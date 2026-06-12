from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .model_ecapa import ECAPASpeakerIdentifier
from .utils import ensure_project_dirs, resolve_path


UNKNOWN_LABEL = "Unknown"


@dataclass
class CompressionResult:
    method: str
    compression_type: str
    embedding_dim: int
    centroid_dtype: str
    closed_set_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    open_set_overall_accuracy: float
    known_accuracy: float
    unknown_rejection_accuracy: float
    false_acceptance_rate: float
    false_rejection_rate: float
    centroid_storage_bytes: int
    storage_ratio_vs_float32: float
    average_scoring_time_ms: float
    notes: str


def load_split_csvs(config: dict) -> dict[str, pd.DataFrame]:
    split_dir = resolve_path(config["data"]["split_dir"])
    paths = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
        "unknown_val": split_dir / "unknown_val.csv",
        "unknown_test": split_dir / "unknown_test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        metadata_path = split_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing split files and metadata.csv in {split_dir}")
        metadata = pd.read_csv(metadata_path)
        for split_name, path in paths.items():
            split_value = split_name
            metadata[metadata["split"] == split_value].to_csv(path, index=False)
    return {name: pd.read_csv(path) for name, path in paths.items()}


def load_or_extract_ecapa_embeddings(
    config: dict,
    rows_by_split: dict[str, pd.DataFrame],
    cache_path: str | Path = "data/processed/ecapa_embedding_cache.pkl",
) -> dict[str, np.ndarray]:
    cache_file = resolve_path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        with cache_file.open("rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    all_paths = []
    for df in rows_by_split.values():
        all_paths.extend(df["path"].astype(str).tolist())
    unique_paths = sorted(set(all_paths))
    missing = [path for path in unique_paths if path not in cache]

    if missing:
        model = ECAPASpeakerIdentifier(config)
        for path in missing:
            cache[path] = model.embed_file(path).astype(np.float32)
        with cache_file.open("wb") as f:
            pickle.dump(cache, f)

    return {path: np.asarray(embedding, dtype=np.float32) for path, embedding in cache.items()}


def embeddings_for_rows(df: pd.DataFrame, embedding_cache: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([embedding_cache[str(path)] for path in df["path"].astype(str)])


def compute_speaker_centroids(
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    speaker_ids = sorted(train_df["speaker_id"].astype(str).unique())
    centroids = []
    speakers = train_df["speaker_id"].astype(str).to_numpy()
    for speaker_id in speaker_ids:
        centroids.append(train_embeddings[speakers == speaker_id].mean(axis=0))
    return speaker_ids, np.vstack(centroids).astype(np.float32)


def fit_pca(train_embeddings: np.ndarray, n_components: int) -> PCA:
    if n_components > min(train_embeddings.shape):
        raise ValueError(f"PCA-{n_components} is invalid for train embedding shape {train_embeddings.shape}")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_embeddings)
    return pca


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine_scores(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return l2_normalize(embeddings.astype(np.float32)) @ l2_normalize(centroids.astype(np.float32)).T


def compress_centroids_float16(centroids: np.ndarray) -> np.ndarray:
    return centroids.astype(np.float16)


def quantize_centroids_int8(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.max(np.abs(centroids), axis=1).astype(np.float32) / 127.0
    scales = np.where(scales <= 1e-12, 1.0, scales).astype(np.float32)
    q = np.round(centroids / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales


def dequantize_centroids_int8(q_centroids: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q_centroids.astype(np.float32) * scales.astype(np.float32)[:, None]


def centroid_storage_bytes(num_speakers: int, embedding_dim: int, centroid_dtype: str) -> int:
    if centroid_dtype == "float32":
        return num_speakers * embedding_dim * 4
    if centroid_dtype == "float16":
        return num_speakers * embedding_dim * 2
    if centroid_dtype == "int8":
        return num_speakers * embedding_dim + num_speakers * 4
    raise ValueError(f"Unsupported centroid dtype: {centroid_dtype}")


def predict_from_scores(scores: np.ndarray, speaker_ids: list[str]) -> tuple[list[str], np.ndarray]:
    idx = np.argmax(scores, axis=1)
    return [speaker_ids[i] for i in idx], scores[np.arange(scores.shape[0]), idx]


def evaluate_closed_set(
    method: str,
    speaker_ids: list[str],
    centroids: np.ndarray,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
) -> tuple[dict, list[dict], float]:
    start = time.perf_counter()
    scores = cosine_scores(test_embeddings, centroids)
    elapsed = time.perf_counter() - start
    y_pred, max_scores = predict_from_scores(scores, speaker_ids)
    y_true = test_df["speaker_id"].astype(str).tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=speaker_ids, average="macro", zero_division=0
    )
    metrics = {
        "closed_set_accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }
    predictions = []
    for row, pred, max_score, score_row in zip(test_df.itertuples(index=False), y_pred, max_scores, scores):
        predictions.append(
            {
                "method": method,
                "utterance_id": row.utterance_id,
                "path": row.path,
                "true_speaker_id": str(row.speaker_id),
                "predicted_speaker_id": pred,
                "max_score": float(max_score),
                "scores": json.dumps({speaker: float(score) for speaker, score in zip(speaker_ids, score_row)}),
            }
        )
    avg_time_ms = (elapsed / max(len(test_df), 1)) * 1000.0
    return metrics, predictions, avg_time_ms


def select_open_set_threshold(known_scores: np.ndarray, unknown_scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate([known_scores, unknown_scores]))
    best_threshold = float(candidates[0])
    best_accuracy = -1.0
    for threshold in candidates:
        known_ok = known_scores >= threshold
        unknown_ok = unknown_scores < threshold
        accuracy = float((known_ok.sum() + unknown_ok.sum()) / (len(known_scores) + len(unknown_scores)))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def evaluate_open_set(
    method: str,
    speaker_ids: list[str],
    centroids: np.ndarray,
    val_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    unknown_val_df: pd.DataFrame,
    unknown_val_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    unknown_test_df: pd.DataFrame,
    unknown_test_embeddings: np.ndarray,
) -> tuple[dict, list[dict]]:
    val_known_scores = cosine_scores(val_embeddings, centroids)
    val_unknown_scores = cosine_scores(unknown_val_embeddings, centroids)
    _, val_known_max = predict_from_scores(val_known_scores, speaker_ids)
    _, val_unknown_max = predict_from_scores(val_unknown_scores, speaker_ids)
    threshold, val_accuracy = select_open_set_threshold(val_known_max, val_unknown_max)

    known_scores = cosine_scores(test_embeddings, centroids)
    unknown_scores = cosine_scores(unknown_test_embeddings, centroids)
    known_pred, known_max = predict_from_scores(known_scores, speaker_ids)
    unknown_pred, unknown_max = predict_from_scores(unknown_scores, speaker_ids)

    predictions = []
    known_correct = 0
    false_reject = 0
    for row, pred, max_score, score_row in zip(test_df.itertuples(index=False), known_pred, known_max, known_scores):
        open_pred = UNKNOWN_LABEL if max_score < threshold else pred
        if open_pred == UNKNOWN_LABEL:
            false_reject += 1
        elif open_pred == str(row.speaker_id):
            known_correct += 1
        predictions.append(_open_prediction_row(method, row, pred, open_pred, max_score, threshold, False, speaker_ids, score_row))

    unknown_rejected = 0
    false_accept = 0
    for row, pred, max_score, score_row in zip(
        unknown_test_df.itertuples(index=False), unknown_pred, unknown_max, unknown_scores
    ):
        open_pred = UNKNOWN_LABEL if max_score < threshold else pred
        if open_pred == UNKNOWN_LABEL:
            unknown_rejected += 1
        else:
            false_accept += 1
        predictions.append(_open_prediction_row(method, row, pred, open_pred, max_score, threshold, True, speaker_ids, score_row))

    known_total = len(test_df)
    unknown_total = len(unknown_test_df)
    total = known_total + unknown_total
    metrics = {
        "threshold": threshold,
        "validation_open_set_accuracy": val_accuracy,
        "known_accuracy": known_correct / known_total if known_total else 0.0,
        "unknown_rejection_accuracy": unknown_rejected / unknown_total if unknown_total else 0.0,
        "false_acceptance_rate": false_accept / unknown_total if unknown_total else 0.0,
        "false_rejection_rate": false_reject / known_total if known_total else 0.0,
        "open_set_overall_accuracy": (known_correct + unknown_rejected) / total if total else 0.0,
    }
    return metrics, predictions


def _open_prediction_row(
    method: str,
    row,
    closed_pred: str,
    open_pred: str,
    max_score: float,
    threshold: float,
    is_unknown: bool,
    speaker_ids: list[str],
    score_row: np.ndarray,
) -> dict:
    return {
        "method": method,
        "utterance_id": row.utterance_id,
        "path": row.path,
        "true_speaker_id": str(row.speaker_id),
        "is_unknown": is_unknown,
        "closed_set_prediction": closed_pred,
        "open_set_prediction": open_pred,
        "max_score": float(max_score),
        "threshold": float(threshold),
        "scores": json.dumps({speaker: float(score) for speaker, score in zip(speaker_ids, score_row)}),
    }


def measure_scoring_time_ms(embeddings: np.ndarray, centroids: np.ndarray, repeats: int = 200) -> float:
    repeats = max(1, int(repeats))
    start = time.perf_counter()
    for _ in range(repeats):
        _ = cosine_scores(embeddings, centroids)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats / max(len(embeddings), 1)) * 1000.0


def run_embedding_compression_experiments(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs(config)
    rows = load_split_csvs(config)
    embedding_cache = load_or_extract_ecapa_embeddings(config, rows)

    split_embeddings = {name: embeddings_for_rows(df, embedding_cache) for name, df in rows.items()}
    original_dim = int(split_embeddings["train"].shape[1])
    experiments = []

    experiments.append(("ecapa_original_float32", "none", original_dim, "float32", None, "Original ECAPA embeddings."))
    for dim in [128, 64, 32, 16]:
        experiments.append((f"ecapa_pca_{dim}", "pca", dim, "float32", dim, "PCA fitted on enrollment embeddings only."))
    experiments.append(("ecapa_float16_centroid", "centroid_quantization", original_dim, "float16", None, "Centroids stored as float16."))
    experiments.append(("ecapa_int8_centroid", "centroid_quantization", original_dim, "int8", None, "Symmetric per-centroid int8 quantization."))

    metrics_rows = []
    closed_predictions = []
    open_predictions = []

    for method, compression_type, embedding_dim, centroid_dtype, pca_dim, notes in experiments:
        try:
            transformed = split_embeddings
            if pca_dim is not None:
                pca = fit_pca(split_embeddings["train"], pca_dim)
                transformed = {name: pca.transform(values).astype(np.float32) for name, values in split_embeddings.items()}

            speaker_ids, centroids = compute_speaker_centroids(rows["train"], transformed["train"])
            scoring_centroids = centroids
            if centroid_dtype == "float16":
                scoring_centroids = compress_centroids_float16(centroids).astype(np.float32)
            elif centroid_dtype == "int8":
                q_centroids, scales = quantize_centroids_int8(centroids)
                scoring_centroids = dequantize_centroids_int8(q_centroids, scales)

            closed_metrics, closed_pred, _ = evaluate_closed_set(
                method, speaker_ids, scoring_centroids, rows["test"], transformed["test"]
            )
            open_metrics, open_pred = evaluate_open_set(
                method,
                speaker_ids,
                scoring_centroids,
                rows["val"],
                transformed["val"],
                rows["unknown_val"],
                transformed["unknown_val"],
                rows["test"],
                transformed["test"],
                rows["unknown_test"],
                transformed["unknown_test"],
            )
            storage_bytes = centroid_storage_bytes(len(speaker_ids), int(scoring_centroids.shape[1]), centroid_dtype)
            reference_bytes = centroid_storage_bytes(len(speaker_ids), original_dim, "float32")
            timing_ms = measure_scoring_time_ms(transformed["test"], scoring_centroids)

            metrics_rows.append(
                CompressionResult(
                    method=method,
                    compression_type=compression_type,
                    embedding_dim=int(scoring_centroids.shape[1]),
                    centroid_dtype=centroid_dtype,
                    closed_set_accuracy=closed_metrics["closed_set_accuracy"],
                    macro_precision=closed_metrics["macro_precision"],
                    macro_recall=closed_metrics["macro_recall"],
                    macro_f1=closed_metrics["macro_f1"],
                    open_set_overall_accuracy=open_metrics["open_set_overall_accuracy"],
                    known_accuracy=open_metrics["known_accuracy"],
                    unknown_rejection_accuracy=open_metrics["unknown_rejection_accuracy"],
                    false_acceptance_rate=open_metrics["false_acceptance_rate"],
                    false_rejection_rate=open_metrics["false_rejection_rate"],
                    centroid_storage_bytes=storage_bytes,
                    storage_ratio_vs_float32=storage_bytes / reference_bytes,
                    average_scoring_time_ms=timing_ms,
                    notes=notes,
                ).__dict__
            )
            closed_predictions.extend(closed_pred)
            open_predictions.extend(open_pred)
        except Exception as exc:
            metrics_rows.append(
                {
                    "method": method,
                    "compression_type": compression_type,
                    "embedding_dim": embedding_dim,
                    "centroid_dtype": centroid_dtype,
                    "closed_set_accuracy": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall": np.nan,
                    "macro_f1": np.nan,
                    "open_set_overall_accuracy": np.nan,
                    "known_accuracy": np.nan,
                    "unknown_rejection_accuracy": np.nan,
                    "false_acceptance_rate": np.nan,
                    "false_rejection_rate": np.nan,
                    "centroid_storage_bytes": np.nan,
                    "storage_ratio_vs_float32": np.nan,
                    "average_scoring_time_ms": np.nan,
                    "notes": f"FAILED: {exc}",
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    closed_pred_df = pd.DataFrame(closed_predictions)
    open_pred_df = pd.DataFrame(open_predictions)
    save_embedding_compression_outputs(metrics_df, closed_pred_df, open_pred_df)
    return metrics_df, closed_pred_df, open_pred_df


def save_embedding_compression_outputs(
    metrics_df: pd.DataFrame,
    closed_pred_df: pd.DataFrame,
    open_pred_df: pd.DataFrame,
) -> None:
    metrics_dir = resolve_path("results/metrics")
    predictions_dir = resolve_path("results/predictions")
    figures_dir = resolve_path("results/figures")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_dir / "embedding_compression_metrics.csv", index=False)
    closed_pred_df.to_csv(predictions_dir / "embedding_compression_closed_set_predictions.csv", index=False)
    open_pred_df.to_csv(predictions_dir / "embedding_compression_open_set_predictions.csv", index=False)
    save_embedding_compression_figures(metrics_df, figures_dir)


def save_embedding_compression_figures(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = metrics_df.dropna(subset=["closed_set_accuracy"]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df["method"], plot_df["closed_set_accuracy"])
    plt.ylim(0, 1)
    plt.ylabel("Closed-set accuracy")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "embedding_compression_closed_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df["method"], plot_df["open_set_overall_accuracy"])
    plt.ylim(0, 1)
    plt.ylabel("Open-set overall accuracy")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "embedding_compression_open_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df["storage_ratio_vs_float32"], plot_df["open_set_overall_accuracy"])
    for row in plot_df.itertuples(index=False):
        plt.annotate(row.method, (row.storage_ratio_vs_float32, row.open_set_overall_accuracy), fontsize=8)
    plt.xlabel("Centroid storage ratio vs original float32")
    plt.ylabel("Open-set overall accuracy")
    plt.tight_layout()
    plt.savefig(figures_dir / "embedding_compression_storage_vs_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df["method"], plot_df["average_scoring_time_ms"])
    plt.ylabel("Average scoring time (ms / utterance)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "embedding_compression_scoring_time.png", dpi=160)
    plt.close()
