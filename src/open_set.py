from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .predict import available_model_names, load_trained_model, score_rows
from .utils import ensure_project_dirs, load_splits, resolve_path, save_json
from .visualize import save_open_set_distribution


def _best_threshold(known_scores: np.ndarray, unknown_scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate([known_scores, unknown_scores]))
    if candidates.size == 0:
        raise RuntimeError("No validation scores available for open-set threshold selection.")
    best_thr = float(candidates[0])
    best_acc = -1.0
    for thr in candidates:
        known_ok = known_scores >= thr
        unknown_ok = unknown_scores < thr
        acc = float((known_ok.sum() + unknown_ok.sum()) / (len(known_scores) + len(unknown_scores)))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def evaluate_open_set(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs(config)
    df = load_splits(config)
    known_val = df[df["split"] == "val"].copy()
    unknown_val = df[df["split"] == "unknown_val"].copy()
    known_test = df[df["split"] == "test"].copy()
    unknown_test = df[df["split"] == "unknown_test"].copy()

    metrics = []
    predictions = []
    dist_rows = []
    thresholds = {}

    for model_name in available_model_names(config):
        model = load_trained_model(model_name)
        val_known_records = score_rows(model, known_val)
        val_unknown_records = score_rows(model, unknown_val)
        known_scores = np.array([r["max_score"] for r in val_known_records], dtype=float)
        unknown_scores = np.array([r["max_score"] for r in val_unknown_records], dtype=float)
        threshold, val_acc = _best_threshold(known_scores, unknown_scores)
        thresholds[model_name] = threshold

        for score in known_scores:
            dist_rows.append({"model": model_name, "group": "known_val", "max_score": score})
        for score in unknown_scores:
            dist_rows.append({"model": model_name, "group": "unknown_val", "max_score": score})

        test_records = score_rows(model, known_test)
        unknown_records = score_rows(model, unknown_test)

        known_correct = 0
        false_reject = 0
        for record in test_records:
            open_pred = "Unknown" if record["max_score"] < threshold else record["predicted_speaker_id"]
            if open_pred == "Unknown":
                false_reject += 1
            elif open_pred == record["true_speaker_id"]:
                known_correct += 1
            predictions.append(_prediction_row(model_name, record, threshold, open_pred, False))

        unknown_rejected = 0
        false_accept = 0
        for record in unknown_records:
            open_pred = "Unknown" if record["max_score"] < threshold else record["predicted_speaker_id"]
            if open_pred == "Unknown":
                unknown_rejected += 1
            else:
                false_accept += 1
            predictions.append(_prediction_row(model_name, record, threshold, open_pred, True))

        known_total = len(test_records)
        unknown_total = len(unknown_records)
        total = known_total + unknown_total
        metrics.append(
            {
                "model": model_name,
                "threshold": threshold,
                "validation_open_set_accuracy": val_acc,
                "known_speaker_accuracy": known_correct / known_total if known_total else 0.0,
                "unknown_rejection_accuracy": unknown_rejected / unknown_total if unknown_total else 0.0,
                "false_acceptance_rate": false_accept / unknown_total if unknown_total else 0.0,
                "false_rejection_rate": false_reject / known_total if known_total else 0.0,
                "overall_open_set_accuracy": (known_correct + unknown_rejected) / total if total else 0.0,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.DataFrame(predictions)
    metrics_df.to_csv(resolve_path("results/metrics/open_set_metrics.csv"), index=False)
    pred_df.to_csv(resolve_path("results/predictions/open_set_predictions.csv"), index=False)
    save_json(thresholds, "models/open_set_thresholds.json")
    save_open_set_distribution(pd.DataFrame(dist_rows))
    return metrics_df, pred_df


def _prediction_row(model_name: str, record: dict, threshold: float, open_pred: str, is_unknown: bool) -> dict:
    return {
        "model": model_name,
        "utterance_id": record["utterance_id"],
        "path": record["path"],
        "true_speaker_id": record["true_speaker_id"],
        "is_unknown": is_unknown,
        "closed_set_prediction": record["predicted_speaker_id"],
        "open_set_prediction": open_pred,
        "max_score": record["max_score"],
        "threshold": threshold,
        "scores": json.dumps(record["scores"]),
    }
