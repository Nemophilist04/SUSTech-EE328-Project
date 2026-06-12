from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import load_joblib, model_path, resolve_path


def available_model_names(config: dict, include_missing: bool = False) -> list[str]:
    names = ["baseline"] + [f"gmm_{int(n)}" for n in config["gmm"]["n_components_list"]]
    if bool(config.get("ecapa", {}).get("enabled", False)):
        names.append("ecapa")
    if include_missing:
        return names
    return [name for name in names if model_path(name).exists()]


def load_trained_model(model_name: str):
    return load_joblib(model_path(model_name))


def score_rows(model, rows: pd.DataFrame) -> list[dict]:
    records = []
    for row in rows.itertuples(index=False):
        pred, max_score, scores = model.predict_file(row.path)
        records.append(
            {
                "utterance_id": row.utterance_id,
                "path": row.path,
                "true_speaker_id": str(row.speaker_id),
                "predicted_speaker_id": str(pred),
                "max_score": float(max_score),
                "scores": scores,
            }
        )
    return records


def predict_file(model_name: str, audio_path: str | Path) -> tuple[str, float, dict[str, float]]:
    model = load_trained_model(model_name)
    return model.predict_file(str(resolve_path(audio_path)))
