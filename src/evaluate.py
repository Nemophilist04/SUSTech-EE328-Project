from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from .predict import available_model_names, load_trained_model, score_rows
from .utils import ensure_project_dirs, load_splits, resolve_path
from .visualize import save_confusion, save_metric_bars


def evaluate_closed_set(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs(config)
    df = load_splits(config)
    test_df = df[df["split"] == "test"].copy()
    labels = sorted(df[df["split"] == "train"]["speaker_id"].astype(str).unique())
    metrics = []
    predictions = []

    for model_name in available_model_names(config):
        model = load_trained_model(model_name)
        records = score_rows(model, test_df)
        y_true = [r["true_speaker_id"] for r in records]
        y_pred = [r["predicted_speaker_id"] for r in records]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        metrics.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "classification_report": json.dumps(classification_report(y_true, y_pred, labels=labels, zero_division=0)),
            }
        )
        for record in records:
            record = dict(record)
            record["model"] = model_name
            record["scores"] = json.dumps(record["scores"])
            predictions.append(record)
        if config["evaluation"].get("save_confusion_matrix", True):
            save_confusion(y_true, y_pred, labels, model_name)

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.DataFrame(predictions)
    metrics_df.to_csv(resolve_path("results/metrics/closed_set_metrics.csv"), index=False)
    pred_df.to_csv(resolve_path("results/predictions/closed_set_predictions.csv"), index=False)
    save_metric_bars(metrics_df)
    return metrics_df, pred_df
