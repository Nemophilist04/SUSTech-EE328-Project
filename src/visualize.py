from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from .utils import resolve_path


def save_metric_bars(metrics_df: pd.DataFrame) -> None:
    fig_dir = resolve_path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename in [("accuracy", "model_comparison_accuracy.png"), ("macro_f1", "model_comparison_f1.png")]:
        if metric not in metrics_df.columns or metrics_df.empty:
            continue
        plt.figure(figsize=(8, 4))
        plt.bar(metrics_df["model"], metrics_df[metric])
        plt.ylim(0, 1)
        plt.xticks(rotation=20)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=160)
        plt.close()


def save_confusion(y_true: list[str], y_pred: list[str], labels: list[str], model_name: str) -> None:
    fig_dir = resolve_path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close()


def save_open_set_distribution(score_df: pd.DataFrame) -> None:
    fig_dir = resolve_path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    if score_df.empty:
        return
    for model_name, model_df in score_df.groupby("model"):
        plt.figure(figsize=(9, 5))
        for group, group_df in model_df.groupby("group"):
            plt.hist(group_df["max_score"], bins=20, alpha=0.45, density=True, label=group)
        plt.title(f"Open-Set Validation Score Distribution - {model_name}")
        plt.xlabel("Max score")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        out_name = "open_set_score_distribution.png" if model_name == score_df["model"].iloc[0] else f"open_set_score_distribution_{model_name}.png"
        plt.savefig(fig_dir / out_name, dpi=160)
        plt.close()
