from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def main() -> None:
    closed_path = PROJECT_ROOT / "results" / "metrics" / "closed_set_metrics.csv"
    open_path = PROJECT_ROOT / "results" / "metrics" / "open_set_metrics.csv"
    closed = _read_csv(closed_path)
    open_set = _read_csv(open_path)

    closed_cols = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    open_cols = [
        "model",
        "threshold",
        "known_speaker_accuracy",
        "unknown_rejection_accuracy",
        "false_acceptance_rate",
        "false_rejection_rate",
        "overall_open_set_accuracy",
    ]

    best_closed = closed.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
    best_open = open_set.sort_values("overall_open_set_accuracy", ascending=False).iloc[0]

    print("\nClosed-set metrics")
    print(closed[closed_cols].to_string(index=False))
    print("\nOpen-set metrics")
    print(open_set[open_cols].to_string(index=False))
    print(
        f"\nBest closed-set model: {best_closed['model']} "
        f"(accuracy={best_closed['accuracy']:.4f}, macro_f1={best_closed['macro_f1']:.4f})"
    )
    print(
        f"Best open-set model: {best_open['model']} "
        f"(overall_open_set_accuracy={best_open['overall_open_set_accuracy']:.4f})"
    )

    figures = [
        "results/figures/model_comparison_accuracy.png",
        "results/figures/model_comparison_f1.png",
        "results/figures/confusion_matrix_baseline.png",
        "results/figures/confusion_matrix_gmm_8.png",
        "results/figures/confusion_matrix_gmm_16.png",
        "results/figures/confusion_matrix_ecapa.png",
        "results/figures/open_set_score_distribution.png",
    ]
    print("\nImportant figures")
    for rel_path in figures:
        path = PROJECT_ROOT / rel_path
        status = "OK" if path.exists() else "MISSING"
        print(f"- {status}: {path}")


if __name__ == "__main__":
    main()
