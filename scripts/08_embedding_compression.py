from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_compression import run_embedding_compression_experiments
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics_df, _, _ = run_embedding_compression_experiments(config)

    display_cols = [
        "method",
        "embedding_dim",
        "centroid_dtype",
        "closed_set_accuracy",
        "open_set_overall_accuracy",
        "storage_ratio_vs_float32",
        "average_scoring_time_ms",
        "notes",
    ]
    print("\nEmbedding compression metrics")
    print(metrics_df[display_cols].to_string(index=False))

    valid = metrics_df.dropna(subset=["closed_set_accuracy", "open_set_overall_accuracy"]).copy()
    compressed = valid[valid["method"] != "ecapa_original_float32"].copy()
    if not compressed.empty:
        best_closed = compressed.sort_values(["closed_set_accuracy", "macro_f1"], ascending=False).iloc[0]
        best_open = compressed.sort_values("open_set_overall_accuracy", ascending=False).iloc[0]
        tradeoff = compressed.assign(
            tradeoff_score=compressed["open_set_overall_accuracy"] / compressed["storage_ratio_vs_float32"].clip(lower=1e-12)
        ).sort_values("tradeoff_score", ascending=False).iloc[0]
        print(
            f"\nBest compressed method by closed-set accuracy: {best_closed['method']} "
            f"({best_closed['closed_set_accuracy']:.4f})"
        )
        print(
            f"Best compressed method by open-set overall accuracy: {best_open['method']} "
            f"({best_open['open_set_overall_accuracy']:.4f})"
        )
        print(
            f"Best storage/accuracy trade-off: {tradeoff['method']} "
            f"(open={tradeoff['open_set_overall_accuracy']:.4f}, "
            f"storage_ratio={tradeoff['storage_ratio_vs_float32']:.4f})"
        )


if __name__ == "__main__":
    main()
