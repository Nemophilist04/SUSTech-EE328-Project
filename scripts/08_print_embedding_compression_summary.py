from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "metrics" / "embedding_compression_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python scripts/08_embedding_compression.py --config config.yaml")
    df = pd.read_csv(path)
    cols = [
        "method",
        "embedding_dim",
        "centroid_dtype",
        "closed_set_accuracy",
        "open_set_overall_accuracy",
        "storage_ratio_vs_float32",
        "average_scoring_time_ms",
        "notes",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
