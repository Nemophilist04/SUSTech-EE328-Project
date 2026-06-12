from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate_closed_set
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics, _ = evaluate_closed_set(config)
    print(metrics[["model", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
