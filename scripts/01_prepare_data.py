from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_prepare import prepare_librispeech_splits
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    df = prepare_librispeech_splits(config)
    print(f"Saved {len(df)} split rows to {Path(config['data']['split_dir']) / 'metadata.csv'}")


if __name__ == "__main__":
    main()
