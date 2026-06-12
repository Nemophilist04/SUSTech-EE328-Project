from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_project_dirs(config: dict[str, Any]) -> None:
    for key in ("root_dir", "processed_dir", "split_dir"):
        resolve_path(config["data"][key]).mkdir(parents=True, exist_ok=True)
    for folder in ("models", "results/metrics", "results/predictions", "results/figures"):
        resolve_path(folder).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_joblib(obj: Any, path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path_like: str | Path) -> Any:
    path = resolve_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Required model file is missing: {path}")
    return joblib.load(path)


def save_json(obj: Any, path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def load_splits(config: dict[str, Any]) -> pd.DataFrame:
    split_path = resolve_path(config["data"]["split_dir"]) / "metadata.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split metadata not found: {split_path}. Run python scripts/01_prepare_data.py --config config.yaml first."
        )
    df = pd.read_csv(split_path)
    df["path"] = df["path"].map(str)
    return df


def model_path(model_name: str) -> Path:
    return resolve_path("models") / f"{model_name}.joblib"
