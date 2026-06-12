from __future__ import annotations

from pathlib import Path

from .model_baseline import BaselineSpeakerIdentifier
from .model_ecapa import ECAPASpeakerIdentifier
from .model_gmm import GMMSpeakerIdentifier
from .utils import ensure_project_dirs, load_splits, model_path, save_joblib, save_json


def train_all_models(config: dict) -> dict[str, str]:
    ensure_project_dirs(config)
    df = load_splits(config)
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise RuntimeError("No training rows found in split metadata.")

    saved: dict[str, str] = {}

    baseline = BaselineSpeakerIdentifier(config).fit(train_df)
    saved["baseline"] = str(save_joblib(baseline, model_path("baseline")))

    for n_components in config["gmm"]["n_components_list"]:
        name = f"gmm_{int(n_components)}"
        model = GMMSpeakerIdentifier(config, n_components=int(n_components)).fit(train_df)
        saved[name] = str(save_joblib(model, model_path(name)))

    if bool(config.get("ecapa", {}).get("enabled", False)):
        try:
            ecapa = ECAPASpeakerIdentifier(config).fit(train_df)
            saved["ecapa"] = str(save_joblib(ecapa, model_path("ecapa")))
        except Exception as exc:
            save_json({"ecapa_status": "skipped", "reason": str(exc)}, Path("models") / "ecapa_skipped.json")
            print(f"ECAPA skipped: {exc}")

    return saved
