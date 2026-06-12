from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .audio_utils import load_audio
from .features import MFCCFeatureExtractor


class GMMSpeakerIdentifier:
    def __init__(self, config: dict, n_components: int):
        self.config = config
        self.n_components = int(n_components)
        self.extractor = MFCCFeatureExtractor(config)
        self.models: dict[str, GaussianMixture] = {}
        self.speaker_ids: list[str] = []

    def features_file(self, path: str) -> np.ndarray:
        audio_cfg = self.config["audio"]
        waveform, _ = load_audio(
            path,
            target_sample_rate=int(audio_cfg["sample_rate"]),
            mono=bool(audio_cfg["mono"]),
            do_trim_silence=bool(audio_cfg["trim_silence"]),
        )
        return self.extractor.extract(waveform)

    def fit(self, train_df: pd.DataFrame) -> "GMMSpeakerIdentifier":
        gmm_cfg = self.config["gmm"]
        self.speaker_ids = sorted(train_df["speaker_id"].astype(str).unique())
        for speaker_id in self.speaker_ids:
            speaker_rows = train_df[train_df["speaker_id"].astype(str) == speaker_id]
            frames = [self.features_file(path) for path in speaker_rows["path"]]
            x = np.vstack(frames)
            model = GaussianMixture(
                n_components=self.n_components,
                covariance_type=gmm_cfg.get("covariance_type", "diag"),
                max_iter=int(gmm_cfg.get("max_iter", 200)),
                random_state=int(gmm_cfg.get("random_state", 42)),
                reg_covar=1e-6,
            )
            model.fit(x)
            self.models[speaker_id] = model
        return self

    def score_file(self, path: str) -> dict[str, float]:
        if not self.models:
            raise RuntimeError("GMM model has not been trained.")
        frames = self.features_file(path)
        return {speaker_id: float(model.score(frames)) for speaker_id, model in self.models.items()}

    def predict_file(self, path: str) -> tuple[str, float, dict[str, float]]:
        scores = self.score_file(path)
        pred = max(scores, key=scores.get)
        return pred, scores[pred], scores
