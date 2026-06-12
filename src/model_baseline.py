from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .audio_utils import load_audio
from .features import MFCCFeatureExtractor, stats_pooling


class BaselineSpeakerIdentifier:
    def __init__(self, config: dict):
        self.config = config
        self.extractor = MFCCFeatureExtractor(config)
        self.speaker_ids: list[str] = []
        self.templates: np.ndarray | None = None

    def embed_file(self, path: str) -> np.ndarray:
        audio_cfg = self.config["audio"]
        waveform, _ = load_audio(
            path,
            target_sample_rate=int(audio_cfg["sample_rate"]),
            mono=bool(audio_cfg["mono"]),
            do_trim_silence=bool(audio_cfg["trim_silence"]),
        )
        return stats_pooling(self.extractor.extract(waveform))

    def fit(self, train_df: pd.DataFrame) -> "BaselineSpeakerIdentifier":
        templates = []
        speaker_ids = sorted(train_df["speaker_id"].astype(str).unique())
        for speaker_id in speaker_ids:
            speaker_rows = train_df[train_df["speaker_id"].astype(str) == speaker_id]
            embeddings = [self.embed_file(path) for path in speaker_rows["path"]]
            templates.append(np.mean(np.vstack(embeddings), axis=0))
        self.speaker_ids = speaker_ids
        self.templates = np.vstack(templates).astype(np.float32)
        return self

    def score_file(self, path: str) -> dict[str, float]:
        if self.templates is None:
            raise RuntimeError("Baseline model has not been trained.")
        embedding = self.embed_file(path).reshape(1, -1)
        scores = cosine_similarity(embedding, self.templates)[0]
        return {speaker_id: float(score) for speaker_id, score in zip(self.speaker_ids, scores)}

    def predict_file(self, path: str) -> tuple[str, float, dict[str, float]]:
        scores = self.score_file(path)
        pred = max(scores, key=scores.get)
        return pred, scores[pred], scores
