from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from .audio_utils import load_audio
from .utils import get_device, resolve_path


class ECAPASpeakerIdentifier:
    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()
        self.speaker_ids: list[str] = []
        self.centroids: np.ndarray | None = None
        self.classifier = None
        cache_dir = config.get("ecapa", {}).get("cache_dir", "data/processed/ecapa_cache")
        self.cache_dir = resolve_path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_backend(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except Exception as exc:
            raise RuntimeError("SpeechBrain is not installed or failed to import. ECAPA is optional.") from exc
        run_device = "cuda:0" if self.device.type == "cuda" else "cpu"
        local_dir = resolve_path(self.config["ecapa"].get("pretrained_dir", "models/ecapa_pretrained"))
        required = [
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "classifier.ckpt",
            "label_encoder.txt",
        ]
        if all((local_dir / name).exists() for name in required):
            self.classifier = EncoderClassifier.from_hparams(
                source=str(local_dir),
                savedir=str(local_dir),
                overrides={"pretrained_path": str(local_dir)},
                local_strategy=LocalStrategy.COPY,
                run_opts={"device": run_device},
            )
            return

        source = self.config["ecapa"].get("source", "speechbrain/spkrec-ecapa-voxceleb")
        savedir = resolve_path("models") / "speechbrain_ecapa"
        self.classifier = EncoderClassifier.from_hparams(source=source, savedir=str(savedir), run_opts={"device": run_device})

    def __getstate__(self):
        state = self.__dict__.copy()
        state["classifier"] = None
        return state

    def _cache_path(self, path: str) -> Path:
        key = hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def embed_file(self, path: str) -> np.ndarray:
        cache_path = self._cache_path(path)
        if cache_path.exists():
            return np.load(cache_path)
        if self.classifier is None:
            self.load_backend()
        audio_cfg = self.config["audio"]
        waveform, sr = load_audio(
            path,
            target_sample_rate=int(audio_cfg["sample_rate"]),
            mono=True,
            do_trim_silence=bool(audio_cfg["trim_silence"]),
        )
        with torch.no_grad():
            wav = waveform.to(self.device)
            emb = self.classifier.encode_batch(wav).squeeze().detach().cpu().numpy().astype(np.float32)
        np.save(cache_path, emb)
        return emb

    def fit(self, train_df: pd.DataFrame) -> "ECAPASpeakerIdentifier":
        self.load_backend()
        centroids = []
        self.speaker_ids = sorted(train_df["speaker_id"].astype(str).unique())
        for speaker_id in self.speaker_ids:
            rows = train_df[train_df["speaker_id"].astype(str) == speaker_id]
            embeddings = [self.embed_file(path) for path in rows["path"]]
            centroids.append(np.mean(np.vstack(embeddings), axis=0))
        self.centroids = np.vstack(centroids).astype(np.float32)
        return self

    def score_file(self, path: str) -> dict[str, float]:
        if self.centroids is None:
            raise RuntimeError("ECAPA model has not been trained.")
        embedding = self.embed_file(path).reshape(1, -1)
        scores = cosine_similarity(embedding, self.centroids)[0]
        return {speaker_id: float(score) for speaker_id, score in zip(self.speaker_ids, scores)}

    def predict_file(self, path: str) -> tuple[str, float, dict[str, float]]:
        scores = self.score_file(path)
        pred = max(scores, key=scores.get)
        return pred, scores[pred], scores
