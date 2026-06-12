from __future__ import annotations

import numpy as np
import torch
import torchaudio


class MFCCFeatureExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = int(config["audio"]["sample_rate"])
        feature_cfg = config["features"]
        self.n_mfcc = int(feature_cfg["n_mfcc"])
        n_fft = int(self.sample_rate * float(feature_cfg["frame_length_ms"]) / 1000)
        hop_length = int(self.sample_rate * float(feature_cfg["hop_length_ms"]) / 1000)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": max(40, self.n_mfcc * 3),
                "center": True,
                "power": 2.0,
            },
        )
        self.delta = torchaudio.transforms.ComputeDeltas()

    def extract(self, waveform: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            mfcc = self.mfcc(waveform).squeeze(0)
            features = [mfcc]
            if self.config["features"].get("use_delta", True):
                d1 = self.delta(mfcc)
                features.append(d1)
                if self.config["features"].get("use_delta_delta", True):
                    features.append(self.delta(d1))
            stacked = torch.cat(features, dim=0).transpose(0, 1).float()
            arr = stacked.cpu().numpy()
        if self.config["features"].get("normalize", True):
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True) + 1e-8
            arr = (arr - mean) / std
        return arr.astype(np.float32)


def stats_pooling(frame_features: np.ndarray) -> np.ndarray:
    mean = frame_features.mean(axis=0)
    std = frame_features.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)
