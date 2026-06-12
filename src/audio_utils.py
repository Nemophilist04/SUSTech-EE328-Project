from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def trim_silence(waveform: torch.Tensor, sample_rate: int, top_db: float = 35.0) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    mono = waveform.mean(dim=0)
    frame_length = max(int(sample_rate * 0.02), 1)
    hop = max(frame_length // 2, 1)
    frames = mono.unfold(0, frame_length, hop)
    rms = torch.sqrt(torch.mean(frames.pow(2), dim=1) + 1e-10)
    threshold = rms.max() * (10.0 ** (-top_db / 20.0))
    voiced = torch.nonzero(rms >= threshold, as_tuple=False).flatten()
    if voiced.numel() == 0:
        return waveform
    start = int(max(voiced[0].item() * hop, 0))
    end = int(min(voiced[-1].item() * hop + frame_length, waveform.shape[1]))
    return waveform[:, start:end]


def load_audio(
    path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
    do_trim_silence: bool = True,
) -> tuple[torch.Tensor, int]:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise
        audio_np, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(audio_np.T)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    if do_trim_silence:
        waveform = trim_silence(waveform, sample_rate)
    if waveform.shape[1] < sample_rate // 10:
        pad = sample_rate // 10 - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform.contiguous(), sample_rate
