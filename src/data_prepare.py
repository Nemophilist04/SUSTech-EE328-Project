from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import pandas as pd
import torchaudio

from .utils import ensure_project_dirs, resolve_path, save_json, set_seed


def _librispeech_audio_path(root_dir: Path, url: str, utterance_key: str) -> Path:
    speaker, chapter, _ = utterance_key.split("-")
    return root_dir / "LibriSpeech" / url / speaker / chapter / f"{utterance_key}.flac"


def prepare_librispeech_splits(config: dict) -> pd.DataFrame:
    ensure_project_dirs(config)
    data_cfg = config["data"]
    set_seed(int(data_cfg["random_seed"]))
    root_dir = resolve_path(data_cfg["root_dir"])
    url = data_cfg["librispeech_url"]

    dataset = torchaudio.datasets.LIBRISPEECH(str(root_dir), url=url, download=True)
    walker = list(getattr(dataset, "_walker", []))
    if not walker:
        raise RuntimeError("Could not read LibriSpeech metadata walker from torchaudio dataset.")

    by_speaker: dict[str, list[str]] = defaultdict(list)
    for utterance_key in sorted(walker):
        speaker_id = utterance_key.split("-")[0]
        by_speaker[speaker_id].append(utterance_key)

    train_n = int(data_cfg["train_per_speaker"])
    val_n = int(data_cfg["val_per_speaker"])
    test_n = int(data_cfg["test_per_speaker"])
    total_n = int(data_cfg["utterances_per_speaker"])
    unknown_n_per_speaker = val_n + test_n

    eligible_registered = [spk for spk, utts in by_speaker.items() if len(utts) >= total_n]
    if len(eligible_registered) < int(data_cfg["num_registered_speakers"]):
        raise RuntimeError(
            f"Need {data_cfg['num_registered_speakers']} speakers with at least {total_n} utterances, "
            f"found {len(eligible_registered)}."
        )

    rng = random.Random(int(data_cfg["random_seed"]))
    rng.shuffle(eligible_registered)
    registered = eligible_registered[: int(data_cfg["num_registered_speakers"])]
    eligible_unknown = [
        spk for spk, utts in by_speaker.items() if spk not in registered and len(utts) >= unknown_n_per_speaker
    ]
    if len(eligible_unknown) < int(data_cfg["num_unknown_speakers"]):
        raise RuntimeError(
            f"Need {data_cfg['num_unknown_speakers']} unknown speakers with at least {unknown_n_per_speaker} utterances, "
            f"found {len(eligible_unknown)}."
        )
    rng.shuffle(eligible_unknown)
    unknown = eligible_unknown[: int(data_cfg["num_unknown_speakers"])]

    rows = []
    for speaker_id in registered:
        utts = by_speaker[speaker_id][:total_n]
        split_plan = (
            [("train", utt) for utt in utts[:train_n]]
            + [("val", utt) for utt in utts[train_n : train_n + val_n]]
            + [("test", utt) for utt in utts[train_n + val_n : train_n + val_n + test_n]]
        )
        for split, utt in split_plan:
            rows.append(
                {
                    "split": split,
                    "speaker_id": speaker_id,
                    "utterance_id": utt,
                    "path": str(_librispeech_audio_path(root_dir, url, utt)),
                    "is_unknown": False,
                }
            )

    for speaker_id in unknown:
        utts = by_speaker[speaker_id][:unknown_n_per_speaker]
        split_plan = [("unknown_val", utt) for utt in utts[:val_n]] + [
            ("unknown_test", utt) for utt in utts[val_n : val_n + test_n]
        ]
        for split, utt in split_plan:
            rows.append(
                {
                    "split": split,
                    "speaker_id": speaker_id,
                    "utterance_id": utt,
                    "path": str(_librispeech_audio_path(root_dir, url, utt)),
                    "is_unknown": True,
                }
            )

    df = pd.DataFrame(rows)
    split_dir = resolve_path(data_cfg["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_dir / "metadata.csv", index=False)
    save_json({"registered_speakers": registered, "unknown_speakers": unknown}, split_dir / "speakers.json")
    return df
