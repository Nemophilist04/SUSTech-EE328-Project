from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_utils import load_audio
from src.features import MFCCFeatureExtractor
from src.predict import available_model_names, load_trained_model
from src.utils import load_config, model_path, resolve_path


st.set_page_config(page_title="Speaker Identification", layout="wide")
st.title("Speaker Identification Demo")


@st.cache_resource
def cached_config() -> dict:
    return load_config("config.yaml")


@st.cache_resource
def cached_model(model_name: str):
    return load_trained_model(model_name)


config = cached_config()
available = available_model_names(config)
configured = available_model_names(config, include_missing=True)
if not available:
    st.error("No trained models found. Run python scripts/02_train_models.py --config config.yaml first.")
    st.stop()

default_model = config.get("demo", {}).get("default_model", "gmm_8")
default_index = available.index(default_model) if default_model in available else 0

uploaded = st.file_uploader("Upload an audio file", type=["wav", "flac", "mp3", "ogg", "m4a"])
model_name = st.selectbox("Model", options=available, index=default_index)
open_set_enabled = st.checkbox("Open-set rejection", value=True)

missing = sorted(set(configured) - set(available))
if missing:
    st.info(f"Models not trained or unavailable: {', '.join(missing)}")

if uploaded is None:
    st.stop()

upload_dir = resolve_path("data/processed/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)
audio_path = upload_dir / uploaded.name
audio_path.write_bytes(uploaded.getbuffer())

st.audio(str(audio_path))
model = cached_model(model_name)
pred, max_score, scores = model.predict_file(str(audio_path))

waveform, sr = load_audio(
    audio_path,
    target_sample_rate=int(config["audio"]["sample_rate"]),
    mono=True,
    do_trim_silence=bool(config["audio"]["trim_silence"]),
)
samples = waveform.squeeze(0).cpu().numpy()
times = pd.Series(range(len(samples))) / sr

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, samples, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    st.pyplot(fig)

with col2:
    extractor = MFCCFeatureExtractor(config)
    mfcc = extractor.extract(waveform)[:, : int(config["features"]["n_mfcc"])]
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(mfcc.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MFCC")
    ax.set_title("MFCC Spectrogram")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

score_df = pd.DataFrame({"speaker_id": list(scores.keys()), "score": list(scores.values())}).sort_values(
    "score", ascending=False
)
st.bar_chart(score_df.set_index("speaker_id"))

threshold = None
threshold_path = resolve_path("models/open_set_thresholds.json")
if threshold_path.exists():
    with threshold_path.open("r", encoding="utf-8") as f:
        threshold = json.load(f).get(model_name)

if open_set_enabled and threshold is not None:
    decision = "Unknown" if max_score < float(threshold) else pred
    st.metric("Prediction", decision)
    st.write(f"Closed-set best speaker: {pred}")
    st.write(f"Max score: {max_score:.4f}; threshold: {float(threshold):.4f}")
elif open_set_enabled:
    st.metric("Prediction", pred)
    st.warning("Open-set threshold is missing. Run python scripts/04_evaluate_open_set.py --config config.yaml first.")
else:
    st.metric("Predicted speaker ID", pred)
    st.write(f"Max score: {max_score:.4f}")
