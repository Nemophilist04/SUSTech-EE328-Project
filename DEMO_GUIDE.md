# Demo Guide

## Start Streamlit

Open Windows PowerShell:

```powershell
conda activate speaker-id
cd C:\Users\ben04\Desktop\speaker_id_project
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## First Model to Show

Start with `ecapa` for the strongest open-set result. Then switch to `gmm_16` to show the best traditional model.

Enable open-set rejection in the app so the threshold and Known/Unknown decision are visible.

## Select a Known-Speaker Sample

Open:

```text
data\splits\test.csv
```

Pick any row and use the `path` column as the audio file to upload. These samples are from registered speakers and should usually produce a known speaker ID.

## Select an Unknown-Speaker Sample

Open:

```text
data\splits\unknown_test.csv
```

Pick any row and use the `path` column as the audio file to upload. These samples are from impostor speakers and should ideally be rejected as `Unknown` when open-set mode is enabled.

## Screenshots for PPT

Capture these screens:

- Streamlit app with uploaded audio player.
- Waveform plot.
- MFCC spectrogram.
- Speaker score bar chart.
- Prediction panel showing `ecapa`.
- Open-set decision showing a known sample accepted.
- Open-set decision showing an unknown sample rejected.
- `results\figures\model_comparison_accuracy.png`.
- `results\figures\model_comparison_f1.png`.
- `results\figures\confusion_matrix_ecapa.png`.
- `results\figures\open_set_score_distribution.png`.

## Open-Set Rejection Demo

1. Choose `ecapa`.
2. Enable open-set rejection.
3. Upload a file from `data\splits\test.csv`.
4. Show that the app predicts a registered speaker ID when the max score is above threshold.
5. Upload a file from `data\splits\unknown_test.csv`.
6. Show that the app predicts `Unknown` when the max score is below threshold.
