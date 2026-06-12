# Speaker Identification Project

Course project implementation of a 10-speaker identification system on LibriSpeech `dev-clean`.

## Environment

This project is set up for Windows PowerShell with the existing Conda environment:

```powershell
conda activate speaker-id
cd C:\Users\ben04\Desktop\speaker_id_project
```

Do not recreate the environment. PyTorch and torchaudio are already installed for the local CUDA/Windows setup, so they are intentionally not pinned in `requirements.txt`.

## Dataset

The project uses `torchaudio.datasets.LIBRISPEECH` with the `dev-clean` subset.

Current `config.yaml` uses:

```yaml
data:
  root_dir: "data/raw_readable"
```

`data\raw_readable` is the working LibriSpeech extraction used by the final run. The first extraction under `data\raw` had Windows file access issues, so the archive was re-extracted into `data\raw_readable`.

Prepared split files:

- `data\splits\metadata.csv`
- `data\splits\train.csv`
- `data\splits\val.csv`
- `data\splits\test.csv`
- `data\splits\unknown_val.csv`
- `data\splits\unknown_test.csv`

Split design:

- 10 registered speakers selected automatically.
- 40 utterances per registered speaker.
- 28 train/enrollment, 6 validation, 6 test.
- 5 additional unknown speakers for open-set rejection.

## Models

Baseline:

- MFCC features: 13 MFCC + delta + delta-delta.
- Cepstral normalization.
- Mean/std utterance pooling.
- Speaker template = average enrollment embedding.
- Prediction by cosine similarity.

GMM:

- Same MFCC frame features.
- One diagonal-covariance `sklearn.mixture.GaussianMixture` per speaker.
- Runs with 4, 8, and 16 mixture components.
- Prediction by average frame log-likelihood.

ECAPA:

- Uses pretrained SpeechBrain ECAPA-TDNN as an embedding extractor.
- No ECAPA training from scratch.
- Loads locally from `models\ecapa_pretrained` without Hugging Face network access.
- Required local files:
  - `hyperparams.yaml`
  - `embedding_model.ckpt`
  - `mean_var_norm_emb.ckpt`
  - `classifier.ckpt`
  - `label_encoder.txt`

## Run Commands

From the project root:

```powershell
python scripts/01_prepare_data.py --config config.yaml
python scripts/02_train_models.py --config config.yaml
python scripts/03_evaluate_closed_set.py --config config.yaml
python scripts/04_evaluate_open_set.py --config config.yaml
python scripts/06_print_summary.py
streamlit run app.py
```

Prediction from Python:

```powershell
python -c "from src.predict import predict_file; print(predict_file('gmm_16', r'path\to\audio.flac'))"
```

Available model names after final training:

- `baseline`
- `gmm_4`
- `gmm_8`
- `gmm_16`
- `ecapa`

## Final Results

Closed-set results:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| baseline | 0.1000 | 0.0100 | 0.1000 | 0.0182 |
| gmm_4 | 0.7833 | 0.7645 | 0.7833 | 0.7526 |
| gmm_8 | 0.8333 | 0.8083 | 0.8333 | 0.8134 |
| gmm_16 | 0.9000 | 0.8274 | 0.9000 | 0.8580 |
| ecapa | 0.9000 | 0.8333 | 0.9000 | 0.8600 |

Open-set results:

| Model | Threshold | Known Accuracy | Unknown Rejection | FAR | FRR | Overall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.0000 | 0.1000 | 0.0333 | 0.9667 | 0.0000 | 0.0778 |
| gmm_4 | -53.4944 | 0.7667 | 0.2333 | 0.7667 | 0.0167 | 0.5889 |
| gmm_8 | -52.4820 | 0.8167 | 0.3333 | 0.6667 | 0.0167 | 0.6556 |
| gmm_16 | -51.6032 | 0.8667 | 0.4000 | 0.6000 | 0.0333 | 0.7111 |
| ecapa | 0.3302 | 0.9000 | 0.8667 | 0.1333 | 0.0667 | 0.8889 |

## Outputs

Metrics and predictions:

- `results\metrics\closed_set_metrics.csv`
- `results\metrics\open_set_metrics.csv`
- `results\predictions\closed_set_predictions.csv`
- `results\predictions\open_set_predictions.csv`

Figures:

- `results\figures\model_comparison_accuracy.png`
- `results\figures\model_comparison_f1.png`
- `results\figures\confusion_matrix_baseline.png`
- `results\figures\confusion_matrix_gmm_8.png`
- `results\figures\confusion_matrix_gmm_16.png`
- `results\figures\confusion_matrix_ecapa.png`
- `results\figures\open_set_score_distribution.png`

## Streamlit Demo

Start the app:

```powershell
streamlit run app.py
```

Use `ecapa` or `gmm_16` first for the best final results. Enable open-set mode to show the Known/Unknown threshold decision.
