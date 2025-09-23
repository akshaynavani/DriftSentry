# DriftSentry — Implementation Plan

LSTM Autoencoder for transaction anomaly detection. The model is trained on sequences of
"normal" per-account financial transaction features; at inference time, sequences that the
autoencoder reconstructs poorly are flagged as anomalous (e.g. structuring / unusual deposit
bursts), and the distribution of reconstruction errors is monitored over time to detect
distribution shift (drift). The trained model is served through a FastAPI inference service.

## Stack
- Python 3.10+, PyTorch (model), Pandas / NumPy (data + features), SciPy (drift statistics)
- FastAPI + Uvicorn + Pydantic (serving), pytest + httpx (tests)

## Key design decisions
- **Synthetic data**: no real financial data is available, so a reproducible generator produces
  per-account transaction streams with injected anomaly types (deposit bursts, structuring just
  under a reporting threshold, odd-hour activity, amount spikes). Labels are kept for evaluation.
- **Sequences**: transactions are grouped by account, sorted by time, featurized, standardized
  with a scaler fitted on training data only, then cut into sliding windows of length `seq_len`.
  A window is labeled anomalous if it contains any injected anomalous transaction.
- **Model**: encoder LSTM -> latent vector -> repeated across time -> decoder LSTM -> linear head.
  Trained with MSE only on normal windows.
- **Threshold**: chosen on a validation split to hit a target recall (default 0.94) on held-out
  anomalies, with fallback to a percentile of normal-window errors.
- **Drift**: compare a batch's reconstruction-error distribution and per-feature distributions
  against a stored reference using the two-sample KS test and Population Stability Index (PSI).
- **Artifacts**: model weights, scaler stats, threshold and reference error sample stored in a
  single directory (`artifacts/`) loaded by the API at startup.

## Phases

### Phase 1 — Scaffold, synthetic data and feature pipeline
- `pyproject.toml`, `requirements.txt`, `driftsentry/__init__.py`, `driftsentry/config.py`
- `driftsentry/data/synthetic.py` — transaction generator with anomaly injection
- `driftsentry/data/features.py` — per-transaction feature engineering
- `driftsentry/data/windowing.py` — scaler + sliding window sequence builder

### Phase 2 — LSTM Autoencoder and training
- `driftsentry/model/autoencoder.py` — `LSTMAutoencoder`
- `driftsentry/model/dataset.py` — torch `Dataset` wrapper
- `driftsentry/train.py` — training loop with early stopping, artifact saving

### Phase 3 — Scoring, thresholding, evaluation and drift detection
- `driftsentry/scoring.py` — reconstruction errors, threshold selection
- `driftsentry/evaluate.py` — recall / precision / F1 / ROC-AUC report on held-out anomalies
- `driftsentry/drift.py` — KS test + PSI drift monitor
- `driftsentry/artifacts.py` — save/load bundle

### Phase 4 — FastAPI inference service
- `driftsentry/api/schemas.py`, `driftsentry/api/service.py`, `driftsentry/api/main.py`
- Endpoints: `GET /health`, `GET /model/info`, `POST /score`, `POST /drift`
- `.env.example`, `Dockerfile`

### Phase 5 — CLI and tests
- `driftsentry/cli.py` — `generate`, `train`, `evaluate`, `serve` commands
- `tests/` — unit tests for features, windowing, model shapes, scoring, drift, and API
