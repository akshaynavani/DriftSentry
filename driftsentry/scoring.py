"""Reconstruction-error scoring and threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from driftsentry.model import LSTMAutoencoder


@torch.no_grad()
def reconstruction_errors(
    model: LSTMAutoencoder, X: np.ndarray, batch_size: int = 512, device: str = "cpu"
) -> np.ndarray:
    """Mean squared reconstruction error for every window in ``X``."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size])).to(device)
        out.append(model.reconstruction_error(batch).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def feature_contributions(
    model: LSTMAutoencoder, X: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Per-feature reconstruction error, shape ``(n_windows, n_features)``."""
    model.eval()
    batch = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    return model.feature_errors(batch).cpu().numpy()


@dataclass
class ThresholdResult:
    threshold: float
    method: str
    val_recall: float | None
    val_false_positive_rate: float


def select_threshold(
    errors: np.ndarray,
    labels: np.ndarray,
    target_recall: float = 0.94,
    fallback_percentile: float = 99.0,
) -> ThresholdResult:
    """Pick the highest threshold whose recall on labeled anomalies is >= ``target_recall``.

    Windows with ``error >= threshold`` are flagged. If no labeled anomalies are present,
    the threshold falls back to a percentile of the normal-window error distribution.
    """
    errors = np.asarray(errors, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    normal, anomalous = errors[~labels], errors[labels]

    if anomalous.size == 0:
        thr = float(np.percentile(normal, fallback_percentile))
        fpr = float((normal >= thr).mean()) if normal.size else 0.0
        return ThresholdResult(thr, f"percentile_{fallback_percentile:g}", None, fpr)

    sorted_anom = np.sort(anomalous)
    # Allow at most floor((1 - target) * n) anomalies to fall below the threshold.
    k = int(np.floor((1.0 - target_recall) * sorted_anom.size + 1e-9))
    thr = float(sorted_anom[min(k, sorted_anom.size - 1)])
    recall = float((anomalous >= thr).mean())
    fpr = float((normal >= thr).mean()) if normal.size else 0.0
    return ThresholdResult(thr, f"target_recall_{target_recall:g}", recall, fpr)
