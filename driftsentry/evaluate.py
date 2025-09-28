"""Classification metrics for reconstruction-error based anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def detection_report(errors: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    preds = (errors >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "threshold": float(threshold),
        "n_windows": int(len(labels)),
        "n_anomalous": int(labels.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
    }
    if 0 < labels.sum() < len(labels):
        report["roc_auc"] = round(float(roc_auc_score(labels, errors)), 4)
        report["average_precision"] = round(float(average_precision_score(labels, errors)), 4)
    return report


def recall_by_anomaly_type(
    errors: np.ndarray, window_types: np.ndarray, threshold: float
) -> dict[str, float]:
    """Recall broken down by the anomaly type present in each window."""
    out: dict[str, float] = {}
    for kind in sorted(set(window_types) - {"none"}):
        mask = window_types == kind
        out[kind] = round(float((errors[mask] >= threshold).mean()), 4)
    return out
