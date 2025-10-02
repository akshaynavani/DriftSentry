"""Distribution-shift monitoring via KS test and Population Stability Index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

PSI_EPS = 1e-4


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """PSI using quantile bins of the reference distribution.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if reference.size == 0 or current.size == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:  # constant reference (e.g. binary feature that never fires)
        ref_frac = np.array([1.0, 0.0])
        same = np.isclose(current, reference[0])
        cur_frac = np.array([same.mean(), 1 - same.mean()])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_frac = np.histogram(reference, bins=edges)[0] / reference.size
        cur_frac = np.histogram(current, bins=edges)[0] / current.size
    ref_frac = np.clip(ref_frac, PSI_EPS, None)
    cur_frac = np.clip(cur_frac, PSI_EPS, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


@dataclass
class DriftThresholds:
    ks_pvalue: float = 0.01
    psi_moderate: float = 0.1
    psi_significant: float = 0.25


def _psi_level(psi: float, t: DriftThresholds) -> str:
    if psi >= t.psi_significant:
        return "significant"
    if psi >= t.psi_moderate:
        return "moderate"
    return "stable"


class DriftMonitor:
    """Compares a batch against reference reconstruction errors and feature values."""

    def __init__(
        self,
        reference_errors: np.ndarray,
        reference_features: np.ndarray,
        feature_names: list[str],
        thresholds: DriftThresholds | None = None,
    ):
        self.reference_errors = np.asarray(reference_errors, dtype=np.float64)
        self.reference_features = np.asarray(reference_features, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.thresholds = thresholds or DriftThresholds()

    def _compare(self, ref: np.ndarray, cur: np.ndarray) -> dict:
        ks = ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur)
        return {
            "ks_statistic": round(float(ks.statistic), 4),
            "ks_pvalue": float(ks.pvalue),
            "psi": round(psi, 4),
            "psi_level": _psi_level(psi, self.thresholds),
            "drifted": bool(
                ks.pvalue < self.thresholds.ks_pvalue and psi >= self.thresholds.psi_moderate
            ),
        }

    def check(self, errors: np.ndarray, features: np.ndarray | None = None) -> dict:
        """Return a drift report.

        Args:
            errors: per-window reconstruction errors for the current batch.
            features: optional raw (unscaled) per-transaction features, shape (n, n_features).
        """
        errors = np.asarray(errors, dtype=np.float64)
        report: dict = {
            "n_windows": int(errors.size),
            "error_drift": self._compare(self.reference_errors, errors),
            "reference_error_mean": round(float(self.reference_errors.mean()), 6),
            "current_error_mean": round(float(errors.mean()), 6) if errors.size else None,
        }
        drifted_features: list[str] = []
        if features is not None and len(features):
            features = np.asarray(features, dtype=np.float64)
            per_feature = {}
            for i, name in enumerate(self.feature_names):
                res = self._compare(self.reference_features[:, i], features[:, i])
                per_feature[name] = res
                if res["drifted"]:
                    drifted_features.append(name)
            report["feature_drift"] = per_feature
        report["drifted_features"] = drifted_features
        report["drift_detected"] = bool(report["error_drift"]["drifted"] or drifted_features)
        return report
