"""Inference service: turns raw transactions into anomaly scores and drift reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from driftsentry.artifacts import ModelBundle
from driftsentry.config import FEATURE_COLUMNS
from driftsentry.data.features import build_features
from driftsentry.data.windowing import make_windows
from driftsentry.drift import DriftMonitor
from driftsentry.scoring import feature_contributions, reconstruction_errors

TOP_K_FEATURES = 3


class InferenceService:
    def __init__(self, bundle: ModelBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = device
        self.monitor = DriftMonitor(
            bundle.reference_errors, bundle.reference_features, list(bundle.scaler.columns)
        )

    @staticmethod
    def _to_frame(transactions: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(transactions)
        if "txn_id" not in df.columns:
            df["txn_id"] = None
        missing = df["txn_id"].isna()
        df.loc[missing, "txn_id"] = [f"req_{i}" for i in np.flatnonzero(missing.to_numpy())]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        return df

    def _windows(self, transactions: list[dict], stride: int):
        feats = build_features(self._to_frame(transactions), self.bundle.reporting_threshold)
        windows = make_windows(
            feats, self.bundle.scaler, self.bundle.window_cfg.seq_len, stride=stride
        )
        return feats, windows

    def score(self, transactions: list[dict], include_windows: bool = False) -> dict:
        feats, windows = self._windows(transactions, stride=1)
        errors = reconstruction_errors(self.bundle.model, windows.X, device=self.device)
        contribs = feature_contributions(self.bundle.model, windows.X, device=self.device)
        thr = self.bundle.threshold
        counts = feats.groupby("account_id").size()

        accounts = []
        for account_id in pd.unique(windows.account_ids):
            idx = np.flatnonzero(windows.account_ids == account_id)
            acc_err = errors[idx]
            flagged = acc_err >= thr
            worst = idx[int(np.argmax(acc_err))]
            top = np.argsort(contribs[worst])[::-1][:TOP_K_FEATURES]
            entry = {
                "account_id": str(account_id),
                "n_transactions": int(counts.loc[account_id]),
                "n_windows": int(idx.size),
                "max_error": float(acc_err.max()),
                "latest_error": float(acc_err[-1]),
                "anomaly_score": float(acc_err.max() / thr) if thr > 0 else 0.0,
                "is_anomalous": bool(flagged.any()),
                "flagged_windows": int(flagged.sum()),
                "top_features": [FEATURE_COLUMNS[i] for i in top],
            }
            if include_windows:
                entry["windows"] = [
                    {
                        "end_txn_id": str(windows.end_txn_ids[i]),
                        "error": float(errors[i]),
                        "is_anomalous": bool(errors[i] >= thr),
                    }
                    for i in idx
                ]
            accounts.append(entry)
        return {"threshold": thr, "accounts": accounts}

    def drift(self, transactions: list[dict], include_features: bool = True) -> dict:
        feats, windows = self._windows(transactions, stride=self.bundle.window_cfg.stride)
        errors = reconstruction_errors(self.bundle.model, windows.X, device=self.device)
        raw = feats[FEATURE_COLUMNS].to_numpy(dtype=np.float64) if include_features else None
        report = self.monitor.check(errors, raw)
        if not include_features:
            report.pop("feature_drift", None)
        return report

    def info(self) -> dict:
        b = self.bundle
        return {
            "version": b.version,
            "threshold": b.threshold,
            "seq_len": b.window_cfg.seq_len,
            "stride": b.window_cfg.stride,
            "features": list(b.scaler.columns),
            "metrics": b.metrics,
        }
