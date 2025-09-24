"""Per-transaction feature engineering.

Input columns required: ``account_id``, ``timestamp``, ``amount``, ``txn_type``, ``channel``.
All features are computed causally per account (only past transactions are used), so the
same function can be applied to a live stream in the inference service.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driftsentry.config import FEATURE_COLUMNS

REQUIRED_COLUMNS = ["account_id", "timestamp", "amount", "txn_type", "channel"]
MAX_GAP_SECONDS = 30 * 24 * 3600.0


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def build_features(df: pd.DataFrame, reporting_threshold: float = 10_000.0) -> pd.DataFrame:
    """Return ``df`` sorted by account/time with the feature columns appended."""
    _validate(df)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    amount = out["amount"].astype(float).clip(lower=0.0)
    out["log_amount"] = np.log1p(amount)
    out["is_deposit"] = (out["txn_type"].str.lower() == "deposit").astype(float)

    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = out["timestamp"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    gaps = out.groupby("account_id")["timestamp"].diff().dt.total_seconds()
    out["log_seconds_since_prev"] = np.log1p(gaps.fillna(MAX_GAP_SECONDS).clip(0, MAX_GAP_SECONDS))

    # Causal z-score of the log amount against the account's own history.
    grouped = out.groupby("account_id")["log_amount"]
    past_mean = grouped.transform(lambda s: s.expanding().mean().shift(1))
    past_std = grouped.transform(lambda s: s.expanding().std().shift(1))
    z = (out["log_amount"] - past_mean) / past_std.where(past_std > 1e-6)
    out["amount_zscore_account"] = z.fillna(0.0).clip(-10, 10)

    out["near_threshold"] = (
        (amount >= 0.85 * reporting_threshold) & (amount < reporting_threshold)
    ).astype(float)

    channel = out["channel"].str.lower()
    out["channel_atm"] = (channel == "atm").astype(float)
    out["channel_wire"] = (channel == "wire").astype(float)
    out["channel_online"] = (channel == "online").astype(float)

    assert all(c in out.columns for c in FEATURE_COLUMNS)
    return out
