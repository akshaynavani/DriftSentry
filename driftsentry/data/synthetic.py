"""Synthetic per-account transaction streams with injected anomalies.

Each account has a "persona" that controls its typical transaction rate, amount scale,
deposit ratio, active hours and preferred channels. A subset of accounts receive one or
more anomalous episodes:

- ``deposit_burst``: many deposits in a short period, far above the account's usual rate.
- ``structuring``: repeated cash deposits just below the reporting threshold.
- ``odd_hours``: a run of large withdrawals in the middle of the night via wire.
- ``amount_spike``: a handful of transactions with amounts orders of magnitude larger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driftsentry.config import DataConfig

CHANNELS = ["branch", "atm", "online", "wire"]
ANOMALY_TYPES = ["deposit_burst", "structuring", "odd_hours", "amount_spike"]
START = pd.Timestamp("2025-01-01")


def _persona(rng: np.random.Generator) -> dict:
    return {
        "rate_per_day": rng.uniform(0.4, 3.0),
        "amount_mu": rng.uniform(3.5, 6.5),  # log-amount mean
        "amount_sigma": rng.uniform(0.3, 0.8),
        "deposit_ratio": rng.uniform(0.25, 0.6),
        "active_hour_center": rng.uniform(9, 19),
        "channel_probs": rng.dirichlet([3.0, 2.0, 4.0, 0.5]),
    }


def _normal_transactions(
    account_id: str, persona: dict, days: int, rng: np.random.Generator, shift: float = 0.0
) -> pd.DataFrame:
    n = rng.poisson(persona["rate_per_day"] * days)
    n = max(n, 5)
    day_offsets = np.sort(rng.uniform(0, days, size=n))
    days_int = np.floor(day_offsets)
    hours = np.clip(rng.normal(persona["active_hour_center"], 2.5, size=n), 0, 23.99)
    timestamps = START + pd.to_timedelta(days_int, unit="D") + pd.to_timedelta(hours, unit="h")
    amounts = np.exp(rng.normal(persona["amount_mu"] + shift, persona["amount_sigma"], size=n))
    is_deposit = rng.random(n) < persona["deposit_ratio"]
    channels = rng.choice(CHANNELS, size=n, p=persona["channel_probs"])
    return pd.DataFrame(
        {
            "account_id": account_id,
            "timestamp": timestamps,
            "amount": amounts.round(2),
            "txn_type": np.where(is_deposit, "deposit", "withdrawal"),
            "channel": channels,
            "is_anomaly": 0,
            "anomaly_type": "none",
        }
    )


def _anomalous_episode(
    account_id: str,
    persona: dict,
    days: int,
    cfg: DataConfig,
    rng: np.random.Generator,
    kind: str,
) -> pd.DataFrame:
    start_day = rng.uniform(5, days - 5)
    if kind == "deposit_burst":
        n = int(rng.integers(12, 30))
        offsets_h = np.sort(rng.uniform(0, 48, size=n))
        amounts = np.exp(rng.normal(persona["amount_mu"] + 0.8, 0.4, size=n))
        txn = ["deposit"] * n
        channels = rng.choice(["atm", "online"], size=n)
    elif kind == "structuring":
        n = int(rng.integers(6, 14))
        offsets_h = np.sort(rng.uniform(0, 24 * 7, size=n))
        amounts = cfg.reporting_threshold * rng.uniform(0.86, 0.995, size=n)
        txn = ["deposit"] * n
        channels = rng.choice(["branch", "atm"], size=n)
    elif kind == "odd_hours":
        n = int(rng.integers(5, 12))
        nights = rng.integers(0, 5, size=n)
        offsets_h = np.sort(nights * 24 + rng.uniform(1.0, 4.5, size=n))
        amounts = np.exp(rng.normal(persona["amount_mu"] + 1.5, 0.3, size=n))
        txn = ["withdrawal"] * n
        channels = np.array(["wire"] * n)
    elif kind == "amount_spike":
        n = int(rng.integers(3, 7))
        offsets_h = np.sort(rng.uniform(0, 24 * 3, size=n))
        amounts = np.exp(rng.normal(persona["amount_mu"] + 3.0, 0.3, size=n))
        txn = list(rng.choice(["deposit", "withdrawal"], size=n))
        channels = rng.choice(["wire", "online"], size=n)
    else:
        raise ValueError(f"unknown anomaly type: {kind}")

    timestamps = START + pd.to_timedelta(start_day, unit="D") + pd.to_timedelta(offsets_h, unit="h")
    return pd.DataFrame(
        {
            "account_id": account_id,
            "timestamp": timestamps,
            "amount": np.asarray(amounts).round(2),
            "txn_type": txn,
            "channel": channels,
            "is_anomaly": 1,
            "anomaly_type": kind,
        }
    )


def generate_transactions(
    cfg: DataConfig | None = None,
    amount_shift: float = 0.0,
    anomalies: bool = True,
    account_prefix: str = "acct",
) -> pd.DataFrame:
    """Generate a transaction table.

    Args:
        cfg: data configuration.
        amount_shift: additive shift to every account's log-amount mean; use a non-zero
            value to simulate a population-level distribution shift for drift testing.
        anomalies: whether to inject anomalous episodes at all.
        account_prefix: prefix for generated account ids.
    """
    cfg = cfg or DataConfig()
    rng = np.random.default_rng(cfg.seed)
    frames: list[pd.DataFrame] = []
    for i in range(cfg.n_accounts):
        account_id = f"{account_prefix}_{i:05d}"
        persona = _persona(rng)
        frames.append(_normal_transactions(account_id, persona, cfg.days, rng, shift=amount_shift))
        if anomalies and rng.random() < cfg.anomaly_account_fraction:
            n_episodes = int(rng.integers(1, 3))
            for kind in rng.choice(ANOMALY_TYPES, size=n_episodes, replace=False):
                frames.append(_anomalous_episode(account_id, persona, cfg.days, cfg, rng, str(kind)))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    df.insert(0, "txn_id", [f"tx_{i:08d}" for i in range(len(df))])
    return df
