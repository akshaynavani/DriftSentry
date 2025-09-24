"""Standardization and sliding-window sequence construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftsentry.config import FEATURE_COLUMNS


@dataclass
class FeatureScaler:
    """Minimal JSON-serializable standard scaler."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        values = df[list(self.columns)].to_numpy(dtype=np.float64)
        self.mean = values.mean(axis=0)
        std = values.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler is not fitted")
        values = df[list(self.columns)].to_numpy(dtype=np.float64)
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "columns": list(self.columns),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureScaler":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
            columns=tuple(data["columns"]),
        )


@dataclass
class WindowSet:
    X: np.ndarray  # (n_windows, seq_len, n_features)
    y: np.ndarray  # (n_windows,) 1 if any transaction in the window is anomalous
    account_ids: np.ndarray  # (n_windows,)
    end_txn_ids: np.ndarray  # (n_windows,) id of last transaction in each window

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(self.X[mask], self.y[mask], self.account_ids[mask], self.end_txn_ids[mask])


def make_windows(
    features: pd.DataFrame,
    scaler: FeatureScaler,
    seq_len: int,
    stride: int = 1,
    label_column: str = "is_anomaly",
) -> WindowSet:
    """Cut each account's standardized feature stream into overlapping windows.

    Accounts with fewer than ``seq_len`` transactions are left-padded by repeating their first
    transaction so that every account yields at least one window.
    """
    scaled = scaler.transform(features)
    labels = (
        features[label_column].to_numpy(dtype=np.int64)
        if label_column in features.columns
        else np.zeros(len(features), dtype=np.int64)
    )
    txn_ids = (
        features["txn_id"].to_numpy()
        if "txn_id" in features.columns
        else np.arange(len(features)).astype(str)
    )
    accounts = features["account_id"].to_numpy()

    xs, ys, accs, ends = [], [], [], []
    boundaries = np.flatnonzero(accounts[1:] != accounts[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(features)]])
    for s, e in zip(starts, stops):
        block, lab, ids = scaled[s:e], labels[s:e], txn_ids[s:e]
        if len(block) < seq_len:
            pad = seq_len - len(block)
            block = np.concatenate([np.repeat(block[:1], pad, axis=0), block])
            lab = np.concatenate([np.zeros(pad, dtype=lab.dtype), lab])
            ids = np.concatenate([np.repeat(ids[:1], pad), ids])
        last_start = len(block) - seq_len
        offsets = list(range(0, last_start + 1, stride))
        if offsets[-1] != last_start:
            offsets.append(last_start)  # always cover the most recent transactions
        for o in offsets:
            xs.append(block[o : o + seq_len])
            ys.append(int(lab[o : o + seq_len].max()))
            accs.append(accounts[s])
            ends.append(ids[o + seq_len - 1])

    n_features = scaled.shape[1]
    X = np.stack(xs).astype(np.float32) if xs else np.zeros((0, seq_len, n_features), np.float32)
    return WindowSet(X, np.asarray(ys, dtype=np.int64), np.asarray(accs), np.asarray(ends))


def split_accounts(
    account_ids: np.ndarray, fractions: tuple[float, ...], seed: int = 0
) -> list[np.ndarray]:
    """Randomly partition unique account ids into groups of the given fractions."""
    unique = np.unique(account_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cuts = np.cumsum([int(round(f * len(unique))) for f in fractions[:-1]])
    return list(np.split(unique, cuts))
