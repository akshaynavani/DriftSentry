"""Request / response models for the inference API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    txn_id: str | None = None
    account_id: str
    timestamp: datetime
    amount: float = Field(ge=0)
    txn_type: Literal["deposit", "withdrawal"]
    channel: Literal["branch", "atm", "online", "wire"]


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1)
    include_windows: bool = False


class WindowScore(BaseModel):
    end_txn_id: str
    error: float
    is_anomalous: bool


class AccountScore(BaseModel):
    account_id: str
    n_transactions: int
    n_windows: int
    max_error: float
    latest_error: float
    anomaly_score: float = Field(description="max_error / threshold; > 1 means anomalous")
    is_anomalous: bool
    flagged_windows: int
    top_features: list[str]
    windows: list[WindowScore] | None = None


class ScoreResponse(BaseModel):
    threshold: float
    accounts: list[AccountScore]


class DriftRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1)
    include_features: bool = True


class DriftResponse(BaseModel):
    drift_detected: bool
    n_windows: int
    reference_error_mean: float
    current_error_mean: float | None
    error_drift: dict
    drifted_features: list[str]
    feature_drift: dict | None = None


class ModelInfo(BaseModel):
    version: str
    threshold: float
    seq_len: int
    stride: int
    features: list[str]
    metrics: dict


class Health(BaseModel):
    status: str
    model_loaded: bool
