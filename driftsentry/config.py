"""Central configuration objects shared across training, scoring and serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


FEATURE_COLUMNS: list[str] = [
    "log_amount",
    "is_deposit",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log_seconds_since_prev",
    "amount_zscore_account",
    "near_threshold",
    "channel_atm",
    "channel_wire",
    "channel_online",
]


@dataclass
class DataConfig:
    n_accounts: int = 400
    days: int = 120
    anomaly_account_fraction: float = 0.15
    reporting_threshold: float = 10_000.0
    seed: int = 42


@dataclass
class WindowConfig:
    seq_len: int = 20
    stride: int = 5


@dataclass
class ModelConfig:
    n_features: int = len(FEATURE_COLUMNS)
    hidden_size: int = 64
    latent_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    val_fraction: float = 0.2
    target_recall: float = 0.94
    fallback_percentile: float = 99.0
    seed: int = 7
    device: str = "cpu"


@dataclass
class DriftSentryConfig:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)
