import pytest

from driftsentry.config import DataConfig, DriftSentryConfig, ModelConfig
from driftsentry.data.synthetic import generate_transactions


@pytest.fixture(scope="session")
def small_data_cfg() -> DataConfig:
    return DataConfig(n_accounts=40, days=30, anomaly_account_fraction=0.4, seed=1)


@pytest.fixture(scope="session")
def transactions(small_data_cfg):
    return generate_transactions(small_data_cfg)


@pytest.fixture(scope="session")
def tiny_cfg(small_data_cfg) -> DriftSentryConfig:
    cfg = DriftSentryConfig(data=small_data_cfg)
    cfg.model = ModelConfig(hidden_size=16, latent_size=4)
    cfg.window.seq_len = 8
    cfg.window.stride = 4
    cfg.train.epochs = 2
    cfg.train.batch_size = 64
    return cfg


@pytest.fixture(scope="session")
def bundle(transactions, tiny_cfg):
    from driftsentry.pipeline import run_training

    return run_training(transactions, tiny_cfg)
