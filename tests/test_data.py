import numpy as np
import pandas as pd
import pytest

from driftsentry.config import FEATURE_COLUMNS
from driftsentry.data.features import build_features
from driftsentry.data.synthetic import ANOMALY_TYPES
from driftsentry.data.windowing import FeatureScaler, make_windows, split_accounts


def test_generator_shapes_and_labels(transactions):
    assert {"account_id", "timestamp", "amount", "is_anomaly", "anomaly_type"} <= set(
        transactions.columns
    )
    assert transactions["is_anomaly"].sum() > 0
    assert set(transactions["anomaly_type"]) - {"none"} <= set(ANOMALY_TYPES)
    assert (transactions["amount"] >= 0).all()


def test_generator_is_deterministic(small_data_cfg):
    from driftsentry.data.synthetic import generate_transactions

    a = generate_transactions(small_data_cfg)
    b = generate_transactions(small_data_cfg)
    pd.testing.assert_frame_equal(a, b)


def test_features_present_and_finite(transactions):
    feats = build_features(transactions)
    values = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert set(np.unique(feats["is_deposit"])) <= {0.0, 1.0}


def test_features_are_causal(transactions):
    feats_full = build_features(transactions)
    first_acc = transactions["account_id"].iloc[0]
    sub = transactions[transactions["account_id"] == first_acc].iloc[:5]
    feats_prefix = build_features(sub)
    full_rows = feats_full[feats_full["account_id"] == first_acc].iloc[:5]
    np.testing.assert_allclose(
        feats_prefix[FEATURE_COLUMNS].to_numpy(), full_rows[FEATURE_COLUMNS].to_numpy()
    )


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame({"account_id": ["a"]}))


def test_scaler_roundtrip(transactions):
    feats = build_features(transactions)
    scaler = FeatureScaler().fit(feats)
    restored = FeatureScaler.from_dict(scaler.to_dict())
    np.testing.assert_allclose(scaler.transform(feats), restored.transform(feats))


def test_windows_shape_and_short_account_padding(transactions):
    feats = build_features(transactions)
    scaler = FeatureScaler().fit(feats)
    w = make_windows(feats, scaler, seq_len=8, stride=4)
    assert w.X.shape[1:] == (8, len(FEATURE_COLUMNS))
    assert len(w.y) == len(w.X) == len(w.anomaly_types)
    # every account yields at least one window
    assert set(w.account_ids) == set(feats["account_id"])

    short = feats[feats["account_id"] == feats["account_id"].iloc[0]].iloc[:3]
    ws = make_windows(short, scaler, seq_len=8)
    assert ws.X.shape == (1, 8, len(FEATURE_COLUMNS))


def test_window_labels_follow_transactions(transactions):
    feats = build_features(transactions)
    scaler = FeatureScaler().fit(feats)
    w = make_windows(feats, scaler, seq_len=8, stride=1)
    assert w.y.sum() > 0
    assert all((t != "none") == bool(y) for t, y in zip(w.anomaly_types, w.y))


def test_split_accounts_is_partition():
    ids = np.array([f"a{i}" for i in range(100)])
    parts = split_accounts(ids, (0.6, 0.2, 0.2), seed=3)
    assert sum(len(p) for p in parts) == 100
    assert len(set(np.concatenate(parts))) == 100
