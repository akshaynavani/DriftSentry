"""End-to-end training and evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from driftsentry.artifacts import ModelBundle, save_bundle
from driftsentry.config import FEATURE_COLUMNS, DriftSentryConfig
from driftsentry.data.features import build_features
from driftsentry.data.windowing import FeatureScaler, make_windows, split_accounts
from driftsentry.evaluate import detection_report, recall_by_anomaly_type
from driftsentry.scoring import reconstruction_errors, select_threshold
from driftsentry.train import fit_autoencoder

log = logging.getLogger(__name__)

MAX_REFERENCE_ROWS = 5000


def run_training(
    transactions: pd.DataFrame,
    cfg: DriftSentryConfig | None = None,
    output_dir: str | Path | None = None,
) -> ModelBundle:
    """Split by account, train on normal windows, calibrate the threshold and evaluate.

    Accounts are partitioned into train / validation / test groups so that no account's
    history leaks across splits. The autoencoder only ever sees normal windows during
    training; validation (normal + anomalous) is used to pick the threshold, and the test
    accounts provide the held-out recall numbers.
    """
    cfg = cfg or DriftSentryConfig()
    tc, wc = cfg.train, cfg.window
    feats = build_features(transactions, cfg.data.reporting_threshold)
    if "is_anomaly" not in feats.columns:
        feats["is_anomaly"] = 0

    vf = tc.val_fraction
    train_acc, val_acc, test_acc = split_accounts(
        feats["account_id"].to_numpy(), (1 - 2 * vf, vf, vf), seed=tc.seed
    )
    train_df = feats[feats["account_id"].isin(train_acc)]
    val_df = feats[feats["account_id"].isin(val_acc)]
    test_df = feats[feats["account_id"].isin(test_acc)]

    normal_train_rows = train_df[train_df["is_anomaly"] == 0]
    scaler = FeatureScaler().fit(normal_train_rows)

    train_w = make_windows(train_df, scaler, wc.seq_len, wc.stride)
    val_w = make_windows(val_df, scaler, wc.seq_len, wc.stride)
    test_w = make_windows(test_df, scaler, wc.seq_len, wc.stride)
    log.info(
        "windows: train=%d val=%d test=%d (anomalous val=%d test=%d)",
        len(train_w), len(val_w), len(test_w), val_w.y.sum(), test_w.y.sum(),
    )

    train_normal = train_w.subset(train_w.y == 0)
    val_normal = val_w.subset(val_w.y == 0)
    model, history = fit_autoencoder(train_normal.X, val_normal.X, cfg.model, tc)

    val_err = reconstruction_errors(model, val_w.X, device=tc.device)
    thr = select_threshold(val_err, val_w.y, tc.target_recall, tc.fallback_percentile)
    log.info("threshold=%.6f (%s)", thr.threshold, thr.method)

    test_err = reconstruction_errors(model, test_w.X, device=tc.device)
    test_report = detection_report(test_err, test_w.y, thr.threshold)
    if test_w.anomaly_types is not None:
        test_report["recall_by_type"] = recall_by_anomaly_type(
            test_err, test_w.anomaly_types, thr.threshold
        )
    log.info("test report: %s", test_report)

    # Reference distributions for drift monitoring come from normal training data.
    train_err = reconstruction_errors(model, train_normal.X, device=tc.device)
    rng = np.random.default_rng(tc.seed)
    ref_rows = normal_train_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if len(ref_rows) > MAX_REFERENCE_ROWS:
        ref_rows = ref_rows[rng.choice(len(ref_rows), MAX_REFERENCE_ROWS, replace=False)]

    bundle = ModelBundle(
        model=model,
        scaler=scaler,
        threshold=thr.threshold,
        model_cfg=cfg.model,
        window_cfg=wc,
        reporting_threshold=cfg.data.reporting_threshold,
        reference_errors=train_err.astype(np.float32),
        reference_features=ref_rows.astype(np.float32),
        metrics={
            "threshold_method": thr.method,
            "val_recall": thr.val_recall,
            "val_false_positive_rate": thr.val_false_positive_rate,
            "best_epoch": history.best_epoch + 1,
            "best_val_loss": history.best_val_loss,
            "test": test_report,
        },
    )
    if output_dir is not None:
        save_bundle(bundle, output_dir)
        log.info("saved model bundle to %s", output_dir)
    return bundle


def evaluate_bundle(bundle: ModelBundle, transactions: pd.DataFrame) -> dict:
    """Score a labeled transaction table with an already trained bundle."""
    feats = build_features(transactions, bundle.reporting_threshold)
    windows = make_windows(
        feats, bundle.scaler, bundle.window_cfg.seq_len, bundle.window_cfg.stride
    )
    errors = reconstruction_errors(bundle.model, windows.X)
    report = detection_report(errors, windows.y, bundle.threshold)
    if windows.anomaly_types is not None:
        report["recall_by_type"] = recall_by_anomaly_type(
            errors, windows.anomaly_types, bundle.threshold
        )
    return report
