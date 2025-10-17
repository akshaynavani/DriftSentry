import numpy as np
import torch

from driftsentry.artifacts import load_bundle, save_bundle
from driftsentry.config import ModelConfig
from driftsentry.evaluate import detection_report
from driftsentry.model import LSTMAutoencoder
from driftsentry.scoring import reconstruction_errors, select_threshold


def test_autoencoder_output_shape():
    cfg = ModelConfig(n_features=5, hidden_size=8, latent_size=3)
    model = LSTMAutoencoder(cfg)
    x = torch.randn(4, 10, 5)
    assert model(x).shape == x.shape
    assert model.encode(x).shape == (4, 3)
    assert model.reconstruction_error(x).shape == (4,)
    assert model.feature_errors(x).shape == (4, 5)


def test_autoencoder_can_overfit_a_batch():
    torch.manual_seed(0)
    model = LSTMAutoencoder(ModelConfig(n_features=3, hidden_size=16, latent_size=8))
    x = torch.randn(8, 6, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        loss = ((model(x) - x) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_select_threshold_hits_target_recall():
    rng = np.random.default_rng(0)
    normal = rng.normal(1.0, 0.2, 1000)
    anomalous = rng.normal(3.0, 0.8, 200)
    errors = np.concatenate([normal, anomalous])
    labels = np.concatenate([np.zeros(1000), np.ones(200)])
    res = select_threshold(errors, labels, target_recall=0.94)
    assert res.val_recall >= 0.94
    assert detection_report(errors, labels, res.threshold)["recall"] >= 0.94


def test_select_threshold_fallback_without_anomalies():
    errors = np.linspace(0, 1, 101)
    res = select_threshold(errors, np.zeros(101), fallback_percentile=99)
    assert res.method.startswith("percentile")
    assert res.val_recall is None
    assert np.isclose(res.threshold, 0.99)


def test_detection_report_counts():
    report = detection_report(np.array([0.1, 0.9, 0.8, 0.2]), np.array([0, 1, 0, 1]), 0.5)
    assert (report["tp"], report["fp"], report["fn"], report["tn"]) == (1, 1, 1, 1)
    assert report["recall"] == 0.5


def test_pipeline_bundle_and_roundtrip(bundle, tmp_path, transactions):
    assert bundle.threshold > 0
    assert "test" in bundle.metrics and "recall" in bundle.metrics["test"]
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.threshold == bundle.threshold
    x = np.random.default_rng(0).normal(size=(3, 8, bundle.model_cfg.n_features)).astype("float32")
    np.testing.assert_allclose(
        reconstruction_errors(bundle.model, x), reconstruction_errors(loaded.model, x), rtol=1e-5
    )
