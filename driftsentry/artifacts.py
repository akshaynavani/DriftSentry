"""Persisting and loading the trained model bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from driftsentry.config import ModelConfig, WindowConfig
from driftsentry.data.windowing import FeatureScaler
from driftsentry.model import LSTMAutoencoder

MODEL_FILE = "model.pt"
META_FILE = "meta.json"
REFERENCE_FILE = "reference.npz"


@dataclass
class ModelBundle:
    model: LSTMAutoencoder
    scaler: FeatureScaler
    threshold: float
    model_cfg: ModelConfig
    window_cfg: WindowConfig
    reporting_threshold: float
    reference_errors: np.ndarray
    reference_features: np.ndarray
    metrics: dict = field(default_factory=dict)
    version: str = "0.1.0"


def save_bundle(bundle: ModelBundle, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.model.state_dict(), out / MODEL_FILE)
    meta = {
        "version": bundle.version,
        "threshold": bundle.threshold,
        "reporting_threshold": bundle.reporting_threshold,
        "model": bundle.model_cfg.__dict__,
        "window": bundle.window_cfg.__dict__,
        "scaler": bundle.scaler.to_dict(),
        "metrics": bundle.metrics,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=float))
    np.savez_compressed(
        out / REFERENCE_FILE,
        errors=bundle.reference_errors,
        features=bundle.reference_features,
    )
    return out


def load_bundle(directory: str | Path, device: str = "cpu") -> ModelBundle:
    src = Path(directory)
    meta = json.loads((src / META_FILE).read_text())
    model_cfg = ModelConfig(**meta["model"])
    model = LSTMAutoencoder(model_cfg)
    state = torch.load(src / MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    ref = np.load(src / REFERENCE_FILE)
    return ModelBundle(
        model=model,
        scaler=FeatureScaler.from_dict(meta["scaler"]),
        threshold=float(meta["threshold"]),
        model_cfg=model_cfg,
        window_cfg=WindowConfig(**meta["window"]),
        reporting_threshold=float(meta["reporting_threshold"]),
        reference_errors=ref["errors"],
        reference_features=ref["features"],
        metrics=meta.get("metrics", {}),
        version=meta.get("version", "unknown"),
    )
