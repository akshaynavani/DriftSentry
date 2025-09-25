"""Training loop for the LSTM autoencoder."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from driftsentry.config import ModelConfig, TrainConfig
from driftsentry.model import LSTMAutoencoder
from driftsentry.model.dataset import WindowDataset

log = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
    return total / max(count, 1)


def fit_autoencoder(
    train_X: np.ndarray,
    val_X: np.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[LSTMAutoencoder, TrainHistory]:
    """Train on normal windows with early stopping on validation reconstruction loss."""
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    model = LSTMAutoencoder(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        WindowDataset(train_X), batch_size=train_cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(WindowDataset(val_X), batch_size=train_cfg.batch_size)

    history = TrainHistory()
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(train_cfg.epochs):
        tr = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        va = _run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(va)
        history.train_loss.append(tr)
        history.val_loss.append(va)
        log.info("epoch %03d train=%.5f val=%.5f", epoch + 1, tr, va)
        if va < history.best_val_loss - 1e-6:
            history.best_val_loss, history.best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= train_cfg.patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history
