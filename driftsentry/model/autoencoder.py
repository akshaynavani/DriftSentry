"""Sequence-to-sequence LSTM autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from driftsentry.config import ModelConfig


class LSTMAutoencoder(nn.Module):
    """Encode a transaction window into a latent vector and reconstruct it.

    encoder LSTM -> last hidden state -> linear to latent -> repeated ``seq_len`` times ->
    decoder LSTM -> linear projection back to feature space.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        lstm_dropout = c.dropout if c.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=c.n_features,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_latent = nn.Linear(c.hidden_size, c.latent_size)
        self.from_latent = nn.Linear(c.latent_size, c.hidden_size)
        self.decoder = nn.LSTM(
            input_size=c.hidden_size,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output = nn.Linear(c.hidden_size, c.n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        return self.to_latent(h_n[-1])

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        h = torch.relu(self.from_latent(z))
        repeated = h.unsqueeze(1).expand(-1, seq_len, -1)
        out, _ = self.decoder(repeated)
        return self.output(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.size(1))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window mean squared reconstruction error, shape ``(batch,)``."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def feature_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window, per-feature mean squared error, shape ``(batch, n_features)``."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)
