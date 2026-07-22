"""Normalization / activation building blocks shared by several components."""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """Channel-first LayerNorm: normalizes over the channel dim of a
    [batch, channels, time] tensor (torch's built-in LayerNorm normalizes
    over the last dim, so channels get transposed there and back)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class Snake1d(nn.Module):
    """Snake activation (BigVGAN): x + (1/alpha) * sin(alpha * x)^2.

    Learnable per-channel alpha gives the activation a periodic inductive
    bias that suits raw waveform generation better than LeakyReLU. This is
    what EdgeTTS's own code actually uses (verified: their README says
    "Snake1d activation (BigVGAN-style)", not SnakeBeta -- see
    conversation history).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x) ** 2


class SnakeBeta(nn.Module):
    """Snake-Beta activation (later BigVGAN revision): x + (1/beta) *
    sin(alpha * x)^2, with alpha (frequency) and beta (amplitude) as two
    *separate* learnable parameters instead of Snake1d's single alpha
    reused for both roles. Both are stored and updated in log-space
    (matching BigVGAN's own released configs, e.g. bigvgan_base_24khz),
    which keeps them positive and stabilizes training instead of the raw
    parameter potentially crossing zero.

    Not present in EdgeTTS (which uses plain Snake1d) -- this is a
    BanhmiTTS-only addition, opt-in via `use_snake_beta`.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.log_beta = nn.Parameter(torch.zeros(1, channels, 1))
        self._eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)
        return x + (1.0 / (beta + self._eps)) * torch.sin(alpha * x) ** 2
