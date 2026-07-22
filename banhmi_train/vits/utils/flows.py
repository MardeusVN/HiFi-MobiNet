"""Invertible flow steps used to build up the normalizing-flow prior
(ResidualCouplingBlock in flow_block.py) and the stochastic duration
predictor's two flow stacks.

Each flow's forward() returns (output, logdet) when reverse=False and just
output when reverse=True -- the log-determinant of the Jacobian is only
needed for the forward (density-evaluation) direction.
"""
import math

import torch
from torch import nn

from .dds_conv import DDSConv
from .transforms import piecewise_rational_quadratic_transform


class Log(nn.Module):
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        return torch.exp(x) * x_mask


class Flip(nn.Module):
    def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).type_as(x)
        return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs):
        if not reverse:
            y = (self.m + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        return (x - self.m) * torch.exp(-self.logs) * x_mask


class ResidualCouplingLayer(nn.Module):
    """Affine coupling: splits channels in half, uses one half (through a
    WaveNet conditioner) to predict an affine transform of the other half.
    x0 always passes through unchanged, which is what keeps the whole flow
    invertible regardless of how complex the conditioner network is."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        # Imported here (not at module scope) to avoid a wavenet<->flows
        # circular import; both are small, leaf-level modules otherwise.
        from .wavenet import WN

        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        stats = self.post(h) * x_mask
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            logdet = torch.sum(logs, [1, 2])
            return torch.cat([x0, x1], 1), logdet

        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ConvFlow(nn.Module):
    """Coupling layer whose affine-transform-predictor output instead
    parameterizes a rational-quadratic spline (a strictly more expressive,
    still-invertible transform of x1)."""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ):
        super().__init__()
        self.half_channels = in_channels // 2
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.convs(self.pre(x0), x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, c*?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        return x
