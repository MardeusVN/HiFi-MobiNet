# Ported from the official NVIDIA/BigVGAN repo's models.py: AMPBlock1 and
# AMPBlock2 -- the Anti-aliased Multi-Periodicity residual blocks that
# replace HiFi-GAN's plain ResBlock1/ResBlock2 (compare against
# vits/utils/resblocks.py, which is the un-anti-aliased ResBlock1/ResBlock2
# this project already had). Same conv layout as those two classes; the
# only structural difference is every activation call goes through
# `alias_free_torch.act.Activation1d(SnakeBeta(...))` instead of a bare
# nonlinearity, so the new high-frequency content Snake introduces gets
# filtered instead of aliasing back in-band.
import typing

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .activations import SnakeBeta
from .alias_free_torch import Activation1d
from .utils import get_padding, init_weights


def _amp_activation(channels: int) -> Activation1d:
    return Activation1d(activation=SnakeBeta(channels))


class AMPBlock1(nn.Module):
    """Three (dilation, dilation=1) conv pairs -- AMP counterpart of
    ResBlock1 (used when `resblock="1"`, the higher-capacity/quality
    config)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: typing.Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([_amp_activation(channels) for _ in range(self.num_layers)])

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = a2(xt)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in (*self.convs1, *self.convs2):
            remove_weight_norm(layer)


class AMPBlock2(nn.Module):
    """Two-conv variant -- AMP counterpart of ResBlock2 (used when
    `resblock="2"`, the lighter/faster config, and this project's default
    -- see `training.py`'s `resblock: str = "2"`)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: typing.Tuple[int, ...] = (1, 3),
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)
        self.num_layers = len(self.convs)
        self.activations = nn.ModuleList([_amp_activation(channels) for _ in range(self.num_layers)])

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            if x_mask is not None:
                xt = xt * x_mask
            x = c(xt) + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)
