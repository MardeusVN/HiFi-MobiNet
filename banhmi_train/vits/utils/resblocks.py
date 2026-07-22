"""HiFi-GAN-style residual dilated-conv blocks used inside Generator's
upsampling stack, using BigVGAN's Snake activation instead of LeakyReLU.
"""
import typing

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .commons import get_padding, init_weights
from .normalization import Snake1d

ActivationCls = typing.Callable[[int], nn.Module]


class ResBlock1(nn.Module):
    """Three (dilation, dilation=1) conv pairs, each gated by Snake."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: typing.Tuple[int, ...] = (1, 3, 5),
        activation_cls: ActivationCls = Snake1d,
    ):
        super().__init__()
        self.snakes1 = nn.ModuleList([activation_cls(channels) for _ in dilation])
        self.snakes2 = nn.ModuleList([activation_cls(channels) for _ in dilation])
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

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        for snake1, c1, snake2, c2 in zip(self.snakes1, self.convs1, self.snakes2, self.convs2):
            xt = snake1(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = snake2(c1(xt))
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


class ResBlock2(nn.Module):
    """Two-conv-pair variant of ResBlock1 (fewer dilations, lighter)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: typing.Tuple[int, ...] = (1, 3),
        activation_cls: ActivationCls = Snake1d,
    ):
        super().__init__()
        self.snakes = nn.ModuleList([activation_cls(channels) for _ in dilation])
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        for snake, conv in zip(self.snakes, self.convs):
            xt = snake(x)
            if x_mask is not None:
                xt = xt * x_mask
            x = conv(xt) + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)
