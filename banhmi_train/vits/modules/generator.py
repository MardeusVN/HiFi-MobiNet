"""HiFi-GAN-style vocoder decoder: upsamples the flow's latent sequence
(one vector per output frame) directly to a raw waveform via transposed
convolutions + residual dilated-conv blocks, using BigVGAN's Snake
activation before each upsampling step instead of LeakyReLU.
"""
import typing

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from ..utils.commons import init_weights
from ..utils.normalization import Snake1d, SnakeBeta
from ..utils.resblocks import ResBlock1, ResBlock2


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        gin_channels: int = 0,
        use_snake_beta: bool = True,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        activation_cls = SnakeBeta if use_snake_beta else Snake1d

        self.ups = nn.ModuleList()
        self.pre_up_snakes = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.pre_up_snakes.append(activation_cls(upsample_initial_channel // (2**i)))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d, activation_cls=activation_cls))

        self.final_snake = activation_cls(ch)
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g=None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = up(self.pre_up_snakes[i](x))
            xs = sum(
                self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels)
            )
            x = xs / self.num_kernels

        x = self.conv_post(self.final_snake(x))
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
