# Ported from the official NVIDIA/BigVGAN repo's models.py (`BigVGAN`
# class), with one deliberate change: the input to `conv_pre` there is
# `h.num_mels` (a mel-spectrogram, since upstream BigVGAN is a standalone
# vocoder trained mel->wav). Here it's `initial_channel`, fed the VITS
# flow's latent `z_slice` instead -- same role `vits/modules/generator.py`'s
# `Generator.initial_channel` plays (see that file: `self.dec(z_slice, g=g)`
# in synthesizer.py). This keeps VITS end-to-end (no mel bottleneck) while
# reusing BigVGAN's actual anti-aliased architecture for the waveform decoder.
#
# Constructor signature otherwise mirrors `vits/modules/generator.py`'s
# `Generator` (resblock/resblock_kernel_sizes/.../gin_channels) so this class
# is a drop-in swap for it once wired into synthesizer.py -- not done in this
# commit, this file only adds the module itself.
import typing

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .activations import SnakeBeta
from .alias_free_torch import Activation1d
from .amp_block import AMPBlock1, AMPBlock2
from .utils import init_weights


class BigVGANGenerator(nn.Module):
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
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        amp_block_cls = AMPBlock1 if resblock == "1" else AMPBlock2

        self.ups = nn.ModuleList()
        self.pre_up_activations = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.pre_up_activations.append(Activation1d(activation=SnakeBeta(upsample_initial_channel // (2**i))))
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
                self.resblocks.append(amp_block_cls(ch, k, d))

        self.final_activation = Activation1d(activation=SnakeBeta(ch))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g=None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = up(self.pre_up_activations[i](x))
            xs = sum(
                self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels)
            )
            x = xs / self.num_kernels

        x = self.conv_post(self.final_activation(x))
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
