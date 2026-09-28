# Custom variant, NOT part of the official NVIDIA/BigVGAN repo -- see
# generator.py for the faithful port. This trades some of full BigVGAN's
# anti-aliasing coverage for speed, based on where aliasing actually gets
# introduced.
"""BigVGANGeneratorLite: AMP (anti-aliased Snake) applied ONLY at the 4
"boundary" activations that sit right next to a temporal-resolution change
-- the 3 pre-upsample activations and the final one, each immediately
before/after a strided `ConvTranspose1d`. That's where Snake's new
high-frequency content actually risks aliasing against a *changing* sample
rate. The resblocks (18 of the 22 activation calls in `generator.py`'s
medium config) run entirely within one fixed resolution and reuse
`use_snake=True`'s plain SnakeBeta (`vits.utils.resblocks.ResBlock1`/
`ResBlock2` -- the exact resblocks the currently-running training job
already uses), no anti-aliasing wrapper.

Benchmark this against `generator.py` (full AMP) and `vits/modules/
generator.py` (`use_snake=True`, no AMP at all) before trusting it's a good
trade -- reduced AMP coverage is a real design choice, not free.
"""
import typing

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from ..vits.utils.resblocks import ResBlock1, ResBlock2
from .activations import SnakeBeta
from .alias_free_torch import Activation1d
from .utils import init_weights


class BigVGANGeneratorLite(nn.Module):
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
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

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
                # No activation_cls override -> ResBlock1/2's own default
                # (vits.utils.normalization.SnakeBeta), plain, no AMP.
                self.resblocks.append(resblock_cls(ch, k, d))

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
