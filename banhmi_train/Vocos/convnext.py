# Ported from the official gemelo-ai/vocos repo's vocos/modules.py
# (ConvNeXtBlock) and vocos/models.py (VocosBackbone) -- verified against
# the actual source (not reconstructed from memory) via
# raw.githubusercontent.com/gemelo-ai/vocos/main/vocos/{modules,models}.py.
# Vocos: Siuzdak, "Closing the gap between time-domain and Fourier-based
# neural vocoders for high-quality audio synthesis" (ICLR 2024).
#
# One deliberate omission: upstream's AdaLayerNorm (discrete
# `adanorm_num_embeddings`-based conditioning, for their multi-bandwidth
# setup) isn't ported -- this project conditions on a continuous speaker
# embedding via `gin_channels` everywhere else (see BigVGan/generator.py,
# vits/modules/generator.py), so `gin_channels` is added here instead,
# following that existing convention rather than upstream's.
from typing import Optional

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted from facebookresearch/ConvNeXt to 1D audio:
    depthwise conv (spatial mixing) -> LayerNorm -> pointwise MLP (4x
    expand, GELU, project back) -> LayerScale -> residual."""

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        return residual + x


class VocosBackbone(nn.Module):
    """Stack of ConvNeXt blocks running entirely at the *input's* time
    resolution (no upsampling/downsampling anywhere) -- Vocos's core
    departure from HiFi-GAN/BigVGAN's Generator, which upsamples to the
    waveform's sample rate via ConvTranspose1d. All the resolution change
    happens in the ISTFTHead instead (see heads.py), which is close to
    parameter-free (a fixed inverse-FFT overlap-add)."""

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        gin_channels: int = 0,
        use_f0: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, dim, 1)
        if use_f0:
            self.f0_cond = nn.Conv1d(1, dim, 1)
        self.apply(self._init_weights)
        if use_f0:
            # Zero-initialized (after _init_weights, which would otherwise
            # overwrite this with its trunc_normal_ default) so this starts as
            # a true no-op when grafted onto an already-trained checkpoint
            # that never had F0 conditioning -- matches this project's
            # Generator.f0_cond convention.
            self.f0_cond.weight.data.zero_()
            self.f0_cond.bias.data.zero_()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, g=None, f0=None) -> torch.Tensor:
        # x: [B, input_channels, T] -> [B, T, dim] (channel-last, ready for a Linear head)
        x = self.embed(x)
        if g is not None:
            x = x + self.cond(g)
        if f0 is not None:
            x = x + self.f0_cond(f0)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x
