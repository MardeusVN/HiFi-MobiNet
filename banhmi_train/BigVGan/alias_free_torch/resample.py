# Ported from the official NVIDIA/BigVGAN repo's alias_free_torch/resample.py,
# itself adapted from https://github.com/junjun3518/alias-free-torch (Apache-2.0).
"""2x up/down-sampling around a periodic activation (see act.Activation1d),
built from the same windowed-sinc low-pass filter as filter.py."""
import torch
from torch import nn
from torch.nn import functional as F

from .filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: "int | None" = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter_ = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        x = x[..., self.pad_left : -self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: "int | None" = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)
