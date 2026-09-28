# Ported from the official NVIDIA/BigVGAN repo's alias_free_torch/act.py,
# itself adapted from https://github.com/junjun3518/alias-free-torch (Apache-2.0).
"""Anti-aliased wrapper around a periodic activation: upsample 2x (headroom
above Nyquist) -> activation -> downsample 2x (drop the new high-frequency
content the activation just introduced back out). This is the piece that
turns plain Snake/SnakeBeta into BigVGAN's AMP (Anti-aliased
Multi-Periodicity) -- see amp_block.py, which is the only place this gets
used in this package."""
from torch import nn

from .resample import DownSample1d, UpSample1d


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x
