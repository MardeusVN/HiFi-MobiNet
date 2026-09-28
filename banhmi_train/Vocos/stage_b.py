"""Stage B -- MarGan's time-domain periodic waveform refiner
(docs/margan_design.md SS4). Runs on x0 (the waveform iSTFT already
produced), correcting harmonic/transient detail that a frequency-domain-only
refiner (Stage A) can't reach -- HiFi-GAN MRF's actual strength (SS3.2).
Returns a residual `r`; the caller adds it to x0 (x = x0 + r), matching
SS3.1's pipeline -- not applied inside this module.

NOTE on an unresolved design-doc inconsistency, not silently picked around:
SS3.1's formula writes `R_B(x_0, h)` (Stage B conditioned on both x0 *and*
the original VITS latent h, at frame rate), but SS4's own architecture
diagram only ever draws a single "Input" arrow into Stage B, with no h
conditioning path. Reconciling the two would need an undefined mechanism to
bring frame-rate h onto x0's sample-rate grid (upsample/broadcast). This
implementation follows SS4's diagram (the actual detailed architecture
spec) and takes x0 only -- h-conditioning is left unresolved, not decided.
"""
import torch
from torch import nn

from ..BigVGan.activations import SnakeBeta


class _MRFBlock(nn.Module):
    """One block of SS4's diagram: 3 parallel DWConv(k=3, dilation=1/3/5)
    branches, each followed by SnakeBeta, concatenated then fused by a 1x1
    conv, with a residual add around the whole block. RF=11 per block
    (SS4.2: parallel branches take the max, not the product); N stacked
    blocks give RF=1+10N (linear, not exponential -- also SS4.2).
    """

    _DILATIONS = (1, 3, 5)

    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size=3, dilation=d, padding=d, groups=channels)
            for d in self._DILATIONS
        )
        self.acts = nn.ModuleList(SnakeBeta(channels) for _ in self._DILATIONS)
        self.fuse = nn.Conv1d(channels * len(self._DILATIONS), channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [act(conv(x)) for conv, act in zip(self.branches, self.acts)]
        fused = self.fuse(torch.cat(branch_outs, dim=1))
        return x + fused


class StageB(nn.Module):
    # channels=224, num_blocks=2 -- ~313K params (measured, not just the
    # hand-derived estimate), inside SS6's 300-500K target and matching
    # Rui ro 3's "few layers, narrow, not many shallow layers" guidance
    # (fewer blocks was the explicit preference, not maximizing the budget).
    def __init__(self, channels: int = 224, num_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Conv1d(1, channels, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(_MRFBlock(channels) for _ in range(num_blocks))
        self.output_proj = nn.Conv1d(channels, 1, kernel_size=7, padding=3)
        # Zero-init so Stage B starts as a no-op (r=0 at init) -- same
        # graft-without-disturbing pattern as Stage A / flow_block.py's
        # zero-initialized post layer.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        # x0: [B, 1, T_sample]
        h = self.input_proj(x0)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
