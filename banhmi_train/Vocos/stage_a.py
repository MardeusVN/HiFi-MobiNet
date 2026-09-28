"""Stage A -- MarGan's frequency-domain phase refiner (docs/margan_design.md
SS3.1). Runs on the Vocos head's predicted (mag, cos_phi_hat, sin_phi_hat)
*before* iSTFT, correcting phase only -- magnitude passes through unchanged.

Formula (SS3.1, all steps proven wrap-safe/finite in that doc's SS3.1+SS9):
    (delta_c, delta_s) = phi_A(mag, cos_phi_hat, sin_phi_hat)   # raw Conv2D
    Delta_c, Delta_s    = k*tanh(delta_c), k*tanh(delta_s)      # bounded
    c', s'              = cos_phi_hat + Delta_c, sin_phi_hat + Delta_s
    (cos_phi', sin_phi')= (c', s') / sqrt(c'^2 + s'^2 + eps)

The tanh(*k) bound is load-bearing, not decorative: without it, a learned
correction that happens to point opposite the original (cos,sin) can drive
c'^2+s'^2 toward 0, and eps stops being negligible in the re-projection.
With k=0.5, margan_design.md SS3.1 proves c'^2+s'^2 is bounded below by
(1-0.5*sqrt(2))^2 ~= 0.0858 unconditionally, regardless of what phi_A
learns -- so the eps guard never actually has to do any work.
"""
import torch
from torch import nn


class StageA(nn.Module):
    def __init__(self, channels: int = 48, k: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.k = k
        self.eps = eps
        # Conv2D over the (freq, frame) grid -- 513 freq bins x T_frame at
        # n_fft=1024 (margan_design.md SS6 commits to n_fft=1024, SS1's
        # ablation showed 512 makes reconstruction worse, not just cheaper).
        # ~43.8K params at channels=48, 3 conv2d(3x3) hidden layers -- inside
        # SS6's 40-80K target.
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=3, padding=1),
        )
        # Zero-init the last conv so Stage A starts as a no-op (raw
        # correction (delta_c, delta_s) = 0 at init) -- same
        # graft-without-disturbing pattern already used elsewhere in this
        # project (flow_block.py's zero-initialized `post` layer,
        # gin_channels' zero-initialized conditioning convs).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        mag: torch.Tensor,
        cos_phi: torch.Tensor,
        sin_phi: torch.Tensor,
        return_correction: bool = False,
    ):
        # mag, cos_phi, sin_phi: each [B, F, T_frame] (F=513 freq bins,
        # from ISTFTHead's mag/phase split -- see heads.py). Stacked as
        # channels so Conv2D sees a proper (F, T_frame) 2D grid.
        x = torch.stack([mag, cos_phi, sin_phi], dim=1)  # [B, 3, F, T]
        raw = self.net(x)  # [B, 2, F, T]
        delta_c = self.k * torch.tanh(raw[:, 0])
        delta_s = self.k * torch.tanh(raw[:, 1])
        c = cos_phi + delta_c
        s = sin_phi + delta_s
        denom = torch.sqrt(c**2 + s**2 + self.eps)
        cos_phi2, sin_phi2 = c / denom, s / denom
        if return_correction:
            # (Delta_c, Delta_s) as actually added to (cos_phi, sin_phi),
            # i.e. post-tanh*k, pre-re-projection -- exactly what Rui ro 4
            # in margan_design.md SS12 asks to monitor (is the learned
            # correction meaningfully nonzero, or did Stage A learn to be a
            # no-op?), not the unbounded raw (delta_c, delta_s) before tanh.
            return cos_phi2, sin_phi2, delta_c, delta_s
        return cos_phi2, sin_phi2
