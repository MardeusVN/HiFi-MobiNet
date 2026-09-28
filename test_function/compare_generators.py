#!/usr/bin/env python3
"""Head-to-head comparison of 5 generator architectures on ONE real wav:

    HiFi-GAN (vanilla LeakyReLU)  vs  HiFi-GAN + SnakeBeta (no AMP)  vs
    BigVGANGenerator (SnakeBeta + AMP, full)  vs  BigVGANGeneratorLite
    (SnakeBeta + AMP only at the 4 upsample-boundary activations)  vs
    VocosGenerator (ConvNeXt backbone + ISTFT head, no time-domain upsampling)

Pipeline per model: wav -> linear STFT magnitude (`spectrogram_torch`) ->
Generator -> waveform -> mel loss against the original (same `loss_mel`
formula -- L1 in mel space * c_mel -- that `vits/training.py` uses). This
bypasses VITS's text encoder / posterior encoder / flow entirely: the
Generator is fed the STFT directly as a stand-in for its usual latent `z`,
purely to isolate the Generator architecture as the only variable.

Random-init weights alone say nothing about which architecture reconstructs
better -- an untrained conv stack produces structured noise regardless of
input. So each model is *overfit* on this single clip for `--steps` steps
(default 800) before its loss_mel is measured; everything else (data,
steps, optimizer, LR, init seed) is held equal, so the final loss reflects
the architecture, not training budget.

Deliberately CPU-only: avoids taking VRAM/compute from the real DDP
training run this repo has going in /home/capstone/overnight_run_full.

Usage:
    python -m test_function.compare_generators
    python -m test_function.compare_generators --wav data/wavs/LJ018-0126.wav --steps 1500
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banhmi_train import wavfile  # noqa: E402
from banhmi_train.BigVGan.generator import BigVGANGenerator  # noqa: E402
from banhmi_train.BigVGan.generator_lite import BigVGANGeneratorLite  # noqa: E402
from banhmi_train.Vocos.generator import VocosGenerator  # noqa: E402
from banhmi_train.Vocos.stage_a import StageA  # noqa: E402
from banhmi_train.Vocos.stage_b import StageB  # noqa: E402
from banhmi_train.mel_processing import (  # noqa: E402
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectrogram_torch,
)
from banhmi_train.vits.modules.generator import Generator  # noqa: E402
from banhmi_train.vits.utils.commons import get_padding, init_weights  # noqa: E402
from banhmi_train.vits.utils.normalization import SnakeBeta  # noqa: E402
from banhmi_train.vits.utils.resblocks import ResBlockInverted  # noqa: E402

# Matches vits/training.py's own defaults (VitsModel.__init__), so loss_mel
# here is on the same scale as the real training run's.
N_FFT = 1024
HOP = 256
WIN = 1024
SAMPLE_RATE = 22050
MEL_CHANNELS = 80
MEL_FMIN = 0.0
MEL_FMAX = None
C_MEL = 45
LR = 2e-4
BETAS = (0.8, 0.99)
EPS = 1e-9

# vits/training.py's "medium" (project default) Generator config. Shared by
# all 3 models -- only the activation (+ AMP, for BigVGAN) differs.
GENERATOR_CONFIG = dict(
    resblock="2",
    resblock_kernel_sizes=(3, 5, 7),
    resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
    upsample_rates=(8, 8, 4),  # product = 256 = HOP, so T_wav_out == T_wav_in exactly
    upsample_initial_channel=256,
    upsample_kernel_sizes=(16, 16, 8),
    gin_channels=0,
)


def load_wav(path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz, expected {SAMPLE_RATE} Hz (resample it first)")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono-mix, just in case
    return torch.from_numpy(audio).unsqueeze(0)  # [1, T]


def crop_to_hop_multiple(y: torch.Tensor, max_seconds: float) -> torch.Tensor:
    y = y[:, : int(max_seconds * SAMPLE_RATE)]
    usable = (y.shape[1] // HOP) * HOP
    return y[:, :usable]


def ground_truth_phase(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int):
    """cos(phi_gt), sin(phi_gt) of the real audio's own complex STFT --
    needed for margan_design.md SS9's L_phi (ground truth in that formula's
    (cos phi_gt, sin phi_gt) is a genuine unit vector, unlike Stage A's own
    output -- see SS9). Frame-aligned with `spec`/`spectrogram_torch` by
    replicating its exact padding/window/center convention (mel_processing.py)
    -- only difference is `return_complex=True` instead of taking the
    magnitude, so phase survives.
    """
    window = torch.hann_window(win_size, dtype=y.dtype, device=y.device)
    pad = (n_fft - hop_size) // 2
    y_padded = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec_complex = torch.stft(
        y_padded, n_fft, hop_length=hop_size, win_length=win_size, window=window,
        center=False, pad_mode="reflect", onesided=True, return_complex=True,
    )
    mag_gt = spec_complex.abs().clamp_min(1e-9)
    return spec_complex.real / mag_gt, spec_complex.imag / mag_gt


class _VocosWithStageA(torch.nn.Module):
    """Test-only harness for margan_design.md Phase 3: backbone -> head.out
    (mag/phase) -> Stage A phase refinement -> reconstruct complex STFT ->
    head.istft. Deliberately does NOT touch VocosGenerator/ISTFTHead's own
    forward -- Phase 3 only asks for an isolated overfit-test to check
    whether Stage A helps at all before committing to real training-loop
    wiring (config flags, checkpoint compatibility, etc.), which is a
    separate, later step.
    """

    def __init__(self, vocos: VocosGenerator, stage_a: StageA):
        super().__init__()
        self.vocos = vocos
        self.stage_a = stage_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vocos.backbone(x)
        h = self.vocos.head.out(feats).transpose(1, 2)  # [B, n_fft+2, T]
        mag, phase = h.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        cos_p2, sin_p2 = self.stage_a(mag, cos_p, sin_p)
        real, imag = mag * cos_p2, mag * sin_p2
        with torch.autocast(device_type=real.device.type, enabled=False):
            spec = torch.complex(real.float(), imag.float())
            audio = self.vocos.head.istft(spec)
        return audio.unsqueeze(1)


class _VocosWithStageB(torch.nn.Module):
    """Test-only harness for margan_design.md Phase 2: plain VocosGenerator
    forward (backbone -> head -> iSTFT, no Stage A) -> Stage B residual ->
    x0 + r. Isolates Stage B's own contribution before combining with
    Stage A (SS3.2's rationale: each stage should prove independent value).
    """

    def __init__(self, vocos: VocosGenerator, stage_b: StageB):
        super().__init__()
        self.vocos = vocos
        self.stage_b = stage_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.vocos(x)  # [B, 1, T_wav]
        return x0 + self.stage_b(x0)


class _MarGan(torch.nn.Module):
    """Test-only harness for margan_design.md Phase 4: full pipeline --
    backbone -> head.out (mag/phase) -> Stage A phase refinement ->
    reconstruct complex STFT -> iSTFT -> x0 -> Stage B residual -> x0 + r.
    Combines the exact same _VocosWithStageA phase-refinement path (SS3.1)
    with _VocosWithStageB's residual add (SS4), not a new formula.
    """

    def __init__(self, vocos: VocosGenerator, stage_a: StageA, stage_b: StageB):
        super().__init__()
        self.vocos = vocos
        self.stage_a = stage_a
        self.stage_b = stage_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vocos.backbone(x)
        h = self.vocos.head.out(feats).transpose(1, 2)  # [B, n_fft+2, T]
        mag, phase = h.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        cos_p2, sin_p2 = self.stage_a(mag, cos_p, sin_p)
        real, imag = mag * cos_p2, mag * sin_p2
        with torch.autocast(device_type=real.device.type, enabled=False):
            spec = torch.complex(real.float(), imag.float())
            x0 = self.vocos.head.istft(spec).unsqueeze(1)
        return x0 + self.stage_b(x0)


class _HiFiGANInverted(torch.nn.Module):
    """Test-only harness for a new (non-MarGan) novelty direction: the exact
    same HiFi-GAN upsample scaffold as the real
    banhmi_train/vits/modules/generator.py's `Generator` (identical
    conv_pre/upsample/conv_post/tanh, same channel/kernel/stride/padding at
    GENERATOR_CONFIG's "medium" tier) -- but each stage's 3-branch dilated
    MRF (ResBlock2 x3, summed/averaged) is replaced by a single
    `banhmi_train.vits.utils.resblocks.ResBlockInverted` (pointwise-expand
    -> depthwise -> pointwise-project linear bottleneck, weight_norm +
    SnakeBeta, one residual around the whole block -- MobileNetV2/Conformer-
    conv-module shape, not ConvNeXt's depthwise-first ordering already used
    in Vocos/convnext.py). Not wired into the real Generator class yet --
    isolated overfit-test first, same discipline used for every MarGan stage.
    """

    def __init__(
        self,
        initial_channel: int,
        upsample_rates=(8, 8, 4),
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes=(16, 16, 8),
        depthwise_kernel_size: int = 7,
        expansion=2,
        blocks_per_stage: int = 1,
    ):
        super().__init__()
        # expansion: int (same at every stage) or a per-stage tuple, e.g.
        # (6, 6, 1) to keep expansion=6 at the cheap early stages (small T)
        # and drop to 1 at the last stage (T=8192, where the expand/project
        # pointwise convs' cost is most exposed -- see the RTF profiling
        # this variant exists to test).
        if isinstance(expansion, int):
            expansion = (expansion,) * len(upsample_rates)
        assert len(expansion) == len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.pre_up_acts = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        # One ModuleList of `blocks_per_stage` stacked ResBlockInverted per
        # upsample stage -- expansion alone can't reach hifigan's param
        # count here (conv_pre is a fixed 513->256 cost, over half the
        # total, that dilutes any expansion change), so stacking blocks
        # sequentially per stage is the lever that actually moves params.
        self.stage_blocks = torch.nn.ModuleList()
        ch_out = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch_in = upsample_initial_channel // (2**i)
            ch_out = upsample_initial_channel // (2 ** (i + 1))
            self.pre_up_acts.append(SnakeBeta(ch_in))
            self.ups.append(
                torch.nn.utils.weight_norm(
                    torch.nn.ConvTranspose1d(ch_in, ch_out, k, u, padding=(k - u) // 2)
                )
            )
            self.stage_blocks.append(
                torch.nn.ModuleList(
                    ResBlockInverted(ch_out, kernel_size=depthwise_kernel_size, activation_cls=SnakeBeta, expansion=expansion[i])
                    for _ in range(blocks_per_stage)
                )
            )
        self.final_act = SnakeBeta(ch_out)
        self.conv_post = torch.nn.Conv1d(ch_out, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up(self.pre_up_acts[i](x))
            for block in self.stage_blocks[i]:
                x = block(x)
        x = self.conv_post(self.final_act(x))
        return torch.tanh(x)


class _ResBlockInvertedNoExpand(torch.nn.Module):
    """Ablation of ResBlockInverted: drops the first pointwise (expand)
    conv + its activation entirely -- Depthwise -> weight_norm -> SnakeBeta
    -> Pointwise-project (linear, zero-init) -> weight_norm + residual, no
    expand step at all. This is MobileNetV1's depthwise-separable block
    shape, not MobileNetV2's inverted-residual shape. Only a meaningful
    comparison at expansion=1 (ResBlockInverted's own `pw_expand` at
    expansion=1 is channels->channels -- same width in/out, so it's not
    load-bearing for widening the depthwise the way it is at expansion>1 --
    the open question is whether that no-op-width pointwise still earns its
    params/compute, or whether it's pure overhead at expansion=1).
    """

    def __init__(self, channels, kernel_size=3, dilation=1, activation_cls=SnakeBeta):
        super().__init__()
        self.dw = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                channels, channels, kernel_size, dilation=dilation, groups=channels,
                padding=get_padding(kernel_size, dilation),
            )
        )
        self.act = activation_cls(channels)
        self.pw_project = torch.nn.utils.weight_norm(torch.nn.Conv1d(channels, channels, 1))
        for layer in (self.dw, self.pw_project):
            layer.apply(init_weights)
        # Same zero-init rationale as ResBlockInverted -- identity no-op at init.
        self.pw_project.weight_g.data.zero_()
        self.pw_project.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.dw(x)
        xt = self.act(xt)
        xt = self.pw_project(xt)
        return xt + x

    def remove_weight_norm(self):
        torch.nn.utils.remove_weight_norm(self.dw)
        torch.nn.utils.remove_weight_norm(self.pw_project)


class _HiFiGANInvertedMRFNoExpand(torch.nn.Module):
    """Same parallel-MRF mechanism as `_HiFiGANInvertedMRF` (3 branches,
    kernels 3/5/7, summed/averaged) but each branch is
    `_ResBlockInvertedNoExpand` instead of `ResBlockInverted` -- isolates
    whether the expand-pointwise (present but width-preserving at
    expansion=1) is worth its cost inside the winning MRF(1,1,1) config.
    """

    def __init__(
        self,
        initial_channel: int,
        upsample_rates=(8, 8, 4),
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes=(16, 16, 8),
        kernel_sizes=(3, 5, 7),
    ):
        super().__init__()
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.pre_up_acts = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        self.branch_sets = torch.nn.ModuleList()
        ch_out = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch_in = upsample_initial_channel // (2**i)
            ch_out = upsample_initial_channel // (2 ** (i + 1))
            self.pre_up_acts.append(SnakeBeta(ch_in))
            self.ups.append(
                torch.nn.utils.weight_norm(
                    torch.nn.ConvTranspose1d(ch_in, ch_out, k, u, padding=(k - u) // 2)
                )
            )
            self.branch_sets.append(
                torch.nn.ModuleList(
                    _ResBlockInvertedNoExpand(ch_out, kernel_size=ks, activation_cls=SnakeBeta)
                    for ks in kernel_sizes
                )
            )
        self.final_act = SnakeBeta(ch_out)
        self.conv_post = torch.nn.Conv1d(ch_out, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up(self.pre_up_acts[i](x))
            branches = self.branch_sets[i]
            x = sum(b(x) for b in branches) / len(branches)
        x = self.conv_post(self.final_act(x))
        return torch.tanh(x)


class _HiFiGANInvertedMRF(torch.nn.Module):
    """Test-only harness: HiFi-GAN's *actual* MRF mechanism (N branches run
    in PARALLEL on the same input, summed then divided by N) but each
    branch is a ResBlockInverted instead of ResBlock1/2 -- `_HiFiGANInverted`
    above only ever stacks blocks_per_stage of them SEQUENTIALLY (one
    block's output feeds the next), never in parallel with different
    kernels the way HiFi-GAN's own MRF does. Tests whether HiFi-GAN's
    actual multi-receptive-field idea (not just picking one kernel) helps
    when combined with depthwise-separable blocks.
    """

    def __init__(
        self,
        initial_channel: int,
        upsample_rates=(8, 8, 4),
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes=(16, 16, 8),
        kernel_sizes=(3, 5, 7),
        expansion=1,
    ):
        super().__init__()
        if isinstance(expansion, int):
            expansion = (expansion,) * len(upsample_rates)
        assert len(expansion) == len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.pre_up_acts = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        # One ModuleList of len(kernel_sizes) PARALLEL branches per stage,
        # summed/averaged in forward() -- matches HiFi-GAN's ResBlock1/2
        # MRF mechanism exactly, just with ResBlockInverted as the branch.
        self.branch_sets = torch.nn.ModuleList()
        ch_out = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch_in = upsample_initial_channel // (2**i)
            ch_out = upsample_initial_channel // (2 ** (i + 1))
            self.pre_up_acts.append(SnakeBeta(ch_in))
            self.ups.append(
                torch.nn.utils.weight_norm(
                    torch.nn.ConvTranspose1d(ch_in, ch_out, k, u, padding=(k - u) // 2)
                )
            )
            self.branch_sets.append(
                torch.nn.ModuleList(
                    ResBlockInverted(ch_out, kernel_size=ks, activation_cls=SnakeBeta, expansion=expansion[i])
                    for ks in kernel_sizes
                )
            )
        self.final_act = SnakeBeta(ch_out)
        self.conv_post = torch.nn.Conv1d(ch_out, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up(self.pre_up_acts[i](x))
            branches = self.branch_sets[i]
            x = sum(b(x) for b in branches) / len(branches)
        x = self.conv_post(self.final_act(x))
        return torch.tanh(x)


def build_generator(kind: str, spec_channels: int) -> torch.nn.Module:
    if kind == "hifigan":
        return Generator(initial_channel=spec_channels, use_snake=False, **GENERATOR_CONFIG)
    if kind == "hifigan_snake":
        return Generator(initial_channel=spec_channels, use_snake=True, **GENERATOR_CONFIG)
    if kind == "bigvgan":
        return BigVGANGenerator(initial_channel=spec_channels, **GENERATOR_CONFIG)
    if kind == "bigvgan_lite":
        return BigVGANGeneratorLite(initial_channel=spec_channels, **GENERATOR_CONFIG)
    if kind == "hifigan_inverted":
        # New novelty direction (not MarGan): baseline's exact upsample
        # scaffold, MRF replaced by a single ResBlockInverted per stage
        # (pointwise-expand -> depthwise -> pointwise-project, linear
        # bottleneck, weight_norm + SnakeBeta). depthwise_kernel_size=7
        # matches this project's own ConvNeXt convention; expansion=2 is a
        # first, modest choice -- both tunable.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=7,
            expansion=2,
        )
    if kind == "hifigan_inverted_matched":
        # Same block as "hifigan_inverted", but blocks_per_stage=7 instead
        # of 1 -- expansion alone can't reach hifigan's param count in this
        # harness (conv_pre's fixed 513->256 cost dominates and dilutes any
        # expansion change), so stacking is the lever used instead.
        # Measured 2,245,440 params vs hifigan's 2,240,000 (+0.24%) --
        # closest match found by sweeping blocks_per_stage=1..7.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=7,
            expansion=2,
            blocks_per_stage=7,
        )
    if kind == "hifigan_inverted_mbv2":
        # Same block, but expansion=6 / kernel=3 -- the actual MobileNetV2
        # (Sandler et al. 2018) paper defaults, not this project's earlier
        # ConvNeXt-borrowed kernel=7 / conservative expansion=2. Needs only
        # blocks_per_stage=2 (6 blocks total, vs 21 for "_matched") to land
        # near hifigan's budget -- measured 2,139,712 params (-4.48%).
        # Much shallower, so should train more stably at a given LR.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=3,
            expansion=6,
            blocks_per_stage=2,
        )
    if kind == "hifigan_inverted_mbv2_lasttiny":
        # Per-stage expansion: (6, 6, 1) -- keeps expansion=6 (the "_mbv2"
        # winner) at stages 0/1, drops to 1 only at stage 2 (T=8192, the
        # full-sample-rate stage where profiling found the expand/project
        # pointwise convs' cost is most exposed -- ~47ms of the ~84ms total
        # forward pass was stage 2's resblock alone at uniform expansion=6).
        # Tests whether this recovers most of hifigan's RTF while keeping
        # most of "_mbv2"'s loss_mel win.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=3,
            expansion=(6, 6, 1),
            blocks_per_stage=2,
        )
    if kind == "hifigan_inverted_mbv2_441":
        # Same idea as "_lasttiny" but also drops stages 0/1 from 6 to 4 --
        # measured 17.6ms/call vs baseline's 24.4ms (faster than baseline,
        # not just closer to it) and 1,364,096 params.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=3,
            expansion=(4, 4, 1),
            blocks_per_stage=2,
        )
    if kind == "hifigan_inverted_mbv2_221":
        # Pushed further than "_441": expansion=(2,2,1) -- measured 10.7ms/call
        # (~2.2x faster than baseline's ~23.9ms), 1,193,344 params.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=3,
            expansion=(2, 2, 1),
            blocks_per_stage=2,
        )
    if kind == "hifigan_inverted_mbv2_111":
        # expansion=(1,1,1) -- no channel expansion anywhere, ResBlockInverted
        # degenerates to a plain depthwise-separable block (no longer really
        # "inverted residual" in the MobileNetV2 sense). Measured 8.4ms/call
        # (~2.9x faster than baseline), 1,107,968 params -- the speed floor
        # of this architecture family at blocks_per_stage=2/kernel=3.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=3,
            expansion=(1, 1, 1),
            blocks_per_stage=2,
        )
    if kind in ("hifigan_inverted_mbv2_111_k5", "hifigan_inverted_mbv2_111_k7"):
        # Same as "_111" but sweeping depthwise kernel_size (3/5/7 --
        # HiFi-GAN's own resblock_kernel_sizes options, for a natural
        # comparison point) now that expansion=(1,1,1) is the leading
        # candidate. Speed barely differs (8.4/8.8/10.5ms at k=3/5/7,
        # measured) -- this isolates whether kernel size affects loss_mel.
        k = 5 if kind.endswith("k5") else 7
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=k,
            expansion=(1, 1, 1),
            blocks_per_stage=2,
        )
    if kind == "hifigan_inverted_mrf_111":
        # HiFi-GAN's real MRF mechanism (3 PARALLEL branches, kernels 3/5/7,
        # summed/averaged) with ResBlockInverted as the branch, expansion=1
        # (the current winner). Compare against "_111" (2 SEQUENTIAL k=3
        # blocks) to isolate parallel-multi-kernel vs sequential-single-kernel.
        return _HiFiGANInvertedMRF(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            kernel_sizes=(3, 5, 7),
            expansion=(1, 1, 1),
        )
    if kind == "hifigan_inverted_mrf_111_noexpand":
        # Ablation of "_mrf_111": drops the first pointwise (expand) conv
        # from every branch's ResBlockInverted -- MobileNetV1-style
        # depthwise-separable (Depthwise -> Pointwise-project) instead of
        # MobileNetV2-style inverted-residual. At expansion=1 the dropped
        # pointwise was channels->channels (no width change), so this tests
        # whether it earned its params/compute or was pure overhead.
        return _HiFiGANInvertedMRFNoExpand(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            kernel_sizes=(3, 5, 7),
        )
    if kind in (
        "hifigan_inverted_mrf_221",
        "hifigan_inverted_mrf_441",
        "hifigan_inverted_mrf_661",
    ):
        # Same MRF mechanism as "_mrf_111" (3 PARALLEL branches, kernels
        # 3/5/7, ResBlockInverted per branch), sweeping the same per-stage
        # expansion schedules already validated for the SEQUENTIAL variant
        # ("_221"/"_441"/"_661" naming matches "_mbv2_221"/"_441" above, plus
        # the uniform-6 case renamed "_661" for consistency since MRF has no
        # single "_mbv2" uniform-t=6 kind of its own yet). Each branch gets
        # its own expansion at that stage -- 3x the pointwise-conv cost of
        # the sequential blocks_per_stage=2 variant at the same expansion,
        # so expect MRF to be slower at every schedule; the open question is
        # whether the parallel multi-receptive-field structure buys enough
        # loss_mel to justify that cost, per HiFi-GAN's own real MRF design.
        exp = {
            "hifigan_inverted_mrf_221": (2, 2, 1),
            "hifigan_inverted_mrf_441": (4, 4, 1),
            "hifigan_inverted_mrf_661": (6, 6, 1),
        }[kind]
        return _HiFiGANInvertedMRF(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            kernel_sizes=(3, 5, 7),
            expansion=exp,
        )
    if kind == "hifigan_inverted_t4":
        # Middle ground between "_matched" (t=2, 21 blocks total) and
        # "_mbv2" (t=6, 6 blocks total): t=4, kernel=3, blocks_per_stage=3
        # (9 blocks total) -- measured 2,140,160 params, essentially the
        # same budget as _mbv2 (2,139,712), so this isolates expansion
        # ratio alone against _mbv2 at a fixed param count.
        return _HiFiGANInverted(
            initial_channel=spec_channels,
            upsample_rates=GENERATOR_CONFIG["upsample_rates"],
            upsample_initial_channel=GENERATOR_CONFIG["upsample_initial_channel"],
            upsample_kernel_sizes=GENERATOR_CONFIG["upsample_kernel_sizes"],
            depthwise_kernel_size=3,
            expansion=4,
            blocks_per_stage=3,
        )
    if kind == "vocos":
        # No resblock/upsample_rates -- Vocos never upsamples in the conv
        # stack, so GENERATOR_CONFIG doesn't apply. n_fft/hop_length match
        # this module's own (same values the STFT target was built with).
        return VocosGenerator(initial_channel=spec_channels, n_fft=N_FFT, hop_length=HOP, gin_channels=0)
    if kind == "vocos_small":
        # dim=160/intermediate_dim=480(3x)/num_layers=8 -> ~1.99M params,
        # matched to the ~2.2M HiFi-GAN-family generators for a fair
        # apples-to-apples capacity comparison (default Vocos is ~15M here).
        return VocosGenerator(initial_channel=spec_channels, n_fft=N_FFT, hop_length=HOP, dim=160, intermediate_dim=480, num_layers=8, gin_channels=0)
    if kind == "vocos_nfft512":
        # Isolates n_fft as the only changed variable vs "vocos" (dim=512
        # unchanged) -- halves head.out's output width (n_fft+2), testing
        # whether the STFT-resolution axis alone affects reconstruction
        # independent of backbone capacity.
        return VocosGenerator(initial_channel=spec_channels, n_fft=512, hop_length=HOP, gin_channels=0)
    if kind == "vocos_small_nfft512":
        # Isolates n_fft as the only changed variable vs "vocos_small" (dim
        # unchanged at 160) -- the actual cell missing from the dim x n_fft
        # ablation grid discussed with the user: does relieving head.out's
        # phase-prediction target size fix vocos_small's crackling without
        # touching backbone width at all?
        return VocosGenerator(initial_channel=spec_channels, n_fft=512, hop_length=HOP, dim=160, intermediate_dim=480, num_layers=8, gin_channels=0)
    if kind in ("vocos_dim96", "vocos_dim112", "vocos_dim128"):
        # MarGan design doc (docs/margan_design.md) Phase 1: find a
        # backbone+head smaller than vocos_small (dim=160, ~1.99M) landing
        # in the 700-900K backbone / 800K-1.1M backbone+head target range
        # from that doc's SS6/SS11.1, at n_fft=1024 (SS1's ablation already
        # ruled out n_fft=512). Keeps intermediate_dim=3x dim and
        # num_layers=8 fixed -- same ratio/depth vocos_small already uses --
        # so dim is the only isolated variable, same methodology as the
        # dim x n_fft grid above.
        dim = int(kind.removeprefix("vocos_dim"))
        return VocosGenerator(initial_channel=spec_channels, n_fft=N_FFT, hop_length=HOP, dim=dim, intermediate_dim=dim * 3, num_layers=8, gin_channels=0)
    if kind == "margan_dim96_stageA":
        # MarGan Phase 3 (docs/margan_design.md SS13): dim=96 backbone (the
        # one chosen in Phase 1, loss_mel=8.53 alone) + Stage A phase
        # refiner on top, trained jointly from scratch via the same overfit
        # protocol -- isolates whether Stage A alone (no Stage B yet) helps,
        # before investing in the full 2-stage pipeline.
        vocos = VocosGenerator(initial_channel=spec_channels, n_fft=N_FFT, hop_length=HOP, dim=96, intermediate_dim=288, num_layers=8, gin_channels=0)
        return _VocosWithStageA(vocos, StageA())
    if kind == "margan_dim96_stageB":
        # MarGan Phase 2 (docs/margan_design.md SS13): dim=96 backbone (same
        # Phase 1 choice) + Stage B waveform refiner alone (no Stage A yet)
        # -- isolates whether time-domain residual correction helps by
        # itself, same "test each stage independently" discipline as Phase 3.
        vocos = VocosGenerator(initial_channel=spec_channels, n_fft=N_FFT, hop_length=HOP, dim=96, intermediate_dim=288, num_layers=8, gin_channels=0)
        return _VocosWithStageB(vocos, StageB())
    if kind.startswith("margan_dim") and kind.endswith("_full"):
        # MarGan Phase 4/5 (docs/margan_design.md SS13): backbone + Stage A
        # + Stage B combined -- the full MarGan pipeline, parametrized by
        # dim so Phase 5's budget-tuning sweep (e.g. dim=112 instead of
        # Phase 1's dim=96) doesn't need a new near-duplicate branch each
        # time -- only the backbone's dim changes, Stage A/B stay fixed.
        dim = int(kind.removeprefix("margan_dim").removesuffix("_full"))
        vocos = VocosGenerator(initial_channel=spec_channels, n_fft=N_FFT, hop_length=HOP, dim=dim, intermediate_dim=dim * 3, num_layers=8, gin_channels=0)
        return _MarGan(vocos, StageA(), StageB())
    raise ValueError(f"unknown generator kind: {kind}")


def mel_loss(y_hat: torch.Tensor, y_mel_target: torch.Tensor) -> torch.Tensor:
    y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1), N_FFT, MEL_CHANNELS, SAMPLE_RATE, HOP, WIN, MEL_FMIN, MEL_FMAX
    )
    t = min(y_hat_mel.shape[-1], y_mel_target.shape[-1])
    return F.l1_loss(y_hat_mel[..., :t], y_mel_target[..., :t]) * C_MEL


def vocos_magnitude_loss(model: torch.nn.Module, spec: torch.Tensor) -> float | None:
    """Isolates magnitude-prediction error from phase/iSTFT/mel entirely --
    per margan_design.md Rui ro 5, needed to tell whether a shrunk Vocos
    backbone's mel-loss gap (e.g. dim=96) comes from bad phase (Stage A's
    job) or bad magnitude too (no amount of phase refinement fixes that).
    Only meaningful for VocosGenerator: pulls its head's predicted |STFT|
    straight from `model.head.out` (replicating heads.py's ISTFTHead.forward
    up to `mag = exp(...).clamp(...)`, stopping before phase/iSTFT), then L1
    against `spec` -- the exact ground-truth linear magnitude already fed to
    this model as its own input, so shapes always match with no extra STFT
    call needed. Returns None for non-Vocos kinds (no `.backbone`/`.head`)."""
    if not (hasattr(model, "backbone") and hasattr(model, "head")):
        return None
    feats = model.backbone(spec)  # [B, T, dim], channel-last
    x = model.head.out(feats).transpose(1, 2)  # [B, n_fft+2, T]
    mag, _phase = x.chunk(2, dim=1)
    mag = torch.exp(mag).clamp(max=1e2)
    t = min(mag.shape[-1], spec.shape[-1])
    return F.l1_loss(mag[..., :t], spec[..., :t]).item()


def save_wav(path: Path, y: torch.Tensor) -> None:
    audio = (y.detach().squeeze().clamp(-1.0, 1.0).numpy() * 32767.0)
    wavfile.write(path, SAMPLE_RATE, audio)


def stage_a_phases(model: torch.nn.Module, spec: torch.Tensor):
    """Runs backbone+head.out+Stage A up to the (cos,sin) pairs, stopping
    before iSTFT -- shared by the training-time L_phi loss term below and
    the post-hoc diagnostics, so both use the exact same computation."""
    feats = model.vocos.backbone(spec)
    h = model.vocos.head.out(feats).transpose(1, 2)
    mag, phase = h.chunk(2, dim=1)
    mag = torch.exp(mag).clamp(max=1e2)
    cos_p, sin_p = torch.cos(phase), torch.sin(phase)
    cos_p2, sin_p2, delta_c, delta_s = model.stage_a(mag, cos_p, sin_p, return_correction=True)
    return cos_p, sin_p, cos_p2, sin_p2, delta_c, delta_s


def stage_a_l_phi_loss(model: torch.nn.Module, spec: torch.Tensor, cos_gt: torch.Tensor, sin_gt: torch.Tensor) -> torch.Tensor:
    """L_phi (margan_design.md SS9's exact formula) on Stage A's *post*-
    refinement (cos,sin), differentiable -- meant to be added into the
    training loss (lambda_phi * this), not just measured after the fact.
    Without this term in the loss, mel_loss alone gives Stage A zero direct
    gradient signal toward matching the *true* phase (mel_loss only
    constrains the reconstructed waveform's own magnitude spectrum, which is
    phase-invariant almost by construction -- see the diagnostic finding
    this was added to explain: L_phi stayed ~2.0 before AND after 800 steps
    of mel_loss-only training, meaning Stage A's substantial learned
    correction, mean||(Delta_c,Delta_s)||=0.61, wasn't moving phase any
    closer to ground truth at all)."""
    _cos_p, _sin_p, cos_p2, sin_p2, _dc, _ds = stage_a_phases(model, spec)
    t = min(cos_p2.shape[-1], cos_gt.shape[-1])
    return ((cos_p2[..., :t] - cos_gt[..., :t]) ** 2 + (sin_p2[..., :t] - sin_gt[..., :t]) ** 2).mean()


def stage_a_diagnostics(model: torch.nn.Module, spec: torch.Tensor, cos_gt: torch.Tensor, sin_gt: torch.Tensor) -> dict | None:
    """L_phi before/after Stage A (margan_design.md SS9's exact formula, not
    mel_loss as a proxy) + mean ||(Delta_c, Delta_s)|| (Rui ro 4: is the
    learned correction meaningfully nonzero, or did Stage A learn to be a
    no-op?). Only meaningful for `_VocosWithStageA`-style models."""
    if not hasattr(model, "stage_a"):
        return None
    cos_p, sin_p, cos_p2, sin_p2, delta_c, delta_s = stage_a_phases(model, spec)

    t = min(cos_p.shape[-1], cos_gt.shape[-1])
    cos_p, sin_p = cos_p[..., :t], sin_p[..., :t]
    cos_p2, sin_p2 = cos_p2[..., :t], sin_p2[..., :t]
    cos_gt_t, sin_gt_t = cos_gt[..., :t], sin_gt[..., :t]

    l_phi_before = ((cos_p - cos_gt_t) ** 2 + (sin_p - sin_gt_t) ** 2).mean().item()
    l_phi_after = ((cos_p2 - cos_gt_t) ** 2 + (sin_p2 - sin_gt_t) ** 2).mean().item()
    correction_norm = torch.sqrt(delta_c**2 + delta_s**2).mean().item()
    return {"l_phi_before": l_phi_before, "l_phi_after": l_phi_after, "correction_norm": correction_norm}


def run_one(
    kind: str, spec: torch.Tensor, y_mel_target: torch.Tensor, steps: int, log_every: int, out_dir: Path,
    cos_gt: torch.Tensor | None = None, sin_gt: torch.Tensor | None = None, lambda_phi: float = 0.0,
    lr: float = LR,
) -> dict:
    # Reseeding per model doesn't make initializations "equivalent" (the 3
    # architectures have different parameter shapes, so they consume the
    # RNG stream differently) -- it just makes each run reproducible.
    torch.manual_seed(1234)
    model = build_generator(kind, spec.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=BETAS, eps=EPS)
    use_l_phi = lambda_phi > 0.0 and hasattr(model, "stage_a") and cos_gt is not None

    print(f"\n=== {kind} ({n_params:,} params) -- overfitting {steps} steps{' + L_phi loss' if use_l_phi else ''} ===")
    for step in range(1, steps + 1):
        optim.zero_grad()
        y_hat = model(spec)
        m_loss = mel_loss(y_hat, y_mel_target)
        loss = m_loss
        if use_l_phi:
            l_phi = stage_a_l_phi_loss(model, spec, cos_gt, sin_gt)
            loss = loss + lambda_phi * l_phi
        loss.backward()
        optim.step()
        if step % log_every == 0 or step == steps:
            extra = f"  l_phi={l_phi.item():.4f}" if use_l_phi else ""
            print(f"  step {step:>5}/{steps}  loss_mel={m_loss.item():.4f}{extra}")

    model.eval()
    with torch.no_grad():
        y_hat_final = model(spec)
        final_loss = mel_loss(y_hat_final, y_mel_target).item()
        mag_loss = vocos_magnitude_loss(model, spec)
        stage_a_diag = stage_a_diagnostics(model, spec, cos_gt, sin_gt) if cos_gt is not None else None

    out_path = out_dir / f"{kind}.wav"
    save_wav(out_path, y_hat_final)

    if mag_loss is not None:
        print(f"  final: loss_mel={final_loss:.4f}  magnitude_l1={mag_loss:.4f}")
    if stage_a_diag is not None:
        print(
            f"  Stage A: L_phi before={stage_a_diag['l_phi_before']:.4f}  "
            f"after={stage_a_diag['l_phi_after']:.4f}  "
            f"mean||(Delta_c,Delta_s)||={stage_a_diag['correction_norm']:.4f}"
        )

    return {
        "kind": kind, "params": n_params, "final_loss_mel": final_loss,
        "magnitude_l1": mag_loss, "stage_a": stage_a_diag, "wav": out_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--wav", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "wavs" / "LJ018-0126.wav",
    )
    parser.add_argument("--steps", type=int, default=800, help="Overfit steps per model")
    parser.add_argument("--max-seconds", type=float, default=2.0, help="Clip length to use (keeps CPU runtime short)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--kinds", type=str, default=None, help="Comma-separated subset, e.g. 'vocos_small' -- default runs all 5")
    parser.add_argument("--lambda-phi", type=float, default=0.0, help="Weight for L_phi (margan_design.md SS9) added to the loss, Stage-A-capable kinds only")
    parser.add_argument("--lr", type=float, default=LR, help="AdamW learning rate (module default matches vits/training.py's own)")
    args = parser.parse_args()

    torch.manual_seed(1234)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    y = crop_to_hop_multiple(load_wav(args.wav), args.max_seconds)
    print(f"Loaded {args.wav.name}: {y.shape[1]} samples ({y.shape[1] / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz)")

    spec = spectrogram_torch(y, N_FFT, HOP, WIN)  # [1, 513, T_frames] -- fed to Generator as its input
    y_mel_target = spec_to_mel_torch(spec, N_FFT, MEL_CHANNELS, SAMPLE_RATE, MEL_FMIN, MEL_FMAX)
    cos_gt, sin_gt = ground_truth_phase(y, N_FFT, HOP, WIN)
    print(f"STFT input shape: {tuple(spec.shape)} (freq_bins={spec.shape[1]}, frames={spec.shape[2]})")

    save_wav(args.out_dir / "original.wav", y)

    kinds = tuple(args.kinds.split(",")) if args.kinds else (
        "hifigan", "hifigan_snake", "bigvgan", "bigvgan_lite", "vocos",
    )
    results = [
        run_one(kind, spec, y_mel_target, args.steps, args.log_every, args.out_dir, cos_gt, sin_gt, args.lambda_phi, args.lr)
        for kind in kinds
    ]

    print("\n=== Summary (lower loss_mel = better reconstruction) ===")
    print(f"{'Generator':<16}{'Params':>12}{'Final loss_mel':>18}{'Magnitude L1':>16}")
    for r in sorted(results, key=lambda r: r["final_loss_mel"]):
        mag_str = f"{r['magnitude_l1']:.4f}" if r["magnitude_l1"] is not None else "n/a"
        print(f"{r['kind']:<16}{r['params']:>12,}{r['final_loss_mel']:>18.4f}{mag_str:>16}")
    print(f"\noriginal.wav + per-kind .wav saved to {args.out_dir}")


if __name__ == "__main__":
    main()
