#!/usr/bin/env python3
"""Isolated speed benchmark for the 2 discriminator combos discussed:
BigVGan (UnivNet-style MRD, project's existing setup) vs Vocos-recipe
(DAC-style multi-band MRD) -- same methodology as
benchmark_generator_speed.py (many reps, warmup, forward-only and
forward+backward), but for D instead of G, and broken down by MPD vs MRD
individually so it's clear which piece is the bottleneck, not just the combo.

Deliberately CPU-only, same reasoning as the other test_function scripts.

Usage:
    python -m test_function.benchmark_discriminator_speed
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banhmi_train.BigVGan.discriminators import (  # noqa: E402
    MultiPeriodDiscriminator as BigVGanMPD,
    MultiResolutionDiscriminator as BigVGanMRD,
)
from banhmi_train.vits.losses import discriminator_loss, feature_loss, generator_loss  # noqa: E402
from banhmi_train.Vocos.discriminators import (  # noqa: E402
    MultiPeriodDiscriminator as VocosMPD,
    MultiResolutionDiscriminator as VocosMRD,
)

SAMPLE_RATE = 22050


def bench(d: torch.nn.Module, y: torch.Tensor, y_hat: torch.Tensor, warmup: int, reps: int):
    n_params = sum(p.numel() for p in d.parameters())

    d.eval()
    with torch.no_grad():
        for _ in range(warmup):
            d(y, y_hat)
        start = time.perf_counter()
        for _ in range(reps):
            d(y, y_hat)
        fwd_s = (time.perf_counter() - start) / reps

    d.train()

    def d_step():
        y_d_r, y_d_g, _, _ = d(y, y_hat.detach())
        loss, *_ = discriminator_loss(y_d_r, y_d_g)
        loss.backward()

    def g_step():
        y_d_r, y_d_g, fmap_r, fmap_g = d(y, y_hat)
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_g)
        (loss_fm + loss_gen).backward()

    for _ in range(warmup):
        d.zero_grad(set_to_none=True)
        d_step()
    start = time.perf_counter()
    for _ in range(reps):
        d.zero_grad(set_to_none=True)
        d_step()
    d_step_s = (time.perf_counter() - start) / reps

    y_hat = y_hat.detach().requires_grad_(True)
    for _ in range(warmup):
        if y_hat.grad is not None:
            y_hat.grad = None
        g_step()
    start = time.perf_counter()
    for _ in range(reps):
        if y_hat.grad is not None:
            y_hat.grad = None
        g_step()
    g_step_s = (time.perf_counter() - start) / reps

    return n_params, fwd_s, d_step_s, g_step_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=0.372, help="Clip length, matches benchmark_generator_speed.py's 32-frame default (8192 samples)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(1234)
    t_wav = int(args.seconds * SAMPLE_RATE)
    y = torch.randn(args.batch_size, 1, t_wav)
    y_hat = torch.randn(args.batch_size, 1, t_wav)

    print(f"batch_size={args.batch_size}  clip={t_wav} samples ({args.seconds:.3f}s)  torch.get_num_threads()={torch.get_num_threads()}\n")

    combos = {
        "BigVGan MPD": BigVGanMPD(),
        "BigVGan MRD (UnivNet)": BigVGanMRD(),
        "Vocos MPD": VocosMPD(),
        "Vocos MRD (DAC)": VocosMRD(),
    }

    rows = {}
    for name, d in combos.items():
        n_params, fwd_s, d_step_s, g_step_s = bench(d, y, y_hat, args.warmup, args.reps)
        rows[name] = (n_params, fwd_s, d_step_s, g_step_s)
        print(f"{name:<24} params={n_params:>12,}  forward={fwd_s*1000:>8.2f}ms  "
              f"D-step(fwd+bwd)={d_step_s*1000:>8.2f}ms  G-step(fwd+bwd)={g_step_s*1000:>8.2f}ms")

    print("\n=== Combined D-step cost (what one training step actually pays) ===")
    for label, mpd_name, mrd_name in [
        ("BigVGan combo (project default)", "BigVGan MPD", "BigVGan MRD (UnivNet)"),
        ("Vocos-recipe combo", "Vocos MPD", "Vocos MRD (DAC)"),
    ]:
        p_mpd, f_mpd, d_mpd, g_mpd = rows[mpd_name]
        p_mrd, f_mrd, d_mrd, g_mrd = rows[mrd_name]
        total_params = p_mpd + p_mrd
        total_d_step = d_mpd + d_mrd
        total_g_step = g_mpd + g_mrd
        print(f"{label:<34} params={total_params:>12,}  D-step={total_d_step*1000:>8.2f}ms  G-step(D's share)={total_g_step*1000:>8.2f}ms  total/train-step={( total_d_step+total_g_step)*1000:>8.2f}ms")


if __name__ == "__main__":
    main()
