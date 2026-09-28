#!/usr/bin/env python3
"""Params + wall-clock speed of the 3 Generator architectures, fed the
*real* VITS input shape this time: latent `z_slice` (initial_channel=192,
`segment_size // hop_length` frames -- see vits/training.py), not the STFT
stand-in `compare_generators.py` used for the reconstruction-quality test.
Measures both:
  - forward-only (eval, no_grad): the inference/vocoder-deployment cost,
    reported as a real-time factor (audio-seconds produced per wall-clock
    second -- >1x means faster than real-time).
  - forward+backward: the per-training-step cost this actually adds on top
    of the discriminator/other VITS components (training_step_g calls
    Generator once per step).

Deliberately CPU-only, same reasoning as compare_generators.py -- the live
2000-epoch DDP run in /home/capstone/overnight_run_full is already flirting
with OOM on both GPUs (see train.log), so this doesn't touch CUDA at all.
CPU numbers aren't the real deployment number, but the *ratio* between the
3 architectures (the actual question here) transfers to GPU reasonably
well since all 3 share the same conv/upsample backbone -- see the printed
overhead caveat.

Usage:
    python -m test_function.benchmark_generator_speed
    python -m test_function.benchmark_generator_speed --batch-size 8 --reps 100
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banhmi_train.BigVGan.generator import BigVGANGenerator  # noqa: E402
from banhmi_train.BigVGan.generator_lite import BigVGANGeneratorLite  # noqa: E402
from banhmi_train.Vocos.generator import VocosGenerator  # noqa: E402
from banhmi_train.vits.modules.generator import Generator  # noqa: E402

# Matches vits/training.py's real defaults: initial_channel = inter_channels
# (192), medium quality's resblock/upsample config, hop_length = 256.
INTER_CHANNELS = 192
N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_RATE = 22050
GENERATOR_CONFIG = dict(
    resblock="2",
    resblock_kernel_sizes=(3, 5, 7),
    resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
    upsample_rates=(8, 8, 4),
    upsample_initial_channel=256,
    upsample_kernel_sizes=(16, 16, 8),
    gin_channels=0,
)


def build_generator(kind: str) -> torch.nn.Module:
    if kind == "hifigan":
        return Generator(initial_channel=INTER_CHANNELS, use_snake=False, **GENERATOR_CONFIG)
    if kind == "hifigan_snake":
        return Generator(initial_channel=INTER_CHANNELS, use_snake=True, **GENERATOR_CONFIG)
    if kind == "bigvgan":
        return BigVGANGenerator(initial_channel=INTER_CHANNELS, **GENERATOR_CONFIG)
    if kind == "bigvgan_lite":
        return BigVGANGeneratorLite(initial_channel=INTER_CHANNELS, **GENERATOR_CONFIG)
    if kind == "vocos":
        return VocosGenerator(initial_channel=INTER_CHANNELS, n_fft=N_FFT, hop_length=HOP_LENGTH, gin_channels=0)
    raise ValueError(kind)


def bench_forward(model: torch.nn.Module, z: torch.Tensor, warmup: int, reps: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(z)
        start = time.perf_counter()
        for _ in range(reps):
            model(z)
        elapsed = time.perf_counter() - start
    return elapsed / reps


def bench_forward_backward(model: torch.nn.Module, z: torch.Tensor, warmup: int, reps: int) -> float:
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        y_hat = model(z)
        y_hat.pow(2).mean().backward()
    start = time.perf_counter()
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        y_hat = model(z)
        y_hat.pow(2).mean().backward()
    elapsed = time.perf_counter() - start
    return elapsed / reps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=16, help="Matches training's default --batch-size")
    parser.add_argument("--frames", type=int, default=32, help="segment_size(8192)/hop_length(256) in training")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=0, help="0 = torch default")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(1234)

    z = torch.randn(args.batch_size, INTER_CHANNELS, args.frames)
    audio_seconds_per_call = (args.frames * HOP_LENGTH * args.batch_size) / SAMPLE_RATE

    print(f"batch_size={args.batch_size} frames={args.frames} -> "
          f"{args.frames * HOP_LENGTH} samples/clip = {args.frames * HOP_LENGTH / SAMPLE_RATE:.3f}s/clip "
          f"({audio_seconds_per_call:.2f}s audio/call across the batch)")
    print(f"torch.get_num_threads() = {torch.get_num_threads()} (CPU-only run)\n")

    rows = []
    for kind in ("hifigan", "hifigan_snake", "bigvgan", "bigvgan_lite", "vocos"):
        model = build_generator(kind)
        n_params = sum(p.numel() for p in model.parameters())

        fwd_s = bench_forward(model, z, args.warmup, args.reps)
        fwd_bwd_s = bench_forward_backward(model, z, args.warmup, args.reps)

        rtf = audio_seconds_per_call / fwd_s
        rows.append(
            {
                "kind": kind,
                "params": n_params,
                "fwd_ms": fwd_s * 1000,
                "fwd_rtf": rtf,
                "fwd_bwd_ms": fwd_bwd_s * 1000,
            }
        )
        print(f"{kind:<16} params={n_params:>10,}  forward={fwd_s*1000:>8.2f}ms "
              f"({rtf:>6.1f}x real-time)  forward+backward={fwd_bwd_s*1000:>8.2f}ms")

    baseline = next(r for r in rows if r["kind"] == "hifigan")
    print("\n=== Summary (relative to vanilla HiFi-GAN) ===")
    print(f"{'Generator':<16}{'Params':>12}{'+Params':>10}{'Forward ms':>13}{'vs base':>9}"
          f"{'Fwd+Bwd ms':>13}{'vs base':>9}")
    for r in rows:
        dp = r["params"] - baseline["params"]
        fwd_ratio = r["fwd_ms"] / baseline["fwd_ms"]
        fb_ratio = r["fwd_bwd_ms"] / baseline["fwd_bwd_ms"]
        print(
            f"{r['kind']:<16}{r['params']:>12,}{dp:>+10,}{r['fwd_ms']:>13.2f}{fwd_ratio:>8.2f}x"
            f"{r['fwd_bwd_ms']:>13.2f}{fb_ratio:>8.2f}x"
        )


if __name__ == "__main__":
    main()
