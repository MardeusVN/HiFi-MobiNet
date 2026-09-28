#!/usr/bin/env python3
"""Same 4-generator comparison as `compare_generators.py`, but with the
*real* adversarial objective this time: MPD + MRD (`BigVGan/
discriminators.py`) alternated G-step/D-step against each Generator, exact
loss formulas from `vits/losses.py` (LSGAN discriminator/generator loss +
feature-matching loss) and the same combination `vits/training.py`'s
`training_step_g` uses (loss_gen + loss_fm + loss_mel, MPD and MRD summed
in). Duration-predictor/discriminator and KL loss are excluded -- those
depend on VITS's text encoder/posterior encoder/flow, which this test
bypasses entirely by feeding the Generator raw STFT (see
compare_generators.py's docstring for why).

Why this exists (not just compare_generators.py): mel-L1-only training
can't show whether a Generator interacts differently with a discriminator
mid-training -- e.g. whether AMP's anti-aliased output is harder/easier for
MRD's STFT-based scoring to critique. Each Generator here gets its own
fresh MPD+MRD (not shared), matching how each ablation would really be
trained standalone.

Each "step" below = one D update + one G update (2x the optimizer steps of
`compare_generators.py`'s pure mel-L1 loop for the same step count, and
each of those touches 2 heavy discriminators too) -- expect this to run
noticeably slower per step; use a smaller --steps than the mel-only test.

Usage:
    python -m test_function.compare_generators_gan --steps 300
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from banhmi_train.BigVGan.discriminators import (  # noqa: E402
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)
from banhmi_train.mel_processing import mel_spectrogram_torch  # noqa: E402
from banhmi_train.vits.losses import discriminator_loss, feature_loss, generator_loss  # noqa: E402
from test_function.compare_generators import (  # noqa: E402
    BETAS,
    C_MEL,
    EPS,
    HOP,
    LR,
    MEL_CHANNELS,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    SAMPLE_RATE,
    WIN,
    build_generator,
    crop_to_hop_multiple,
    load_wav,
    save_wav,
)


def mel_of(y: torch.Tensor) -> torch.Tensor:
    return mel_spectrogram_torch(y.squeeze(1), N_FFT, MEL_CHANNELS, SAMPLE_RATE, HOP, WIN, MEL_FMIN, MEL_FMAX)


def run_one(kind: str, spec: torch.Tensor, y: torch.Tensor, y_mel_target: torch.Tensor, steps: int, log_every: int, out_dir: Path) -> dict:
    torch.manual_seed(1234)
    gen = build_generator(kind, spec.shape[1])
    mpd = MultiPeriodDiscriminator()
    mrd = MultiResolutionDiscriminator()
    n_params_g = sum(p.numel() for p in gen.parameters())

    optim_g = torch.optim.AdamW(gen.parameters(), lr=LR, betas=BETAS, eps=EPS)
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()), lr=LR, betas=BETAS, eps=EPS
    )

    y = y.unsqueeze(1)  # [B, T] -> [B, 1, T], matching Generator's own [B, 1, T] output

    print(f"\n=== {kind} ({n_params_g:,} G params) -- {steps} adversarial steps ===")
    start = time.perf_counter()
    for step in range(1, steps + 1):
        # D-step
        with torch.no_grad():
            y_hat = gen(spec)
        optim_d.zero_grad()
        y_d_r, y_d_g, _, _ = mpd(y, y_hat)
        y_d_r_mrd, y_d_g_mrd, _, _ = mrd(y, y_hat)
        loss_disc, *_ = discriminator_loss(y_d_r, y_d_g)
        loss_disc_mrd, *_ = discriminator_loss(y_d_r_mrd, y_d_g_mrd)
        (loss_disc + loss_disc_mrd).backward()
        optim_d.step()

        # G-step
        optim_g.zero_grad()
        y_hat = gen(spec)
        y_hat_mel = mel_of(y_hat)
        t = min(y_hat_mel.shape[-1], y_mel_target.shape[-1])
        loss_mel = F.l1_loss(y_hat_mel[..., :t], y_mel_target[..., :t]) * C_MEL

        y_d_r, y_d_g, fmap_r, fmap_g = mpd(y, y_hat)
        y_d_r_mrd, y_d_g_mrd, fmap_r_mrd, fmap_g_mrd = mrd(y, y_hat)
        loss_fm = feature_loss(fmap_r, fmap_g) + feature_loss(fmap_r_mrd, fmap_g_mrd)
        loss_gen, _ = generator_loss(y_d_g)
        loss_gen_mrd, _ = generator_loss(y_d_g_mrd)
        loss_gen_all = loss_gen + loss_gen_mrd + loss_fm + loss_mel
        loss_gen_all.backward()
        optim_g.step()

        if step % log_every == 0 or step == steps:
            print(
                f"  step {step:>4}/{steps}  loss_mel={loss_mel.item():.4f}  "
                f"loss_gen_all={loss_gen_all.item():.4f}  loss_disc={(loss_disc+loss_disc_mrd).item():.4f}"
            )
    elapsed = time.perf_counter() - start

    gen.eval()
    with torch.no_grad():
        y_hat_final = gen(spec)
        final_mel = mel_of(y_hat_final)
        t = min(final_mel.shape[-1], y_mel_target.shape[-1])
        final_loss_mel = (F.l1_loss(final_mel[..., :t], y_mel_target[..., :t]) * C_MEL).item()

    out_path = out_dir / f"{kind}_gan.wav"
    save_wav(out_path, y_hat_final)

    return {
        "kind": kind,
        "params": n_params_g,
        "final_loss_mel": final_loss_mel,
        "seconds_per_step": elapsed / steps,
        "wav": out_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--wav", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "wavs" / "LJ018-0126.wav",
    )
    parser.add_argument("--steps", type=int, default=300, help="Fewer than compare_generators.py -- D+G both run 2 discriminators/step")
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()

    torch.manual_seed(1234)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from banhmi_train.mel_processing import spec_to_mel_torch, spectrogram_torch

    y = crop_to_hop_multiple(load_wav(args.wav), args.max_seconds)
    print(f"Loaded {args.wav.name}: {y.shape[1]} samples ({y.shape[1] / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz)")

    spec = spectrogram_torch(y, N_FFT, HOP, WIN)
    y_mel_target = spec_to_mel_torch(spec, N_FFT, MEL_CHANNELS, SAMPLE_RATE, MEL_FMIN, MEL_FMAX)

    results = [
        run_one(kind, spec, y, y_mel_target, args.steps, args.log_every, args.out_dir)
        for kind in ("hifigan", "hifigan_snake", "bigvgan", "bigvgan_lite", "vocos")
    ]

    print("\n=== Summary: adversarial training (lower loss_mel = better) ===")
    print(f"{'Generator':<16}{'Params':>12}{'Final loss_mel':>18}{'s/step (G+D)':>16}")
    for r in sorted(results, key=lambda r: r["final_loss_mel"]):
        print(f"{r['kind']:<16}{r['params']:>12,}{r['final_loss_mel']:>18.4f}{r['seconds_per_step']:>16.3f}")
    print(f"\n*_gan.wav saved to {args.out_dir}")


if __name__ == "__main__":
    main()
