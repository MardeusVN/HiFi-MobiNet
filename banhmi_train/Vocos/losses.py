# Ported from the official gemelo-ai/vocos repo's vocos/loss.py (hinge
# formulas) + vocos/experiment.py's training_step (the sub-discriminator
# averaging pattern) -- verified against the actual source via
# raw.githubusercontent.com, not reconstructed from memory.
"""Vocos trains its discriminators with a HINGE loss, not the LSGAN loss
`vits/losses.py` uses for the HiFi-GAN-family discriminators -- these are
the correct pairing for `Vocos/discriminators.py`'s MPD/MRD when
use_vocos=True (see vits/training.py). Two things differ from
vits/losses.py's discriminator_loss/generator_loss/feature_loss, both
load-bearing, not stylistic:

  - Hinge, not squared-error: discriminator wants real logit > 1 / fake
    logit < -1 via clamp(1 - x, min=0) / clamp(1 + x, min=0); generator
    wants clamp(1 - fake_logit, min=0) pushed to 0 (fake logit >= 1) --
    LSGAN's mean((1-x)^2) is a different loss landscape entirely.
  - Averaged over sub-discriminators *inside* the loss (loss / len(...)),
    matching vocos/experiment.py's `loss_mp /= len(loss_mp_real)` (and the
    same for loss_mrd, loss_gen_mp, loss_gen_mrd, loss_fm_mp, loss_fm_mrd)
    -- vits/losses.py's versions are a plain sum with no normalization.
    feature_loss here also skips vits/losses.py's *2 scale and its
    .detach() on the real feature maps (real Vocos has neither).

Only for Vocos/discriminators.py's MPD/MRD -- model_d_dur (VITS2's
duration discriminator, orthogonal to vocoder choice) keeps using
vits/losses.py's LSGAN functions regardless of use_vocos.
"""
import torch


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss / len(fmap_r)


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Hinge loss, averaged over sub-discriminators."""
    loss = 0.0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(torch.clamp(1 - dr, min=0))
        g_loss = torch.mean(torch.clamp(1 + dg, min=0))
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss / len(disc_real_outputs), r_losses, g_losses


def generator_loss(disc_outputs):
    """Hinge loss, averaged over sub-discriminators."""
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(torch.clamp(1 - dg, min=0))
        gen_losses.append(l)
        loss = loss + l
    return loss / len(disc_outputs), gen_losses
