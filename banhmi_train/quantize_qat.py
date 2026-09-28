#!/usr/bin/env python3
"""Quantization-aware fine-tuning for BanhmiTTS's SynthesizerTrn. Ported
from the QAT-Training side-project's qat_transfer/piper_train/quantize_qat.py,
adapted for this project's automatic-optimization + optimizer_idx training
loop (VitsModel.training_step(batch, batch_idx, optimizer_idx), not manual
optimization -- QAT-Training's own project uses manual opt, this one
doesn't, so the discriminator-warmup mechanism differs: instead of
conditionally skipping opt_d.step() inside a manual training_step, this
returns None from training_step when optimizer_idx==1 during warmup, which
tells PL's automatic optimizer to skip that optimizer's step for the batch
entirely (no forward/backward for D at all during warmup, not just no
step -- a stronger form of freezing but same intent: model_g's own
training_step_g adversarial loss term still uses D's current, frozen,
un-stepped weights).

Resumes from a converged FP32 checkpoint (not an existing QAT run --
that would be quantize_qat's --resume-qat-checkpoint mode in the original
project; not ported here since this is always a fresh QAT start), wraps
the validated PTQ scope's Conv1d/ConvTranspose1d/Linear layers with
fake-quantize wrappers (see vits/quantize.py), and continues training at a
low learning rate so the weights adapt to INT8 quantization noise before
ONNX export.

Usage (baseline, matching the PTQ sweep's winning scope -- flow+enc_p+dp,
Conv/ConvTranspose only, dec left untouched):
    python3 -m banhmi_train.quantize_qat \\
        --resume-from-checkpoint baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt \\
        --dataset-dir training_output/ljspeech/medium \\
        --qat-submodules flow,enc_p,dp \\
        --max_epochs 150 --d-warmup-steps 785 --batch-size 16 --devices 2
"""
import argparse
import logging
import pathlib
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from .vits.dataset import Batch
from .vits.training import VitsModel
from .vits.quantize import (
    count_qat_layers,
    prepare_qat,
    remove_all_weight_norm,
    set_observer_enabled,
)

_LOGGER = logging.getLogger("banhmi_train.quantize_qat")

torch.serialization.add_safe_globals([pathlib.PosixPath])

_WRAP_TYPE_MAP = {"conv": (nn.Conv1d, nn.ConvTranspose1d), "all": (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)}


class QATVitsModel(VitsModel):
    """VitsModel + generator-only discriminator warmup: model_d/model_d_mrd/
    model_d_dur's weights stay frozen (no forward, no backward, no
    optimizer step at all) for the first `d_warmup_steps` batches, while
    model_g still trains fully -- including its own adversarial loss term,
    computed against D's current (frozen) weights. Recommended practice
    for fine-tuning quantized GANs: without it, D immediately reacts to the
    generator's fresh fake-quant noise from step 1, which can destabilize
    the adversarial balance before the generator has had any chance to
    adapt."""

    def __init__(self, *args, d_warmup_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_warmup_steps = d_warmup_steps
        self._g_steps_seen = 0

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            self._g_steps_seen += 1
            return self.training_step_g(batch)
        if self._g_steps_seen <= self.d_warmup_steps:
            return None  # skip this batch's D step entirely during warmup
        return self.training_step_d(batch)


class FreezeObserverCallback(Callback):
    """Standard QAT practice: let the FakeQuantize observers calibrate
    scale/zero_point against real activation statistics for the first N
    epochs, then freeze them so the exported values stop drifting while
    the remaining epochs just let the weights settle around the now-fixed
    quantization grid."""

    def __init__(self, freeze_after_epoch: int):
        self.freeze_after_epoch = freeze_after_epoch
        self._frozen = False

    def on_train_epoch_start(self, trainer, pl_module):
        if (not self._frozen) and trainer.current_epoch >= self.freeze_after_epoch:
            _LOGGER.info("Freezing FakeQuantize observers at epoch %s", trainer.current_epoch)
            set_observer_enabled(pl_module.model_g, False)
            self._frozen = True


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-from-checkpoint", required=True,
        help="Path to a converged FP32 .ckpt to start QAT fine-tuning from (fresh "
        "start: weights only, optimizer/scheduler/epoch state reset)",
    )
    parser.add_argument("--dataset-dir", required=True, help="Path to pre-processed dataset directory")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6,
        help="QAT fine-tuning LR (~40x lower than the 2e-4 base-training default -- GAN "
        "adversarial training is sensitive to LR, and published QAT recipes for other "
        "architectures land in the 1e-6-1e-5 range, not the 'divide by 10' heuristic used "
        "for plain supervised fine-tuning). Deliberately independent of the source "
        "checkpoint's own decayed LR -- this is a fresh, much-lower-LR regime by design, "
        "not a resume of the original training's schedule.",
    )
    parser.add_argument(
        "--lr-decay-target", type=float, default=0.1,
        help="Total multiplicative LR decay to reach by the end of training (default: 10x "
        "reduction). Computes a per-epoch gamma = lr_decay_target ** (1/max_epochs) instead "
        "of the checkpoint's own lr_decay=0.999875 (tuned for ~1500-epoch base training, "
        "barely moves the LR over a short QAT run).",
    )
    parser.add_argument(
        "--d-warmup-steps", type=int, required=True,
        help="Keep model_d/model_d_mrd/model_d_dur frozen for this many initial batches "
        "while model_g still trains. Recommended: one epoch's worth of batches for this "
        "checkpoint's batch_size/dataset (compute from the source checkpoint's own "
        "global_step / epoch).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--freeze-observer-epoch", type=int, default=None,
        help="Freeze FakeQuantize observers after this epoch (default: 40%% of max_epochs)",
    )
    parser.add_argument(
        "--qat-submodules", required=True,
        help="Comma-separated model_g submodule names to QAT-wrap, e.g. flow,enc_p,dp -- "
        "must match whichever scope the PTQ sweep validated as working for this "
        "architecture (see PTQ_Sweep_Report).",
    )
    parser.add_argument(
        "--qat-exclude-leaf-names", default="",
        help="Comma-separated leaf module names to skip within the wrapped submodules "
        "(e.g. conv_post for baseline's dec, out for Vocos's dec.head) -- empty when the "
        "wrapped scope doesn't include dec at all.",
    )
    parser.add_argument(
        "--qat-wrap-types", choices=("conv", "all"), default="conv",
        help="'conv' wraps only Conv1d/ConvTranspose1d (baseline's validated PTQ scope -- "
        "never touches enc_p/dp's attention Linear layers, so QAT-wrapping those too would "
        "waste training capacity adapting weights that will never actually be exported as "
        "INT8). 'all' additionally wraps Linear (needed for vocos_small's dec, where "
        "pwconv1/pwconv2/head.out genuinely are nn.Linear and do get quantized).",
    )
    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    parser.add_argument("--default_root_dir", help="Trainer log/checkpoint directory")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    dataset_dir = Path(args.dataset_dir)
    dataset_path = dataset_dir / "dataset.jsonl"
    if not args.default_root_dir:
        args.default_root_dir = str(dataset_dir / "qat_runs")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    lr_decay = args.lr_decay_target ** (1.0 / args.max_epochs)
    _LOGGER.info(
        "LR schedule: %.2e -> %.2e over %d epochs (gamma=%.5f)",
        args.learning_rate, args.learning_rate * args.lr_decay_target, args.max_epochs, lr_decay,
    )
    _LOGGER.info("Discriminator warmup: frozen for the first %d batch(es)", args.d_warmup_steps)

    _LOGGER.info("Loading FP32 checkpoint (fresh QAT start): %s", args.resume_from_checkpoint)
    model = QATVitsModel.load_from_checkpoint(
        args.resume_from_checkpoint,
        dataset=[dataset_path],
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=lr_decay,
        num_workers=args.num_workers,
        d_warmup_steps=args.d_warmup_steps,
        use_f0=False,  # checkpoints predating the F0 feature have no use_f0 hparam saved
        strict=False,
        map_location="cpu",
        weights_only=False,
    )

    _LOGGER.info("Preparing model_g for QAT (removing weight_norm, wrapping layers)...")
    remove_all_weight_norm(model.model_g)
    qat_submodules = tuple(s.strip() for s in args.qat_submodules.split(",") if s.strip())
    qat_exclude_leaf_names = tuple(s.strip() for s in args.qat_exclude_leaf_names.split(",") if s.strip())
    wrap_types = _WRAP_TYPE_MAP[args.qat_wrap_types]
    _LOGGER.info(
        "QAT wrap scope: submodules=%s exclude_leaf_names=%s wrap_types=%s",
        qat_submodules, qat_exclude_leaf_names, args.qat_wrap_types,
    )
    prepare_qat(model.model_g, submodule_names=qat_submodules, exclude_leaf_names=qat_exclude_leaf_names, wrap_types=wrap_types)
    n_layers = count_qat_layers(model.model_g)
    _LOGGER.info("QAT-wrapped %d layer(s)", n_layers)

    freeze_epoch = args.freeze_observer_epoch
    if freeze_epoch is None:
        freeze_epoch = max(1, int(args.max_epochs * 0.4))

    devices = int(args.devices) if str(args.devices).isdigit() else args.devices
    strategy = "ddp" if isinstance(devices, int) and devices > 1 else "auto"
    callbacks = [
        ModelCheckpoint(
            filename="qat-best-{epoch}-{val_loss_mel:.4f}",
            monitor="val_loss_mel",
            mode="min",
            save_top_k=3,
            save_last=False,
        ),
        FreezeObserverCallback(freeze_epoch),
    ]
    if args.checkpoint_epochs is not None:
        callbacks.append(
            ModelCheckpoint(every_n_epochs=args.checkpoint_epochs, save_top_k=1, save_last=True, filename="qat-last-{epoch}")
        )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        logger=TensorBoardLogger(save_dir=args.default_root_dir),
        callbacks=callbacks,
        replace_sampler_ddp=False,
        gradient_clip_val=1.0,
    )
    trainer.fit(model)

    if model._test_dataset is not None and len(model._test_dataset) > 0:
        trainer.test(model)


if __name__ == "__main__":
    main()
