#!/usr/bin/env python3
"""Trains a single-speaker VITS voice from a banhmi_train.preprocess output
directory (dataset.jsonl + config.json).

Usage:
    python -m banhmi_train.train \
        --dataset-dir training_output/ljspeech/medium \
        --quality medium \
        --batch-size 16 --accelerator gpu --devices 1

    # Or edit one file instead of typing every flag (see configs/*.yaml for
    # examples). Anything in the YAML overrides --quality and any other
    # flag above it -- including VitsModel parameters that have no --flag
    # at all (e.g. resblock_kernel_sizes), since VitsModel's **kwargs
    # catch-all accepts them by name.
    python -m banhmi_train.train \
        --dataset-dir training_output/ljspeech/medium \
        --config configs/banhmi.yaml \
        --batch-size 16 --accelerator gpu --devices 1
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .vits.training import VitsModel

_LOGGER = logging.getLogger("banhmi_train.train")


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to preprocessed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs", type=int, help="Save checkpoint every N epochs (default: 1)"
    )
    parser.add_argument(
        "--quality", default="medium", choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--config", type=Path,
        help="YAML file of parameter overrides, applied on top of --quality and every other flag",
    )
    parser.add_argument(
        "--use-bigvgan", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Bundle the BigVGAN paper's own recipe (SnakeBeta activation + MPD + MRD) as one "
            "switch -- equivalent to --use-snake-beta --use-mrd together. Leave unset to control "
            "them individually (e.g. SnakeBeta without MRD, or vice versa). CLI-only: this "
            "bundling logic lives in this file's main(), not in VitsModel, so `use_bigvgan: true` "
            "inside a --config YAML has no effect -- set use_snake_beta/use_mrd there instead."
        ),
    )
    parser.add_argument(
        "--use-vits2", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Bundle VITS2's own actual contributions we've implemented (transformer-flow "
            "coupling layers + duration discriminator + noise-scaled MAS) as one switch -- "
            "equivalent to --use-transformer-flows --use-duration-discriminator "
            "--use-noise-scaled-mas together. Does NOT cover VITS2's mel-posterior-encoder, "
            "which isn't ported (see conversation history) -- this bundle is only what's "
            "actually implemented. CLI-only, same as --use-bigvgan: has no effect written "
            "inside a --config YAML."
        ),
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See banhmi_train.preprocess.config for the schema
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        sample_rate = int(config["audio"]["sample_rate"])

    trainer = Trainer.from_argparse_args(args)
    if args.checkpoint_epochs is not None:
        trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)]
        _LOGGER.debug("Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs)

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    if args.use_bigvgan is not None:
        dict_args["use_snake_beta"] = args.use_bigvgan
        dict_args["use_mrd"] = args.use_bigvgan

    if args.use_vits2 is not None:
        dict_args["use_transformer_flows"] = args.use_vits2
        dict_args["use_duration_discriminator"] = args.use_vits2
        dict_args["use_noise_scaled_mas"] = args.use_vits2

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as config_yaml_file:
            overrides = yaml.safe_load(config_yaml_file) or {}
        _LOGGER.info("Applying overrides from %s: %s", args.config, overrides)
        dict_args.update(overrides)

    model = VitsModel(
        num_symbols=num_symbols,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    trainer.fit(model)

    # test_split is held out entirely until now -- a final, unbiased check
    # on data the training process never touched for monitoring/tuning.
    if model._test_dataset is not None and len(model._test_dataset) > 0:
        trainer.test(model)
    else:
        _LOGGER.warning("No test set to evaluate (test_split too small for this dataset size)")


if __name__ == "__main__":
    main()
