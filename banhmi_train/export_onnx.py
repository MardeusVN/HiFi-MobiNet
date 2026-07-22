#!/usr/bin/env python3
"""Exports a trained VitsModel checkpoint to an ONNX voice, matching the
input/output schema banhmi_train.say expects ("input", "input_lengths",
"scales" -- no "sid", since this project is permanently single-speaker).

Usage:
    python -m banhmi_train.export_onnx path/to/checkpoint.ckpt voice.onnx
"""
import argparse
import logging
from pathlib import Path

import torch

from .vits.training import VitsModel

_LOGGER = logging.getLogger("banhmi_train.export_onnx")

_OPSET_VERSION = 15


def main() -> None:
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output", help="Path to output model (.onnx)")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG messages to the console")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)
    model_g = model.model_g

    num_symbols = model_g.n_vocab

    model_g.eval()
    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales):
        noise_scale, length_scale, noise_scale_w = scales[0], scales[1], scales[2]
        audio = model_g.infer(
            text, text_lengths, noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w
        )[0].unsqueeze(1)
        return audio

    model_g.forward = infer_forward

    dummy_input_length = 50
    sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long)
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])  # noise, length, noise_w

    torch.onnx.export(
        model=model_g,
        args=(sequences, sequence_lengths, scales),
        f=str(output_path),
        verbose=False,
        opset_version=_OPSET_VERSION,
        input_names=["input", "input_lengths", "scales"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Exported model to %s", output_path)


if __name__ == "__main__":
    main()
