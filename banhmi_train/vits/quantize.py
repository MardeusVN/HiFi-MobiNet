"""Quantization-aware training (QAT) support for BanhmiTTS's SynthesizerTrn
generator. Ported from the QAT-Training side-project's proven
qat_transfer/piper_train/vits/quantize.py (same wrap-in-place approach,
same FakeQuantize config for ONNX Runtime's QDQ u8s8 scheme) -- see that
repo's module docstring for the full rationale on why eager
`prepare_qat`/FX-mode quantization can't be used directly on this
architecture (no nnqat.Conv1d/ConvTranspose1d; FX symbolic_trace breaks on
this codebase's data-dependent reverse/forward branching).

Architecture-agnostic: works for both the HiFi-GAN-family `dec` (baseline)
and the Vocos `dec` (ConvNeXt+ISTFT) -- prepare_qat's submodule_names
selects by attribute name only, and the PTQ sweep run separately for each
(see QAT-Training-style PTQ_Sweep_Report) determined that quantizing
flow+enc_p+dp while leaving dec at FP32 is the config that actually works
well for BOTH architectures -- unlike this project's own dec, which
differs enough between the two that no single quantization scope for it
transfers.
"""
import logging
from typing import Iterable

import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn import functional as F

_LOGGER = logging.getLogger("banhmi_train.vits.quantize")

# All of SynthesizerTrn's infer()-path submodules. enc_q (posterior
# encoder) and the discriminators are training-only and never touched.
INFERENCE_SUBMODULE_NAMES = ("enc_p", "dp", "flow", "dec", "f0_predictor")


def _make_weight_fake_quant(ch_axis: int) -> FakeQuantize:
    # ONNX's QuantizeLinear/DequantizeLinear only accept (quant_min, quant_max)
    # of (0, 127), (0, 255) or (-128, 127) -- torch.onnx.export rejects the
    # (-127, 127) range PyTorch's own default qat qconfigs typically use.
    return FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
    )()


def _make_act_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )()


class QATConv1d(nn.Module):
    """Drop-in replacement for nn.Conv1d that fake-quantizes its input
    activation and weight (per-channel, ch_axis=0) on every forward call."""

    def __init__(self, orig: nn.Conv1d):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.stride = orig.stride
        self.padding = orig.padding
        self.dilation = orig.dilation
        self.groups = orig.groups
        self.weight_fake_quant = _make_weight_fake_quant(ch_axis=0)
        self.act_fake_quant = _make_act_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QATConvTranspose1d(nn.Module):
    """Drop-in replacement for nn.ConvTranspose1d. ch_axis=1 -- out_channels
    is dim 1 of a ConvTranspose1d weight ([in, out, k])."""

    def __init__(self, orig: nn.ConvTranspose1d):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.stride = orig.stride
        self.padding = orig.padding
        self.output_padding = orig.output_padding
        self.dilation = orig.dilation
        self.groups = orig.groups
        self.weight_fake_quant = _make_weight_fake_quant(ch_axis=1)
        self.act_fake_quant = _make_act_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return F.conv_transpose1d(
            x, w, self.bias, self.stride, self.padding, self.output_padding,
            self.groups, self.dilation,
        )


class QATLinear(nn.Module):
    """Drop-in replacement for nn.Linear (ch_axis=0 -- out_features). Needed
    for Vocos's dec (pwconv1/pwconv2/head.out are nn.Linear, not Conv) --
    unused for baseline's flow+enc_p+dp scope today, but kept generic so the
    same module serves vocos_small's future QAT run."""

    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.weight_fake_quant = _make_weight_fake_quant(ch_axis=0)
        self.act_fake_quant = _make_act_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return F.linear(x, w, self.bias)


_WRAPPABLE = {
    nn.Conv1d: QATConv1d,
    nn.ConvTranspose1d: QATConvTranspose1d,
    nn.Linear: QATLinear,
}
_QAT_WRAPPER_TYPES = tuple(_WRAPPABLE.values())


def remove_all_weight_norm(model_g) -> None:
    """Fuse every weight_norm-parametrized Conv/ConvTranspose in model_g
    into a plain weight Parameter -- required before wrapping (eager
    weight_norm reassigns `.weight` via a forward pre-hook on every call, so
    a naive wrapper that captures `orig.weight` once would hold a stale
    tensor). model_g.dec.remove_weight_norm() and model_g.flow.
    remove_weight_norm() already recurse through their own submodules
    (ResidualCouplingBlock.remove_weight_norm() loops model_g.flow.flows
    itself -- see vits/modules/flow_block.py); VocosGenerator.
    remove_weight_norm() is a documented no-op (LayerNorm-based, no
    weight_norm anywhere), safe to call unconditionally for both dec
    architectures."""
    with torch.no_grad():
        model_g.dec.remove_weight_norm()
        model_g.flow.remove_weight_norm()


def _wrap_leaves(module: nn.Module, exclude_names: frozenset, wrap_types: tuple) -> int:
    n_wrapped = 0
    for name, child in list(module.named_children()):
        if name in exclude_names:
            continue  # leaf name explicitly excluded (e.g. dec.conv_post)
        if isinstance(child, _QAT_WRAPPER_TYPES):
            continue  # already wrapped (idempotent re-entry)
        wrapper_cls = _WRAPPABLE.get(type(child))
        if wrapper_cls is not None and type(child) in wrap_types:
            setattr(module, name, wrapper_cls(child))
            n_wrapped += 1
        else:
            n_wrapped += _wrap_leaves(child, exclude_names, wrap_types)
    return n_wrapped


def prepare_qat(
    model_g,
    submodule_names: Iterable[str] = INFERENCE_SUBMODULE_NAMES,
    exclude_leaf_names: Iterable[str] = (),
    wrap_types: Iterable[type] = (nn.Conv1d, nn.ConvTranspose1d, nn.Linear),
):
    """Wrap every leaf inside the named inference-path submodules of
    model_g (a SynthesizerTrn) whose type is in `wrap_types` with a
    fake-quantizing replacement, in place. Call remove_all_weight_norm()
    first. Returns model_g.

    exclude_leaf_names: leaf attribute names to skip wrapping regardless of
    which submodule they're found in (e.g. "conv_post" for baseline's dec,
    "out" for Vocos's dec.head) -- use this to keep QAT wrapping scope
    consistent with whichever layers actually end up quantized at export
    time (see the PTQ sweep report for how this scope was determined).

    wrap_types: restrict which nn.Module types get wrapped -- e.g. baseline's
    validated PTQ scope only ever quantizes Conv1d/ConvTranspose1d (never the
    Linear layers inside enc_p/dp's attention blocks), so QAT-wrapping those
    Linear layers too would spend training capacity adapting weights to
    fake-quant noise for a layer that will never actually be exported as
    INT8 -- pure waste, not just a harmless mismatch. Default wraps all
    three (matches vocos_small's scope, where dec's pwconv1/pwconv2/head.out
    genuinely are nn.Linear and do get quantized at export).
    """
    exclude = frozenset(exclude_leaf_names)
    wrap_types = tuple(wrap_types)
    total = 0
    for name in submodule_names:
        submodule = getattr(model_g, name, None)
        if submodule is None:
            continue
        n = _wrap_leaves(submodule, exclude, wrap_types)
        _LOGGER.info("Wrapped %d layer(s) in model_g.%s for QAT", n, name)
        total += n
    _LOGGER.info("Total quantized layers: %d", total)
    return model_g


def set_fake_quant_enabled(model_g, enabled: bool) -> None:
    for m in model_g.modules():
        if isinstance(m, _QAT_WRAPPER_TYPES):
            m.weight_fake_quant.enable_fake_quant(enabled)
            m.act_fake_quant.enable_fake_quant(enabled)


def set_observer_enabled(model_g, enabled: bool) -> None:
    """Freeze/unfreeze the running min/max statistics. Standard QAT
    practice: disable observer updates for the last chunk of fine-tuning so
    exported scale/zero_point stop drifting right before convert/export."""
    for m in model_g.modules():
        if isinstance(m, _QAT_WRAPPER_TYPES):
            m.weight_fake_quant.enable_observer(enabled)
            m.act_fake_quant.enable_observer(enabled)


def count_qat_layers(model_g) -> int:
    return sum(1 for m in model_g.modules() if isinstance(m, _QAT_WRAPPER_TYPES))
