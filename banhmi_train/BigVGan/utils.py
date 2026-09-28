# Ported from the official NVIDIA/BigVGAN repo's utils.py (init_weights /
# get_padding only -- the checkpoint I/O and plotting helpers there aren't
# needed here, this package plugs into banhmi_train's own Trainer instead).
import torch


def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if module.__class__.__name__.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2
