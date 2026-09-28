# Combines convnext.py + heads.py, both verified against the official
# gemelo-ai/vocos source (see their docstrings). This VocosGenerator class
# itself has no upstream equivalent -- upstream wires
# feature_extractor/backbone/head together via a jsonargparse config
# (vocos/experiment.py's VocosExp), not a single nn.Module; this is that
# wiring collapsed into one class to match this project's other Generator
# classes' interface.
"""VocosGenerator: VocosBackbone (ConvNeXt stack at frame rate) + ISTFTHead
(predicted STFT -> inverse STFT). Input here is `initial_channel`, fed the
VITS latent `z_slice` exactly like `BigVGan/generator.py` and
`vits/modules/generator.py` -- not upstream Vocos's own mel-spectrogram
input, and not `vits.modules.generator.Generator`'s family, which
upsamples to sample rate with `ConvTranspose1d`. Vocos's whole premise is
that this network never needs to run above the input's own frame rate at
all -- see convnext.py/heads.py.

Default dim/intermediate_dim/num_layers/n_fft/hop_length below match the
released 24kHz mel-conditioned checkpoint's actual config
(configs/vocos.yaml in the upstream repo) -- confirmed ~13.5M backbone
params at these settings, notably *larger* than the ~1.6-2.2M-param
HiFi-GAN-family generators benchmarked in test_function/, since Vocos
trades parameter count for eliminating the upsampling stack's compute.
"""
import torch
from torch import nn

from .convnext import VocosBackbone
from .heads import ISTFTHead


class VocosGenerator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        gin_channels: int = 0,
        use_f0: bool = False,
    ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels=initial_channel,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            gin_channels=gin_channels,
            use_f0=use_f0,
        )
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length)

    def forward(self, x: torch.Tensor, g=None, f0=None, onnx_export: bool = False) -> torch.Tensor:
        x = self.backbone(x, g=g, f0=f0)  # [B, T, dim], channel-last
        return self.head(x, onnx_export=onnx_export)  # [B, 1, T_wav]

    def remove_weight_norm(self):
        # No weight_norm anywhere in this architecture (LayerNorm-based, not
        # weight-normed convs like the HiFi-GAN family) -- kept for
        # interface parity with the other Generator classes.
        pass
