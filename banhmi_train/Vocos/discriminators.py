# Ported from the official gemelo-ai/vocos repo's vocos/discriminators.py
# -- verified against the actual source via raw.githubusercontent.com, not
# reconstructed from memory. Two deliberate substitutions, both dependency-
# only (architecture and math unchanged):
#   - `torchaudio.transforms.Spectrogram` -> plain `torch.stft` (equivalent
#     at power=None: complex STFT, Hann window, center=True) -- this
#     project doesn't depend on torchaudio anywhere else (mel_processing.py
#     uses raw torch.stft too), and it isn't installed in this env.
#   - `einops.rearrange` -> `.permute()` -- einops isn't installed either;
#     one rearrange call doesn't justify adding the dependency.
# Also omitted: `num_embeddings`/`cond_embedding_id` (upstream's
# conditional-bandwidth embedding for their multi-bandwidth model) -- this
# project has no equivalent concept, single output per forward call.
#
# IMPORTANT: despite the matching class name, `MultiResolutionDiscriminator`
# here is NOT the same architecture as `BigVGan/discriminators.py`'s (which
# is UnivNet-style: one conv stack over the *whole* spectrum, per
# resolution). This one is DAC-style (Descript Audio Codec): at each STFT
# resolution, the frequency axis is split into 5 sub-bands and each gets
# its *own* independent conv stack. `MultiPeriodDiscriminator` matches
# HiFi-GAN/BigVGan's exactly (upstream reuses it as-is).
import typing

import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.utils import weight_norm

_LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
                weight_norm(Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(kernel_size // 2, 0))),
            ]
        )
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        # Upstream takes raw [B, T] and unsqueezes to [B, 1, T] itself; this
        # project's convention (Generator output, BigVGan/discriminators.py)
        # is [B, 1, T] already -- interface-only adaptation, same as the
        # torchaudio/einops substitutions noted at the top of this file.
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, layer in enumerate(self.convs):
            x = F.leaky_relu(layer(x), _LRELU_SLOPE)
            if i > 0:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """Identical to HiFi-GAN's / BigVGan/discriminators.py's -- upstream
    Vocos reuses this architecture as-is, no changes."""

    _PERIODS = (2, 3, 5, 7, 11)

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in self._PERIODS])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _run_discriminators(self.discriminators, y, y_hat)


class DiscriminatorR(nn.Module):
    """Single-resolution, multi-band STFT discriminator (DAC-style): the
    frequency axis is split into 5 fixed sub-bands (as fractions of
    Nyquist), each scored by its own independent conv stack, then
    concatenated back together before the shared `conv_post`."""

    _BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))

    def __init__(self, window_length: int, channels: int = 32, hop_factor: float = 0.25):
        super().__init__()
        self.window_length = window_length
        self.hop_length = int(window_length * hop_factor)
        self.register_buffer("window", torch.hann_window(window_length))

        n_freq = window_length // 2 + 1
        self.bands = [(int(lo * n_freq), int(hi * n_freq)) for lo, hi in self._BANDS]

        def make_band_convs() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    weight_norm(Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                    weight_norm(Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                    weight_norm(Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
                ]
            )

        self.band_convs = nn.ModuleList([make_band_convs() for _ in self.bands])
        self.conv_post = weight_norm(Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def _spectrogram(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        x = x - x.mean(dim=-1, keepdim=True)  # remove DC offset
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)  # peak-normalize
        # cuFFT doesn't support half/bf16 input (mel_processing.spectrogram_torch
        # works around the same constraint) -- not present upstream, added
        # here for this project's bf16 training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(
                x.float(), n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length,
                window=self.window, center=True, return_complex=True,
            )
        spec = torch.view_as_real(spec)  # [B, Freq, Frames, 2]
        spec = spec.permute(0, 3, 2, 1)  # [B, 2, Frames, Freq]
        return [spec[..., lo:hi] for lo, hi in self.bands]

    def forward(self, x: torch.Tensor):
        x_bands = self._spectrogram(x.squeeze(1))
        fmap = []
        band_outputs = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = F.leaky_relu(layer(band), _LRELU_SLOPE)
                if i > 0:
                    fmap.append(band)
            band_outputs.append(band)
        x = torch.cat(band_outputs, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    _FFT_SIZES = (2048, 1024, 512)

    def __init__(self, fft_sizes: typing.Optional[typing.Tuple[int, ...]] = None):
        super().__init__()
        fft_sizes = fft_sizes or self._FFT_SIZES
        self.discriminators = nn.ModuleList([DiscriminatorR(w) for w in fft_sizes])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _run_discriminators(self.discriminators, y, y_hat)


def _run_discriminators(discriminators, y: torch.Tensor, y_hat: torch.Tensor):
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
    for d in discriminators:
        y_d_r, fmap_r = d(y)
        y_d_g, fmap_g = d(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)
    return y_d_rs, y_d_gs, fmap_rs, fmap_gs
