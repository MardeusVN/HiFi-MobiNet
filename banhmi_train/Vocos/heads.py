# Ported from the official gemelo-ai/vocos repo's vocos/spectral_ops.py
# (ISTFT) and vocos/heads.py (ISTFTHead) -- verified against the actual
# source via raw.githubusercontent.com/gemelo-ai/vocos/main/vocos/
# {spectral_ops,heads}.py, not reconstructed from memory.
"""Turns the backbone's per-frame features into a waveform via a predicted
STFT (log-magnitude + phase per bin) and an inverse STFT -- the piece that
replaces HiFi-GAN/BigVGAN's entire ConvTranspose1d upsampling stack. This
is why Vocos is fast: the only resolution-changing operation left is the
ISTFT, and it's close to parameter-free (a fixed overlap-add), not a
learned conv stack.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ISTFT(nn.Module):
    """Custom ISTFT supporting "same" padding (analogous to a CNN's "same"
    padding), not just "center" -- plain `torch.istft` only supports
    "center" because its NOLA (Nonzero Overlap-Add) check fails at the
    edges otherwise (pytorch/pytorch#62323). "same" is fine here because
    the edge samples get trimmed away regardless, so the NOLA violation at
    the very edge never surfaces in the output.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ("center", "same"):
            raise ValueError('padding must be "center" or "same"')
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        # Precomputed real-valued one-sided IDFT basis, mathematically
        # identical to torch.fft.irfft(..., norm="backward") -- used only by
        # forward_real() below (the ONNX-exportable path: torch.onnx's
        # exporter has no op for aten::complex, so torch.complex()+
        # torch.fft.irfft() can't be traced into an ONNX graph at all).
        # x[n] = (1/n_fft) * sum_k weight[k] * (Re[k]*cos(2*pi*k*n/n_fft) -
        # Im[k]*sin(2*pi*k*n/n_fft)), weight=1 at k=0 and Nyquist (self-
        # conjugate bins under Hermitian symmetry), 2 elsewhere (each mirrors
        # a folded-back negative-frequency bin). Verified numerically against
        # torch.fft.irfft in test_function/ before this was wired in.
        num_bins = n_fft // 2 + 1
        k = torch.arange(num_bins).unsqueeze(1)  # [num_bins, 1]
        n = torch.arange(n_fft).unsqueeze(0)  # [1, n_fft]
        angle = 2 * math.pi * k * n / n_fft  # [num_bins, n_fft]
        weight = torch.full((num_bins, 1), 2.0)
        weight[0, 0] = 1.0
        if n_fft % 2 == 0:
            weight[-1, 0] = 1.0
        # persistent=False: purely a deterministic function of n_fft, not
        # learned/data-dependent -- keeping it out of state_dict() means old
        # checkpoints (saved before this buffer existed) still load cleanly.
        self.register_buffer("idft_cos", torch.cos(angle) * weight / n_fft, persistent=False)
        self.register_buffer("idft_sin", -torch.sin(angle) * weight / n_fft, persistent=False)

        # Identity-scatter kernel for the ONNX-exportable overlap-add below
        # (conv_transpose1d in place of F.fold/col2im): weight[c, 0, c] = 1
        # means input channel c (== sample c within a win_length-long frame)
        # lands at kernel offset c within conv_transpose1d's placement of
        # that frame -- i.e. exactly what F.fold's col2im does, frame by
        # frame, with the same automatic overlap-add across frames via
        # stride=hop_length. Only needed for forward_real's onnx_export
        # path -- F.fold is fine (and already relied upon) elsewhere.
        self.register_buffer(
            "_scatter_kernel", torch.eye(win_length).unsqueeze(1), persistent=False
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [B, n_fft//2+1, T] complex
        if self.padding == "center":
            return torch.istft(
                spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True
            )
        return self._overlap_add(torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward"))

    def forward_real(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """Same result as forward() with padding="same", but takes the
        spectrum as separate real/imag tensors and reconstructs frames via
        the matmul-based IDFT above instead of torch.complex+torch.fft --
        the ONNX-exportable path. O(n_fft^2) per frame vs FFT's O(n_fft log
        n_fft), so noticeably slower; only meant for tracing an export, not
        for training or normal PyTorch-side inference (use forward() there).
        """
        if self.padding != "same":
            raise NotImplementedError('forward_real only supports padding="same"')
        # [B, num_bins, T] x [num_bins, n_fft] -> [B, n_fft, T]
        ifft = torch.einsum("bkt,kn->bnt", real, self.idft_cos) + torch.einsum(
            "bkt,kn->bnt", imag, self.idft_sin
        )
        return self._overlap_add_onnx(ifft)

    def _overlap_add(self, ifft: torch.Tensor) -> torch.Tensor:
        ifft = ifft * self.window[None, :, None]

        pad = (self.win_length - self.hop_length) // 2
        t = ifft.shape[-1]
        output_size = (t - 1) * self.hop_length + self.win_length
        y = F.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length)
        )[:, 0, 0, pad:-pad]

        window_sq = self.window.square().expand(1, t, -1).transpose(1, 2)
        window_envelope = F.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length)
        ).squeeze()[pad:-pad]

        return y / window_envelope.clamp_min(1e-11)

    def _overlap_add_onnx(self, ifft: torch.Tensor) -> torch.Tensor:
        """Same result as _overlap_add, via conv_transpose1d instead of
        F.fold -- torch.onnx's col2im symbolic (added opset 18) breaks on a
        dynamic (trace-time-computed) output_size, which F.fold's call
        needs here since output length depends on the input phoneme count.
        conv_transpose1d has no such issue and is supported since opset 1.
        """
        ifft = ifft * self.window[None, :, None]  # [B, win_length, T]

        pad = (self.win_length - self.hop_length) // 2
        t = ifft.shape[-1]
        y = F.conv_transpose1d(ifft, self._scatter_kernel, stride=self.hop_length)[:, 0, pad:-pad]

        window_sq = self.window.square().expand(1, t, -1).transpose(1, 2)  # [1, win_length, T]
        window_envelope = F.conv_transpose1d(window_sq, self._scatter_kernel, stride=self.hop_length)[
            0, 0, pad:-pad
        ]

        return y / window_envelope.clamp_min(1e-11)


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2  # log-magnitude + phase per one-sided freq bin
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor, onnx_export: bool = False) -> torch.Tensor:
        # x: [B, T, dim] (backbone output, channel-last)
        x = self.out(x).transpose(1, 2)  # [B, n_fft + 2, T]
        mag, phase = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)  # guards against inf blowing up istft early in training
        # Directly producing the complex value (mag * (cos(p) + j sin(p))) is
        # both correct and cheaper than atan2->exp(phase*1j) round-tripping.
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        real, imag = mag * cos_p, mag * sin_p
        if onnx_export:
            # torch.onnx has no op for aten::complex, so the fft path below
            # can't be traced -- route through the matmul-based IDFT instead
            # (see ISTFT.forward_real). Slower, only used for export tracing.
            return self.istft.forward_real(real.float(), imag.float()).unsqueeze(1)
        # cuFFT (torch.fft's backend on CUDA) doesn't support bf16/fp16,
        # same constraint mel_processing.spectrogram_torch works around --
        # not present upstream, added here for this project's bf16 training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.complex(real.float(), imag.float())
            audio = self.istft(spec)
        return audio.unsqueeze(1)  # [B, 1, T_wav]
