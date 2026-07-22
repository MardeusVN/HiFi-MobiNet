"""SynthesizerTrn: the top-level generator model (text -> audio), tying
together the text encoder, posterior encoder, normalizing flow, duration
predictor, and HiFi-GAN decoder described in the other vits/ files.

`voice_conversion()` (upstream: encode a source speaker's audio, re-decode
with a different speaker's embedding) is dropped here -- it's multi-speaker
-only functionality (asserts n_speakers > 1) that can never run for this
project's permanently-single-speaker scope, unlike the gin_channels/g=None
plumbing elsewhere which is exercised (as a no-op) on every single-speaker
forward pass too.
"""
import math
import typing

import torch
from torch import nn

from ..utils import commons
from ..utils.monotonic_align import maximum_path
from .duration_predictor import DurationPredictor, StochasticDurationPredictor
from .flow_block import ResidualCouplingBlock
from .generator import Generator
from .posterior_encoder import PosteriorEncoder
from .text_encoder import TextEncoder


class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        n_speakers: int = 1,
        gin_channels: int = 0,
        use_sdp: bool = True,
        posterior_encoder_kernel_size: int = 5,
        posterior_encoder_dilation_rate: int = 1,
        posterior_encoder_layers: int = 16,
        flow_kernel_size: int = 5,
        flow_dilation_rate: int = 1,
        flow_n_flows: int = 4,
        use_transformer_flows: bool = True,
        use_snake_beta: bool = True,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            use_snake_beta=use_snake_beta,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            posterior_encoder_kernel_size,
            posterior_encoder_dilation_rate,
            posterior_encoder_layers,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            flow_kernel_size,
            flow_dilation_rate,
            flow_n_flows,
            gin_channels=gin_channels,
            use_transformer_flows=use_transformer_flows,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def _speaker_embedding(self, sid):
        if self.n_speakers > 1:
            return self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        return None

    def forward(self, x, x_lengths, y, y_lengths, sid=None, mas_noise_scale: float = 0.0):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker_embedding(sid)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            attn = self._monotonic_alignment(x_mask, y_mask, z_p, m_p, logs_p, mas_noise_scale)

        w = attn.sum(2)
        logw_ = torch.log(w + 1e-6) * x_mask
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            # Sample a predicted duration (reverse mode) purely to feed the
            # duration discriminator -- same call infer() uses, does not
            # affect l_length (the SDP's own NLL loss).
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
        else:
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        # Expand the per-phoneme prior to per-frame using the alignment.
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
        )

    def _monotonic_alignment(self, x_mask, y_mask, z_p, m_p, logs_p, mas_noise_scale: float = 0.0):
        """Finds the most likely phoneme<->frame alignment given the
        current prior, via the negative cross-entropy between each frame's
        posterior sample and each phoneme's prior distribution.

        mas_noise_scale (VITS2's "noise-scaled MAS"): early in training the
        prior/posterior are poorly calibrated, so the argmax alignment can
        lock onto a bad path and never recover (monotonic search has no way
        to "undo" a bad early commitment). Injecting noise proportional to
        neg_cent's own spread lets a few different alignments win early on;
        the caller anneals this to 0 over training so it converges to the
        exact deterministic search vanilla VITS always used.
        """
        s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_s]
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
        neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)  # [b, t_t, t_s]
        neg_cent3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)  # [b, t_t, t_s]
        neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        if mas_noise_scale > 0:
            neg_cent = neg_cent + torch.randn_like(neg_cent) * torch.std(neg_cent) * mas_noise_scale

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        return maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker_embedding(sid)

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask * length_scale)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        # [b, t', t] x [b, t, d] -> [b, d, t']: expand per-phoneme prior to per-frame
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)
