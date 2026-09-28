# Ported from the official NVIDIA/BigVGAN repo's activations.py, which is
# itself adapted from https://github.com/EdwardDixon/snake (MIT license).
# Snake: https://arxiv.org/abs/2006.08195 (Liu, Hartwig, Ueda -- "Neural
# Networks Fail to Learn Periodic Functions and How to Fix It").
"""Periodic activation functions: Snake (single learnable alpha) and
SnakeBeta (separate alpha/beta). Used *only* wrapped in
`alias_free_torch.act.Activation1d` inside this package's AMPBlocks -- calling
these directly (as `vits.utils.normalization.SnakeBeta` does) omits BigVGAN's
anti-aliasing and is a different, weaker variant (see that file's docstring).
"""
import torch
from torch import nn, pow, sin
from torch.nn import Parameter


class Snake(nn.Module):
    """Snake := x + (1/alpha) * sin^2(alpha * x). alpha is a single
    learnable parameter per channel (shared frequency and inverse-magnitude
    role); higher alpha = higher-frequency periodicity."""

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [C] -> [1, C, 1]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    """SnakeBeta := x + (1/beta) * sin^2(alpha * x). Separates the
    frequency role (alpha) from the inverse-magnitude role (beta) into two
    independent learnable parameters per channel -- this is the variant
    BigVGAN's own released configs (e.g. bigvgan_base_24khz) actually use,
    and the one `vits.utils.normalization.SnakeBeta` mirrors (minus AMP)."""

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
