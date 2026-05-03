"""Gradient Reversal Layer (Ganin & Lempitsky, 2015).

The GRL is the identity in the forward pass and multiplies the gradient by
``-lambda`` in the backward pass. This allows training a domain discriminator
adversarially against the feature encoder using a single backward call.
"""

from typing import Any

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Custom autograd Function: forward is identity, backward multiplies by -lambda."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        # negate gradient and scale by lambda; lambda is not a learnable param
        return grad_output.neg() * ctx.lambda_, None


def gradient_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Functional API for the gradient reversal layer.

    Args:
        x: Input feature tensor.
        lambda_: Scalar weight for the reverse-gradient. Typically scheduled
            from 0 (warm-up) to 1 over training; see ``utils.schedules.grl_lambda_ganin``.

    Returns:
        Tensor identical to ``x`` in forward, but with negated/scaled gradient
        in backward.
    """
    return GradientReversalFunction.apply(x, lambda_)


class GradientReversalLayer(torch.nn.Module):
    """nn.Module wrapper around ``gradient_reverse``.

    Stores ``lambda_`` as a buffer that can be updated step-by-step from outside
    (e.g., from the training loop) without affecting the optimizer state.
    """

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        # we keep lambda as a Python float to avoid device sync issues;
        # it is NOT a learnable parameter
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gradient_reverse(x, self.lambda_)