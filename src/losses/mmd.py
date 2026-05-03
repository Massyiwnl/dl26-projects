"""Maximum Mean Discrepancy (MMD) loss with multi-kernel Gaussian RBF.

Reference: Gretton et al. (2012), "A Kernel Two-Sample Test", JMLR.
Used in DAN (Long et al., 2015) as a feature alignment loss for DA.

The squared MMD between two distributions P (source) and Q (target),
estimated from samples, is:

    MMD^2(P, Q) = E_{x,x'~P}[k(x,x')] + E_{y,y'~Q}[k(y,y')]
                  - 2 * E_{x~P, y~Q}[k(x,y)]

We use a mixture of Gaussian RBF kernels with multiple bandwidths to be more
robust than a single-bandwidth kernel:

    k(x, y) = sum_i exp( -|x - y|^2 / (2 * sigma_i^2) )

Bandwidths sigma_i are chosen as multiples of the median pairwise distance
(median heuristic, standard in the literature).
"""

from typing import Sequence

import torch


def _pairwise_squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the (n, m) matrix of squared L2 distances between rows of a and b."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a_sq = (a * a).sum(dim=1, keepdim=True)            # (n, 1)
    b_sq = (b * b).sum(dim=1, keepdim=True).t()        # (1, m)
    return a_sq + b_sq - 2.0 * a @ b.t()


def _gaussian_multi_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    sigmas: Sequence[float],
) -> torch.Tensor:
    """Sum of Gaussian RBF kernels evaluated on every pair of rows of (a, b).

    Returns a tensor of shape (n, m).
    """
    d2 = _pairwise_squared_distances(a, b)             # (n, m)
    kernel = torch.zeros_like(d2)
    for sigma in sigmas:
        kernel = kernel + torch.exp(-d2 / (2.0 * sigma * sigma + 1e-12))
    return kernel


def _median_heuristic_sigma(features: torch.Tensor) -> float:
    """Median pairwise L2 distance over the rows of ``features``.

    Used as the central bandwidth from which a multi-kernel set is built.
    Returns a scalar Python float.
    """
    with torch.no_grad():
        d2 = _pairwise_squared_distances(features, features)
        # take strict upper triangle, off-diagonal, to avoid zeros
        n = features.size(0)
        mask = torch.triu(torch.ones(n, n, device=features.device), diagonal=1).bool()
        d = torch.sqrt(d2[mask].clamp_min(1e-12))
        median = d.median().item()
    return float(max(median, 1e-6))


def multi_kernel_mmd2(
    source: torch.Tensor,
    target: torch.Tensor,
    sigma_multipliers: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    sigma: float | None = None,
) -> torch.Tensor:
    """Compute the squared multi-kernel MMD between source and target features.

    Args:
        source: (n, d) source-domain features.
        target: (m, d) target-domain features.
        sigma_multipliers: Multiplicative factors applied to ``sigma`` to obtain
            the bandwidths of the kernel mixture.
        sigma: If None, computed via the median heuristic on the merged batch.

    Returns:
        A scalar tensor: MMD^2(source, target). Always >= 0 in the population
        limit, but the unbiased empirical estimator can be slightly negative
        on small batches; we clamp the output at zero for numerical sanity.
    """
    if sigma is None:
        merged = torch.cat([source, target], dim=0)
        sigma = _median_heuristic_sigma(merged)

    sigmas = [sigma * m for m in sigma_multipliers]

    k_ss = _gaussian_multi_kernel(source, source, sigmas).mean()
    k_tt = _gaussian_multi_kernel(target, target, sigmas).mean()
    k_st = _gaussian_multi_kernel(source, target, sigmas).mean()

    mmd2 = k_ss + k_tt - 2.0 * k_st
    return mmd2.clamp_min(0.0)