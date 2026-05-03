"""Schedule functions used across DA trainers (lambda for GRL, learning rate)."""

import math


def grl_lambda_ganin(progress: float, gamma: float = 10.0, lambda_max: float = 1.0) -> float:
    """Compute the GRL lambda following Ganin et al. (2016).

    The original schedule is:
        lambda_p = (2 / (1 + exp(-gamma * p))) - 1
    where p in [0, 1] is the training progress (current_step / total_steps).

    This makes lambda grow smoothly from 0 to 1 (or to ``lambda_max``),
    so the discriminator has time to warm up before pushing the encoder hard.

    Args:
        progress: Float in [0, 1].
        gamma: Steepness of the sigmoid; higher = faster ramp up. Default 10.
        lambda_max: Final value of lambda at the end of training.

    Returns:
        Current lambda for GRL.
    """
    progress = max(0.0, min(1.0, progress))
    coeff = (2.0 / (1.0 + math.exp(-gamma * progress))) - 1.0
    return float(coeff * lambda_max)