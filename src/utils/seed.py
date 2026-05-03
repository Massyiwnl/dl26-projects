"""Reproducibility utilities for deterministic training."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set RNG seeds for Python, NumPy and PyTorch (CPU/CUDA).

    Args:
        seed: Integer seed used everywhere.
        deterministic: If True, force CuDNN deterministic algorithms (slower
            but fully reproducible). Use False for production speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True