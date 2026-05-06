"""Lightweight checkpoint I/O for trainers in this project.

A checkpoint is a single .pt file containing:
    state_dict   : model weights
    optimizer    : optimizer state (optional, for resume)
    epoch        : last completed epoch
    best_metric  : best validation metric so far (used for "best.pt" tracking)
    config       : the YAML config used for this run, as a plain dict
    extra        : any free-form metadata (e.g., per-epoch losses)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    config: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Save a checkpoint to ``path``. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if best_metric is not None:
        state["best_metric"] = float(best_metric)
    if config is not None:
        state["config"] = config
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint from ``path``. If ``model`` is given, load weights into it.
    If ``optimizer`` is given and the checkpoint has optimizer state, load it.
    Returns the full checkpoint dict.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt