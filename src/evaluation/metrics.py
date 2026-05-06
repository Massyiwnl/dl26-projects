"""Classification metrics for action recognition under domain adaptation.

Given a 1-D tensor of `pred` (predicted class indices) and `target`
(ground-truth class indices), all functions return a single metric or a
small dict of metrics.

We rely on torchmetrics-style numpy/sklearn implementations for clarity:
    - top-k accuracy is computed manually (so we don't add a dep just for that)
    - balanced accuracy and per-class P/R/F1 use sklearn (already in env)
    - confusion matrix is plain numpy
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, ks: Sequence[int] = (1, 5)) -> dict[str, float]:
    """Top-k accuracy for k in ``ks``.

    Args:
        logits: (N, C) raw model outputs.
        target: (N,) int64 ground-truth class indices.
    Returns:
        dict like {"top1": 0.46, "top5": 0.83}
    """
    out: dict[str, float] = {}
    with torch.no_grad():
        max_k = max(ks)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # (N, max_k)
        correct = pred.eq(target.unsqueeze(1))                          # (N, max_k)
        for k in ks:
            acc = correct[:, :k].any(dim=1).float().mean().item()
            out[f"top{k}"] = acc
    return out


def aggregate_metrics(
    logits_or_pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    num_classes: int,
    is_logits: bool = True,
) -> dict[str, float | np.ndarray]:
    """Return the bundle of metrics we report in the paper:
        top1, top5 (only if logits given), balanced_accuracy, macro_f1,
        per-class precision / recall / f1, confusion_matrix.

    Args:
        logits_or_pred: (N, C) logits if ``is_logits`` else (N,) predicted classes.
        target: (N,) ground-truth classes.
        num_classes: total number of classes (24 for our task).
        is_logits: True if first arg is logits, False if already argmaxed.
    """
    if is_logits:
        logits = logits_or_pred
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        topk = topk_accuracy(logits, target, ks=(1, 5))
        pred = logits.argmax(dim=1).cpu().numpy()
        target = target.cpu().numpy()
    else:
        topk = {}
        pred = logits_or_pred.cpu().numpy() if isinstance(logits_or_pred, torch.Tensor) else np.asarray(logits_or_pred)
        target = target.cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    bal_acc = float(balanced_accuracy_score(target, pred))
    macro_f1 = float(f1_score(target, pred, average="macro", zero_division=0))

    p, r, f, support = precision_recall_fscore_support(
        target, pred, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(target, pred, labels=list(range(num_classes)))

    return {
        **topk,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "per_class_precision": p,
        "per_class_recall": r,
        "per_class_f1": f,
        "per_class_support": support,
        "confusion_matrix": cm,
    }


def format_metrics_summary(metrics: dict, num_classes: int = 24) -> str:
    """Human-readable summary string of the headline metrics."""
    lines = []
    if "top1" in metrics:
        lines.append(f"  top1                = {metrics['top1']:.4f}")
    if "top5" in metrics:
        lines.append(f"  top5                = {metrics['top5']:.4f}")
    lines.append(f"  balanced_accuracy   = {metrics['balanced_accuracy']:.4f}")
    lines.append(f"  macro_f1            = {metrics['macro_f1']:.4f}")
    lines.append(f"  classes covered     = {(metrics['per_class_support'] > 0).sum()}/{num_classes}")
    return "\n".join(lines)