"""Combined source/target data loader for DA training (DANN, MMD).

At each training step we need a labelled batch from the SOURCE domain and
an unlabelled batch from the TARGET domain. Since the two domains have
different sizes, we want the loader to:
  * never run out of one domain before the other (we keep cycling),
  * use independent random sampling for each domain (otherwise the batches
    are biased by sequential order on disk).

`PairedDomainIterator` wraps two PyTorch DataLoaders and yields one
(src_batch, tgt_batch) tuple per call to next(). It defines an "epoch" as
`steps_per_epoch` calls; usually we set this to len(source_loader), so each
training epoch corresponds to one full pass over the labelled source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader


@dataclass
class PairedBatch:
    """One step of training: a labelled source batch and an unlabelled target batch.

    Both `src_x` and `tgt_x` have shape (B, feature_dim). `src_y` is (B,) int64.
    `tgt_y` is also kept (we have it because the .npz has labels for evaluation,
    but DA training MUST NOT use it).
    """
    src_x: torch.Tensor
    src_y: torch.Tensor
    tgt_x: torch.Tensor
    tgt_y: torch.Tensor   # for evaluation only; MUST NOT be passed to the loss


class PairedDomainIterator:
    """Iterate over source and target loaders simultaneously, cycling each.

    Args:
        src_loader: DataLoader over the labelled source dataset.
        tgt_loader: DataLoader over the (nominally unlabelled) target dataset.
        steps_per_epoch: How many paired batches make one epoch. Usually
            len(src_loader). If None, defaults to max(len(src_loader),
            len(tgt_loader)).
        device: Optional device to move batches to before returning. If None,
            batches stay on CPU and the trainer is responsible for the move.
    """

    def __init__(
        self,
        src_loader: DataLoader,
        tgt_loader: DataLoader,
        steps_per_epoch: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.device = device

        if steps_per_epoch is None:
            steps_per_epoch = max(len(src_loader), len(tgt_loader))
        self.steps_per_epoch = steps_per_epoch

        self._src_iter: Iterator | None = None
        self._tgt_iter: Iterator | None = None

    def _reset_src(self) -> None:
        self._src_iter = iter(self.src_loader)

    def _reset_tgt(self) -> None:
        self._tgt_iter = iter(self.tgt_loader)

    def _next_src(self):
        if self._src_iter is None:
            self._reset_src()
        try:
            return next(self._src_iter)
        except StopIteration:
            self._reset_src()
            return next(self._src_iter)

    def _next_tgt(self):
        if self._tgt_iter is None:
            self._reset_tgt()
        try:
            return next(self._tgt_iter)
        except StopIteration:
            self._reset_tgt()
            return next(self._tgt_iter)

    def __iter__(self) -> "PairedDomainIterator":
        self._reset_src()
        self._reset_tgt()
        self._step = 0
        return self

    def __next__(self) -> PairedBatch:
        if self._step >= self.steps_per_epoch:
            raise StopIteration
        self._step += 1

        src_x, src_y = self._next_src()
        tgt_x, tgt_y = self._next_tgt()

        if self.device is not None:
            src_x = src_x.to(self.device, non_blocking=True)
            src_y = src_y.to(self.device, non_blocking=True)
            tgt_x = tgt_x.to(self.device, non_blocking=True)
            tgt_y = tgt_y.to(self.device, non_blocking=True)

        return PairedBatch(src_x=src_x, src_y=src_y, tgt_x=tgt_x, tgt_y=tgt_y)

    def __len__(self) -> int:
        return self.steps_per_epoch