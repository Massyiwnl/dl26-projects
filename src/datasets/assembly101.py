"""PyTorch Dataset for Assembly101 segment-level features.

Reads the .npz produced by `precompute_segment_features.py`
(or `make_synthetic_segment_features.py` for local development).

Each .npz contains:
    features    : (N, FEATURE_DIM) float32  — segment embeddings
    labels      : (N,) int64                — verb_id (0..23)
    segment_ids : (N,) int64                — original CSV row id

The whole .npz fits comfortably in RAM (~700 MB at full scale), so we load
it eagerly in the constructor: this gives effectively zero-cost __getitem__
during training, which removes the data loader from the bottleneck path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class Assembly101SegmentDataset(Dataset):
    """In-memory dataset of pre-extracted segment-level features.

    Args:
        npz_path: Path to the .npz file (one per (split, domain) combination).
        return_segment_id: If True, __getitem__ also returns the original CSV
            row id; useful for per-segment analysis and debugging.
    """

    def __init__(self, npz_path: str | Path, return_segment_id: bool = False) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Segment-feature file not found: {npz_path}")

        with np.load(npz_path) as data:
            self.features = torch.from_numpy(data["features"].astype(np.float32))
            self.labels = torch.from_numpy(data["labels"].astype(np.int64))
            self.segment_ids = torch.from_numpy(data["segment_ids"].astype(np.int64))

        self.npz_path = npz_path
        self.return_segment_id = return_segment_id

        if not (len(self.features) == len(self.labels) == len(self.segment_ids)):
            raise ValueError(
                f"Inconsistent lengths in {npz_path}: "
                f"features={len(self.features)}, labels={len(self.labels)}, "
                f"segment_ids={len(self.segment_ids)}"
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]
        if self.return_segment_id:
            return x, y, self.segment_ids[idx]
        return x, y

    @property
    def num_classes(self) -> int:
        return int(self.labels.max().item()) + 1

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def class_counts(self) -> torch.Tensor:
        """Return a (num_classes,) tensor with the number of samples per class."""
        return torch.bincount(self.labels, minlength=self.num_classes)