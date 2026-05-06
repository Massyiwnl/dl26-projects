"""Smoke test for the Assembly101 segment Dataset and the paired DA loader.

Verifies:
  1. Each .npz loads correctly and has consistent (features, labels, segment_ids).
  2. DataLoader yields tensors of the expected shape and dtype.
  3. PairedDomainIterator yields one source+target batch per step,
     respects the requested steps_per_epoch, and cycles correctly.
  4. The 24 classes are present in source and target train splits.

Run from repo root:
    python -m src.datasets.smoke_test_loaders
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.assembly101 import Assembly101SegmentDataset
from src.datasets.pair_loader import PairedDomainIterator


REPO_ROOT = Path(__file__).resolve().parents[2]
SEG_DIR = REPO_ROOT / "data" / "processed" / "segment_features"


def main() -> None:
    print("Smoke test: Assembly101SegmentDataset + PairedDomainIterator\n")

    # 1. Datasets
    train_src = Assembly101SegmentDataset(SEG_DIR / "train_source.npz")
    train_tgt = Assembly101SegmentDataset(SEG_DIR / "train_target.npz")
    val_src = Assembly101SegmentDataset(SEG_DIR / "val_source.npz")

    print(f"  train_source: {len(train_src):>7} segments, {train_src.num_classes} classes, dim={train_src.feature_dim}")
    print(f"  train_target: {len(train_tgt):>7} segments, {train_tgt.num_classes} classes, dim={train_tgt.feature_dim}")
    print(f"  val_source:   {len(val_src):>7} segments")

    assert train_src.feature_dim == 2048
    assert train_src.num_classes == 24
    assert train_tgt.num_classes == 24
    print("[PASS] Dataset shapes/classes correct")

    # 2. Class counts (long-tail check)
    src_counts = train_src.class_counts()
    print(f"\n  train_source class count: min={src_counts.min().item()}, max={src_counts.max().item()}, total={src_counts.sum().item()}")

    # 3. Single DataLoader smoke
    BATCH = 64
    src_loader = DataLoader(train_src, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(train_tgt, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_src, batch_size=BATCH, shuffle=False, num_workers=0)

    x, y = next(iter(src_loader))
    print(f"\n  src batch: x={tuple(x.shape)} {x.dtype}, y={tuple(y.shape)} {y.dtype}")
    assert x.shape == (BATCH, 2048) and y.shape == (BATCH,)
    assert x.dtype == torch.float32 and y.dtype == torch.int64
    print("[PASS] Source DataLoader yields correct shape/dtype")

    # 4. PairedDomainIterator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_iter = PairedDomainIterator(
        src_loader=src_loader,
        tgt_loader=tgt_loader,
        steps_per_epoch=5,        # tiny for smoke test
        device=device,
    )

    print(f"\n  Iterating 1 fake epoch of {len(pair_iter)} steps on device={device}...")
    n_steps = 0
    for batch in pair_iter:
        assert batch.src_x.shape == (BATCH, 2048)
        assert batch.tgt_x.shape == (BATCH, 2048)
        assert batch.src_y.shape == (BATCH,)
        assert batch.src_x.device.type == device.type
        n_steps += 1
    assert n_steps == 5
    print(f"[PASS] PairedDomainIterator yielded {n_steps} steps as requested")

    # 5. The iterator must cycle (run two epochs back-to-back)
    pair_iter2 = PairedDomainIterator(src_loader, tgt_loader, steps_per_epoch=3, device=device)
    s1 = sum(1 for _ in pair_iter2)
    s2 = sum(1 for _ in pair_iter2)
    assert s1 == 3 and s2 == 3
    print(f"[PASS] PairedDomainIterator cycles cleanly across multiple epochs")

    print("\nAll loader smoke tests passed.")


if __name__ == "__main__":
    main()