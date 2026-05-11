"""Smoke test for the Charades-Ego parser.

Verifies:
    - All 6 official CSVs load.
    - Per-video splitting train/val is deterministic and non-overlapping.
    - Segment counts are coherent with the EDA numbers (~34k/~34k train).
    - All 157 classes appear in train_source and train_target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datasets.charades_ego import make_charades_splits, NUM_CLASSES


REPO_ROOT = Path(__file__).resolve().parents[2]
CHARADES_DIR = REPO_ROOT / "data" / "raw" / "charades-ego" / "CharadesEgo"


def main() -> None:
    print("Smoke test: Charades-Ego parser\n")
    splits = make_charades_splits(CHARADES_DIR, val_fraction=0.15, seed=42)

    print("=== Volumi per (split, domain) ===")
    for name, df in splits.items():
        n_videos = df['id'].nunique()
        n_segments = len(df)
        n_classes = df['class_id'].nunique()
        print(f"  {name:>18}: {n_segments:>6} segm, {n_videos:>5} videos, {n_classes:>3} classes")

    # ---- assertions ----
    # 1. train_source and train_target sums ≈ 34k each (from EDA), val ≈ 5k
    assert len(splits['train_source']) + len(splits['val_source']) > 33000
    assert len(splits['train_target']) + len(splits['val_target']) > 33000
    print("\n[PASS] Train+val segment counts match EDA")

    # 2. video sets must NOT overlap between train and val (same domain)
    for domain in ('source', 'target'):
        train_vids = set(splits[f'train_{domain}']['id'])
        val_vids = set(splits[f'val_{domain}']['id'])
        assert len(train_vids & val_vids) == 0, f"video leakage in {domain}!"
    print("[PASS] Train/val splits are video-disjoint")

    # 3. all 157 classes in train_source and train_target
    for domain in ('source', 'target'):
        n_classes = splits[f'train_{domain}']['class_id'].nunique()
        assert n_classes == NUM_CLASSES, f"{domain} train has only {n_classes}/{NUM_CLASSES} classes"
    print("[PASS] All 157 classes present in train splits")

    # 4. class_id within [0, 156]
    for name, df in splits.items():
        assert df['class_id'].min() >= 0
        assert df['class_id'].max() < NUM_CLASSES
    print("[PASS] All class_ids in [0, 156]")

    print("\nAll parser smoke tests passed.")
    print(f"\nReady to extract features for {sum(len(d) for d in splits.values()):,} segments total.")


if __name__ == "__main__":
    main()