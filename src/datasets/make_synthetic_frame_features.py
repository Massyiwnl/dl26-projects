"""Generate a small SUBSET of synthetic per-video frame-level features
that mimic the expected output of `1_lmdb_to_npy.py`.

Why a subset?
  Generating frame-level features for all ~600 (sequence, view) pairs would
  take ~60 GB of disk space, which is excessive on a development laptop.
  A subset of 20 sequences is enough to unit-test the segment-level
  pre-processing in `2_precompute_segment_features.py`. The full-scale data
  pipeline is meant to run on the DMI cluster, not locally.

  For local end-to-end DANN/MMD smoke tests we use
  `make_synthetic_segment_features.py` instead, which produces directly
  the segment-level .npz files (~1 GB total).

Output: `data/synthetic/frame_features/<sequence_id>__<view>.npy`
        shape (N_frames, 2048), dtype float16 (saves 50% disk).

Run from repo root:
    python -m src.datasets.make_synthetic_frame_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = REPO_ROOT / "data" / "raw" / "assembly101-annotations" / "fine-grained-annotations"
OUT_DIR = REPO_ROOT / "data" / "synthetic" / "frame_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_VIEW = "C10095_rgb"
TARGET_VIEW = "HMC_21176875_mono10bit"
FEATURE_DIM = 2048
NUM_VERBS = 24

# Subset config: only 20 train sequences. They MUST have both views available
# (we keep an exo + ego pair per sequence so all downstream tests work).
NUM_TEST_SEQUENCES = 20
SEED = 42

DOMAIN_SHIFT_NORM = 5.0
CLASS_NOISE_STD = 1.5


def main() -> None:
    rng = np.random.default_rng(SEED)

    class_means = rng.standard_normal((NUM_VERBS, FEATURE_DIM)).astype(np.float32) * 2.0
    d_vec = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    d_vec = (d_vec / np.linalg.norm(d_vec)) * DOMAIN_SHIFT_NORM

    print("Synthetic frame-level features (SUBSET for unit tests)")
    print(f"  classes={NUM_VERBS}, feature_dim={FEATURE_DIM}, dtype=float16")
    print(f"  test sequences: {NUM_TEST_SEQUENCES}")
    print(f"  output dir: {OUT_DIR}")

    # Pick sequences from train that have BOTH views
    train = pd.read_csv(ANN_DIR / "train.csv")
    parts = train["video"].str.split("/", n=1, expand=True)
    train["sequence_id"] = parts[0]
    train["view"] = parts[1].str.replace(".mp4", "", regex=False)
    train_2v = train[train["view"].isin([SOURCE_VIEW, TARGET_VIEW])]

    # Sequences with BOTH views
    seqs_per_view = (
        train_2v.groupby("sequence_id")["view"].nunique()
    )
    paired_sequences = sorted(seqs_per_view[seqs_per_view == 2].index)
    rng.shuffle(paired_sequences)
    chosen = paired_sequences[:NUM_TEST_SEQUENCES]
    print(f"\n  Total paired sequences available: {len(paired_sequences)}")
    print(f"  Chosen subset: {len(chosen)} sequences -> 2 .npy files each\n")

    subset = train_2v[train_2v["sequence_id"].isin(chosen)]
    groups = subset.groupby(["sequence_id", "view"], sort=False)

    total_size_mb = 0.0
    for (seq, view), df in tqdm(groups, total=groups.ngroups, desc="generating"):
        out_path = OUT_DIR / f"{seq}__{view}.npy"
        if out_path.exists():
            continue

        max_frame = int(df["end_frame"].max()) + 1
        feats = (
            rng.standard_normal((max_frame, FEATURE_DIM)).astype(np.float32)
            * CLASS_NOISE_STD
        )

        bias = d_vec if view == SOURCE_VIEW else -d_vec

        for _, row in df.iterrows():
            sf, ef = int(row["start_frame"]), int(row["end_frame"]) + 1
            verb = int(row["verb_id"])
            n = ef - sf
            seg = (
                class_means[verb][None, :]
                + rng.standard_normal((n, FEATURE_DIM)).astype(np.float32)
                * CLASS_NOISE_STD
                + bias[None, :]
            )
            feats[sf:ef] = seg

        # half-precision: 50% smaller, no measurable accuracy loss for our use
        np.save(out_path, feats.astype(np.float16))
        total_size_mb += out_path.stat().st_size / 1e6

    print(f"\nDone. {groups.ngroups} files saved, total ~{total_size_mb:.0f} MB")
    print(f"Files in: {OUT_DIR}")


if __name__ == "__main__":
    main()