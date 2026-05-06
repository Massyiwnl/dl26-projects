"""Generate synthetic SEGMENT-LEVEL features at full scale (~130k segments
covering all official Assembly101 fine-grained train/val/test rows for the
2 chosen views).

Why a separate script (vs precompute_segment_features.py)?
  - `precompute_segment_features.py` is the REAL pipeline: it reads frame-level
    .npy files (produced by 1_lmdb_to_npy.py from the official LMDB) and does
    mean-pooling per segment.
  - This script SHORT-CIRCUITS that pipeline for local development: it
    generates one synthetic segment-level vector per CSV row directly,
    using the same domain-shifted Gaussian model as the unit-test script.

Output: same as precompute_segment_features.py
    data/processed/segment_features/
        train_source.npz, train_target.npz
        val_source.npz,   val_target.npz
        test_source.npz,  test_target.npz

OVERWRITES any existing .npz in the same directory.

Run from repo root:
    python -m src.datasets.make_synthetic_segment_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = REPO_ROOT / "data" / "raw" / "assembly101-annotations" / "fine-grained-annotations"
OUT_DIR = REPO_ROOT / "data" / "processed" / "segment_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_VIEW = "C10095_rgb"
TARGET_VIEW = "HMC_21176875_mono10bit"
FEATURE_DIM = 2048
NUM_VERBS = 24
SEED = 42

# Same shift/noise as in make_synthetic_frame_features.py for consistency.
DOMAIN_SHIFT_NORM = 5.0
CLASS_NOISE_STD = 1.5


def generate_features(
    df: pd.DataFrame,
    class_means: np.ndarray,
    domain_bias: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (features, labels, segment_ids) for a (split, domain) subset."""
    n = len(df)
    feats = np.empty((n, FEATURE_DIM), dtype=np.float32)
    labels = df["verb_id"].to_numpy(dtype=np.int64)
    seg_ids = df["id"].to_numpy(dtype=np.int64)

    # vectorised generation: for each row, sample mean[verb] + N(0,sigma) + bias
    # We do it in a loop with a single rng call per chunk to keep it readable
    # and memory-friendly.
    chunk = 8192
    for start in tqdm(range(0, n, chunk), desc="  generating", leave=False):
        end = min(start + chunk, n)
        cls = labels[start:end]
        mean = class_means[cls]                                          # (B, D)
        noise = rng.standard_normal((end - start, FEATURE_DIM)).astype(np.float32) * CLASS_NOISE_STD
        feats[start:end] = mean + noise + domain_bias[None, :]

    return feats, labels, seg_ids


def main() -> None:
    rng = np.random.default_rng(SEED)

    class_means = rng.standard_normal((NUM_VERBS, FEATURE_DIM)).astype(np.float32) * 2.0
    d_vec = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    d_vec = (d_vec / np.linalg.norm(d_vec)) * DOMAIN_SHIFT_NORM

    print("Synthetic SEGMENT-level features (full scale, all CSV rows)")
    print(f"  classes={NUM_VERBS}, feature_dim={FEATURE_DIM}, dtype=float32")
    print(f"  domain_shift_norm={DOMAIN_SHIFT_NORM}, class_noise_std={CLASS_NOISE_STD}")
    print(f"  output dir: {OUT_DIR}\n")

    splits = [("train", "train"), ("val", "validation"), ("test", "test")]
    for out_split, csv_split in splits:
        print(f"== {out_split} ==")
        df = pd.read_csv(ANN_DIR / f"{csv_split}.csv")
        parts = df["video"].str.split("/", n=1, expand=True)
        df["sequence_id"] = parts[0]
        df["view"] = parts[1].str.replace(".mp4", "", regex=False)

        for domain_name, view in [("source", SOURCE_VIEW), ("target", TARGET_VIEW)]:
            sub = df[df["view"] == view].copy()
            if len(sub) == 0:
                continue
            bias = d_vec if view == SOURCE_VIEW else -d_vec
            feats, labels, seg_ids = generate_features(sub, class_means, bias, rng)

            out_path = OUT_DIR / f"{out_split}_{domain_name}.npz"
            np.savez_compressed(
                out_path,
                features=feats,
                labels=labels,
                segment_ids=seg_ids,
            )
            mb = out_path.stat().st_size / 1e6
            print(f"  -> {out_path.name}  ({feats.shape}, {mb:.1f} MB)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()