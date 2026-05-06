"""Generate synthetic SEGMENT-LEVEL features at full scale.

This is the CHALLENGING variant: it uses a non-linear per-domain transform
(QR-orthogonal rotation + element-wise tanh on a low-rank subspace + bias)
combined with low signal-to-noise ratio. The result is a non-trivial DA
problem in which:
    - in-domain accuracy is reachable but not perfect (B2 oracle ~75-90%)
    - cross-domain accuracy crashes without DA (B1 ~30-50%)
    - DANN/MMD have measurable room to close the gap

Output: same as precompute_segment_features.py
    data/processed/segment_features/
        train_source.npz, train_target.npz
        val_source.npz,   val_target.npz
        test_source.npz,  test_target.npz

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

# ---- Signal calibration (HARDER than before) -----------------------------
# Class signal lives only in the first SIGNAL_DIM coordinates.
SIGNAL_DIM = 200
CLASS_MEAN_SCALE = 1.5     # strong enough for in-domain learning
CLASS_NOISE_STD = 2.0      # moderate noise, in-domain ~90% achievable

# ---- Per-domain transform (NON-LINEAR, hard to undo) ---------------------
SHIFT_RANK = 128            # very small subspace shifted
NONLIN_GAIN = 12.0          # tanh now acts as ~identity in the bulk
BIAS_NORM = 1.0             # very small bias
DOMAIN_SCALE_DIFF = 0.0     # no per-domain rescaling


def random_rotation(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Return a (dim, dim) orthogonal matrix sampled uniformly from O(dim)."""
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q


def make_per_domain_transform(rng: np.random.Generator):
    """Build (R_src, scale_src, b_src, R_tgt, scale_tgt, b_tgt).

    Each domain applies (in order):
        x' = R_d @ x   on the first SHIFT_RANK coordinates
        x' = x' * scale_d        (per-domain element-wise scale)
        x' = NONLIN_GAIN * tanh(x' / NONLIN_GAIN)  on the first SHIFT_RANK coords
        x' = x' + b_d
    The tanh squashing is the non-linear ingredient that prevents a single
    affine encoder from undoing the shift.
    """
    R_src_block = random_rotation(rng, SHIFT_RANK)
    R_tgt_block = random_rotation(rng, SHIFT_RANK)
    R_src = np.eye(FEATURE_DIM, dtype=np.float32)
    R_tgt = np.eye(FEATURE_DIM, dtype=np.float32)
    R_src[:SHIFT_RANK, :SHIFT_RANK] = R_src_block
    R_tgt[:SHIFT_RANK, :SHIFT_RANK] = R_tgt_block

    scale_src = 1.0 + DOMAIN_SCALE_DIFF / 2.0
    scale_tgt = 1.0 - DOMAIN_SCALE_DIFF / 2.0

    b_dir = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    b_dir = b_dir / np.linalg.norm(b_dir)
    b_src = b_dir * BIAS_NORM
    b_tgt = -b_dir * BIAS_NORM

    return R_src, scale_src, b_src, R_tgt, scale_tgt, b_tgt


def apply_domain(
    x: np.ndarray, R: np.ndarray, scale: float, bias: np.ndarray
) -> np.ndarray:
    # rotation
    x = x @ R.T
    # per-domain scale (element-wise)
    x = x * scale
    # non-linearity on the first SHIFT_RANK coordinates
    head = np.tanh(x[:, :SHIFT_RANK] / NONLIN_GAIN) * NONLIN_GAIN
    x = np.concatenate([head, x[:, SHIFT_RANK:]], axis=1)
    # translation
    x = x + bias[None, :]
    return x


def generate_features(
    df: pd.DataFrame,
    class_means: np.ndarray,
    R: np.ndarray,
    scale: float,
    bias: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    feats = np.empty((n, FEATURE_DIM), dtype=np.float32)
    labels = df["verb_id"].to_numpy(dtype=np.int64)
    seg_ids = df["id"].to_numpy(dtype=np.int64)

    chunk = 8192
    for start in tqdm(range(0, n, chunk), desc="  generating", leave=False):
        end = min(start + chunk, n)
        cls = labels[start:end]
        # weak class signal in first SIGNAL_DIM dims, zeros elsewhere, plus noise
        x = np.zeros((end - start, FEATURE_DIM), dtype=np.float32)
        x[:, :SIGNAL_DIM] = class_means[cls]
        x = x + rng.standard_normal(x.shape).astype(np.float32) * CLASS_NOISE_STD
        # apply domain transform
        x = apply_domain(x, R, scale, bias)
        feats[start:end] = x

    return feats, labels, seg_ids


def main() -> None:
    rng = np.random.default_rng(SEED)

    # class means live in R^SIGNAL_DIM (zeros in the rest)
    class_means_signal = rng.standard_normal((NUM_VERBS, SIGNAL_DIM)).astype(np.float32) * CLASS_MEAN_SCALE
    R_src, scale_src, b_src, R_tgt, scale_tgt, b_tgt = make_per_domain_transform(rng)

    print("Synthetic SEGMENT-level features (CHALLENGING SHIFT)")
    print(f"  classes={NUM_VERBS}, feature_dim={FEATURE_DIM}, dtype=float32")
    print(f"  signal_dim={SIGNAL_DIM}, class_mean_scale={CLASS_MEAN_SCALE}, class_noise_std={CLASS_NOISE_STD}")
    print(f"  shift_rank={SHIFT_RANK}, nonlin_gain={NONLIN_GAIN}, bias_norm={BIAS_NORM}, domain_scale_diff={DOMAIN_SCALE_DIFF}")
    print(f"  output dir: {OUT_DIR}\n")

    splits = [("train", "train"), ("val", "validation"), ("test", "test")]
    for out_split, csv_split in splits:
        print(f"== {out_split} ==")
        df = pd.read_csv(ANN_DIR / f"{csv_split}.csv")
        parts = df["video"].str.split("/", n=1, expand=True)
        df["sequence_id"] = parts[0]
        df["view"] = parts[1].str.replace(".mp4", "", regex=False)

        for domain_name, view, R, scale, bias in [
            ("source", SOURCE_VIEW, R_src, scale_src, b_src),
            ("target", TARGET_VIEW, R_tgt, scale_tgt, b_tgt),
        ]:
            sub = df[df["view"] == view].copy()
            if len(sub) == 0:
                continue
            feats, labels, seg_ids = generate_features(sub, class_means_signal, R, scale, bias, rng)

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