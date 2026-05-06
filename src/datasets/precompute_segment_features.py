"""Step 2 of the data pipeline: aggregate frame-level features into
segment-level features.

Pipeline overview (Track 7 — Domain Adaptation Exo->Ego):
    1. lmdb_to_npy.py  -> .npy frame-level features per (sequence, view)
                          [executed only on the cluster, on the real LMDB]
    2. THIS SCRIPT     -> .npz segment-level features per (split, domain)
                          [reads the .npy from step 1 and the official
                           Assembly101 fine-grained-annotations CSV files]
    3. assembly101.py  -> PyTorch Dataset reading the .npz
    4. pair_loader.py  -> Combined source+target loader for DANN/MMD

For each segment in the official train/validation/test CSVs, restricted to
SOURCE_VIEW (exocentric) or TARGET_VIEW (egocentric):
    1. Open the corresponding .npy.
    2. Slice rows [start_frame : end_frame + 1].
    3. Mean-pool over the temporal axis -> single 2048-D vector.
    4. Append to the per-(split, domain) collection.

Output:
    data/processed/segment_features/
        train_source.npz
        train_target.npz
        val_source.npz
        val_target.npz
        test_source.npz
        test_target.npz

Each .npz contains:
    features    : (N, FEATURE_DIM) float32
    labels      : (N,) int64           # verb_id
    segment_ids : (N,) int64           # original CSV row id (for traceability)

CLI:
    # Default: read from the SYNTHETIC subset (only sequences with .npy present)
    python -m src.datasets.precompute_segment_features

    # On the cluster (real data):
    python -m src.datasets.precompute_segment_features \
        --frame-dir /home/$USER/data/processed/frame_features \
        --out-dir   /home/$USER/data/processed/segment_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------- defaults --------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANN_DIR = REPO_ROOT / "data" / "raw" / "assembly101-annotations" / "fine-grained-annotations"
DEFAULT_FRAME_DIR = REPO_ROOT / "data" / "synthetic" / "frame_features"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "segment_features"

SOURCE_VIEW = "C10095_rgb"
TARGET_VIEW = "HMC_21176875_mono10bit"
FEATURE_DIM = 2048


# ---------------------------- core ------------------------------------------

def aggregate_split(
    df: pd.DataFrame,
    frame_dir: Path,
    domain_name: str,
    expected_view: str,
) -> dict[str, np.ndarray] | None:
    """Aggregate one (split, domain) subset into segment-level features.

    Returns dict with keys 'features', 'labels', 'segment_ids', or None
    if no segments are available (e.g. on the synthetic subset where most
    sequences are missing).
    """
    sub = df[df["view"] == expected_view].copy()
    if len(sub) == 0:
        return None

    # cache last-loaded .npy to avoid re-reading the same file across rows
    cache_key: str | None = None
    cache_arr: np.ndarray | None = None

    feats_list: list[np.ndarray] = []
    labels_list: list[int] = []
    ids_list: list[int] = []
    skipped_missing_file = 0
    skipped_out_of_range = 0

    for _, row in tqdm(
        sub.iterrows(), total=len(sub), desc=f"  {domain_name}", leave=False
    ):
        seq = row["sequence_id"]
        view = row["view"]
        key = f"{seq}__{view}"
        npy_path = frame_dir / f"{key}.npy"

        if not npy_path.exists():
            skipped_missing_file += 1
            continue

        if key != cache_key:
            cache_arr = np.load(npy_path).astype(np.float32, copy=False)
            cache_key = key
        assert cache_arr is not None  # for type checker

        sf, ef = int(row["start_frame"]), int(row["end_frame"]) + 1
        if ef > cache_arr.shape[0] or sf < 0:
            skipped_out_of_range += 1
            continue

        seg = cache_arr[sf:ef]
        if seg.shape[0] == 0:
            skipped_out_of_range += 1
            continue

        feats_list.append(seg.mean(axis=0).astype(np.float32))
        labels_list.append(int(row["verb_id"]))
        ids_list.append(int(row["id"]))

    if not feats_list:
        print(f"    [WARN] {domain_name}: no segments produced")
        return None

    print(
        f"    {domain_name}: {len(feats_list):,} segments aggregated"
        f" | skipped: {skipped_missing_file} (missing .npy),"
        f" {skipped_out_of_range} (out of range)"
    )

    return {
        "features": np.stack(feats_list, axis=0),
        "labels": np.asarray(labels_list, dtype=np.int64),
        "segment_ids": np.asarray(ids_list, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR,
                    help="Directory with the fine-grained train/val/test CSV files.")
    ap.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR,
                    help="Directory with per-(sequence,view) .npy frame features.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Where to write the segment-level .npz files.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Segment feature pre-computation")
    print(f"  annotations: {args.ann_dir}")
    print(f"  frame .npy:  {args.frame_dir}")
    print(f"  output:      {args.out_dir}")
    print(f"  source view: {SOURCE_VIEW}")
    print(f"  target view: {TARGET_VIEW}\n")

    splits = ["train", "validation", "test"]
    for split in splits:
        print(f"== {split} ==")
        csv_path = args.ann_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)
        parts = df["video"].str.split("/", n=1, expand=True)
        df["sequence_id"] = parts[0]
        df["view"] = parts[1].str.replace(".mp4", "", regex=False)

        # restrict to our 2 chosen views
        df = df[df["view"].isin([SOURCE_VIEW, TARGET_VIEW])].copy()

        for domain_name, view in [("source", SOURCE_VIEW), ("target", TARGET_VIEW)]:
            agg = aggregate_split(df, args.frame_dir, domain_name, view)
            if agg is None:
                continue
            out_name = f"{'val' if split == 'validation' else split}_{domain_name}.npz"
            out_path = args.out_dir / out_name
            np.savez_compressed(
                out_path,
                features=agg["features"],
                labels=agg["labels"],
                segment_ids=agg["segment_ids"],
            )
            mb = out_path.stat().st_size / 1e6
            print(f"    -> {out_name}  ({agg['features'].shape}, {mb:.1f} MB)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()