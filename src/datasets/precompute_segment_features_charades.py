"""Step 2 of the Charades-Ego data pipeline: aggregate frame-level
ResNet-50 features into segment-level features.

Pipeline overview:
    1. extract_features.py -> .npy per video (N_sampled_frames, 2048) float16
                              + manifest.json with timestamps_sec[i] for each frame i
    2. THIS SCRIPT         -> .npz per (split, domain) with segment-level features

For each segment (class_id, start_sec, end_sec) of each video, we:
    1. Open the corresponding video .npy and the manifest entry.
    2. Find all sampled frames whose timestamp falls in [start_sec, end_sec].
    3. Mean-pool the corresponding rows -> single 2048-D vector.

Output:
    data/processed/charades-ego/segment_features/
        train_source.npz, train_target.npz
        val_source.npz,   val_target.npz
        test_source.npz,  test_target.npz

CLI:
    python -m src.datasets.precompute_segment_features_charades
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets.charades_ego import make_charades_splits, NUM_CLASSES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHARADES_DIR = REPO_ROOT / "data" / "raw" / "charades-ego" / "CharadesEgo"
DEFAULT_FRAME_DIR = REPO_ROOT / "data" / "processed" / "charades-ego" / "frame_features"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "charades-ego" / "segment_features"


def aggregate_split(
    df: pd.DataFrame,
    frame_dir: Path,
    manifest: dict,
    domain_name: str,
) -> dict[str, np.ndarray] | None:
    """Aggregate one (split, domain) subset into segment-level features.

    Returns dict with keys 'features', 'labels', 'segment_ids', or None
    if no segments are available.
    """
    if len(df) == 0:
        return None

    cache_id: str | None = None
    cache_feats: np.ndarray | None = None
    cache_ts: np.ndarray | None = None

    feats_list: list[np.ndarray] = []
    labels_list: list[int] = []
    ids_list: list[int] = []
    skipped_missing_npy = 0
    skipped_missing_manifest = 0
    skipped_empty_range = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {domain_name}", leave=False):
        video_id = row['id']
        npy_path = frame_dir / f"{video_id}.npy"

        if not npy_path.exists():
            skipped_missing_npy += 1
            continue
        if video_id not in manifest:
            skipped_missing_manifest += 1
            continue

        if video_id != cache_id:
            cache_feats = np.load(npy_path).astype(np.float32, copy=False)
            cache_ts = np.asarray(manifest[video_id]['timestamps_sec'], dtype=np.float32)
            cache_id = video_id
        assert cache_feats is not None and cache_ts is not None

        sf, ef = float(row['start_sec']), float(row['end_sec'])
        mask = (cache_ts >= sf) & (cache_ts <= ef)
        if not mask.any():
            # fallback: take the single nearest frame to the segment midpoint
            mid = (sf + ef) / 2.0
            idx = int(np.argmin(np.abs(cache_ts - mid)))
            seg_feats = cache_feats[idx:idx + 1]
        else:
            seg_feats = cache_feats[mask]

        if seg_feats.shape[0] == 0:
            skipped_empty_range += 1
            continue

        feats_list.append(seg_feats.mean(axis=0).astype(np.float32))
        labels_list.append(int(row['class_id']))
        ids_list.append(int(row['segment_id']))

    if not feats_list:
        print(f"    [WARN] {domain_name}: no segments produced")
        return None

    print(
        f"    {domain_name}: {len(feats_list):>5} segments aggregated"
        f" | skipped: {skipped_missing_npy} (no .npy), {skipped_missing_manifest} (no manifest), {skipped_empty_range} (empty range)"
    )

    return {
        "features": np.stack(feats_list, axis=0),
        "labels": np.asarray(labels_list, dtype=np.int64),
        "segment_ids": np.asarray(ids_list, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--charades-dir", type=Path, default=DEFAULT_CHARADES_DIR,
                    help="Directory with the Charades-Ego CSV files.")
    ap.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR,
                    help="Directory with per-video .npy frame features + manifest.json.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Where to write the segment-level .npz files.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Charades-Ego segment feature pre-computation")
    print(f"  CSVs:       {args.charades_dir}")
    print(f"  frame .npy: {args.frame_dir}")
    print(f"  output:     {args.out_dir}\n")

    manifest_path = args.frame_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Did you run extract_features.py first?"
        )
    manifest = json.loads(manifest_path.read_text())
    print(f"  Manifest contains {len(manifest)} videos\n")

    splits = make_charades_splits(args.charades_dir, val_fraction=0.15, seed=42)
    print("=== Building segment-level features for each (split, domain) ===\n")

    for name, df in splits.items():
        print(f"  -- {name} --")
        agg = aggregate_split(df, args.frame_dir, manifest, name)
        if agg is None:
            continue
        out_path = args.out_dir / f"{name}.npz"
        np.savez_compressed(
            out_path,
            features=agg["features"],
            labels=agg["labels"],
            segment_ids=agg["segment_ids"],
        )
        mb = out_path.stat().st_size / 1e6
        print(f"    -> {out_path.name}  ({agg['features'].shape}, {mb:.1f} MB)\n")

    print("Done.")


if __name__ == "__main__":
    main()