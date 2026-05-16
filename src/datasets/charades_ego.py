"""PyTorch Dataset and parser utilities for Charades-Ego segment-level features.

Charades-Ego (Sigurdsson et al., CVPR 2018) is a paired ego/exo video
dataset: 7860 videos (3935 ego + 3925 exo), 157 action classes, multi-label
temporal annotations.

For our DA setting we treat each (action_class, start_sec, end_sec) interval
as a single labelled segment (Strategy B - single-label). When two classes
overlap in time, they become two separate segments, each with a single
class label. This converts Charades-Ego from multi-label to single-label,
matching the structure of our existing pipeline 1:1.

Domain mapping:
    source (exocentric / third-person) <- CharadesEgo_v1_*_only3rd.csv
    target (egocentric / first-person) <- CharadesEgo_v1_*_only1st.csv

Splits:
    train -> 85% of train_only*.csv (per-video split, seed=42)
    val   -> 15% of train_only*.csv
    test  -> CharadesEgo_v1_test_only*.csv

This file provides:
    - parse_charades_actions: convert the 'c001 1.0 5.0;...' string to a
      list of (class_id, start_sec, end_sec) tuples.
    - load_charades_split: read one of the official CSVs and expand it to
      a per-segment DataFrame (one row per segment).
    - make_charades_splits: convenience function that loads all 4 splits
      (train_source/target, val_source/target, test_source/target) and
      returns a dict of DataFrames ready for feature extraction.
    - CharadesEgoSegmentDataset: PyTorch Dataset over .npz segment files,
      identical interface to Assembly101SegmentDataset.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------- constants ------------------------------------

NUM_CLASSES = 157
ACTION_RE = re.compile(r'\s*c(\d+)\s+([\d.]+)\s+([\d.]+)\s*')


# ---------------------------- parsing --------------------------------------

def parse_charades_actions(actions_str: str | float) -> list[tuple[int, float, float]]:
    """Parse a Charades-Ego 'actions' field into a list of segments.

    Args:
        actions_str: e.g. "c156 3.90 12.00;c061 8.20 12.50".
            May be NaN or empty (video with no annotated actions).

    Returns:
        List of (class_id, start_sec, end_sec) tuples. Empty list if no actions.
    """
    if pd.isna(actions_str) or actions_str == '':
        return []
    segs = []
    for piece in str(actions_str).split(';'):
        m = ACTION_RE.match(piece)
        if m:
            segs.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return segs


def load_charades_split(csv_path: Path) -> pd.DataFrame:
    """Load one of the official Charades-Ego CSVs and explode it to one
    row per (video, segment).

    Returns a DataFrame with columns:
        id, segment_idx_in_video, class_id, start_sec, end_sec, length, charades_video
    """
    raw = pd.read_csv(csv_path)
    rows = []
    for _, r in raw.iterrows():
        segs = parse_charades_actions(r['actions'])
        for k, (cid, s, e) in enumerate(segs):
            rows.append({
                'id': r['id'],
                'segment_idx_in_video': k,
                'class_id': cid,
                'start_sec': s,
                'end_sec': e,
                'length': r['length'],
                'charades_video': r.get('charades_video', None),
            })
    df = pd.DataFrame(rows)
    df['segment_id'] = range(len(df))  # global integer id, used to track samples
    return df


def make_charades_splits(
    base_dir: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build the 6 split DataFrames the rest of the pipeline expects:
        train_source, train_target, val_source, val_target, test_source, test_target.

    Train and val are obtained by splitting the official train CSVs at the
    *video level* (not segment level) so segments of the same video never
    cross the split boundary. The split is deterministic in `seed`.
    """
    base = Path(base_dir)
    out: dict[str, pd.DataFrame] = {}

    csvs = {
        ('train', 'source'): base / 'CharadesEgo_v1_train_only3rd.csv',
        ('train', 'target'): base / 'CharadesEgo_v1_train_only1st.csv',
        ('test', 'source'):  base / 'CharadesEgo_v1_test_only3rd.csv',
        ('test', 'target'):  base / 'CharadesEgo_v1_test_only1st.csv',
    }

    # --- train and val (carved from train via per-video split) ---
    rng = np.random.default_rng(seed)
    for domain in ('source', 'target'):
        df = load_charades_split(csvs[('train', domain)])
        # determine video-level split (consistent across domains via the same seed
        # would not be appropriate because the video sets are different per domain;
        # we just split each domain independently with the same seed for reproducibility)
        unique_videos = df['id'].unique()
        rng_local = np.random.default_rng(seed if domain == 'source' else seed + 1)
        shuffled = rng_local.permutation(unique_videos)
        n_val = int(len(shuffled) * val_fraction)
        val_videos = set(shuffled[:n_val])
        out[f'train_{domain}'] = df[~df['id'].isin(val_videos)].reset_index(drop=True)
        out[f'val_{domain}']   = df[df['id'].isin(val_videos)].reset_index(drop=True)

    # --- test (taken verbatim from official test_only*) ---
    for domain in ('source', 'target'):
        out[f'test_{domain}'] = load_charades_split(csvs[('test', domain)])

    return out


# ---------------------------- Dataset --------------------------------------

class CharadesEgoSegmentDataset(Dataset):
    """In-memory dataset of pre-extracted Charades-Ego segment-level features.

    Identical interface to Assembly101SegmentDataset; the only structural
    difference is num_classes==157 instead of 24, but it's inferred from
    the labels at load time.
    """

    def __init__(self, npz_path: str | Path, return_segment_id: bool = False) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Segment-feature file not found: {npz_path}")

        with np.load(npz_path) as data:
            self.features = torch.from_numpy(data['features'].astype(np.float32))
            self.labels = torch.from_numpy(data['labels'].astype(np.int64))
            self.segment_ids = torch.from_numpy(data['segment_ids'].astype(np.int64))

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
        return NUM_CLASSES  # fissato a 157 per Charades-Ego (anche se la split potrebbe non coprirle tutte)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def class_counts(self) -> torch.Tensor:
        """Return a (NUM_CLASSES,) tensor with the number of samples per class."""
        return torch.bincount(self.labels, minlength=NUM_CLASSES)