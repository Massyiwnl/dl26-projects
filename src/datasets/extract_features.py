"""Extract per-frame visual features from Charades-Ego videos.

For each .mp4 file in the input directory:
    1. Decode video frames at TARGET_FPS (default 5).
    2. Apply ImageNet normalization.
    3. Pass through ResNet-50 (pre-trained on ImageNet) up to the 2048-D
       average-pool layer.
    4. Save as a single .npy file: (N_sampled_frames, 2048) float16.

Output: data/processed/charades-ego/frame_features/<video_id>.npy
        plus a manifest mapping video_id -> sampled timestamps (sec).

The downstream pipeline (precompute_segment_features.py) will then
mean-pool these frame features over each (class, start_sec, end_sec)
segment to produce the segment-level .npz files used by the trainer.

Run from repo root:
    python -m src.datasets.extract_features \
        --video-dir data/raw/charades-ego/CharadesEgo_v1_480 \
        --output-dir data/processed/charades-ego/frame_features \
        --target-fps 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "raw" / "charades-ego" / "CharadesEgo_v1_480"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "charades-ego" / "frame_features"


# ---------------------------- model ---------------------------------------

def build_feature_extractor(device: torch.device) -> nn.Module:
    """ResNet-50 ImageNet, output the 2048-D pre-fc embedding."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()  # output (B, 2048)
    model.eval()
    return model.to(device)


# ---------------------------- video reader --------------------------------

def get_video_frames(video_path: Path, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode the video at target_fps. Returns (frames, timestamps).

    frames: (N, H, W, 3) uint8, RGB
    timestamps: (N,) float, seconds
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps < 1:
        src_fps = 30.0
    step = max(int(round(src_fps / target_fps)), 1)

    frames = []
    timestamps = []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(idx / src_fps)
        idx += 1
    cap.release()

    if not frames:
        return np.zeros((0,), dtype=np.uint8), np.zeros((0,), dtype=np.float32)

    return np.stack(frames, axis=0), np.asarray(timestamps, dtype=np.float32)


# ---------------------------- dataset for batched inference ---------------

class _VideoDataset(Dataset):
    """Iterate over the frames of a single video. Used by DataLoader for
    batched ImageNet preprocessing + GPU transfer."""

    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __init__(self, frames: np.ndarray) -> None:
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._transform(self.frames[idx])


# ---------------------------- main ----------------------------------------

@torch.no_grad()
def extract_for_video(model: nn.Module, video_path: Path, target_fps: float,
                       batch_size: int, num_workers: int, device: torch.device
                       ) -> tuple[np.ndarray, np.ndarray]:
    frames, ts = get_video_frames(video_path, target_fps)
    if len(frames) == 0:
        return np.zeros((0, 2048), dtype=np.float16), ts

    ds = _VideoDataset(frames)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    feats_list = []
    for batch in dl:
        batch = batch.to(device, non_blocking=True)
        out = model(batch).cpu().numpy().astype(np.float16)
        feats_list.append(out)

    feats = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 2048), dtype=np.float16)
    return feats, ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--target-fps", type=float, default=5.0,
                    help="Sample one frame every 1/target_fps seconds.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip videos for which the .npy already exists.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Video dir:  {args.video_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target FPS: {args.target_fps}")

    if not args.video_dir.exists():
        raise FileNotFoundError(f"Video dir not found: {args.video_dir}")

    videos = sorted(args.video_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos\n")

    model = build_feature_extractor(device)

    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    n_done = 0
    n_skipped = 0
    for video_path in tqdm(videos, desc="extracting"):
        video_id = video_path.stem
        out_path = args.output_dir / f"{video_id}.npy"
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        feats, ts = extract_for_video(
            model, video_path, args.target_fps,
            batch_size=args.batch_size, num_workers=args.num_workers, device=device,
        )
        np.save(out_path, feats)
        manifest[video_id] = {
            "n_frames": int(feats.shape[0]),
            "timestamps_sec": ts.tolist(),
            "target_fps": args.target_fps,
        }
        n_done += 1

        # periodically dump the manifest in case of interruption
        if n_done % 100 == 0:
            manifest_path.write_text(json.dumps(manifest))

    manifest_path.write_text(json.dumps(manifest))
    print(f"\nDone. {n_done} new, {n_skipped} skipped. Manifest at {manifest_path}")


if __name__ == "__main__":
    main()