"""Extract per-frame visual features from Charades-Ego videos.

For each .mp4 file in the input directory:
    1. Decode video frames at TARGET_FPS (default 5).
    2. Resize 256 short-side + center-crop 224 (ImageNet standard).
    3. Batch-transfer to GPU, normalize with ImageNet stats.
    4. Forward through ResNet-50 (pre-trained on ImageNet) up to the
       2048-D average-pool layer.
    5. Save as a single .npy file: (N_sampled_frames, 2048) float16.

Output: data/processed/charades-ego/frame_features/<video_id>.npy
        plus a manifest mapping video_id -> sampled timestamps (sec).

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

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "raw" / "charades-ego" / "CharadesEgo_v1_480"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "charades-ego" / "frame_features"


# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_feature_extractor(device: torch.device) -> nn.Module:
    """ResNet-50 ImageNet, output the 2048-D pre-fc embedding."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()  # output (B, 2048)
    model.eval()
    return model.to(device)


def get_video_frames(video_path: Path, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode the video at target_fps. Returns (frames, timestamps).

    frames: (N, H, W, 3) uint8, RGB (already short-side resized to 256 and center-cropped to 224)
    timestamps: (N,) float, seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps < 1:
        src_fps = 30.0
    step = max(int(round(src_fps / target_fps)), 1)

    crops = []
    timestamps = []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # Resize so the short side is 256 (matches torchvision Resize(256))
            if h < w:
                new_h, new_w = 256, int(round(w * 256 / h))
            else:
                new_h, new_w = int(round(h * 256 / w)), 256
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Center crop 224x224
            y0 = (new_h - 224) // 2
            x0 = (new_w - 224) // 2
            cropped = resized[y0:y0 + 224, x0:x0 + 224]
            crops.append(cropped)
            timestamps.append(idx / src_fps)
        idx += 1
    cap.release()

    if not crops:
        return np.zeros((0, 224, 224, 3), dtype=np.uint8), np.zeros((0,), dtype=np.float32)

    return np.stack(crops, axis=0), np.asarray(timestamps, dtype=np.float32)


@torch.no_grad()
def extract_for_video(
    model: nn.Module,
    video_path: Path,
    target_fps: float,
    batch_size: int,
    device: torch.device,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode + preprocess + forward, no DataLoader overhead."""
    frames, ts = get_video_frames(video_path, target_fps)
    if len(frames) == 0:
        return np.zeros((0, 2048), dtype=np.float16), ts

    feats_list = []
    n = len(frames)
    for i in range(0, n, batch_size):
        batch_np = frames[i:i + batch_size]  # (B, 224, 224, 3) uint8
        t = torch.from_numpy(batch_np).to(device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).float().div_(255.0)  # (B, 3, 224, 224)
        t.sub_(mean_gpu).div_(std_gpu)
        out = model(t).cpu().numpy().astype(np.float16)
        feats_list.append(out)

    feats = np.concatenate(feats_list, axis=0)
    return feats, ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Kept for CLI compatibility; ignored (no DataLoader).")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Video dir:  {args.video_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Batch size: {args.batch_size}")

    if not args.video_dir.exists():
        raise FileNotFoundError(f"Video dir not found: {args.video_dir}")

    videos = sorted(args.video_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos\n")

    model = build_feature_extractor(device)
    mean_gpu = torch.from_numpy(_IMAGENET_MEAN).to(device).view(1, 3, 1, 1)
    std_gpu = torch.from_numpy(_IMAGENET_STD).to(device).view(1, 3, 1, 1)

    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    n_done = 0
    n_skipped = 0
    pbar = tqdm(videos, desc="extracting")
    for video_path in pbar:
        video_id = video_path.stem
        out_path = args.output_dir / f"{video_id}.npy"
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        feats, ts = extract_for_video(
            model, video_path, args.target_fps,
            batch_size=args.batch_size, device=device,
            mean_gpu=mean_gpu, std_gpu=std_gpu,
        )
        np.save(out_path, feats)
        manifest[video_id] = {
            "n_frames": int(feats.shape[0]),
            "timestamps_sec": ts.tolist(),
            "target_fps": args.target_fps,
        }
        n_done += 1

        # update progress bar with running stats
        if n_done % 50 == 0:
            pbar.set_postfix({"done": n_done, "skipped": n_skipped})

        # periodically dump the manifest in case of interruption
        if n_done % 200 == 0:
            manifest_path.write_text(json.dumps(manifest))

    manifest_path.write_text(json.dumps(manifest))
    print(f"\nDone. {n_done} new, {n_skipped} skipped. Manifest at {manifest_path}")


if __name__ == "__main__":
    main()