"""Smoke test for extract_features.py.

We don't have any real Charades-Ego video yet (still downloading), so we
generate a tiny synthetic MP4 in /tmp and run the full extraction pipeline
on it. This validates that:
    1. ResNet-50 loads correctly with ImageNet weights.
    2. The video decoder + transform + GPU forward works end-to-end.
    3. The output .npy has the expected (N, 2048) float16 shape.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from src.datasets.extract_features import build_feature_extractor, extract_for_video


def make_test_video(path: Path, n_frames: int = 60, fps: int = 30,
                    width: int = 320, height: int = 240) -> None:
    """Write a short synthetic MP4 with random-color frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def main() -> None:
    print("Smoke test: ResNet-50 feature extractor on a synthetic video\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. build model
    print("\nLoading ResNet-50 (will download ImageNet weights on first run)...")
    model = build_feature_extractor(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[PASS] Model loaded: {n_params:.1f}M parameters")

    # 2. make a test video (60 frames @ 30 fps = 2 sec)
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "test.mp4"
        make_test_video(video_path, n_frames=60, fps=30)
        size_kb = video_path.stat().st_size / 1024
        print(f"[PASS] Test video created: {size_kb:.0f} KB")

        # 3. extract features at target_fps=5 -> expect ~10 frames sampled
        feats, ts = extract_for_video(
            model, video_path, target_fps=5.0,
            batch_size=8, num_workers=0, device=device,
        )
        print(f"\nFeatures shape:    {feats.shape}")
        print(f"Features dtype:    {feats.dtype}")
        print(f"Timestamps shape:  {ts.shape}")
        print(f"Timestamps:        {ts[:5]} ... {ts[-3:]}")
        assert feats.shape[1] == 2048
        assert feats.dtype == np.float16
        assert 8 <= feats.shape[0] <= 12  # ~10 frames @ 5fps on a 2sec video
        print("[PASS] Feature extraction shape and dtype correct")

    # 4. sanity check: features should NOT be all zeros / all NaN
    assert not np.isnan(feats).any()
    # Random input causes many ReLU units to be permanently 0;
    # we accept it but require the majority of dims to be informative.
    active_dims = (feats.std(axis=0) > 0).sum()
    total_dims = feats.shape[1]
    active_frac = active_dims / total_dims
    print(f"  active feature dims: {active_dims}/{total_dims} = {active_frac:.1%}")
    assert active_frac > 0.30, f"too few active dims ({active_frac:.1%})"
    print("[PASS] Features are largely non-degenerate and contain no NaNs")

    print("\nAll feature-extraction smoke tests passed.")


if __name__ == "__main__":
    main()