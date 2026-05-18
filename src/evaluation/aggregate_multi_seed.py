"""Aggregate multi-seed cluster runs and produce a mean ± std results table.

For each method in {b1, b2, dann, mmd}, finds checkpoints matching
'experiments/checkpoints/CLUSTER_{method}_seed*', re-evaluates each best.pt
on the target val split, then prints aggregated metrics + a markdown table.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from src.models.encoder import FeatureEncoder
from src.models.classifier import ActionClassifier
from src.datasets.charades_ego import CharadesEgoSegmentDataset
from src.evaluation.metrics import aggregate_metrics


def build_from_ckpt(ckpt_path, device, in_dim=2048, embed_dim=256, num_classes=157):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    cls_sd = {k[len("classifier."):]: v for k, v in sd.items() if k.startswith("classifier.")}
    cls_hidden = int(cls_sd["net.0.weight"].shape[0])
    encoder = FeatureEncoder(in_dim=in_dim, hidden_dims=(1024, 512),
                             embed_dim=embed_dim, dropout=0.3).to(device)
    classifier = ActionClassifier(embed_dim=embed_dim, hidden_dim=cls_hidden,
                                  num_classes=num_classes, dropout=0.1).to(device)
    encoder.load_state_dict(enc_sd)
    classifier.load_state_dict(cls_sd)
    encoder.eval()
    classifier.eval()
    return encoder, classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-npz",
                    default="data/processed/charades-ego/segment_features/val_target.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--methods", nargs="+", default=["b1", "b2", "dann", "mmd"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_ds = CharadesEgoSegmentDataset(args.val_npz)
    X_val = val_ds.features.to(device)
    y_val = val_ds.labels
    print(f"Val target: {len(val_ds):,} segments, {val_ds.num_classes} classes\n")

    method_names = {
        "b1":   "B1 — Source-only (exo->ego, zero-shot)",
        "dann": "DANN (lambda_max=0.5) — main",
        "mmd":  "MMD (lambda_mmd=1.0) — main",
        "b2":   "B2 — Target-only oracle (ego->ego, upper bound)",
    }

    results_by_method = {}
    for method in args.methods:
        pattern = f"CLUSTER_{method}_seed*"
        ckpt_dirs = sorted(Path(args.checkpoints_dir).glob(pattern))
        if not ckpt_dirs:
            print(f"[WARN] no checkpoints for method '{method}' (pattern: {pattern})")
            continue
        print(f"=== {method.upper()} — {len(ckpt_dirs)} seed(s) ===")
        results = []
        for d in ckpt_dirs:
            ckpt = d / "best.pt"
            if not ckpt.exists():
                print(f"  [WARN] best.pt missing in {d}")
                continue
            seed = d.name.split("_seed")[-1]
            enc, cls = build_from_ckpt(ckpt, device)
            with torch.no_grad():
                logits = cls(enc(X_val))
            m = aggregate_metrics(logits.cpu(), y_val,
                                  num_classes=val_ds.num_classes, is_logits=True)
            results.append({"seed": seed, **m})
            print(f"  seed={seed}: bal={m['balanced_accuracy']:.4f}  "
                  f"top1={m['top1']:.4f}  top5={m['top5']:.4f}  "
                  f"F1={m['macro_f1']:.4f}")
        results_by_method[method] = results
        print()

    print("=" * 80)
    print(" AGGREGATED RESULTS (mean ± std across seeds)")
    print("=" * 80)
    print(f"\n{'Method':<10} {'balanced':<18} {'top-1':<18} {'top-5':<18} {'macro-F1':<18}")
    print("-" * 80)
    for method in args.methods:
        rs = results_by_method.get(method, [])
        if not rs:
            print(f"{method:<10} (no data)")
            continue
        line = f"{method:<10} "
        for key in ["balanced_accuracy", "top1", "top5", "macro_f1"]:
            vals = [r[key] for r in rs]
            line += f"{np.mean(vals):.4f} ± {np.std(vals):.4f}    "
        print(line)

    print("\n\n--- Markdown table for REPORT ---\n")
    print("| Model | balanced acc | top-1 | top-5 | macro-F1 |")
    print("|---|---|---|---|---|")
    for method in ["b1", "dann", "mmd", "b2"]:  # canonical order in REPORT
        rs = results_by_method.get(method, [])
        if not rs:
            continue
        b   = (np.mean([r["balanced_accuracy"] for r in rs]),
               np.std([r["balanced_accuracy"] for r in rs]))
        t1  = (np.mean([r["top1"] for r in rs]),
               np.std([r["top1"] for r in rs]))
        t5  = (np.mean([r["top5"] for r in rs]),
               np.std([r["top5"] for r in rs]))
        f1  = (np.mean([r["macro_f1"] for r in rs]),
               np.std([r["macro_f1"] for r in rs]))
        name = method_names.get(method, method)
        bold = "**" if method in {"dann", "mmd"} else ""
        print(f"| {bold}{name}{bold} | "
              f"{bold}{b[0]:.3f} ± {b[1]:.3f}{bold} | "
              f"{bold}{t1[0]:.3f} ± {t1[1]:.3f}{bold} | "
              f"{bold}{t5[0]:.3f} ± {t5[1]:.3f}{bold} | "
              f"{bold}{f1[0]:.3f} ± {f1[1]:.3f}{bold} |")


if __name__ == "__main__":
    main()
