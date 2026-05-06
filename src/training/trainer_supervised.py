"""Supervised baseline trainer for B1 (source-only) and B2 (target-only).

Both baselines share the exact same training code: a Encoder + Classifier
trained with cross-entropy on a single labelled dataset. The two settings
differ only in the .npz used for the train DataLoader:

    B1: train_source.npz  (labelled exocentric)  -> evaluate on target val
    B2: train_target.npz  (labelled egocentric)  -> evaluate on target val

The trainer:
    - Sets the seed for reproducibility.
    - Uses Adam + cosine LR schedule.
    - Logs to TensorBoard at every epoch and prints a one-line summary.
    - Saves the latest checkpoint, plus a "best.pt" tracking best
      balanced_accuracy on target val.

CLI:
    python -m src.training.trainer_supervised \
        --train-npz data/processed/segment_features/train_source.npz \
        --val-npz   data/processed/segment_features/val_target.npz \
        --output-dir experiments/checkpoints/baseline_source_only \
        --epochs 30 --batch-size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.assembly101 import Assembly101SegmentDataset
from src.evaluation.metrics import aggregate_metrics, format_metrics_summary
from src.models.classifier import ActionClassifier
from src.models.encoder import FeatureEncoder
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def build_model(in_dim: int, embed_dim: int, num_classes: int, device: torch.device):
    encoder = FeatureEncoder(
        in_dim=in_dim, hidden_dims=(1024, 512), embed_dim=embed_dim, dropout=0.5
    ).to(device)
    classifier = ActionClassifier(
        embed_dim=embed_dim, hidden_dim=128, num_classes=num_classes, dropout=0.3
    ).to(device)
    return encoder, classifier


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    encoder.eval()
    classifier.eval()
    logits_all, targets_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = classifier(encoder(x))
        logits_all.append(logits.cpu())
        targets_all.append(y)
    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    return aggregate_metrics(logits_all, targets_all, num_classes=num_classes, is_logits=True)


def train_one_epoch(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    encoder.train()
    classifier.train()
    total_loss = 0.0
    n_correct = 0
    n_seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = classifier(encoder(x))
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_correct += (logits.argmax(dim=1) == y).sum().item()
        n_seen += x.size(0)

    return {"loss": total_loss / max(n_seen, 1), "top1": n_correct / max(n_seen, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--val-npz", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir / "tb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- data ----
    train_ds = Assembly101SegmentDataset(args.train_npz)
    val_ds = Assembly101SegmentDataset(args.val_npz)
    num_classes = max(train_ds.num_classes, val_ds.num_classes)
    print(f"Train npz : {args.train_npz.name}  -> {len(train_ds):,} segments")
    print(f"Val   npz : {args.val_npz.name}    -> {len(val_ds):,} segments")
    print(f"num_classes: {num_classes}\n")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # ---- model ----
    encoder, classifier = build_model(
        in_dim=train_ds.feature_dim, embed_dim=args.embed_dim,
        num_classes=num_classes, device=device,
    )
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    # ---- train loop ----
    best_bal_acc = -1.0
    config_dict = vars(args).copy()
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in config_dict.items()}

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(encoder, classifier, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        val = evaluate(encoder, classifier, val_loader, num_classes, device)

        writer.add_scalar("train/loss", tr["loss"], epoch)
        writer.add_scalar("train/top1", tr["top1"], epoch)
        writer.add_scalar("val/top1", val["top1"], epoch)
        writer.add_scalar("val/top5", val["top5"], epoch)
        writer.add_scalar("val/balanced_accuracy", val["balanced_accuracy"], epoch)
        writer.add_scalar("val/macro_f1", val["macro_f1"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"epoch {epoch:>3}/{args.epochs} | "
            f"train loss={tr['loss']:.4f} top1={tr['top1']:.3f} | "
            f"val top1={val['top1']:.3f} top5={val['top5']:.3f} "
            f"bal={val['balanced_accuracy']:.3f} f1={val['macro_f1']:.3f}"
        )

        # save latest
        save_checkpoint(
            args.output_dir / "latest.pt",
            model=nn.ModuleDict({"encoder": encoder, "classifier": classifier}),
            optimizer=optimizer, epoch=epoch,
            best_metric=best_bal_acc, config=config_dict,
        )

        # save best
        if val["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = val["balanced_accuracy"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model=nn.ModuleDict({"encoder": encoder, "classifier": classifier}),
                optimizer=None, epoch=epoch, best_metric=best_bal_acc, config=config_dict,
            )

    writer.close()

    print("\n=== Final eval (best.pt on val set) ===")
    print(format_metrics_summary(val, num_classes=num_classes))
    print(f"\nBest balanced accuracy: {best_bal_acc:.4f}")
    print(f"Checkpoints saved in: {args.output_dir}")


if __name__ == "__main__":
    main()