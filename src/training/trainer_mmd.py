"""MMD trainer (Deep Adaptation Networks / Long et al. 2015) for exo->ego DA
on Charades-Ego segment-level features.

Loss = CE(class_logits_src, y_src) + lambda_mmd * MMD^2(z_src, z_tgt)

where MMD^2 is a multi-kernel Gaussian Maximum Mean Discrepancy on the
encoder embeddings (see `src/losses/mmd.py`). Unlike DANN, this requires
no domain discriminator and no Ganin schedule: MMD is a stable
distribution-matching loss and lambda_mmd can be a fixed scalar. We still
apply a short warmup (lambda_mmd = 0 for the first `warmup_epochs`)
so the classifier head can take its first steps before alignment kicks in.

CLI:
    python -m src.training.trainer_mmd \
        --train-source data/processed/charades-ego/segment_features/train_source.npz \
        --train-target data/processed/charades-ego/segment_features/train_target.npz \
        --val-target   data/processed/charades-ego/segment_features/val_target.npz \
        --output-dir   experiments/checkpoints/CHAR_mmd \
        --epochs 50 --batch-size 256 --lambda-mmd 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.charades_ego import CharadesEgoSegmentDataset
from src.datasets.pair_loader import PairedDomainIterator
from src.evaluation.metrics import aggregate_metrics, format_metrics_summary
from src.losses.mmd import multi_kernel_mmd2
from src.models.classifier import ActionClassifier
from src.models.encoder import FeatureEncoder
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


# ----------------------------- model wrapper ----------------------------


class MMDModel(nn.Module):
    """Encoder + classifier inside a single Module for clean checkpointing.

    `forward` returns ``(embeddings, class_logits)`` — paralleling DANNModel
    but without the domain discriminator. The MMD alignment is applied
    externally in the training loop, directly on the embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        encoder_hidden: tuple = (1024, 512),
        embed_dim: int = 256,
        cls_hidden: int = 256,
        num_classes: int = 157,
        dropout_encoder: float = 0.3,
        dropout_cls: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(
            in_dim=in_dim, hidden_dims=encoder_hidden,
            embed_dim=embed_dim, dropout=dropout_encoder,
        )
        self.classifier = ActionClassifier(
            embed_dim=embed_dim, hidden_dim=cls_hidden,
            num_classes=num_classes, dropout=dropout_cls,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.classifier(z)
        return z, logits


# ----------------------------- evaluation -------------------------------


@torch.no_grad()
def evaluate_target(
    model: MMDModel,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    model.eval()
    logits_all, targets_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        _, logits = model(x)
        logits_all.append(logits.cpu())
        targets_all.append(y)
    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    return aggregate_metrics(
        logits_all, targets_all, num_classes=num_classes, is_logits=True
    )


# ----------------------------- training ---------------------------------


def train_one_epoch(
    model: MMDModel,
    pair_iter: PairedDomainIterator,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    lambda_mmd: float,
    epoch: int,
    warmup_epochs: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    sums = {"L_cls": 0.0, "L_mmd": 0.0, "L_tot": 0.0, "src_top1": 0.0}
    n_batches = 0

    active_lambda = 0.0 if epoch <= warmup_epochs else lambda_mmd

    for batch in pair_iter:
        # joint forward through the encoder (src + tgt together for efficiency)
        x = torch.cat([batch.src_x, batch.tgt_x], dim=0)
        z_all = model.encoder(x)
        bs = batch.src_x.size(0)
        z_src = z_all[:bs]
        z_tgt = z_all[bs:]

        # classification only on the source
        logits_src = model.classifier(z_src)
        L_cls = ce_loss(logits_src, batch.src_y)

        # MMD only when active (skipped during warmup, saves a few seconds)
        if active_lambda > 0:
            L_mmd = multi_kernel_mmd2(z_src, z_tgt)
        else:
            L_mmd = torch.zeros((), device=x.device)

        loss = L_cls + active_lambda * L_mmd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            src_top1 = (
                logits_src.argmax(dim=1) == batch.src_y
            ).float().mean().item()

        sums["L_cls"] += L_cls.item()
        sums["L_mmd"] += L_mmd.item()
        sums["L_tot"] += loss.item()
        sums["src_top1"] += src_top1
        n_batches += 1
        global_step += 1

        writer.add_scalar("step/L_cls", L_cls.item(), global_step)
        writer.add_scalar("step/L_mmd", L_mmd.item(), global_step)
        writer.add_scalar("step/lambda_mmd_active", active_lambda, global_step)

    return {k: v / max(n_batches, 1) for k, v in sums.items()}, global_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", type=Path, required=True)
    ap.add_argument("--train-target", type=Path, required=True)
    ap.add_argument("--val-target", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--lambda-mmd", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir / "tb")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- data ----
    src_ds = CharadesEgoSegmentDataset(args.train_source)
    tgt_ds = CharadesEgoSegmentDataset(args.train_target)
    val_ds = CharadesEgoSegmentDataset(args.val_target)
    num_classes = max(src_ds.num_classes, tgt_ds.num_classes, val_ds.num_classes)

    print(f"Train source: {len(src_ds):,} segments")
    print(f"Train target: {len(tgt_ds):,} segments")
    print(f"Val target:   {len(val_ds):,} segments")
    print(f"num_classes:  {num_classes}")
    print(f"lambda_mmd:   {args.lambda_mmd}")
    print(f"warmup:       {args.warmup_epochs} epoch(s)\n")

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ---- model ----
    model = MMDModel(
        in_dim=src_ds.feature_dim,
        encoder_hidden=(1024, 512),
        embed_dim=args.embed_dim,
        cls_hidden=256,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()

    # ---- training loop ----
    best_bal_acc = -1.0
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        pair_iter = PairedDomainIterator(
            src_loader=src_loader, tgt_loader=tgt_loader,
            steps_per_epoch=len(src_loader), device=device,
        )
        tr, global_step = train_one_epoch(
            model, pair_iter, optimizer, ce_loss,
            lambda_mmd=args.lambda_mmd, epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            writer=writer, global_step=global_step,
        )
        scheduler.step()

        val = evaluate_target(model, val_loader, num_classes, device)

        writer.add_scalar("epoch/L_cls", tr["L_cls"], epoch)
        writer.add_scalar("epoch/L_mmd", tr["L_mmd"], epoch)
        writer.add_scalar("epoch/L_tot", tr["L_tot"], epoch)
        writer.add_scalar("epoch/src_top1", tr["src_top1"], epoch)
        writer.add_scalar("val/top1", val["top1"], epoch)
        writer.add_scalar("val/top5", val["top5"], epoch)
        writer.add_scalar("val/balanced_accuracy", val["balanced_accuracy"], epoch)
        writer.add_scalar("val/macro_f1", val["macro_f1"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        active = 0.0 if epoch <= args.warmup_epochs else args.lambda_mmd
        print(
            f"epoch {epoch:>3}/{args.epochs} | "
            f"L_cls={tr['L_cls']:.3f} L_mmd={tr['L_mmd']:.4f} L_tot={tr['L_tot']:.3f} "
            f"lam={active:.2f} src_top1={tr['src_top1']:.3f} | "
            f"val target: top1={val['top1']:.3f} bal={val['balanced_accuracy']:.3f} "
            f"f1={val['macro_f1']:.3f}"
        )

        save_checkpoint(
            args.output_dir / "latest.pt",
            model=model, optimizer=optimizer, epoch=epoch,
            best_metric=best_bal_acc, config=config_dict,
        )
        if val["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = val["balanced_accuracy"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model, optimizer=None, epoch=epoch,
                best_metric=best_bal_acc, config=config_dict,
            )

    writer.close()

    print("\n=== Final eval (best.pt on target val) ===")
    print(format_metrics_summary(val, num_classes=num_classes))
    print(f"\nBest target val balanced accuracy: {best_bal_acc:.4f}")
    print(f"Checkpoints in: {args.output_dir}")


if __name__ == "__main__":
    main()