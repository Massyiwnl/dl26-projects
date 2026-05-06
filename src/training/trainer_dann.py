"""DANN trainer (Domain-Adversarial Neural Network) for exo->ego DA on
Assembly101 segment-level features.

Pipeline at training time:
    1. Sample a labelled source batch (B_s features + verb labels).
    2. Sample an unlabelled target batch (B_t features).
    3. Forward source: encoder -> classifier  -> classification logits
    4. Forward source+target: encoder -> GRL -> discriminator -> domain logits
    5. Loss = CE(class_logits_src, y_src) + BCE(domain_logits, domain_labels)
    6. Backward through GRL flips the gradient of L_dom on the encoder side,
       making the encoder PRODUCE domain-invariant features over time.

Lambda follows Ganin et al. (2016):
    lambda_p = (2 / (1 + exp(-gamma * p))) - 1
where p = current_step / total_steps. We also implement a "warmup" of
warmup_epochs epochs during which lambda is forced to 0 so the classifier
gets a head-start before the discriminator pushes back.

CLI:
    python -m src.training.trainer_dann \
        --train-source data/processed/segment_features/train_source.npz \
        --train-target data/processed/segment_features/train_target.npz \
        --val-target   data/processed/segment_features/val_target.npz \
        --output-dir   experiments/checkpoints/SYNTH_dann \
        --epochs 30 --batch-size 256 --gamma 10 --lambda-max 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.assembly101 import Assembly101SegmentDataset
from src.datasets.pair_loader import PairedDomainIterator
from src.evaluation.metrics import aggregate_metrics, format_metrics_summary
from src.models.dann import DANNModel
from src.utils.checkpoint import save_checkpoint
from src.utils.schedules import grl_lambda_ganin
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate_target(
    model: DANNModel,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    model.eval()
    logits_all, targets_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits_all.append(out.class_logits.cpu())
        targets_all.append(y)
    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    return aggregate_metrics(logits_all, targets_all, num_classes=num_classes, is_logits=True)


def train_one_epoch(
    model: DANNModel,
    pair_iter: PairedDomainIterator,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    bce_loss: nn.Module,
    lambda_max: float,
    gamma: float,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()

    sums = {"L_cls": 0.0, "L_dom": 0.0, "dom_acc": 0.0, "src_top1": 0.0}
    n_batches = 0

    for batch in pair_iter:
        # ---- compute lambda ----
        if epoch <= warmup_epochs:
            lambda_p = 0.0
        else:
            # progress over the post-warmup horizon
            steps_per_epoch = len(pair_iter)
            p = ((epoch - warmup_epochs - 1) * steps_per_epoch + n_batches) / max(
                (epochs - warmup_epochs) * steps_per_epoch, 1
            )
            lambda_p = grl_lambda_ganin(p, gamma=gamma, lambda_max=lambda_max)
        model.set_grl_lambda(lambda_p)

        # ---- forward ----
        x = torch.cat([batch.src_x, batch.tgt_x], dim=0)
        out = model(x)
        bs = batch.src_x.size(0)
        cls_logits_src = out.class_logits[:bs]
        dom_logits = out.domain_logits  # (2B, 1)

        # ---- losses ----
        L_cls = ce_loss(cls_logits_src, batch.src_y)
        dom_labels = torch.cat(
            [
                torch.zeros(bs, 1, device=x.device),
                torch.ones(bs, 1, device=x.device),
            ],
            dim=0,
        )
        L_dom = bce_loss(dom_logits, dom_labels)
        loss = L_cls + L_dom

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ---- metrics ----
        with torch.no_grad():
            dom_pred = (torch.sigmoid(dom_logits) >= 0.5).float()
            dom_acc = (dom_pred == dom_labels).float().mean().item()
            src_top1 = (cls_logits_src.argmax(dim=1) == batch.src_y).float().mean().item()

        sums["L_cls"] += L_cls.item()
        sums["L_dom"] += L_dom.item()
        sums["dom_acc"] += dom_acc
        sums["src_top1"] += src_top1
        n_batches += 1
        global_step += 1

        # per-step TB log
        writer.add_scalar("step/L_cls", L_cls.item(), global_step)
        writer.add_scalar("step/L_dom", L_dom.item(), global_step)
        writer.add_scalar("step/dom_acc", dom_acc, global_step)
        writer.add_scalar("step/lambda_p", lambda_p, global_step)

    return {k: v / max(n_batches, 1) for k, v in sums.items()}, global_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", type=Path, required=True)
    ap.add_argument("--train-target", type=Path, required=True)
    ap.add_argument("--val-target", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir / "tb")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- data ----
    src_ds = Assembly101SegmentDataset(args.train_source)
    tgt_ds = Assembly101SegmentDataset(args.train_target)
    val_ds = Assembly101SegmentDataset(args.val_target)
    num_classes = max(src_ds.num_classes, tgt_ds.num_classes, val_ds.num_classes)

    print(f"Train source: {len(src_ds):,} segments")
    print(f"Train target: {len(tgt_ds):,} segments")
    print(f"Val target:   {len(val_ds):,} segments")
    print(f"num_classes:  {num_classes}\n")

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ---- model ----
    model = DANNModel(
        in_dim=src_ds.feature_dim,
        encoder_hidden=(1024, 512),
        embed_dim=args.embed_dim,
        cls_hidden=128,
        num_classes=num_classes,
        disc_hidden=(256, 128),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

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
            model, pair_iter, optimizer, ce_loss, bce_loss,
            lambda_max=args.lambda_max, gamma=args.gamma,
            epoch=epoch, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
            writer=writer, global_step=global_step,
        )
        scheduler.step()

        val = evaluate_target(model, val_loader, num_classes, device)

        writer.add_scalar("epoch/L_cls", tr["L_cls"], epoch)
        writer.add_scalar("epoch/L_dom", tr["L_dom"], epoch)
        writer.add_scalar("epoch/dom_acc", tr["dom_acc"], epoch)
        writer.add_scalar("epoch/src_top1", tr["src_top1"], epoch)
        writer.add_scalar("val/top1", val["top1"], epoch)
        writer.add_scalar("val/top5", val["top5"], epoch)
        writer.add_scalar("val/balanced_accuracy", val["balanced_accuracy"], epoch)
        writer.add_scalar("val/macro_f1", val["macro_f1"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"epoch {epoch:>3}/{args.epochs} | "
            f"L_cls={tr['L_cls']:.3f} L_dom={tr['L_dom']:.3f} "
            f"dom_acc={tr['dom_acc']:.3f} src_top1={tr['src_top1']:.3f} | "
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