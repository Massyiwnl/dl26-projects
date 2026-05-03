"""End-to-end smoke test for MMD-based domain adaptation on synthetic data.

The architecture is the same as DANN but without GRL/discriminator:
    encoder -> classifier (on source)
            -> MMD(embeddings_source, embeddings_target)  --minimised together

Expected behaviour:
* ``L_cls`` decreases on source.
* ``L_mmd`` decreases over training (encoder learns to align source/target).
* ``L_total = L_cls + lambda_mmd * L_mmd`` decreases.

Run from the repo root:
    python -m src.training.smoke_test_mmd
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.losses.mmd import multi_kernel_mmd2
from src.models.classifier import ActionClassifier
from src.models.encoder import FeatureEncoder
from src.utils.seed import set_seed


def make_fake_dataset(
    n_samples: int,
    in_dim: int,
    num_classes: int,
    domain_shift: float,
    domain_id: int,
    seed: int,
) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    mu_class = torch.randn(num_classes, in_dim, generator=g) * 2.0
    d_vec = torch.randn(in_dim, generator=g)
    d_vec = d_vec / d_vec.norm() * domain_shift
    labels = torch.randint(0, num_classes, (n_samples,), generator=g)
    noise = torch.randn(n_samples, in_dim, generator=g)
    if domain_id == 0:
        x = mu_class[labels] + noise + d_vec.unsqueeze(0)
    else:
        x = mu_class[labels] + noise - d_vec.unsqueeze(0)
    return TensorDataset(x, labels)


def main() -> None:
    set_seed(0)

    in_dim = 64
    num_classes = 5
    n_samples_per_domain = 1024
    batch_size = 64
    total_steps = 200
    log_every = 20
    lambda_mmd = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    src_ds = make_fake_dataset(n_samples_per_domain, in_dim, num_classes, 3.0, 0, 1)
    tgt_ds = make_fake_dataset(n_samples_per_domain, in_dim, num_classes, 3.0, 1, 2)

    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = FeatureEncoder(in_dim=in_dim, hidden_dims=(64, 32), embed_dim=32).to(device)
    classifier = ActionClassifier(embed_dim=32, hidden_dim=32, num_classes=num_classes).to(device)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)

    print(f"{'step':>5} | {'L_cls':>7} | {'L_mmd':>7} | {'L_total':>8}")
    print("-" * 45)

    for step in range(1, total_steps + 1):
        try:
            x_s, y_s = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            x_s, y_s = next(src_iter)
        try:
            x_t, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            x_t, _ = next(tgt_iter)

        x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)

        z_s = encoder(x_s)
        z_t = encoder(x_t)
        logits_s = classifier(z_s)

        L_cls = ce_loss(logits_s, y_s)
        L_mmd = multi_kernel_mmd2(z_s, z_t)
        loss = L_cls + lambda_mmd * L_mmd

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % log_every == 0 or step == 1:
            print(
                f"{step:>5} | {L_cls.item():>7.3f} | {L_mmd.item():>7.4f} | "
                f"{loss.item():>8.3f}"
            )

    print("\nExpected behaviour:")
    print("  - L_cls should DECREASE on source.")
    print("  - L_mmd should DECREASE (encoder aligns source/target distributions).")


if __name__ == "__main__":
    main()