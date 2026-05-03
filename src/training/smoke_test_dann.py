"""End-to-end smoke test for the DANN model on synthetic data.

This script generates two fake "domains" of features with a clear domain bias
plus a class signal. It then trains DANN for a few hundred steps and prints
diagnostics that should match the theoretical behaviour:

* Classifier loss on the SOURCE batch decreases.
* Domain discriminator accuracy starts high (the discriminator easily tells
  source from target) and drops toward 50% (i.e., the encoder is confusing it).
* Lambda_p ramps from 0 toward 1 following Ganin's schedule.

Run from the repo root:
    python -m src.training.smoke_test_dann
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.dann import DANNModel
from src.utils.schedules import grl_lambda_ganin
from src.utils.seed import set_seed


# ---------------------------- synthetic data --------------------------------

def make_fake_dataset(
    n_samples: int,
    in_dim: int,
    num_classes: int,
    domain_shift: float,
    domain_id: int,
    seed: int,
) -> TensorDataset:
    """Generate a domain-biased classification dataset.

    Each class has a distinct mean vector ``mu_c``. Samples are drawn from
    ``N(mu_c + domain_shift * d, I)`` where ``d`` is a domain-specific direction.
    This way the class signal is recoverable across domains, but a simple
    classifier trained only on one domain will be biased by ``domain_shift``.
    """
    g = torch.Generator().manual_seed(seed)

    # class means (shared across domains)
    mu_class = torch.randn(num_classes, in_dim, generator=g) * 2.0
    # domain direction (different per domain)
    d_vec = torch.randn(in_dim, generator=g)
    d_vec = d_vec / d_vec.norm() * domain_shift

    labels = torch.randint(0, num_classes, (n_samples,), generator=g)
    noise = torch.randn(n_samples, in_dim, generator=g)

    if domain_id == 0:
        x = mu_class[labels] + noise + d_vec.unsqueeze(0)
    else:
        x = mu_class[labels] + noise - d_vec.unsqueeze(0)

    return TensorDataset(x, labels)


# ---------------------------- training loop ---------------------------------

def main() -> None:
    set_seed(0)

    # hyperparameters (toy scale, runs in seconds)
    in_dim = 64
    num_classes = 5
    n_samples_per_domain = 1024
    batch_size = 64
    total_steps = 200
    log_every = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- data ---
    src_ds = make_fake_dataset(
        n_samples=n_samples_per_domain,
        in_dim=in_dim,
        num_classes=num_classes,
        domain_shift=3.0,
        domain_id=0,
        seed=1,
    )
    tgt_ds = make_fake_dataset(
        n_samples=n_samples_per_domain,
        in_dim=in_dim,
        num_classes=num_classes,
        domain_shift=3.0,
        domain_id=1,
        seed=2,
    )

    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- model ---
    model = DANNModel(
        in_dim=in_dim,
        encoder_hidden=(64, 32),
        embed_dim=32,
        cls_hidden=32,
        num_classes=num_classes,
        disc_hidden=(32, 16),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)

    print(f"{'step':>5} | {'lambda':>7} | {'L_cls':>7} | {'L_dom':>7} | {'dom_acc':>8}")
    print("-" * 55)

    for step in range(1, total_steps + 1):
        # refresh iterators when exhausted
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

        # ---- update lambda_p (Ganin schedule) ----
        progress = step / total_steps
        lambda_p = grl_lambda_ganin(progress, gamma=10.0, lambda_max=1.0)
        model.set_grl_lambda(lambda_p)

        # ---- forward on source + target ----
        x = torch.cat([x_s, x_t], dim=0)
        out = model(x)

        bs = x_s.size(0)
        cls_logits_src = out.class_logits[:bs]
        dom_logits_all = out.domain_logits  # (2B, 1)

        # ---- losses ----
        L_cls = ce_loss(cls_logits_src, y_s)
        # domain labels: 0 source, 1 target
        dom_labels = torch.cat(
            [torch.zeros(bs, 1, device=device), torch.ones(bs, 1, device=device)],
            dim=0,
        )
        L_dom = bce_loss(dom_logits_all, dom_labels)
        loss = L_cls + L_dom

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ---- logging ----
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                dom_pred = (torch.sigmoid(dom_logits_all) >= 0.5).float()
                dom_acc = (dom_pred == dom_labels).float().mean().item()
            print(
                f"{step:>5} | {lambda_p:>7.3f} | {L_cls.item():>7.3f} | "
                f"{L_dom.item():>7.3f} | {dom_acc:>8.3f}"
            )

    print("\nExpected behaviour:")
    print("  - L_cls should DECREASE (classifier learns the source classes).")
    print("  - dom_acc should start near 1.0 and DRIFT TOWARD 0.5")
    print("    (encoder is confusing the discriminator).")
    print("  - lambda climbs smoothly from 0 to ~1 (Ganin schedule).")


if __name__ == "__main__":
    main()