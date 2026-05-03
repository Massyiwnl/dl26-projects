"""DANN model: shared encoder + action classifier + domain discriminator (via GRL).

The key idea (Ganin et al., 2016) is that the encoder is trained to:
1. Help the action classifier (by minimising classification loss).
2. Confuse the domain discriminator (because gradients from the discriminator
   are reversed by the GRL before reaching the encoder).

This module exposes a single ``forward`` that returns both class logits and
domain logits in one pass, which keeps the training loop tidy.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from src.models.classifier import ActionClassifier
from src.models.discriminator import DomainDiscriminator
from src.models.encoder import FeatureEncoder
from src.models.grl import GradientReversalLayer


@dataclass
class DANNOutput:
    """Container for the multi-head outputs of DANN."""

    embeddings: torch.Tensor   # (B, embed_dim)
    class_logits: torch.Tensor  # (B, num_classes)
    domain_logits: torch.Tensor  # (B, 1)


class DANNModel(nn.Module):
    """Composed model: FeatureEncoder + ActionClassifier + GRL + DomainDiscriminator."""

    def __init__(
        self,
        in_dim: int = 2048,
        encoder_hidden: Sequence[int] = (1024, 512),
        embed_dim: int = 256,
        cls_hidden: int = 128,
        num_classes: int = 17,
        disc_hidden: Sequence[int] = (256, 128),
        encoder_dropout: float = 0.5,
        cls_dropout: float = 0.3,
        disc_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = FeatureEncoder(
            in_dim=in_dim,
            hidden_dims=encoder_hidden,
            embed_dim=embed_dim,
            dropout=encoder_dropout,
        )
        self.classifier = ActionClassifier(
            embed_dim=embed_dim,
            hidden_dim=cls_hidden,
            num_classes=num_classes,
            dropout=cls_dropout,
        )
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.discriminator = DomainDiscriminator(
            embed_dim=embed_dim,
            hidden_dims=tuple(disc_hidden),
            dropout=disc_dropout,
        )

    def set_grl_lambda(self, lambda_: float) -> None:
        """Update the GRL coefficient (called from the training loop schedule)."""
        self.grl.set_lambda(lambda_)

    def forward(self, x: torch.Tensor) -> DANNOutput:
        """
        Args:
            x: (B, in_dim) input features (concatenation of source + target
               batches is typical at training time).
        Returns:
            DANNOutput with embeddings, class logits, domain logits.
        """
        z = self.encoder(x)
        class_logits = self.classifier(z)
        z_reversed = self.grl(z)
        domain_logits = self.discriminator(z_reversed)
        return DANNOutput(
            embeddings=z,
            class_logits=class_logits,
            domain_logits=domain_logits,
        )