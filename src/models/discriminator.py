"""Binary domain discriminator for adversarial DA (DANN).

Input: embedding produced by ``FeatureEncoder`` (typically already passed
through a Gradient Reversal Layer).
Output: a single logit per sample. After sigmoid, this is the predicted
probability that the sample comes from the source domain.

Convention used in this project:
    label 0 = source domain (exocentric)
    label 1 = target domain (egocentric)

Use ``nn.BCEWithLogitsLoss`` on the raw logit during training.
"""

import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    """MLP that predicts the domain label (source vs. target) from embeddings."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            embed_dim: Input embedding dimensionality (must match encoder).
            hidden_dims: Sequence of hidden layer widths.
            dropout: Dropout probability after each hidden layer.
        """
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h

        # binary output: single logit (use BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim) embeddings.
        Returns:
            (B, 1) raw logits. Apply sigmoid to get domain probability.
        """
        return self.net(x)