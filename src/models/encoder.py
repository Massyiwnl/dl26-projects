"""Feature encoder for DA on Assembly101 TSM features.

Input: per-segment TSM feature vector (typically 2048-D).
Output: lower-dimensional embedding (typically 256-D).

This is the SHARED encoder of the DANN architecture: the action classifier
and the domain discriminator both branch from its output.
"""

from typing import Sequence

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """Multi-layer perceptron encoder over pre-extracted segment features.

    Architecture: Linear -> BatchNorm -> ReLU -> Dropout, repeated for each
    hidden layer, then a final Linear projection to ``embed_dim``.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dims: Sequence[int] = (1024, 512),
        embed_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        """
        Args:
            in_dim: Input feature dimensionality (2048 for TSM-ResNet50).
            hidden_dims: Sequence of hidden layer widths.
            embed_dim: Output embedding dimensionality.
            dropout: Dropout probability applied after each hidden block.
        """
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h

        # final projection (no activation, no BN: this is the embedding space)
        layers.append(nn.Linear(prev_dim, embed_dim))

        self.net = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) batch of pre-extracted features.
        Returns:
            (B, embed_dim) embeddings.
        """
        return self.net(x)