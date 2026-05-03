"""Action classification head for DA on Assembly101.

Input: embedding produced by ``FeatureEncoder``.
Output: logits over ``num_classes`` action classes (coarse verbs).
"""

import torch
import torch.nn as nn


class ActionClassifier(nn.Module):
    """Simple MLP classifier head over the encoder embeddings."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 17,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            embed_dim: Input embedding dimensionality (must match encoder).
            hidden_dim: Hidden layer width.
            num_classes: Number of action classes.
            dropout: Dropout probability after the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim) embeddings.
        Returns:
            (B, num_classes) raw logits.
        """
        return self.net(x)