"""V3-MC discounted Monte Carlo critic + WM-terminal nano outcome predictor.

Both share the same backbone (2-layer MLP, hidden=256, dropout=0.2);
they differ only in the output head and training objective.

V3-MC:  output = scalar logit → sigmoid → P(success); BCE loss on γ^(T−t)·R_term
WMT:    output = 4-d (eef_xyz, gripper_open); MSE loss on terminal_state
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MLPBackbone(nn.Module):
    """Shared 2-layer MLP backbone."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class V3MC(nn.Module):
    """Discounted Monte Carlo critic.

    Predicts P(eventual success | obs, chunk) trained on discounted MC labels
    y_t = γ^(T-t) · R_terminal.
    """

    def __init__(
        self,
        obs_dim: int,
        chunk_dim: int,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.chunk_dim = chunk_dim
        self.backbone = _MLPBackbone(obs_dim + chunk_dim, hidden, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self, obs_features: torch.Tensor, chunk_features: torch.Tensor
    ) -> torch.Tensor:
        """Returns logits (B,)."""
        x = torch.cat([obs_features, chunk_features], dim=-1)
        return self.head(self.backbone(x)).squeeze(-1)

    @torch.no_grad()
    def score(
        self, obs_features: torch.Tensor, chunk_features: torch.Tensor
    ) -> torch.Tensor:
        """Returns sigmoid probabilities (B,) in [0,1]. Higher = better."""
        return torch.sigmoid(self.forward(obs_features, chunk_features))


class WMTerminal(nn.Module):
    """Nano outcome predictor.

    Predicts the terminal observation summary (eef_xyz + gripper_open) that
    would result from executing this chunk from this observation. At inference,
    score = -L2(predicted, per-task success centroid).
    """

    def __init__(
        self,
        obs_dim: int,
        chunk_dim: int,
        out_dim: int = 4,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.chunk_dim = chunk_dim
        self.out_dim = out_dim
        self.backbone = _MLPBackbone(obs_dim + chunk_dim, hidden, dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(
        self, obs_features: torch.Tensor, chunk_features: torch.Tensor
    ) -> torch.Tensor:
        """Returns predictions (B, out_dim)."""
        x = torch.cat([obs_features, chunk_features], dim=-1)
        return self.head(self.backbone(x))

    @torch.no_grad()
    def score(
        self,
        obs_features: torch.Tensor,
        chunk_features: torch.Tensor,
        targets: torch.Tensor,  # (B, out_dim) per-example target
    ) -> torch.Tensor:
        """Returns -L2 distance (B,). Higher (less negative) = better."""
        pred = self.forward(obs_features, chunk_features)
        return -torch.norm(pred - targets, dim=-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
