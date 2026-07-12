# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Physics-Guided Multi-domain Enhancement Module (PGME) for CargoDefect-YOLOv26."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv
from .edge_branch import EdgeBranch
from .frequency_branch import FrequencyBranch


class AttentionFusion(nn.Module):
    """Lightweight channel attention fusion for multi-domain features."""

    def __init__(self, c: int, n_branches: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_branches) / n_branches)
        self.fuse = Conv(c * n_branches, c, k=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        weighted = sum(w[i] * feats[i] for i in range(len(feats)))
        return self.fuse(torch.cat(feats, dim=1)) + weighted


class PGME(nn.Module):
    """Physics-Guided Multi-domain Enhancement: Sobel Edge + Haar-DWT + RGB with attention fusion."""

    def __init__(self, c_out: int, use_edge: bool = True, use_frequency: bool = True):
        super().__init__()
        self.use_edge = use_edge
        self.use_frequency = use_frequency
        self.rgb_proj = Conv(c_out, c_out, k=1)
        self.edge_branch = EdgeBranch(c_out) if use_edge else None
        self.frequency_branch = FrequencyBranch(c_out) if use_frequency else None
        n_branches = 1 + int(use_edge) + int(use_frequency)
        self.fusion = AttentionFusion(c_out, n_branches)
        # Learnable PGME fusion strength (default 1.0 = full PGME, 0.0 = backbone-only)
        self.alpha_pgme = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Fuse edge, frequency, and RGB features into backbone feature map."""
        if not isinstance(x, list):
            return x
        image, feat = x[0], x[1]
        branches = [self.rgb_proj(feat)]
        if self.use_edge and self.edge_branch is not None:
            branches.append(self.edge_branch(image, target=feat))
        if self.use_frequency and self.frequency_branch is not None:
            branches.append(self.frequency_branch(image, target=feat))
        alpha = self.alpha_pgme if hasattr(self, "alpha_pgme") else 1.0
        return feat + alpha * self.fusion(branches)
