# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Edge enhancement branch for CargoDefect-YOLOv26."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class SobelEdge(nn.Module):
    """Compute Sobel edge magnitude: Edge = sqrt(Gx^2 + Gy^2)."""

    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edge magnitude from RGB input."""
        gray = x.mean(1, keepdim=True)
        gx = F.conv2d(gray, self.gx, padding=1)
        gy = F.conv2d(gray, self.gy, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class EdgeBranch(nn.Module):
    """Lightweight edge feature encoder: Sobel -> 3x3 Conv + BN + SiLU."""

    def __init__(self, c_out: int):
        """Initialize EdgeBranch.

        Args:
            c_out (int): Output channel dimension for encoded edge features.
        """
        super().__init__()
        self.sobel = SobelEdge()
        self.encode = Conv(1, c_out, k=3)

    @staticmethod
    def _match_size(feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if feat.shape[-2:] != target.shape[-2:]:
            feat = F.interpolate(feat, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return feat

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """Encode edge map to feature tensor matching target spatial size."""
        edge = self.sobel(x)
        edge_feat = self.encode(edge)
        if target is not None:
            edge_feat = self._match_size(edge_feat, target)
        return edge_feat


class EdgeEnhance(nn.Module):
    """Fuse edge features with backbone features: feature = feature + edge_feature."""

    def __init__(self, c_out: int, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.edge_branch = EdgeBranch(c_out) if enabled else None

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Fuse edge branch output with backbone feature map."""
        if not self.enabled or self.edge_branch is None:
            return x[-1] if isinstance(x, list) else x
        image, feat = x[0], x[1]
        return feat + self.edge_branch(image, target=feat)
