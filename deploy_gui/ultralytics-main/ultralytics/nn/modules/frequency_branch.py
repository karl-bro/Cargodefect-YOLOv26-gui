# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Frequency enhancement branch for CargoDefect-YOLOv26."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class HaarDWT(nn.Module):
    """2D Haar discrete wavelet transform producing LL, LH, HL, HH sub-bands."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Haar DWT on each channel independently."""
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) * 0.25
        lh = (a + b - c - d) * 0.25
        hl = (a - b + c - d) * 0.25
        hh = (a - b - c + d) * 0.25
        return ll, lh, hl, hh


class FrequencyBranch(nn.Module):
    """Haar-DWT frequency encoder: DWT -> Conv3x3 -> Concat -> 1x1 Conv."""

    def __init__(self, c_out: int):
        super().__init__()
        self.dwt = HaarDWT()
        self.conv_ll = Conv(3, c_out // 4, k=3)
        self.conv_lh = Conv(3, c_out // 4, k=3)
        self.conv_hl = Conv(3, c_out // 4, k=3)
        self.conv_hh = Conv(3, c_out // 4, k=3)
        self.fuse = Conv(c_out, c_out, k=1)

    @staticmethod
    def _match_size(feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if feat.shape[-2:] != target.shape[-2:]:
            feat = F.interpolate(feat, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return feat

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """Encode high-frequency texture features from input image."""
        ll, lh, hl, hh = self.dwt(x)
        freq = torch.cat(
            [self.conv_ll(ll), self.conv_lh(lh), self.conv_hl(hl), self.conv_hh(hh)],
            dim=1,
        )
        freq_feat = self.fuse(freq)
        if target is not None:
            freq_feat = self._match_size(freq_feat, target)
        return freq_feat


class FrequencyEnhance(nn.Module):
    """Fuse frequency features with backbone features: feature = feature + frequency_feature."""

    def __init__(self, c_out: int, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.frequency_branch = FrequencyBranch(c_out) if enabled else None

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.frequency_branch is None:
            return x[-1] if isinstance(x, list) else x
        image, feat = x[0], x[1]
        return feat + self.frequency_branch(image, target=feat)
