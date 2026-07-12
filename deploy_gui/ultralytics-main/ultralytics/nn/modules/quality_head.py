# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Quality Head (OK/NG) for CargoDefect-YOLOv26."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_QUALITY_THRESHOLDS = {
    "defect_score": 0.5,
    "defect_area_ratio": 0.05,
    "severe_classes": [1, 2, 3],  # crack, dent, stain (defect classifier indices; none=4)
}

DEFECT_NONE_IDX = 4


class QualityHead(nn.Module):
    """Image-level OK/NG quality head (binary BCE when nc_quality=2)."""

    export = False

    def __init__(
        self,
        ch: tuple[int, ...],
        nc_quality: int = 2,
        thresholds: dict[str, Any] | None = None,
    ):
        """Initialize QualityHead.

        Args:
            ch (tuple): Channel sizes from detection feature maps.
            nc_quality (int): 2 for OK/NG binary, 3 for legacy OK/Minor/Severe.
            thresholds (dict, optional): OK/NG decision thresholds.
        """
        super().__init__()
        self.nc_quality = nc_quality
        self.thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})}
        total_c = sum(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        out_dim = 1 if nc_quality == 2 else nc_quality
        self.fc = nn.Linear(total_c, out_dim)

    def extract_features(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Global average pooling over multi-scale ROI features."""
        pooled = [self.pool(f).flatten(1) for f in feats]
        return torch.cat(pooled, dim=1)

    def forward(
        self,
        feats: list[torch.Tensor],
        detect_preds: dict | None = None,
        defect_preds: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass returning logits (train) or OK/NG verdict (inference)."""
        feat = self.extract_features(feats)
        logits = self.fc(feat)
        if self.training:
            return logits
        if self.nc_quality == 2:
            probs = torch.sigmoid(logits[:, 0])
            verdict = (probs >= self.thresholds["defect_score"]).long()
        else:
            probs = F.softmax(logits, dim=-1)
            verdict = self.judge_ok_ng(probs, detect_preds)
        if defect_preds is not None:
            defect_cls = defect_preds.argmax(dim=-1)
            verdict = torch.where(defect_cls < DEFECT_NONE_IDX, torch.ones_like(verdict), verdict)
        if self.export:
            return torch.cat([probs.unsqueeze(-1), verdict.float().unsqueeze(-1)], dim=-1)
        return {"logits": logits, "probs": probs, "verdict": verdict}

    def judge_ok_ng(
        self,
        probs: torch.Tensor,
        detect_preds: dict | None = None,
    ) -> torch.Tensor:
        """Apply industrial OK/NG rules from config thresholds."""
        batch_size = probs.shape[0]
        device = probs.device
        verdict = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0=OK, 1=NG
        minor_idx, severe_idx = 1, 2

        for i in range(batch_size):
            if detect_preds is None:
                pred_class = probs[i].argmax().item()
                verdict[i] = 0 if pred_class == 0 else 1
                continue

            scores = detect_preds.get("scores")
            if scores is None:
                verdict[i] = 0 if probs[i, 0] >= probs[i, 1:].max() else 1
                continue

            if isinstance(scores, torch.Tensor):
                sig_scores = scores.sigmoid() if scores.dtype != torch.float16 else scores
                max_score = sig_scores.max().item() if sig_scores.numel() else 0.0
            else:
                max_score = 0.0

            severe_list = self.thresholds.get("severe_classes", [])
            has_severe = False
            if isinstance(scores, torch.Tensor) and scores.numel():
                cls_idx = sig_scores.max(dim=-1).values.max(dim=-1).values
                cls_vals = [c for c in cls_idx.flatten().tolist() if c == c]  # drop NaN
                has_severe = any(int(c) in severe_list for c in cls_vals)

            if max_score < self.thresholds["defect_score"] and probs[i, 0] > probs[i, minor_idx]:
                verdict[i] = 0
            elif probs[i, severe_idx] > self.thresholds["defect_score"]:
                verdict[i] = 1
            elif has_severe:
                verdict[i] = 1
            elif probs[i, minor_idx] > self.thresholds["defect_score"]:
                verdict[i] = 1
            else:
                verdict[i] = 0 if probs[i].argmax().item() == 0 else 1
        return verdict

    @staticmethod
    def derive_quality_labels(
        batch: dict[str, torch.Tensor],
        severe_classes: list[int] | None = None,
    ) -> torch.Tensor:
        """Derive quality labels from batch; prefer explicit quality_label (0=OK, 1=NG)."""
        if "quality_label" in batch:
            from ultralytics.data.cargodefect import quality_targets_from_batch

            nc_quality = batch.get("nc_quality", 3)
            return quality_targets_from_batch(batch, nc_quality=int(nc_quality))
        if "quality" in batch:
            return batch["quality"].long()
        severe_classes = severe_classes or DEFAULT_QUALITY_THRESHOLDS["severe_classes"]
        batch_size = batch["img"].shape[0]
        device = batch["img"].device
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        batch_idx = batch["batch_idx"]
        cls = batch["cls"].view(-1).long()
        for i in range(batch_size):
            mask = batch_idx.view(-1) == i
            img_cls = cls[mask]
            if img_cls.numel() == 0:
                labels[i] = 0
            elif any(int(c) in severe_classes for c in img_cls.tolist()):
                labels[i] = 2
            else:
                labels[i] = 1
        return labels
