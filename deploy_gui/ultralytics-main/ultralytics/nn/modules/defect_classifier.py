# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ROI-based defect classifier for CargoDefect inspection pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy


class DefectClassifierHead(nn.Module):
    """Classify defect type on cargo ROI crops: scratch / crack / dent / stain / none."""

    export = False

    def __init__(self, nc_defect: int = 5, roi_size: int = 128):
        super().__init__()
        self.nc_defect = nc_defect
        self.roi_size = roi_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, nc_defect)

    @staticmethod
    def crop_roi(image: torch.Tensor, box_xywh: torch.Tensor, pad: float = 0.05) -> torch.Tensor:
        """Crop a normalized xywh box from CHW image tensor."""
        _, h, w = image.shape
        box = xywh2xyxy(box_xywh.view(1, 4)).squeeze(0)
        bw, bh = box[2] - box[0], box[3] - box[1]
        box[0] = (box[0] - pad * bw).clamp(0, 1)
        box[1] = (box[1] - pad * bh).clamp(0, 1)
        box[2] = (box[2] + pad * bw).clamp(0, 1)
        box[3] = (box[3] + pad * bh).clamp(0, 1)
        x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h], device=image.device)).round().long().tolist()
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
        return image[:, y1:y2, x1:x2]

    def build_roi_batch(self, batch: dict[str, torch.Tensor], nc_cargo: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one ROI per image using the largest cargo GT box, else full frame."""
        images = batch["img"]
        device = images.device
        batch_idx = batch["batch_idx"].view(-1)
        cls = batch["cls"].view(-1).long()
        bboxes = batch["bboxes"]
        defect_labels = batch["defect_label"].view(-1).long()
        b = images.shape[0]
        rois = []
        targets = []
        for i in range(b):
            mask = (batch_idx == i) & (cls < nc_cargo)
            if mask.any():
                cargo_boxes = bboxes[mask]
                areas = cargo_boxes[:, 2] * cargo_boxes[:, 3]
                box = cargo_boxes[areas.argmax()]
                roi = self.crop_roi(images[i], box)
            else:
                roi = images[i]
            if roi.numel() == 0 or min(roi.shape[-2:]) < 2:
                roi = images[i]
            rois.append(
                F.interpolate(
                    roi.unsqueeze(0),
                    size=(self.roi_size, self.roi_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            )
            targets.append(defect_labels[i])
        if not rois:
            return torch.zeros(0, 3, self.roi_size, self.roi_size, device=device), torch.zeros(0, device=device)
        return torch.stack(rois, 0), torch.stack(targets, 0)

    def forward(self, batch: dict[str, torch.Tensor], nc_cargo: int) -> torch.Tensor:
        """Return defect logits for one ROI per image in the batch."""
        rois, _ = self.build_roi_batch(batch, nc_cargo)
        if rois.numel() == 0:
            return rois.new_zeros((0, self.nc_defect))
        feats = self.encoder(rois).flatten(1)
        return self.fc(feats)
