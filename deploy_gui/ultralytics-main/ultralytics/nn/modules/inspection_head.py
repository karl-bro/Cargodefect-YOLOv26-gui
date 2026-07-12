# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Split-task inspection head: YOLO detect (cargo only) + defect classifier + quality."""

from __future__ import annotations

import torch
import torch.nn as nn

from .defect_classifier import DefectClassifierHead
from .head import Detect
from .quality_head import QualityHead


class CargoInspectionHead(nn.Module):
    """Cargo detection + ROI defect classification + image-level quality (OK/NG)."""

    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    xyxy = False

    def __init__(
        self,
        nc: int,
        nc_defect: int,
        nc_quality: int,
        reg_max: int,
        end2end: bool,
        ch: tuple,
        use_defect_classifier: bool = True,
        use_quality_head: bool = True,
        quality_thresholds: dict | None = None,
        roi_size: int = 128,
    ):
        super().__init__()
        self.nc = nc
        self.nc_defect = nc_defect
        self.nc_quality = nc_quality
        self.reg_max = reg_max
        self._end2end = end2end
        self.use_defect_classifier = use_defect_classifier
        self.use_quality_head = use_quality_head

        self.detect = Detect(nc, reg_max, end2end, ch)
        self.defect_classifier = (
            DefectClassifierHead(nc_defect=nc_defect, roi_size=roi_size) if use_defect_classifier else None
        )
        self.quality = QualityHead(ch, nc_quality, quality_thresholds) if use_quality_head else None

    @property
    def end2end(self):
        return self._end2end and hasattr(self.detect, "one2one")

    @end2end.setter
    def end2end(self, value):
        self._end2end = value
        self.detect.end2end = value

    @property
    def stride(self):
        return self.detect.stride

    @stride.setter
    def stride(self, value):
        self.detect.stride = value

    @property
    def nl(self):
        return self.detect.nl

    @property
    def inplace(self):
        return self.detect.inplace

    @inplace.setter
    def inplace(self, value):
        self.detect.inplace = value

    def bias_init(self):
        self.detect.bias_init()

    def forward(self, x: list) -> dict | torch.Tensor | tuple:
        feats = x[0] if isinstance(x[0], list) else x
        detect_out = self.detect(feats)
        outputs = {"detect": detect_out, "feats": feats}

        if self.quality is not None:
            outputs["quality"] = self.quality(feats)

        if self.training:
            return outputs

        det_inference = detect_out[0] if isinstance(detect_out, tuple) else detect_out
        if self.export:
            return det_inference
        return det_inference, outputs

    def defect_logits(self, batch: dict[str, torch.Tensor], nc_cargo: int) -> torch.Tensor:
        if self.defect_classifier is None:
            return batch["img"].new_zeros((0, self.nc_defect))
        return self.defect_classifier(batch, nc_cargo)
