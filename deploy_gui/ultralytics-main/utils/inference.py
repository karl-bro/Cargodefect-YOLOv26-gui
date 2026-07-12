#!/usr/bin/env python3
"""Inference engine for defect detection with OK/NG rule."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import yaml


class DefectInferenceEngine:
    """Load model and run inference with OK/NG classification."""

    def __init__(self, config_path: str = "gui/config.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Resolve relative to project root (find by looking for ultralytics dir)
            for parent in [Path.cwd()] + list(Path.cwd().parents):
                if (parent / config_path).exists():
                    config_path = (parent / config_path).resolve()
                    break

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        weights_path = self.config["weights_path"]
        if not Path(weights_path).is_absolute():
            # Resolve relative to config file directory
            weights_path = str(Path(config_path).parent / weights_path)
        self.weights_path = Path(weights_path)
        self.package_conf = float(self.config.get("package_conf", 0.35))
        self.defect_conf = float(self.config.get("defect_conf", 0.15))
        self.iou_threshold = float(self.config.get("iou_threshold", 0.5))
        self.class_names = self.config.get("class_names", ["package", "defect"])

        self.model = None
        self._load_model()

    def _load_model(self):
        from ultralytics import YOLO
        self.model = YOLO(str(self.weights_path))

    def process_image(self, image: np.ndarray) -> Dict:
        if self.model is None:
            self._load_model()

        # Run inference at the lower threshold to capture all candidates
        min_conf = min(self.package_conf, self.defect_conf)
        t0 = time.perf_counter()
        results = self.model(image, conf=min_conf,
                             iou=self.iou_threshold, verbose=False)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        result = results[0]
        # Default annotated image (will redraw with per-class thresholds)
        annotated = image.copy()
        if annotated.shape[2] != 3:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        detections = []
        has_defect = False
        max_conf = 0.0
        cargo_class = "none"
        defect_status = "normal"

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                # Per-class confidence filtering
                if cls_id == 0 and conf < self.package_conf:
                    continue
                if cls_id == 1 and conf < self.defect_conf:
                    continue

                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"cls{cls_id}"
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in xyxy],
                })

                # Draw filtered bbox
                color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)
                pt1 = (int(xyxy[0]), int(xyxy[1]))
                pt2 = (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(annotated, pt1, pt2, color, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated, label, (pt1[0], pt1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if conf > max_conf:
                    max_conf = conf

                if cls_name == "package":
                    cargo_class = "package"
                elif cls_name == "defect":
                    has_defect = True
                    defect_status = "defect"

        quality_result = "NG" if has_defect else "OK"

        return {
            "cargo_class": cargo_class,
            "defect_status": defect_status,
            "quality_result": quality_result,
            "confidence": round(max_conf, 4),
            "inference_time_ms": round(inference_ms, 2),
            "detections": detections,
            "annotated_image": annotated,
        }

    def process_file(self, image_path: str) -> Dict:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.process_image(image)


def draw_results(image: np.ndarray, result: Dict) -> np.ndarray:
    """Draw inspection results on image (uses pre-annotated image from engine)."""
    h, w = image.shape[:2]
    annotated = cv2.cvtColor(result.get("annotated_image", image), cv2.COLOR_BGR2RGB) \
        if result.get("annotated_image") is not None else image.copy()
    if annotated.shape[:2] != (h, w):
        annotated = cv2.resize(annotated, (w, h))

    lines = [
        f"Result: {result['quality_result']}",
        f"Cargo: {result['cargo_class']} | Defect: {result['defect_status']}",
        f"Conf: {result['confidence']:.2f}",
        f"Time: {result['inference_time_ms']:.1f}ms",
    ]
    y0 = 30
    for i, line in enumerate(lines):
        color = (0, 0, 255) if result["quality_result"] == "NG" else (0, 255, 0)
        cv2.putText(annotated, line, (10, y0 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return annotated
