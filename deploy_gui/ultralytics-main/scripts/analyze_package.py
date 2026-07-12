#!/usr/bin/env python3
"""Analyze package class weaknesses: missed detections, label stats, augmentation suggestions."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

OUT = ROOT / "results/package_analysis"
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = ROOT / "runs/detect/runs/cargodefect/fusion_v3/weights/best.pt"
PACKAGE_CLASS_ID = 3  # 0=box, 1=bottle, 2=can, 3=package

CARGO_NAMES = ["box", "bottle", "can", "package"]


def main():
    from ultralytics.data import build_cargodefect_dataset, build_dataloader
    from ultralytics.data.cargodefect import resolve_cargodefect_data
    from ultralytics.data.utils import check_det_dataset
    from copy import deepcopy
    from ultralytics.utils import DEFAULT_CFG

    data_cfg = resolve_cargodefect_data(check_det_dataset("ultralytics/cfg/datasets/cargodefect-fusion.yaml"))

    # Cargo class stats from labels
    split = data_cfg["val"]
    args = deepcopy(DEFAULT_CFG)
    args.imgsz = 640
    args.mosaic = 0.0
    args.mixup = 0.0
    args.cutmix = 0.0
    args.copy_paste = 0.0
    args.fliplr = 0.0
    args.hsv_h = 0.0
    args.hsv_s = 0.0
    args.hsv_v = 0.0
    args.degrees = 0.0
    args.translate = 0.0
    args.scale = 0.0
    args.shear = 0.0
    args.perspective = 0.0
    args.auto_augment = None
    args.erasing = 0.0

    ds = build_cargodefect_dataset(args, split, 8, data_cfg, mode="val", stride=32)
    loader = build_dataloader(ds, batch=8, workers=4, shuffle=False)

    # Count package instances
    total_images = len(ds)
    package_img_count = 0
    package_box_count = 0
    class_counts = {i: 0 for i in range(4)}

    for batch in loader:
        for i in range(len(batch["im_file"])):
            cls = batch["cls"][batch["batch_idx"] == i] if "batch_idx" in batch else batch["cls"]
            if isinstance(cls, list):
                cls = cls[i]
            unique = np.unique(cls.cpu().numpy() if hasattr(cls, "cpu") else cls)
            if PACKAGE_CLASS_ID in unique:
                package_img_count += 1
            for c in cls.cpu().numpy().flatten() if hasattr(cls, "cpu") else cls.flatten():
                c_int = int(c)
                if c_int < 4:
                    class_counts[c_int] += 1

    # Inference for missed package analysis
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(WEIGHTS))
    model = yolo.model.eval().to(device)

    missed_samples = []
    sample_dir = OUT / "missed_package"
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    max_samples = 30

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            imgs = batch["img"].float() / 255.0
            preds = model.predict(imgs)
            det = preds[0] if isinstance(preds, tuple) else preds

            for i in range(imgs.shape[0]):
                gt_boxes = batch.get("bboxes", [])
                gt_cls = batch.get("cls", [])
                # Build GT package boxes for this image
                has_package = False
                if "batch_idx" in batch:
                    mask = batch["batch_idx"] == i
                    cls_i = batch["cls"][mask]
                    boxes_i = batch["bboxes"][mask]
                else:
                    cls_i = batch["cls"][i] if isinstance(batch["cls"], list) else batch["cls"]
                    boxes_i = batch["bboxes"][i] if isinstance(batch["bboxes"], list) else batch["bboxes"]

                package_mask = cls_i == PACKAGE_CLASS_ID
                n_gt_packages = int(package_mask.sum())

                if n_gt_packages > 0:
                    has_package = True
                    # Check predictions
                    pred_boxes = det.boxes if hasattr(det, "boxes") else None
                    n_det_packages = 0
                    if pred_boxes is not None and len(pred_boxes) > 0:
                        pred_cls = pred_boxes.cls.cpu().numpy()
                        n_det_packages = int((pred_cls == PACKAGE_CLASS_ID).sum())

                    if n_det_packages < n_gt_packages and saved < max_samples:
                        img = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        h, w = img.shape[:2]
                        # Draw GT boxes
                        for j in range(len(cls_i)):
                            if cls_i[j] == PACKAGE_CLASS_ID:
                                bx = boxes_i[j].cpu().numpy() * np.array([w, h, w, h])
                                x1, y1, x2, y2 = bx.astype(int)
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(img, "GT_package(missed)", (x1, max(y1 - 4, 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        # Draw predicted boxes
                        if pred_boxes is not None and len(pred_boxes) > 0:
                            for pb in pred_boxes:
                                if int(pb.cls) == PACKAGE_CLASS_ID:
                                    x1, y1, x2, y2 = pb.xyxy[0].cpu().numpy().astype(int)
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(img, f"GT={n_gt_packages} DET={n_det_packages}", (8, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                        cv2.imwrite(str(sample_dir / f"missed_{saved:03d}.jpg"), img)
                        saved += 1

                if has_package:
                    missed_samples.append({
                        "image": str(batch["im_file"][i]),
                        "n_gt": n_gt_packages,
                        "n_det": n_det_packages if "n_det_packages" in dir() else 0,
                    })

    # Report
    report_lines = [
        "# Package Class Weak Analysis",
        "",
        f"- Weights: `{WEIGHTS}`",
        f"- Total val images: {total_images}",
        "",
        "## Category Distribution (val set)",
        f"- Box instances: {class_counts[0]}",
        f"- Bottle instances: {class_counts[1]}",
        f"- Can instances: {class_counts[2]}",
        f"- **Package instances: {class_counts[3]}**",
        f"- Images with packages: {package_img_count} / {total_images} ({100*package_img_count/max(total_images,1):.1f}%)",
        "",
        "## Missed Package Detection Samples",
        f"- Saved to `{sample_dir}/` ({saved} samples, GT=red, Pred=green)",
        "",
        "## Analysis & Recommendations",
        "- Package is the most challenging class: similar appearance to background, variable box stacking.",
        "- Augmentation suggestions:",
        "  1. Increase mosaic + mixup for package-heavy batches",
        "  2. Add copy_paste augmentation with package crops",
        "  3. Consider class-specific oversampling in dataloader",
        "- Label check: verify bounding box tightness on package annotations.",
        "",
        "## Next Steps",
        "- Run `scripts/check_package_labels.py` to audit annotation quality",
        "- Try class-balanced sampling or focal loss on detection head",
    ]
    (OUT / "package_analysis.md").write_text("\n".join(report_lines), encoding="utf-8")

    # CSV with per-image stats
    csv_path = OUT / "package_detection.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_gt", "n_det"])
        w.writeheader()
        for s in missed_samples:
            w.writerow(s)

    print(f"\n".join(report_lines))
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
