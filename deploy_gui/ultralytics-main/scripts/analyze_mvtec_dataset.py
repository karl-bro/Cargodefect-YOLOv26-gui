#!/usr/bin/env python3
"""
MVTec LOCO AD dataset scanner and YOLO label generator.

Thoroughly analyzes the dataset structure, maps anomaly types correctly,
converts ground-truth masks to bounding boxes, and generates YOLO-format labels.

Key findings to correct:
- Old code heuristically mapped logical_anomalies→stain, structural_anomalies→scratch
- This was WRONG. MVtec LOCO AD only has two anomaly types: "logical" and "structural"
- There are NO scratch/crack/dent/stain labels in this dataset
- We unify to a single "defect" class for detection
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MVtec_ROOT = Path("/home/swot2486/0701/MVTec LOCO AD")
OUT_DIR = ROOT / "results/mvtec_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_DIR = ROOT / "datasets/cargodefect_mvtec/labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_LINK_DIR = ROOT / "datasets/cargodefect_mvtec/images"
IMAGES_LINK_DIR.mkdir(parents=True, exist_ok=True)

# Map "good" to OK (0), any anomaly to NG (1)
# For detection: class 0-3 = cargo, class 4 = defect
DEFECT_CLASS_ID = 4  # global defect class ID

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag"]
SPLITS = ["train", "validation", "test"]
ANOMALY_TYPES = ["logical_anomalies", "structural_anomalies"]


def mask_to_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Convert binary mask to list of (x1, y1, x2, y2) bounding boxes."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:  # filter tiny regions
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))
    # If no contours, use the full nonzero bounding box
    if not bboxes:
        ys, xs = np.nonzero(binary)
        if len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                bboxes.append((x1, y1, x2, y2))
    return bboxes


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h) -> tuple[float, float, float, float]:
    """Convert pixel bbox to YOLO normalized format (x_center, y_center, width, height)."""
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    return cx, cy, w, h


def main():
    stats = {
        "categories": {},
        "total_train": 0, "total_val": 0, "total_test": 0,
        "total_logical": 0, "total_structural": 0,
        "total_masks": 0, "total_bboxes": 0, "total_masked_images": 0,
    }
    per_anomaly_stats = {
        "logical_anomalies": {"samples": 0, "has_mask": 0, "missing_mask": 0, "bboxes": 0},
        "structural_anomalies": {"samples": 0, "has_mask": 0, "missing_mask": 0, "bboxes": 0},
    }
    anomaly_samples = []  # list of dicts for detailed CSV

    for cat in CATEGORIES:
        cat_dir = MVtec_ROOT / cat / cat
        if not cat_dir.exists():
            print(f"WARNING: {cat_dir} not found, skipping")
            continue

        cat_stat = {
            "train_good": 0, "val_good": 0, "test_good": 0,
            "logical_anomalies": 0, "structural_anomalies": 0,
            "logical_has_mask": 0, "structural_has_mask": 0,
            "logical_missing_mask": 0, "structural_missing_mask": 0,
            "logical_bboxes": 0, "structural_bboxes": 0,
        }

        # count "good" samples
        for split_name, split_key in [("train", "train"), ("validation", "val"), ("test", "test")]:
            good_dir = cat_dir / split_name / "good"
            if good_dir.exists():
                n = len([f for f in os.listdir(good_dir) if f.endswith(".png")])
                cat_stat[f"{split_key}_good"] = n
                if split_key == "test":
                    stats["total_test"] += n
                elif split_key == "val":
                    stats["total_val"] += n
                elif split_key == "train":
                    stats["total_train"] += n

        # Process each anomaly type
        for anom_type in ANOMALY_TYPES:
            img_dir = cat_dir / "test" / anom_type
            gt_dir = cat_dir / "ground_truth" / anom_type
            if not img_dir.exists():
                continue

            anom_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            cat_stat[anom_type] = len(anom_images)
            per_anomaly_stats[anom_type]["samples"] += len(anom_images)
            if anom_type == "logical_anomalies":
                stats["total_logical"] += len(anom_images)
            else:
                stats["total_structural"] += len(anom_images)

            for img_name in anom_images:
                img_path = img_dir / img_name
                stem = Path(img_name).stem  # e.g. "000"
                mask_dir = gt_dir / stem
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                has_mask = False
                yolo_lines = []
                num_bboxes = 0

                if mask_dir.exists():
                    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
                    if mask_files:
                        has_mask = True
                        stats["total_masks"] += len(mask_files)
                        stats["total_masked_images"] += 1
                        # Combine all masks for this image
                        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        for mf in mask_files:
                            mask = cv2.imread(str(mask_dir / mf), cv2.IMREAD_GRAYSCALE)
                            if mask is not None:
                                combined_mask = np.maximum(combined_mask, mask)
                        bboxes = mask_to_bboxes(combined_mask)
                        num_bboxes = len(bboxes)
                        stats["total_bboxes"] += num_bboxes

                        for (x1, y1, x2, y2) in bboxes:
                            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                            yolo_lines.append(f"{DEFECT_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                # Write YOLO label
                rel_label_path = f"{cat}/test_{anom_type}/{stem}.txt"
                label_path = LABELS_DIR / rel_label_path
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

                if has_mask:
                    if anom_type == "logical_anomalies":
                        cat_stat["logical_has_mask"] += 1
                        cat_stat["logical_bboxes"] += num_bboxes
                    else:
                        cat_stat["structural_has_mask"] += 1
                        cat_stat["structural_bboxes"] += num_bboxes
                else:
                    if anom_type == "logical_anomalies":
                        cat_stat["logical_missing_mask"] += 1
                    else:
                        cat_stat["structural_missing_mask"] += 1

                anomaly_samples.append({
                    "category": cat,
                    "anomaly_type": anom_type.replace("_anomalies", ""),
                    "image": str(img_path),
                    "image_stem": stem,
                    "has_mask": has_mask,
                    "num_masks": len(mask_files) if has_mask else 0,
                    "num_bboxes": num_bboxes,
                    "img_w": img_w,
                    "img_h": img_h,
                })

        # Update per-anomaly aggregate stats
        for at in ANOMALY_TYPES:
            per_anomaly_stats[at]["has_mask"] += cat_stat.get(f"{at.split('_')[0]}_has_mask", 0)
            per_anomaly_stats[at]["missing_mask"] += cat_stat.get(f"{at.split('_')[0]}_missing_mask", 0)
            per_anomaly_stats[at]["bboxes"] += cat_stat.get(f"{at.split('_')[0]}_bboxes", 0)

        stats["categories"][cat] = cat_stat

    # --- REPORT ---
    report_lines = [
        "# MVTec LOCO AD Dataset Analysis Report",
        "",
        "## 1. Dataset Structure",
        "",
        f"- Root: `{MVtec_ROOT}`",
        f"- Categories: {len(CATEGORIES)}  ({', '.join(CATEGORIES)})",
        f"- Anomaly types: **logical_anomalies** and **structural_anomalies** only",
        "",
        "### Important: No scratch/crack/dent/stain labels exist",
        "- Old heuristics mapping logical→stain, structural→scratch were **incorrect**",
        "- Correct mapping: ALL anomalies → `defect` (class_id=4)",
        "",
        "## 2. Sample Counts",
        "",
        "| Category | train/good | val/good | test/good | logical_ano | structural_ano | logical_mask | structural_mask | logical_bbox | structural_bbox |",
        "|----------|-----------|---------|-----------|-------------|----------------|-------------|----------------|-------------|----------------|",
    ]

    for cat in CATEGORIES:
        cs = stats["categories"].get(cat, {})
        report_lines.append(
            f"| {cat} | {cs.get('train_good', 0)} | {cs.get('val_good', 0)} | {cs.get('test_good', 0)} "
            f"| {cs.get('logical_anomalies', 0)} | {cs.get('structural_anomalies', 0)} "
            f"| {cs.get('logical_has_mask', 0)} | {cs.get('structural_has_mask', 0)} "
            f"| {cs.get('logical_bboxes', 0)} | {cs.get('structural_bboxes', 0)} |"
        )

    total_anomalies = stats["total_logical"] + stats["total_structural"]
    total_good = stats["total_train"] + stats["total_val"] + stats["total_test"]
    report_lines += [
        f"| **Total** | {stats['total_train']} | {stats['total_val']} | {stats['total_test']} "
        f"| {stats['total_logical']} | {stats['total_structural']} "
        f"| -- | -- | {stats['total_bboxes']} | -- |",
        "",
        "## 3. Mask Coverage",
        "",
        f"- Total anomaly images: **{total_anomalies}**",
        f"- Images with ground-truth masks: **{stats['total_masked_images']}**",
        f"- Total mask files: **{stats['total_masks']}**",
        f"- Total bounding boxes generated: **{stats['total_bboxes']}**",
        "",
        "| Anomaly Type | Samples | Has Mask | Missing Mask | Masks | BBoxes |",
        "|--------------|---------|----------|-------------|-------|--------|",
    ]
    for at, pas in per_anomaly_stats.items():
        report_lines.append(f"| {at} | {pas['samples']} | {pas['has_mask']} | {pas['missing_mask']} | "
                            f"{pas['has_mask']} | {pas['bboxes']} |")

    report_lines += [
        "",
        "## 4. Quality Classification Summary (OK/NG)",
        "",
        f"- OK samples (all good): {total_good}",
        f"- NG samples (all anomalies): {total_anomalies}",
        f"- Total: {total_good + total_anomalies}",
        "",
        "## 5. Generated Labels",
        "",
        f"- YOLO labels: `{LABELS_DIR}`",
        f"  - Format: class x_center y_center width height (all normalized)",
        f"  - class=4 = defect (unified)",
        "",
        "## 6. Recommendations",
        "",
        "1. **Detection**: Use class 0-3 for cargo + class 4 for unified defect",
        "2. **Quality head**: binary OK (0) / NG (1) — no fine-grained anomaly types",
        "3. **Defect classifier**: binary none (0) / defect (1) — no scratch/crack/dent/stain",
        "4. **Training strategy**: pseudo-bboxes from mask conversion give weak supervision for defect localization",
        "",
        "## 7. Artifacts",
        f"- Detailed CSV: `{OUT_DIR / 'anomaly_samples.csv'}`",
        f"- JSON stats: `{OUT_DIR / 'mvtec_stats.json'}`",
        f"- Full report: `{OUT_DIR / 'mvtec_analysis_report.md'}`",
    ]

    report_path = OUT_DIR / "mvtec_analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # JSON
    serializable_stats = {
        "total_train": stats["total_train"],
        "total_val": stats["total_val"],
        "total_test": stats["total_test"],
        "total_good": total_good,
        "total_logical": stats["total_logical"],
        "total_structural": stats["total_structural"],
        "total_anomalies": total_anomalies,
        "total_masked_images": stats["total_masked_images"],
        "total_masks": stats["total_masks"],
        "total_bboxes": stats["total_bboxes"],
        "categories": stats["categories"],
        "per_anomaly_type": per_anomaly_stats,
    }
    (OUT_DIR / "mvtec_stats.json").write_text(json.dumps(serializable_stats, indent=2), encoding="utf-8")

    # CSV
    csv_path = OUT_DIR / "anomaly_samples.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "anomaly_type", "image_stem", "has_mask", "num_masks", "num_bboxes", "img_w", "img_h"])
        w.writeheader()
        for s in anomaly_samples:
            del s["image"]
            w.writerow(s)

    # Print summary
    print(f"Report: {report_path}")
    print(f"CSV:    {csv_path}")
    print(f"JSON:   {OUT_DIR / 'mvtec_stats.json'}")
    print(f"Labels: {LABELS_DIR}")
    print()
    print(f"Summary: {total_good} good + {total_anomalies} anomaly = {total_good + total_anomalies} total")
    print(f"  Logical:  {stats['total_logical']}  (masks: {per_anomaly_stats['logical_anomalies']['has_mask']})")
    print(f"  Structural: {stats['total_structural']}  (masks: {per_anomaly_stats['structural_anomalies']['has_mask']})")
    print(f"  BBoxes generated: {stats['total_bboxes']}")


if __name__ == "__main__":
    main()
