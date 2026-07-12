#!/usr/bin/env python3
"""Comprehensive label diagnostic script for cargodefect_package dataset."""
from __future__ import annotations
import cv2, csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets/cargodefect_package"
OUT = ROOT / "results/package_label_debug"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "visualizations").mkdir(exist_ok=True)

def read_labels(lbl_path):
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                boxes.append({"error": f"short_line: {len(parts)} cols"})
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                boxes.append({"error": f"parse: {line[:50]}"})
                continue
            boxes.append({"cls": cls_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes

def validate_boxes(boxes):
    issues = []
    for i, b in enumerate(boxes):
        if "error" in b:
            issues.append(f"box[{i}]: {b['error']}")
            continue
        cls_id = b["cls"]
        if cls_id not in (0, 1):
            issues.append(f"box[{i}]: bad class_id={cls_id}")
        cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
        if not (0 <= cx <= 1): issues.append(f"box[{i}]: cx={cx} out of [0,1]")
        if not (0 <= cy <= 1): issues.append(f"box[{i}]: cy={cy} out of [0,1]")
        if not (0 < w <= 1):   issues.append(f"box[{i}]: w={w} not in (0,1]")
        if not (0 < h <= 1):   issues.append(f"box[{i}]: h={h} not in (0,1]")
        if w > 1:  issues.append(f"box[{i}]: w={w} > 1")
        if h > 1:  issues.append(f"box[{i}]: h={h} > 1")
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        if x1 < -0.001: issues.append(f"box[{i}]: x1={x1:.4f} < 0")
        if y1 < -0.001: issues.append(f"box[{i}]: y1={y1:.4f} < 0")
        if x1 >= x2:    issues.append(f"box[{i}]: x1>=x2 ({x1}>= {x2})")
        if y1 >= y2:    issues.append(f"box[{i}]: y1>=y2 ({y1}>= {y2})")
    return issues

# ============ COUNT PER CLASS ============
print("=== CLASS DISTRIBUTION ===")
all_counts = Counter()
per_split = {}
for split in ["train", "val"]:
    lbl_dir = DS / "labels" / split
    counts = Counter()
    empty_files = 0
    bad_files = 0
    total_issue_files = 0
    all_bad = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        boxes = read_labels(lbl)
        if not boxes:
            empty_files += 1
        has_issue = False
        for b in boxes:
            if "error" in b:
                bad_files += 1
                has_issue = True
                break
            cls_id = b.get("cls", -1)
            counts[cls_id] += 1
            all_counts[cls_id] += 1
        issues = validate_boxes(boxes)
        if issues:
            total_issue_files += 1
            all_bad.append((str(lbl), issues))
    per_split[split] = dict(counts)
    print(f"{split}: labels={len(list(lbl_dir.glob('*.txt')))} | class_dist={dict(counts)} | empty={empty_files} | bad_parse={bad_files} | bounds_issue={total_issue_files}")
    if all_bad[:5]:
        print(f"  First 5 bad files:")
        for fname, isss in all_bad[:5]:
            print(f"    {fname}: {isss[:2]}...")
    total_issues = len(all_bad)
    print(f"  Total files with bounds issues: {total_issues}")

# Save distribution CSV
dist_csv = OUT / "label_distribution.csv"
with open(dist_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class_id", "class_name", "train", "val", "total"])
    names = {0: "package", 1: "defect"}
    for cls_id in sorted(all_counts.keys()):
        w.writerow([cls_id, names.get(cls_id, f"unknown_{cls_id}"),
                    per_split["train"].get(cls_id, 0),
                    per_split["val"].get(cls_id, 0),
                    all_counts[cls_id]])
print(f"\nDistribution CSV: {dist_csv}")

# ============ SAMPLE BBOX VISUALIZATION ============
print("\n=== VISUALIZING 100 TRAIN LABELS ===")
import random
random.seed(42)
lbls = list((DS / "labels" / "train").glob("*.txt"))
random.shuffle(lbls)
vis_n = min(100, len(lbls))

colors = {0: (0, 255, 0), 1: (0, 0, 255)}
names_cls = {0: "package", 1: "defect"}
vis_stats = Counter()
vis_issues = []

for i, lbl in enumerate(lbls[:vis_n]):
    stem = lbl.stem
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = DS / "images" / "train" / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    if not img_path:
        vis_issues.append(f"{stem}: image not found")
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        vis_issues.append(f"{stem}: image read failed")
        continue
    boxes = read_labels(lbl)
    for b in boxes:
        if "error" in b: continue
        cls_id = b["cls"]
        vis_stats[cls_id] += 1
        cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
        ih, iw = img.shape[:2]
        x1 = int((cx - w/2) * iw)
        y1 = int((cy - h/2) * ih)
        x2 = int((cx + w/2) * iw)
        y2 = int((cy + h/2) * ih)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(iw-1, x2); y2 = min(ih-1, y2)
        color = colors.get(cls_id, (128,128,128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, names_cls.get(cls_id, f"cls{cls_id}"), (x1, max(y1-4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    out_path = OUT / "visualizations" / f"{stem}.jpg"
    cv2.imwrite(str(out_path), img)

print(f"Vis: {vis_n} images → {OUT}/visualizations/")
print(f"  Boxes drawn: {dict(vis_stats)}")
if vis_issues:
    print(f"  Issues: {len(vis_issues)} images")
    for iss in vis_issues[:5]:
        print(f"    {iss}")

# ============ VERIFY KAGGLE XML CONVERSION ============
print("\n=== KAGGLE XML CHECK ===")
import xml.etree.ElementTree as ET
kaggle_dir = Path("/home/swot2486/0701/Images with cardboard defects (dents, holes, dirt)")
kaggle_objs = Counter()
kaggle_sizes = []
for xml_path in sorted(kaggle_dir.glob("*.xml"))[:20]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    kaggle_sizes.append((w, h))
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        kaggle_objs[name] += 1
print(f"Kaggle objects (first 20 XMLs): {dict(kaggle_objs)}")
print(f"Kaggle image sizes (first 20): {kaggle_sizes[:5]} min/max w={min(s[0] for s in kaggle_sizes)}-{max(s[0] for s in kaggle_sizes)} h={min(s[1] for s in kaggle_sizes)}-{max(s[1] for s in kaggle_sizes)}")

# Check a converted Kaggle label
print("\nSample Kaggle→YOLO labels:")
for lbl in sorted((DS / "labels" / "train").glob("kaggle_*"))[:3]:
    boxes = read_labels(lbl)
    print(f"  {lbl.name}: {len(boxes)} boxes -> {[{'cls':b.get('cls'), 'w':b.get('w','?'), 'h':b.get('h','?')} for b in boxes if 'error' not in b]}")

# ============ REPORT ============
report = [
    "# Package Label Debug Report",
    "",
    "## Class Distribution",
    "| class_id | class | train | val | total |",
    "|---|---|---|---|---|",
]
for cls_id in sorted(all_counts.keys()):
    report.append(f"| {cls_id} | {names.get(cls_id, '?')} | {per_split['train'].get(cls_id,0)} | {per_split['val'].get(cls_id,0)} | {all_counts[cls_id]} |")

total_train_labels = len(list((DS/"labels"/"train").glob("*.txt")))
total_val_labels = len(list((DS/"labels"/"val").glob("*.txt")))
report += [
    "",
    "## Label Files",
    f"- Train: {total_train_labels} files",
    f"- Val: {total_val_labels} files",
    f"- Class 0 (package): {all_counts[0]} bboxes",
    f"- Class 1 (defect): {all_counts[1]} bboxes",
    f"- Other IDs: {sum(v for k,v in all_counts.items() if k not in (0,1))}",
    "",
    "## Kaggle Verification",
    f"- Objects found: {dict(kaggle_objs)}",
    f"- Image sizes: W={kaggle_sizes[0][0] if kaggle_sizes else '?'} × H={kaggle_sizes[0][1] if kaggle_sizes else '?'}",
    f"- Mapping: dent/dirt/hole → class 1 (defect)",
    "",
    "## Roboflow Verification",
    "- Cardboard Box Defect: no defect(0)→package, torn(1)→defect, wrinkle(2)→defect",
    "- Corrugated Box Defect: No Defect(0)→package, Torn(1)→defect, Wrinkle(2)→defect",
    "",
    "## Recommendations",
    "1. Full-image pseudo bbox for package dominates → defect model ignores defect class",
    "2. Need defect-only dataset variant (nc=1) for baseline verification",
    "3. Need package+defect WITHOUT pseudo bbox (real boxes only)",
]
(OUT / "label_visualization_report.md").write_text("\n".join(report), encoding="utf-8")
print(f"\nReport: {OUT / 'label_visualization_report.md'}")
print("DONE")
