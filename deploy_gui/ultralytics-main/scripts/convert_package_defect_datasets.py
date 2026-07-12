#!/usr/bin/env python3
"""
Convert 3 cardboard defect datasets into unified cargodefect_package format.

CRITICAL FIX: USE_FULL_IMAGE_PACKAGE_BBOX = False by default.
The pseudo full-image bbox (0 0.5 0.5 1.0 1.0) causes the model to assign
every anchor to class-0 (package), drowning the defect class signal.
When package bbox is needed, only source-provided "no defect" bboxes are used.

Output:
  datasets/cargodefect_package/          (package+defect, nc=2)
  datasets/cargodefect_package_defect/   (defect-only, nc=1)

Datasets:
  1. Cardboard Box Defect (Roboflow YOLO) — no defect/torn/wrinkle
  2. Corrugated Box Defect (Roboflow YOLO) — No Defect/Torn/Wrinkle
  3. Kaggle cardboard defects (Pascal VOC XML) — dent/hole/dirt
"""
from __future__ import annotations

import csv, os, shutil, sys, yaml
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─── CONFIG ──────────────────────────────────────────────
USE_BINARY_DEFECT = True
USE_FULL_IMAGE_PACKAGE_BBOX = False  # CRITICAL FIX: was True, now False
OUT_DIR = ROOT / "datasets/cargodefect_package"
DEFECT_ONLY_DIR = ROOT / "datasets/cargodefect_package_defect"

DS1 = Path("/home/swot2486/0701/Cardboard Box Defect.v2i.yolo26")
DS2 = Path("/home/swot2486/0701/Corrugated Box Defect.v14-80-20.yolo26")
DS3 = Path("/home/swot2486/0701/Images with cardboard defects (dents, holes, dirt)")

# ─── CLASS MAPPING ──────────────────────────────────────
DEFECT_BINARY_MAP = {"torn": 1, "wrinkle": 1, "dent": 1, "hole": 1, "dirt": 1}

PACKAGE_CLASS = 0
DEFECT_CLASS = 1

DEFECT_CLASS_NAMES = {0: "package", 1: "defect"}
DEFECT_ONLY_NAMES = {0: "defect"}

def map_defect(name: str) -> int:
    name = name.lower()
    return DEFECT_BINARY_MAP.get(name, -1)

# ─── Helpers ─────────────────────────────────────────────
def yolo_box_to_wh(yolo_line: str):
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return cls_id, cx, cy, w, h

def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        objs.append((name, xmin, ymin, xmax, ymax))
    return objs, w, h

def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return cx, cy, bw, bh

# ─── MAIN ───────────────────────────────────────────────
def main():
    # Clear outputs
    for d in [OUT_DIR, DEFECT_ONLY_DIR]:
        if d.exists():
            shutil.rmtree(d)
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (d / sub).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    label_dist = defaultdict(Counter)   # train/val -> class distribution
    defect_only_dist = defaultdict(Counter)
    quality_records = []
    train_idx = val_idx = 0
    log_lines = ["# Dataset Conversion Report", ""]

    # == Track per-image class presence for reporting ==
    only_pkg = 0
    only_def = 0
    both_cls = 0
    neither_cls = 0

    def write_outputs(target_split, new_name, defect_lines, img_path):
        """Write package+defect and defect-only labels."""
        # --- Package+Defect ---
        lbl_out = OUT_DIR / f"labels/{target_split}/{new_name}.txt"
        with open(lbl_out, "w") as f:
            for line in defect_lines:
                f.write(line + "\n")

        # --- Defect Only ---
        def_only_lines = []
        for line in defect_lines:
            if line.startswith("1 "):  # only keep defect class
                parts = line.strip().split()
                # renumber: class 1 -> class 0
                def_only_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        def_lbl_out = DEFECT_ONLY_DIR / f"labels/{target_split}/{new_name}.txt"
        with open(def_lbl_out, "w") as f:
            for line in def_only_lines:
                f.write(line + "\n")

        # --- Copy images to both ---
        for out_dir in [OUT_DIR, DEFECT_ONLY_DIR]:
            shutil.copy2(img_path, out_dir / f"images/{target_split}/{new_name}{img_path.suffix}")

    def process_roboflow(ds_path, split_map, desc, cls0_is_package=True):
        nonlocal train_idx, val_idx, only_pkg, only_def, both_cls, neither_cls
        n_imgs = 0

        for split_name, (img_sub, lbl_sub) in split_map.items():
            img_dir = ds_path / img_sub
            lbl_dir = ds_path / lbl_sub
            target_split = "train" if "train" in split_name else "val"

            if not img_dir.exists():
                log_lines.append(f"  WARNING: {img_dir} not found")
                continue

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                n_imgs += 1

                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                defect_lines = []
                has_package_bbox = False
                has_defect_bbox = False

                if lbl_path.exists():
                    for line in lbl_path.read_text().strip().splitlines():
                        parsed = yolo_box_to_wh(line)
                        if parsed is None:
                            continue
                        cls_id, cx, cy, w, h = parsed
                        if cls_id == 0 and cls0_is_package:
                            # "no defect"/"No Defect" bbox → package bbox
                            defect_lines.append(f"{PACKAGE_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                            has_package_bbox = True
                            label_dist[target_split]["package_real"] += 1
                        elif cls_id >= 1:
                            defect_name = {1: "torn", 2: "wrinkle"}.get(cls_id, "unknown")
                            new_cls = map_defect(defect_name)
                            if new_cls >= 0:
                                defect_lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                                has_defect_bbox = True
                                label_dist[target_split][f"defect_{defect_name}"] += 1
                            else:
                                log_lines.append(f"  WARNING: unknown class {cls_id} in {lbl_path}")
                        else:
                            log_lines.append(f"  WARNING: unexpected cls_id={cls_id} in {lbl_path}")

                # NO PSEUDO BBOX: only use source-provided "no defect" bbox for package
                # If no package bbox from source, don't add one.
                # This allows defect-only images to exist.

                if target_split == "train":
                    train_idx += 1
                    new_id = train_idx
                else:
                    val_idx += 1
                    new_id = val_idx

                new_name = f"{desc}_{new_id:05d}"

                # Track class presence
                if has_package_bbox and has_defect_bbox:
                    both_cls += 1
                elif has_package_bbox and not has_defect_bbox:
                    only_pkg += 1
                elif not has_package_bbox and has_defect_bbox:
                    only_def += 1
                else:
                    neither_cls += 1

                write_outputs(target_split, new_name, defect_lines, img_path)

                # Quality label
                quality = 1 if has_defect_bbox else 0
                quality_records.append({
                    "split": target_split,
                    "image": f"{new_name}{img_path.suffix}",
                    "quality": quality,
                })
                stats[f"{target_split}_total"] += 1
                if quality == 1:
                    stats[f"{target_split}_NG"] += 1
                else:
                    stats[f"{target_split}_OK"] += 1

                if not has_defect_bbox:
                    label_dist[target_split]["defect_none"] += 1

        log_lines.append(f"  {desc}: {n_imgs} images")
        return n_imgs

    def process_kaggle(ds_path, split="train"):
        nonlocal train_idx, val_idx, only_pkg, only_def, both_cls, neither_cls
        n_imgs = 0
        target_split = split

        for xml_path in sorted(ds_path.glob("*.xml")):
            img_path = ds_path / f"{xml_path.stem}.jpg"
            if not img_path.exists():
                img_path = ds_path / f"{xml_path.stem}.jpeg"
            if not img_path.exists():
                img_path = ds_path / f"{xml_path.stem}.png"
            if not img_path.exists():
                log_lines.append(f"  WARNING: image not found for {xml_path.name}")
                continue
            n_imgs += 1

            objs, img_w, img_h = parse_voc_xml(xml_path)
            defect_lines = []
            has_defect_bbox = False

            for name, xmin, ymin, xmax, ymax in objs:
                new_cls = map_defect(name.lower())
                if new_cls >= 0:
                    cx, cy, bw, bh = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                    defect_lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    has_defect_bbox = True
                    label_dist[target_split][f"defect_{name.lower()}"] += 1
                else:
                    log_lines.append(f"  WARNING: unknown class {name} in {xml_path}")

            # NO PSEUDO BBOX for Kaggle either.
            # Kaggle only has defect annotations (dent, hole, dirt) — no "no defect"
            # So all Kaggle images become defect-only (only class-1, no class-0)

            if target_split == "train":
                train_idx += 1
                new_id = train_idx
            else:
                val_idx += 1
                new_id = val_idx

            new_name = f"kaggle_{new_id:05d}"

            # Kaggle images have no package bbox → defect-only
            only_def += 1

            write_outputs(target_split, new_name, defect_lines, img_path)

            # Quality
            quality = 1  # All Kaggle images are NG (have defects)
            quality_records.append({
                "split": target_split,
                "image": f"{new_name}{img_path.suffix}",
                "quality": quality,
            })
            stats[f"{target_split}_total"] += 1
            stats[f"{target_split}_NG"] += 1

        log_lines.append(f"  kaggle {split}: {n_imgs} images")
        return n_imgs

    # ─── Process all datasets ────────────────────────────
    log_lines.append("## Cardboard Box Defect (Roboflow)")
    process_roboflow(DS1, {
        "train": ("train/images", "train/labels"),
        "val": ("valid/images", "valid/labels"),
        "test": ("test/images", "test/labels"),
    }, desc="cb1")

    log_lines.append("\n## Corrugated Box Defect (Roboflow)")
    process_roboflow(DS2, {
        "train": ("train/images", "train/labels"),
        "val": ("valid/images", "valid/labels"),
    }, desc="cb2")

    log_lines.append("\n## Kaggle Cardboard Defects (Pascal VOC)")
    process_kaggle(DS3)

    # ─── Generate quality CSVs ──────────────────────────
    for split in ["train", "val"]:
        recs = [r for r in quality_records if r["split"] == split]
        csv_path = OUT_DIR / f"quality_labels_{split}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "quality_label"])
            for r in recs:
                w.writerow([r["image"], r["quality"]])

    # ─── Generate label_distribution.csv ─────────────────
    all_classes = sorted(set().union(
        *[label_dist[s].keys() for s in label_dist]
    ))
    dist_path = OUT_DIR / "label_distribution.csv"
    with open(dist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "val", "total"])
        for cls_name in all_classes:
            tr = label_dist["train"].get(cls_name, 0)
            vl = label_dist["val"].get(cls_name, 0)
            w.writerow([cls_name, tr, vl, tr + vl])

    # ─── Generate data.yaml for BOTH datasets ────────────
    for out_dir, nc, names_dict, yaml_name in [
        (OUT_DIR, 2, DEFECT_CLASS_NAMES, "cargodefect-package.yaml"),
        (DEFECT_ONLY_DIR, 1, DEFECT_ONLY_NAMES, "cargodefect-defect-only.yaml"),
    ]:
        data_yaml = {
            "path": str(out_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": nc,
            "names": [names_dict[i] for i in range(nc)],
        }
        with open(out_dir / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

        # Copy to ultralytics cfg
        cfg_dir = ROOT / "ultralytics/cfg/datasets"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / yaml_name, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    # Also write quality-aware yaml for cargodefect training
    quality_yaml = {
        "path": str(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["package", "defect"],
        "quality_names": {0: "OK", 1: "NG"},
    }
    with open(OUT_DIR / "data_quality.yaml", "w") as f:
        yaml.dump(quality_yaml, f, default_flow_style=False, allow_unicode=True)

    # Count labels per class from actual label files
    def count_bboxes(label_dir):
        counts = Counter()
        for lbl in sorted(label_dir.glob("*.txt")):
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
        return counts

    pkg_counts_train = count_bboxes(OUT_DIR / "labels/train")
    pkg_counts_val = count_bboxes(OUT_DIR / "labels/val")
    def_counts_train = count_bboxes(DEFECT_ONLY_DIR / "labels/train")
    def_counts_val = count_bboxes(DEFECT_ONLY_DIR / "labels/val")

    # ─── Report ──────────────────────────────────────────
    total_train_imgs = len(list((OUT_DIR / "images/train").iterdir()))
    total_val_imgs = len(list((OUT_DIR / "images/val").iterdir()))

    log_lines += [
        "",
        "## Package+Defect Dataset (nc=2)",
        f"- Path: {OUT_DIR}",
        f"- Train: {total_train_imgs} images",
        f"- Val:   {total_val_imgs} images",
        f"- Class 0 (package): train={pkg_counts_train.get(0,0)}, val={pkg_counts_val.get(0,0)}",
        f"- Class 1 (defect):  train={pkg_counts_train.get(1,0)}, val={pkg_counts_val.get(1,0)}",
        f"- OK (no defect): {stats.get('train_OK',0)+stats.get('val_OK',0)}",
        f"- NG (has defect): {stats.get('train_NG',0)+stats.get('val_NG',0)}",
        "",
        "### Images per class combination",
        f"- Only package:  {only_pkg}",
        f"- Only defect:   {only_def}",
        f"- Both:          {both_cls}",
        f"- Neither:       {neither_cls}",
        "",
        "## Defect-Only Dataset (nc=1)",
        f"- Path: {DEFECT_ONLY_DIR}",
        f"- Train: {total_train_imgs} images (class-0 bboxes: {def_counts_train.get(0,0)})",
        f"- Val:   {total_val_imgs} images (class-0 bboxes: {def_counts_val.get(0,0)})",
        "",
        "## Configuration",
        f"- USE_BINARY_DEFECT: {USE_BINARY_DEFECT}",
        f"- USE_FULL_IMAGE_PACKAGE_BBOX: {USE_FULL_IMAGE_PACKAGE_BBOX}  (FIX: was True, now False)",
        f"- Source class mapping: torn/wrinkle/dent/hole/dirt → defect (class 1)",
        f"- 'no defect' / 'No Defect' → package (class 0)",
    ]

    report_path = OUT_DIR / "convert_report.md"
    report_path.write_text("\n".join(log_lines))
    print(f"\nConversion complete!")
    print(f"  Package+Defect (nc=2):      {OUT_DIR}")
    print(f"    Train: {total_train_imgs} imgs, class_dist={dict(pkg_counts_train)}")
    print(f"    Val:   {total_val_imgs} imgs, class_dist={dict(pkg_counts_val)}")
    print(f"    Only_pkg={only_pkg}, Only_def={only_def}, Both={both_cls}, Neither={neither_cls}")
    print(f"  Defect-Only (nc=1):         {DEFECT_ONLY_DIR}")
    print(f"    Train: {total_train_imgs} imgs, defects={def_counts_train.get(0,0)}")
    print(f"    Val:   {total_val_imgs} imgs, defects={def_counts_val.get(0,0)}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
