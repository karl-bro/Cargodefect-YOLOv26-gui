#!/usr/bin/env python3
"""Analyze pseudo bbox impact on label distribution."""
from pathlib import Path

DS = Path("/home/swot2486/0701/ultralytics-main/datasets/cargodefect_package")

for split in ["train", "val"]:
    lbl_dir = DS / "labels" / split
    files = list(lbl_dir.glob("*.txt"))
    only_pkg = 0   # only class 0
    only_defect = 0  # only class 1
    both = 0       # both classes
    neither = 0    # no bboxes
    pseudo_count = 0  # files with full-image pseudo bbox for class 0
    real_pkg_total = 0  # class-0 bboxes that are NOT pseudo (cx=0.5,w=1,h=1)

    for lbl in files:
        text = lbl.read_text().strip()
        lines = [l for l in text.split("\n") if l.strip()]
        has0 = any(l.startswith("0 ") for l in lines)
        has1 = any(l.startswith("1 ") for l in lines)
        if has0 and has1: both += 1
        elif has0 and not has1: only_pkg += 1
        elif not has0 and has1: only_defect += 1
        else: neither += 1

        found_pseudo = False
        for l in lines:
            if not l.startswith("0 "): continue
            parts = l.split()
            if len(parts) < 5: continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            is_pseudo = (abs(cx - 0.5) < 0.02 and abs(cy - 0.5) < 0.02
                         and abs(w - 1.0) < 0.02 and abs(h - 1.0) < 0.02)
            if is_pseudo:
                found_pseudo = True
            else:
                real_pkg_total += 1
        if found_pseudo:
            pseudo_count += 1

    print(f"=== {split.upper()} ({len(files)} files) ===")
    print(f"  only package:      {only_pkg}")
    print(f"  only defect:       {only_defect}")
    print(f"  both package+defect: {both}")
    print(f"  neither (empty):   {neither}")
    print(f"  files with pseudo full-img bbox: {pseudo_count}")
    print(f"  non-pseudo class-0 bboxes: {real_pkg_total}")
    print()
