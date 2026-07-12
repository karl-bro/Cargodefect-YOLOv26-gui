#!/usr/bin/env python3
"""
Defect error analysis for package+defect baseline.
Produces: FN/FP samples, confidence distribution, per-source stats, conf sweep, visualizations.
"""
from __future__ import annotations

import csv, json, os, shutil, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/detect/runs/cargodefect/package_defect_baseline/weights/best.pt"
DATA_YAML = ROOT / "ultralytics/cfg/datasets/cargodefect-package.yaml"
OUT_DIR = ROOT / "results/package_defect_error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "fn_samples").mkdir(exist_ok=True)
(OUT_DIR / "fp_samples").mkdir(exist_ok=True)
(OUT_DIR / "low_conf_samples").mkdir(exist_ok=True)

DS_DIR = ROOT / "datasets/cargodefect_package"
VAL_IMG_DIR = DS_DIR / "images/val"
VAL_LABEL_DIR = DS_DIR / "labels/val"

CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
IOU_THRESH = 0.5


def load_model():
    model = YOLO(str(WEIGHTS))
    return model


def load_val_ground_truth():
    gt = {}
    for label_file in sorted(VAL_LABEL_DIR.glob("*.txt")):
        stem = label_file.stem
        lines = label_file.read_text().strip().splitlines()
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append({"class": cls, "bbox": [cx, cy, w, h]})
        if lines and lines[0].strip():
            gt[stem] = boxes
        else:
            gt[stem] = []
    return gt


def xywh2xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def run_validation_with_predictions(model, conf=0.001):
    results = model.predict(
        source=str(VAL_IMG_DIR),
        conf=conf,
        iou=0.65,
        save=False,
        verbose=False,
    )
    preds = {}
    for r in results:
        path = Path(r.path)
        stem = path.stem
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf_val = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append({"class": cls, "conf": conf_val, "xyxy": xyxy})
        preds[stem] = boxes
    return preds


def match_defect_predictions(gt_boxes, pred_boxes):
    gt_defect = [(i, b) for i, b in enumerate(gt_boxes) if b["class"] == 1]
    pred_defect = [(j, b) for j, b in enumerate(pred_boxes) if b["class"] == 1]

    if not pred_defect:
        return [], [], list(range(len(gt_defect))), pred_defect, gt_defect

    matched_pos = set()
    matched_gt = set()
    pairs = []

    # Sort by confidence descending, using position within pred_defect
    pred_ranked = sorted(enumerate(pred_defect), key=lambda x: x[1][1]["conf"], reverse=True)
    for pos, (orig_j, pred) in pred_ranked:
        best_iou = 0
        best_g = None
        for g_idx, gt_b in gt_defect:
            if g_idx in matched_gt:
                continue
            gt_xyxy = xywh2xyxy(*gt_b["bbox"], 640, 640)
            pred_xyxy = list(pred["xyxy"])
            iou = compute_iou(gt_xyxy, pred_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_g = g_idx
        if best_iou >= IOU_THRESH and best_g is not None:
            matched_pos.add(pos)
            matched_gt.add(best_g)
            pairs.append({"pos": pos, "g_idx": best_g, "iou": best_iou})

    unmatched_pred = [pos for pos in range(len(pred_defect)) if pos not in matched_pos]
    unmatched_gt = [g_idx for g_idx, _ in gt_defect if g_idx not in matched_gt]
    return pairs, unmatched_pred, unmatched_gt, pred_defect, gt_defect


def match_package_predictions(gt_boxes, pred_boxes):
    gt_pkg = [(i, b) for i, b in enumerate(gt_boxes) if b["class"] == 0]
    pred_pkg = [(j, b) for j, b in enumerate(pred_boxes) if b["class"] == 0]
    matched_pred = set()
    matched_gt = set()
    pred_pkg_sorted = sorted(pred_pkg, key=lambda x: x[1]["conf"], reverse=True)
    for p_idx, pred in pred_pkg_sorted:
        best_iou = 0
        best_g = None
        for g_idx, gt_b in gt_pkg:
            if g_idx in matched_gt:
                continue
            gt_xyxy = xywh2xyxy(*gt_b["bbox"], 640, 640)
            pred_xyxy = list(pred["xyxy"])
            iou = compute_iou(gt_xyxy, pred_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_g = g_idx
        if best_iou >= IOU_THRESH and best_g is not None:
            matched_pred.add(p_idx)
            matched_gt.add(best_g)
    return len(matched_pred), len(pred_pkg) - len(matched_pred), len(gt_pkg) - len(matched_gt)


def visualize_box(img_path, boxes, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for box in boxes:
        if "xyxy" in box:
            x1, y1, x2, y2 = [int(v) for v in box["xyxy"]]
        else:
            x1, y1, x2, y2 = xywh2xyxy(*box["bbox"], w, h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        color = (0, 0, 255) if box.get("type") == "fn" else (0, 165, 255)
        label = box.get("label", "")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out_path), img)


def main():
    print("=" * 60)
    print("Defect Error Analysis for package+defect baseline")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model = load_model()

    print("[2/5] Loading val ground truth...")
    gt_dict = load_val_ground_truth()
    total_defect_gt = sum(sum(1 for b in boxes if b["class"] == 1) for boxes in gt_dict.values())
    total_pkg_gt = sum(sum(1 for b in boxes if b["class"] == 0) for boxes in gt_dict.values())
    print(f"  Val images: {len(gt_dict)}, package GT: {total_pkg_gt}, defect GT: {total_defect_gt}")

    print("[3/5] Running predictions (conf=0.001)...")
    preds_dict = run_validation_with_predictions(model, conf=0.001)

    print("[4/5] Analyzing error patterns...")
    all_conf_tp = []
    all_conf_fp = []
    fn_samples = []
    fp_samples = []
    low_conf_tp = []
    defect_tp, defect_fp, defect_fn = 0, 0, 0

    for stem, gt_list in tqdm(gt_dict.items(), desc="Matching"):
        img_path = VAL_IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            img_path = VAL_IMG_DIR / f"{stem}.png"
        if not img_path.exists() or stem not in preds_dict:
            continue

        pred_list = preds_dict[stem]
        pairs, unmatched_pred, unmatched_gt, pred_defect_list, gt_defect_list = \
            match_defect_predictions(gt_list, pred_list)

        defect_tp += len(pairs)
        defect_fn += len(unmatched_gt)
        defect_fp += len(unmatched_pred)

        # Collect TP confidences
        for pair in pairs:
            conf = pred_defect_list[pair["pos"]][1]["conf"]
            all_conf_tp.append(conf)
            if conf < 0.2:
                low_conf_tp.append((stem, conf))

        # FN visualization
        if unmatched_gt:
            fn_boxes = []
            for g_idx in unmatched_gt:
                fn_boxes.append({"bbox": gt_defect_list[g_idx][1]["bbox"], "type": "fn", "label": "FN"})
            fn_samples.append((stem, fn_boxes))
            if len(fn_samples) <= 50:
                visualize_box(img_path, fn_boxes, OUT_DIR / "fn_samples" / f"{stem}.jpg")

        # FP visualization
        if unmatched_pred:
            fp_boxes = []
            for pos in unmatched_pred:
                pred_b = pred_defect_list[pos][1]
                all_conf_fp.append(pred_b["conf"])
                fp_boxes.append({**pred_b, "type": "fp", "label": f"FP {pred_b['conf']:.2f}"})
            fp_samples.append((stem, fp_boxes))
            if len(fp_samples) <= 50:
                visualize_box(img_path, fp_boxes, OUT_DIR / "fp_samples" / f"{stem}.jpg")

    recall = defect_tp / (defect_tp + defect_fn) if (defect_tp + defect_fn) > 0 else 0
    precision = defect_tp / (defect_tp + defect_fp) if (defect_tp + defect_fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  Defect: TP={defect_tp}, FP={defect_fp}, FN={defect_fn}")
    print(f"  Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
    print(f"  FN images: {len(fn_samples)}, FP images: {len(fp_samples)}")

    conf_tp_arr = np.array(all_conf_tp) if all_conf_tp else np.array([])
    conf_fp_arr = np.array(all_conf_fp) if all_conf_fp else np.array([])
    if len(conf_tp_arr) > 0:
        print(f"  TP conf: mean={conf_tp_arr.mean():.4f}, median={np.median(conf_tp_arr):.4f}")
    if len(conf_fp_arr) > 0:
        print(f"  FP conf: mean={conf_fp_arr.mean():.4f}, median={np.median(conf_fp_arr):.4f}")

    print("[5/5] Running conf threshold sweep...")
    sweep_results = []
    for conf_th in CONF_THRESHOLDS:
        tp, fp, fn = 0, 0, 0
        tp_p, fp_p, fn_p = 0, 0, 0
        for stem, gt_list in gt_dict.items():
            if stem not in preds_dict:
                continue
            pred_list = [p for p in preds_dict[stem] if p["conf"] >= conf_th]
            pairs, unmatched_pred, unmatched_gt, _, _ = match_defect_predictions(gt_list, pred_list)
            tp += len(pairs)
            fn += len(unmatched_gt)
            fp += len(unmatched_pred)
            tp_p_i, fp_p_i, fn_p_i = match_package_predictions(gt_list, pred_list)
            tp_p += tp_p_i
            fp_p += fp_p_i
            fn_p += fn_p_i

        def_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        def_r = tp / (tp + fn) if (tp + fn) > 0 else 0
        def_f1 = 2 * def_p * def_r / (def_p + def_r) if (def_p + def_r) > 0 else 0
        pkg_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0
        pkg_r = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0
        sweep_results.append({
            "conf_threshold": conf_th,
            "defect_TP": tp, "defect_FP": fp, "defect_FN": fn,
            "defect_P": round(def_p, 4),
            "defect_R": round(def_r, 4),
            "defect_F1": round(def_f1, 4),
            "package_P": round(pkg_p, 4),
            "package_R": round(pkg_r, 4),
        })

    # ---- Save outputs ----
    print("\n=== Generating reports ===")

    sweep_csv = OUT_DIR / "conf_threshold_sweep.csv"
    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
        w.writeheader()
        w.writerows(sweep_results)
    print(f"Saved: {sweep_csv}")

    report_path = OUT_DIR / "error_report.md"
    with open(report_path, "w") as f:
        f.write("# Defect Error Analysis Report\n\n")
        f.write(f"**Model**: `package_defect_baseline/weights/best.pt`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Defect GT in val: {total_defect_gt}\n")
        f.write(f"- True Positives: {defect_tp}\n")
        f.write(f"- False Negatives: {defect_fn}\n")
        f.write(f"- False Positives: {defect_fp}\n")
        f.write(f"- **Recall**: {recall:.4f}\n")
        f.write(f"- **Precision**: {precision:.4f}\n")
        f.write(f"- **F1**: {f1:.4f}\n\n")

        f.write("## Confidence Distribution\n\n")
        if len(conf_tp_arr) > 0:
            f.write(f"- TP confidence: mean={conf_tp_arr.mean():.4f}, median={np.median(conf_tp_arr):.4f}, min={conf_tp_arr.min():.4f}, max={conf_tp_arr.max():.4f}\n")
        if len(conf_fp_arr) > 0:
            f.write(f"- FP confidence: mean={conf_fp_arr.mean():.4f}, median={np.median(conf_fp_arr):.4f}, min={conf_fp_arr.min():.4f}, max={conf_fp_arr.max():.4f}\n")
        f.write("\n")

        f.write("## Low Confidence TPs (conf < 0.2)\n\n")
        f.write(f"Count: {len(low_conf_tp)}\n")
        for stem, conf in low_conf_tp[:30]:
            f.write(f"- `{stem}`: {conf:.4f}\n")
        f.write("\n")

        f.write("## Confidence Threshold Sweep\n\n")
        f.write("| conf | defect_P | defect_R | defect_F1 | package_P | package_R |\n")
        f.write("|------|----------|----------|-----------|-----------|----------|\n")
        for r in sweep_results:
            f.write(f"| {r['conf_threshold']:.2f} | {r['defect_P']:.4f} | {r['defect_R']:.4f} | {r['defect_F1']:.4f} | {r['package_P']:.4f} | {r['package_R']:.4f} |\n")

        best = max(sweep_results, key=lambda x: x["defect_F1"])
        f.write(f"\n**Recommended threshold: {best['conf_threshold']:.2f}** (defect F1={best['defect_F1']:.4f})\n")

        f.write(f"\n## FN Samples ({len(fn_samples)})\n\n")
        for stem, _ in fn_samples[:30]:
            f.write(f"- `{stem}`\n")

        f.write(f"\n## FP Samples ({len(fp_samples)})\n\n")
        for stem, _ in fp_samples[:30]:
            f.write(f"- `{stem}`\n")

    print(f"Saved: {report_path}")
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
