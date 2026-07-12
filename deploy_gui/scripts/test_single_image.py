#!/usr/bin/env python3
"""Test single-image detection with full pipeline (YOLO + ROI fallback).

Usage:
    python scripts/test_single_image.py --image path/to/photo.jpg
    python scripts/test_single_image.py --image path/to/photo.jpg --weights custom.pt
    python scripts/test_single_image.py --image path/to/photo.jpg --no-fallback

Outputs:
    - Console: YOLO raw boxes, ROI fallback result, final decision
    - Image:   <input>_detected.jpg (annotated with boxes, ROI, and fallback label)
"""

from __future__ import annotations

import argparse, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np, yaml

from utils.detector import CargoDefectDetector, PACKAGE_ID, DEFECT_ID

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Single-image test with ROI fallback")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default=None, help="Model weights path (defaults to config)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable package fallback")
    parser.add_argument("--fixed-station", action="store_true", help="Enable fixed station mode")
    parser.add_argument("--debug", action="store_true", help="Print per-box debug info")
    args = parser.parse_args()

    cfg = load_config()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    # Override weights if provided
    if args.weights:
        cfg["model"]["weights"] = args.weights

    # Override debug
    if args.debug:
        cfg["decision"]["debug"] = True

    # Override fallback
    if args.no_fallback:
        cfg["package_fallback"]["enable"] = False

    # Override fixed station
    if args.fixed_station:
        cfg["decision"]["fixed_station_mode"] = True

    # -- Create detector + load model --
    detector = CargoDefectDetector(cfg)
    try:
        detector.load()
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        sys.exit(1)

    # -- Read image --
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        sys.exit(1)
    print(f"[INFO] Image: {img_path}")
    print(f"[INFO] Shape: {image.shape}")

    # -- Config summary --
    print(f"\n{'='*60}")
    print(f"  Configuration")
    print(f"{'='*60}")
    print(f"  imgsz          : {detector.imgsz}")
    print(f"  raw_conf       : {detector.raw_conf}")
    print(f"  iou            : {detector.iou}")
    print(f"  max_det        : {detector.max_det}")
    print(f"  package_conf   : {detector.package_conf}")
    print(f"  defect_show_conf: {detector.defect_show_conf}")
    print(f"  defect_ng_conf : {detector.defect_ng_conf}")
    print(f"  fixed_station  : {detector.fixed_station_mode}")
    print(f"  fallback_enable: {detector.fallback_enable}")
    print(f"  fallback_roi   : {detector.fallback_roi}")
    print(f"  fallback_min_area_r: {detector.fallback_min_area_r}")
    print(f"  device         : {detector.device}")

    # -- Run full detection pipeline --
    t0 = datetime.now()
    result = detector.detect(image)
    elapsed = (datetime.now() - t0).total_seconds() * 1000

    # -- Print raw YOLO info --
    print(f"\n{'='*60}")
    print(f"  YOLO RAW DETECTION")
    print(f"{'='*60}")
    print(f"  inference   : {result['inference_ms']:.1f} ms  (total: {elapsed:.0f} ms)")
    print(f"  n_raw       : {result['n_raw']}")
    print(f"  n_pkg (YOLO): {result['n_pkg']}")
    print(f"  n_def_ng    : {result['n_def_ng']}")
    print(f"  n_def_sus   : {result['n_def_suspect']}")

    # -- ROI fallback result --
    print(f"\n{'='*60}")
    print(f"  PACKAGE FALLBACK")
    print(f"{'='*60}")
    print(f"  fallback_enable: {detector.fallback_enable}")
    print(f"  YOLO found pkg : {result['n_pkg'] > 0}")
    if detector.fallback_enable:
        fb_result = CargoDefectDetector.has_large_object_in_roi(
            image, detector.fallback_roi,
            detector.fallback_min_area_r,
            detector.fallback_min_w_r,
            detector.fallback_min_h_r,
        )
        print(f"  ROI fallback   : {fb_result} (large object in ROI)")
    else:
        print(f"  ROI fallback   : (disabled)")
    print(f"  package_source : {result['package_source']}")

    # -- Decision --
    print(f"\n{'='*60}")
    print(f"  FINAL DECISION")
    print(f"{'='*60}")
    print(f"  quality       : {result['quality']}")
    print(f"  has_package   : {result['has_package']}")
    print(f"  has_defect    : {result['has_defect']}")
    print(f"  defect_status : {result['defect_status']}")
    print(f"  cargo_class   : {result['cargo_class']}")
    print(f"  max_conf      : {result['max_conf']:.4f}")
    print(f"  package_source: {result['package_source']}")

    # -- Boxes detail --
    boxes = result["boxes"]
    print(f"\n{'='*60}")
    print(f"  DRAWN BOXES (n={len(boxes)})")
    print(f"{'='*60}")
    if boxes:
        print(f"  {'#':<4} {'cls':<4} {'label':<10} {'conf':<8} {'color':<8}  {'rect'}")
        print(f"  {'-'*60}")
        for i, b in enumerate(boxes):
            print(f"  {i:<4} {b['cls']:<4} {b['label']:<10} {b['conf']:<8.4f} {b['color']:<8}  "
                  f"({int(b['x1'])},{int(b['y1'])}-{int(b['x2'])},{int(b['y2'])})")
    else:
        print(f"  (no boxes drawn)")

    # -- Annotated image --
    annotated = image.copy()

    # ROI
    if detector.enable_roi:
        rx, ry, rx2, ry2 = detector.roi
        cv2.rectangle(annotated, (rx, ry), (rx2, ry2), (255, 255, 128), 1)

    # Fallback label
    if result.get("package_source") == "ROI兜底" and detector.enable_roi:
        rx, ry, _, _ = detector.roi
        cv2.putText(annotated, "package fallback", (rx + 4, ry + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 128), 1)

    # Boxes
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        label = b["label"]
        if b["color"] == "blue":
            color = (255, 0, 0)
            text = f"package {b['conf']:.2f}"
        elif b["color"] == "red":
            color = (0, 0, 255)
            text = f"defect {b['conf']:.2f}"
        elif b["color"] == "yellow":
            color = (0, 220, 255)
            text = f"suspect {b['conf']:.2f}"
        else:
            color = (255, 255, 255)
            text = f"?? {b['conf']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, text, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    q = result["quality"]
    q_color = (0, 200, 50) if q == "OK" else (0, 0, 255) if q == "NG" else (0, 180, 255)
    cv2.putText(annotated, f"{q}  ({result['package_source']})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, q_color, 2)

    out_path = img_path.parent / f"{img_path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n[INFO] Annotated image saved: {out_path}")


if __name__ == "__main__":
    main()
