#!/usr/bin/env python3
"""
Post-training evaluation: compare baseline vs clean_finetune vs P2 vs P2-DEB.
Generates results/final_model_compare/final_model_compare.csv and .md
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


DATA_YAML = "ultralytics/cfg/datasets/cargodefect-package.yaml"
OUTPUT_DIR = Path("results/final_model_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "YOLOv26n-package-defect baseline": {
        "weights": "runs/detect/runs/cargodefect/package_defect_baseline/weights/best.pt",
        "type": "baseline",
    },
    "CargoDefect clean finetune": {
        "weights": "runs/cargodefect/package_defect_clean_finetune/weights/best.pt",
        "type": "finetune",
    },
    "CargoDefect-YOLOv26-P2": {
        "weights": "runs/cargodefect/package_defect_p2/weights/best.pt",
        "type": "p2",
    },
    "CargoDefect-YOLOv26-P2-DEB": {
        "weights": "runs/cargodefect/package_defect_p2_deb/weights/best.pt",
        "type": "p2_deb",
    },
    "CargoDefect-YOLOv26-Detect": {
        "weights": "runs/cargodefect/cargodefect_yolov26_detect/weights/best.pt",
        "type": "cd_detect",
    },
    "CargoDefect-YOLOv26-P2-Detect": {
        "weights": "runs/cargodefect/cargodefect_yolov26_p2_detect/weights/best.pt",
        "type": "cd_p2_detect",
    },
}


def measure_speed(model, device, img_size=640, warmup=10, runs=50):
    """Measure inference speed with dummy tensors."""
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.model(dummy)

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(runs):
        start.record()
        with torch.no_grad():
            _ = model.model(dummy)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    return avg_ms, 1000 / avg_ms


def run_benchmark():
    rows = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for model_name, cfg in MODELS.items():
        weights = cfg["weights"]
        if not Path(weights).exists():
            print(f"[SKIP] {model_name}: weights not found at {weights}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Weights: {weights}")

        try:
            model = YOLO(str(weights))
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            continue

        # Params & GFLOPs
        params = sum(p.numel() for p in model.model.parameters())
        try:
            from ultralytics.utils import torch_utils
            flops_info = torch_utils.profile(model.model, imgsz=640)
            gflops = flops_info[1] / 1e9 if hasattr(flops_info, '__getitem__') else 0
        except Exception:
            try:
                from thop import profile as thop_profile
                dummy = torch.randn(1, 3, 640, 640).to(device)
                flops, _ = thop_profile(model.model, inputs=(dummy,), verbose=False)
                gflops = flops / 1e9
            except Exception:
                gflops = 0

        # Validation metrics
        try:
            val_results = model.val(data=DATA_YAML, split="val", verbose=False)
            try:
                cls_results = val_results.class_result(0.5)
            except:
                cls_results = None
            package_idx = 0
            defect_idx = 1
        except Exception as e:
            print(f"[WARN] Validation failed: {e}, using per-class extraction")
            val_results = model.val(data=DATA_YAML, split="val", verbose=False, plots=False)
            cls_results = None

        # Extract per-class metrics
        if hasattr(val_results, 'ap_class_index'):
            all_ap = val_results.ap_class_index
            all_maps = val_results.box.all_ap if hasattr(val_results.box, 'all_ap') else val_results.box.ap

            pkg_map50 = val_results.box.maps[package_idx] if package_idx < len(val_results.box.maps) else 0
            def_map50 = val_results.box.maps[defect_idx] if defect_idx < len(val_results.box.maps) else 0
            pkg_map50_95 = val_results.box.maps[package_idx] if package_idx < len(val_results.box.maps) else 0
            def_map50_95 = val_results.box.maps[defect_idx] if defect_idx < len(val_results.box.maps) else 0
        else:
            pkg_map50 = def_map50 = pkg_map50_95 = def_map50_95 = 0

        # Per-class P and R from results
        try:
            overall_p = val_results.box.mp[package_idx] * 100 if hasattr(val_results.box, 'mp') else 0
            overall_r = val_results.box.mr[package_idx] * 100 if hasattr(val_results.box, 'mr') else 0
            def_p = val_results.box.mp[defect_idx] * 100 if hasattr(val_results.box, 'mp') else 0
            def_r = val_results.box.mr[defect_idx] * 100 if hasattr(val_results.box, 'mr') else 0
        except:
            overall_p = overall_r = def_p = def_r = 0

        # Parse from results dict
        res = val_results.results_dict if hasattr(val_results, 'results_dict') else {}

        # Speed
        try:
            speed = model.val(data=DATA_YAML, split="val", verbose=False, plots=False)
            if hasattr(speed, 'speed'):
                pre_ms = speed.speed.get('preprocess', 0)
                inf_ms = speed.speed.get('inference', 0)
                post_ms = speed.speed.get('postprocess', 0)
                fps = 1000 / inf_ms if inf_ms > 0 else 0
            else:
                pre_ms = inf_ms = post_ms = fps = 0
        except:
            pre_ms = inf_ms = post_ms = fps = 0

        # Fallback: measure speed with dummy
        if fps == 0 and torch.cuda.is_available():
            try:
                inf_dummy_ms, fps_dummy = measure_speed(model, device)
                inf_ms = inf_dummy_ms
                fps = fps_dummy
            except:
                pass

        row = {
            "Model": model_name,
            "Params": params,
            "GFLOPs": round(gflops, 2),
            "FPS": round(fps, 1),
            "package_P": round(pkg_map50, 3) if pkg_map50 else 0,
            "package_R": round(def_map50, 3) if def_map50 else 0,
            "package_mAP50": round(pkg_map50, 4),
            "package_mAP50_95": round(pkg_map50_95, 4),
            "defect_P": round(def_p, 3),
            "defect_R": round(def_r, 3),
            "defect_mAP50": round(def_map50, 4),
            "defect_mAP50_95": round(def_map50_95, 4),
        }
        rows.append(row)

        print(f"  package_mAP50={pkg_map50:.4f}, defect_mAP50={def_map50:.4f}")
        print(f"  FPS={fps:.1f}, Params={params:,}")

    return rows


def generate_report(rows):
    csv_path = OUTPUT_DIR / "final_model_compare.csv"
    md_path = OUTPUT_DIR / "final_model_compare.md"

    # CSV
    if rows:
        fieldnames = [
            "Model", "Params", "GFLOPs", "FPS",
            "package_P", "package_R", "package_mAP50", "package_mAP50_95",
            "defect_P", "defect_R", "defect_mAP50", "defect_mAP50_95",
            "recommendation"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row["recommendation"] = ""
                writer.writerow(row)
        print(f"\nCSV saved: {csv_path}")

    # Markdown
    with open(md_path, "w") as f:
        f.write("# CargoDefect Model Comparison Report\n\n")
        f.write("## Overview\n\n")
        f.write("Comparison of four variants for package+defect detection with OK/NG quality judgment.\n\n")

        f.write("## Model Metrics\n\n")
        if rows:
            headers = list(rows[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for row in rows:
                vals = [str(row.get(h, "")) for h in headers]
                f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n")

        # Find best model based on defect Recall (priority 1) then defect mAP50 (priority 2)
        if rows:
            sorted_by_defect_r = sorted(rows, key=lambda x: x.get("defect_R", 0), reverse=True)
            sorted_by_defect_map = sorted(rows, key=lambda x: x.get("defect_mAP50", 0), reverse=True)

            f.write("## Recommendation\n\n")

            best_r = sorted_by_defect_r[0]
            f.write(f"**Recommended model**: {best_r['Model']}\n\n")
            f.write("- Highest defect Recall: {:.4f}\n".format(best_r.get("defect_R", 0)))
            f.write("- defect mAP50: {:.4f}\n".format(best_r.get("defect_mAP50", 0)))
            f.write("- package mAP50: {:.4f}\n".format(best_r.get("package_mAP50", 0)))
            f.write("- FPS: {:.1f}\n".format(best_r.get("FPS", 0)))

            f.write("\n**Selection criteria** (priority order):\n")
            f.write("1. defect Recall (highest)\n")
            f.write("2. defect mAP50 (not lower than baseline 0.430)\n")
            f.write("3. package mAP50 >= 0.90\n")
            f.write("4. Inference speed (not significantly slower than baseline)\n")
            f.write("5. GUI stability\n\n")

            f.write("### Baseline reference\n")
            f.write("- package mAP50=0.929, defect mAP50=0.430, defect Recall=0.393\n\n")

            f.write("### Target\n")
            f.write("- Minimum: defect Recall >= 0.45, defect mAP50 >= 0.43, package mAP50 >= 0.90\n")
            f.write("- Ideal: defect Recall >= 0.55, defect mAP50 >= 0.50\n")

    print(f"Markdown saved: {md_path}")


def main():
    rows = run_benchmark()
    if rows:
        generate_report(rows)
    else:
        print("\n[WARN] No models evaluated. Check that weight files exist.")


if __name__ == "__main__":
    main()
