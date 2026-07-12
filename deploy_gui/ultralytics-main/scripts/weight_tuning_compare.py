#!/usr/bin/env python3
"""Compare weight tuning experiments vs baseline finetune model."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

DATA_YAML = "ultralytics/cfg/datasets/cargodefect-package.yaml"
OUTPUT_DIR = Path("results/weight_tuning_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "CargoDefect-Detect finetune (baseline)": {
        "weights": "runs/detect/runs/cargodefect/cargodefect_detect_finetune_from_baseline/weights/best.pt",
    },
    "defectw2 (class_weights 2.0)": {
        "weights": "runs/detect/runs/cargodefect/cargodefect_detect_defectw2/weights/best.pt",
    },
    "defectw2_cls075": {
        "weights": "runs/detect/runs/cargodefect/cargodefect_detect_defectw2_cls075/weights/best.pt",
    },
    "pgme_alpha05": {
        "weights": "runs/detect/runs/cargodefect/cargodefect_detect_pgme_alpha05/weights/best.pt",
    },
    "p2loss15": {
        "weights": "runs/detect/runs/cargodefect/cargodefect_detect_p2loss15/weights/best.pt",
    },
}


def measure_fps(model, device="cuda:0", img_size=640, warmup=10, runs=50):
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    model.model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.model(dummy)
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
    return round(1000 / avg_ms, 1) if avg_ms > 0 else 0.0


def eval_model(name, weights):
    if not Path(weights).exists():
        print(f"[SKIP] {name}: not found {weights}")
        return None

    print(f"\nEvaluating: {name}")
    model = YOLO(str(weights))
    params = sum(p.numel() for p in model.model.parameters())

    gflops = 0.0
    try:
        from ultralytics.utils import torch_utils

        info = torch_utils.profile(model.model, imgsz=640)
        gflops = round(info[1] / 1e9, 2)
    except Exception:
        pass

    val = model.val(data=DATA_YAML, split="val", verbose=False, plots=False)
    maps = list(val.box.maps) if hasattr(val.box, "maps") else [0, 0]
    ap50 = list(getattr(val.box, "ap50", maps))
    ap = list(getattr(val.box, "ap", maps))

    # Per-class P/R from results_dict (mp/mr are macro averages, not per-class arrays)
    rd = getattr(val, "results_dict", {}) or {}
    pkg_i, def_i = 0, 1

    def _metric(prefix, cls_idx):
        for key in (f"{prefix}/B", prefix):
            v = rd.get(key)
            if v is not None:
                return float(v)
        return 0.0

    # Try per-class from box arrays if available
    def _per_class(arr, idx):
        if arr is None:
            return 0.0
        if hasattr(arr, "__len__") and len(arr) > idx:
            return float(arr[idx])
        return float(arr) if idx == 0 else 0.0

    pkg_p = _per_class(getattr(val.box, "p", None), pkg_i) * 100
    pkg_r = _per_class(getattr(val.box, "r", None), pkg_i) * 100
    def_p = _per_class(getattr(val.box, "p", None), def_i) * 100
    def_r = _per_class(getattr(val.box, "r", None), def_i) * 100

    if def_r == 0 and "metrics/recall(B)" in rd:
        # Fallback: parse from printed table via box class results
        try:
            for i, cname in enumerate(val.names.values()):
                cr = val.box.class_result(i)
                if i == pkg_i:
                    pkg_p, pkg_r = cr[0] * 100, cr[1] * 100
                elif i == def_i:
                    def_p, def_r = cr[0] * 100, cr[1] * 100
        except Exception:
            pass

    fps = measure_fps(model) if torch.cuda.is_available() else 0.0

    row = {
        "Model": name,
        "Params": params,
        "GFLOPs": gflops,
        "FPS": fps,
        "package_P": round(pkg_p, 2),
        "package_R": round(pkg_r, 2),
        "package_mAP50": round(float(maps[pkg_i]), 4),
        "package_mAP50_95": round(float(ap[pkg_i]), 4),
        "defect_P": round(def_p, 2),
        "defect_R": round(def_r, 2),
        "defect_mAP50": round(float(maps[def_i]), 4),
        "defect_mAP50_95": round(float(ap[def_i]), 4),
        "all_mAP50": round(float(val.box.map50), 4),
        "all_mAP50_95": round(float(val.box.map), 4),
    }
    print(f"  defect_R={row['defect_R']}, defect_mAP50={row['defect_mAP50']}, package_mAP50={row['package_mAP50']}")
    return row


def main():
    rows = []
    for name, cfg in MODELS.items():
        row = eval_model(name, cfg["weights"])
        if row:
            rows.append(row)

    if not rows:
        print("No models evaluated.")
        return

    fields = [
        "Model", "Params", "GFLOPs", "FPS",
        "package_P", "package_R", "package_mAP50", "package_mAP50_95",
        "defect_P", "defect_R", "defect_mAP50", "defect_mAP50_95",
        "all_mAP50", "all_mAP50_95",
    ]
    csv_path = OUTPUT_DIR / "weight_tuning_compare.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUTPUT_DIR / "weight_tuning_compare.md"
    with open(md_path, "w") as f:
        f.write("# Weight Tuning Comparison\n\n")
        f.write("Selection criteria: defect Recall > defect mAP50 >= baseline > package_mAP50 >= 0.90\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in fields) + " |\n")

        best = max(rows[1:], key=lambda x: (x["defect_R"], x["defect_mAP50"]), default=None)
        if best:
            f.write(f"\n**Recommended**: {best['Model']} (defect_R={best['defect_R']}, defect_mAP50={best['defect_mAP50']})\n")

    print(f"\nSaved: {csv_path}\nSaved: {md_path}")


if __name__ == "__main__":
    main()
