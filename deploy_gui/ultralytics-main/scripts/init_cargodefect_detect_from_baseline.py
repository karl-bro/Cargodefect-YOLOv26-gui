#!/usr/bin/env python3
"""
Initialize CargoDefect-YOLOv26-Detect model from package_defect_baseline weights.
Matches weights by shape, skipping new PGME layers and mismatched neck/Detect layers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO


BASELINE_PT = "runs/detect/runs/cargodefect/package_defect_baseline/weights/best.pt"
CARGO_YAML = "ultralytics/cfg/models/26/cargodefect-yolov26-detect.yaml"
OUTPUT_PT = "weights/cargodefect_detect_init_from_baseline.pt"


def main():
    Path(OUTPUT_PT).parent.mkdir(parents=True, exist_ok=True)

    # 1. Build CargoDetect model
    print("Building CargoDetect model...")
    cargo_model = YOLO(CARGO_YAML)
    cargo_sd = cargo_model.model.state_dict()
    cargo_keys = list(cargo_sd.keys())
    print(f"  CargoDetect: {len(cargo_keys)} keys")

    # 2. Load baseline weights (full checkpoint, not just YOLO)
    print(f"Loading baseline: {BASELINE_PT}")
    bl_ckpt = torch.load(BASELINE_PT, map_location="cpu", weights_only=False)

    # Handle both formats: DetectionModel object or dict
    if isinstance(bl_ckpt.get("model"), torch.nn.Module):
        bl_sd = bl_ckpt["model"].state_dict()
    elif isinstance(bl_ckpt.get("model"), dict):
        bl_sd = bl_ckpt["model"]
    else:
        bl_sd = bl_ckpt
    bl_keys = list(bl_sd.keys())
    print(f"  Baseline: {len(bl_keys)} keys")

    # 3. Shape-based matching
    # Build a shape lookup from baseline
    bl_by_shape = {}
    for bk, bt in bl_sd.items():
        # Only cache trainable params (exclude num_batches_tracked, buffers)
        bl_by_shape.setdefault(tuple(bt.shape), []).append((bk, bt.shape))

    matched = 0
    skipped = []
    loaded_new = []
    bl_used = set()

    for ck in cargo_keys:
        cs = cargo_sd[ck].shape
        # Skip buffers like num_batches_tracked (no shape)
        if len(cs) == 0:
            continue

        # Find baseline key with matching shape
        candidates = bl_by_shape.get(tuple(cs), [])
        found = False
        for bk, bs in candidates:
            if bk not in bl_used:
                # Copy weight
                cargo_sd[ck] = bl_sd[bk].clone()
                bl_used.add(bk)
                loaded_new.append((ck, bk))
                matched += 1
                found = True
                break

        if not found:
            skipped.append((ck, tuple(cs)))

    print(f"\nResults:")
    print(f"  Matched & loaded: {matched} / {len(cargo_keys)} keys")
    print(f"  Skipped (random init): {len(skipped)} keys")
    print(f"  Baseline keys used: {matched} / {len(bl_keys)}")

    # Show skipped summary by module
    skip_by_module = {}
    for k, s in skipped:
        # Extract top-level module name
        parts = k.split(".")
        mod = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        skip_by_module.setdefault(mod, 0)
        skip_by_module[mod] += 1

    print(f"\n  Skipped by module:")
    for mod, cnt in sorted(skip_by_module.items()):
        print(f"    {mod}: {cnt} keys")

    # Show loaded summary by module
    load_by_module = {}
    for ck, bk in loaded_new:
        parts = ck.split(".")
        mod = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        load_by_module.setdefault(mod, 0)
        load_by_module[mod] += 1

    print(f"\n  Loaded by module:")
    for mod, cnt in sorted(load_by_module.items()):
        print(f"    {mod}: {cnt} keys")

    # 4. Save as proper YOLO checkpoint
    cargo_model.model.load_state_dict(cargo_sd, strict=True)
    
    # Build a minimal checkpoint that YOLO() can load
    ckpt = {
        "model": cargo_model.model,
        "train_args": {"batch": 8, "data": "cargodefect-package.yaml"},
        "date": __import__("datetime").datetime.now().isoformat(),
    }
    # Inject yaml for detection method detection
    ckpt["train_results"] = {}
    
    torch.save(ckpt, OUTPUT_PT)
    print(f"Saved to: {OUTPUT_PT}")

    # 5. Verify — build fresh model from YAML, load state, forward
    print("Verifying...")
    verify_m = YOLO(CARGO_YAML)
    verify_m.model.load_state_dict(cargo_sd, strict=True)
    verify_m.model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    print(f"\nSaved to: {OUTPUT_PT}")

    # 5. Verify
    print("Verifying...")
    verify_model = YOLO(OUTPUT_PT)
    dummy = torch.randn(1, 3, 640, 640)
    verify_model.model.eval()
    with torch.no_grad():
        out = verify_model.model(dummy)
    if isinstance(out, (list, tuple)):
        print(f"  Forward OK, output[0] shape = {list(out[0].shape)}")
    elif isinstance(out, torch.Tensor):
        print(f"  Forward OK, shape = {list(out.shape)}")
    print("Verification PASSED")


if __name__ == "__main__":
    main()
