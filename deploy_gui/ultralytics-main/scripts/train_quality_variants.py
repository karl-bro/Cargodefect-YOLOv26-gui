"""Batch train CargoDefect-YOLOv26 v4 with 5 quality loss variants."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import yaml
import shutil
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

BASE_YAML = "ultralytics/cfg/models/26/cargodefect-yolov26.yaml"
DATA_YAML = "ultralytics/cfg/datasets/cargodefect-fusion.yaml"
CONFIG_DIR = ROOT / "configs/ablation/quality_loss"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_KWARGS = dict(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    device=0,
    workers=4,
    project="runs/cargodefect",
    exist_ok=True,
    amp=True,
    cos_lr=True,
    grad_clip=1.0,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    close_mosaic=10,
    patience=100,
    seed=0,
    deterministic=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    auto_augment="randaugment",
    erasing=0.4,
)

VARIANTS = [
    {"name": "fusion_v4_pos1.0", "quality_pos_weight": 1.0},
    {"name": "fusion_v4_pos1.2", "quality_pos_weight": 1.2},
    {"name": "fusion_v4_pos1.5", "quality_pos_weight": 1.5},
    {"name": "fusion_v4_focal0.25", "quality_focal_gamma": 2.0, "quality_focal_alpha": 0.25},
    {"name": "fusion_v4_focal0.35", "quality_focal_gamma": 2.0, "quality_focal_alpha": 0.35},
]

OUT_DIR = ROOT / "results/quality_loss_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_YAML) as f:
    base_cfg = yaml.safe_load(f)


def _is_oom(exc):
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


results = []
for v in VARIANTS:
    name = v.pop("name")
    cfg = deepcopy(base_cfg)
    for k, val in v.items():
        cfg[k] = val
    # Remove pos_weight if using focal
    if "quality_focal_gamma" in cfg and cfg.get("quality_focal_gamma", 0) > 0:
        cfg.pop("quality_pos_weight", None)
    else:
        cfg.pop("quality_focal_gamma", None)
        cfg.pop("quality_focal_alpha", None)

    yaml_path = CONFIG_DIR / f"{name}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Generated {yaml_path}")

    batch = 8
    trained = False
    while batch >= 1:
        try:
            print(f"\n=== Training {name} (batch={batch}) ===")
            model = YOLO(str(yaml_path))
            model.train(batch=batch, name=name, **TRAIN_KWARGS)
            trained = True
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            if batch == 1:
                raise RuntimeError(f"CUDA OOM at batch=1 for {name}") from exc
            # Fully release CUDA before retry
            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            batch //= 2
            print(f"CUDA OOM, retrying batch={batch}")

    if trained:
        results.append(f"  {name}: done")
    else:
        results.append(f"  {name}: FAILED")

# Summary
summary = ["# v4 Quality Loss Batch Training Results", "", "## Runs"] + results
(OUT_DIR / "batch_summary.md").write_text("\n".join(summary), encoding="utf-8")
print("\n\n=== BATCH COMPLETE ===")
print("\n".join(results))
print(f"Summary: {OUT_DIR / 'batch_summary.md'}")
