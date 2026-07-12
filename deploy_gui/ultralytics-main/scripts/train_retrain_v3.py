"""Retrain CargoDefect-YOLOv26 v3 with fixed dataset (NG supervision in training)."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

MODEL_YAML = "ultralytics/cfg/models/26/cargodefect-yolov26.yaml"
DATA_YAML = "ultralytics/cfg/datasets/cargodefect-fusion.yaml"

TRAIN_KWARGS = dict(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    device=0,
    workers=4,
    project="runs/cargodefect",
    name="fusion_v3",
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


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


batch = 8
trained = False
while batch >= 1:
    try:
        print(f"Starting v3 training with batch={batch}")
        model = YOLO(MODEL_YAML)
        model.train(batch=batch, **TRAIN_KWARGS)
        trained = True
        break
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        if batch == 1:
            raise RuntimeError("CUDA OOM at batch=1") from exc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        batch //= 2
        print(f"CUDA OOM, retrying batch={batch}")

if trained:
    print(f"Done. Results: {ROOT/'runs/detect/runs/cargodefect/fusion_v3'}")
else:
    print("Training failed.")
    sys.exit(1)
