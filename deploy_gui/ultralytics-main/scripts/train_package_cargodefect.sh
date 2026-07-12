#!/bin/bash
# Train CargoDefect-YOLOv26 Package with quality head (OK/NG)

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cargodefect

MODEL_YAML="ultralytics/cfg/models/26/cargodefect-yolov26-package.yaml"
DATA_YAML="ultralytics/cfg/datasets/cargodefect-package.yaml"
NAME="cargodefect_package"

python -c "
from ultralytics import YOLO
model = YOLO('${MODEL_YAML}')
model.train(
    data='${DATA_YAML}',
    epochs=100, imgsz=640, batch=8, device=0, workers=4,
    project='runs/cargodefect', name='${NAME}', exist_ok=True,
    amp=True, cos_lr=True, grad_clip=1.0, optimizer='auto',
    lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    close_mosaic=10, patience=100, seed=0, deterministic=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, cutmix=0.0,
    copy_paste=0.0, auto_augment='randaugment', erasing=0.4,
)
print('cargodefect_package DONE')
"
