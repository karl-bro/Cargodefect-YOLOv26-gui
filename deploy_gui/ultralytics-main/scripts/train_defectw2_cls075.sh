#!/bin/bash
# Experiment 2: class_weights=[1.0,2.0] + cls=0.75 box=7.5 dfl=1.5
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cargodefect

PRETRAINED="runs/detect/runs/cargodefect/cargodefect_detect_finetune_from_baseline/weights/best.pt"
DATA_YAML="ultralytics/cfg/datasets/cargodefect-package.yaml"
NAME="cargodefect_detect_defectw2_cls075"

python -c "
from ultralytics import YOLO
m = YOLO('${PRETRAINED}')
if hasattr(m.model.args, '__dict__'):
    m.model.args.class_weights = [1.0, 2.0]
else:
    m.model.args['class_weights'] = [1.0, 2.0]
m.train(
    data='${DATA_YAML}',
    epochs=30, imgsz=640, batch=4, device=0, workers=4,
    project='runs/cargodefect', name='${NAME}', exist_ok=True,
    amp=True, cos_lr=True, optimizer='AdamW', lr0=0.0005, lrf=0.05,
    warmup_epochs=1, close_mosaic=20, patience=30, seed=0,
    deterministic=False, plots=False, val=False,
    erasing=0.0, mosaic=0.5, copy_paste=0.0,
    box=7.5, cls=0.75, dfl=1.5,
)
print('${NAME} DONE')
"
