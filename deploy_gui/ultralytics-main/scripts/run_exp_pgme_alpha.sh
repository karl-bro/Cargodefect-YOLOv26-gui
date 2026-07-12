#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cargodefect

python -c "
from ultralytics import YOLO
import torch, shutil

m = YOLO('runs/detect/runs/cargodefect/cargodefect_detect_finetune_from_baseline/weights/best.pt')
pgme = m.model.model[2]
print(f'Pre: alpha_pgme={pgme.alpha_pgme.item():.4f}')
pgme.alpha_pgme.data.fill_(0.5)
print(f'Post: alpha_pgme={pgme.alpha_pgme.item():.4f}')

m.train(
    data='ultralytics/cfg/datasets/cargodefect-package.yaml',
    epochs=30, imgsz=640, batch=4, device=0, workers=4,
    project='runs/cargodefect', name='cargodefect_detect_pgme_alpha05', exist_ok=True,
    amp=True, cos_lr=True, optimizer='AdamW', lr0=0.0005, lrf=0.05,
    warmup_epochs=1, close_mosaic=20, patience=30, seed=0,
    deterministic=False, plots=False, val=False,
    erasing=0.0, mosaic=0.5, copy_paste=0.0,
)
print(f'Final alpha_pgme={m.model.model[2].alpha_pgme.item():.4f}')
print('cargodefect_detect_pgme_alpha05 DONE')
" > runs/cargodefect/train_pgme_alpha05.log 2>&1
