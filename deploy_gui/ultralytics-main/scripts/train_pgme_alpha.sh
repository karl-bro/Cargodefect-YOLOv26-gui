#!/bin/bash
# Experiment 3: PGME alpha_pgme initialized at 0.5 (learnable)
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cargodefect

PRETRAINED="runs/detect/runs/cargodefect/cargodefect_detect_finetune_from_baseline/weights/best.pt"
DATA_YAML="ultralytics/cfg/datasets/cargodefect-package.yaml"
NAME="cargodefect_detect_pgme_alpha05"
INIT_PT="weights/cargodefect_detect_pgme_alpha05_init.pt"

python -c "
from ultralytics import YOLO
import torch

# Build from finetune weight, then override alpha_pgme
m = YOLO('${PRETRAINED}')
pgme = m.model.model[2]
old_alpha = pgme.alpha_pgme.item()
print(f'Pre-training alpha_pgme: {old_alpha:.4f}')
pgme.alpha_pgme.data.fill_(0.5)
print(f'New alpha_pgme: {pgme.alpha_pgme.item():.4f}')
# Save as intermediate weight
torch.save(m.model.state_dict(), '${INIT_PT}')
print(f'Saved alpha=0.5 init to ${INIT_PT}')

# Now train
m.train(
    data='${DATA_YAML}',
    epochs=30, imgsz=640, batch=4, device=0, workers=4,
    project='runs/cargodefect', name='${NAME}', exist_ok=True,
    amp=True, cos_lr=True, optimizer='AdamW', lr0=0.0005, lrf=0.05,
    warmup_epochs=1, close_mosaic=20, patience=30, seed=0,
    deterministic=False, plots=False, val=False,
    erasing=0.0, mosaic=0.5, copy_paste=0.0,
)
print('${NAME} DONE')
# Show final alpha_pgme
final_alpha = m.model.model[2].alpha_pgme.item()
print(f'Final alpha_pgme after training: {final_alpha:.4f}')
"
