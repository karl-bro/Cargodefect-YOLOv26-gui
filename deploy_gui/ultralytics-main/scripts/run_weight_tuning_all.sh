#!/bin/bash
# Run all 4 weight tuning experiments sequentially, then validate and compare.
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cargodefect

mkdir -p runs/cargodefect results/weight_tuning_compare

BASE="runs/detect/runs/cargodefect/cargodefect_detect_finetune_from_baseline/weights/best.pt"
DATA="ultralytics/cfg/datasets/cargodefect-package.yaml"

run_val() {
  local name=$1
  local pt="runs/detect/runs/cargodefect/${name}/weights/best.pt"
  if [ -f "$pt" ]; then
    echo "=== VAL: $name ==="
    yolo detect val model="$pt" data="$DATA" imgsz=640 batch=4 device=0 \
      > "runs/cargodefect/val_${name}.log" 2>&1 || true
  fi
}

for exp in 1 2 3 4; do
  echo "========== Experiment $exp =========="
  python scripts/run_all_weight_experiments.py --exp "$exp" \
    > "runs/cargodefect/train_exp${exp}.log" 2>&1
  echo "Experiment $exp finished"
done

run_val "cargodefect_detect_defectw2"
run_val "cargodefect_detect_defectw2_cls075"
run_val "cargodefect_detect_pgme_alpha05"
run_val "cargodefect_detect_p2loss15"

python scripts/weight_tuning_compare.py > results/weight_tuning_compare/run.log 2>&1
echo "All experiments and comparison complete."
