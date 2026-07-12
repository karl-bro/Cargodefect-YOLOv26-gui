#!/usr/bin/env bash
# Sequential ablation training for CargoDefect-YOLOv26 (A→F).
# All experiments share identical train/val split, hyperparameters, and augmentation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v conda &>/dev/null; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cargodefect 2>/dev/null || true
fi

export KMP_DUPLICATE_LIB_OK=TRUE
LOG_DIR="$ROOT/runs/cargodefect/ablation_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/ablation/yolov26_baseline.yaml"
  "configs/ablation/cargodefect_roi_classifier.yaml"
  "configs/ablation/cargodefect_quality_head.yaml"
  "configs/ablation/cargodefect_edge_enhance.yaml"
  "configs/ablation/cargodefect_dwt_enhance.yaml"
  "configs/ablation/cargodefect_full.yaml"
)

run_one() {
  local cfg="$1"
  local name
  name="$(basename "$cfg" .yaml)"
  echo "========================================"
  echo "Ablation: $name"
  echo "Config: $cfg"
  echo "========================================"
  python scripts/run_ablation_train.py \
    --config "$cfg" \
    2>&1 | tee "$LOG_DIR/${name}.log"
}

for cfg in "${CONFIGS[@]}"; do
  run_one "$cfg"
done

echo "All ablation runs finished. Logs: $LOG_DIR"
