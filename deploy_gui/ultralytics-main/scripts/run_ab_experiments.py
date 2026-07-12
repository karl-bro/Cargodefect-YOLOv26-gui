#!/usr/bin/env python3
"""Experiment A (Focal Loss) then B (imgsz=1280 high resolution) for defect recall."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

BASE_PT = "runs/detect/runs/cargodefect/cargodefect_detect_finetune_from_baseline/weights/best.pt"
DATA_YAML = "ultralytics/cfg/datasets/cargodefect-package.yaml"

COMMON = dict(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    workers=4,
    project="runs/cargodefect",
    exist_ok=True,
    amp=True,
    cos_lr=True,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.05,
    warmup_epochs=1,
    close_mosaic=20,
    patience=50,
    seed=0,
    deterministic=False,
    plots=False,
    erasing=0.0,
    mosaic=0.5,
    copy_paste=0.0,
    pretrained=False,
)


def train_and_val(name, **kwargs):
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")
    m = YOLO(BASE_PT)
    if kwargs.get("fl_gamma", 0) > 0:
        LOGGER.info(f"Focal Loss enabled: fl_gamma={kwargs['fl_gamma']}")
    cfg = dict(COMMON)
    cfg.update(kwargs)
    m.train(name=name, val=True, **cfg)

    best_pt = f"runs/detect/runs/cargodefect/{name}/weights/best.pt"
    print(f"\n{'='*60}")
    print(f"VALIDATING: {name}")
    print(f"{'='*60}")
    import subprocess

    subprocess.run(
        [
            sys.executable, "-m", "ultralytics",
            "detect", "val", f"model={best_pt}", f"data={DATA_YAML}",
            "imgsz=640", "batch=4", "device=0",
        ],
        check=False,
    )
    print(f"\n{name} DONE\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["A", "B"])
    args = parser.parse_args()

    if args.exp == "A":
        train_and_val("cargodefect_detect_focal2", fl_gamma=2.0)
    elif args.exp == "B":
        train_and_val(
            "cargodefect_detect_imgsz1280",
            imgsz=1280,
            batch=2,
        )
