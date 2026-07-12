#!/usr/bin/env python3
"""Run all 4 weight tuning experiments for CargoDefect-YOLOv26-Detect."""
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
    epochs=30,
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
    patience=30,
    seed=0,
    deterministic=False,
    plots=False,
    val=False,
    pretrained=False,
    erasing=0.0,
    mosaic=0.5,
    copy_paste=0.0,
)


class DefectWeightTrainer(DetectionTrainer):
    """Log custom weight settings and ensure criterion is refreshed."""

    def set_model_attributes(self):
        super().set_model_attributes()
        cw = getattr(self.args, "class_weights", None)
        if cw:
            LOGGER.info(f"Class weights (explicit): {cw}")
        lw = getattr(self.args, "level_loss_weights", None)
        if lw:
            LOGGER.info(f"Level loss weights: {lw}")
        if hasattr(self.model, "criterion"):
            self.model.criterion = None


def train_experiment(name, **extra):
    m = YOLO(BASE_PT)
    m.train(name=name, trainer=DefectWeightTrainer, **COMMON, **extra)
    print(f"{name} DONE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.exp == 1:
        train_experiment("cargodefect_detect_defectw2", class_weights=[1.0, 2.0])
    elif args.exp == 2:
        train_experiment(
            "cargodefect_detect_defectw2_cls075",
            class_weights=[1.0, 2.0],
            box=7.5,
            cls=0.75,
            dfl=1.5,
        )
    elif args.exp == 3:
        m = YOLO(BASE_PT)
        pgme = m.model.model[2]
        if not hasattr(pgme, "alpha_pgme"):
            pgme.register_parameter("alpha_pgme", torch.nn.Parameter(torch.tensor(1.0)))
            print("PGME alpha_pgme added (BC for old checkpoint)")
        print(f"PGME alpha_pgme pre: {pgme.alpha_pgme.item():.4f}")
        with torch.no_grad():
            pgme.alpha_pgme.fill_(0.5)
        print(f"PGME alpha_pgme post: {pgme.alpha_pgme.item():.4f}")
        m.train(name="cargodefect_detect_pgme_alpha05", trainer=DefectWeightTrainer, **COMMON)
        print(f"PGME alpha_pgme final: {m.model.model[2].alpha_pgme.item():.4f}")
        print("cargodefect_detect_pgme_alpha05 DONE")
    elif args.exp == 4:
        train_experiment("cargodefect_detect_p2loss15", level_loss_weights=[1.5, 1.0, 1.0, 0.75])
