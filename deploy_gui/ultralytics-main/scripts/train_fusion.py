"""Train CargoDefect-YOLOv26 v2 (split-task inspection) on fusion dataset."""
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
    raise RuntimeError(
        "CUDA 不可用。请先安装 GPU 版 PyTorch，例如:\n"
        "  conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia\n"
        "或:\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    )

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

MODEL_YAML = "ultralytics/cfg/models/26/cargodefect-yolov26.yaml"
DATA_YAML = "ultralytics/cfg/datasets/cargodefect-fusion.yaml"
RUN_DIR = ROOT / "runs/detect/runs/cargodefect/fusion_v2"
LAST_CKPT = RUN_DIR / "weights/last.pt"

TRAIN_KWARGS = dict(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    device=0,
    workers=4,
    project="runs/cargodefect",
    name="fusion_v2",
    exist_ok=True,
    amp=True,
    cos_lr=True,
    grad_clip=1.0,
)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


resume_ckpt = str(LAST_CKPT) if LAST_CKPT.exists() else None
if resume_ckpt:
    print(f"Resuming from checkpoint: {resume_ckpt}")

batch = 8
while batch >= 1:
    try:
        print(f"Starting training with batch={batch}")
        model = YOLO(MODEL_YAML)
        train_kwargs = {**TRAIN_KWARGS, "batch": batch}
        if resume_ckpt:
            train_kwargs["resume"] = resume_ckpt
        model.train(**train_kwargs)
        break
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        if batch == 1:
            raise RuntimeError("CUDA OOM at batch=1; cannot reduce batch further.") from exc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        batch //= 2
        print(f"CUDA OOM detected, retrying with batch={batch}")
