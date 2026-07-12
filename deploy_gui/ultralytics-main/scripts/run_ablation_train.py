"""Train a single ablation experiment from configs/ablation/*.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ablation YAML under configs/ablation/")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_yaml = cfg["model"]
    data_yaml = cfg["data"]
    train = cfg["train"]
    run_name = train.pop("run_name", cfg.get("name", "ablation"))
    batch = int(train.pop("batch", 8))

    print(f"Experiment {cfg.get('experiment', '?')}: {cfg.get('description', cfg_path.name)}")
    print(f"Model: {model_yaml}")

    while batch >= 1:
        try:
            model = YOLO(model_yaml)
            model.train(data=data_yaml, name=run_name, batch=batch, **train)
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            if batch == 1:
                raise RuntimeError("CUDA OOM at batch=1") from exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch //= 2
            print(f"CUDA OOM, retrying with batch={batch}")


if __name__ == "__main__":
    main()
