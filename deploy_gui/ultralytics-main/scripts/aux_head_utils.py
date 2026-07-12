"""Shared helpers for quality/defect head debugging and standalone training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = "ultralytics/cfg/datasets/cargodefect-fusion.yaml"
DEFECT_NAMES_6 = ["none", "scratch", "crack", "dent", "stain", "anomaly"]
DEFECT_NAMES_5 = ["scratch", "crack", "dent", "stain", "none"]


def load_dataset_yaml():
    from ultralytics.data.utils import check_det_dataset

    return check_det_dataset(DATA_YAML)


def build_split_loader(mode: str = "val", batch: int = 8, augment: bool = False):
    from copy import deepcopy

    from ultralytics.data import build_cargodefect_dataset, build_dataloader
    from ultralytics.data.cargodefect import resolve_cargodefect_data
    from ultralytics.utils import DEFAULT_CFG

    data = resolve_cargodefect_data(load_dataset_yaml())
    args = deepcopy(DEFAULT_CFG)
    args.imgsz = 224
    if not augment:
        args.mosaic = 0.0
        args.mixup = 0.0
        args.cutmix = 0.0
        args.copy_paste = 0.0
        args.fliplr = 0.0
        args.flipud = 0.0
        args.hsv_h = 0.0
        args.hsv_s = 0.0
        args.hsv_v = 0.0
        args.degrees = 0.0
        args.translate = 0.0
        args.scale = 0.0
        args.shear = 0.0
        args.perspective = 0.0
        args.auto_augment = None
        args.erasing = 0.0
    split = data["val" if mode == "val" else "train"]
    ds = build_cargodefect_dataset(args, split, batch, data, mode=mode, stride=32)
    ds.imgsz = 224
    loader = build_dataloader(ds, batch=batch, workers=4, shuffle=mode == "train")
    return data, ds, loader


def remap_defect_label_6(label: int) -> int:
    """Map dataset defect index to standalone 6-class order: none, scratch, crack, dent, stain, anomaly."""
    mapping = {4: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    return mapping.get(int(label), 5)


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[int(t), int(p)] += 1
    total = max(len(y_true), 1)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {"cm": cm, "accuracy": acc, "labels": labels, "total": total}


def quality_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """OK=0, NG=1. Report False OK (NG->OK) and False NG (OK->NG)."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    ok_mask = y_true == 0
    ng_mask = y_true == 1
    fp_ok = int(((y_pred == 0) & ng_mask).sum())  # defect missed as OK
    fp_ng = int(((y_pred == 1) & ok_mask).sum())  # normal flagged NG
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "ok_accuracy": float(((y_pred == 0) & ok_mask).sum() / max(ok_mask.sum(), 1)),
        "ng_recall": float(((y_pred == 1) & ng_mask).sum() / max(ng_mask.sum(), 1)),
        "false_ok_rate": fp_ok / max(ng_mask.sum(), 1),
        "false_ng_rate": fp_ng / max(ok_mask.sum(), 1),
        "fn_ng_as_ok": fp_ok,
        "fp_ok_as_ng": fp_ng,
        "n_ok": int(ok_mask.sum()),
        "n_ng": int(ng_mask.sum()),
    }


def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], path: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_label_distribution_csv(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "total",
        "quality_ok",
        "quality_ng",
        "ng_ratio",
        "packaging",
        "mvtec_loco",
        "defect_none",
        "defect_scratch",
        "defect_crack",
        "defect_dent",
        "defect_stain",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            defect = row.get("defect", {})
            total = row.get("total", 0) or 1
            w.writerow(
                {
                    "mode": row["mode"],
                    "total": row["total"],
                    "quality_ok": row.get("quality_ok", 0),
                    "quality_ng": row.get("quality_ng", 0),
                    "ng_ratio": round(row.get("quality_ng", 0) / total, 4),
                    "packaging": row.get("source", {}).get("packaging", 0),
                    "mvtec_loco": row.get("source", {}).get("mvtec_loco", 0),
                    "defect_none": defect.get("none", 0),
                    "defect_scratch": defect.get("scratch", 0),
                    "defect_crack": defect.get("crack", 0),
                    "defect_dent": defect.get("dent", 0),
                    "defect_stain": defect.get("stain", 0),
                }
            )


def crop_for_defect(batch: dict, nc_cargo: int = 4, size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ROI crops and defect labels for a batch."""
    from ultralytics.nn.modules.defect_classifier import DefectClassifierHead

    batch = dict(batch)
    img = batch["img"]
    if img.dtype != torch.float32:
        batch["img"] = img.float() / 255.0
    head = DefectClassifierHead(nc_defect=6, roi_size=size)
    rois, targets = head.build_roi_batch(batch, nc_cargo)
    return rois, targets


def crop_for_quality(batch: dict, nc_cargo: int = 4, size: int = 224) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize full frame for image-level OK/NG (works for MVTec images without cargo boxes)."""
    import torch.nn.functional as F

    imgs = batch["img"]
    if imgs.dtype != torch.float32:
        imgs = imgs.float() / 255.0
    rois = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    quality = batch["quality_label"].view(-1).long()
    return rois, quality
