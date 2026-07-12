#!/usr/bin/env python3
"""Standalone quality head trainer (OK/NG) on ROI/full-frame crops."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aux_head_utils import (
    build_split_loader,
    crop_for_quality,
    quality_binary_metrics,
    save_confusion_matrix_png,
)

OUT = ROOT / "results/quality_debug"


class QualityCropNet(nn.Module):
    """Lightweight OK/NG classifier on 224x224 crops."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        imgs = batch["img"].float() / 255.0
        rois, targets = crop_for_quality(batch, size=224)
        if rois.numel() == 0:
            continue
        logits = model(rois)
        pred = (torch.sigmoid(logits) >= threshold).long()
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    import numpy as np

    return quality_binary_metrics(np.array(y_true), np.array(y_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    _, _, train_loader = build_split_loader("train", batch=args.batch, augment=False)
    _, _, val_loader = build_split_loader("val", batch=args.batch, augment=False)

    from ultralytics.data.cargodefect import collect_label_statistics, resolve_cargodefect_data
    from scripts.aux_head_utils import load_dataset_yaml

    stats = collect_label_statistics(resolve_cargodefect_data(load_dataset_yaml()), mode="train")
    ok, ng = stats["quality_ok"], stats["quality_ng"]
    pos_weight = torch.tensor([max(ok / max(ng, 1), 1.0)], device=device)
    model = QualityCropNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Train OK={ok} NG={ng} pos_weight={pos_weight.item():.2f}")

    best_score = 0.0
    save_path = OUT / "quality_head_standalone.pt"
    OUT.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            imgs = batch["img"].float() / 255.0
            batch["img"] = imgs
            rois, targets = crop_for_quality(batch, size=224)
            if rois.numel() == 0:
                continue
            logits = model(rois)
            loss = criterion(logits, targets.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * rois.shape[0]
            n += rois.shape[0]
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device, args.threshold)
            score = metrics["accuracy"] if metrics["ng_recall"] >= 0.85 else metrics["ng_recall"] * 0.5
            print(
                f"epoch {epoch:03d} loss={total_loss/max(n,1):.4f} "
                f"acc={metrics['accuracy']:.3f} ng_recall={metrics['ng_recall']:.3f} "
                f"false_ok={metrics['false_ok_rate']:.3f} score={score:.3f}"
            )
            if score >= best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "metrics": metrics, "threshold": args.threshold}, save_path)
        elif epoch % 1 == 0:
            print(f"epoch {epoch:03d} loss={total_loss/max(n,1):.4f}")

    metrics = evaluate(model, val_loader, device, args.threshold)
    cm = __import__("numpy").zeros((2, 2), dtype=int)
    # re-run for cm
    import numpy as np

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch["img"] = batch["img"].float() / 255.0
            rois, targets = crop_for_quality(batch, size=224)
            if rois.numel() == 0:
                continue
            pred = (torch.sigmoid(model(rois)) >= args.threshold).long()
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    save_confusion_matrix_png(cm, ["OK", "NG"], OUT / "confusion_matrix_standalone.png", title="Standalone Quality Head")
    (OUT / "standalone_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Best score: {best_score:.3f}; saved {save_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
