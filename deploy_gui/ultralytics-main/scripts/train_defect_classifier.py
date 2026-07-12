#!/usr/bin/env python3
"""Standalone defect classifier trainer on ROI crops."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.defect_classifier import DefectClassifierHead
from scripts.aux_head_utils import (
    DEFECT_NAMES_6,
    build_split_loader,
    confusion_metrics,
    crop_for_defect,
    remap_defect_label_6,
    save_confusion_matrix_png,
)

OUT = ROOT / "results/quality_debug"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float() / 255.0
        rois, targets = crop_for_defect(batch, size=128)
        if rois.numel() == 0:
            continue
        logits = model.fc(model.encoder(rois).flatten(1))
        pred = logits.argmax(1)
        y_true.extend([remap_defect_label_6(int(t)) for t in targets.cpu().tolist()])
        y_pred.extend(pred.cpu().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    stats = confusion_metrics(y_true, y_pred, DEFECT_NAMES_6)
    # per-class recall
    recalls = {}
    for i, name in enumerate(DEFECT_NAMES_6):
        mask = y_true == i
        recalls[name] = float(((y_pred == i) & mask).sum() / max(mask.sum(), 1))
    stats["recall_per_class"] = recalls
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    _, _, train_loader = build_split_loader("train", batch=args.batch, augment=True)
    _, _, val_loader = build_split_loader("val", batch=args.batch, augment=False)

    model = DefectClassifierHead(nc_defect=6, roi_size=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    save_path = OUT / "defect_classifier_standalone.pt"
    OUT.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch["img"] = batch["img"].float() / 255.0
            rois, targets = crop_for_defect(batch, size=128)
            if rois.numel() == 0:
                continue
            targets6 = torch.tensor([remap_defect_label_6(int(t)) for t in targets], device=device, dtype=torch.long)
            logits = model.encoder(rois).flatten(1)
            logits = model.fc(logits)
            loss = criterion(logits, targets6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * rois.shape[0]
            n += rois.shape[0]
        stats = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} loss={total_loss/max(n,1):.4f} acc={stats['accuracy']:.3f}")
        if stats["accuracy"] >= best_acc:
            best_acc = stats["accuracy"]
            torch.save({"model": model.state_dict(), "metrics": stats}, save_path)

    stats = evaluate(model, val_loader, device)
    save_confusion_matrix_png(
        stats["cm"], DEFECT_NAMES_6, OUT / "defect_confusion_matrix.png", title="Defect Classifier (val)"
    )
    (OUT / "defect_metrics.json").write_text(json.dumps({k: v for k, v in stats.items() if k != "cm"}, indent=2, default=str))
    print(f"Best acc: {best_acc:.3f}; saved {save_path}")
    print("Per-class recall:", stats.get("recall_per_class"))


if __name__ == "__main__":
    main()
