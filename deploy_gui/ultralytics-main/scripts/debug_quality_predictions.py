#!/usr/bin/env python3
"""Debug CargoDefect quality head predictions on val set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from scripts.aux_head_utils import (
    DATA_YAML,
    build_split_loader,
    quality_binary_metrics,
    save_confusion_matrix_png,
)

OUT = ROOT / "results/quality_debug"
DEFAULT_WEIGHTS = ROOT / "runs/detect/runs/cargodefect/fusion_v2/weights/best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-samples", type=int, default=40)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(args.weights)
    model = yolo.model.eval().to(device)
    head = model.model[-1]
    if head.quality is None:
        raise RuntimeError("Checkpoint has no quality head.")

    _, ds, loader = build_split_loader("val", batch=8, augment=False)
    y_true, y_pred, y_prob = [], [], []
    sample_dir = OUT / "prediction_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            imgs = batch["img"].float() / 255.0
            preds = model.predict(imgs)
            feats = preds[1]["feats"] if isinstance(preds, tuple) else None
            if feats is None:
                continue
            logits = head.quality(feats)
            if isinstance(logits, dict):
                logits = logits["logits"]
            probs = torch.sigmoid(logits[:, 0])
            verdict = (probs >= args.threshold).long()
            gt = batch["quality_label"].view(-1).cpu().numpy()
            pr = verdict.cpu().numpy()
            pb = probs.cpu().numpy()
            y_true.extend(gt.tolist())
            y_pred.extend(pr.tolist())
            y_prob.extend(pb.tolist())

            if saved < args.max_samples:
                for i in range(min(imgs.shape[0], args.max_samples - saved)):
                    img = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    tag = f"gt={'NG' if gt[i]==1 else 'OK'} pred={'NG' if pr[i]==1 else 'OK'} p={pb[i]:.2f}"
                    color = (0, 0, 255) if pr[i] != gt[i] else (0, 180, 0)
                    cv2.putText(img, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.imwrite(str(sample_dir / f"sample_{saved:03d}.jpg"), img)
                    saved += 1

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    metrics = quality_binary_metrics(y_true, y_pred)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    save_confusion_matrix_png(cm, ["OK", "NG"], OUT / "confusion_matrix.png", title="Quality Head (val)")

    report = [
        "# Quality Debug Report",
        "",
        f"- Weights: `{args.weights}`",
        f"- Threshold: {args.threshold}",
        f"- Val samples: {len(y_true)}",
        "",
        "## Metrics",
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- OK Accuracy: {metrics['ok_accuracy']:.4f}",
        f"- NG Recall: **{metrics['ng_recall']:.4f}**",
        f"- False OK Rate (NG missed): **{metrics['false_ok_rate']:.4f}** ({metrics['fn_ng_as_ok']} / {metrics['n_ng']})",
        f"- False NG Rate (OK flagged): {metrics['false_ng_rate']:.4f} ({metrics['fp_ok_as_ng']} / {metrics['n_ok']})",
        "",
        "## Root cause (fixed in dataset pipeline)",
        "- Training previously used only MVTec `train/good` + packaging (all OK=0).",
        "- Enable `mvtec_train_include_anomalies: true` in `cargodefect-fusion.yaml` to add NG supervision.",
        "- Use `quality_pos_weight` + retrain or `scripts/train_quality_head.py` for standalone warmup.",
        "",
        "## Artifacts",
        f"- `{OUT / 'confusion_matrix.png'}`",
        f"- `{OUT / 'prediction_samples/'}`",
        f"- `{OUT / 'label_distribution.csv'}`",
    ]
    (OUT / "quality_debug_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {OUT / 'quality_debug_report.md'}")


if __name__ == "__main__":
    main()
