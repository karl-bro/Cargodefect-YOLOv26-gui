#!/usr/bin/env python3
"""Threshold sweep for CargoDefect quality head on fusion_v3 weights."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from scripts.aux_head_utils import build_split_loader

OUT = ROOT / "results/quality_debug_v3"
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_WEIGHTS = ROOT / "runs/detect/runs/cargodefect/fusion_v3/weights/best.pt"

THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Compute binary classification metrics at given threshold. OK=0, NG=1."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = len(y_true)
    acc = (tp + tn) / total if total else 0
    ok_acc = tn / max(tn + fp, 1)
    ng_rec = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "ok_accuracy": round(ok_acc, 4),
        "ng_recall": round(ng_rec, 4),
        "false_ok_rate": round(fn / max(tp + fn, 1), 4),
        "false_ng_rate": round(fp / max(tn + fp, 1), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.weights} ...")
    yolo = YOLO(args.weights)
    model = yolo.model.eval().to(device)
    head = model.model[-1]
    if head.quality is None:
        raise RuntimeError("Checkpoint has no quality head.")

    _, _, loader = build_split_loader("val", batch=8, augment=False)
    y_true_list, y_prob_list = [], []

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
            gt = batch["quality_label"].view(-1).cpu().numpy()
            pb = probs.cpu().numpy()
            y_true_list.extend(gt.tolist())
            y_prob_list.extend(pb.tolist())

    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_prob_list, dtype=float)
    print(f"Collected {len(y_true)} samples (OK={int((y_true==0).sum())}, NG={int((y_true==1).sum())})")

    # Sweep
    results = []
    best = None
    best_score = -1

    for th in THRESHOLDS:
        m = compute_metrics(y_true, y_prob, th)
        results.append(m)

        # Scoring: prioritize NG Recall >= 0.85, OK Accuracy >= 0.75
        score = 0.0
        if m["ng_recall"] >= 0.85 and m["ok_accuracy"] >= 0.75:
            score = m["accuracy"] * 0.5 + m["f1"] * 0.3 + (1 - m["false_ok_rate"]) * 0.1 + (1 - m["false_ng_rate"]) * 0.1
        elif m["ng_recall"] >= 0.85:
            score = m["ng_recall"] * 0.4 + m["f1"] * 0.3 + (1 - m["false_ok_rate"]) * 0.2 + m["accuracy"] * 0.1
        else:
            score = m["f1"] * 0.4 + m["accuracy"] * 0.3 + m["ng_recall"] * 0.3

        if score > best_score:
            best_score = score
            best = m

    # Save CSV
    csv_path = OUT / "threshold_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in results[0] if k not in ("tp", "fp", "tn", "fn", "total")])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if k not in ("tp", "fp", "tn", "fn", "total")})

    # Save detailed JSON
    (OUT / "threshold_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Report
    feasible = [r for r in results if r["ng_recall"] >= 0.85 and r["ok_accuracy"] >= 0.75]
    relaxed = [r for r in results if r["ng_recall"] >= 0.85 and r["ok_accuracy"] >= 0.50]

    report_lines = [
        "# Quality Head Threshold Sweep Report",
        "",
        f"- Weights: `{args.weights}`",
        f"- Val samples: {len(y_true)} (OK={int((y_true==0).sum())}, NG={int((y_true==1).sum())})",
        "",
        "## Best Threshold (by composite score)",
        f"Threshold: **{best['threshold']}**",
        f"- Accuracy: {best['accuracy']:.4f}",
        f"- OK Accuracy: {best['ok_accuracy']:.4f}",
        f"- NG Recall: {best['ng_recall']:.4f}",
        f"- False OK Rate: {best['false_ok_rate']:.4f}",
        f"- False NG Rate: {best['false_ng_rate']:.4f}",
        f"- Precision: {best['precision']:.4f}",
        f"- Recall: {best['recall']:.4f}",
        f"- F1: {best['f1']:.4f}",
    ]

    if feasible:
        report_lines += [
            "",
            "## Thresholds meeting NG Recall >= 0.85 AND OK Accuracy >= 0.75",
            "",
            "| Thresh | Acc | OK Acc | NG Rec | False OK | False NG | Prec | Rec | F1 |",
            "|--------|-----|--------|--------|----------|----------|------|-----|----|",
        ]
        for r in feasible:
            report_lines.append(f"| {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['ok_accuracy']:.4f} | {r['ng_recall']:.4f} | {r['false_ok_rate']:.4f} | {r['false_ng_rate']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    else:
        report_lines += [
            "",
            "## No threshold meets both NG Recall >= 0.85 AND OK Accuracy >= 0.75",
            "Model is too biased towards NG. Retraining with lower pos_weight is needed.",
        ]
        if relaxed:
            report_lines += [
                "",
                "## Relaxed: NG Recall >= 0.85 AND OK Accuracy >= 0.50",
                "",
                "| Thresh | Acc | OK Acc | NG Rec | False OK | False NG | Prec | Rec | F1 |",
                "|--------|-----|--------|--------|----------|----------|------|-----|----|",
            ]
            for r in relaxed:
                report_lines.append(f"| {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['ok_accuracy']:.4f} | {r['ng_recall']:.4f} | {r['false_ok_rate']:.4f} | {r['false_ng_rate']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")

    report_lines += [
        "",
        "## Full Sweep Table",
        "",
        "| Thresh | Acc | OK Acc | NG Rec | False OK | False NG | Prec | Rec | F1 |",
        "|--------|-----|--------|--------|----------|----------|------|-----|----|",
    ]
    for r in results:
        report_lines.append(f"| {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['ok_accuracy']:.4f} | {r['ng_recall']:.4f} | {r['false_ok_rate']:.4f} | {r['false_ng_rate']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")

    report_lines += [
        "",
        "## Artifacts",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{OUT / 'threshold_sweep.json'}`",
    ]

    (OUT / "best_threshold_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved {csv_path}")
    print(f"Saved {OUT / 'best_threshold_report.md'}")
    print(f"Best threshold: {best['threshold']}, Accuracy: {best['accuracy']:.4f}")
    if not feasible:
        print("WARNING: No threshold meets NG Recall >= 0.85 AND OK Accuracy >= 0.75")


if __name__ == "__main__":
    main()
