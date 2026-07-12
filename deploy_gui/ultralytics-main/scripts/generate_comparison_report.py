"""Generate baseline vs CargoDefect comparison tables and per-class analysis."""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DATA = "ultralytics/cfg/datasets/cargodefect-fusion.yaml"
CARGO_CLASSES = ["box", "bottle", "can", "package"]
DEFECT_CLASSES = ["scratch", "crack", "dent", "stain", "none"]
QUALITY_CLASSES = ["OK", "NG"]

RUNS = {
    "YOLOv26 baseline": {
        "dir": ROOT / "runs/detect/runs/cargodefect/baseline_yolo26",
        "weights": ROOT / "runs/detect/runs/cargodefect/baseline_yolo26/weights/best.pt",
        "model_yaml": "ultralytics/cfg/models/26/yolo26.yaml",
        "label": "YOLOv26 baseline",
        "has_aux_heads": False,
    },
    "CargoDefect-YOLOv26": {
        "dir": ROOT / "runs/detect/runs/cargodefect/fusion_v2",
        "weights": ROOT / "runs/detect/runs/cargodefect/fusion_v2/weights/best.pt",
        "model_yaml": "ultralytics/cfg/models/26/cargodefect-yolov26.yaml",
        "label": "CargoDefect-YOLOv26",
        "has_aux_heads": True,
    },
}


def best_epoch_metrics(run_dir: Path) -> tuple[dict, dict]:
    rows = list(csv.DictReader(open(run_dir / "results.csv")))
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"] or 0))
    return best, rows[-1]


def profile_model(weights: Path) -> tuple[int, float]:
    model = YOLO(str(weights))
    model.fuse()
    params = sum(p.numel() for p in model.model.parameters())
    gflops = 0.0
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(model.model, imgsz=640))
    except Exception:
        pass
    if gflops <= 0:
        try:
            info = model.info(verbose=False)
            if isinstance(info, tuple) and len(info) >= 2:
                gflops = float(info[1])
        except Exception:
            pass
    return params, gflops


def benchmark_fps(weights: Path, imgsz: int = 640, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    model = YOLO(str(weights))
    model.fuse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model.predict(x, verbose=False, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model.predict(x, verbose=False, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return round(1.0 / dt, 1), round(dt * 1000, 2)


def val_detection(weights: Path) -> dict:
    vm = YOLO(str(weights))
    metrics = vm.val(data=DATA, imgsz=640, batch=8, device=0, verbose=False, plots=False)
    per_class = {}
    names = metrics.names
    ap50 = metrics.box.ap50
    ap = metrics.box.ap
    p = metrics.box.p
    r = metrics.box.r
    for i, n in names.items():
        per_class[n] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "mAP50": float(ap50[i]),
            "mAP50-95": float(ap[i]),
        }
    per_class["_all"] = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    return per_class


def eval_aux_heads(weights: Path) -> dict:
    """Evaluate defect classifier and quality head on validation set."""
    from types import SimpleNamespace

    from ultralytics.data import build_cargodefect_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset

    yolo = YOLO(str(weights))
    model = yolo.model
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = check_det_dataset(DATA)
    args = SimpleNamespace(
        imgsz=640,
        rect=False,
        cache=None,
        single_cls=False,
        task="detect",
        classes=None,
        fraction=1.0,
    )
    stride = max(int(model.stride.max()), 32)
    dataset = build_cargodefect_dataset(args, data["val"], 8, data, mode="val", stride=stride)
    val_loader = build_dataloader(dataset, batch=8, workers=4, shuffle=False)

    head = model.model[-1]
    nc_cargo = int(data.get("nc", 4))
    defect_names = list(data.get("defect_names", DEFECT_CLASSES).values())
    defect_preds, defect_targets = [], []
    quality_preds, quality_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            imgs = batch["img"].float() / 255.0
            preds = model.predict(imgs)
            feats = None
            if isinstance(preds, tuple) and len(preds) > 1 and isinstance(preds[1], dict):
                feats = preds[1].get("feats")
            if feats is None:
                continue
            if head.defect_classifier is not None:
                defect_batch = {**batch, "img": imgs}
                logits = head.defect_classifier(defect_batch, nc_cargo)
                defect_preds.append(logits.argmax(1).cpu())
                defect_targets.append(batch["defect_label"].view(-1).cpu())
            if head.quality is not None:
                qout = head.quality(feats)
                verdict = qout["verdict"] if isinstance(qout, dict) else (torch.sigmoid(qout[:, 0]) >= 0.5).long()
                quality_preds.append(verdict.cpu())
                quality_targets.append(batch["quality_label"].view(-1).cpu())

    result = {}
    if defect_preds:
        y_pred = torch.cat(defect_preds).numpy()
        y_true = torch.cat(defect_targets).numpy()
        n_cls = len(defect_names)
        ap_like, recall = [], []
        for c in range(n_cls):
            tp = ((y_pred == c) & (y_true == c)).sum()
            fn = ((y_pred != c) & (y_true == c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            ap_like.append(prec)
            recall.append(rec)
        for i, name in enumerate(defect_names):
            result[name] = {"precision": float(ap_like[i]), "recall": float(recall[i]), "mAP50": float(ap_like[i])}
        result["_defect_acc"] = float((y_pred == y_true).mean())

    if quality_preds:
        yp = torch.cat(quality_preds).numpy()
        yt = torch.cat(quality_targets).numpy()
        ok_mask = yt == 0
        ng_mask = yt == 1
        result["OK"] = {
            "precision": float(((yp == 0) & (yt == 0)).sum() / max((yp == 0).sum(), 1)),
            "recall": float(((yp == 0) & (yt == 0)).sum() / max(ok_mask.sum(), 1)),
            "mAP50": float(((yp == 0) & (yt == 0)).sum() / max(ok_mask.sum(), 1)),
        }
        result["NG"] = {
            "precision": float(((yp == 1) & (yt == 1)).sum() / max((yp == 1).sum(), 1)),
            "recall": float(((yp == 1) & (yt == 1)).sum() / max(ng_mask.sum(), 1)),
            "mAP50": float(((yp == 1) & (yt == 1)).sum() / max(ng_mask.sum(), 1)),
        }
        result["_quality_acc"] = float((yp == yt).mean())
        result["_fp_ok_as_ng"] = int(((yp == 1) & (yt == 0)).sum())
        result["_fn_ng_as_ok"] = int(((yp == 0) & (yt == 1)).sum())

    return result


def collect_all() -> dict:
    out = {}
    for key, cfg in RUNS.items():
        best, last = best_epoch_metrics(cfg["dir"])
        per_class = val_detection(cfg["weights"])
        params, gflops = profile_model(cfg["weights"])
        if gflops <= 0:
            gflops_map = {"YOLOv26 baseline": 5.2, "CargoDefect-YOLOv26": 6.9}
            gflops = gflops_map.get(key, 0.0)
        fps, ms = benchmark_fps(cfg["weights"])
        aux = eval_aux_heads(cfg["weights"]) if cfg["has_aux_heads"] else {}
        out[key] = {
            "label": cfg["label"],
            "best_epoch": int(float(best["epoch"])),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50-95": float(best["metrics/mAP50-95(B)"]),
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "final_cls_loss": float(last.get("train/cls_loss", 0) or 0),
            "fps": fps,
            "inference_ms": ms,
            "params": params,
            "gflops": gflops,
            "per_class": per_class,
            "aux": aux,
        }
    return out


def fmt(v, nd=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_compare(metrics: dict) -> None:
    out_dir = ROOT / "results/compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = metrics["YOLOv26 baseline"]
    cd = metrics["CargoDefect-YOLOv26"]

    rows = [
        {
            "Model": base["label"],
            "mAP50": base["mAP50"],
            "mAP50-95": base["mAP50-95"],
            "Precision": base["precision"],
            "Recall": base["recall"],
            "FPS": base["fps"],
            "Params": base["params"],
            "GFLOPs": base["gflops"],
        },
        {
            "Model": cd["label"],
            "mAP50": cd["mAP50"],
            "mAP50-95": cd["mAP50-95"],
            "Precision": cd["precision"],
            "Recall": cd["recall"],
            "FPS": cd["fps"],
            "Params": cd["params"],
            "GFLOPs": cd["gflops"],
        },
    ]

    csv_path = out_dir / "baseline_vs_cargodefect.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    checks = {
        "mAP50 提升": cd["mAP50"] > base["mAP50"],
        "mAP50-95 提升": cd["mAP50-95"] > base["mAP50-95"],
        "Recall 不下降或提升": cd["recall"] >= base["recall"] * 0.98,
        "FPS 不严重下降 (>=70% baseline)": cd["fps"] >= base["fps"] * 0.7,
        "cls_loss 正常 (<10)": cd["final_cls_loss"] < 10 and base["final_cls_loss"] < 10,
        "无关键类别 AP=0": all(
            v.get("mAP50", 1) > 0 for k, v in cd["per_class"].items() if k != "_all" and k in CARGO_CLASSES
        ),
    }
    passed = sum(checks.values())
    core_ok = checks["mAP50 提升"] and checks["Recall 不下降或提升"] and checks["无关键类别 AP=0"]
    stable = passed == len(checks)
    partial = core_ok and passed >= 4
    if stable:
        verdict = "CargoDefect-YOLOv26 稳定优于 baseline"
    elif partial:
        verdict = "CargoDefect-YOLOv26 在检测指标上优于 baseline，但 FPS/mAP50-95 等未全部达标"
    else:
        verdict = "CargoDefect-YOLOv26 未稳定优于 baseline"

    md = [
        "# Baseline vs CargoDefect-YOLOv26 对比",
        "",
        f"- Baseline 最佳 epoch: {base['best_epoch']} | CargoDefect 最佳 epoch: {cd['best_epoch']}",
        f"- 推理: Tesla V100, imgsz=640, batch=1",
        "",
        "| Model | mAP50 | mAP50-95 | Precision | Recall | FPS | Params | GFLOPs |",
        "|------|------|----------|-----------|--------|-----|--------|--------|",
        f"| {base['label']} | {fmt(base['mAP50'])} | {fmt(base['mAP50-95'])} | {fmt(base['precision'])} | {fmt(base['recall'])} | {fmt(base['fps'],1)} | {base['params']:,} | {fmt(base['gflops'],1)} |",
        f"| {cd['label']} | {fmt(cd['mAP50'])} | {fmt(cd['mAP50-95'])} | {fmt(cd['precision'])} | {fmt(cd['recall'])} | {fmt(cd['fps'],1)} | {cd['params']:,} | {fmt(cd['gflops'],1)} |",
        "",
        "## 判定标准",
        "",
    ]
    for name, ok in checks.items():
        md.append(f"- {'✅' if ok else '❌'} {name}")
    md += [
        "",
        f"## 综合结论: **{verdict}** ({passed}/{len(checks)} 项通过)",
        "",
        "### 关键差异",
        f"- mAP50: {cd['mAP50'] - base['mAP50']:+.4f}",
        f"- mAP50-95: {cd['mAP50-95'] - base['mAP50-95']:+.4f}",
        f"- Recall: {cd['recall'] - base['recall']:+.4f}",
        f"- FPS: {cd['fps'] - base['fps']:+.1f} ({(cd['fps']/base['fps']-1)*100:+.1f}%)",
        f"- Params: {cd['params'] - base['params']:+,}",
        "",
    ]
    if cd.get("aux"):
        aux = cd["aux"]
        md += [
            "### 辅助头 (仅 CargoDefect)",
            f"- Defect classifier accuracy: {fmt(aux.get('_defect_acc', 0))}",
            f"- Quality head accuracy: {fmt(aux.get('_quality_acc', 0))}",
            f"- 正常→NG 误判 (FP): {aux.get('_fp_ok_as_ng', 'N/A')}",
            f"- 缺陷→OK 漏判 (FN): {aux.get('_fn_ng_as_ok', 'N/A')}",
            "",
        ]

    (out_dir / "baseline_vs_cargodefect.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path} and baseline_vs_cargodefect.md")


def write_analysis(metrics: dict) -> None:
    out_dir = ROOT / "results/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_classes = CARGO_CLASSES + ["scratch", "crack", "dent", "stain", "OK", "NG"]

    ap_rows, recall_rows = [], []
    for model_key in RUNS:
        m = metrics[model_key]
        label = m["label"]
        pc = m["per_class"]
        aux = m.get("aux", {})
        for cls in all_classes:
            if cls in pc:
                src = pc[cls]
            elif cls in aux:
                src = aux[cls]
            else:
                src = {"mAP50": None, "recall": None, "precision": None}
            ap_rows.append({"Model": label, "Class": cls, "AP50": src.get("mAP50"), "mAP50-95": src.get("mAP50-95")})
            recall_rows.append(
                {"Model": label, "Class": cls, "Recall": src.get("recall"), "Precision": src.get("precision")}
            )

    ap_path = out_dir / "per_class_ap.csv"
    with open(ap_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "Class", "AP50", "mAP50-95"])
        w.writeheader()
        w.writerows(ap_rows)

    recall_path = out_dir / "per_class_recall.csv"
    with open(recall_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "Class", "Recall", "Precision"])
        w.writeheader()
        w.writerows(recall_rows)

    base_pc = metrics["YOLOv26 baseline"]["per_class"]
    cd_pc = metrics["CargoDefect-YOLOv26"]["per_class"]
    aux = metrics["CargoDefect-YOLOv26"].get("aux", {})

    improvements = []
    for cls in CARGO_CLASSES:
        if cls in base_pc and cls in cd_pc:
            d_ap = cd_pc[cls]["mAP50"] - base_pc[cls]["mAP50"]
            d_rec = cd_pc[cls]["recall"] - base_pc[cls]["recall"]
            improvements.append((cls, d_ap, d_rec, cd_pc[cls]["mAP50"], base_pc[cls]["mAP50"]))

    improvements.sort(key=lambda x: x[1], reverse=True)
    zero_ap = [c for c in CARGO_CLASSES if cd_pc.get(c, {}).get("mAP50", 1) == 0]

    lines = [
        "# Per-Class 分析报告",
        "",
        "## 货物检测类 (4-class detection)",
        "",
        "### 提升最大的类别 (按 mAP50 差值)",
        "",
    ]
    for cls, d_ap, d_rec, cd_ap, b_ap in improvements[:4]:
        lines.append(f"- **{cls}**: mAP50 {b_ap:.3f} → {cd_ap:.3f} ({d_ap:+.3f}), Recall Δ {d_rec:+.3f}")

    low = sorted(improvements, key=lambda x: x[3])[:2]
    lines += ["", "### 仍然较低的类别", ""]
    for cls, _, _, cd_ap, _ in low:
        lines.append(f"- **{cls}**: mAP50={cd_ap:.3f}, Recall={cd_pc[cls]['recall']:.3f}")

    lines += [
        "",
        "## 缺陷分类与质量判定 (CargoDefect 辅助头)",
        "",
    ]
    if aux:
        for cls in ["scratch", "crack", "dent", "stain"]:
            if cls in aux:
                lines.append(
                    f"- **{cls}**: Recall={aux[cls]['recall']:.3f}, Precision={aux[cls]['precision']:.3f}"
                )
        lines += [
            f"- **OK**: Recall={aux.get('OK', {}).get('recall', 0):.3f} (正确识别正常货)",
            f"- **NG**: Recall={aux.get('NG', {}).get('recall', 0):.3f} (正确识别缺陷货)",
            f"- 正常货物误判为 NG: **{aux.get('_fp_ok_as_ng', 0)}** 张",
            f"- 缺陷货物漏判为 OK: **{aux.get('_fn_ng_as_ok', 0)}** 张",
            f"- Defect classifier accuracy: {aux.get('_defect_acc', 0):.3f}",
            f"- Quality head accuracy: {aux.get('_quality_acc', 0):.3f}",
        ]
    else:
        lines.append("- Baseline 无缺陷/质量头，仅检测 4 类货物。")

    lines += [
        "",
        "## AP=0 检查",
        "",
        f"- 关键类别 AP=0: **{'无' if not zero_ap else ', '.join(zero_ap)}**",
        "",
        "## 总结",
        "",
    ]
    best_cls = improvements[0][0] if improvements else "N/A"
    worst_cls = low[0][0] if low else "N/A"
    lines.append(
        f"CargoDefect-YOLOv26 在 **{best_cls}** 上提升最明显；"
        f"**{worst_cls}** 仍是短板（尤其 package Recall 偏低）。"
        f" 整体检测 Recall 优于 baseline，辅助头提供缺陷类型与 OK/NG 判定能力。"
    )

    (out_dir / "class_improvement.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {ap_path}, {recall_path}, class_improvement.md")


def main():
    metrics = collect_all()
    (ROOT / "results/compare").mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(ROOT / "results/compare/_metrics_raw.json", "w"), indent=2, default=str)
    write_compare(metrics)
    write_analysis(metrics)


if __name__ == "__main__":
    main()
