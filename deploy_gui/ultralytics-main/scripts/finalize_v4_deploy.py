#!/usr/bin/env python3
"""
Post-training script: copy best_v4.pt to deploy_gui, generate comparison report, zip package.
Run AFTER v4_train_eval_pipeline.py completes.
"""
import os, sys, shutil, yaml, csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEPLOY = ROOT / "deploy_gui"
RESULTS = ROOT / "results/v4_final"
V4_RUNS = ROOT / "runs/detect/runs/cargodefect"
BASELINE = ROOT / "runs/detect/runs/cargodefect/baseline_yolo26"
FUSION_V3 = ROOT / "runs/detect/runs/cargodefect/fusion_v3"

VARIANTS = ["fusion_v4_pos1.0", "fusion_v4_pos1.2", "fusion_v4_pos1.5", "fusion_v4_focal0.25", "fusion_v4_focal0.35"]

def load_results_csv(path):
    """Load YOLO training results.csv (last row = best epoch)."""
    if not path.exists():
        return {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    return dict(zip(header, last))

def extract_metrics(results_csv_path):
    """Extract key metrics from results.csv (YOLO format)."""
    d = load_results_csv(results_csv_path)
    if not d:
        return {"mAP50": "N/A", "mAP50_95": "N/A", "precision": "N/A", "recall": "N/A"}
    # YOLO results.csv columns (varies by version):
    # epoch, train/box_loss, train/cls_loss, train/dfl_loss, ...,
    # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
    metrics = {}
    for k, v in d.items():
        if "mAP50(B)" in k or "metrics/mAP50(B)" in k:
            metrics["mAP50"] = round(float(v), 4)
        elif "mAP50-95(B)" in k or "metrics/mAP50-95(B)" in k:
            metrics["mAP50_95"] = round(float(v), 4)
        elif "precision(B)" in k or "metrics/precision(B)" in k:
            metrics["precision"] = round(float(v), 4)
        elif "recall(B)" in k or "metrics/recall(B)" in k:
            metrics["recall"] = round(float(v), 4)
    return metrics

def main():
    print("=" * 60)
    print("Phase 5: Finalize v4 and package deploy")
    print("=" * 60)

    # Read v4 evaluation report
    v4_report = RESULTS / "v4_final_report.md"
    if not v4_report.exists():
        print(f"ERROR: v4_final_report.md not found at {RESULTS}")
        print("Run v4_train_eval_pipeline.py first!")
        sys.exit(1)

    # Parse best model name from report
    best_name = None
    for line in v4_report.read_text().split("\n"):
        if line.startswith("## Best:"):
            best_name = line.split(":")[1].strip()
            break

    if not best_name:
        print("Could not determine best model. Using first available.")
        for v in VARIANTS:
            wp = V4_RUNS / v / "weights" / "best.pt"
            if wp.exists():
                best_name = v
                break

    best_wp = V4_RUNS / best_name / "weights" / "best.pt"
    print(f"\nBest model: {best_name}")
    print(f"Source: {best_wp}")

    if not best_wp.exists():
        print(f"ERROR: best.pt not found at {best_wp}")
        sys.exit(1)

    # Copy to deploy_gui
    dest = DEPLOY / "weights" / "best_v4.pt"
    print(f"Copying to: {dest}")
    shutil.copy2(best_wp, dest)
    print(f"  Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")

    # --- Generate Final Comparison Report ---
    print("\n--- Comparison Report ---")
    compare_dir = ROOT / "results/final_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "baseline_yolo26": BASELINE / "weights" / "best.pt",
        "fusion_v3": FUSION_V3 / "weights" / "best.pt",
        **{v: V4_RUNS / v / "weights" / "best.pt" for v in VARIANTS},
    }

    # Read quality eval CSV
    qeval = {}
    qcsv = RESULTS / "quality_evaluation.csv"
    if qcsv.exists():
        with open(qcsv) as f:
            for row in csv.DictReader(f):
                qeval[row["name"]] = row

    rows = []
    for model_name, wp in models.items():
        if not wp.exists():
            continue
        metrics = extract_metrics(wp.parent.parent / "results.csv")
        qm = qeval.get(model_name, {})
        rows.append([
            model_name,
            metrics.get("mAP50", "N/A"),
            metrics.get("mAP50_95", "N/A"),
            metrics.get("precision", "N/A"),
            metrics.get("recall", "N/A"),
            qm.get("accuracy", "N/A"),
            qm.get("ok_accuracy", "N/A"),
            qm.get("ng_recall", "N/A"),
            qm.get("f1", "N/A"),
        ])

    # CSV
    csv_path = compare_dir / "baseline_vs_cargodefect_v4.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mAP50", "mAP50_95", "precision", "recall", "q_accuracy", "q_ok_acc", "q_ng_recall", "q_f1"])
        w.writerows(rows)

    # MD
    md_lines = [
        "# CargoDefect-YOLOv26 Final Comparison Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Detection + Quality Comparison",
        "",
        "| Model | mAP50 | mAP50-95 | Precision | Recall | Q_Acc | Q_OK_Acc | Q_NG_Rec | Q_F1 |",
        "|-------|-------|----------|-----------|--------|-------|----------|----------|------|",
    ]
    for row in rows:
        md_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |")

    md_lines += [
        "",
        f"## Best Model: {best_name}",
        f"Deployed to: deploy_gui/weights/best_v4.pt",
        "",
        "## Artifacts",
        f"- {csv_path}",
        f"- {compare_dir / 'baseline_vs_cargodefect_v4.md'}",
        f"- {DEPLOY}/weights/best_v4.pt",
    ]
    md_path = compare_dir / "baseline_vs_cargodefect_v4.md"
    md_path.write_text("\n".join(md_lines))
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")

    # --- Zip Package ---
    print("\n--- Packaging ---")
    zip_path = ROOT / "deploy_gui_v4.zip"
    if zip_path.exists():
        zip_path.unlink()

    import subprocess
    result = subprocess.run(
        ["zip", "-r", str(zip_path), "deploy_gui"],
        cwd=str(ROOT), capture_output=True, text=True
    )
    if result.returncode == 0:
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"Package: {zip_path} ({size_mb:.1f} MB)")
    else:
        print(f"Zip failed: {result.stderr}")

    print("\n" + "=" * 60)
    print("FINALIZE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
