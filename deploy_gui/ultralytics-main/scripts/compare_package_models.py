#!/usr/bin/env python3
"""Generate package baseline vs cargodefect comparison report."""
import csv, sys
from pathlib import Path
from datetime import datetime
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMPARE_DIR = ROOT / "results/package_compare"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

def load_csv_last(path):
    if not path.exists(): return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: round(float(v), 4) for k, v in rows[-1].items()} if rows else {}

def extract_metrics(csv_path):
    d = load_csv_last(csv_path)
    m = {}
    for k, v in d.items():
        if "mAP50(B)" in k: m["mAP50"] = v
        elif "mAP50-95(B)" in k: m["mAP50_95"] = v
        elif "precision(B)" in k: m["precision"] = v
        elif "recall(B)" in k: m["recall"] = v
    return m

baseline_csv = ROOT / "runs/detect/runs/cargodefect/baseline_package/results.csv"
cargod_csv = ROOT / "runs/detect/runs/cargodefect/cargodefect_package/results.csv"

bm = extract_metrics(baseline_csv)
cm = extract_metrics(cargod_csv)

rows = [["baseline_package", bm.get("mAP50","?"), bm.get("mAP50_95","?"), bm.get("precision","?"), bm.get("recall","?"),
         "?", "?", "?", "?"],
        ["cargodefect_package", cm.get("mAP50","?"), cm.get("mAP50_95","?"), cm.get("precision","?"), cm.get("recall","?"),
         "?", "?", "?", "?"]]

csv_path = COMPARE_DIR / "baseline_vs_cargodefect_package.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","mAP50","mAP50_95","precision","recall","q_accuracy","q_ok_acc","q_ng_recall","q_f1"])
    w.writerows(rows)

md = ["# Package Baseline vs CargoDefect Comparison",
      f"Generated: {datetime.now()}",
      "|Model|mAP50|mAP50-95|Precision|Recall|Q_Acc|Q_OK|Q_NG_Rec|Q_F1|",
      "|---|---|---|---|---|---|---|---|---|"]
for r in rows: md.append(f"|{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}|{r[7]}|{r[8]}|")
(COMPARE_DIR / "baseline_vs_cargodefect_package.md").write_text("\n".join(md))
print(f"CSV: {csv_path}")
print(f"MD: {COMPARE_DIR / 'baseline_vs_cargodefect_package.md'}")
