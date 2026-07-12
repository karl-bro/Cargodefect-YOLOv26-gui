#!/usr/bin/env python3
"""Verify quality label mapping: good/OK=0, anomaly/NG=1."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.cargodefect import collect_label_statistics, load_quality_label, resolve_cargodefect_data
from ultralytics.data.utils import check_det_dataset
from scripts.aux_head_utils import DATA_YAML, write_label_distribution_csv

OUT = ROOT / "results/quality_debug"


def main():
    data = resolve_cargodefect_data(check_det_dataset(DATA_YAML))
    rows = [collect_label_statistics(data, mode=m) for m in ("train", "val")]
    out_csv = OUT / "label_distribution.csv"
    write_label_distribution_csv(rows, out_csv)

    print("Quality label rules:")
    print("  - packaging quality txt: 0=OK, >0=NG (default 0 if missing)")
    print("  - MVTec good/validation/test/good -> OK=0")
    print("  - MVTec logical/structural anomalies -> NG=1")
    print()
    for row in rows:
        total = row["total"] or 1
        print(
            f"[{row['mode']}] total={row['total']} OK={row['quality_ok']} NG={row['quality_ng']} "
            f"(NG ratio={row['quality_ng']/total:.1%}) sources={row['source']}"
        )
        if row["mode"] == "train" and row["quality_ng"] == 0:
            print("  WARNING: train split has zero NG labels — quality head cannot learn NG.")
        if row["mode"] == "val" and row["quality_ng"] == 0:
            print("  WARNING: val split has zero NG labels.")

    # Spot-check packaging quality file reader
    packaging = Path(data["packaging_path"])
    sample = next(packaging.glob("images/Train/*.jpg"), None)
    if sample:
        qpath = packaging / "quality" / "Train" / f"{sample.stem}.txt"
        print(f"\nPackaging quality sample: {sample.name} -> {load_quality_label(qpath, default=0)} (file exists: {qpath.exists()})")

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
