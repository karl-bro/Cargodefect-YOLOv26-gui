#!/usr/bin/env python3
"""Verify defect label mapping and class coverage."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.cargodefect import collect_label_statistics, resolve_cargodefect_data
from ultralytics.data.utils import check_det_dataset
from scripts.aux_head_utils import DATA_YAML, DEFECT_NAMES_5, DEFECT_NAMES_6, remap_defect_label_6

OUT = ROOT / "results/quality_debug"


def main():
    data = resolve_cargodefect_data(check_det_dataset(DATA_YAML))
    defect_names = data.get("defect_names", DEFECT_NAMES_5)

    print("Defect label schema (dataset nc_defect=5):")
    for i, name in enumerate(defect_names.values() if isinstance(defect_names, dict) else defect_names):
        print(f"  {i}: {name if isinstance(defect_names, dict) else name}")
    print("\nStandalone 6-class order:", ", ".join(DEFECT_NAMES_6))
    print("  remap: dataset scratch/crack/dent/stain/none -> 1/2/3/4/0; unknown -> anomaly(5)")
    print()

    for mode in ("train", "val"):
        stats = collect_label_statistics(data, mode=mode)
        print(f"[{mode}] defect counts: {stats['defect']}")
        if mode == "train" and stats["defect"].get("none", 0) == stats["total"]:
            print("  WARNING: all train defect labels are 'none' — defect classifier has no defect supervision.")
    print(f"\nDone. See also {OUT / 'label_distribution.csv'} (run check_quality_labels.py).")


if __name__ == "__main__":
    main()
