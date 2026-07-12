#!/usr/bin/env python3
"""
CargoDefect GUI Application with OK/NG inspection.
Temporary rule: defect detected -> NG, else -> OK.
Supports both GUI mode and CLI test mode.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from utils.inference import DefectInferenceEngine, draw_results
from utils.result_logger import ResultLogger

QT_AVAILABLE = False
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                                  QPushButton, QVBoxLayout, QHBoxLayout,
                                  QGridLayout, QFileDialog, QDoubleSpinBox,
                                  QGroupBox)
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QImage, QPixmap
    QT_AVAILABLE = True
except ImportError:
    pass


def run_cli_test():
    """CLI test mode for cloud server validation."""
    print("=" * 60)
    print("CargoDefect GUI - CLI Test Mode")
    print("=" * 60)

    engine = DefectInferenceEngine("gui/config.yaml")
    logger = ResultLogger("gui_results")

    val_dir = Path("datasets/cargodefect_package/images/val")
    if not val_dir.exists():
        print(f"Val dir not found: {val_dir}")
        return

    images = sorted(val_dir.glob("*.jpg"))[:20]
    if not images:
        images = sorted(val_dir.glob("*.png"))[:20]

    print(f"Testing with {len(images)} images...\n")

    ok_count = 0
    ng_count = 0
    total_time = 0.0

    for img_path in images:
        try:
            result = engine.process_file(str(img_path))
            logger.log_result(result, str(img_path),
                              fps=1000 / result["inference_time_ms"]
                              if result["inference_time_ms"] > 0 else 0)

            status = result["quality_result"]
            if status == "OK":
                ok_count += 1
            else:
                ng_count += 1

            total_time += result["inference_time_ms"]

            nb = len(result["detections"])
            det_info = ",".join(f"{d['class_name']}({d['confidence']:.2f})" for d in result["detections"])
            print(f"  {img_path.name}: {status} | conf={result['confidence']:.3f} | "
                  f"{result['inference_time_ms']:.1f}ms | {nb} boxes | {det_info}")

        except Exception as e:
            print(f"  {img_path.name}: ERROR - {e}")

    avg_time = total_time / len(images) if images else 0
    print(f"\nSummary: {len(images)} images, OK={ok_count}, NG={ng_count}")
    if avg_time > 0:
        print(f"Avg inference: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")

    csv_path = logger.csv_path
    if csv_path.exists():
        print(f"\nResults saved to: {csv_path}")
        with open(csv_path) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                prefix = "  " if i == 0 else "  "
                print(prefix + " | ".join(row))
                if i >= 10:
                    print("  ...")
                    break

    print("\nCLI test complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CargoDefect Inspection GUI")
    parser.add_argument("--test", action="store_true", help="Run CLI test mode")
    parser.add_argument("--config", default="gui/config.yaml", help="Config path")
    args = parser.parse_args()

    if args.test or not QT_AVAILABLE:
        run_cli_test()
    elif QT_AVAILABLE:
        # Import GUI class only when needed
        from gui.app_gui import DefectInspectionGUI
        app = QApplication(sys.argv)
        window = DefectInspectionGUI()
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
