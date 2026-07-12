#!/usr/bin/env python3
"""Result logger for saving inference results and scoring."""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ResultLogger:
    def __init__(self, save_dir: str = "gui_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "images").mkdir(exist_ok=True)
        (self.save_dir / "reports").mkdir(exist_ok=True)

        self.csv_path = self.save_dir / "results.csv"
        self._init_csv()

        self.history: List[Dict] = []
        self.max_history = 20
        self.correct_count = 0
        self.total_count = 0

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image_id", "timestamp", "cargo_class", "defect_status",
                    "result", "confidence", "inference_time", "fps",
                    "serial_status", "judge_result"
                ])

    def log_result(self, result: Dict, image_path: str = "",
                   serial_status: str = "N/A", fps: float = 0.0) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        image_id = Path(image_path).stem if image_path else f"img_{len(self.history):04d}"

        row = [
            image_id, timestamp,
            result.get("cargo_class", ""),
            result.get("defect_status", ""),
            result.get("quality_result", ""),
            result.get("confidence", 0),
            result.get("inference_time_ms", 0),
            round(fps, 2),
            serial_status,
            "",  # judge_result filled later
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        entry = {
            "image_id": image_id,
            "timestamp": timestamp,
            "result": result,
            "row": row,
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return image_id

    def judge_result(self, image_id: str, is_correct: bool):
        rows = []
        if self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

        for i, row in enumerate(rows):
            if row[0] == image_id:
                row[9] = "correct" if is_correct else "incorrect"
                break

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        if is_correct:
            self.correct_count += 1
        self.total_count += 1

    def get_score(self) -> Dict:
        accuracy = self.correct_count / self.total_count if self.total_count > 0 else 0
        return {
            "correct": self.correct_count,
            "total": self.total_count,
            "accuracy": round(accuracy, 4),
        }

    def export_score_report(self):
        score = self.get_score()
        report_path = self.save_dir / "reports" / "score_report.md"
        with open(report_path, "w") as f:
            f.write("# Score Report\n\n")
            f.write(f"- Correct: {score['correct']}\n")
            f.write(f"- Total: {score['total']}\n")
            f.write(f"- Accuracy: {score['accuracy']:.2%}\n")
        return report_path

    def get_history(self) -> List[Dict]:
        return self.history
