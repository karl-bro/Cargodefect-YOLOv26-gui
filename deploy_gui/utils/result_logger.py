#!/usr/bin/env python3
"""Result logging: save images, CSV, scoring, and report generation."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2, numpy as np


class ResultLogger:
    """Handles image captures, CSV logging, scoring, and report."""

    def __init__(self, config: dict, base_dir: Path):
        sc = config["save"]
        self.result_dir = base_dir / sc.get("result_dir", "results")
        self.auto_capture_num = int(sc.get("auto_capture_num", 20))
        self.img_dir = self.result_dir / "images"
        self.ann_dir = self.result_dir / "annotated"
        self.csv_path = self.result_dir / "results.csv"
        self.report_path = self.result_dir / "score_report.md"

        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.ann_dir.mkdir(parents=True, exist_ok=True)

        self._captured = 0
        self._target = self.auto_capture_num
        self._records: list[dict] = []
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "timestamp", "image_path", "annotated_path",
                            "pred_result", "defect_status", "max_conf",
                            "package_conf", "defect_conf", "judge_result", "is_correct"])

    @property
    def captured_count(self) -> int:
        return self._captured

    @property
    def target_count(self) -> int:
        return self._target

    @property
    def capture_complete(self) -> bool:
        return self._captured >= self._target

    def save_frame(self, raw_img: np.ndarray, annotated_img: np.ndarray,
                   detection: dict, pkg_th: float, def_th: float) -> str | None:
        """Save original + annotated images, append CSV. Returns filename or None."""
        if self.capture_complete:
            return None

        idx = self._captured + 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"capture_{idx:02d}_{ts}.jpg"

        cv2.imwrite(str(self.img_dir / fname), raw_img)
        cv2.imwrite(str(self.ann_dir / fname), annotated_img)

        row = {
            "id": idx,
            "timestamp": ts,
            "image_path": str(self.img_dir / fname),
            "annotated_path": str(self.ann_dir / fname),
            "pred_result": detection.get("quality", ""),
            "defect_status": detection.get("defect_status", ""),
            "max_conf": detection.get("max_conf", 0.0),
            "package_conf": pkg_th,
            "defect_conf": def_th,
            "judge_result": "",
            "is_correct": "",
        }
        self._records.append(row)
        self._save_csv_row(row)
        self._captured += 1
        return str(self.ann_dir / fname)

    def _save_csv_row(self, row: dict):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([row["id"], row["timestamp"], row["image_path"], row["annotated_path"],
                        row["pred_result"], row["defect_status"], row["max_conf"],
                        row["package_conf"], row["defect_conf"], row["judge_result"], row["is_correct"]])

    def set_judge_result(self, index: int, result: str):
        if 0 <= index < len(self._records):
            self._records[index]["judge_result"] = result
            self._records[index]["is_correct"] = "yes" if result == "correct" else "no"
            self._rewrite_csv()

    def _rewrite_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "timestamp", "image_path", "annotated_path",
                        "pred_result", "defect_status", "max_conf",
                        "package_conf", "defect_conf", "judge_result", "is_correct"])
            for r in self._records:
                w.writerow([r["id"], r["timestamp"], r["image_path"], r["annotated_path"],
                            r["pred_result"], r["defect_status"], r["max_conf"],
                            r["package_conf"], r["defect_conf"], r["judge_result"], r["is_correct"]])

    @property
    def judged_count(self) -> int:
        return sum(1 for r in self._records if r["judge_result"] in ("correct", "wrong"))

    @property
    def correct_count(self) -> int:
        return sum(1 for r in self._records if r["judge_result"] == "correct")

    @property
    def wrong_count(self) -> int:
        return sum(1 for r in self._records if r["judge_result"] == "wrong")

    @property
    def score_accuracy(self) -> float:
        t = self.judged_count
        return self.correct_count / t * 100 if t > 0 else 0.0

    def generate_report(self) -> Path:
        total = len(self._records)
        judged = self.judged_count
        correct = self.correct_count
        wrong = self.wrong_count
        acc = self.score_accuracy

        lines = [
            "# 评分报表",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 汇总",
            f"- 总采集数: {total}",
            f"- 已评分: {judged}",
            f"- 正确: {correct}",
            f"- 错误: {wrong}",
            f"- 准确率: {acc:.1f}%",
            "",
            "## 明细",
            "| # | 预测结果 | 缺陷状态 | 最高置信度 | 评委判定 |",
            "|---|----------|----------|------------|----------|",
        ]
        for i, r in enumerate(self._records, 1):
            jr = r["judge_result"] or "--"
            lines.append(f"| {i} | {r['pred_result']} | {r['defect_status']} | {r['max_conf']:.2f} | {jr} |")

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.report_path

    def clear(self):
        self._records.clear()
        self._captured = 0
        if self.csv_path.exists():
            self.csv_path.unlink()
        self._init_csv()
