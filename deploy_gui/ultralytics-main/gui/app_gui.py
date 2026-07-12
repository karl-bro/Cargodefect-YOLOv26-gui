#!/usr/bin/env python3
"""PyQt5 GUI component for CargoDefect inspection with per-class thresholds."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                              QPushButton, QVBoxLayout, QHBoxLayout,
                              QGridLayout, QFileDialog, QDoubleSpinBox,
                              QGroupBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.inference import DefectInferenceEngine
from utils.result_logger import ResultLogger


class DefectInspectionGUI(QMainWindow):
    """Main GUI window for defect inspection with per-class confidence thresholds."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CargoDefect Inspection System")
        self.setMinimumSize(1200, 800)

        self.engine = DefectInferenceEngine("gui/config.yaml")
        self.logger = ResultLogger("gui_results")
        self.camera = None
        self.camera_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.current_frame = None
        self.current_result = None
        self.fps = 0.0
        self.last_frame_time = time.time()

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel
        left_panel = QVBoxLayout()
        self.video_label = QLabel("No image loaded")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray; background: #222;")
        left_panel.addWidget(self.video_label)

        cam_group = QGroupBox("Camera")
        cam_layout = QHBoxLayout()
        self.btn_open_cam = QPushButton("Open Camera")
        self.btn_open_cam.clicked.connect(self.toggle_camera)
        cam_layout.addWidget(self.btn_open_cam)
        self.btn_load_img = QPushButton("Load Image")
        self.btn_load_img.clicked.connect(self.load_image)
        cam_layout.addWidget(self.btn_load_img)
        cam_group.setLayout(cam_layout)
        left_panel.addWidget(cam_group)
        left_panel.addStretch()
        main_layout.addLayout(left_panel, 2)

        # Right panel
        right_panel = QVBoxLayout()

        result_group = QGroupBox("Current Result")
        result_layout = QGridLayout()
        self.lbl_cargo = QLabel("--")
        self.lbl_defect = QLabel("--")
        self.lbl_quality = QLabel("--")
        self.lbl_conf = QLabel("--")
        self.lbl_time = QLabel("--")
        self.lbl_serial = QLabel("--")

        result_layout.addWidget(QLabel("Cargo:"), 0, 0)
        result_layout.addWidget(self.lbl_cargo, 0, 1)
        result_layout.addWidget(QLabel("Defect:"), 1, 0)
        result_layout.addWidget(self.lbl_defect, 1, 1)
        result_layout.addWidget(QLabel("Quality:"), 2, 0)
        result_layout.addWidget(self.lbl_quality, 2, 1)
        result_layout.addWidget(QLabel("Confidence:"), 3, 0)
        result_layout.addWidget(self.lbl_conf, 3, 1)
        result_layout.addWidget(QLabel("Time:"), 4, 0)
        result_layout.addWidget(self.lbl_time, 4, 1)
        result_layout.addWidget(QLabel("Serial:"), 5, 0)
        result_layout.addWidget(self.lbl_serial, 5, 1)
        result_group.setLayout(result_layout)
        right_panel.addWidget(result_group)

        thresh_group = QGroupBox("Thresholds (package / defect)")
        thresh_layout = QGridLayout()
        thresh_layout.addWidget(QLabel("Package:"), 0, 0)
        self.spin_pkg = QDoubleSpinBox()
        self.spin_pkg.setRange(0.01, 1.0)
        self.spin_pkg.setValue(self.engine.package_conf)
        self.spin_pkg.setSingleStep(0.05)
        self.spin_pkg.valueChanged.connect(self._on_pkg_changed)
        thresh_layout.addWidget(self.spin_pkg, 0, 1)
        thresh_layout.addWidget(QLabel("Defect:"), 1, 0)
        self.spin_def = QDoubleSpinBox()
        self.spin_def.setRange(0.01, 1.0)
        self.spin_def.setValue(self.engine.defect_conf)
        self.spin_def.setSingleStep(0.05)
        self.spin_def.valueChanged.connect(self._on_def_changed)
        thresh_layout.addWidget(self.spin_def, 1, 1)
        thresh_group.setLayout(thresh_layout)
        right_panel.addWidget(thresh_group)

        score_group = QGroupBox("Scoring")
        score_layout = QVBoxLayout()
        self.lbl_score = QLabel("Correct: 0 / Total: 0")
        score_layout.addWidget(self.lbl_score)
        btn_layout = QHBoxLayout()
        self.btn_correct = QPushButton("Correct")
        self.btn_correct.clicked.connect(lambda: self.judge(True))
        self.btn_wrong = QPushButton("Wrong")
        self.btn_wrong.clicked.connect(lambda: self.judge(False))
        btn_layout.addWidget(self.btn_correct)
        btn_layout.addWidget(self.btn_wrong)
        score_layout.addLayout(btn_layout)
        self.btn_export = QPushButton("Export Report")
        self.btn_export.clicked.connect(self.export_report)
        score_layout.addWidget(self.btn_export)
        score_group.setLayout(score_layout)
        right_panel.addWidget(score_group)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)

    def toggle_camera(self):
        if self.camera_running:
            self.timer.stop()
            if self.camera:
                self.camera.release()
            self.camera = None
            self.camera_running = False
            self.btn_open_cam.setText("Open Camera")
            self.video_label.setText("Camera stopped")
        else:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    self.video_label.setText("Camera not available")
                    return
                self.camera_running = True
                self.btn_open_cam.setText("Close Camera")
                self.timer.start(30)
            except Exception as e:
                self.video_label.setText(f"Camera error: {e}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.jpg *.png *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._display_frame(img)
                self._run_inference()

    def _run_inference(self):
        if self.current_frame is None:
            return
        try:
            result = self.engine.process_image(self.current_frame)
            self.current_result = result
            img_id = self.logger.log_result(result, fps=0)

            self.lbl_cargo.setText(result["cargo_class"])
            self.lbl_defect.setText(result["defect_status"])
            self.lbl_quality.setText(result["quality_result"])
            self.lbl_conf.setText(f"{result['confidence']:.4f}")
            self.lbl_time.setText(f"{result['inference_time_ms']:.2f} ms")

            color = "red" if result["quality_result"] == "NG" else "green"
            self.lbl_quality.setStyleSheet(f"font-weight: bold; color: {color};")

            # Build serial message
            serial_msg = self._build_serial_msg(result)
            self.lbl_serial.setText(serial_msg)

            annotated = result["annotated_image"]
            self._display_frame(annotated)
        except Exception as e:
            self.lbl_quality.setText(f"Error: {e}")

    def _build_serial_msg(self, result: dict) -> str:
        """Format: <cargo_class,defect_status,OK_or_NG,confidence>"""
        cargo = result.get("cargo_class", "none")
        defect = result.get("defect_status", "normal")
        quality = result.get("quality_result", "OK")
        conf = result.get("confidence", 0.0)
        return f"<{cargo},{defect},{quality},{conf:.2f}>"

    def update_frame(self):
        if self.camera and self.camera_running:
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.time()
                if now - self.last_frame_time > 0:
                    self.fps = 1.0 / (now - self.last_frame_time)
                self.last_frame_time = now
                self._run_inference()

    def _display_frame(self, frame):
        if frame is None:
            return
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            h, w, ch = frame.shape
            bytes_per_line = 3 * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = frame.shape[:2]
            bytes_per_line = w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def judge(self, is_correct: bool):
        if self.current_result:
            self.logger.judge_result(
                f"img_{len(self.logger.history):04d}", is_correct)
            score = self.logger.get_score()
            self.lbl_score.setText(
                f"Correct: {score['correct']} / Total: {score['total']} "
                f"({score['accuracy']:.1%})")

    def export_report(self):
        path = self.logger.export_score_report()
        self.lbl_score.setText(f"Report: {path}")

    def _on_pkg_changed(self, val):
        self.engine.package_conf = val

    def _on_def_changed(self, val):
        self.engine.defect_conf = val
