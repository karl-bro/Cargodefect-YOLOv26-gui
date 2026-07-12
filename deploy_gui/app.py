#!/usr/bin/env python3
"""CargoDefect-YOLOv26-Detect GUI -- threaded camera, inference, scoring."""

from __future__ import annotations

import sys, traceback, time
from datetime import datetime
from pathlib import Path

import cv2, numpy as np, yaml
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QVBoxLayout, QWidget, QGridLayout, QGroupBox, QFrame, QSizePolicy,
)

from utils.detector import CargoDefectDetector, PACKAGE_ID, DEFECT_ID
from utils.serial_sender import SerialSender
from utils.result_logger import ResultLogger
from utils.camera_worker import CameraWorker
from utils.infer_worker import InferWorker

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
GREEN = (0, 200, 50)
WHITE = (255, 255, 255)
CYAN = (255, 255, 128)
GRAY = (160, 160, 160)

OK_STYLE = "background-color:#e8f5e9;color:#2e7d32;font-weight:bold;font-size:16px;padding:6px;border-radius:4px;border:2px solid #4caf50"
NG_STYLE = "background-color:#ffebee;color:#c62828;font-weight:bold;font-size:16px;padding:6px;border-radius:4px;border:2px solid #f44336"
WAIT_STYLE = "background-color:#fff8e1;color:#e65100;font-weight:bold;font-size:16px;padding:6px;border-radius:4px;border:2px solid #ff9800"
VAL_STYLE = "font-size:12px;padding:2px;"
CARD_OK = "border:2px solid #4caf50;background:#e8f5e9;padding:4px"
CARD_NG = "border:2px solid #f44336;background:#ffebee;padding:4px"
CARD_JUDGED = "border:2px solid #2196f3;background:#e3f2fd;padding:4px"

_QUALITY_CN = {"OK": "合格", "NG": "不合格", "WAIT": "待检"}
_DEFECT_CN = {"normal": "正常", "defect": "缺陷", "none": "无"}
_PKG_SRC_CN = {"yolo": "YOLO", "background_diff": "背景差分", "fixed_station": "固定工位", "none": "无"}


class ScoreCard(QFrame):
    clicked_correct = Signal(int)
    clicked_wrong = Signal(int)

    def __init__(self, idx: int, pixmap: QPixmap, pred: str, conf: float, parent=None):
        super().__init__(parent)
        self.index = idx
        self.setFixedSize(200, 160)
        self.setFrameStyle(QFrame.Box)
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        lay.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel(f"#{idx + 1}  预测: {pred}  置信度:{conf:.2f}")
        lbl.setStyleSheet("font-size:10px;font-weight:bold")
        lay.addWidget(lbl)
        thumb = QLabel()
        thumb.setPixmap(pixmap.scaled(180, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        thumb.setAlignment(Qt.AlignCenter)
        lay.addWidget(thumb)
        br = QHBoxLayout()
        btn_ok = QPushButton("正确"); btn_ok.setFixedHeight(24)
        btn_ng = QPushButton("错误"); btn_ng.setFixedHeight(24)
        btn_ok.clicked.connect(lambda: self.clicked_correct.emit(self.index))
        btn_ng.clicked.connect(lambda: self.clicked_wrong.emit(self.index))
        br.addWidget(btn_ok); br.addWidget(btn_ng)
        lay.addLayout(br)
        self.update_style("")

    def update_style(self, judge: str):
        if judge == "correct" or judge == "wrong":
            self.setStyleSheet(CARD_JUDGED)
        elif "NG" in self.findChild(QLabel).text():
            self.setStyleSheet(CARD_NG)
        else:
            self.setStyleSheet(CARD_OK)


class CargoDefectGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("货件缺陷检测系统 -- CargoDefect-YOLOv26-Detect")
        self.resize(1500, 900)
        self.cfg = load_config()
        self.detector = CargoDefectDetector(self.cfg, BASE_DIR)
        self.serial = SerialSender(self.cfg)
        self.logger = ResultLogger(self.cfg, BASE_DIR)
        self.camera_worker: CameraWorker | None = None
        self.infer_worker: InferWorker | None = None
        self._latest_frame: np.ndarray | None = None
        self.capturing = False
        self._cards: list[ScoreCard] = []
        self._frame_idx = 0
        self._skip = max(1, int(self.cfg.get("performance", {}).get("frame_skip", 5)))
        self._last_detection: dict | None = None
        self._err_count = 0
        self._last_err = ""
        self._fps_c = 0
        self._fps_t = datetime.now()
        self._fps_v = 0.0
        self.detector.load_background()
        self._build_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._update_status)
        self._refresh_timer.start(500)

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QHBoxLayout(cw); split = QSplitter(Qt.Horizontal)

        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("摄像头未开启"); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480); self.video_label.setStyleSheet("background:#1a1a1a;color:#888;")
        lv.addWidget(self.video_label)
        self.fps_label = QLabel("帧率: --"); self.fps_label.setStyleSheet("color:#0f0;background:rgba(0,0,0,150);padding:4px")
        lv.addWidget(self.fps_label)
        split.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right); rv.setSpacing(6)

        gs = QGroupBox("系统状态"); gsl = QVBoxLayout(gs)
        self.lbl_model = QLabel("模型: --")
        self.lbl_camera = QLabel("摄像头: 关")
        self.lbl_bg = QLabel("空背景: 未采集")
        self.lbl_quality = QLabel("合格 / 不合格 / 待检"); self.lbl_quality.setAlignment(Qt.AlignCenter)
        self.lbl_quality.setStyleSheet(WAIT_STYLE)
        self.lbl_defect = QLabel("缺陷: --")
        self.lbl_class = QLabel("类别: --")
        self.lbl_pkg_src = QLabel("包裹来源: --")
        self.lbl_diff = QLabel("背景差异率: --")
        self.lbl_conf = QLabel("最高置信度: --")
        self.lbl_inf = QLabel("推理: -- ms")
        self.lbl_err = QLabel(""); self.lbl_err.setStyleSheet("color:red;font-size:11px"); self.lbl_err.hide()
        for w in [self.lbl_model, self.lbl_camera, self.lbl_bg, self.lbl_defect,
                   self.lbl_class, self.lbl_pkg_src, self.lbl_diff, self.lbl_conf, self.lbl_inf]:
            w.setStyleSheet(VAL_STYLE); gsl.addWidget(w)
        gsl.addWidget(self.lbl_quality); gsl.addWidget(self.lbl_err)
        rv.addWidget(gs)

        gt = QGroupBox("阈值设置"); gtl = QGridLayout(gt)
        gtl.addWidget(QLabel("包裹:"), 0, 0)
        self.sp_pkg = QDoubleSpinBox(); self.sp_pkg.setRange(0.01, 1.0); self.sp_pkg.setSingleStep(0.05); self.sp_pkg.setDecimals(2)
        self.sp_pkg.setValue(self.detector.package_conf); self.sp_pkg.valueChanged.connect(self._sync_thresh)
        gtl.addWidget(self.sp_pkg, 0, 1)
        gtl.addWidget(QLabel("缺陷展示:"), 1, 0)
        self.sp_dshow = QDoubleSpinBox(); self.sp_dshow.setRange(0.01, 1.0); self.sp_dshow.setSingleStep(0.05); self.sp_dshow.setDecimals(2)
        self.sp_dshow.setValue(self.detector.defect_show_conf); self.sp_dshow.valueChanged.connect(self._sync_thresh)
        gtl.addWidget(self.sp_dshow, 1, 1)
        gtl.addWidget(QLabel("缺陷NG:"), 2, 0)
        self.sp_dng = QDoubleSpinBox(); self.sp_dng.setRange(0.01, 1.0); self.sp_dng.setSingleStep(0.05); self.sp_dng.setDecimals(2)
        self.sp_dng.setValue(self.detector.defect_ng_conf); self.sp_dng.valueChanged.connect(self._sync_thresh)
        gtl.addWidget(self.sp_dng, 2, 1)
        rv.addWidget(gt)

        go = QGroupBox("运行选项"); gol = QVBoxLayout(go)
        self.chk_fixed = QCheckBox("固定工位模式"); self.chk_fixed.setChecked(self.detector.fixed_station_mode)
        self.chk_fixed.toggled.connect(self._sync_thresh); gol.addWidget(self.chk_fixed)
        self.chk_debug = QCheckBox("调试输出"); self.chk_debug.setChecked(self.detector.debug)
        self.chk_debug.toggled.connect(self._sync_thresh); gol.addWidget(self.chk_debug)
        rv.addWidget(go)

        gb = QGroupBox("操作"); gbl = QGridLayout(gb)
        self.btn_load = QPushButton("加载模型"); self.btn_load.clicked.connect(self._on_load)
        self.btn_cam_on = QPushButton("打开摄像头"); self.btn_cam_on.clicked.connect(self._on_cam_on)
        self.btn_cam_off = QPushButton("关闭摄像头"); self.btn_cam_off.clicked.connect(self._on_cam_off)
        self.btn_ser_on = QPushButton("连接串口"); self.btn_ser_on.clicked.connect(self._on_ser_on)
        self.btn_ser_off = QPushButton("断开串口"); self.btn_ser_off.clicked.connect(self._on_ser_off)
        self.btn_bg = QPushButton("采集空工位背景"); self.btn_bg.clicked.connect(self._on_cap_bg)
        self.btn_cap = QPushButton("采集20张"); self.btn_cap.clicked.connect(self._on_cap)
        self.btn_manual = QPushButton("采集单张"); self.btn_manual.clicked.connect(self._on_cap_single)
        self.btn_clear = QPushButton("清空"); self.btn_clear.clicked.connect(self._on_clear)
        self.btn_rpt = QPushButton("保存报表"); self.btn_rpt.clicked.connect(self._on_rpt)
        gbl.addWidget(self.btn_load, 0, 0); gbl.addWidget(self.btn_cam_on, 0, 1)
        gbl.addWidget(self.btn_cam_off, 1, 0); gbl.addWidget(self.btn_ser_on, 1, 1)
        gbl.addWidget(self.btn_ser_off, 2, 0); gbl.addWidget(self.btn_bg, 2, 1)
        gbl.addWidget(self.btn_cap, 3, 0); gbl.addWidget(self.btn_manual, 3, 1)
        gbl.addWidget(self.btn_clear, 4, 0); gbl.addWidget(self.btn_rpt, 4, 1)
        rv.addWidget(gb)

        self.lbl_stats = QLabel("已采集: 0/20  已评分: 0  正确: 0  错误: 0  准确率: --")
        self.lbl_stats.setStyleSheet("font-size:14px;font-weight:bold;padding:6px;background:#f5f5f5;border-radius:4px")
        rv.addWidget(self.lbl_stats)

        self.card_area = QScrollArea(); self.card_area.setWidgetResizable(True)
        self.card_container = QWidget(); self.card_layout = QGridLayout(self.card_container)
        self.card_layout.setSpacing(6); self.card_area.setWidget(self.card_container)
        self.card_area.setMinimumHeight(200); rv.addWidget(self.card_area)

        split.addWidget(right); split.setSizes([960, 520]); root.addWidget(split)

    def _sync_thresh(self):
        self.detector.set_package_conf(self.sp_pkg.value())
        self.detector.set_defect_show_conf(self.sp_dshow.value())
        self.detector.set_defect_ng_conf(self.sp_dng.value())
        self.detector.set_fixed_station_mode(self.chk_fixed.isChecked())
        self.detector.set_debug(self.chk_debug.isChecked())

    def _update_status(self):
        self.lbl_model.setText(f"模型: {'已加载' if self.detector.loaded else '未加载'}")
        self.lbl_camera.setText(f"摄像头: {'开' if self.camera_worker and self.camera_worker._running else '关'}")
        self.lbl_bg.setText(f"空背景: {'已采集' if self.detector.bg_loaded else '未采集'}")
        jc = self.logger.judged_count; tc = self.logger.captured_count
        self.lbl_stats.setText(f"已采集: {tc}/{self.logger.target_count}  已评分: {jc}  "
                               f"正确: {self.logger.correct_count}  错误: {self.logger.wrong_count}  "
                               f"准确率: {self.logger.score_accuracy:.1f}%")

    def _on_load(self):
        try: self.detector.load(); QMessageBox.information(self, "模型", "加载成功")
        except Exception as e: QMessageBox.critical(self, "模型错误", str(e))

    def _on_cam_on(self):
        if self.camera_worker and self.camera_worker._running: return
        if not self.detector.loaded:
            try: self.detector.load()
            except Exception as e: QMessageBox.critical(self, "模型错误", str(e)); return
        try:
            cc = self.cfg["camera"]; w = cc.get("width", 960); h = cc.get("height", 540)
            # Start inference worker first
            if self.infer_worker is None or not self.infer_worker.isRunning():
                self.infer_worker = InferWorker(self.detector)
                self.infer_worker.result_ready.connect(self._on_infer_result)
                self.infer_worker.infer_error.connect(self._on_infer_error)
                self.infer_worker.start()
            self.camera_worker = CameraWorker(cc.get("source", 0), w, h)
            self.camera_worker.frame_ready.connect(self._on_frame)
            self.camera_worker.camera_error.connect(lambda m: QMessageBox.warning(self, "摄像头", m))
            self.camera_worker.start(); self._err_count = 0; self.lbl_err.hide()
        except Exception as e: traceback.print_exc(); QMessageBox.critical(self, "摄像头错误", str(e))

    def _on_cam_off(self):
        if self.camera_worker: self.camera_worker.stop(); self.camera_worker = None
        if self.infer_worker: self.infer_worker.stop(); self.infer_worker = None
        self.video_label.setText("摄像头未开启")

    def _on_ser_on(self):
        try: self.serial.connect(); QMessageBox.information(self, "串口", "已连接")
        except Exception as e: QMessageBox.warning(self, "串口错误", str(e))

    def _on_ser_off(self): self.serial.disconnect()

    def _on_cap_bg(self):
        if self._latest_frame is None: QMessageBox.warning(self, "背景采集", "请先打开摄像头，确保画面中无包裹"); return
        if self.detector.capture_background(self._latest_frame): QMessageBox.information(self, "背景采集", "空工位背景已保存")
        else: QMessageBox.critical(self, "背景采集", "背景保存失败")

    def _on_cap(self):
        self._cards.clear()
        while self.card_layout.count():
            w = self.card_layout.takeAt(0).widget()
            if w: w.deleteLater()
        self.capturing = True; self.logger.clear()
        QMessageBox.information(
            self, "采集",
            f"开始自动采集 {self.logger.target_count} 张。\n"
            "提示：需检测到包裹才会抓拍。\n"
            "若一直待检，请先「采集空工位背景」或勾选「固定工位模式」。"
        )

    def _on_cap_single(self):
        """Manual single-frame capture — works even without package detection."""
        if not self.camera_worker or not self.camera_worker._running:
            QMessageBox.warning(self, "采集", "请先打开摄像头"); return
        if self.logger.capture_complete:
            QMessageBox.information(self, "采集", f"已采集满 {self.logger.target_count} 张，请先清空记录"); return
        if self._latest_frame is None:
            QMessageBox.warning(self, "采集", "暂无可用画面，请稍候再试"); return
        detection = self._last_detection or {
            "quality": "WAIT", "defect_status": "none", "max_conf": 0.0,
            "has_package": False, "has_defect": False, "boxes": [],
        }
        annotated = self._draw_detection(self._latest_frame, detection)
        self._do_capture(annotated, detection)

    def _on_clear(self):
        self.capturing = False; self.logger.clear()
        for c in self._cards: c.deleteLater()
        self._cards.clear()
        while self.card_layout.count():
            w = self.card_layout.takeAt(0).widget()
            if w: w.deleteLater()
        self.lbl_stats.setText("已采集: 0/20  已评分: 0  正确: 0  错误: 0  准确率: --")

    def _on_rpt(self):
        try: path = self.logger.generate_report(); QMessageBox.information(self, "报表", f"已保存: {path}")
        except Exception as e: QMessageBox.critical(self, "报表错误", str(e))

    def _on_frame(self, frame: np.ndarray):
        """Camera callback — always refresh display; submit inference asynchronously."""
        self._latest_frame = frame
        self._frame_idx += 1

        # FPS counter for display refresh
        self._fps_c += 1
        now = datetime.now()
        if (now - self._fps_t).total_seconds() >= 1.0:
            self._fps_v = self._fps_c / (now - self._fps_t).total_seconds()
            self._fps_c = 0
            self._fps_t = now

        # Submit to background infer every N frames (non-blocking)
        if self._frame_idx % self._skip == 0 and self.infer_worker is not None:
            self.infer_worker.submit(frame.copy())

        # Always show latest frame with last known overlays — never blocks
        annotated = self._draw_detection(frame, self._last_detection)
        self._display_frame(annotated)

    def _on_infer_error(self, msg: str):
        self._err_count += 1
        self._last_err = msg
        if self._err_count <= 3:
            print(f"[GUI ERR] {msg}")
        self.lbl_inf.setText("推理: 错误")
        self.lbl_err.setText(msg)
        self.lbl_err.show()

    def _on_infer_result(self, frame: np.ndarray, detection: dict):
        """Called from InferWorker when a detection finishes."""
        self._last_detection = detection
        self._err_count = 0
        self.lbl_err.hide()

        q = detection["quality"]
        self.lbl_quality.setText(_QUALITY_CN.get(q, q))
        if q == "OK": self.lbl_quality.setStyleSheet(OK_STYLE)
        elif q == "NG": self.lbl_quality.setStyleSheet(NG_STYLE)
        else: self.lbl_quality.setStyleSheet(WAIT_STYLE)

        self.lbl_defect.setText(f"缺陷: {_DEFECT_CN.get(detection['defect_status'], detection['defect_status'])}")

        pkg_src = detection.get("package_source", "none")
        self.lbl_pkg_src.setText(f"包裹来源: {_PKG_SRC_CN.get(pkg_src, pkg_src)}")

        has_pkg = detection["has_package"]
        has_def = detection["has_defect"]
        if has_pkg:
            self.lbl_class.setText(f"类别: 包裹  包裹=是  缺陷={'是' if has_def else '否'}")
        else:
            self.lbl_class.setText("类别: 无  包裹=否  缺陷=否")

        if pkg_src == "yolo":
            ypkc = detection.get("yolo_pkg_max_conf", detection["max_conf"])
            self.lbl_conf.setText(f"最高置信度: {ypkc:.3f}")
        elif has_def:
            self.lbl_conf.setText(f"最高置信度: {detection['max_conf']:.3f}")
        else:
            self.lbl_conf.setText("最高置信度: --")

        diff_r = detection.get("diff_ratio", 0.0)
        if self.detector.bg_loaded:
            self.lbl_diff.setText(f"背景差异率: {diff_r * 100:.2f}%")
        else:
            self.lbl_diff.setText("背景差异率: --")

        self.lbl_inf.setText(f"推理: {detection['inference_ms']:.0f} ms")
        self.serial.push_decision(q, detection["defect_status"], detection["max_conf"])

        # Auto-capture only when package is present
        if self.capturing and has_pkg and not self.logger.capture_complete:
            annotated = self._draw_detection(frame, detection)
            self._do_capture(annotated, detection)

    def _do_capture(self, annotated: np.ndarray, detection: dict):
        raw = self._latest_frame
        if raw is None: return
        path = self.logger.save_frame(raw, annotated, detection, self.sp_pkg.value(), self.sp_dng.value())
        if path is None: return
        # Use a contiguous copy for QImage safety
        ann = np.ascontiguousarray(annotated)
        h, w, _ = ann.shape
        qimg = QImage(ann.data, w, h, w * 3, QImage.Format_BGR888).copy()
        pix = QPixmap.fromImage(qimg); idx = len(self._cards)
        card = ScoreCard(idx, pix, detection.get("quality", "WAIT"), float(detection.get("max_conf", 0.0)))
        card.clicked_correct.connect(self._on_card_ok); card.clicked_wrong.connect(self._on_card_wrong)
        r, c = divmod(idx, 5); self.card_layout.addWidget(card, r, c); self._cards.append(card)
        if self.logger.capture_complete:
            self.capturing = False
            QMessageBox.information(self, "采集完成", f"已采集 {self.logger.target_count} 张，请评分")

    def _on_card_ok(self, index: int):
        self.logger.set_judge_result(index, "correct"); self._cards[index].update_style("correct")
    def _on_card_wrong(self, index: int):
        self.logger.set_judge_result(index, "wrong"); self._cards[index].update_style("wrong")

    def _draw_detection(self, frame: np.ndarray, detection: dict | None) -> np.ndarray:
        display = frame.copy()
        if self.detector.enable_roi:
            rx, ry, rx2, ry2 = self.detector.roi
            cv2.rectangle(display, (rx, ry), (rx2, ry2), CYAN, 1)
        if detection is None:
            return display

        pkg_src = detection.get("package_source", "none")
        diff_r = detection.get("diff_ratio", 0.0)
        if self.detector.enable_roi:
            rx, ry, _, _ = self.detector.roi
            if pkg_src == "background_diff":
                cv2.putText(display, f"package bg-diff {diff_r:.3f}", (rx + 4, ry + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
            elif not detection.get("has_package", False):
                cv2.putText(display, "WAIT", (rx + 4, ry + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        for b in detection.get("boxes", []):
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            conf, cc = float(b["conf"]), b.get("color", "white")
            if cc == "blue": color, text = BLUE, f"package {conf:.2f}"
            elif cc == "red": color, text = RED, f"defect {conf:.2f}"
            elif cc == "yellow": color, text = YELLOW, f"suspect {conf:.2f}"
            elif cc == "gray": color, text = GRAY, f"raw_pkg {conf:.3f}"
            else: color, text = WHITE, f"?? {conf:.2f}"
            thickness = 1 if cc == "gray" else 2
            font_scale = 0.4 if cc == "gray" else 0.5
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(display, text, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        q = detection.get("quality", "WAIT")
        qc = GREEN if q == "OK" else (0, 0, 255) if q == "NG" else (0, 180, 255)
        cv2.putText(display, q, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, qc, 3)
        return display

    def _display_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        # Contiguous copy so QImage buffer stays valid
        frame = np.ascontiguousarray(frame)
        qt = QImage(frame.data, w, h, w * ch, QImage.Format_BGR888).copy()
        pix = QPixmap.fromImage(qt).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.video_label.setPixmap(pix)
        self.fps_label.setText(f"帧率: {self._fps_v:.1f}")

    def closeEvent(self, event):
        self._on_cam_off(); self.serial.disconnect(); self._refresh_timer.stop(); event.accept()


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))
    w = CargoDefectGUI(); w.show(); sys.exit(app.exec())

if __name__ == "__main__": main()
