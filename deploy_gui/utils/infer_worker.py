#!/usr/bin/env python3
"""Background inference worker — keeps GUI thread free for smooth display."""

from __future__ import annotations

import traceback
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex

from utils.detector import CargoDefectDetector


class InferWorker(QThread):
    """Runs detector.detect() off the GUI thread.

    Only keeps the latest pending frame (frame dropping).
    """

    result_ready = Signal(object, object)  # (frame, detection_dict)
    infer_error = Signal(str)

    def __init__(self, detector: CargoDefectDetector):
        super().__init__()
        self.detector = detector
        self._running = False
        self._mutex = QMutex()
        self._pending: np.ndarray | None = None
        self._has_pending = False

    def submit(self, frame: np.ndarray):
        """Queue latest frame for inference (overwrites older pending)."""
        self._mutex.lock()
        self._pending = frame
        self._has_pending = True
        self._mutex.unlock()

    def run(self):
        self._running = True
        while self._running:
            self._mutex.lock()
            if self._has_pending and self._pending is not None:
                frame = self._pending
                self._pending = None
                self._has_pending = False
                self._mutex.unlock()
            else:
                self._mutex.unlock()
                self.msleep(5)
                continue

            try:
                detection = self.detector.detect(frame)
                self.result_ready.emit(frame, detection)
            except Exception as e:
                traceback.print_exc()
                self.infer_error.emit(f"{type(e).__name__}: {e}")

    def stop(self):
        self._running = False
        self.wait(5000)
