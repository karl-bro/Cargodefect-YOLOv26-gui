#!/usr/bin/env python3
"""Camera worker thread with MJPG, fixed resolution, and frame dropping."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class CameraWorker(QThread):
    """Reads frames from camera in a dedicated thread. Only keeps the latest frame."""

    frame_ready = Signal(np.ndarray)
    camera_error = Signal(str)

    def __init__(self, source: int = 0, width: int = 960, height: int = 540):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def run(self):
        self._running = True
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self.camera_error.emit(f"Cannot open camera source: {self.source}")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self.camera_error.emit("Failed to read frame from camera")
                break
            # Drop buffered frames to keep latency low
            for _ in range(1):
                self._cap.grab()
            self.frame_ready.emit(frame)
            # Cap camera callback rate ~25 FPS
            self.msleep(40)

        if self._cap is not None:
            self._cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)
