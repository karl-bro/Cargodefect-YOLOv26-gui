#!/usr/bin/env python3
"""Serial communication sender for STM32 / PLC."""

from __future__ import annotations

import time
from collections import deque

class SerialSender:
    """Manages serial connection with cooldown and frame stability."""

    def __init__(self, config: dict):
        sc = config["serial"]
        self.port = sc.get("port", "COM3")
        self.baudrate = int(sc.get("baudrate", 115200))
        self.enabled = bool(sc.get("enable", False))

        dc = config["decision"]
        self.stable_frames = int(dc.get("stable_frames", 5))
        self.cooldown_ms = int(dc.get("send_cooldown_ms", 1500))

        self._ser = None
        self.connected = False
        self._last_quality: str | None = None
        self._decision_queue: deque[str] = deque(maxlen=self.stable_frames)
        self._last_send_time = 0.0

    def connect(self) -> bool:
        """Open serial port and return success."""
        if not self.enabled:
            return True
        try:
            import serial as pyserial

            self._ser = pyserial.Serial(self.port, self.baudrate, timeout=1)
            self.connected = True
            return True
        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Serial connect failed ({self.port}): {e}")

    def disconnect(self):
        """Close serial port if open."""
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self.connected = False

    def _format_message(self, package: str, defect: str, quality: str, conf: float) -> str:
        return f"<{package},{defect},{quality},{conf:.2f}>\n"

    def push_decision(self, quality: str, defect_status: str, max_conf: float) -> bool:
        """Add a frame decision to the queue.  Returns True if a stable message was sent."""
        self._decision_queue.append(quality)

        if len(self._decision_queue) < self.stable_frames:
            return False

        # All recent decisions must agree
        if all(q == quality for q in self._decision_queue):
            now_ms = time.time() * 1000
            if quality != self._last_quality or (now_ms - self._last_send_time) > self.cooldown_ms:
                self._last_quality = quality
                self._last_send_time = now_ms

                pkg_token = "package" if quality in ("OK", "NG") else "none"
                def_token = "defect" if defect_status == "defect" else "normal"
                msg = self._format_message(pkg_token, def_token, quality, max_conf)
                return self.send(msg)
        return False

    def send(self, message: str) -> bool:
        """Write bytes to serial port."""
        if not self.connected or self._ser is None or not self._ser.is_open:
            return False
        try:
            self._ser.write(message.encode("utf-8"))
            return True
        except Exception:
            return False
