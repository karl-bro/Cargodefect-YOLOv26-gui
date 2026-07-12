#!/usr/bin/env python3
"""Serial sender for STM32 communication (safe stub)."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SerialSender:
    """Send OK/NG results to STM32 via serial port."""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False

    def connect(self) -> bool:
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connected = True
            logger.info(f"Connected to {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            logger.warning(f"Serial connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False

    def send_result(self, cargo_class: str, defect_status: str,
                    quality_result: str, confidence: float) -> bool:
        """
        Send format: <cargo_class,defect_status,OK_or_NG,confidence>
        """
        if not self.connected:
            return False

        try:
            msg = f"<{cargo_class},{defect_status},{quality_result},{confidence:.2f}>"
            self.serial.write(msg.encode())
            logger.info(f"Sent: {msg}")
            return True
        except Exception as e:
            logger.error(f"Serial send failed: {e}")
            self.connected = False
            return False
