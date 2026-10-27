"""Camera capture worker.

Runs OpenCV video capture on a background thread so the GUI stays responsive.
Emits a QImage for display and keeps the latest raw BGR frame available for
inference.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


class CameraThread(QThread):
    """Background thread that reads frames from a webcam."""

    frame_ready = Signal(QImage)          # RGB QImage for the GUI
    error_occurred = Signal(str)          # emitted if camera cannot be opened

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent BGR frame, or None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    # ------------------------------------------------------------------ #
    # QThread overrides
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera index {self.camera_index}")
            self._running = False
            return

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Keep a BGR copy for inference (thread-safe)
                with self._lock:
                    self._latest_frame = frame.copy()

                # Convert to RGB QImage for display (deep copy = thread-safe)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(
                    rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                ).copy()
                self.frame_ready.emit(qimg)
        finally:
            cap.release()

    def stop(self) -> None:
        self._running = False
        self.wait(3000)  # wait up to 3 s for the thread to finish
