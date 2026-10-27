"""Inference worker.

Pulls the latest camera frame at a controlled rate and runs the gesture model
on a background thread, emitting results plus an annotated frame (with hand
landmarks drawn) back to the GUI thread.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from model_adapter import ModelAdapter


class InferenceThread(QThread):
    result_ready = Signal(str, float)           # (label, confidence)
    annotated_frame_ready = Signal(QImage)      # RGB QImage with landmarks drawn
    landmarks_ready = Signal(object)            # {landmark_index: (x, y)} pixel coords
    error_occurred = Signal(str)

    def __init__(self, camera_thread, adapter: ModelAdapter, fps: float = 10.0):
        super().__init__()
        self.camera_thread = camera_thread
        self.adapter = adapter
        self._period = 1.0 / fps
        self._running = False

    def run(self) -> None:
        self._running = True
        try:
            while self._running:
                frame = self.camera_thread.latest_frame()
                if frame is not None:
                    try:
                        label, confidence, annotated_bgr, landmarks_dict = self.adapter.predict(frame)
                        self.result_ready.emit(label, confidence)
                        self.landmarks_ready.emit(landmarks_dict)

                        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        bytes_per_line = ch * w
                        qimg = QImage(
                            rgb.data, w, h, bytes_per_line,
                            QImage.Format.Format_RGB888,
                        ).copy()
                        self.annotated_frame_ready.emit(qimg)
                    except Exception as exc:  # pragma: no cover - defensive
                        self.error_occurred.emit(str(exc))
                time.sleep(self._period)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
