"""Styled gesture result display with cozy warm theme.

Uses gesture config for dynamic colors so new gestures automatically
get the right color without code changes.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

from config.gesture_config import GestureConfig


class ResultWidget(QFrame):
    """Large readout showing the predicted gesture and confidence."""

    def __init__(
        self,
        config: Optional[GestureConfig] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config or GestureConfig()
        self.setObjectName("resultWidget")
        self.setFixedHeight(130)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.setSpacing(16)

        self.label_gesture = QLabel("Gesture: --")
        self.label_confidence = QLabel("Confidence: --%")

        gesture_font = QFont("Segoe UI", 26, QFont.Weight.DemiBold)
        conf_font = QFont("Segoe UI", 15, QFont.Weight.Medium)
        self.label_gesture.setFont(gesture_font)
        self.label_confidence.setFont(conf_font)

        self.label_gesture.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.label_confidence.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout.addWidget(self.label_gesture, stretch=1)
        layout.addWidget(self.label_confidence)

    def set_result(self, label: str, confidence: float) -> None:
        self.label_gesture.setText(f"Gesture: {label}")
        self.label_confidence.setText(f"Confidence: {confidence * 100:.0f}%")

        # Dynamic color from gesture config
        gesture = self.config.get_gesture(label)
        color = gesture["color"] if gesture else "#5a4d43"
        self.label_gesture.setStyleSheet(f"color: {color};")
