"""Hand landmark position panel.

Shows live pixel (x, y) coordinates for key hand nodes next to the result box.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

_KEY_LANDMARKS = [
    (0, "Wrist"),
    (4, "Thumb Tip"),
    (8, "Index Tip"),
    (12, "Middle Tip"),
    (16, "Ring Tip"),
    (20, "Pinky Tip"),
]

_NODE_COLOR = "#409cff"   # bright blue for active node text
_DIM_COLOR = "#7a6d62"    # muted warm gray for inactive


class LandmarkPanel(QFrame):
    """Scrollable panel showing hand landmark (node) pixel coordinates."""

    def __init__(self):
        super().__init__()
        self.setObjectName("landmarkPanel")
        self.setMinimumHeight(200)          # tall enough for all 6 rows
        self.setMaximumHeight(260)
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("Hand Node Positions")
        header.setObjectName("landmarkHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #5a4d43; background: transparent;")
        layout.addWidget(header)

        # Content widget holding the grid
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(content)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(4)

        self._coord_labels: dict[int, QLabel] = {}
        self._name_labels: dict[int, QLabel] = {}

        for row, (idx, name) in enumerate(_KEY_LANDMARKS):
            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Segoe UI", 10))
            name_lbl.setStyleSheet(f"color: {_DIM_COLOR}; background: transparent;")

            coord_lbl = QLabel("--, --")
            coord_lbl.setFont(QFont("Consolas", 10, QFont.Weight.Medium))
            coord_lbl.setStyleSheet("color: #5a4d43; background: transparent;")
            coord_lbl.setMinimumWidth(60)
            coord_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            self.grid.addWidget(name_lbl, row, 0)
            self.grid.addWidget(coord_lbl, row, 1)

            self._name_labels[idx] = name_lbl
            self._coord_labels[idx] = coord_lbl

        layout.addWidget(content)

        # Footer hint
        hint = QLabel("Pixel (x, y)")
        hint.setObjectName("landmarkHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setFont(QFont("Segoe UI", 9, QFont.Weight.Light))
        hint.setStyleSheet("color: #9a8c7f; background: transparent;")
        layout.addWidget(hint)

        self.clear()

    def set_landmarks(self, landmarks_dict: dict) -> None:
        """Update displayed coordinates from {index: (x, y)} dict."""
        if not landmarks_dict:
            self.clear()
            return

        for idx, (x, y, _) in landmarks_dict.items():
            if idx in self._coord_labels:
                self._coord_labels[idx].setText(f"{x}, {y}")
                self._coord_labels[idx].setStyleSheet(
                    f"color: {_NODE_COLOR}; background: transparent; font-weight: 600;"
                )
                self._name_labels[idx].setStyleSheet(
                    f"color: {_NODE_COLOR}; background: transparent; font-weight: 600;"
                )

    def clear(self) -> None:
        """Reset all coordinates to dashes."""
        for idx in self._coord_labels:
            self._coord_labels[idx].setText("--, --")
            self._coord_labels[idx].setStyleSheet(
                f"color: #5a4d43; background: transparent;"
            )
            self._name_labels[idx].setStyleSheet(
                f"color: {_DIM_COLOR}; background: transparent;"
            )
