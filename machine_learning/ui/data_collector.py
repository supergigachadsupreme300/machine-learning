"""Data collection widget for gesture training data.

Saves hand landmark (node) coordinates into per-gesture CSV files under
``data/raw/`` instead of full images — much smaller and more useful for
training a landmark-based gesture model.

Gestures are loaded dynamically from config/gestures.json so new gestures
can be added or removed without touching code.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config.gesture_config import GestureConfig

# Landmark names for all 21 MediaPipe hand landmarks
_LANDMARK_NAMES = {
    0:  "wrist",
    1:  "thumb_cmc",
    2:  "thumb_mcp",
    3:  "thumb_ip",
    4:  "thumb_tip",
    5:  "index_mcp",
    6:  "index_pip",
    7:  "index_dip",
    8:  "index_tip",
    9:  "middle_mcp",
    10: "middle_pip",
    11: "middle_dip",
    12: "middle_tip",
    13: "ring_mcp",
    14: "ring_pip",
    15: "ring_dip",
    16: "ring_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip",
}

# Landmark indices we save (all 21 MediaPipe hand landmarks)
LANDMARK_INDICES = list(range(21))

# Project root (ui/ is at machine_learning/ui/, so parents[2] = machine_learning/)
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data" / "raw"


class AddGestureDialog(QDialog):
    """Dialog for adding a new gesture."""

    def __init__(self, config: GestureConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Add New Gesture")
        self.setModal(True)
        self.resize(400, 250)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Gesture ID
        layout.addWidget(QLabel("Gesture ID (e.g., 'peace', 'thumbs_up'):"))
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("peace")
        layout.addWidget(self.id_edit)

        # Display label
        layout.addWidget(QLabel("Display Label:"))
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Peace Sign")
        layout.addWidget(self.label_edit)

        # Category
        layout.addWidget(QLabel("Category:"))
        self.category_combo = QPushButton()  # placeholder, replaced below
        from PySide6.QtWidgets import QComboBox

        self.category_combo = QComboBox()
        for cat in self.config.get_all_categories():
            self.category_combo.addItem(cat["label"], cat["id"])
        layout.addWidget(self.category_combo)

        # Color
        layout.addWidget(QLabel("Color:"))
        self.color_edit = QLineEdit("#5a4d43")
        self.color_edit.setPlaceholderText("#hexcolor")
        layout.addWidget(self.color_edit)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        gid = self.id_edit.text().strip()
        label = self.label_edit.text().strip()
        category = self.category_combo.currentData()
        color = self.color_edit.text().strip() or "#5a4d43"

        if not gid:
            QMessageBox.warning(self, "Invalid Input", "Gesture ID is required.")
            return
        if not label:
            label = gid

        if not self.config.add_gesture(gid, label, category, color):
            QMessageBox.warning(self, "Duplicate", f"Gesture '{gid}' already exists.")
            return

        self.accept()

    def get_gesture_id(self) -> Optional[str]:
        return self.id_edit.text().strip() or None


class DataCollector(QObject):
    """Handles saving landmark coordinates to per-gesture CSV files."""

    saved = Signal(str, str)   # (gesture_id, filepath)

    def __init__(self, config: Optional[GestureConfig] = None):
        super().__init__()
        self.config = config or GestureConfig()
        self._counts: dict[str, int] = {
            g["id"]: 0 for g in self.config.get_all_gestures()
        }
        self._latest_landmarks: Optional[dict] = None
        self._ensure_dirs()
        self._init_csvs()

    def _ensure_dirs(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for g in self.config.get_all_gestures():
            (DATA_DIR / g["id"]).mkdir(parents=True, exist_ok=True)

    def _init_csvs(self) -> None:
        """Create per-gesture CSV files with headers if they don't exist."""
        for g in self.config.get_all_gestures():
            csv_path = DATA_DIR / g["id"] / f"{g['id']}.csv"
            if not csv_path.exists():
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    header = ["timestamp", "gesture"]
                    for i in LANDMARK_INDICES:
                        name = _LANDMARK_NAMES.get(i, f"lm{i}")
                        header += [f"{name}_x", f"{name}_y", f"{name}_z"]
                    writer.writerow(header)

    def reload_config(self) -> None:
        """Reload gesture config and re-initialize directories/CSVs."""
        self.config._load()
        self._counts = {g["id"]: 0 for g in self.config.get_all_gestures()}
        self._ensure_dirs()
        self._init_csvs()

    def update_landmarks(self, landmarks_dict: dict) -> None:
        """Store the latest landmark coordinates from the inference thread."""
        self._latest_landmarks = landmarks_dict

    def save_landmarks(self, gesture_id: str) -> Optional[str]:
        """Save the latest landmark coordinates to the gesture's CSV file.

        Returns the saved file path, or None if no landmarks available.
        """
        if gesture_id not in self.config.get_gesture_ids():
            return None
        if not self._latest_landmarks:
            return None

        self._counts[gesture_id] = self._counts.get(gesture_id, 0) + 1
        count = self._counts[gesture_id]
        ts_iso = datetime.now().isoformat()

        csv_path = DATA_DIR / gesture_id / f"{gesture_id}.csv"

        # Build row: timestamp, gesture, then x,y,z for each landmark
        row = [ts_iso, gesture_id]
        for i in LANDMARK_INDICES:
            if i in self._latest_landmarks:
                x, y, z = self._latest_landmarks[i]
                row += [x, y, z]
            else:
                row += [None, None, None]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.saved.emit(gesture_id, str(csv_path))
        return str(csv_path)

    def count(self, gesture_id: str) -> int:
        return self._counts.get(gesture_id, 0)

    def total(self) -> int:
        return sum(self._counts.values())

    def has_landmarks(self) -> bool:
        return self._latest_landmarks is not None and len(self._latest_landmarks) > 0

    def remove_gesture_data(self, gesture_id: str) -> bool:
        """Remove all saved data for a gesture (CSV file and folder)."""
        if gesture_id not in self.config.get_gesture_ids():
            return False

        csv_path = DATA_DIR / gesture_id / f"{gesture_id}.csv"
        gesture_dir = DATA_DIR / gesture_id

        try:
            if csv_path.exists():
                csv_path.unlink()
            if gesture_dir.exists():
                gesture_dir.rmdir()  # only removes if empty
        except OSError:
            pass

        self._counts.pop(gesture_id, None)
        self.config.remove_gesture(gesture_id)
        return True


class DataCollectorWidget(QWidget):
    """UI for collecting training data — one button per gesture.

    Gestures are loaded dynamically from config. Supports adding/removing
    gestures at runtime.
    """

    def __init__(
        self,
        collector: DataCollector,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.collector = collector
        self.config = collector.config

        self.setObjectName("dataCollectorWidget")
        self._buttons: dict[str, QPushButton] = {}
        self._count_labels: dict[str, QLabel] = {}
        self._remove_buttons: dict[str, QPushButton] = {}
        self._build_ui()
        self._refresh_counts()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scroll area so the panel stays compact
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(600)

        content = QWidget()
        content.setObjectName("dataCollectorContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)

        # Header with add button
        header_row = QHBoxLayout()
        header = QLabel("Collect Training Data")
        header.setObjectName("dataHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        header.setStyleSheet("color: #5a4d43; background: transparent;")

        self.btn_add = QPushButton("+ Add Gesture")
        self.btn_add.setObjectName("btnAddGesture")
        self.btn_add.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        self.btn_add.setMaximumWidth(110)
        self.btn_add.clicked.connect(self._add_gesture)

        header_row.addWidget(header)
        header_row.addStretch()
        header_row.addWidget(self.btn_add)
        layout.addLayout(header_row)

        # Instructions
        instr = QLabel("Click a gesture button to save the current hand node positions.")
        instr.setObjectName("dataInstr")
        instr.setAlignment(Qt.AlignmentFlag.AlignLeft)
        instr.setFont(QFont("Segoe UI", 8, QFont.Weight.Light))
        instr.setStyleSheet("color: #7a6d62; background: transparent;")
        layout.addWidget(instr)

        # Category filter tabs
        self.category_combo = QPushButton()  # placeholder
        from PySide6.QtWidgets import QComboBox

        self.category_combo = QComboBox()
        self.category_combo.addItem("All Categories", "all")
        for cat in self.config.get_all_categories():
            self.category_combo.addItem(cat["label"], cat["id"])
        self.category_combo.currentIndexChanged.connect(self._filter_by_category)
        self.category_combo.setMaximumWidth(150)
        layout.addWidget(self.category_combo)

        # Landmark availability indicator
        self.landmark_status = QLabel("Waiting for hand landmarks...")
        self.landmark_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.landmark_status.setFont(QFont("Segoe UI", 8, QFont.Weight.Light))
        self.landmark_status.setStyleSheet("color: #7a6d62; background: transparent;")
        layout.addWidget(self.landmark_status)

        # Gesture button grid (scrollable within the main scroll area)
        # Use QVBoxLayout with QHBoxLayout rows so delete buttons stay square
        self.grid_widget = QWidget()
        self.grid = QVBoxLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)

        self._rebuild_gesture_grid()

        layout.addWidget(self.grid_widget)

        # Total + status
        status_layout = QHBoxLayout()
        self.total_lbl = QLabel("Total saved: 0")
        self.total_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self.total_lbl.setStyleSheet("color: #5a4d43; background: transparent;")
        self.status_lbl = QLabel("Waiting for hand landmarks...")
        self.status_lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Light))
        self.status_lbl.setStyleSheet("color: #7a6d62; background: transparent;")

        status_layout.addWidget(self.total_lbl)
        status_layout.addStretch()
        status_layout.addWidget(self.status_lbl)
        layout.addLayout(status_layout)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------ #
    # Dynamic grid
    # ------------------------------------------------------------------ #
    def _rebuild_gesture_grid(self) -> None:
        """Rebuild the gesture button grid from current config."""
        # Clear existing widgets
        for btn in self._buttons.values():
            btn.deleteLater()
        for lbl in self._count_labels.values():
            lbl.deleteLater()
        for btn in self._remove_buttons.values():
            btn.deleteLater()

        self._buttons.clear()
        self._count_labels.clear()
        self._remove_buttons.clear()

        # Get filtered gestures
        category_id = self.category_combo.currentData()
        if category_id == "all":
            gestures = self.config.get_all_gestures()
        else:
            gestures = self.config.get_gestures_by_category(category_id)

        # Stack gestures vertically (one per row using QHBoxLayout)
        for gesture in gestures:
            gid = gesture["id"]
            label = gesture["label"]
            color = gesture.get("color", "#5a4d43")

            # Row layout
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)

            # Gesture button
            btn = QPushButton(label)
            btn.setObjectName(f"btnCollect{gid}")
            btn.setMinimumHeight(36)
            btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
            btn.setStyleSheet(
                f"QPushButton {{ background: #ffffff; border: 1.5px solid {color}; "
                f"border-radius: 8px; color: {color}; text-align: left; padding-left: 12px; }}"
            )
            btn.clicked.connect(lambda checked=False, g=gid: self._capture(g))

            # Count label
            count_lbl = QLabel("0")
            count_lbl.setObjectName(f"count{gid}")
            count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            count_lbl.setFont(QFont("Consolas", 9, QFont.Weight.Medium))
            count_lbl.setMinimumWidth(30)
            count_lbl.setStyleSheet(f"color: {color}; background: transparent;")

            # Remove button (small square)
            remove_btn = QPushButton("×")
            remove_btn.setObjectName(f"btnRemove{gid}")
            remove_btn.setMinimumSize(24, 24)
            remove_btn.setMaximumSize(24, 24)
            remove_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            remove_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            remove_btn.setStyleSheet(
                "QPushButton { background: #f7e7e0; border: 1px solid #e3c8b8; "
                "border-radius: 4px; color: #9c5a46; padding: 0px; }"
                "QPushButton:hover { background: #f1d7c9; padding: 0px; }"
            )
            remove_btn.clicked.connect(lambda checked=False, g=gid: self._remove_gesture(g))

            row.addWidget(btn, stretch=1)
            row.addWidget(count_lbl, stretch=0)
            row.addWidget(remove_btn, stretch=0)

            self.grid.addLayout(row)

            self._buttons[gid] = btn
            self._count_labels[gid] = count_lbl
            self._remove_buttons[gid] = remove_btn

        # Add stretch to push buttons to top
        self.grid.addStretch()

    # ------------------------------------------------------------------ #
    # Category filter
    # ------------------------------------------------------------------ #
    def _filter_by_category(self) -> None:
        self._rebuild_gesture_grid()
        self._refresh_counts()

    # ------------------------------------------------------------------ #
    # Add / Remove gesture
    # ------------------------------------------------------------------ #
    def _add_gesture(self) -> None:
        dialog = AddGestureDialog(self.config, self)
        if dialog.exec():
            new_id = dialog.get_gesture_id()
            if new_id:
                self.collector.reload_config()
                self._rebuild_gesture_grid()
                self._refresh_counts()
                self.status_lbl.setText(f"Added gesture: {new_id}")
                self.status_lbl.setStyleSheet("color: #2e7d32; background: transparent;")

    def _remove_gesture(self, gesture_id: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Remove Gesture",
            f"Remove gesture '{gesture_id}' and delete its saved data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.collector.remove_gesture_data(gesture_id)
            self.collector.reload_config()
            self._rebuild_gesture_grid()
            self._refresh_counts()
            self.status_lbl.setText(f"Removed gesture: {gesture_id}")
            self.status_lbl.setStyleSheet("color: #b3261e; background: transparent;")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update_landmarks(self, landmarks_dict: dict) -> None:
        """Receive latest landmarks from the inference thread."""
        self.collector.update_landmarks(landmarks_dict)

        if landmarks_dict:
            self.landmark_status.setText(
                f"Hand detected — {len(landmarks_dict)} landmarks ready"
            )
            self.landmark_status.setStyleSheet(
                "color: #2e7d32; background: transparent;"
            )
        else:
            self.landmark_status.setText("No hand detected")
            self.landmark_status.setStyleSheet(
                "color: #b3261e; background: transparent;"
            )

    def _capture(self, gesture_id: str) -> None:
        if not self.collector.has_landmarks():
            self.status_lbl.setText("No hand landmarks available yet.")
            self.status_lbl.setStyleSheet("color: #b3261e; background: transparent;")
            return

        path = self.collector.save_landmarks(gesture_id)
        if path is None:
            self.status_lbl.setText(f"Failed to save {gesture_id}.")
            self.status_lbl.setStyleSheet("color: #b3261e; background: transparent;")
            return

        self._refresh_counts()
        gesture = self.config.get_gesture(gesture_id)
        label = gesture["label"] if gesture else gesture_id
        self.status_lbl.setText(f"Saved {label}")
        self.status_lbl.setStyleSheet("color: #2e7d32; background: transparent;")

    def _refresh_counts(self) -> None:
        for gesture_id in self.config.get_gesture_ids():
            if gesture_id in self._count_labels:
                self._count_labels[gesture_id].setText(
                    str(self.collector.count(gesture_id))
                )
        self.total_lbl.setText(f"Total saved: {self.collector.total()}")
