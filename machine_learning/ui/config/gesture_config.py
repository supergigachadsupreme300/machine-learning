"""Gesture configuration manager.

Loads gestures and categories from gestures.json and provides
dynamic CRUD operations so gestures can be added/removed without
touching code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Project root (ui/ is at machine_learning/ui/, so parents[2] = machine_learning/)
PROJ_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent
GESTURES_CONFIG = CONFIG_DIR / "gestures.json"


class GestureConfig:
    """Manages gesture definitions loaded from gestures.json."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or GESTURES_CONFIG
        self._gestures: list[dict] = []
        self._categories: list[dict] = []
        self._gesture_map: dict[str, dict] = {}
        self._category_map: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._gestures = data.get("gestures", [])
        self._categories = data.get("categories", [])
        self._gesture_map = {g["id"]: g for g in self._gestures}
        self._category_map = {c["id"]: c for c in self._categories}

    def save(self) -> None:
        """Write current state back to gestures.json."""
        data = {
            "gestures": self._gestures,
            "categories": self._categories,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------ #
    # Gestures
    # ------------------------------------------------------------------ #
    def get_gesture(self, gesture_id: str) -> Optional[dict]:
        return self._gesture_map.get(gesture_id)

    def get_all_gestures(self) -> list[dict]:
        return list(self._gestures)

    def get_gestures_by_category(self, category_id: str) -> list[dict]:
        return [g for g in self._gestures if g.get("category") == category_id]

    def get_gesture_ids(self) -> list[str]:
        return [g["id"] for g in self._gestures]

    def add_gesture(
        self,
        gesture_id: str,
        label: str,
        category: str = "custom",
        color: str = "#5a4d43",
    ) -> bool:
        """Add a new gesture. Returns False if ID already exists."""
        if gesture_id in self._gesture_map:
            return False

        gesture = {
            "id": gesture_id,
            "label": label,
            "category": category,
            "color": color,
        }
        self._gestures.append(gesture)
        self._gesture_map[gesture_id] = gesture
        self.save()
        return True

    def remove_gesture(self, gesture_id: str) -> bool:
        """Remove a gesture. Returns False if not found."""
        if gesture_id not in self._gesture_map:
            return False

        self._gestures = [g for g in self._gestures if g["id"] != gesture_id]
        del self._gesture_map[gesture_id]
        self.save()
        return True

    def update_gesture(
        self,
        gesture_id: str,
        label: Optional[str] = None,
        category: Optional[str] = None,
        color: Optional[str] = None,
    ) -> bool:
        """Update an existing gesture's properties."""
        if gesture_id not in self._gesture_map:
            return False

        gesture = self._gesture_map[gesture_id]
        if label is not None:
            gesture["label"] = label
        if category is not None:
            gesture["category"] = category
        if color is not None:
            gesture["color"] = color

        self.save()
        return True

    # ------------------------------------------------------------------ #
    # Categories
    # ------------------------------------------------------------------ #
    def get_all_categories(self) -> list[dict]:
        return list(self._categories)

    def get_category_color(self, category_id: str) -> str:
        cat = self._category_map.get(category_id)
        return cat["color"] if cat else "#5a4d43"
