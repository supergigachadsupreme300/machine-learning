"""Utility helpers for Quy-model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_bundle(path, model, class_names, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "class_names": class_names, "config": config or {}}, path)


def load_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
