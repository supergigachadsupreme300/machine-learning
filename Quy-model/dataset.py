"""Dataset loading helpers for Quy-model (landmark-based)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3


def _normalize_landmarks(vec: np.ndarray) -> np.ndarray:
    """Translate by wrist and scale by hand size for invariance."""
    pts = vec.reshape(NUM_LANDMARKS, 3).copy()
    wrist = pts[0].copy()
    pts = pts - wrist
    # scale by distance wrist -> middle mcp (landmark 9)
    scale = np.linalg.norm(pts[9]) if np.linalg.norm(pts[9]) > 1e-6 else 1.0
    pts = pts / scale
    return pts.flatten().astype(np.float32)


def load_csv_folder(folder: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all gesture CSVs under a folder (raw or split subset).

    Returns (X, y, class_names).
    """
    folder = Path(folder)
    X_list = []
    y_list = []
    if not folder.exists():
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.array([]), []

    # Find CSV files: either folder/<gesture>/<gesture>.csv or folder/*.csv
    csv_files = list(folder.rglob("*.csv"))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "gesture" not in df.columns:
            continue
        # landmark columns: all numeric columns except timestamp/gesture
        feat_cols = [c for c in df.columns if c not in ("timestamp", "gesture")]
        if len(feat_cols) < FEATURE_DIM:
            continue
        feat_cols = feat_cols[:FEATURE_DIM]
        for _, row in df.iterrows():
            label = str(row["gesture"])
            try:
                vec = row[feat_cols].to_numpy(dtype=float)
            except Exception:
                continue
            if np.isnan(vec).any():
                continue
            vec = _normalize_landmarks(vec)
            X_list.append(vec)
            y_list.append(label)

    if not X_list:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.array([]), []

    X = np.stack(X_list)
    y = np.array(y_list)
    class_names = sorted(set(y_list))
    return X, y, class_names


def load_split_data(root: Path):
    """Load from data/split/{train,val,test} if available."""
    root = Path(root)
    train_X, train_y, classes = load_csv_folder(root / "train")
    val_X, val_y, _ = load_csv_folder(root / "val")
    test_X, test_y, _ = load_csv_folder(root / "test")
    return (train_X, train_y), (val_X, val_y), (test_X, test_y), classes


def prepare_data(source, test_size=0.15, val_size=0.15, seed=42):
    """Load data from source folder and split into train/val/test."""
    X, y, class_names = load_csv_folder(source)
    if X.shape[0] == 0:
        return None

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Split train/val
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=seed, stratify=y_trainval
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
    }
