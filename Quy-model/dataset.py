"""Dataset loading for Quy-model (MediaPipe hand landmarks)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

from config import HAND_LANDMARKER

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

_landmarker = None


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        base = mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER))
        opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1)
        _landmarker = vision.HandLandmarker.create_from_options(opts)
    return _landmarker


def extract_landmarks(img_bgr) -> Optional[np.ndarray]:
    """Return 63-dim normalized landmark vector or None."""
    if isinstance(img_bgr, (str, Path)):
        img = cv2.imread(str(img_bgr))
        if img is None:
            return None
        img_bgr = img
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = get_landmarker().detect(mp_image)
    if not res.hand_landmarks:
        return None
    lm = res.hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    # normalize: translate by wrist, scale by hand size
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9]) if np.linalg.norm(pts[9]) > 1e-6 else 1.0
    pts = pts / scale
    return pts.flatten()


def load_data(data_dir, max_per_class: int = 0):
    data_dir = Path(data_dir)
    # Support nested layout: data/raw/A/CLICK/*.jpg
    class_dirs_set = {}
    for d in data_dir.rglob("*"):
        if d.is_dir() and any(d.glob("*.jpg")):
            class_dirs_set[d.name] = d
    class_dirs = sorted(class_dirs_set.values(), key=lambda p: p.name)
    class_names = [d.name for d in class_dirs]
    X, y = [], []
    for ci, d in enumerate(class_dirs):
        files = list(d.rglob("*.jpg"))
        if max_per_class and max_per_class > 0:
            files = files[:max_per_class]
        for f in files:
            vec = extract_landmarks(str(f))
            if vec is not None:
                X.append(vec)
                y.append(ci)
    return np.array(X), np.array(y), class_names
