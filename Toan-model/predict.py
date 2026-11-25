"""Inference for Toan-model: gesture landmarks -> answer (label)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import MODELS_DIR, NUM_LANDMARKS, FEATURE_DIM
from dataset import _normalize_landmarks
from utils import load_model_bundle


def _load_bundle(model_path: Optional[Path] = None):
    if model_path is None:
        model_path = MODELS_DIR / "best.joblib"
        if not model_path.exists():
            # legacy fallback
            for cand in ("model.joblib", "best.pkl", "model.pkl"):
                if (MODELS_DIR / cand).exists():
                    model_path = MODELS_DIR / cand
                    break
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train.py first.")
    bundle = load_model_bundle(model_path)
    return bundle["model"], bundle["class_names"], bundle.get("config", {})


def _extract_landmarks_from_image(image_path: Path) -> Optional[np.ndarray]:
    """Try to extract 21 hand landmarks from an image using MediaPipe (if available)."""
    try:
        import mediapipe as mp
    except Exception:
        return None
    try:
        import cv2
    except Exception:
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Try tasks API first, then legacy
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        model_path = MODELS_DIR / "hand_landmarker.task"
        if model_path.exists():
            base = mp_python.BaseOptions(model_asset_path=str(model_path))
            opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1)
            landmarker = vision.HandLandmarker.create_from_options(opts)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
                vec = []
                for p in lm:
                    vec += [p.x * img.shape[1], p.y * img.shape[0], p.z]
                return np.array(vec, dtype=float)
    except Exception:
        pass

    try:
        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            vec = []
            for p in lm:
                vec += [p.x * img.shape[1], p.y * img.shape[0], p.z]
            return np.array(vec, dtype=float)
    except Exception:
        pass

    return None


def predict_from_landmarks(
    landmarks,
    model_path: Optional[Path] = None,
    topk: int = 3,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    arr = np.asarray(landmarks, dtype=float).flatten()
    if arr.size != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} landmark values, got {arr.size}.")
    model, class_names, cfg = _load_bundle(model_path)
    vec = _normalize_landmarks(arr)
    proba = None
    try:
        proba = model.predict_proba(vec.reshape(1, -1))[0]
    except Exception:
        proba = None
    pred = model.predict(vec.reshape(1, -1))[0]
    if proba is not None:
        idx = list(class_names).index(pred) if pred in class_names else 0
        conf = float(proba[idx]) if idx < len(proba) else 1.0
        # top-k
        order = np.argsort(proba)[::-1][:topk]
        topk_list = [(class_names[i], float(proba[i])) for i in order]
    else:
        conf = 1.0
        topk_list = [(str(pred), 1.0)]

    # Reject option: a resting hand forces a label among the trained classes,
    # so treat low-confidence predictions as "no gesture".
    if min_confidence is not None and conf < min_confidence:
        label: Optional[str] = None
    else:
        label = str(pred)
    return {
        "label": label,
        "confidence": conf,
        "topk": topk_list,
        "all_classes": class_names,
    }


def predict_from_image(image_path: Path, model_path: Optional[Path] = None, topk: int = 3) -> Dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    vec = _extract_landmarks_from_image(image_path)
    if vec is None:
        raise RuntimeError(
            "Could not extract hand landmarks from image. "
            "Install mediapipe and ensure a hand is visible, or pass landmarks directly."
        )
    result = predict_from_landmarks(vec, model_path=model_path, topk=topk)
    result["image_path"] = str(image_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict gesture from landmarks or an image.")
    parser.add_argument("image_path", type=Path, nargs="?", help="Optional image path (requires mediapipe).")
    parser.add_argument("--image", type=Path, default=None, help="Image path.")
    parser.add_argument("--landmarks", type=str, default=None, help="JSON list of 63 landmark floats.")
    parser.add_argument("--csv", type=Path, default=None, help="CSV file with one landmark row.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if args.landmarks is not None:
        vec = json.loads(args.landmarks)
        result = predict_from_landmarks(vec, model_path=args.model_path, topk=args.topk)
    elif args.csv is not None:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if df.empty:
            print("ERROR: CSV empty.")
            sys.exit(1)
        row = df.iloc[0]
        feat_cols = [c for c in df.columns if c not in ("timestamp", "gesture")]
        vec = row[feat_cols].to_numpy(dtype=float)[:FEATURE_DIM]
        result = predict_from_landmarks(vec, model_path=args.model_path, topk=args.topk)
    elif args.image is not None or args.image_path is not None:
        img = args.image if args.image is not None else args.image_path
        result = predict_from_image(img, model_path=args.model_path, topk=args.topk)
    else:
        print("Provide --landmarks, --csv, or an image path.")
        sys.exit(1)

    print(f"Answer: {result['label']}  (confidence={result['confidence']:.4f})")
    print("Top-k:")
    for label, prob in result["topk"]:
        print(f"  {label}: {prob:.4f}")
    print("\nJSON: " + json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
