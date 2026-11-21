"""Inference for Quy-model (landmark-based)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config import MODELS_DIR
from dataset import extract_landmarks
from utils import load_bundle


def predict_image(image_path, model_path=None):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    bundle = load_bundle(model_path if model_path else MODELS_DIR / "best.joblib")
    model = bundle["model"]
    class_names = bundle["class_names"]

    vec = extract_landmarks(str(image_path))
    if vec is None:
        return {"image_path": str(image_path), "label": None,
                "confidence": 0.0, "all_classes": class_names,
                "error": "No hand detected"}
    vec = vec.reshape(1, -1)
    pred = model.predict(vec)[0]
    label = class_names[pred] if isinstance(pred, (int, np.integer)) else str(pred)
    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(vec)[0][pred])
    return {"image_path": str(image_path), "label": str(label),
            "confidence": conf if conf is not None else 1.0,
            "all_classes": class_names}


def main():
    parser = argparse.ArgumentParser(description="Predict gesture from an image.")
    parser.add_argument("image_path", type=Path, nargs="?")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    img = args.image if args.image is not None else args.image_path
    if img is None:
        print("Provide an image path.")
        sys.exit(1)
    result = predict_image(img, model_path=args.model_path)
    if result.get("error"):
        print(result["error"])
        sys.exit(1)
    print(f"Image: {result['image_path']}")
    print(f"Answer: {result['label']}  (confidence={result['confidence']:.4f})")


if __name__ == "__main__":
    main()
