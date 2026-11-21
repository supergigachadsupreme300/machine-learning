"""Training script for Quy-model (landmark-based)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import DATA_ROOT, MODELS_DIR, REPORTS_DIR, TEST_SIZE, SEED, MODEL_NAME
from dataset import load_data
from model import build_model
from utils import save_bundle, save_json


def main():
    parser = argparse.ArgumentParser(description="Train Quy-model (landmarks).")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-per-class", type=int, default=80, help="Max images per class (0=all).")
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else DATA_ROOT
    if not Path(data_dir).exists():
        print(f"ERROR: data not found at {data_dir}")
        sys.exit(1)

    print("Extracting landmarks from images (may take a while)...")
    X, y, class_names = load_data(data_dir, max_per_class=args.max_per_class)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Samples with landmarks: {X.shape[0]}")

    if len(class_names) < 2 or X.shape[0] < 10:
        print("ERROR: not enough data")
        sys.exit(1)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"Train: {Xtr.shape[0]} | Test: {Xte.shape[0]}")

    model = build_model(args.model_name)
    print(f"Model: {args.model_name}")
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(yte, pred, target_names=class_names, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_bundle(MODELS_DIR / "best.joblib", model, class_names,
                config={"model_name": args.model_name})
    save_json(MODELS_DIR / "classes.json", class_names)
    save_json(REPORTS_DIR / "evaluation.json", {
        "accuracy": float(acc), "macro_f1": float(f1),
        "num_samples": int(Xte.shape[0]), "classes": class_names,
    })
    print(f"\nModel saved to {MODELS_DIR / 'best.joblib'}")


if __name__ == "__main__":
    main()
