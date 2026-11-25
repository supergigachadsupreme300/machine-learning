"""Evaluate Quy-model (classical ML)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import DATA_ROOT, MODELS_DIR, REPORTS_DIR, IMG_SIZE, TEST_SIZE, SEED
from dataset import load_data
from utils import load_bundle, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quy-model.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else DATA_ROOT
    if not Path(data_dir).exists():
        print(f"ERROR: data not found at {data_dir}")
        sys.exit(1)

    X, y, class_names = load_data(data_dir, IMG_SIZE)
    _, Xte, _, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    bundle = load_bundle(args.model_path if args.model_path else MODELS_DIR / "best.joblib")
    model = bundle["model"]
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(yte, pred, target_names=class_names, zero_division=0))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(REPORTS_DIR / "evaluation.json", {
        "accuracy": float(acc), "macro_f1": float(f1),
        "num_samples": int(Xte.shape[0]), "classes": class_names,
    })
    print(f"Saved report to {REPORTS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
