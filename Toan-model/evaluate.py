"""Evaluate a trained Toan-model on landmark test data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from config import RAW_DATA_DIR, SPLIT_DATA_DIR, MODELS_DIR, REPORTS_DIR
from dataset import load_csv_folder, load_split_data, prepare_data
from utils import load_model_bundle, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate Toan-model on test set.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    if args.data_dir is not None:
        source = args.data_dir
    elif SPLIT_DATA_DIR.exists() and (SPLIT_DATA_DIR / "test").exists():
        source = SPLIT_DATA_DIR
    else:
        source = RAW_DATA_DIR

    if args.model_path is None:
        mp = MODELS_DIR / "best.joblib"
    else:
        mp = args.model_path
    bundle = load_model_bundle(mp)
    model = bundle["model"]
    class_names = bundle["class_names"]

    if SPLIT_DATA_DIR.exists() and (SPLIT_DATA_DIR / "test").exists() and args.data_dir is None:
        (_, _), (_, _), (X_test, y_test), _ = load_split_data(SPLIT_DATA_DIR)
    else:
        prepared = prepare_data(source, test_size=0.15, val_size=0.15, seed=42)
        if prepared is None:
            print("ERROR: No data found.")
            sys.exit(1)
        X_test = prepared["X_test"]
        y_test = prepared["y_test"]

    if X_test.shape[0] == 0:
        print("ERROR: No test samples.")
        sys.exit(1)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "num_samples": int(X_test.shape[0]),
        "classes": class_names,
    }
    save_json(REPORTS_DIR / "evaluation.json", result)
    print(f"Saved report to {REPORTS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
