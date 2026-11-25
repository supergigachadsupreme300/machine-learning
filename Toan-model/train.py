"""Training script for Toan-model (landmark-based gesture classifier)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import (
    RAW_DATA_DIR, SPLIT_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    MODELS_DIR, REPORTS_DIR, MODEL_NAME, TEST_SIZE, VAL_SIZE, SEED, CV_FOLDS,
)
from dataset import load_csv_folder, prepare_data, load_split_data
from model import build_model
from utils import save_model_bundle, save_json


def _resolve_source():
    # Prefer split data if present
    if SPLIT_DATA_DIR.exists() and (SPLIT_DATA_DIR / "train").exists():
        return ("split", SPLIT_DATA_DIR)
    if RAW_DATA_DIR.exists():
        return ("raw", RAW_DATA_DIR)
    return (None, None)


def main():
    parser = argparse.ArgumentParser(description="Train Toan-model gesture classifier.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder with CSVs (raw or split).")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation on train set.")
    args = parser.parse_args()

    if args.data_dir is not None:
        source = args.data_dir
        mode = "raw"
    else:
        mode, source = _resolve_source()

    if source is None or not Path(source).exists():
        print("ERROR: No training data found.")
        print("Collect landmarks into data/raw/<gesture>/<gesture>.csv or run split_dataset.py.")
        sys.exit(1)

    print(f"Data source: {source} (mode={mode})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "split":
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = load_split_data(source)
        if X_train.shape[0] == 0:
            print("ERROR: No training samples found in split data.")
            sys.exit(1)
    else:
        prepared = prepare_data(source, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED)
        if prepared is None:
            print("ERROR: No training samples found. Collect data first.")
            sys.exit(1)
        X_train = prepared["X_train"]; y_train = prepared["y_train"]
        X_val = prepared["X_val"]; y_val = prepared["y_val"]
        X_test = prepared["X_test"]; y_test = prepared["y_test"]
        class_names = prepared["class_names"]

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    if len(class_names) < 2:
        print("ERROR: Need at least 2 classes.")
        sys.exit(1)

    model = build_model(args.model_name)
    print(f"Model: {args.model_name}")

    # Cross-validation
    if args.cv:
        try:
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
            print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
        except Exception as e:
            print(f"CV failed: {e}")

    model.fit(X_train, y_train)

    # Validation
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    print(f"Val accuracy: {val_acc:.4f} | Val macro F1: {val_f1:.4f}")

    # Test
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
    print(f"Test accuracy: {test_acc:.4f} | Test macro F1: {test_f1:.4f}")
    print("\nClassification report (test):")
    print(classification_report(y_test, test_pred, target_names=class_names, zero_division=0))

    # Save model bundle
    model_path = MODELS_DIR / "best.joblib"
    save_model_bundle(
        model_path,
        model=model,
        class_names=class_names,
        config={"model_name": args.model_name, "feature_dim": X_train.shape[1]},
    )
    save_json(MODELS_DIR / "classes.json", class_names)

    # Save report
    report = {
        "model_name": args.model_name,
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "num_train": int(X_train.shape[0]),
        "num_val": int(X_val.shape[0]),
        "num_test": int(X_test.shape[0]),
        "classes": class_names,
        "classification_report": classification_report(
            y_test, test_pred, target_names=class_names, zero_division=0, output_dict=True
        ),
    }
    save_json(REPORTS_DIR / "training_report.json", report)

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, test_pred, labels=class_names)
        _plot_confusion(cm, class_names, REPORTS_DIR / "confusion_matrix.png")
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")

    print("\n" + "=" * 60)
    print(f"Training complete. Test accuracy: {test_acc:.4f}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


def _plot_confusion(cm, class_names, path):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
