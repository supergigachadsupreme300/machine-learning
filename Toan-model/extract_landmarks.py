"""Convert gesture images in data/raw/ into landmark CSVs for Toan-model training.

Reads images from the repo-root data/raw/<CLASS>/<CLASS>_NNNNNN.jpg and uses
MediaPipe Tasks HandLandmarker to extract 21 hand landmarks (x, y, z) per image.
Writes one CSV per class to Toan-model/data/raw/<CLASS>/<CLASS>.csv with columns:

    timestamp, gesture, wrist_x, wrist_y, wrist_z, thumb_cmc_x, ..., pinky_tip_z

Usage:
    cd Toan-model
    python extract_landmarks.py

Options:
    --source   Source image folder (default: repo-root data/raw)
    --dest     Destination CSV folder (default: Toan-model/data/raw)
    --limit    Max images per class (for testing; 0 = all)
    --model    Path to hand_landmarker.task model (default: machine_learning/ui/models/hand_landmarker.task)
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Root of the whole repo (Toan-model/ is one level below repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw"
DEFAULT_DEST = Path(__file__).resolve().parent / "data" / "raw"
DEFAULT_MODEL = REPO_ROOT / "machine_learning" / "ui" / "models" / "hand_landmarker.task"

# Landmark names matching MediaPipe hand landmark indices 0..20
_LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def _build_landmarker(model_path: Path) -> vision.HandLandmarker:
    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def extract_landmarks_from_image(image_path: Path, landmarker) -> Optional[list[float]]:
    """Return 63 landmark values [x0,y0,z0, x1,y1,z1, ...] or None if no hand found."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    vec = []
    for p in lm:
        vec += [p.x, p.y, p.z]  # normalized 0..1 coords
    return vec


def convert_class(source_dir: Path, dest_dir: Path, class_name: str, landmarker, limit: int = 0) -> int:
    """Convert all images for one class to a single CSV. Returns rows written."""
    images = sorted(
        f for f in source_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if limit > 0:
        images = images[:limit]

    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / f"{class_name}.csv"

    written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header: timestamp, gesture, then x,y,z for each of 21 landmarks
        header = ["timestamp", "gesture"]
        for name in _LANDMARK_NAMES:
            header += [f"{name}_x", f"{name}_y", f"{name}_z"]
        writer.writerow(header)

        for img_path in images:
            vec = extract_landmarks_from_image(img_path, landmarker)
            if vec is None:
                continue
            row = [datetime.now().isoformat(), class_name] + vec
            writer.writerow(row)
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from images to CSV.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Folder with class subfolders of images.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination folder for per-class CSVs.")
    parser.add_argument("--limit", type=int, default=0, help="Max images per class (0 = all).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to hand_landmarker.task model.")
    args = parser.parse_args()

    source = args.source
    dest = args.dest
    model_path = args.model

    if not source.exists():
        print(f"ERROR: Source folder not found: {source}")
        return
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("The UI downloads it automatically; run machine_learning/ui/main.py once, or copy the .task file.")
        return

    class_dirs = sorted(d for d in source.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"ERROR: No class folders found in {source}")
        return

    print(f"Loading MediaPipe HandLandmarker model: {model_path}")
    landmarker = _build_landmarker(model_path)
    print(f"Converting images -> landmarks")
    print(f"  Source: {source}")
    print(f"  Dest:   {dest}")
    print(f"  Classes: {[d.name for d in class_dirs]}")
    print()

    total = 0
    for class_dir in class_dirs:
        n = convert_class(class_dir, dest, class_dir.name, landmarker, limit=args.limit)
        total += n
        print(f"  {class_dir.name}: {n} landmark rows written to {dest / f'{class_dir.name}.csv'}")

    landmarker.close()

    print(f"\nTotal: {total} landmark rows across {len(class_dirs)} classes.")
    print(f"\nNext step: cd Toan-model && python train.py")


if __name__ == "__main__":
    main()
