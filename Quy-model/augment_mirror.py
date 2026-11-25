"""Mirror (horizontal flip) all gesture images to augment the dataset.

Mirroring swaps left/right gestures, so the mirrored image is saved under
the opposite class folder.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_ROOT

SWAP = {
    "MOUSE_LEFT": "MOUSE_RIGHT",
    "MOUSE_RIGHT": "MOUSE_LEFT",
    "SCROLL_LEFT": "SCROLL_RIGHT",
    "SCROLL_RIGHT": "SCROLL_LEFT",
}

data_dir = DATA_ROOT
count = 0
for class_dir in data_dir.iterdir():
    if not class_dir.is_dir():
        continue
    label = class_dir.name
    target_label = SWAP.get(label, label)
    target_dir = data_dir / target_label
    target_dir.mkdir(parents=True, exist_ok=True)
    for img_path in list(class_dir.glob("*.jpg")):
        if "_mirror" in img_path.stem:
            continue
        out_path = target_dir / (img_path.stem + "_mirror.jpg")
        if out_path.exists():
            continue
        with Image.open(img_path) as img:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped.save(out_path)
        count += 1
print(f"Created {count} mirrored images (left/right swapped) in {data_dir}")
