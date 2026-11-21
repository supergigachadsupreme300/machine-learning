"""Mirror (horizontal flip) all gesture images to augment the dataset."""
from __future__ import annotations

from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_ROOT

data_dir = DATA_ROOT
count = 0
for class_dir in data_dir.iterdir():
    if not class_dir.is_dir():
        continue
    for img_path in list(class_dir.glob("*.jpg")):
        if "_mirror" in img_path.stem:
            continue
        out_path = class_dir / (img_path.stem + "_mirror.jpg")
        if out_path.exists():
            continue
        with Image.open(img_path) as img:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped.save(out_path)
        count += 1
print(f"Created {count} mirrored images in {data_dir}")
