
from pathlib import Path
import random
import shutil

# =========================
# CONFIG
# =========================

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)

# Kiểm tra tổng tỉ lệ
assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6


# =========================
# SPLIT DATA
# =========================

for class_dir in RAW_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    label = class_dir.name

    images = [
        file for file in class_dir.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    datasets = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in datasets.items():

        target_dir = (
            OUTPUT_DIR / split_name / label
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_path in split_images:

            destination = target_dir / image_path.name

            shutil.copy2(
                image_path,
                destination
            )

    print(
        f"{label}: "
        f"train={len(train_images)}, "
        f"val={len(val_images)}, "
        f"test={len(test_images)}"
    )

print("Dataset split completed.")
