
from pathlib import Path

DATA_DIR = Path("data/split")

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for split in SPLITS:

    print(f"\n===== {split.upper()} =====")

    split_dir = DATA_DIR / split

    total = 0

    for class_dir in sorted(split_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        count = sum(
            1 for file in class_dir.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        )

        total += count

        print(f"{class_dir.name:15s}: {count}")

    print("Total:", total)
