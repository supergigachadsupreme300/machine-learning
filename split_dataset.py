
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. CONFIGURATION
# ==========================================

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/split")

TRAIN_MEMBERS = ["A", "B", "C"]
TEST_MEMBER = "D"

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

EXPECTED_CLASSES = 11

# ==========================================
# 2. CHECK CONFIGURATION
# ==========================================

if not RAW_DIR.exists():
    raise FileNotFoundError(f"Không tìm thấy {RAW_DIR}")

if abs(TRAIN_RATIO + VAL_RATIO - 1.0) > 1e-9:
    raise ValueError("TRAIN_RATIO + VAL_RATIO phải bằng 1.0")

all_members = TRAIN_MEMBERS + [TEST_MEMBER]

for member in all_members:
    if not (RAW_DIR / member).is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục thành viên: {RAW_DIR / member}"
        )

# ==========================================
# 3. DETECT 11 GESTURE CLASSES
# ==========================================

reference_dir = RAW_DIR / TRAIN_MEMBERS[0]

labels = sorted(
    folder.name
    for folder in reference_dir.iterdir()
    if folder.is_dir()
)

if len(labels) != EXPECTED_CLASSES:
    raise ValueError(
        f"Yêu cầu {EXPECTED_CLASSES} gesture, "
        f"nhưng tìm thấy {len(labels)} gesture: {labels}"
    )

# Kiểm tra các thành viên đều có cùng 11 thư mục
for member in all_members:

    member_dir = RAW_DIR / member

    member_labels = sorted(
        folder.name
        for folder in member_dir.iterdir()
        if folder.is_dir()
    )

    if member_labels != labels:
        raise ValueError(
            f"Thư mục gesture của {member} không đồng nhất.\n"
            f"Đang có: {member_labels}\n"
            f"Yêu cầu: {labels}"
        )

print("Đã xác nhận 11 gesture:")
for label in labels:
    print(" -", label)

# ==========================================
# 4. CHECK OUTPUT
# ==========================================

if OUTPUT_DIR.exists():

    existing_files = [
        path for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
    ]

    if existing_files:
        raise FileExistsError(
            f"\n{OUTPUT_DIR} đã có file!\n"
            "Hãy xóa thư mục split cũ trước khi chạy lại."
        )

# Tạo đúng 11 thư mục gesture trong mỗi tập
for split_name in ["train", "val", "test"]:

    for label in labels:

        (OUTPUT_DIR / split_name / label).mkdir(
            parents=True,
            exist_ok=True
        )

# ==========================================
# 5. SPLIT DATASET
# ==========================================

rng = random.Random(RANDOM_SEED)

summary = defaultdict(dict)

total_train = 0
total_val = 0
total_test = 0

print("\n========== SPLITTING DATASET ==========")

for label in labels:

    print(f"\nProcessing: {label}")

    # --------------------------------------
    # A, B, C -> TRAIN + VALIDATION
    # --------------------------------------

    combined_images = []

    for member in TRAIN_MEMBERS:

        source_dir = RAW_DIR / member / label

        images = sorted(
            path for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"  {member}: {len(images)} ảnh")

        for image_path in images:
            combined_images.append((member, image_path))

    # Trộn ảnh A, B, C trong cùng gesture
    rng.shuffle(combined_images)

    total_combined = len(combined_images)

    if total_combined < 2:
        raise ValueError(
            f"Không đủ ảnh để chia Train/Val cho {label}"
        )

    # Chia theo tỷ lệ
    train_count = round(total_combined * TRAIN_RATIO)

    train_count = max(
        1,
        min(train_count, total_combined - 1)
    )

    train_images = combined_images[:train_count]
    val_images = combined_images[train_count:]

    # Copy ảnh Train
    for member, image_path in train_images:

        destination = (
            OUTPUT_DIR / "train" / label
            / f"{member}__{image_path.name}"
        )

        shutil.copy2(image_path, destination)

    # Copy ảnh Validation
    for member, image_path in val_images:

        destination = (
            OUTPUT_DIR / "val" / label
            / f"{member}__{image_path.name}"
        )

        shutil.copy2(image_path, destination)

    # --------------------------------------
    # D -> TEST
    # --------------------------------------

    test_source_dir = RAW_DIR / TEST_MEMBER / label

    test_images = sorted(
        path for path in test_source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in test_images:

        destination = (
            OUTPUT_DIR / "test" / label
            / f"{TEST_MEMBER}__{image_path.name}"
        )

        shutil.copy2(image_path, destination)

    # --------------------------------------
    # STATISTICS
    # --------------------------------------

    train_num = len(train_images)
    val_num = len(val_images)
    test_num = len(test_images)

    summary[label] = {
        "train": train_num,
        "val": val_num,
        "test": test_num
    }

    total_train += train_num
    total_val += val_num
    total_test += test_num

    print(f"  Train      : {train_num}")
    print(f"  Validation : {val_num}")
    print(f"  Test       : {test_num}")

# ==========================================
# 6. FINAL REPORT
# ==========================================

total_all = total_train + total_val + total_test

print("\n" + "=" * 70)
print("             DATASET SPLITTING REPORT")
print("=" * 70)

print(
    f"{'GESTURE':<22}"
    f"{'TRAIN':>10}"
    f"{'VAL':>10}"
    f"{'TEST':>10}"
    f"{'TOTAL':>10}"
)

print("-" * 70)

for label in labels:

    result = summary[label]

    print(
        f"{label:<22}"
        f"{result['train']:>10}"
        f"{result['val']:>10}"
        f"{result['test']:>10}"
        f"{sum(result.values()):>10}"
    )

print("-" * 70)

print(
    f"{'TOTAL':<22}"
    f"{total_train:>10}"
    f"{total_val:>10}"
    f"{total_test:>10}"
    f"{total_all:>10}"
)

if total_all:
    print("\nACTUAL DATASET RATIOS")

    print(f"Train      : {total_train / total_all * 100:.2f}%")
    print(f"Validation : {total_val / total_all * 100:.2f}%")
    print(f"Test       : {total_test / total_all * 100:.2f}%")

print("\nOUTPUT DIRECTORY:", OUTPUT_DIR.resolve())
print("Hoàn thành! Ảnh gốc trong raw vẫn được giữ nguyên.")
