
import cv2
from pathlib import Path

# =========================
# 1. CẤU HÌNH
# =========================

DATA_DIR = Path("data/raw")

LABELS = [
    "MOUSE_UP",
    "MOUSE_DOWN",
    "MOUSE_LEFT",
    "MOUSE_RIGHT",
    "SCROLL_UP",
    "SCROLL_DOWN",
    "SCROLL_LEFT",
    "SCROLL_RIGHT",
    "CLICK",
    "ZOOM_IN",
    "ZOOM_OUT",
    "NEUTRAL"
]

# Tạo thư mục cho từng nhãn
for label in LABELS:
    (DATA_DIR / label).mkdir(parents=True, exist_ok=True)


# =========================
# 2. ĐẾM SỐ ẢNH ĐÃ CÓ
# =========================

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png"}

    return sum(
        1 for file in folder.iterdir()
        if file.suffix.lower() in extensions
    )


# =========================
# 3. MỞ WEBCAM
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

current_label = LABELS[0]

print("===== HUONG DAN =====")
print("U: MOUSE_UP")
print("D: MOUSE_DOWN")
print("L: MOUSE_LEFT")
print("R: MOUSE_RIGHT")
print("I: SCROLL_UP")
print("K: SCROLL_DOWN")
print("J: SCROLL_LEFT")
print("H: SCROLL_RIGHT")
print("O: CLICK")
print("Z: ZOOM_IN")
print("X: ZOOM_OUT")
print("N: NEUTRAL")
print("SPACE: Chup anh")
print("Q: Thoat")


# =========================
# 4. VÒNG LẶP CAMERA
# =========================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc webcam.")
        break

    # Lật hình camera cho dễ thao tác như gương
    frame = cv2.flip(frame, 1)

    folder = DATA_DIR / current_label
    image_count = count_images(folder)

    # Hiển thị nhãn và số ảnh
    cv2.putText(
        frame,
        f"Label: {current_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Images: {image_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE: Capture | Q: Quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # CHỌN NHÃN
    # =========================

    key_map = {
        ord("u"): "MOUSE_UP",
        ord("d"): "MOUSE_DOWN",
        ord("l"): "MOUSE_LEFT",
        ord("r"): "MOUSE_RIGHT",

        ord("i"): "SCROLL_UP",
        ord("k"): "SCROLL_DOWN",
        ord("j"): "SCROLL_LEFT",
        ord("h"): "SCROLL_RIGHT",

        ord("o"): "CLICK",
        ord("z"): "ZOOM_IN",
        ord("x"): "ZOOM_OUT",
        ord("n"): "NEUTRAL"
    }

    if key in key_map:
        current_label = key_map[key]
        print("Da chon nhan:", current_label)

    # =========================
    # CHỤP ẢNH
    # =========================

    elif key == ord(" "):
        folder = DATA_DIR / current_label
        folder.mkdir(parents=True, exist_ok=True)

        image_count = count_images(folder)

        filename = (
            f"{current_label}_{image_count + 1:06d}.jpg"
        )

        save_path = folder / filename

        cv2.imwrite(str(save_path), frame)

        print("Saved:", save_path)

    # =========================
    # THOÁT
    # =========================

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
