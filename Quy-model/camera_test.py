"""Live camera test for Quy-model."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IMG_SIZE, MODELS_DIR
from utils import load_bundle
from link import LABEL_TO_GESTURE
from dataset import extract_landmarks
from functions import execute_gesture
from functions.config import MOVE_DISTANCE, SCROLL_VERTICAL, SCROLL_HORIZONTAL
import functions.mouse_actions as ma
import functions.screen_actions as sa

parser = argparse.ArgumentParser()
parser.add_argument("--auto-execute", action="store_true")
parser.add_argument("--predict-every", type=int, default=3)
args = parser.parse_args()

bundle = load_bundle(MODELS_DIR / "best.joblib")
model = bundle["model"]
class_names = bundle["class_names"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam.")
    sys.exit(1)

print("Press 'q' to quit." + (" Auto-execute ON." if args.auto_execute else " (display only)"))

last_action_time = 0
cooldown = 0.8
frame_count = 0
current_label = None
current_gesture = None
current_conf = None
stable_count = 0
stable_label = None
repeat_count = 0
last_executed_gesture = None
last_action_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_count += 1

    if frame_count % args.predict_every == 0:
        vec = extract_landmarks(frame)
        if vec is not None:
            pred = model.predict(vec.reshape(1, -1))[0]
            new_label = class_names[int(pred)]
            if hasattr(model, "predict_proba"):
                new_conf = float(model.predict_proba(vec.reshape(1, -1))[0][pred])
            else:
                new_conf = 1.0

            if new_label == stable_label:
                stable_count += 1
            else:
                stable_label = new_label
                stable_count = 1

            current_label = new_label
            current_conf = new_conf
            current_gesture = LABEL_TO_GESTURE.get(str(new_label).upper())
            print(f"Pred: {current_label} -> {current_gesture}")

            if args.auto_execute and current_gesture is not None and stable_count >= 1:
                now = time.time()
                is_click = (current_gesture == "ok")
                # cooldown only for click (and zoom); movement/scroll run every cycle
                if is_click and (now - last_action_time < 1.0):
                    pass
                else:
                    try:
                        if current_gesture != last_executed_gesture:
                            repeat_count = 0
                        else:
                            repeat_count += 1
                        scale = 1 + repeat_count

                        if current_gesture.startswith("point_"):
                            dist = MOVE_DISTANCE + repeat_count * 10
                            getattr(ma, current_gesture)(distance=dist)
                            ok = True
                        elif current_gesture.startswith("swipe_"):
                            if "left" in current_gesture or "right" in current_gesture:
                                amt = SCROLL_HORIZONTAL * scale
                            else:
                                amt = SCROLL_VERTICAL * scale
                            getattr(sa, current_gesture)(amount=amt)
                            ok = True
                        elif current_gesture in ("fist", "open_hand"):
                            if now - last_action_time >= 1.0:
                                ok = execute_gesture(current_gesture, force=True)
                            else:
                                ok = False
                        else:
                            ok = execute_gesture(current_gesture, force=True)

                        if ok:
                            last_executed_gesture = current_gesture
                            last_action_time = now
                            last_action_text = f"ACTION: {current_gesture} x{scale}"
                            print(f"Executed: {current_gesture} x{scale} -> {ok}")
                    except Exception as e:
                        print("Action error:", e)
        else:
            current_label = "No hand"
            current_conf = None
            current_gesture = None
            stable_count = 0
            stable_label = None

    if current_label is not None:
        text = f"{current_label} ({current_conf:.2f})" if current_conf is not None else current_label
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if current_gesture:
            cv2.putText(frame, f"-> {current_gesture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if last_action_text:
            cv2.putText(frame, last_action_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Quy-model camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
