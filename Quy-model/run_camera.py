"""Live camera demo: Quy-model (Decision Tree) -> gesture -> action.

Loads models/best.joblib, predicts the gesture from the webcam, maps it to a
gesture id from functions/, and (optionally) executes the bound action.

Usage:
    python run_camera.py                 # display predictions only
    python run_camera.py --execute       # also perform the mapped action
    python run_camera.py --execute --predict-every 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from config import MODELS_DIR
from dataset import extract_landmarks
from utils import load_bundle

from functions import execute_gesture, get_gesture_label

# Model label (from training) -> function gesture id.
LABEL_TO_GESTURE = {
    "MOUSE_UP": "point_up",
    "MOUSE_DOWN": "point_down",
    "MOUSE_LEFT": "point_left",
    "MOUSE_RIGHT": "point_right",
    "SCROLL_UP": "swipe_up",
    "SCROLL_DOWN": "swipe_down",
    "SCROLL_LEFT": "swipe_left",
    "SCROLL_RIGHT": "swipe_right",
    "CLICK": "ok",
    "ZOOM_IN": "fist",
    "ZOOM_OUT": "open_hand",
}

# Only gestures below repeat while held; others fire once per stable episode.
REPEAT_WHILE_HELD = {"point_up", "point_down", "point_left", "point_right"}

CONFIDENCE_THRESHOLD = 0.6
GESTURE_HOLD_SECONDS = 0.3
DISCRETE_COOLDOWN_EXTRA = 0.0


def main():
    parser = argparse.ArgumentParser(description="Quy-model live camera demo.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the mapped gesture action.")
    parser.add_argument("--predict-every", type=int, default=2,
                        help="Run prediction every N frames.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    args = parser.parse_args()

    bundle = load_bundle(MODELS_DIR / "best.joblib")
    model = bundle["model"]
    class_names = bundle["class_names"]
    has_proba = hasattr(model, "predict_proba")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    # State for stability / hold.
    stable_label = None          # current stable model label
    stable_since = None         # when it became stable
    fired_for_episode = False   # whether a discrete gesture already fired
    current_label = None
    current_conf = None
    current_gesture = None
    last_action_text = ""
    frame_count = 0

    print("Press 'q' to quit." + (" Auto-execute ON." if args.execute else " (display only)"))

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
                label = class_names[int(pred)]
                conf = (float(model.predict_proba(vec.reshape(1, -1))[0][pred])
                        if has_proba else 1.0)

                current_label = label
                current_conf = conf

                if conf >= CONFIDENCE_THRESHOLD:
                    if label != stable_label:
                        stable_label = label
                        stable_since = time.time()
                        fired_for_episode = False
                    gesture_id = LABEL_TO_GESTURE.get(str(label).upper())
                    current_gesture = gesture_id

                    held_long_enough = (stable_since is not None and
                                       (time.time() - stable_since) >= GESTURE_HOLD_SECONDS)

                    if args.execute and gesture_id is not None and held_long_enough:
                        is_repeat = gesture_id in REPEAT_WHILE_HELD
                        if is_repeat or not fired_for_episode:
                            try:
                                ok = execute_gesture(gesture_id, force=False)
                            except Exception as e:
                                ok = False
                                print("Action error:", e)
                            if ok:
                                last_action_text = f"ACTION: {gesture_id}"
                                print(f"Executed: {gesture_id}")
                                if not is_repeat:
                                    fired_for_episode = True
                else:
                    current_label = "low conf"
                    current_gesture = None
                    stable_label = None
                    stable_since = None
                    fired_for_episode = False
            else:
                current_label = "No hand"
                current_conf = None
                current_gesture = None
                stable_label = None
                stable_since = None
                fired_for_episode = False

        # Overlay
        if current_label is not None:
            text = (f"{current_label} ({current_conf:.2f})"
                    if current_conf is not None else current_label)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
        if current_gesture:
            cv2.putText(frame, f"-> {current_gesture} ({get_gesture_label(current_gesture)})",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if last_action_text:
            cv2.putText(frame, last_action_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Quy-model camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
