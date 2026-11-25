"""Link Quy-model predictions to gesture action functions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Map model labels -> function gesture ids
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
    "NEUTRAL": None,
}

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions import execute_gesture, get_gesture_label
from predict import predict_image


def predict_and_map(image_path, model_path=None):
    result = predict_image(image_path, model_path=model_path)
    label = result["label"].upper()
    gesture_id = LABEL_TO_GESTURE.get(label)
    return result, gesture_id


def main():
    parser = argparse.ArgumentParser(description="Predict gesture and optionally execute action.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run the mapped action.")
    args = parser.parse_args()

    result, gesture_id = predict_and_map(args.image_path, model_path=args.model_path)
    print(f"Predicted: {result['label']} (conf={result['confidence']:.4f})")
    if gesture_id is None:
        print("Mapped action: none (NEUTRAL)")
        return
    print(f"Mapped gesture: {gesture_id}")
    if args.execute:
        ok = execute_gesture(gesture_id, force=True)
        print(f"Action executed: {ok}")
    else:
        print("Run with --execute to perform the action.")


if __name__ == "__main__":
    main()
