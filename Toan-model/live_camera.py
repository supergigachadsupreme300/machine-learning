"""Live camera inference for Toan-model.

Opens your webcam, extracts hand landmarks with MediaPipe, runs your trained
Toan-model on every frame, and shows the predicted gesture + confidence on
the video feed.

Usage:
    cd Toan-model
    python live_camera.py

Controls:
    q   - quit
    s   - save current frame (with prediction overlay) to data/live_captures/
    c   - clear the prediction history / reset

Requires:
    - mediapipe (Tasks HandLandmarker model from machine_learning/ui/models/)
    - opencv-python
    - your trained Toan-model/models/best.joblib
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from dataset import _normalize_landmarks
from config import CONFIDENCE_THRESHOLD, GESTURE_HOLD_SECONDS

# Make the workspace-root ``functions`` package importable so the live demo can
# dispatch actions (click, move, scroll, ...) just like the PySide6 UI does.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from functions import execute_gesture  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = _REPO_ROOT
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.joblib"
CLASSES_PATH = Path(__file__).resolve().parent / "models" / "classes.json"
LANDMARKER_MODEL = REPO_ROOT / "machine_learning" / "ui" / "models" / "hand_landmarker.task"
CAPTURE_DIR = Path(__file__).resolve().parent / "data" / "live_captures"

# Landmark names (MediaPipe indices 0..20) — matches extract_landmarks.py
_LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Colors (BGR)
_COLOR_OK = (0, 200, 0)      # green for high confidence
_COLOR_WARN = (0, 200, 255)  # yellow for medium
_COLOR_BAD = (0, 0, 255)     # red for low
_COLOR_TEXT = (255, 255, 255)

# The trained model uses one label convention (CLICK, MOUSE_UP, ...) while the
# ``functions`` package expects another (ok, point_up, ...). Map model labels
# to the function gesture ids so actions actually fire.
_LABEL_TO_GESTURE = {
    "CLICK": "ok",
    "MOUSE_UP": "point_up",
    "MOUSE_DOWN": "point_down",
    "MOUSE_LEFT": "point_left",
    "MOUSE_RIGHT": "point_right",
    "SCROLL_UP": "swipe_up",
    "SCROLL_DOWN": "swipe_down",
    "SCROLL_LEFT": "swipe_left",
    "SCROLL_RIGHT": "swipe_right",
    "ZOOM_IN": "fist",
    "ZOOM_OUT": "open_hand",
}


def _to_gesture_id(label: str) -> Optional[str]:
    """Translate a model label to a ``functions`` gesture id.

    Returns None when the label has no mapped action (do nothing).
    """
    if label in _LABEL_TO_GESTURE:
        return _LABEL_TO_GESTURE[label]
    # Already in gesture-id form (e.g. data collected via the UI).
    from functions import GESTURE_ACTIONS

    return label if label in GESTURE_ACTIONS else None


def _load_model():
    """Load the trained Toan-model bundle and class list."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run: python train.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    class_names = bundle.get("class_names") or bundle["config"].get("classes", [])
    if not class_names and CLASSES_PATH.exists():
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    return model, list(class_names)


def _build_landmarker() -> vision.HandLandmarker:
    if not LANDMARKER_MODEL.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {LANDMARKER_MODEL}\n"
            "Run machine_learning/ui/main.py once to download it, or copy "
            "hand_landmarker.task into machine_learning/ui/models/."
        )
    base = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL))
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,  # video mode for camera
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def _extract_landmarks(frame_rgb, landmarker, timestamp_ms: int) -> Optional[list[float]]:
    """Return 63 landmark values [x0,y0,z0, ...] or None if no hand."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    vec = []
    for p in lm:
        vec += [p.x, p.y, p.z]
    return vec


def _predict(model, class_names, vec: list[float]):
    """Run prediction. Returns (label, confidence, topk_list)."""
    import numpy as np

    arr = np.asarray(vec, dtype=float).flatten()
    # Normalize the same way training does (wrist-centered, scale-invariant)
    arr = _normalize_landmarks(arr)
    pred = model.predict(arr.reshape(1, -1))[0]
    proba = None
    try:
        proba = model.predict_proba(arr.reshape(1, -1))[0]
    except Exception:
        proba = None

    if proba is not None:
        conf = float(proba[list(class_names).index(pred)]) if pred in class_names else 1.0
        order = np.argsort(proba)[::-1][:3]
        topk = [(class_names[i], float(proba[i])) for i in order]
    else:
        conf = 1.0
        topk = [(str(pred), 1.0)]
    return str(pred), conf, topk


def _color_for_conf(conf: float):
    if conf >= 0.8:
        return _COLOR_OK
    if conf >= 0.5:
        return _COLOR_WARN
    return _COLOR_BAD


def _draw_overlay(frame, label, confidence, topk, fps: float, hold_status: str = ""):
    """Draw prediction overlay on the frame."""
    h, w = frame.shape[:2]

    # Status bar at top
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"Toan-model — {label}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        _COLOR_TEXT,
        2,
    )
    conf_color = _color_for_conf(confidence)
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        conf_color,
        2,
    )

    # Hold/confirmation status (center-top)
    if hold_status:
        cv2.putText(
            frame,
            hold_status,
            (w // 2 - 60, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

    # FPS top-right
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        _COLOR_TEXT,
        2,
    )

    # Top-3 list on right side
    y_start = 90
    cv2.putText(
        frame,
        "Top-3:",
        (15, y_start),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    for i, (lbl, prob) in enumerate(topk):
        y = y_start + 25 + i * 22
        cv2.putText(
            frame,
            f"  {i + 1}. {lbl}: {prob:.2f}",
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _COLOR_TEXT,
            1,
        )

    # Controls hint bottom-left
    cv2.putText(
        frame,
        "q: quit | s: save frame | c: clear",
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )


def main():
    # Load model
    try:
        model, class_names = _load_model()
    except FileNotFoundError as exc:
        print("ERROR:", exc)
        return
    print(f"Loaded model. Classes ({len(class_names)}): {class_names}")

    # Load MediaPipe landmarker
    try:
        landmarker = _build_landmarker()
    except FileNotFoundError as exc:
        print("ERROR:", exc)
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        landmarker.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Camera started. Press 'q' to quit, 's' to save frame, 'c' to reset.")
    print("Show your hand to the camera — prediction appears on screen.")

    prev_label = None
    prev_conf = 0.0
    frame_count = 0
    fps_time = cv2.getTickCount()

    # Gesture-confirmation hold: a gesture must remain stable for
    # GESTURE_HOLD_SECONDS before its action fires. If the detected gesture
    # changes (or disappears) during the hold, the pending gesture is reset
    # and nothing fires.
    import time as _time

    pending_gesture: Optional[str] = None
    pending_since: float = 0.0
    fired_gesture: Optional[str] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read frame.")
                break

            # Mirror frame to match data collection (cv2.flip(frame, 1) in collect_dataset.py)
            frame = cv2.flip(frame, 1)

            frame_count += 1
            now = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (now - fps_time) if frame_count > 1 else 0.0
            fps_time = now

            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            label = "--"
            confidence = 0.0
            topk = []

            vec = _extract_landmarks(rgb, landmarker, timestamp_ms)
            if vec is not None:
                label, confidence, topk = _predict(model, class_names, vec)
                # Reject option: a resting hand forces a label among the trained
                # classes, so treat low-confidence predictions as "no gesture".
                if confidence < CONFIDENCE_THRESHOLD:
                    label = "--"

                # Confirmation hold: a gesture must stay stable for
                # GESTURE_HOLD_SECONDS before its action fires. Transitions
                # between gestures reset the hold, so a quick change does not
                # trigger an unintended action. A resting hand (label == "--")
                # also clears the pending gesture.
                gesture_id = _to_gesture_id(label) if label != "--" else None

                if gesture_id is None:
                    # No gesture / unmapped label: cancel any pending hold.
                    pending_gesture = None
                    fired_gesture = None
                elif gesture_id != pending_gesture:
                    # New (or changed) gesture: start a fresh hold.
                    pending_gesture = gesture_id
                    pending_since = _time.time()
                    fired_gesture = None
                else:
                    # Same gesture held: fire once the hold elapses, then keep
                    # firing while it stays held. ``execute_gesture``'s own
                    # cooldown (functions/config.py ACTION_COOLDOWN for mouse
                    # moves, DISCRETE_COOLDOWN for click/zoom/swipe) controls
                    # the repeat rate, so a held gesture repeats instead of
                    # firing only once.
                    held = _time.time() - pending_since
                    if held >= GESTURE_HOLD_SECONDS:
                        try:
                            executed = execute_gesture(gesture_id)
                        except Exception as exc:  # defensive: keep camera running
                            print(f"Action error ({label}): {exc}")
                            executed = False
                        if executed:
                            fired_gesture = gesture_id
                            print(
                                f"-> {label} => {gesture_id} held {held:.2f}s "
                                f"(conf={confidence:.2f})"
                            )
            else:
                # No hand detected: cancel any pending hold.
                pending_gesture = None
                fired_gesture = None

            # Build the hold-confirmation status text for the overlay.
            if pending_gesture is None:
                hold_status = ""
            elif fired_gesture == pending_gesture:
                hold_status = f"FIRED: {pending_gesture}"
            else:
                held_pct = min(
                    (_time.time() - pending_since) / GESTURE_HOLD_SECONDS, 1.0
                )
                hold_status = f"HOLDING {held_pct * 100:.0f}%"

            # Only print when label changes meaningfully
            if label != prev_label or abs(confidence - prev_conf) > 0.1:
                if label != "--":
                    print(f"{label}  (conf={confidence:.2f})")
                prev_label = label
                prev_conf = confidence

            _draw_overlay(frame, label, confidence, topk, fps, hold_status)
            cv2.imshow("Toan-model — Live Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = CAPTURE_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(fname), frame)
                print(f"Saved: {fname}")
            elif key == ord("c"):
                prev_label = None
                prev_conf = 0.0
                pending_gesture = None
                fired_gesture = None
                print("Reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
