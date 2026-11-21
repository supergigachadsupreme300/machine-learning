"""Gesture inference adapter.

Supports two MediaPipe backends:
  1. Legacy ``mp.solutions.hands`` (older mediapipe versions)
  2. Tasks API ``HandLandmarker`` (mediapipe 1.0+ / Python 3.12+)

Recognises:
  * Numbering 0-5  — static finger-count from hand landmarks (rule-based).
  * Swiping        — wrist trajectory over a short sliding window.

Returns an annotated BGR frame with hand landmarks drawn so the UI can show
the hand nodes on the camera feed.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

# Finger tip / PIP landmark indices (MediaPipe Hands)
_TIP_IDS = [4, 8, 12, 16, 20]      # thumb, index, middle, ring, pinky tips
_PIP_IDS = [2, 6, 10, 14, 18]      # corresponding proximal interphalangeal joints
_THUMB_TIP = 4
_THUMB_IP = 2

# MediaPipe hand skeleton connections (landmark index pairs)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)  # palm
]

# Landmark drawing style — nodes distinctly different from lines
_LANDMARK_COLOR = (64, 156, 255)    # bright blue for nodes (pops against skin)
_CONNECTION_COLOR = (200, 160, 130) # muted warm terracotta for lines (recedes)
_LANDMARK_RADIUS = 5
_CONNECTION_THICKNESS = 2
_LANDMARK_OUTLINE = (255, 255, 255) # white outline for nodes

# Landmark names for display (MediaPipe hand landmark indices)
_LANDMARK_NAMES = {
    0:  "Wrist",
    4:  "Thumb Tip",
    8:  "Index Tip",
    12: "Middle Tip",
    16: "Ring Tip",
    20: "Pinky Tip",
}

# MediaPipe Tasks model bundle URL

# MediaPipe Tasks model bundle URL
_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)


class GestureSmoother:
    """Temporal smoothing to reduce jittery predictions.

    Only emits a new gesture when it has been consistent for ``stable_frames``
    consecutive predictions. Returns ``None`` while unstable so the UI can keep
    showing the last stable result.
    """

    def __init__(self, stable_frames: int = 3, history_len: int = 5):
        self.stable_frames = stable_frames
        self.history: deque[str] = deque(maxlen=history_len)
        self._last_stable: Optional[str] = None
        self._run_count = 0
        self._run_gesture: Optional[str] = None

    def update(self, gesture: str) -> Optional[str]:
        """Feed a raw prediction. Returns a stable gesture or None."""
        self.history.append(gesture)

        if gesture == self._run_gesture:
            self._run_count += 1
        else:
            self._run_gesture = gesture
            self._run_count = 1

        if self._run_count >= self.stable_frames:
            self._last_stable = gesture
            return gesture
        return None

    def reset(self) -> None:
        self.history.clear()
        self._run_count = 0
        self._run_gesture = None
        self._last_stable = None

    @property
    def last_stable(self) -> Optional[str]:
        return self._last_stable


class ModelAdapter:
    def __init__(
        self,
        use_mediapipe: bool = True,
        swipe_window: int = 15,
        model_dir: Optional[str | Path] = None,
        stable_frames: int = 3,
    ):
        self.use_mediapipe = False
        self.swipe_window = swipe_window
        self._wrist_history: deque[Tuple[float, float]] = deque(maxlen=swipe_window)
        self._backend: Optional[str] = None   # "legacy" | "tasks" | None
        self._hands = None                    # legacy backend handle
        self._landmarker = None               # tasks backend handle
        self.smoother = GestureSmoother(stable_frames=stable_frames)

        # Mock-mode state (deterministic cycle)
        self._mock_gestures = [
            ("0", 0.99),
            ("1", 0.97),
            ("2", 0.95),
            ("3", 0.94),
            ("4", 0.96),
            ("5", 0.98),
            ("swipe_right", 0.91),
            ("swipe_left", 0.89),
        ]
        self._mock_index = 0

        if not use_mediapipe:
            return

        if self._try_legacy_backend():
            self.use_mediapipe = True
            self._backend = "legacy"
        elif self._try_tasks_backend(model_dir):
            self.use_mediapipe = True
            self._backend = "tasks"

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #
    def _try_legacy_backend(self) -> bool:
        try:
            import mediapipe as mp

            hands_module = mp.solutions.hands
            self._hands = hands_module.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            return True
        except (ImportError, AttributeError, Exception):
            self._hands = None
            return False

    def _try_tasks_backend(self, model_dir: Optional[str | Path]) -> bool:
        try:
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision
        except ImportError:
            return False

        if model_dir is None:
            model_dir = Path(__file__).resolve().parent / "models"
        model_path = Path(model_dir) / "hand_landmarker.task"

        if not model_path.exists():
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(_HAND_LANDMARKER_URL, str(model_path))
            except Exception:
                return False

        try:
            base_options = mp_tasks_python.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            return True
        except Exception:
            self._landmarker = None
            return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, frame_bgr: np.ndarray) -> Tuple[str, float, np.ndarray, dict]:
        """Run inference on a single BGR frame.

        Returns ``(label, confidence, annotated_frame_bgr, landmarks_dict)``
        where ``landmarks_dict`` maps landmark index to ``(x, y)`` pixel coords.
        """
        h, w = frame_bgr.shape[:2]

        if not self.use_mediapipe:
            label, conf = self._mock_predict()
            annotated = frame_bgr.copy()
            return (label, conf, annotated, {})

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated = frame_bgr.copy()

        if self._backend == "legacy":
            import mediapipe as mp

            results = self._hands.process(rgb)
            if not results.multi_hand_landmarks:
                self._wrist_history.clear()
                self.smoother.reset()
                self._draw_landmarks(annotated, None)
                return ("No hand", 0.0, annotated, {})

            lm = results.multi_hand_landmarks[0]
            landmarks_dict = {
                i: (int(p.x * w), int(p.y * h), p.z)
                for i, p in enumerate(lm.landmark)
            }
            self._draw_landmarks(annotated, lm)

            # Use detection confidence from handedness if available
            conf = 0.90
            if results.multi_handedness:
                conf = float(results.multi_handedness[0].classification[0].score)

            label, confidence = self._classify(lm, conf)
            stable = self.smoother.update(label)
            if stable is None:
                label = self.smoother.last_stable or label
                confidence = max(confidence, conf)
            return (label, confidence, annotated, landmarks_dict)

        if self._backend == "tasks":
            import mediapipe as mp

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )
            result = self._landmarker.detect(mp_image)
            if not result.hand_landmarks:
                self._wrist_history.clear()
                self.smoother.reset()
                self._draw_landmarks(annotated, None)
                return ("No hand", 0.0, annotated, {})

            lm_list = result.hand_landmarks[0]
            landmarks_dict = {
                i: (int(p.x * w), int(p.y * h), p.z)
                for i, p in enumerate(lm_list)
            }
            self._draw_landmarks_from_list(annotated, lm_list)

            conf = 0.90
            if result.handedness:
                conf = float(result.handedness[0][0].score)

            # Build a small object with .landmark[i].x/.y for the classifier
            class _FakeLandmarks:
                def __init__(self, landmark_list):
                    self.landmark = landmark_list

            fake_lm = _FakeLandmarks(lm_list)
            label, confidence = self._classify(fake_lm, conf)
            stable = self.smoother.update(label)
            if stable is None:
                label = self.smoother.last_stable or label
                confidence = max(confidence, conf)
            return (label, confidence, annotated, landmarks_dict)

        self._wrist_history.clear()
        self.smoother.reset()
        return ("No hand", 0.0, annotated, {})

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_landmarks(frame_bgr: np.ndarray, landmarks) -> None:
        if landmarks is None:
            return
        h, w = frame_bgr.shape[:2]
        points = []
        for lm in landmarks.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))
        ModelAdapter._draw_skeleton(frame_bgr, points)

    @staticmethod
    def _draw_landmarks_from_list(frame_bgr: np.ndarray, landmark_list) -> None:
        h, w = frame_bgr.shape[:2]
        points = []
        for lm in landmark_list:
            points.append((int(lm.x * w), int(lm.y * h)))
        ModelAdapter._draw_skeleton(frame_bgr, points)

    @staticmethod
    def _draw_skeleton(frame_bgr: np.ndarray, points: list) -> None:
        # Draw connections
        for a, b in _HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(
                    frame_bgr, points[a], points[b],
                    _CONNECTION_COLOR, _CONNECTION_THICKNESS,
                    cv2.LINE_AA,
                )
        # Draw joints
        for x, y in points:
            cv2.circle(
                frame_bgr, (x, y), _LANDMARK_RADIUS,
                _LANDMARK_COLOR, -1, cv2.LINE_AA,
            )
            cv2.circle(
                frame_bgr, (x, y), _LANDMARK_RADIUS + 1,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------ #
    # Internal: numbering (static pose)
    # ------------------------------------------------------------------ #
    def _classify(self, landmarks, detection_conf: float) -> Tuple[str, float]:
        fingers = self._count_extended_fingers(landmarks)
        swipe = self._detect_swipe(landmarks)

        # Dynamic gestures (wrist motion) take priority.
        if swipe is not None:
            return (swipe, min(0.92, detection_conf + 0.02))

        lm = landmarks.landmark

        # 1. OK gesture — thumb tip touching index tip -> left click.
        if self._is_ok_gesture(lm):
            return ("ok", min(0.95, detection_conf + 0.03))

        # 2. Pointing with the index finger only -> move mouse 4 directions.
        if self._is_index_pointing(lm):
            direction = self._point_direction(lm)
            return (f"point_{direction}", min(0.95, detection_conf + 0.03))

        # 3. Fist (no finger extended) -> zoom in.
        if fingers == 0:
            return ("fist", min(0.95, detection_conf + 0.03))

        # 4. Open hand (all fingers extended) -> zoom out.
        if fingers == 5:
            return ("open_hand", min(0.95, detection_conf + 0.03))

        return (str(fingers), min(0.95, detection_conf + 0.03))

    # ------------------------------------------------------------------ #
    # Helpers: control-gesture detection (MediaPipe hand landmark indices)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _finger_extended(lm, tip: int, pip: int) -> bool:
        """True if a finger is extended (tip above its PIP joint)."""
        return lm[tip].y < lm[pip].y - 0.015

    @staticmethod
    def _is_ok_gesture(lm) -> bool:
        """OK: thumb tip (4) touching index tip (8)."""
        thumb = lm[4]
        index = lm[8]
        return math.hypot(thumb.x - index.x, thumb.y - index.y) < 0.05

    @staticmethod
    def _is_index_pointing(lm) -> bool:
        """Index extended while middle, ring and pinky stay closed."""
        return (
            ModelAdapter._finger_extended(lm, 8, 6)
            and not ModelAdapter._finger_extended(lm, 12, 10)
            and not ModelAdapter._finger_extended(lm, 16, 14)
            and not ModelAdapter._finger_extended(lm, 20, 18)
        )

    @staticmethod
    def _point_direction(lm) -> str:
        """Direction of the index finger (tip vs MCP joint)."""
        tip = lm[8]
        mcp = lm[5]
        dx = tip.x - mcp.x
        dy = tip.y - mcp.y  # image coords: y increases downward
        if abs(dx) > abs(dy):
            return "left" if dx < 0 else "right"
        return "up" if dy < 0 else "down"

    @staticmethod
    def _count_extended_fingers(landmarks) -> int:
        """Count extended fingers using tip vs PIP joint height."""
        count = 0
        for tip, pip in zip(_TIP_IDS, _PIP_IDS):
            tip_y = landmarks.landmark[tip].y
            pip_y = landmarks.landmark[pip].y
            # In image coords y increases downward; smaller y = higher up.
            # Require a small margin to avoid jitter on borderline frames.
            if tip_y < pip_y - 0.015:
                count += 1
        # Thumb: tip x vs IP x (handedness-agnostic approximation)
        if landmarks.landmark[_THUMB_TIP].x < landmarks.landmark[_THUMB_IP].x:
            count += 1
        return max(0, min(5, count))

    # ------------------------------------------------------------------ #
    # Internal: swiping (dynamic motion)
    # ------------------------------------------------------------------ #
    def _detect_swipe(self, landmarks) -> Optional[str]:
        wrist = landmarks.landmark[0]
        self._wrist_history.append((wrist.x, wrist.y))
        if len(self._wrist_history) < self.swipe_window:
            return None

        start = self._wrist_history[0]
        end = self._wrist_history[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Require meaningful movement before declaring a swipe.
        if abs(dx) < 0.10 and abs(dy) < 0.10:
            return None

        if abs(dx) > abs(dy):
            return "swipe_right" if dx > 0 else "swipe_left"
        return "swipe_down" if dy > 0 else "swipe_up"

    # ------------------------------------------------------------------ #
    # Internal: mock mode
    # ------------------------------------------------------------------ #
    def _mock_predict(self) -> Tuple[str, float]:
        gesture, conf = self._mock_gestures[self._mock_index]
        self._mock_index = (self._mock_index + 1) % len(self._mock_gestures)
        return (gesture, conf)
