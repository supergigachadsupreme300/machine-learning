"""Gesture -> action functions.

11 functions for 11 gestures:

    * ``ok``              -> click chuột (cử chỉ OK)
    * ``point_up/down/left/right`` -> di chuột 4 hướng (1 ngón chỉ)
    * ``swipe_up/down/left/right`` -> lướt màn hình 4 hướng (2 ngón)
    * ``fist``            -> phóng to màn hình (nắm 5 ngón lại)
    * ``open_hand``       -> thu nhỏ màn hình (xòe 5 ngón ra)
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .config import (
    ACTION_COOLDOWN,
    DISCRETE_COOLDOWN,
    REPEAT_WHILE_HELD,
)
from .mouse_actions import (
    click_ok,
    point_down,
    point_left,
    point_right,
    point_up,
)
from .screen_actions import (
    swipe_down,
    swipe_left,
    swipe_right,
    swipe_up,
    zoom_in,
    zoom_out,
)

__all__ = [
    "click_ok",
    "point_up",
    "point_down",
    "point_left",
    "point_right",
    "swipe_up",
    "swipe_down",
    "swipe_left",
    "swipe_right",
    "zoom_in",
    "zoom_out",
    "GESTURE_ACTIONS",
    "execute_gesture",
    "get_gesture_label",
]

# Map gesture id -> action callable (11 entries).
GESTURE_ACTIONS: dict[str, Callable[[], None]] = {
    "ok": click_ok,
    "point_up": point_up,
    "point_down": point_down,
    "point_left": point_left,
    "point_right": point_right,
    "swipe_up": swipe_up,
    "swipe_down": swipe_down,
    "swipe_left": swipe_left,
    "swipe_right": swipe_right,
    "fist": zoom_in,
    "open_hand": zoom_out,
}

# Human-friendly Vietnamese labels for the result overlay / status bar.
_GESTURE_LABELS = {
    "ok": "OK (click chuột)",
    "point_up": "Chuột lên",
    "point_down": "Chuột xuống",
    "point_left": "Chuột trái",
    "point_right": "Chuột phải",
    "swipe_up": "Lướt lên",
    "swipe_down": "Lướt xuống",
    "swipe_left": "Lướt trái",
    "swipe_right": "Lướt phải",
    "fist": "Phóng to (nắm tay)",
    "open_hand": "Thu nhỏ (xòe tay)",
}

# Last time each gesture was executed, used for the per-gesture cooldown.
_LAST_ACTION_TIME: dict[str, float] = {}


def get_gesture_label(gesture_id: str) -> str:
    """Return a human-friendly label for a gesture id."""
    return _GESTURE_LABELS.get(gesture_id, gesture_id)


def execute_gesture(gesture_id: str, force: bool = False) -> bool:
    """Run the action bound to ``gesture_id``.

    A per-gesture cooldown prevents a held gesture from spamming the action:
    * repeat-while-held gestures (mouse move) fire every ``ACTION_COOLDOWN``
    * one-shot gestures (click, zoom, swipe) fire every ``DISCRETE_COOLDOWN``

    Set ``force=True`` to bypass the cooldown.

    Returns True if the action ran, False if the gesture is unknown or still
    in cooldown.
    """
    action = GESTURE_ACTIONS.get(gesture_id)
    if action is None:
        return False

    now = time.time()
    last = _LAST_ACTION_TIME.get(gesture_id, 0.0)
    cooldown = (
        ACTION_COOLDOWN if gesture_id in REPEAT_WHILE_HELD else DISCRETE_COOLDOWN
    )
    if not force and now - last < cooldown:
        return False

    try:
        action()
    except Exception:
        # Re-raise so the caller (UI) can surface the error instead of
        # silently swallowing it and getting stuck in cooldown.
        raise
    _LAST_ACTION_TIME[gesture_id] = now
    return True
