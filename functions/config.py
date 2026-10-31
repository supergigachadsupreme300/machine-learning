"""Configuration for gesture -> screen/mouse action functions.

11 gestures -> 11 actions:
    1 click chuột xài cử chỉ OK                  -> ok            -> click_ok()
    4 điều khiển chuột theo 4 hướng (1 ngón chỉ) -> point_*       -> move con trỏ
    4 điều khiển màn hình lướt 4 hướng (2 ngón)  -> swipe_*       -> cuộn màn hình
    1 phóng to màn hình (nắm 5 ngón lại / nắm tay) -> fist        -> zoom_in()
    1 thu nhỏ màn hình (xòe 5 ngón ra)           -> open_hand    -> zoom_out()
"""

from __future__ import annotations

# Mouse movement distance (pixels) per pointing gesture.
MOVE_DISTANCE = 120

# Scroll "clicks" per swipe gesture (pyautogui.scroll: positive = up).
SCROLL_VERTICAL = 6
SCROLL_HORIZONTAL = 6

# Cooldown (seconds) between action triggers so a held gesture does not spam.
ACTION_COOLDOWN = 0.6          # repeat-while-held gestures (mouse move)
DISCRETE_COOLDOWN = 1.0        # one-shot gestures (click, zoom, swipe)

# Gestures that repeat while the hand is held (continuous mouse movement).
REPEAT_WHILE_HELD = {
    "point_up",
    "point_down",
    "point_left",
    "point_right",
}

# Every supported gesture id (11 total).
ALL_GESTURES = [
    "ok",
    "point_up",
    "point_down",
    "point_left",
    "point_right",
    "swipe_up",
    "swipe_down",
    "swipe_left",
    "swipe_right",
    "fist",
    "open_hand",
]
