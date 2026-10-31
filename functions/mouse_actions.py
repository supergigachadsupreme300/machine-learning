"""Mouse control actions driven by hand gestures.

Requires ``pyautogui`` (added to ``requirements.txt``). PyAutoGUI's fail-safe is
enabled by default: slamming the mouse into the upper-left corner aborts the
program, which is a handy emergency stop when experimenting with gestures.
"""

from __future__ import annotations

try:
    import pyautogui as _pag
except Exception:  # pragma: no cover - pyautogui may not be installed yet
    _pag = None

from .config import MOVE_DISTANCE


def _require_pyautogui() -> None:
    if _pag is None:
        raise RuntimeError(
            "pyautogui is required for mouse control. "
            "Install it with: pip install pyautogui"
        )


def click_ok() -> None:
    """Left mouse click.

    Gesture: OK (thumb tip touching index finger).
    """
    _require_pyautogui()
    _pag.click()


def point_up(distance: int | None = None) -> None:
    """Move the mouse cursor up.

    Gesture: index finger pointing up.
    """
    _require_pyautogui()
    _pag.moveRel(0, -(distance or MOVE_DISTANCE))


def point_down(distance: int | None = None) -> None:
    """Move the mouse cursor down.

    Gesture: index finger pointing down.
    """
    _require_pyautogui()
    _pag.moveRel(0, (distance or MOVE_DISTANCE))


def point_left(distance: int | None = None) -> None:
    """Move the mouse cursor left.

    Gesture: index finger pointing left.
    """
    _require_pyautogui()
    _pag.moveRel(-(distance or MOVE_DISTANCE), 0)


def point_right(distance: int | None = None) -> None:
    """Move the mouse cursor right.

    Gesture: index finger pointing right.
    """
    _require_pyautogui()
    _pag.moveRel((distance or MOVE_DISTANCE), 0)
