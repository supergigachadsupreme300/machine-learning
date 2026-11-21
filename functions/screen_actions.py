"""Screen control actions (scroll / zoom) driven by hand gestures.

Requires ``pyautogui``. Note: ``pyautogui.hscroll`` (horizontal scroll) is only
available on Windows; on other platforms horizontal swipes are silently skipped.
"""

from __future__ import annotations

try:
    import pyautogui as _pag
except Exception:  # pragma: no cover - pyautogui may not be installed yet
    _pag = None

from .config import SCROLL_HORIZONTAL, SCROLL_VERTICAL


def _require_pyautogui() -> None:
    if _pag is None:
        raise RuntimeError(
            "pyautogui is required for screen control. "
            "Install it with: pip install pyautogui"
        )


def swipe_up() -> None:
    """Scroll the screen up.

    Gesture: two fingers swiping up.
    """
    _require_pyautogui()
    _pag.scroll(SCROLL_VERTICAL)


def swipe_down() -> None:
    """Scroll the screen down.

    Gesture: two fingers swiping down.
    """
    _require_pyautogui()
    _pag.scroll(-SCROLL_VERTICAL)


def swipe_left() -> None:
    """Scroll the screen left.

    Gesture: two fingers swiping left.
    """
    _require_pyautogui()
    try:
        _pag.hscroll(-SCROLL_HORIZONTAL)
    except Exception:
        # hscroll is Windows-only in pyautogui; ignore elsewhere.
        pass


def swipe_right() -> None:
    """Scroll the screen right.

    Gesture: two fingers swiping right.
    """
    _require_pyautogui()
    try:
        _pag.hscroll(SCROLL_HORIZONTAL)
    except Exception:
        pass


def zoom_in() -> None:
    """Zoom the screen in.

    Gesture: close all five fingers into a fist.
    Works in most Windows apps (browser, Explorer, ...) via Ctrl + wheel up.
    """
    _require_pyautogui()
    _pag.keyDown("ctrl")
    _pag.scroll(SCROLL_VERTICAL)
    _pag.keyUp("ctrl")


def zoom_out() -> None:
    """Zoom the screen out.

    Gesture: open all five fingers.
    Works in most Windows apps (browser, Explorer, ...) via Ctrl + wheel down.
    """
    _require_pyautogui()
    _pag.keyDown("ctrl")
    _pag.scroll(-SCROLL_VERTICAL)
    _pag.keyUp("ctrl")
