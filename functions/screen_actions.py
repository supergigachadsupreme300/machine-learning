"""Screen control actions (scroll / zoom) driven by hand gestures."""

from __future__ import annotations

try:
    import pyautogui as _pag
except Exception:
    _pag = None

from .config import SCROLL_HORIZONTAL, SCROLL_VERTICAL


def _require_pyautogui() -> None:
    if _pag is None:
        raise RuntimeError("pyautogui is required for screen control.")


def swipe_up(amount: int | None = None) -> None:
    _require_pyautogui()
    _pag.scroll(amount if amount is not None else SCROLL_VERTICAL)


def swipe_down(amount: int | None = None) -> None:
    _require_pyautogui()
    _pag.scroll(-(amount if amount is not None else SCROLL_VERTICAL))


def swipe_left(amount: int | None = None) -> None:
    _require_pyautogui()
    amt = amount if amount is not None else SCROLL_HORIZONTAL
    _pag.keyDown("shift")
    _pag.scroll(-amt)
    _pag.keyUp("shift")


def swipe_right(amount: int | None = None) -> None:
    _require_pyautogui()
    amt = amount if amount is not None else SCROLL_HORIZONTAL
    _pag.keyDown("shift")
    _pag.scroll(amt)
    _pag.keyUp("shift")


def zoom_in() -> None:
    _require_pyautogui()
    _pag.keyDown("ctrl")
    _pag.scroll(SCROLL_VERTICAL)
    _pag.keyUp("ctrl")


def zoom_out() -> None:
    _require_pyautogui()
    _pag.keyDown("ctrl")
    _pag.scroll(-SCROLL_VERTICAL)
    _pag.keyUp("ctrl")
