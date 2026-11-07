from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial

import flet as ft

from puripuly_heart.ui.flet_runtime import control_page, update_control_if_mounted

MODAL_START_SCALE = 0.9
MODAL_OPEN = ft.Animation(320, ft.AnimationCurve.EASE_OUT_BACK)
MODAL_CLOSE = ft.Animation(150, ft.AnimationCurve.EASE_OUT)
MODAL_OPEN_DELAY_S = 0.03
PRESS_SCALE = 0.97
PRESS = ft.Animation(100, ft.AnimationCurve.EASE_OUT)
PRESS_HOLD_S = 0.1


def show_modal(page: object, dialog: ft.AlertDialog) -> None:
    """Opens a dialog whose card grows from 90% with a small overshoot."""
    content = dialog.content
    animated = isinstance(content, ft.Control) and _loop(page) is not None
    if animated:
        content.animate_scale = MODAL_OPEN
        content.scale = MODAL_START_SCALE
    page.show_dialog(dialog)
    if animated:
        _later(page, MODAL_OPEN_DELAY_S, partial(_settle, content, MODAL_OPEN, lambda: 1.0))


def close_modal(page: object, dialog: ft.AlertDialog | None) -> None:
    """Shrinks the dialog card while the dialog route fades out."""
    content = getattr(dialog, "content", None)
    if isinstance(content, ft.Control) and content.scale is not None:
        content.animate_scale = MODAL_CLOSE
        content.scale = MODAL_START_SCALE
        update_control_if_mounted(content)
    page.pop_dialog()


def press(
    control: ft.Control,
    *,
    rest: Callable[[], float] = lambda: 1.0,
    animation: ft.Animation = PRESS,
) -> None:
    """Dips a pressed control to 97% and lets it settle back to its rest scale."""
    page = control_page(control)
    if _loop(page) is None:
        return
    control.animate_scale = PRESS
    control.scale = PRESS_SCALE
    update_control_if_mounted(control)
    _later(page, PRESS_HOLD_S, partial(_settle, control, animation, rest))


def press_event(event: object) -> None:
    control = getattr(event, "control", None)
    if isinstance(control, ft.Control):
        press(control)


def _loop(page: object) -> asyncio.AbstractEventLoop | None:
    try:
        loop = getattr(page, "loop", None)
    except Exception:
        return None
    if not isinstance(loop, asyncio.AbstractEventLoop) or loop.is_closed():
        return None
    return loop


def _later(page: object, delay: float, callback: Callable[[], None]) -> None:
    loop = _loop(page)
    if loop is not None:
        loop.call_soon_threadsafe(loop.call_later, delay, callback)


def _settle(control: ft.Control, animation: ft.Animation, rest: Callable[[], float]) -> None:
    control.animate_scale = animation
    control.scale = rest()
    update_control_if_mounted(control)


__all__ = [
    "MODAL_CLOSE",
    "MODAL_OPEN",
    "MODAL_START_SCALE",
    "PRESS",
    "PRESS_SCALE",
    "close_modal",
    "press",
    "press_event",
    "show_modal",
]
