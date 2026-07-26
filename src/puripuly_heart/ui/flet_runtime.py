from __future__ import annotations


def control_page(control: object) -> object | None:
    try:
        return getattr(control, "page", None)
    except RuntimeError:
        return None


def is_control_mounted(control: object) -> bool:
    return control_page(control) is not None


def update_control_if_mounted(control: object) -> bool:
    if not is_control_mounted(control):
        return False
    update = getattr(control, "update", None)
    if not callable(update):
        return False
    try:
        update()
    except (AssertionError, RuntimeError) as exc:
        if "Control must be added" not in str(exc):
            raise
        return False
    return True
