from __future__ import annotations

import inspect


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


async def invoke_control_method(
    control: object,
    method_name: str,
    *args: object,
) -> object:
    method = getattr(control, method_name)
    result = method(*args)
    if inspect.isawaitable(result):
        return await result
    return result


def run_control_method(
    control: object,
    method_name: str,
    *args: object,
) -> bool:
    page = control_page(control)
    run_task = getattr(page, "run_task", None)
    if not callable(run_task):
        return False
    run_task(invoke_control_method, control, method_name, *args)
    return True
