from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from puripuly_heart.ui import app as app_module

MISSING = object()


class PageDouble:
    def __init__(self, window) -> None:
        self.window = window
        self.visibility_updates: list[bool] = []

    def update(self) -> None:
        self.visibility_updates.append(self.window.visible)


def _make_page(*, wait_until_ready_to_show=MISSING, center=None) -> PageDouble:
    window = SimpleNamespace(visible=False, center=center or (lambda: None))
    if wait_until_ready_to_show is not MISSING:
        window.wait_until_ready_to_show = wait_until_ready_to_show
    return PageDouble(window)


@pytest.mark.asyncio
async def test_prepare_and_show_main_window_awaits_center_before_showing() -> None:
    events: list[str] = []

    async def wait_until_ready_to_show() -> None:
        events.append("ready")

    async def center() -> None:
        events.append("center-start")
        await asyncio.sleep(0)
        events.append("center-end")

    page = _make_page(
        wait_until_ready_to_show=wait_until_ready_to_show,
        center=center,
    )

    await app_module._prepare_and_show_main_window(page)

    assert events == ["ready", "center-start", "center-end"]
    assert page.window.visible is True
    assert page.visibility_updates == [False, True]


@pytest.mark.asyncio
async def test_prepare_and_show_main_window_tolerates_synchronous_center() -> None:
    events: list[str] = []

    def center() -> None:
        events.append("center")

    page = _make_page(center=center)

    await app_module._prepare_and_show_main_window(page)

    assert events == ["center"]
    assert page.window.visible is True
    assert page.visibility_updates == [False, True]


@pytest.mark.asyncio
async def test_prepare_and_show_main_window_tolerates_missing_readiness_method() -> None:
    events: list[str] = []
    page = _make_page(center=lambda: events.append("center"))

    await app_module._prepare_and_show_main_window(page)

    assert events == ["center"]
    assert page.window.visible is True


@pytest.mark.asyncio
async def test_prepare_and_show_main_window_shows_after_readiness_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def wait_until_ready_to_show() -> None:
        raise RuntimeError("window not ready")

    page = _make_page(wait_until_ready_to_show=wait_until_ready_to_show)

    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        await app_module._prepare_and_show_main_window(page)

    assert page.window.visible is True
    assert page.visibility_updates == [False, True]
    assert "Failed to center the main window before showing it" in caplog.text


@pytest.mark.asyncio
async def test_prepare_and_show_main_window_shows_after_center_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def center() -> None:
        raise RuntimeError("center failed")

    page = _make_page(center=center)

    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        await app_module._prepare_and_show_main_window(page)

    assert page.window.visible is True
    assert page.visibility_updates == [False, True]
    assert "Failed to center the main window before showing it" in caplog.text
