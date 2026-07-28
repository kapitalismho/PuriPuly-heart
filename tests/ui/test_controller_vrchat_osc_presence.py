from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.config.settings import AppSettings
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


@pytest.mark.asyncio
async def test_controller_owns_injected_osc_presence_probe_until_cancel() -> None:
    notices: list[bool] = []
    probed_ports: list[int] = []

    class PresencePort:
        async def should_prompt_enable_osc(self, *, port: int) -> bool | None:
            probed_ports.append(port)
            return True

    dashboard = SimpleNamespace(set_vrchat_osc_notice=lambda active: notices.append(bool(active)))
    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(SimpleNamespace(view_dashboard=dashboard)),
        config_path=Path("settings.json"),
        vrchat_osc_presence=PresencePort(),
    )
    controller.settings = AppSettings()

    controller._schedule_vrchat_osc_presence_probe(force=True)
    for _ in range(20):
        if notices:
            break
        await asyncio.sleep(0)

    assert probed_ports == [9000]
    assert notices == [True]

    await controller._cancel_vrchat_osc_presence_probe()

    assert controller._vrchat_osc_probe_task is None
    assert notices == [True, False]
