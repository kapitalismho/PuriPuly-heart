from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.wiring_composition import (
    create_vrchat_osc_presence_probe_owner,
)


@pytest.mark.asyncio
async def test_composed_presence_owner_uses_injected_port_notice_and_cancel_contract() -> None:
    notices: list[bool] = []
    probed_ports: list[int] = []

    class PresencePort:
        async def should_prompt_enable_osc(self, *, port: int) -> bool | None:
            probed_ports.append(port)
            return True

    owner = create_vrchat_osc_presence_probe_owner(
        presence_provider=PresencePort,
        port_provider=lambda: 9000,
        publish_notice=notices.append,
    )

    owner.schedule(force=True)
    for _ in range(20):
        if notices:
            break
        await asyncio.sleep(0)

    assert probed_ports == [9000]
    assert notices == [True]
    assert owner.task is not None

    await owner.cancel()

    assert owner.task is None
    assert notices == [True, False]
