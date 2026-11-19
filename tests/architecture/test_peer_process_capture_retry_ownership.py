from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiPeerCaptureRuntimeAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [True, False], ids=["restarted", "not-restarted"])
async def test_ui_retry_returns_the_application_owner_outcome(outcome) -> None:
    owner = SimpleNamespace(retry_process_capture=AsyncMock(return_value=outcome))
    adapter = UiPeerCaptureRuntimeAdapter(peer=SimpleNamespace(owner=owner), overlay=None)

    assert await adapter.retry_peer_process_capture() is outcome
    owner.retry_process_capture.assert_awaited_once_with()
