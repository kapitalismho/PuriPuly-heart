from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiInputRuntimeAdapter
from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel_present", [True, False], ids=["self-ready", "pipeline-unavailable"]
)
async def test_ui_manual_submit_preserves_self_source(channel_present) -> None:
    admission = TextSubmissionAdmission(
        "admitted", UUID("00000000-0000-0000-0000-000000000295"), ()
    )
    channel = SimpleNamespace(submit_text=AsyncMock(return_value=admission))
    typing = SimpleNamespace(submit=AsyncMock(return_value=admission))
    adapter = UiInputRuntimeAdapter(
        pipeline=SimpleNamespace(self_translation_channel=channel if channel_present else None),
        manual_typing=typing,
        translation=None,
        self_capture=None,
    )

    assert await adapter.submit_text("manual synthetic text") is admission
    channel.submit_text.assert_not_awaited()
    typing.submit.assert_awaited_once()
    (submit,) = typing.submit.await_args.args
    if channel_present:
        assert await submit() is admission
        channel.submit_text.assert_awaited_once_with("manual synthetic text", source="You")
    else:
        assert submit is None
