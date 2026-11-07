from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiInputRuntimeAdapter
from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission
from tests.helpers.paths import REPO_ROOT

COMPOSITION_PATH = REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"


def test_application_manual_typing_owner_is_only_factory_composition() -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "def _begin_manual_submit_typing(" not in source
    assert "def _manual_typing_idle_task(" not in source
    assert source.count("create_manual_typing_owner(") == 1
    assert "return pipeline.translation_output_projection" in source
    assert 'getattr(hub, "set_self_chatbox_typing_reason", None)' not in source
    assert '"clear_self_chatbox_typing_reasons",' not in source
    assert 'getattr(runtime, "translation_tasks", None)' in source
    assert "idle_timeout_seconds=MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S" in source
    assert "submit_timeout_seconds=MANUAL_SUBMIT_TYPING_TIMEOUT_S" in source


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
