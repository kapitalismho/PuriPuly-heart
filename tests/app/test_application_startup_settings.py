from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from puripuly_heart.app.ports.application_startup import ApplicationStartupState
from puripuly_heart.app.wiring.wiring_application_runtime_logging import (
    compose_application_runtime_logging,
)
from puripuly_heart.composition.application_startup import ApplicationStartupAdapter
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.conversation_history import ConversationHistory
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
from puripuly_heart.ui.i18n import get_locale, set_locale
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


class _FailingPersistOwner:
    def __init__(self, settings: AppSettingsVNext) -> None:
        self.canonical = settings
        self.authoritative = False

    def remember_projection(self, _settings: AppSettingsVNext) -> None:
        return None

    def save_current(self, failure_sink=None) -> bool:
        if failure_sink is not None:
            failure_sink(RuntimeError("persist failed"))
        return False

    def set_overlay_enabled(self, _enabled: bool) -> None:
        return None

    def set_peer_translation_enabled(self, _enabled: bool) -> None:
        return None


class _RecordingPersistOwner(_FailingPersistOwner):
    def __init__(self, settings: AppSettingsVNext, *, save_succeeds: bool = True) -> None:
        super().__init__(settings)
        self.save_succeeds = save_succeeds
        self.saved: list[AppSettingsVNext] = []

    def save_current(self, failure_sink=None) -> bool:
        if not self.save_succeeds:
            return super().save_current(failure_sink)
        self.saved.append(self.canonical)
        return True


def _startup_adapter(
    owner: _FailingPersistOwner,
    settings: AppSettingsVNext,
    *,
    presentation: object | None = None,
) -> ApplicationStartupAdapter:
    provisioning = Mock()
    provisioning.snapshot = SimpleNamespace(cpu_auto_available=True)
    provisioning.inspect_cpu = AsyncMock()
    provisioning.inspect_gpu = AsyncMock()
    return ApplicationStartupAdapter(
        settings=owner,
        settings_loader=lambda: settings,
        provisioning=provisioning,
        gpu_state=lambda: SimpleNamespace(selected_provider_requires_model=False),
        calibration=Mock(),
        presentation=presentation if presentation is not None else Mock(),
        sync_presentation=lambda: None,
        runtime_logging=Mock(),
        initial_llm_binding_sink=lambda _settings, _provider: None,
        pipeline_launcher=Mock(),
        pipeline=Mock(),
        sync_local_asr_notice=lambda: None,
        stt_requires_secret=lambda _provider: False,
        llm_requires_secret=lambda _provider: False,
        alibaba_verified_key=lambda: "",
        managed_translation_available=lambda: False,
        receiver_active=lambda: False,
        create_event_bridge=lambda _logging: Mock(),
        start_event_bridge=lambda _bridge: None,
        wait_for_event_bridge=AsyncMock(),
        sync_clipboard=AsyncMock(),
        start_photo=AsyncMock(),
    )


def _with_main_input_first_launch_completed(settings: AppSettingsVNext) -> AppSettingsVNext:
    return replace(
        settings,
        state=replace(
            settings.state,
            main_input_hint=replace(settings.state.main_input_hint, first_launch_completed=True),
        ),
    )


@pytest.mark.asyncio
async def test_first_launch_is_marked_during_startup_and_shown_with_the_locale() -> None:
    loaded = AppSettingsVNext()
    owner = _RecordingPersistOwner(loaded)
    presentation = Mock()
    adapter = _startup_adapter(owner, loaded, presentation=presentation)

    state = await adapter.prepare_startup_settings()
    adapter.apply_startup_presentation(state)

    assert state.main_input_first_launch is True
    assert [saved.state.main_input_hint.first_launch_completed for saved in owner.saved] == [True]
    assert owner.canonical.state.main_input_hint.first_launch_completed is True
    presentation.set_dashboard_launch_input_hint.assert_called_once_with(first_launch=True)
    call_names = [call[0] for call in presentation.method_calls]
    assert call_names.index("set_dashboard_launch_input_hint") < call_names.index("apply_locale")


@pytest.mark.asyncio
async def test_later_launch_skips_the_startup_save_and_requests_a_random_hint() -> None:
    loaded = _with_main_input_first_launch_completed(AppSettingsVNext())
    owner = _RecordingPersistOwner(loaded)
    presentation = Mock()
    adapter = _startup_adapter(owner, loaded, presentation=presentation)

    state = await adapter.prepare_startup_settings()
    adapter.apply_startup_presentation(state)

    assert state.main_input_first_launch is False
    assert owner.saved == []
    presentation.set_dashboard_launch_input_hint.assert_called_once_with(first_launch=False)


@pytest.mark.asyncio
async def test_unsaved_first_launch_mark_is_retried_on_the_next_launch() -> None:
    loaded = AppSettingsVNext()
    owner = _RecordingPersistOwner(loaded, save_succeeds=False)
    adapter = _startup_adapter(owner, loaded)

    state = await adapter.prepare_startup_settings()

    assert state.main_input_first_launch is True
    assert state.settings is loaded
    assert owner.canonical.state.main_input_hint.first_launch_completed is False


def test_gui_startup_uses_logging_owner_history_binding_without_rebinding_its_facade(
    tmp_path, monkeypatch
) -> None:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "level", root.level)
    log_file = tmp_path / "startup-history.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    history = ConversationHistory()
    presentation = FletUiPresentationAdapter(SimpleNamespace(conversation_history=history))
    runtime_logging = compose_application_runtime_logging(
        presentation=presentation,
        sinks=RuntimeLoggingSinks(logging.NullHandler(), file_handler, log_file),
    )
    settings = AppSettingsVNext()
    adapter = _startup_adapter(
        _RecordingPersistOwner(settings), settings, presentation=presentation
    )
    adapter.runtime_logging = runtime_logging
    previous_locale = get_locale()
    try:
        adapter.apply_startup_presentation(
            ApplicationStartupState(settings=settings, main_input_first_launch=False)
        )
        runtime_logging.record_conversation_observation(
            utterance_id="synthetic-startup-turn",
            speaker_channel="self",
            transcript_text="synthetic startup speech",
            translation_text=None,
            source_language="en",
            target_language=None,
        )
        assert len(history.conversation_records) == 1
        assert history.conversation_records[0].source_text == "synthetic startup speech"
        assert "synthetic startup speech" in log_file.read_text(encoding="utf-8")
    finally:
        runtime_logging.service.close()
        file_handler.close()
        set_locale(previous_locale)
