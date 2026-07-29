from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.app.services.provider_runtime_apply import LlmProviderRebuildOwner
from puripuly_heart.app.services.self_capture_application import (
    SelfCaptureApplicationOwner,
)
from puripuly_heart.app.wiring_provider_runtime import ProviderRuntimeEffects
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    TranslationConnection,
    TranslationModel,
    from_dict,
    materialize_translation_settings,
    to_dict,
)
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.openrouter import OpenRouterKeyMetadata
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.event_bridge import (
    AppConversationEventDestination,
    AppDashboardEventDestination,
    AppHistoryEventDestination,
    UIEventBridge,
)
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from tests.helpers.fakes import install_test_runtime_composition


def _controller_for(settings: AppSettings) -> GuiController:
    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    return install_test_runtime_composition(controller)


def _patch_settings_save(monkeypatch: pytest.MonkeyPatch, callback) -> None:
    def persist(owner) -> None:
        callback(owner.path, owner.compatibility_projection())

    monkeypatch.setattr(controller_module.SettingsOwner, "persist", persist)


def _settings_for_connection(connection: TranslationConnection) -> AppSettings:
    settings = AppSettings()
    settings.translation.connection = connection
    if connection == TranslationConnection.MANAGED_CHINA:
        settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    elif connection == TranslationConnection.OFFICIAL_BYOK:
        settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    elif connection == TranslationConnection.OLLAMA:
        settings.translation.model = TranslationModel.LOCAL_LLM
    settings.translation.connection_history[settings.translation.model.value] = connection
    materialize_translation_settings(settings)
    return settings


async def _wait_until(predicate, *, attempts: int = 20) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not met in time")


async def _async_noop(*_args: object, **_kwargs: object) -> None:
    return None


def _patch_stop_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GuiController, "_close_local_asr_provisioning", _async_noop)
    monkeypatch.setattr(GuiController, "set_stt_enabled", _async_noop)
    monkeypatch.setattr(GuiController, "_configure_vrc_mic_receiver", _async_noop)
    monkeypatch.setattr(OverlayApplicationOwner, "shutdown", _async_noop)
    monkeypatch.setattr(
        GuiController,
        "_close_managed_openrouter_release_service",
        _async_noop,
    )


def test_official_byok_fixture_uses_supported_model_provider_combo() -> None:
    settings = _settings_for_connection(TranslationConnection.OFFICIAL_BYOK)

    settings.validate()

    assert settings.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert settings.provider.llm == LLMProviderName.DEEPSEEK


def test_github_star_prompt_is_eligible_for_managed_remaining_percent_at_threshold() -> None:
    controller = _controller_for(_settings_for_connection(TranslationConnection.MANAGED))
    controller._get_managed_usage_owner().usage_metadata = OpenRouterKeyMetadata(
        limit_usd=100.0,
        remaining_usd=60.0,
        usage_usd=40.0,
    )

    assert controller.is_github_star_prompt_eligible() is True


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        OpenRouterKeyMetadata(limit_usd=None, remaining_usd=60.0, usage_usd=40.0),
        OpenRouterKeyMetadata(limit_usd=100.0, remaining_usd=None, usage_usd=40.0),
        OpenRouterKeyMetadata(limit_usd=0.0, remaining_usd=0.0, usage_usd=0.0),
    ],
)
def test_github_star_prompt_skips_managed_when_usage_metadata_is_unavailable(
    metadata: OpenRouterKeyMetadata | None,
) -> None:
    controller = _controller_for(_settings_for_connection(TranslationConnection.MANAGED))
    controller._get_managed_usage_owner().usage_metadata = metadata

    assert controller.is_github_star_prompt_eligible() is False


@pytest.mark.parametrize(
    "connection",
    [TranslationConnection.OPENROUTER, TranslationConnection.OFFICIAL_BYOK],
)
def test_github_star_prompt_is_eligible_for_recorded_user_owned_cloud_success(
    connection: TranslationConnection,
) -> None:
    settings = _settings_for_connection(connection)
    settings.ui.github_star_prompt_translation_success_observed = True
    controller = _controller_for(settings)

    assert controller.is_github_star_prompt_eligible() is True


def test_github_star_prompt_skips_user_owned_cloud_without_recorded_success() -> None:
    controller = _controller_for(_settings_for_connection(TranslationConnection.OPENROUTER))

    assert controller.is_github_star_prompt_eligible() is False


def test_github_star_prompt_excludes_local_ollama_from_user_owned_cloud_path() -> None:
    settings = _settings_for_connection(TranslationConnection.OLLAMA)
    settings.ui.github_star_prompt_translation_success_observed = True
    controller = _controller_for(settings)

    assert controller.is_github_star_prompt_eligible() is False


@pytest.mark.parametrize(
    "connection",
    [TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA],
)
def test_github_star_prompt_excludes_managed_connections_from_user_owned_cloud_path(
    connection: TranslationConnection,
) -> None:
    settings = _settings_for_connection(connection)
    settings.ui.github_star_prompt_translation_success_observed = True
    controller = _controller_for(settings)

    assert controller.is_github_star_prompt_eligible() is False


def test_github_star_prompt_skips_ineligible_new_user_state() -> None:
    controller = _controller_for(AppSettings())

    assert controller.is_github_star_prompt_eligible() is False


@pytest.mark.parametrize(
    "connection",
    [TranslationConnection.OPENROUTER, TranslationConnection.OFFICIAL_BYOK],
)
def test_user_owned_cloud_translation_success_observation_persists_through_settings(
    monkeypatch: pytest.MonkeyPatch,
    connection: TranslationConnection,
) -> None:
    settings = _settings_for_connection(connection)
    controller = _controller_for(settings)
    saved_payloads: list[dict[str, object]] = []

    def fake_save_settings(_path: Path, updated: AppSettings) -> None:
        saved_payloads.append(to_dict(updated))

    _patch_settings_save(monkeypatch, fake_save_settings)

    assert controller._get_github_star_prompt_owner().record_translation_success_observed() is True

    assert settings.ui.github_star_prompt_translation_success_observed is True
    assert saved_payloads
    restored = from_dict(saved_payloads[-1])
    assert restored.ui.github_star_prompt_translation_success_observed is True


@pytest.mark.parametrize(
    "connection",
    [
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OLLAMA,
    ],
)
def test_translation_success_observation_ignores_non_user_owned_cloud_connections(
    monkeypatch: pytest.MonkeyPatch,
    connection: TranslationConnection,
) -> None:
    settings = _settings_for_connection(connection)
    controller = _controller_for(settings)
    save_calls: list[str] = []

    _patch_settings_save(
        monkeypatch,
        lambda _path, _updated: save_calls.append("save"),
    )

    assert controller._get_github_star_prompt_owner().record_translation_success_observed() is False

    assert settings.ui.github_star_prompt_translation_success_observed is False
    assert save_calls == []


@pytest.mark.asyncio
async def test_event_bridge_schedules_github_star_observation_after_translation_ui_updates() -> (
    None
):
    calls: list[str] = []

    class Dashboard:
        def set_display_translation_text(self, *_args: object, **_kwargs: object) -> None:
            calls.append("dashboard")

    class Controller:
        settings = SimpleNamespace(
            languages=SimpleNamespace(source_language="ko", target_language="en")
        )
        hub = SimpleNamespace(translation_enabled=False)

        def record_github_star_prompt_translation_success_observed(self) -> bool:
            raise AssertionError("event bridge must not synchronously persist prompt state")

        def schedule_github_star_prompt_translation_success_observed(self) -> bool:
            calls.append("schedule")
            return True

    app = SimpleNamespace(
        controller=Controller(),
        view_dashboard=Dashboard(),
        view_logs=None,
        add_history_entry=lambda *_args, **_kwargs: calls.append("history"),
    )
    bridge = UIEventBridge(
        event_queue=object(),
        dashboard_destination=AppDashboardEventDestination(app.view_dashboard),
        history_destination=AppHistoryEventDestination(app.add_history_entry),
        conversation_destination=AppConversationEventDestination(None),
        on_github_star_translation_success=(
            app.controller.schedule_github_star_prompt_translation_success_observed
        ),
    )

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(utterance_id=uuid4(), text="translated"),
            source="Mic",
        )
    )

    assert calls == ["dashboard", "history", "schedule"]


@pytest.mark.asyncio
async def test_event_bridge_records_successful_translation_for_user_owned_cloud_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_for_connection(TranslationConnection.OPENROUTER)
    controller = _controller_for(settings)
    saved_payloads: list[dict[str, object]] = []

    class Dashboard:
        def set_display_translation_text(self, *_args: object, **_kwargs: object) -> None:
            return None

    def fake_save_settings(_path: Path, updated: AppSettings) -> None:
        saved_payloads.append(to_dict(updated))

    _patch_settings_save(monkeypatch, fake_save_settings)

    app = SimpleNamespace(
        controller=controller,
        view_dashboard=Dashboard(),
        view_logs=None,
        add_history_entry=lambda *_args, **_kwargs: None,
    )
    bridge = UIEventBridge(
        event_queue=object(),
        dashboard_destination=AppDashboardEventDestination(app.view_dashboard),
        history_destination=AppHistoryEventDestination(app.add_history_entry),
        conversation_destination=AppConversationEventDestination(None),
        on_github_star_translation_success=(
            controller.schedule_github_star_prompt_translation_success_observed
        ),
    )

    await bridge._handle_event(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            payload=Translation(utterance_id=uuid4(), text="translated"),
            source="Mic",
        )
    )

    await _wait_until(lambda: bool(saved_payloads))

    assert settings.ui.github_star_prompt_translation_success_observed is True
    assert saved_payloads


@pytest.mark.asyncio
async def test_apply_providers_drains_pending_observation_before_settings_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_settings = _settings_for_connection(TranslationConnection.OPENROUTER)
    controller = _controller_for(initial_settings)
    replacement_settings = _settings_for_connection(TranslationConnection.OLLAMA)
    saved_payloads: list[dict[str, object]] = []
    first_to_thread_started = asyncio.Event()
    release_first_to_thread = asyncio.Event()

    def fake_save_settings(_path: Path, updated: AppSettings) -> None:
        saved_payloads.append(to_dict(updated))

    async def delayed_to_thread(func, /, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        first_to_thread_started.set()
        await release_first_to_thread.wait()
        return func(*args, **kwargs)

    _patch_settings_save(monkeypatch, fake_save_settings)
    monkeypatch.setattr(controller_module.asyncio, "to_thread", delayed_to_thread)
    monkeypatch.setattr(LlmProviderRebuildOwner, "rebuild", _async_noop)
    monkeypatch.setattr(PeerApplicationOwner, "refresh_runtime", _async_noop)
    monkeypatch.setattr(SelfCaptureApplicationOwner, "replace_provider", _async_noop)
    monkeypatch.setattr(ProviderRuntimeEffects, "rebuild_self_stt", _async_noop)

    runtime = controller._get_github_star_prompt_owner().get_runtime()
    persist_task = runtime.start_translation_success_observation(
        controller.persist_github_star_prompt_translation_success_observed()
    )
    await asyncio.wait_for(first_to_thread_started.wait(), timeout=1.0)

    apply_task = asyncio.create_task(controller.apply_providers(replacement_settings))
    await asyncio.sleep(0)

    assert not apply_task.done()

    release_first_to_thread.set()
    try:
        await asyncio.wait_for(apply_task, timeout=1.0)
        assert await asyncio.wait_for(persist_task, timeout=1.0) is True
        assert controller.settings is not None
        assert controller.settings.ui.github_star_prompt_translation_success_observed is True
        assert controller.is_github_star_prompt_eligible() is False
        assert saved_payloads[-1]["ui"]["github_star_prompt_translation_success_observed"] is True
    finally:
        if not apply_task.done():
            apply_task.cancel()
            await asyncio.gather(apply_task, return_exceptions=True)
        if not persist_task.done():
            persist_task.cancel()
            await asyncio.gather(persist_task, return_exceptions=True)
