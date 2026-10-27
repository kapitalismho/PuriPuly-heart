from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.github_star_prompt_settings import (
    compose_github_star_prompt_owner,
)
from puripuly_heart.app.services.managed.managed_usage import ManagedUsageOwner
from puripuly_heart.config.llm_profiles import OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.settings_vnext.serialization import to_dict as canonical_to_dict
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.openrouter import OpenRouterKeyMetadata
from puripuly_heart.ui.event_bridge import (
    AppDashboardEventDestination,
    AppHistoryEventDestination,
    UIEventBridge,
)


@pytest.fixture(autouse=True)
def _isolated_prompt_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


class PromptBackend:
    def __init__(self, settings: AppSettingsVNext) -> None:
        path = Path("settings.json").resolve()
        SettingsVNextCanonicalPersistenceAdapter().persist(path, settings)
        self.settings_owner = compose_settings_owner(path)
        self.settings_owner.start()
        self.settings_owner.remember_projection(self.settings_owner.require_canonical())
        self.usage = SimpleNamespace(usage_metadata=None)
        self.owner = compose_github_star_prompt_owner(
            settings=self.settings_owner,
            managed_remaining_percent=lambda: ManagedUsageOwner.remaining_percent_for(
                self.usage.usage_metadata
            ),
            transaction_result_sink=lambda _result: None,
            save_failure_sink=lambda _context, _exc: None,
            runtime_diagnostics_sink=lambda _event, _metadata: None,
            mutation_service_provider=lambda: None,
        )

    @property
    def settings(self) -> AppSettingsVNext:
        return self.settings_owner.canonical

    def _get_managed_usage_owner(self) -> object:
        return self.usage

    def _get_github_star_prompt_owner(self):
        return self.owner

    def is_github_star_prompt_eligible(self) -> bool:
        return self.owner.is_eligible()

    def schedule_github_star_prompt_translation_success_observed(self) -> bool:
        return self.owner.schedule_translation_success_observed()

    async def persist_github_star_prompt_translation_success_observed(self) -> bool:
        return await self.owner.persist_translation_success_observed()


def _prompt_backend_for(settings: AppSettingsVNext) -> PromptBackend:
    return PromptBackend(settings)


def _patch_settings_save(
    monkeypatch: pytest.MonkeyPatch, controller: PromptBackend, callback
) -> None:
    persistence = controller.settings_owner.persistence
    original = persistence.persist

    def persist(path: Path, updated: AppSettingsVNext) -> None:
        callback(path, updated)
        original(path, updated)

    monkeypatch.setattr(persistence, "persist", persist)


def _persisted_settings(controller: PromptBackend) -> AppSettingsVNext:
    return (
        SettingsVNextCanonicalPersistenceAdapter()
        .load_active(controller.settings_owner.path)
        .canonical_settings
    )


def _settings_for_connection(connection: str) -> AppSettingsVNext:
    settings = AppSettingsVNext()
    model = settings.intent.translation.model
    if connection == "managed_china":
        model = "deepseek_v4_flash_41"
    elif connection == "official_byok":
        model = "deepseek_v4_flash_41"
    elif connection == "ollama":
        model = "local_llm"
    history = dict(settings.intent.translation.connection_history)
    history[model] = connection
    return materialize_canonical_translation_settings(
        replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    settings.intent.translation,
                    model=model,
                    connection=connection,
                    connection_history=history,
                    openrouter_model=(
                        OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41
                        if connection == "official_byok"
                        else settings.intent.translation.openrouter_model
                    ),
                ),
            ),
        )
    )


def _with_star_success(settings: AppSettingsVNext) -> AppSettingsVNext:
    return replace(
        settings,
        state=replace(
            settings.state,
            github_star_prompt=replace(
                settings.state.github_star_prompt,
                translation_success_observed=True,
            ),
        ),
    )


def test_official_byok_fixture_uses_supported_model_provider_combo() -> None:
    settings = _settings_for_connection("official_byok")

    assert settings.intent.translation.model == "deepseek_v4_flash_41"
    assert settings.intent.translation.connection == "official_byok"


def test_github_star_prompt_is_eligible_for_managed_remaining_percent_at_threshold() -> None:
    controller = _prompt_backend_for(_settings_for_connection("managed"))
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
    controller = _prompt_backend_for(_settings_for_connection("managed"))
    controller._get_managed_usage_owner().usage_metadata = metadata

    assert controller.is_github_star_prompt_eligible() is False


@pytest.mark.parametrize(
    "connection",
    ["openrouter", "official_byok"],
)
def test_github_star_prompt_is_eligible_for_recorded_user_owned_cloud_success(
    connection: str,
) -> None:
    settings = _with_star_success(_settings_for_connection(connection))
    controller = _prompt_backend_for(settings)

    assert controller.is_github_star_prompt_eligible() is True


def test_github_star_prompt_skips_user_owned_cloud_without_recorded_success() -> None:
    controller = _prompt_backend_for(_settings_for_connection("openrouter"))

    assert controller.is_github_star_prompt_eligible() is False


def test_github_star_prompt_excludes_local_ollama_from_user_owned_cloud_path() -> None:
    settings = _with_star_success(_settings_for_connection("ollama"))
    controller = _prompt_backend_for(settings)

    assert controller.is_github_star_prompt_eligible() is False


@pytest.mark.parametrize(
    "connection",
    ["managed", "managed_china"],
)
def test_github_star_prompt_excludes_managed_connections_from_user_owned_cloud_path(
    connection: str,
) -> None:
    settings = _with_star_success(_settings_for_connection(connection))
    controller = _prompt_backend_for(settings)

    assert controller.is_github_star_prompt_eligible() is False


def test_github_star_prompt_skips_ineligible_new_user_state() -> None:
    controller = _prompt_backend_for(AppSettingsVNext())

    assert controller.is_github_star_prompt_eligible() is False


@pytest.mark.parametrize(
    "connection",
    ["openrouter", "official_byok"],
)
def test_user_owned_cloud_translation_success_observation_persists_through_settings(
    monkeypatch: pytest.MonkeyPatch,
    connection: str,
) -> None:
    settings = _settings_for_connection(connection)
    controller = _prompt_backend_for(settings)
    saved_payloads: list[dict[str, object]] = []

    def fake_save_settings(_path: Path, updated: AppSettingsVNext) -> None:
        saved_payloads.append(canonical_to_dict(updated))

    _patch_settings_save(monkeypatch, controller, fake_save_settings)

    assert controller._get_github_star_prompt_owner().record_translation_success_observed() is True

    assert controller.settings.state.github_star_prompt.translation_success_observed is True
    assert saved_payloads
    restored = _persisted_settings(controller)
    assert restored.state.github_star_prompt.translation_success_observed is True
    assert saved_payloads[-1]["state"]["github_star_prompt"]["translation_success_observed"] is True


@pytest.mark.parametrize(
    "connection",
    [
        "managed",
        "managed_china",
        "ollama",
    ],
)
def test_translation_success_observation_ignores_non_user_owned_cloud_connections(
    monkeypatch: pytest.MonkeyPatch,
    connection: str,
) -> None:
    settings = _settings_for_connection(connection)
    controller = _prompt_backend_for(settings)
    save_calls: list[str] = []

    _patch_settings_save(
        monkeypatch,
        controller,
        lambda _path, _updated: save_calls.append("save"),
    )

    assert controller._get_github_star_prompt_owner().record_translation_success_observed() is False

    assert controller.settings.state.github_star_prompt.translation_success_observed is False
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
    settings = _settings_for_connection("openrouter")
    controller = _prompt_backend_for(settings)
    saved_payloads: list[dict[str, object]] = []

    class Dashboard:
        def set_display_translation_text(self, *_args: object, **_kwargs: object) -> None:
            return None

    def fake_save_settings(_path: Path, updated: AppSettingsVNext) -> None:
        saved_payloads.append(canonical_to_dict(updated))

    _patch_settings_save(monkeypatch, controller, fake_save_settings)

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

    await asyncio.wait_for(controller.owner.drain_translation_success_observation(), timeout=1)

    assert controller.settings.state.github_star_prompt.translation_success_observed is True
    assert saved_payloads
    assert _persisted_settings(controller).state.github_star_prompt.translation_success_observed
