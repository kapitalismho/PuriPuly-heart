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
from puripuly_heart.config.llm_profiles import OPENROUTER_MODEL_DEEPSEEK_V4_FLASH_41
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.settings_vnext.serialization import to_dict as canonical_to_dict
from puripuly_heart.config.translation_values import TranslationConnection
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Translation
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
        self.owner = compose_github_star_prompt_owner(
            settings=self.settings_owner,
            transaction_result_sink=lambda _result: None,
            save_failure_sink=lambda _context, _exc: None,
            runtime_diagnostics_sink=lambda _event, _metadata: None,
            mutation_service_provider=lambda: None,
        )

    @property
    def settings(self) -> AppSettingsVNext:
        return self.settings_owner.canonical

    def is_github_star_prompt_eligible(self) -> bool:
        return self.owner.is_eligible()

    def schedule_github_star_prompt_translation_success_observed(self) -> bool:
        return self.owner.schedule_translation_success_observed()



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
    model = {
        "public_api": "index_translate_35b",
        "managed": "gemma4_26b_31b",
        "managed_china": "deepseek_v4_flash_41",
        "chatgpt": "gpt_6_luna",
        "openrouter": "gemma4_26b_31b",
        "official_byok": "deepseek_v4_flash_41",
        "ollama": "local_llm",
        "cpu": "managed_gemma",
        "gpu": "managed_gemma",
        "custom_http": "custom_http",
    }[connection]
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


@pytest.mark.parametrize("connection", [connection.value for connection in TranslationConnection])
@pytest.mark.parametrize("success_observed", [False, True])
def test_github_star_prompt_eligibility_depends_only_on_translation_success(
    connection: str,
    success_observed: bool,
) -> None:
    settings = _settings_for_connection(connection)
    if success_observed:
        settings = _with_star_success(settings)
    controller = _prompt_backend_for(settings)

    assert controller.is_github_star_prompt_eligible() is success_observed


@pytest.mark.parametrize("connection", [connection.value for connection in TranslationConnection])
def test_translation_success_observation_persists_through_settings(
    connection: str,
) -> None:
    controller = _prompt_backend_for(_settings_for_connection(connection))

    assert controller.owner.record_translation_success_observed() is True

    restarted = _prompt_backend_for(_persisted_settings(controller))
    assert restarted.is_github_star_prompt_eligible() is True
    assert restarted.settings.state.github_star_prompt.translation_success_observed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("connection", ["public_api", "managed", "ollama"])
@pytest.mark.parametrize(("text", "expected"), [("translated", True), (" \n\t", False)])
async def test_event_bridge_records_only_nonempty_successful_translation(
    monkeypatch: pytest.MonkeyPatch,
    connection: str,
    text: str,
    expected: bool,
) -> None:
    settings = _settings_for_connection(connection)
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
            payload=Translation(utterance_id=uuid4(), text=text),
            source="Mic",
        )
    )

    await asyncio.wait_for(controller.owner.drain_translation_success_observation(), timeout=1)

    assert controller.settings.state.github_star_prompt.translation_success_observed is expected
    assert bool(saved_payloads) is expected
    assert (
        _persisted_settings(controller).state.github_star_prompt.translation_success_observed
        is expected
    )
