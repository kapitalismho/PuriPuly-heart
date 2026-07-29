from __future__ import annotations

import copy
from pathlib import Path

import pytest

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.github_star_prompt_settings import (
    compose_github_star_prompt_owner,
)
from puripuly_heart.config.settings import AppSettings, TranslationConnection
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TransactionResult,
)


class RecordingPersistence(SettingsVNextCanonicalPersistenceAdapter):
    def __init__(self, *, failure_text: str | None = None) -> None:
        self.saved: list[AppSettingsVNext] = []
        self.failure_text = failure_text

    def persist(self, path: Path, settings: AppSettingsVNext) -> None:
        _ = path
        if self.failure_text is not None:
            raise RuntimeError(self.failure_text)
        self.saved.append(copy.deepcopy(settings))


def _prompt_owner(
    settings: AppSettings,
    *,
    remaining_percent: int | None = None,
    persistence: RecordingPersistence | None = None,
    diagnostics: list[tuple[str, dict[str, object]]] | None = None,
) -> tuple[
    SettingsOwner,
    GithubStarPromptOwner,
    list[TransactionResult],
    RecordingPersistence,
]:
    resolved_persistence = persistence or RecordingPersistence()
    resolved_diagnostics = diagnostics if diagnostics is not None else []
    settings_owner = SettingsOwner(
        path=Path("settings.json"),
        persistence=resolved_persistence,
        canonical=AppSettingsVNext(),
        current=settings,
        authoritative=True,
        projection_snapshot=copy.deepcopy(settings),
    )
    results: list[TransactionResult] = []
    prompt = compose_github_star_prompt_owner(
        settings=settings_owner,
        managed_remaining_percent=lambda: remaining_percent,
        transaction_result_sink=results.append,
        save_failure_sink=lambda _context, _error: None,
        runtime_diagnostics_sink=lambda event, metadata: resolved_diagnostics.append(
            (event, dict(metadata))
        ),
        mutation_service_provider=lambda: None,
    )
    return settings_owner, prompt, results, resolved_persistence


@pytest.mark.asyncio
async def test_prompt_settings_owner_persists_user_owned_click_without_controller() -> None:
    settings = AppSettings()
    settings.translation.connection = TranslationConnection.OPENROUTER
    settings.ui.github_star_prompt_translation_success_observed = True
    settings_owner, prompt, results, persistence = _prompt_owner(settings)

    assert prompt.is_eligible() is True
    assert await prompt.persist_clicked() is True

    assert settings.ui.github_star_prompt_clicked is True
    assert settings_owner.mutation_depth == 0
    assert results[-1].status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert persistence.saved[-1].state.github_star_prompt.clicked is True


@pytest.mark.asyncio
async def test_prompt_settings_owner_reports_safe_order24_persistence_failure() -> None:
    raw_failure = "secret-value-must-not-appear"
    settings = AppSettings()
    settings.translation.connection = TranslationConnection.OPENROUTER
    settings.ui.github_star_prompt_translation_success_observed = True
    diagnostics: list[tuple[str, dict[str, object]]] = []
    settings_owner, prompt, results, _persistence = _prompt_owner(
        settings,
        persistence=RecordingPersistence(failure_text=raw_failure),
        diagnostics=diagnostics,
    )

    assert await prompt.persist_clicked() is False

    assert settings.ui.github_star_prompt_clicked is False
    assert settings_owner.mutation_depth == 0
    assert settings_owner.rollback_pending is False
    assert results[-1].status == TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert diagnostics == [
        (
            "github_star_prompt_settings_persistence_failed",
            {
                "component": "settings_repository",
                "operation": "save",
                "code": "settings_save_failed",
                "surface": "ui_prompt_clipboard_state",
            },
        )
    ]
    assert raw_failure not in repr(diagnostics)


@pytest.mark.parametrize(
    ("remaining_percent", "expected"),
    (
        (None, False),
        (61, False),
        (60, True),
        (0, True),
    ),
)
def test_prompt_settings_owner_preserves_managed_usage_threshold(
    remaining_percent: int | None,
    expected: bool,
) -> None:
    settings = AppSettings()
    settings.translation.connection = TranslationConnection.MANAGED
    _settings_owner, prompt, _results, _persistence = _prompt_owner(
        settings,
        remaining_percent=remaining_percent,
    )

    assert prompt.is_eligible() is expected
