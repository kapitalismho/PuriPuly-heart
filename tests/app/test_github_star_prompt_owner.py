from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        ui=SimpleNamespace(
            github_star_prompt_clicked=False,
            github_star_prompt_last_shown_at=None,
            github_star_prompt_show_count=0,
            github_star_prompt_translation_success_observed=False,
            github_star_prompt_eligible_launch_count=0,
        )
    )


def _owner(
    current: list[object | None],
    *,
    eligible=lambda: True,
    user_owned=lambda _settings: True,
    persist=None,
) -> tuple[GithubStarPromptOwner, list[str]]:
    failures: list[str] = []

    async def default_persist(_base: object, _committed: object) -> bool:
        return True

    owner = GithubStarPromptOwner(
        settings_provider=lambda: current[0],
        persist_settings_state=persist or default_persist,
        is_eligible=eligible,
        has_user_owned_cloud_connection=user_owned,
        log_save_failure=lambda context, error: failures.append(
            f"{context}:{type(error).__name__}"
        ),
        runtime_diagnostics_sink=lambda _event, _metadata: None,
    )
    return owner, failures


@pytest.mark.asyncio
async def test_owner_persists_open_click_and_translation_observation_state() -> None:
    settings = _settings()
    current: list[object | None] = [settings]
    snapshots: list[tuple[bool, int, bool]] = []

    async def persist(_base: object, committed: object) -> bool:
        ui = committed.ui
        snapshots.append(
            (
                ui.github_star_prompt_clicked,
                ui.github_star_prompt_show_count,
                ui.github_star_prompt_translation_success_observed,
            )
        )
        return True

    owner, _ = _owner(current, persist=persist)

    assert await owner.persist_opened(opened_at=datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert await owner.persist_clicked()
    settings.ui.github_star_prompt_clicked = False
    assert await owner.persist_translation_success_observed()

    assert snapshots == [
        (False, 1, False),
        (True, 1, False),
        (False, 1, True),
    ]
    assert settings.ui.github_star_prompt_last_shown_at == "2026-07-01T00:00:00Z"


@pytest.mark.asyncio
async def test_owner_rolls_back_failed_mutation_and_logs_failure() -> None:
    settings = _settings()
    current: list[object | None] = [settings]

    async def fail(_base: object, _committed: object) -> bool:
        raise OSError("write")

    owner, failures = _owner(current, persist=fail)

    assert await owner.persist_translation_success_observed() is False

    assert settings.ui.github_star_prompt_translation_success_observed is False
    assert failures == ["translation success observation:OSError"]


@pytest.mark.asyncio
async def test_owner_retargets_concurrent_mutation_to_replaced_settings() -> None:
    first = _settings()
    replacement = _settings()
    current: list[object | None] = [first]
    release = asyncio.Event()
    calls = 0

    async def persist(_base: object, _committed: object) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
        return True

    owner, _ = _owner(current, persist=persist)
    task = asyncio.create_task(owner.persist_translation_success_observed())
    await asyncio.sleep(0)
    current[0] = replacement
    release.set()

    assert await task is True
    assert replacement.ui.github_star_prompt_translation_success_observed is True
    assert calls == 2


@pytest.mark.asyncio
async def test_owner_preserves_durable_prompt_state_before_settings_replace() -> None:
    current_settings = _settings()
    current_settings.ui.github_star_prompt_clicked = True
    current_settings.ui.github_star_prompt_show_count = 2
    current_settings.ui.github_star_prompt_last_shown_at = "2026-07-02T00:00:00Z"
    replacement = _settings()
    replacement.ui.github_star_prompt_show_count = 1
    replacement.ui.github_star_prompt_last_shown_at = "2026-07-01T00:00:00Z"
    owner, _ = _owner([current_settings])

    await owner.preserve_before_settings_replace(replacement)

    assert replacement.ui.github_star_prompt_clicked is True
    assert replacement.ui.github_star_prompt_show_count == 2
    assert replacement.ui.github_star_prompt_last_shown_at == "2026-07-02T00:00:00Z"


def test_owner_enforces_launch_gate_recency_and_eligibility() -> None:
    settings = _settings()
    current: list[object | None] = [settings]
    owner, _ = _owner(current)

    assert owner.should_show(now=datetime(2026, 7, 20, tzinfo=timezone.utc)) is False
    settings.ui.github_star_prompt_eligible_launch_count = 3
    assert owner.should_show(now=datetime(2026, 7, 20, tzinfo=timezone.utc)) is True
    settings.ui.github_star_prompt_last_shown_at = "2026-07-10T00:00:00Z"
    assert owner.should_show(now=datetime(2026, 7, 20, tzinfo=timezone.utc)) is False
    assert owner.should_show(now=datetime(2026, 7, 25, tzinfo=timezone.utc)) is True
