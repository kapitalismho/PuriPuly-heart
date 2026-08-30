from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def _settings() -> AppSettingsVNext:
    return AppSettingsVNext()


def _prompt(settings: AppSettingsVNext):
    return settings.state.github_star_prompt


def _with_prompt(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        state=replace(
            settings.state,
            github_star_prompt=replace(settings.state.github_star_prompt, **changes),
        ),
    )


def _owner(
    current: list[object | None],
    *,
    eligible=lambda: True,
    user_owned=lambda _settings: True,
    persist=None,
) -> tuple[GithubStarPromptOwner, list[str]]:
    failures: list[str] = []

    async def default_persist(_base: object, committed: object) -> bool:
        current[0] = committed
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
        current[0] = committed
        prompt = _prompt(committed)
        snapshots.append(
            (
                prompt.clicked,
                prompt.show_count,
                prompt.translation_success_observed,
            )
        )
        return True

    owner, _ = _owner(current, persist=persist)

    assert await owner.persist_opened(opened_at=datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert await owner.persist_clicked()
    current[0] = _with_prompt(current[0], clicked=False)
    assert await owner.persist_translation_success_observed()

    assert snapshots == [
        (False, 1, False),
        (True, 1, False),
        (False, 1, True),
    ]
    assert _prompt(current[0]).last_shown_at == "2026-07-01T00:00:00Z"


@pytest.mark.asyncio
async def test_owner_rolls_back_failed_mutation_and_logs_failure() -> None:
    settings = _settings()
    current: list[object | None] = [settings]

    async def fail(_base: object, _committed: object) -> bool:
        raise OSError("write")

    owner, failures = _owner(current, persist=fail)

    assert await owner.persist_translation_success_observed() is False

    assert _prompt(current[0]).translation_success_observed is False
    assert failures == ["translation success observation:OSError"]


@pytest.mark.asyncio
async def test_owner_rolls_back_open_state_when_should_open_fails_after_persist() -> None:
    settings = _settings()
    current: list[object | None] = [settings]
    snapshots: list[tuple[int, str | None]] = []

    async def persist(_base: object, committed: object) -> bool:
        current[0] = committed
        prompt = _prompt(committed)
        snapshots.append((prompt.show_count, prompt.last_shown_at))
        return True

    owner, _ = _owner(current, persist=persist)
    should_open_values = [True, False]

    def should_open() -> bool:
        return bool(should_open_values.pop(0)) if should_open_values else False

    assert (
        await owner.persist_opened(
            opened_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
            should_open=should_open,
        )
        is False
    )

    assert snapshots == [(1, "2026-07-01T00:00:00Z"), (0, None)]
    assert _prompt(current[0]).show_count == 0
    assert _prompt(current[0]).last_shown_at is None


@pytest.mark.asyncio
async def test_owner_persists_mutation_against_settings_snapshot() -> None:
    first = _settings()
    replacement = _settings()
    current: list[object | None] = [first]
    committed_settings: list[object] = []
    release = asyncio.Event()

    async def persist(_base: object, committed: object) -> bool:
        await release.wait()
        committed_settings.append(committed)
        return True

    owner, _ = _owner(current, persist=persist)
    task = asyncio.create_task(owner.persist_translation_success_observed())
    await asyncio.sleep(0)
    current[0] = replacement
    release.set()

    assert await task is True
    assert len(committed_settings) == 1
    assert _prompt(committed_settings[0]).translation_success_observed is True
    assert _prompt(replacement).translation_success_observed is False


@pytest.mark.asyncio
async def test_owner_preserves_durable_prompt_state_before_settings_replace() -> None:
    current_settings = _with_prompt(
        _settings(),
        clicked=True,
        show_count=2,
        last_shown_at="2026-07-02T00:00:00Z",
    )
    replacement = _with_prompt(
        _settings(),
        show_count=1,
        last_shown_at="2026-07-01T00:00:00Z",
    )
    owner, _ = _owner([current_settings])

    merged = await owner.preserve_before_settings_replace(replacement)

    assert _prompt(merged).clicked is True
    assert _prompt(merged).show_count == 2
    assert _prompt(merged).last_shown_at == "2026-07-02T00:00:00Z"


def test_owner_enforces_launch_gate_recency_and_eligibility() -> None:
    settings = _settings()
    current: list[object | None] = [settings]
    owner, _ = _owner(current)

    assert owner.should_show(now=datetime(2026, 7, 20, tzinfo=timezone.utc)) is False
    current[0] = _with_prompt(current[0], eligible_launch_count=3)
    assert owner.should_show(now=datetime(2026, 7, 20, tzinfo=timezone.utc)) is True
    current[0] = _with_prompt(current[0], last_shown_at="2026-07-10T00:00:00Z")
    assert owner.should_show(now=datetime(2026, 7, 20, tzinfo=timezone.utc)) is False
    assert owner.should_show(now=datetime(2026, 7, 25, tzinfo=timezone.utc)) is True
