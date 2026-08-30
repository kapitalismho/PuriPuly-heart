from __future__ import annotations

from dataclasses import replace

from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, GithubStarPromptState

PROMPT_STATE_DEFAULTS = {
    "clicked": False,
    "last_shown_at": None,
    "show_count": 0,
    "translation_success_observed": False,
    "eligible_launch_count": 0,
}


def _prompt(settings: AppSettingsVNext) -> GithubStarPromptState:
    return settings.state.github_star_prompt


def _with_prompt(prompt: GithubStarPromptState) -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(settings, state=replace(settings.state, github_star_prompt=prompt))


def test_github_star_prompt_state_defaults_round_trip() -> None:
    settings = AppSettingsVNext()
    persisted = serialization.to_dict(settings)
    loaded = serialization.from_dict(persisted)

    assert _prompt(settings) == GithubStarPromptState()
    assert persisted["state"]["github_star_prompt"] == PROMPT_STATE_DEFAULTS
    assert _prompt(loaded) == GithubStarPromptState()


def test_github_star_prompt_state_round_trips_through_canonical_settings() -> None:
    persisted_timestamp = "2026-05-24T12:34:56Z"
    settings = _with_prompt(
        GithubStarPromptState(
            clicked=True,
            last_shown_at=persisted_timestamp,
            show_count=3,
            translation_success_observed=True,
            eligible_launch_count=2,
        )
    )

    serialized = serialization.to_dict(settings)
    restored = serialization.from_dict(serialized)
    prompt = serialized["state"]["github_star_prompt"]

    assert _prompt(settings).clicked is True
    assert _prompt(settings).last_shown_at == persisted_timestamp
    assert _prompt(settings).show_count == 3
    assert _prompt(settings).translation_success_observed is True
    assert _prompt(settings).eligible_launch_count == 2
    assert prompt["clicked"] is True
    assert prompt["last_shown_at"] == persisted_timestamp
    assert prompt["show_count"] == 3
    assert prompt["translation_success_observed"] is True
    assert prompt["eligible_launch_count"] == 2
    assert _prompt(restored) == _prompt(settings)
