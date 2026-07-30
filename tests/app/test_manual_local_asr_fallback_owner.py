from __future__ import annotations

import pytest

from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
    ManualLocalASRFallbackState,
)


def _state(
    *,
    self_provider: str = "local_parakeet_v3",
    peer_provider: str = "local_parakeet_ja",
    self_language: str = "ko",
    peer_language: str = "en",
    cpu_auto_available: bool = True,
) -> ManualLocalASRFallbackState:
    return ManualLocalASRFallbackState(
        self_provider=self_provider,
        peer_provider=peer_provider,
        self_source_language=self_language,
        peer_source_language=peer_language,
        cpu_auto_available=cpu_auto_available,
    )


def test_owner_plans_both_manual_language_mismatches_without_mutating_state() -> None:
    state = _state()

    plan = ManualLocalASRFallbackOwner().plan(state)

    assert plan.self_provider == "local_qwen"
    assert plan.peer_provider == "local_qwen"
    assert plan.fallback_channels == ("self", "peer")
    assert plan.installation_fallback is False
    assert state.self_provider == "local_parakeet_v3"
    assert state.peer_provider == "local_parakeet_ja"


@pytest.mark.parametrize(
    ("channel", "expected_self", "expected_peer", "expected_channels"),
    (
        ("self", "local_qwen", "local_parakeet_ja", ("self",)),
        ("peer", "local_parakeet_v3", "local_qwen", ("peer",)),
    ),
)
def test_owner_scopes_fallback_to_requested_channel(
    channel: str,
    expected_self: str,
    expected_peer: str,
    expected_channels: tuple[str, ...],
) -> None:
    plan = ManualLocalASRFallbackOwner().plan(_state(), channel=channel)

    assert plan.self_provider == expected_self
    assert plan.peer_provider == expected_peer
    assert plan.fallback_channels == expected_channels


def test_owner_marks_cpu_auto_installation_fallback() -> None:
    plan = ManualLocalASRFallbackOwner().plan(
        _state(
            self_provider="local_cpu_auto",
            peer_provider="local_qwen",
            cpu_auto_available=False,
        )
    )

    assert plan.self_provider == "local_qwen"
    assert plan.fallback_channels == ("self",)
    assert plan.installation_fallback is True


def test_owner_rejects_unknown_channel() -> None:
    with pytest.raises(ValueError, match="channel must be"):
        ManualLocalASRFallbackOwner().plan(_state(), channel="system")


def test_owner_classifies_only_changed_manual_channels_for_normalization() -> None:
    owner = ManualLocalASRFallbackOwner()
    current = _state(
        self_provider="deepgram",
        peer_provider="local_parakeet_ja",
        self_language="en",
        peer_language="ja",
    )
    pending = _state(
        self_provider="local_parakeet_v3",
        peer_provider="local_parakeet_ja",
        self_language="ko",
        peer_language="en",
    )

    assert owner.normalization_channels(current=current, pending=pending) == {
        "self",
        "peer",
    }
    assert owner.normalization_channels(current=None, pending=pending) == {
        "self",
        "peer",
    }
