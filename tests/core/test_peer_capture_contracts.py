from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from puripuly_heart.core.peer_capture import (
    PeerCapturedFinalFacts,
    PeerCaptureFinalLanguageState,
    PeerCaptureLanguageFacts,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)


def test_peer_capture_contract_keeps_target_language_and_runtime_facts_immutable() -> None:
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="discord",
        discord_channel="stable",
        executable_basename="Discord.exe",
    )
    language = PeerCaptureLanguageFacts(
        source_mode="auto",
        source_language="en",
        expected_languages=("en", "ko"),
    )
    config = PeerCaptureSessionConfig(
        provider_id="soniox",
        provider_signature=("soniox", "auto", ("en", "ko")),
        runtime_signature=(target, "soniox", "auto", ("en", "ko")),
        capture_signature=(target, 16000, 0.6, 900, 500),
        capture_target=target,
        language=language,
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.6,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
    )

    assert config.capture_target.kind == "process"
    assert config.language.source_mode == "auto"
    assert config.language.expected_languages == ("en", "ko")
    with pytest.raises(FrozenInstanceError):
        config.language = PeerCaptureLanguageFacts("manual", "ja")


def test_target_resolution_distinguishes_pending_unavailable_and_resolved() -> None:
    intent = PeerCaptureTargetIntent(kind="default_output_device")
    resolved = PeerCaptureResolvedTarget(intent=intent, capture_descriptor={"device": 3})

    assert PeerCaptureTargetResolution(PeerCaptureTargetStatus.PENDING).target is None
    assert (
        PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.UNAVAILABLE,
            reason="target_not_running",
        ).reason
        == "target_not_running"
    )
    assert (
        PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED,
            target=resolved,
        ).target
        is resolved
    )


@pytest.mark.parametrize(
    ("state", "detected"),
    [
        (PeerCaptureFinalLanguageState.WHOLE_UTTERANCE, ("en",)),
        (PeerCaptureFinalLanguageState.MIXED, ("en", "ko")),
        (PeerCaptureFinalLanguageState.MISSING, ()),
        (PeerCaptureFinalLanguageState.UNSUPPORTED, ("fr",)),
    ],
)
def test_captured_final_contract_preserves_identity_order_and_language_outcome(
    state: PeerCaptureFinalLanguageState,
    detected: tuple[str, ...],
) -> None:
    utterance_id = uuid4()
    facts = PeerCapturedFinalFacts(
        utterance_id=utterance_id,
        capture_sequence=7,
        language=PeerCaptureLanguageFacts(
            source_mode="auto",
            source_language="en",
            expected_languages=("en", "ko"),
        ),
        language_state=state,
        detected_languages=detected,
    )

    assert facts.utterance_id == utterance_id
    assert facts.capture_sequence == 7
    assert facts.language_state is state
    assert facts.detected_languages == detected
