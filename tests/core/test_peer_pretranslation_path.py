from __future__ import annotations

from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    AudioSegmentSnapshot,
    AudioSegmentTerminalReceipt,
)
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTTimedToken,
)
from puripuly_heart.domain.models import FinalLanguageRun
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


def _settings() -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id="soniox",
        provider_signature=("soniox",),
        runtime_signature=("soniox",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def _receipt(segment: AudioSegmentIdentity) -> AudioSegmentTerminalReceipt:
    snapshot = AudioSegmentSnapshot(
        identity=segment,
        settings=_settings(),
        content_ranges=(),
        context_ranges=(),
        failed_ranges=(),
        content_sample_count=0,
        context_sample_count=0,
        failed_normalized_sample_count=0,
        failed_source_sample_count=0,
        prefix_context_sample_count=0,
        synthetic_context_sample_count=0,
        genuine_onset=True,
        state="terminal",
        opened_at_monotonic_s=0.0,
        sealed_at_monotonic_s=1.0,
        seal_reason="silence",
    )
    return AudioSegmentTerminalReceipt(
        identity=segment,
        outcome="final",
        segment=snapshot,
        terminal_at_monotonic_s=1.0,
        text_authority="authoritative",
    )


@pytest.mark.asyncio
async def test_enabled_path_splits_one_parent_into_ownership_children() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = PretranslationOwnershipOwner(enabled=True)
    harness.peer_owner.pretranslation_ownership = owner
    await harness.start()
    segment = AudioSegmentIdentity(
        activation_generation=1,
        segment_order=1,
        segment_id=uuid4(),
        capture_epoch=1,
    )
    identity = STTProviderTurnIdentity(
        segment=segment,
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    owner.observe(
        ProspectiveSpeakerHypothesis(
            hypothesis_id="cut",
            revision=0,
            capture_epoch=1,
            support_start_sample=1500,
            support_end_sample=1700,
            estimated_transition_sample=1600,
            observed_frontier_sample=2000,
            available_at_monotonic_s=0.5,
            producer_generation=1,
            reference_generation=1,
            producer_valid=True,
            reference_valid=True,
            local_slot=1,
        )
    )
    created: list[str] = []
    inner = harness.translation_turns.on_child_created

    async def created_cb(child):
        created.append(child.ownership_group_id)
        await inner(child)

    harness.translation_turns.on_child_created = created_cb
    terminal = STTProviderTurnTerminal(
        identity=identity,
        outcome="final",
        text="Hello there",
        final_language_runs=(FinalLanguageRun("Hello there", "en"),),
        text_authority="authoritative",
        timed_tokens=(
            STTTimedToken(
                text="Hello ",
                language="en",
                start_ms=0,
                end_ms=100,
                timing="interval",
                source_start_sample=0,
                source_end_sample=1600,
            ),
            STTTimedToken(
                text="there",
                language="en",
                start_ms=100,
                end_ms=200,
                timing="interval",
                source_start_sample=1600,
                source_end_sample=3200,
            ),
        ),
    )
    event = await harness.peer_owner.handle_provider_turn_terminal(_receipt(segment), terminal)
    await harness.translation_turns.wait_for_idle()
    await harness.stop()
    assert event is not None
    assert event.transcript.utterance_id == segment.segment_id
    assert created == ["CURRENT-0", "OTHER-1"]


@pytest.mark.asyncio
async def test_disabled_path_does_not_split() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    await harness.start()
    created: list[str] = []
    inner = harness.translation_turns.on_child_created

    async def created_cb(child):
        created.append(child.ownership_group_id or child.transcript.text)
        await inner(child)

    harness.translation_turns.on_child_created = created_cb
    segment = AudioSegmentIdentity(
        activation_generation=1,
        segment_order=1,
        segment_id=uuid4(),
        capture_epoch=1,
    )
    identity = STTProviderTurnIdentity(
        segment=segment,
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    terminal = STTProviderTurnTerminal(
        identity=identity,
        outcome="final",
        text="Hello there",
        final_language_runs=(FinalLanguageRun("Hello there", "en"),),
        text_authority="authoritative",
        timed_tokens=(
            STTTimedToken(
                text="Hello ",
                language="en",
                start_ms=0,
                end_ms=100,
                timing="interval",
                source_start_sample=0,
                source_end_sample=1600,
            ),
            STTTimedToken(
                text="there",
                language="en",
                start_ms=100,
                end_ms=200,
                timing="interval",
                source_start_sample=1600,
                source_end_sample=3200,
            ),
        ),
    )
    await harness.peer_owner.handle_provider_turn_terminal(_receipt(segment), terminal)
    await harness.translation_turns.wait_for_idle()
    await harness.stop()
    assert created == ["Hello there"]
