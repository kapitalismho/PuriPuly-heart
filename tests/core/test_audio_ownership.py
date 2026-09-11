from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioRetentionBudget,
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart


def test_retention_budget_counts_alias_once_and_distinct_copies_at_actual_bytes() -> None:
    budget = AudioRetentionBudget(
        capacity_bytes=40,
        capacity_sample_equivalents=12,
    )
    source_float = np.zeros(4, dtype=np.float32)
    pcm16_copy = bytearray(8)
    native_float_copy = source_float.copy()

    assert budget.try_reserve(
        source_float,
        source_float.nbytes,
        sample_equivalents=source_float.size,
    )
    assert budget.try_reserve(
        source_float,
        source_float.nbytes,
        sample_equivalents=source_float.size,
    )
    assert budget.used_bytes == 16
    assert budget.used_sample_equivalents == 4
    assert budget.try_reserve(
        pcm16_copy,
        len(pcm16_copy),
        sample_equivalents=source_float.size,
    )
    assert budget.try_reserve(
        native_float_copy,
        native_float_copy.nbytes,
        sample_equivalents=native_float_copy.size,
    )
    assert budget.used_bytes == 40
    assert budget.high_water_bytes == 40
    assert budget.high_water_sample_equivalents == 12
    assert not budget.try_reserve(object(), 1, sample_equivalents=1)

    budget.release(pcm16_copy)
    assert budget.used_bytes == 32
    budget.release(source_float)
    budget.release(native_float_copy)
    assert budget.used_bytes == 0


def test_retention_budget_rejects_sample_equivalent_limit_before_byte_limit() -> None:
    budget = AudioRetentionBudget(
        capacity_bytes=11_520_000,
        capacity_sample_equivalents=2_880_000,
    )
    sample_count = 1_440_001
    source_float = object()
    pcm16_copy = object()

    assert budget.try_reserve(
        source_float,
        sample_count * 4,
        sample_equivalents=sample_count,
    )
    assert not budget.try_reserve(
        pcm16_copy,
        sample_count * 2,
        sample_equivalents=sample_count,
    )
    assert budget.used_bytes == sample_count * 4
    assert budget.used_sample_equivalents == sample_count


def _settings(provider_id: str = "test") -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id=provider_id,
        provider_signature=(provider_id,),
        runtime_signature=(provider_id, "runtime"),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.5,
        vad_hangover_ms=640,
        vad_pre_roll_ms=500,
    )


def _span(start: int, end: int, *, sequence: int = 0) -> AudioCaptureSpan:
    return AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=sequence,
        source_sample_rate_hz=16000,
        source_start_sample=start,
        source_end_sample=end,
        source_start_monotonic_s=start / 16000,
        source_end_monotonic_s=end / 16000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def test_open_segment_rejects_early_terminal_and_terminal_snapshot_is_immutable() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=_settings())
    segment_id = uuid4()
    first = _span(0, 8)
    second = _span(8, 16, sequence=1)
    ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(first,),
        ),
        now_monotonic_s=0.0,
    )

    with pytest.raises(RuntimeError, match="open audio segment"):
        ledger.terminalize(segment_id, outcome="final", now_monotonic_s=0.1)

    ledger.observe_vad_event(
        SpeechChunk(segment_id, chunk=np.ones((8,), dtype=np.float32), chunk_capture=(second,)),
        now_monotonic_s=0.2,
    )
    ledger.observe_vad_event(
        SpeechEnd(segment_id, trailing_silence_ms=0, reason="silence"),
        now_monotonic_s=0.3,
    )
    with pytest.raises(ValueError, match="authoritative provider completion"):
        ledger.terminalize(segment_id, outcome="empty", now_monotonic_s=0.4)
    receipt = ledger.terminalize(segment_id, outcome="final", now_monotonic_s=0.4)

    with pytest.raises(RuntimeError, match="terminal audio segment"):
        ledger.observe_vad_event(
            SpeechChunk(segment_id, chunk=np.ones((8,), dtype=np.float32), chunk_capture=(second,)),
            now_monotonic_s=0.5,
        )
    assert receipt.segment.content_sample_count == 16
    assert receipt.segment.state == "terminal"
    assert ledger.terminalize(segment_id, outcome="failed", now_monotonic_s=0.6) is receipt


def test_ongoing_content_ranges_coalesce_and_open_failure_retires_metadata() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=_settings())
    segment_id = uuid4()
    ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(_span(0, 8),),
        ),
        now_monotonic_s=0.0,
    )
    for sequence in range(1, 5_000):
        start = sequence * 8
        ledger.observe_vad_event(
            SpeechChunk(
                segment_id,
                chunk=np.ones((8,), dtype=np.float32),
                chunk_capture=(_span(start, start + 8, sequence=sequence),),
            ),
            now_monotonic_s=sequence / 100.0,
        )

    snapshot = ledger.snapshots[0]
    assert snapshot.content_sample_count == 40_000
    assert len(snapshot.content_ranges) == 1
    receipt = ledger.terminalize_for_failure(
        segment_id,
        now_monotonic_s=50.0,
        failure_reason="buffer_exhausted",
    )

    assert receipt.outcome == "failed"
    assert receipt.failure_reason == "buffer_exhausted"
    assert ledger.snapshots == ()
    assert ledger.terminal_receipts == (receipt,)


def test_retired_segment_retention_is_bounded_with_recent_dedupe() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=_settings())
    first_id = None
    last_id = None
    sample = np.ones((1,), dtype=np.float32)

    for order in range(4_097):
        segment_id = uuid4()
        if first_id is None:
            first_id = segment_id
        last_id = segment_id
        capture = _span(order, order + 1, sequence=order)
        ledger.observe_vad_event(
            SpeechStart(
                segment_id,
                pre_roll=np.empty((0,), dtype=np.float32),
                chunk=sample,
                chunk_capture=(capture,),
            ),
            now_monotonic_s=float(order),
        )
        ledger.observe_vad_event(
            SpeechEnd(segment_id, trailing_silence_ms=0, reason="silence"),
            now_monotonic_s=float(order) + 0.1,
        )
        ledger.terminalize(
            segment_id,
            outcome="empty",
            now_monotonic_s=float(order) + 0.2,
            text_authority="authoritative",
        )

    assert ledger.snapshots == ()
    assert len(ledger.terminal_receipts) == 4096
    assert last_id is not None and ledger.contains_segment(last_id)
    assert first_id is not None and not ledger.contains_segment(first_id)


def test_reused_prefix_is_accounted_as_context_without_duplicate_content() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=_settings())
    first_id = uuid4()
    prefix = _span(0, 8)
    ledger.observe_vad_event(
        SpeechStart(
            first_id,
            pre_roll=np.empty((0,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            chunk_capture=(prefix,),
        ),
        now_monotonic_s=0.0,
    )
    ledger.observe_vad_event(
        SpeechEnd(first_id, trailing_silence_ms=0, reason="silence"),
        now_monotonic_s=0.1,
    )
    ledger.terminalize(first_id, outcome="final", now_monotonic_s=0.2)

    second_id = uuid4()
    content = _span(8, 16, sequence=1)
    owned = ledger.observe_vad_event(
        SpeechStart(
            second_id,
            pre_roll=np.ones((8,), dtype=np.float32),
            chunk=np.ones((8,), dtype=np.float32),
            pre_roll_capture=(prefix,),
            chunk_capture=(content,),
        ),
        now_monotonic_s=0.3,
    )

    assert owned.segment.prefix_context_sample_count == 8
    assert owned.segment.context_sample_count == 8
    assert owned.segment.content_sample_count == 8
    assert owned.segment.context_ranges == (prefix,)
    assert owned.segment.content_ranges == (content,)
