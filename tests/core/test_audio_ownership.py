from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart


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
    receipt = ledger.terminalize(segment_id, outcome="final", now_monotonic_s=0.4)

    with pytest.raises(RuntimeError, match="terminal audio segment"):
        ledger.observe_vad_event(
            SpeechChunk(segment_id, chunk=np.ones((8,), dtype=np.float32), chunk_capture=(second,)),
            now_monotonic_s=0.5,
        )
    assert receipt.segment.content_sample_count == 16
    assert receipt.segment.state == "terminal"
    assert ledger.terminalize(segment_id, outcome="failed", now_monotonic_s=0.6) is receipt


def test_retired_segment_retention_is_bounded_with_recent_dedupe() -> None:
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=_settings())
    first_id = None
    last_id = None
    sample = np.ones((1,), dtype=np.float32)

    for order in range(10_000):
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
        receipt = ledger.terminalize(
            segment_id,
            outcome="empty",
            now_monotonic_s=float(order) + 0.2,
        )
        assert ledger.drain_ready_terminal_receipts() == (receipt,)

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
    ledger.drain_ready_terminal_receipts()

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
