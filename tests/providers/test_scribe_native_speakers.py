from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.stt.backend import (
    STTFinalSpeakerEvidence,
    STTNativeSegmentRef,
    STTProviderEpochEnded,
    STTProviderTurnTerminal,
    scoped_event_for_delivery,
)
from tests.helpers.scoped_provider_sessions import FakeScribeWebSocket
from tests.providers.test_elevenlabs_scribe_reuse import _next, _request, _seal, _session, _spans


def diarized(ws, index, *words, text=None):
    ws.push(
        "diarized_transcript",
        segment_index=index,
        text="".join(token for token, _speaker in words) if text is None else text,
        words=[{"text": token, "speaker_id": speaker} for token, speaker in words],
    )


@pytest.mark.asyncio
async def test_indexed_text_progress_and_delayed_null_and_chained_merge_evidence():
    ws = FakeScribeWebSocket("actual-native-id")
    session, _ = await _session(ws, diarize=True)
    try:
        for index, text in enumerate(("Repeated phrase", "", "Repeated phrase")):
            turn = _request(index + 1, channel="peer")
            await session.begin_turn(turn)
            await _seal(session, turn)
            ws.push("committed_transcript", segment_index=index, text=text)
            terminal = await _next(session)
            assert isinstance(terminal, STTProviderTurnTerminal)
            assert terminal.native_segment == STTNativeSegmentRef("actual-native-id", index)
            assert terminal.identity == turn.identity
            assert terminal.outcome == ("final" if text else "empty")
            if text:
                assert terminal.final_speaker_runs[0].attribution.state == "missing"
        diarized(ws, 1, ("", None))
        diarized(ws, 2, ("wrong coverage", "A"), text="Repeated phrase")
        diarized(ws, 0, ("Repeated ", "C"), ("phrase", None))
        evidence = await _next(session)
        assert isinstance(evidence, STTFinalSpeakerEvidence)
        assert [item.native_segment.segment_index for item in evidence.revisions] == [0]
        assert [run.speaker_id for run in evidence.revisions[0].final_speaker_runs] == ["C", None]
        diarized(ws, 0, ("Repeated ", "C"), ("phrase", None))
        ws.push("speakers_merged", from_speaker_id="C", to_speaker_id="B")
        assert (await _next(session)).revisions[0].final_speaker_runs[0].speaker_id == "B"
        ws.push("speakers_merged", from_speaker_id="B", to_speaker_id="A")
        merged = await _next(session)
        assert merged.revisions[0].final_speaker_runs[0].speaker_id == "A"
        diarized(ws, 2, ("Repeated phrase", "C"))
        future_alias = await _next(session)
        assert future_alias.revisions[0].native_segment.segment_index == 2
        assert future_alias.revisions[0].final_speaker_runs[0].speaker_id == "A"
        await session.stop()
        assert scoped_event_for_delivery(future_alias) is None
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [1, -1, None, True])
async def test_unsynchronised_native_index_retires_the_manual_slot(index):
    ws = FakeScribeWebSocket()
    session, _ = await _session(ws, diarize=True)
    request = _request(1, channel="peer")
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        ws.push("committed_transcript", text="must not bind", segment_index=index)
        terminal = await _next(session)
        assert terminal.outcome == "failed"
        assert terminal.text_authority == "none"
        assert terminal.epoch_disposition == "retire"
        assert isinstance(await _next(session), STTProviderEpochEnded)
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("send_pending", [True, False])
async def test_metadata_already_queued_is_part_of_authoritative_text_terminal(send_pending):
    ws = FakeScribeWebSocket()
    session, _ = await _session(ws, diarize=True)
    request = _request(1, channel="peer")
    try:
        await session.begin_turn(request)
        if send_pending:
            ws.commit_gate.clear()
        seal = asyncio.create_task(_seal(session, request))
        while ws.commits == 0:
            await asyncio.sleep(0)
        ws.push("committed_transcript", text="  Spoken here.  ", segment_index=0)
        diarized(ws, 0, ("  Spoken ", "speaker-a"), ("here.  ", "speaker-b"))
        await asyncio.sleep(0)
        if send_pending:
            assert session._event_projection.scoped_event_depth == 0
        ws.commit_gate.set()
        await seal
        event = await _next(session)
        assert isinstance(event, STTProviderTurnTerminal)
        assert event.text == "Spoken here."
        assert [(item.text, item.speaker_id) for item in event.final_speaker_runs] == [
            ("Spoken ", "speaker-a"),
            ("here.", "speaker-b"),
        ]
        assert session._event_projection.scoped_event_depth == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_bounded_old_evidence_does_not_block_current_evidence_or_text():
    ws = FakeScribeWebSocket()
    session, _ = await _session(ws, diarize=True)
    try:
        for index in range(35):
            request = _request(index + 1, channel="peer")
            await session.begin_turn(request)
            await _seal(session, request)
            ws.push("committed_transcript", text=f"Sentence {index}.", segment_index=index)
            assert (await _next(session)).identity == request.identity
        diarized(ws, 0, ("Sentence 0.", "stale"))
        diarized(ws, 34, ("Sentence 34.", "current"))
        evidence = await _next(session)
        assert [item.native_segment.segment_index for item in evidence.revisions] == [34]
        assert len(session._committed) <= 32
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_manual_commit_safety_counts_all_transmitted_audio_without_new_cut():
    ws = FakeScribeWebSocket()
    session, _ = await _session(ws)
    request = _request(1)
    try:
        await session.begin_turn(request)
        for sequence, seconds in ((1, 15), (2, 15)):
            await session.send_turn_audio(
                request.identity,
                bytes(seconds * 16000 * 2),
                payload_sequence=sequence,
                source_ranges=_spans(),
                context_only=sequence == 1,
            )
        with pytest.raises(RuntimeError, match="manual commit audio limit"):
            await session.send_turn_audio(
                request.identity,
                b"\x00\x00",
                payload_sequence=3,
                source_ranges=_spans(),
                context_only=False,
            )
        assert (await _next(session)).failure_reason == "scribe_manual_commit_audio_limit"
        assert len(ws.sent) == 2 and ws.commits == 0
    finally:
        await session.close()
