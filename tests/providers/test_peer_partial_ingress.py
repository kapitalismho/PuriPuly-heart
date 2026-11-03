from __future__ import annotations

import json
from dataclasses import replace

import pytest

from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.stt.backend import (
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.domain.recognition import RecognitionStreamIdentity
from tests.core.test_stt_scoped_engine import segment_events, settings, wait_until
from tests.providers.test_deepgram_reuse import _request as deepgram_request
from tests.providers.test_deepgram_reuse import _result as deepgram_result
from tests.providers.test_deepgram_reuse import _session as deepgram_session
from tests.providers.test_gemini_transcribe_backend import (
    _backend,
    _FakeLiveSession,
    _final,
    _interim,
)
from tests.providers.test_gemini_transcribe_backend import _stub_gemini_setup as _stub_gemini_setup
from tests.providers.test_qwen_audio import native
from tests.providers.test_qwen_audio_peer import make_engine
from tests.providers.test_soniox_reuse import _next as soniox_next
from tests.providers.test_soniox_reuse import _request as soniox_request
from tests.providers.test_soniox_reuse import _seal as soniox_seal
from tests.providers.test_soniox_reuse import _session as soniox_session


@pytest.mark.asyncio
async def test_soniox_latest_confirmed_plus_provisional_and_retraction_then_authoritative_final():
    session, _ = soniox_session()
    request = soniox_request(1)
    await session.begin_turn(request)
    normalizer = STTScopedTurnNormalizer(request.identity)
    try:
        session._handle_message(
            json.dumps(
                {
                    "tokens": [
                        {"text": "I agree", "is_final": True},
                        {"text": ". But I do not.", "is_final": False},
                    ]
                }
            )
        )
        stable = await soniox_next(session)
        partial = await soniox_next(session)
        assert stable.stability == "stable" and stable.text == "I agree"
        assert partial.stability == "provisional" and partial.text == "I agree. But I do not."
        normalizer.apply_update(stable)
        assert normalizer.apply_update(partial).text == partial.text
        first_at = partial.first_nonempty_at
        session._handle_message(json.dumps({"tokens": [{"text": ",", "is_final": False}]}))
        changed = await soniox_next(session)
        assert changed.text == "I agree,"
        assert changed.first_nonempty_at == first_at
        session._handle_message(json.dumps({"tokens": []}))
        retracted = await soniox_next(session)
        assert retracted.text == "I agree"
        assert retracted.first_nonempty_at == first_at
        await soniox_seal(session, request)
        session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
        terminal = await soniox_next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert normalizer.apply_terminal(terminal).text == "I agree"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_soniox_pending_snapshot_keeps_first_nonempty_arrival_when_coalesced():
    session, _ = soniox_session()
    request = soniox_request(1)
    await session.begin_turn(request)
    try:
        session._handle_message(json.dumps({"tokens": [{"text": "first", "is_final": False}]}))
        first_at = session._last_rx_at
        session._handle_message(
            json.dumps({"tokens": [{"text": "first longer,", "is_final": False}]})
        )
        latest = await soniox_next(session)
        assert latest.text == "first longer,"
        assert latest.first_nonempty_at == first_at
        assert latest.observed_at >= first_at
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_soniox_self_remains_confirmed_only():
    session, _ = soniox_session()
    request = replace(soniox_request(1), channel="self")
    await session.begin_turn(request)
    try:
        session._handle_message(
            json.dumps(
                {
                    "tokens": [
                        {"text": "confirmed", "is_final": True},
                        {"text": " provisional", "is_final": False},
                    ]
                }
            )
        )
        event = await soniox_next(session)
        assert event.stability == "stable" and event.text == "confirmed"
        assert session._event_projection.scoped_event_depth == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_deepgram_interims_keep_finalize_bundle_as_only_final():
    session = deepgram_session()
    request = replace(deepgram_request(1), channel="peer")
    await session.begin_turn(request)
    normalizer = STTScopedTurnNormalizer(request.identity)
    try:
        session._build_transcript_event(deepgram_result("First", is_final=True))
        stable = await anext(session.turn_events())
        normalizer.apply_update(stable)
        await anext(session.turn_events())
        session._build_transcript_event(deepgram_result("tail?", is_final=False))
        partial = await anext(session.turn_events())
        assert partial.stability == "provisional" and partial.text == "First tail?"
        assert normalizer.apply_update(partial).text == "First tail?"
        session._build_transcript_event(deepgram_result("tail.", is_final=True))
        normalizer.apply_update(await anext(session.turn_events()))
        await anext(session.turn_events())
        session._event_projection.seal(request.identity)
        session._build_transcript_event(deepgram_result("", is_final=False, top_level_ack=True))
        terminal = await anext(session.turn_events())
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert normalizer.apply_terminal(terminal).text == "Firsttail."
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_gemini_native_interims_and_final_share_one_caption_identity(_stub_gemini_setup):
    live = _FakeLiveSession()
    backend, _ = _backend(live)
    session = await backend.open_session(projection=STTSessionProjection("scoped", "epoch"))
    await session.begin_stream(
        RecognitionStreamIdentity("peer", 1, 0, "epoch", ("gemini_transcribe",))
    )
    try:
        session._handle_message(_interim("Observed first"))
        first = await anext(session.turn_events())
        session._handle_message(_interim("Observed corrected longer"))
        refresh = await anext(session.turn_events())
        session._handle_message(_final("Authoritative final"))
        final = await anext(session.turn_events())
        assert first.is_final is False and refresh.is_final is False and final.is_final is True
        assert first.identity.unit_id == refresh.identity.unit_id == final.identity.unit_id
        assert first.text == "Observed first" and refresh.text == "Observed corrected longer"
        session._handle_message(_interim("Successor"))
        successor = await anext(session.turn_events())
        assert successor.identity.unit_id != final.identity.unit_id
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_qwen_partial_fixed_and_empty_restart_remain_snapshots_not_finals():
    events = []
    engine, factory, _ = await make_engine("peer", sink=events.append)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        socket = factory.sockets[0]
        task = socket.task_ids[0]
        await socket.push(native(task, text="provisional", sentence_end=False))
        await wait_until(lambda: len(events) >= 1)
        assert events[-1].text == "provisional" and events[-1].stability == "provisional"
        await socket.push(native(task, text="", sentence_end=False))
        await wait_until(lambda: len(events) >= 2)
        assert events[-1].text == ""
        await socket.push(native(task, text="confirmed"))
        await wait_until(lambda: any(event.stability == "stable" for event in events))
        await socket.push(native(task, sentence_id=2, text=" restarted", sentence_end=False))
        await wait_until(lambda: events[-1].text.endswith("restarted"))
        assert events[-1].text == "confirmed restarted"
        assert all(isinstance(event, STTProviderTurnUpdate) for event in events)
    finally:
        await engine.close()
