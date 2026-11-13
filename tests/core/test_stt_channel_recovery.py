from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.stt.backend import STTChannelAvailability, STTProviderTurnTerminal
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from tests.core.test_stt_scoped_engine import (
    ControlledScopedSession,
    segment_events,
    settings,
    wait_until,
    watchdogs,
)

ORDINARY_PROVIDERS = (
    "soniox", "deepgram", "elevenlabs_scribe", "custom", "rolling_free",
    "local_cpu_auto", "local_granite", "local_gigaam", "local_fun_asr_nano",
    "local_fun_asr_mlt_nano", "local_parakeet_ja",
)


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("provider_id", ORDINARY_PROVIDERS)
async def test_ordinary_recovery_is_bounded_next_input_not_replay(channel, provider_id):
    sessions, events, failures, delays = [], [], [], []

    async def factory(_settings, _epoch):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    async def sleep(delay):
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=factory, channel=channel, event_sink=events.append,
        terminal_failure_sink=failures.append,
        watchdog_resolver=lambda _: watchdogs(connect_retry_base_s=0.8, connect_retry_max_s=1.6),
        sleep=sleep, session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings(provider_id))
    try:
        for order in range(4):
            start, _, end = segment_events(ledger, start_sample=order * 20, now=order)
            await engine.handle_owned_vad_event(start)
            if order < 3:
                session = sessions[-1]
                session.emit(STTProviderTurnTerminal(
                    session.requests[-1].identity, "failed", failure_reason="local_decode_failed",
                    failure_retryable=True, epoch_disposition="retire",
                ))
            await engine.handle_owned_vad_event(end)
            if order < 3:
                await wait_until(lambda: ("close",) in sessions[-1].calls)
        terminals = [event for event in events if isinstance(event, STTProviderTurnTerminal)]
        assert [event.channel_availability for event in terminals] == [
            STTChannelAvailability.RECOVERING, STTChannelAvailability.RECOVERING,
            STTChannelAvailability.UNAVAILABLE, STTChannelAvailability.UNAVAILABLE,
        ]
        assert len(sessions) == 3
        assert [len(session.requests) for session in sessions] == [1, 1, 1]
        assert delays == [0.8, 1.6]
        assert len(failures) == 1
        assert not any(event.text for event in terminals)
    finally:
        await engine.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_correlated_failure_and_old_success_do_not_reset_new_failure(channel):
    sessions, events, failures, delays = [], [], [], []

    async def factory(_settings, _epoch):
        session = ControlledScopedSession()
        session.allows_sealed_turn_overlap = True
        sessions.append(session)
        return session

    async def sleep(delay):
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=factory, channel=channel, event_sink=events.append,
        terminal_failure_sink=failures.append, sleep=sleep, session_lifetime_enabled=False,
        watchdog_resolver=lambda _: watchdogs(connect_retry_base_s=0.8, connect_retry_max_s=1.6),
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("local_granite"))
    try:
        ends = []
        for order in range(3):
            start, _, end = segment_events(ledger, start_sample=order * 20, now=order)
            await engine.handle_owned_vad_event(start)
            ends.append(asyncio.create_task(engine.handle_owned_vad_event(end)))
            await wait_until(lambda: any(call[0] == "seal" for call in sessions[0].calls))
            await asyncio.sleep(0)
        identities = [request.identity for request in sessions[0].requests]
        assert len(identities) == 3
        for identity in identities[1:]:
            sessions[0].emit(STTProviderTurnTerminal(
                identity, "failed", failure_reason="local_decode_failed", failure_retryable=True,
                failure_resource_id="resource-generation-one", epoch_disposition="retire",
            ))
        await asyncio.sleep(0)
        sessions[0].emit(STTProviderTurnTerminal(identities[0], "final", text="Old queued success"))
        await asyncio.gather(*ends)
        await wait_until(lambda: ("close",) in sessions[0].calls)
        assert engine._episode_failures == 1
        assert not failures
        for order in (3, 4):
            start, _, end = segment_events(ledger, start_sample=order * 20, now=order)
            await engine.handle_owned_vad_event(start)
            session = sessions[-1]
            session.emit(STTProviderTurnTerminal(
                session.requests[-1].identity, "failed", failure_reason="local_decode_failed",
                failure_retryable=True, epoch_disposition="retire",
            ))
            await engine.handle_owned_vad_event(end)
            await wait_until(lambda: ("close",) in session.calls)
        assert len(sessions) == 3
        assert delays == [0.8, 1.6]
        assert len(failures) == 1
        assert events[-1].channel_availability is STTChannelAvailability.UNAVAILABLE
        assert [event.text for event in events if isinstance(event, STTProviderTurnTerminal)] == [
            "Old queued success", "", "", "", "",
        ]
    finally:
        await engine.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("reason", ["toggle_off", "source_discontinuity", "cancelled"])
async def test_recovery_backoff_cannot_restore_revoked_capture_authority(channel, reason):
    sessions, events = [], []
    backoff, release = asyncio.Event(), asyncio.Event()

    async def factory(_settings, _epoch):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    async def sleep(_delay):
        backoff.set()
        await release.wait()

    engine = ScopedRecognitionEngine(
        session_factory=factory, channel=channel, event_sink=events.append,
        watchdog_resolver=lambda _: watchdogs(), sleep=sleep, session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("local_fun_asr_nano"))
    pending = None
    try:
        start, _, end = segment_events(ledger, start_sample=0, now=0)
        await engine.handle_owned_vad_event(start)
        session = sessions[0]
        session.emit(STTProviderTurnTerminal(
            session.requests[0].identity, "failed", failure_reason="local_decode_failed",
            failure_retryable=True, epoch_disposition="retire",
        ))
        await engine.handle_owned_vad_event(end)
        await wait_until(lambda: ("close",) in session.calls)
        next_start, _, _ = segment_events(ledger, start_sample=20, now=1)
        pending = asyncio.create_task(engine.handle_owned_vad_event(next_start))
        await asyncio.wait_for(backoff.wait(), 1)
        await engine.abort(reason=reason)
        release.set()
        await pending
        assert len(sessions) == 1
        fresh = PeerAudioSegmentLedger(
            activation_generation=2, settings=replace(settings("local_fun_asr_nano"), runtime_signature=("new",))
        )
        fresh_start, _, fresh_end = segment_events(fresh, start_sample=40, now=2)
        await engine.handle_owned_vad_event(fresh_start)
        sessions[-1].terminal_on_seal = ("final", "Authorized new configuration")
        await engine.handle_owned_vad_event(fresh_end)
        assert len(sessions) == 2
        assert [event.text for event in events if isinstance(event, STTProviderTurnTerminal) and event.text] == [
            "Authorized new configuration"
        ]
    finally:
        release.set()
        if pending is not None:
            await pending
        await engine.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("error", [RuntimeError("transient decode retryable"), PermissionError("blocked")])
async def test_unknown_or_permanent_readiness_failure_never_uses_generic_retry(channel, error):
    attempts, failures, events = [], [], []

    async def factory(_settings, _epoch):
        attempts.append(_epoch)
        raise error

    engine = ScopedRecognitionEngine(
        session_factory=factory, channel=channel, event_sink=events.append,
        terminal_failure_sink=failures.append, watchdog_resolver=lambda _: watchdogs(),
        session_lifetime_enabled=False,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("local_fun_asr_nano"))
    try:
        for order in range(2):
            start, _, end = segment_events(ledger, start_sample=order * 20, now=order)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
        assert len(attempts) == 1
        assert len(failures) == 1
        assert all(event.channel_availability is STTChannelAvailability.UNAVAILABLE for event in events)
    finally:
        await engine.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("provider_id", ORDINARY_PROVIDERS)
async def test_ordinary_recovery_requires_new_recognition_progress(channel, provider_id):
    sessions, events, failures, delays = [], [], [], []

    async def factory(_settings, _epoch):
        session = ControlledScopedSession()
        sessions.append(session)
        return session

    async def sleep(delay):
        delays.append(delay)

    engine = ScopedRecognitionEngine(
        session_factory=factory, channel=channel, event_sink=events.append,
        terminal_failure_sink=failures.append, sleep=sleep, session_lifetime_enabled=False,
        watchdog_resolver=lambda _: watchdogs(connect_retry_base_s=0.8, connect_retry_max_s=1.6),
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings(provider_id))
    try:
        for order in range(3):
            start, _, end = segment_events(ledger, start_sample=order * 20, now=order)
            await engine.handle_owned_vad_event(start)
            session = sessions[-1]
            if order == 1:
                session.terminal_on_seal = ("final", "New authorized recognition")
            else:
                session.terminal_on_seal = None
                session.emit(STTProviderTurnTerminal(
                    session.requests[-1].identity, "failed", failure_reason="local_decode_failed",
                    failure_retryable=True, epoch_disposition="retire",
                ))
            await engine.handle_owned_vad_event(end)
            if order != 1:
                await wait_until(lambda: ("close",) in session.calls)
        terminals = [event for event in events if isinstance(event, STTProviderTurnTerminal)]
        assert [(event.outcome, event.channel_availability) for event in terminals] == [
            ("failed", STTChannelAvailability.RECOVERING),
            ("final", STTChannelAvailability.CONTINUE),
            ("failed", STTChannelAvailability.RECOVERING),
        ]
        assert [event.text for event in terminals] == ["", "New authorized recognition", ""]
        assert len(sessions) == 2
        assert [len(session.requests) for session in sessions] == [1, 2]
        assert delays == [0.8]
        assert not failures
    finally:
        await engine.close()
