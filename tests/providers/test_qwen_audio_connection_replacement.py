from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTSessionProjection
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend
from tests.core.test_stt_scoped_engine import (
    ControlledMonotonicClock,
    settings,
    span,
    wait_until,
    watchdogs,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import ScriptedSocket, control, native


async def checkpoint():
    for _ in range(12):
        await asyncio.sleep(0)


async def reached(event):
    await asyncio.wait_for(event.wait(), timeout=1)


class ControlledSocket(ScriptedSocket):
    def __init__(self, *, ready=True, hold_close=False, readiness_failure=None):
        super().__init__(ready=ready)
        self.close_entered = asyncio.Event()
        self.close_release = asyncio.Event()
        if not hold_close:
            self.close_release.set()
        self.closed_event = asyncio.Event()
        self.abort_entered = asyncio.Event()
        self.wait_closed_entered = asyncio.Event()
        self.close_attempts = 0
        self.abort_attempts = 0
        self.reject_close = False
        self.reject_wait_closed = False
        self.readiness_failure = readiness_failure
        self.history = []
        self.transport = SimpleNamespace(abort=self.abort)

    async def send(self, value):
        await super().send(value)
        if isinstance(value, bytes):
            self.history.append("pcm")
        elif json.loads(value)["header"]["action"] == "run-task":
            self.history.append("run-task")
            if self.readiness_failure is not None:
                await self.push(self.readiness_failure)

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            self.history.append(value["header"]["event"])
        return value

    async def close(self):
        self.close_attempts += 1
        self.close_entered.set()
        self.history.append("close")
        if self.reject_close:
            raise TimeoutError("controlled unconfirmed websocket close")
        await self.close_release.wait()
        self.confirm_closed()

    def abort(self):
        self.abort_attempts += 1
        self.abort_entered.set()

    async def wait_closed(self):
        self.wait_closed_entered.set()
        if self.reject_wait_closed:
            raise TimeoutError("controlled unconfirmed transport abort")
        await self.closed_event.wait()

    def confirm_closed(self):
        if not self.closed:
            self.closed = True
            self.closed_event.set()
            self.incoming.put_nowait(None)
        self.close_release.set()

    @property
    def pcm(self):
        return b"".join(value for value in self.sent if isinstance(value, bytes))

    async def finish(self, text, *, task_id=None):
        task_id = task_id or self.task_ids[-1]
        await self.push(native(task_id, text=text))
        await self.push(control(task_id, "task-finished"))


class ControlledFactory:
    def __init__(self, *sockets):
        self.planned = list(sockets)
        self.sockets = []
        self.peak = 0
        self.attempts = 0

    async def __call__(self, *_args, **_kwargs):
        self.attempts += 1
        socket = self.planned.pop(0) if self.planned else ControlledSocket()
        if isinstance(socket, Exception):
            raise socket
        self.sockets.append(socket)
        self.peak = max(self.peak, len(self.live))
        return socket

    @property
    def live(self):
        return [socket for socket in self.sockets if not socket.closed]


@dataclass
class Segment:
    start: object
    chunk: object
    end: object
    pcm: bytes


class ReplacementHarness:
    def __init__(self, factory):
        self.factory = factory
        self.clock = ControlledMonotonicClock()
        self.ledger = PeerAudioSegmentLedger(
            activation_generation=1, settings=settings("qwen_audio")
        )
        self.consumer = compose_translation_test_harness(
            stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
        )
        self.events = []
        self.transcripts = []
        self.terminal_failures = []
        self.retry_delays = []
        self.sessions = []
        self.engine = ScopedRecognitionEngine(
            channel="peer",
            session_factory=self.open_session,
            event_sink=self.consume,
            terminal_failure_sink=self.terminal_failures.append,
            watchdog_resolver=lambda _: watchdogs(
                readiness_timeout_s=2,
                write_timeout_s=1,
                final_timeout_s=5,
                drain_timeout_s=0.2,
                idle_timeout_s=3600,
                connect_retry_base_s=0.001,
                connect_retry_max_s=0.002,
            ),
            monotonic_clock=self.clock.now,
            sleep=self.sleep,
            event_drain_timeout_s=0.2,
        )

    async def open_session(self, _settings, epoch):
        backend = QwenAudioStreamingSTTBackend(
            api_key="offline",
            channel="peer",
            language_hints=("en",),
            websocket_factory=self.factory,
            monotonic_clock=self.clock.now,
            task_start_timeout_s=1,
            task_finish_timeout_s=5,
            send_timeout_s=1,
        )
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        self.sessions.append(session)
        return session

    async def sleep(self, delay):
        if delay >= 30:
            await self.clock.sleep(delay)
        else:
            self.retry_delays.append(delay)
            await asyncio.sleep(0)

    async def consume(self, event):
        self.events.append(event)
        assert isinstance(event, STTProviderTurnTerminal)
        receipt = self.ledger.terminalize(
            event.identity.segment.segment_id,
            outcome=event.outcome,
            now_monotonic_s=self.clock.now(),
            provider_epoch_id=event.identity.provider_epoch_id,
            provider_turn_id=event.identity.provider_turn_id,
            text_authority=event.text_authority,
            failure_reason=event.failure_reason,
        )
        admitted = await self.consumer.peer_owner.handle_provider_turn_terminal(receipt, event)
        if admitted is not None:
            self.transcripts.append(admitted.transcript)

    def segment(self, order):
        identity = uuid4()
        offset = order * 100
        first = np.full(8, order / 32, dtype=np.float32)
        tail = np.full(4, -order / 32, dtype=np.float32)
        start = self.ledger.observe_vad_event(
            SpeechStart(
                identity,
                np.empty(0, dtype=np.float32),
                first,
                chunk_capture=(span(offset, offset, offset + 8),),
            ),
            now_monotonic_s=self.clock.now(),
        )
        chunk = self.ledger.observe_vad_event(
            SpeechChunk(
                identity,
                tail,
                chunk_capture=(span(offset + 1, offset + 8, offset + 12),),
            ),
            now_monotonic_s=self.clock.now(),
        )
        end = self.ledger.observe_vad_event(
            SpeechEnd(identity, trailing_silence_ms=224, reason="silence"),
            now_monotonic_s=self.clock.now(),
        )
        return Segment(start, chunk, end, float32_to_pcm16le_bytes(np.concatenate((first, tail))))

    async def start(self, segment):
        await self.engine.handle_owned_vad_event(segment.start)
        await self.engine.handle_owned_vad_event(segment.chunk)

    async def seal(self, segment):
        await self.engine.handle_owned_vad_event(segment.end)

    async def require_texts(self, *texts):
        await wait_until(lambda: len(self.transcripts) == len(texts))
        assert [transcript.text for transcript in self.transcripts] == list(texts)
        assert [transcript.source_order for transcript in self.transcripts] == list(
            range(1, len(texts) + 1)
        )
        assert all(transcript.recognition_origins for transcript in self.transcripts)

    async def seed_pair(self):
        segments = [self.segment(order) for order in (1, 2)]
        for index, segment in enumerate(segments):
            await self.start(segment)
            await self.seal(segment)
            await self.factory.sockets[index].finish("AB"[index])
            await self.require_texts(*"AB"[: index + 1])
        return segments


@asynccontextmanager
async def replacement_harness(factory):
    env = ReplacementHarness(factory)
    await env.consumer.start()
    env.consumer.output_runtime.activate_peer_generation(1)
    try:
        yield env
    finally:
        for socket in factory.sockets:
            socket.reject_close = False
            socket.reject_wait_closed = False
            socket.confirm_closed()
        await env.engine.close()
        await env.consumer.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [54.999, 55.0, 55.001])
async def test_idle_boundary_waits_for_fresh_readiness_before_retiring_old_socket(elapsed):
    old = ControlledSocket(hold_close=True)
    other = ControlledSocket()
    fresh = ControlledSocket(ready=False)
    factory = ControlledFactory(old, other, fresh)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(elapsed)
        assert factory.sockets == [old, other]
        assert not old.close_entered.is_set() and not other.close_entered.is_set()
        assert sum(len(socket.task_ids) for socket in factory.sockets) == 2
        third = env.segment(3)
        writing = asyncio.create_task(env.start(third))
        try:
            if elapsed < 55:
                await writing
                current = old
                assert factory.sockets == [old, other]
                assert not old.close_entered.is_set()
            else:
                await wait_until(lambda: bool(fresh.task_ids))
                assert not writing.done()
                assert old in factory.live and not old.close_entered.is_set()
                assert fresh.pcm == b""
                assert old.pcm == first.pcm and other.pcm == second.pcm
                await old.push(native(fresh.task_ids[0], text="obsolete before readiness"))
                await fresh.push(control(fresh.task_ids[0], "task-started"))
                await writing
                await reached(old.close_entered)
                assert "task-started" in fresh.history
                assert fresh.history.index("task-started") < fresh.history.index("pcm")
                assert not old.closed and fresh in factory.live
                await old.push(native(fresh.task_ids[0], text="obsolete after retirement"))
                await old.push(control(fresh.task_ids[0], "task-failed", error_code="500"))
                await old.push(control(fresh.task_ids[0], "task-finished"))
                current = fresh
            await env.seal(third)
            await current.finish("C")
            await env.require_texts("A", "B", "C")
            assert [event.outcome for event in env.events] == ["final"] * 3
            assert current.pcm == (first.pcm + third.pcm if current is old else third.pcm)
            assert other.pcm == second.pcm
            assert sum(len(socket.task_ids) for socket in factory.sockets) == 3
            if elapsed >= 55:
                assert old.pcm == first.pcm
                old.confirm_closed()
                await wait_until(lambda: factory.live == [other, fresh])
        finally:
            if not writing.done():
                writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_retirements_allow_four_physical_sockets_not_a_third_recognition_slot():
    old_a = ControlledSocket(hold_close=True)
    old_b = ControlledSocket(hold_close=True)
    current_a, current_b = ControlledSocket(), ControlledSocket()
    factory = ControlledFactory(old_a, old_b, current_a, current_b)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(55)
        third, fourth, fifth = [env.segment(order) for order in (3, 4, 5)]
        await env.start(third)
        await env.seal(third)
        await env.start(fourth)
        await env.seal(fourth)
        await reached(old_a.close_entered)
        await reached(old_b.close_entered)
        assert set(factory.live) == {old_a, old_b, current_a, current_b}
        assert factory.peak == 4
        pending = asyncio.create_task(env.start(fifth))
        try:
            await checkpoint()
            assert not pending.done()
            assert current_a.pcm == third.pcm and current_b.pcm == fourth.pcm
            assert sum(len(socket.task_ids) for socket in factory.sockets) == 4
            await current_b.finish("D")
            await checkpoint()
            assert not pending.done()
            assert [transcript.text for transcript in env.transcripts] == ["A", "B"]
            await old_a.push(native(current_a.task_ids[0], text="retired A"))
            await old_b.push(native(current_b.task_ids[0], text="retired B"))
            await current_a.finish("C")
            await pending
            await env.require_texts("A", "B", "C", "D")
            await env.seal(fifth)
            await current_a.finish("E")
            await env.require_texts("A", "B", "C", "D", "E")
            assert old_a.pcm == first.pcm and old_b.pcm == second.pcm
            assert current_a.pcm == third.pcm + fifth.pcm
            assert current_b.pcm == fourth.pcm
            assert len(factory.sockets) == 4
            assert sum(len(socket.task_ids) for socket in factory.sockets) == 5
            old_a.confirm_closed()
            old_b.confirm_closed()
            await wait_until(lambda: factory.live == [current_a, current_b])
            assert not env.terminal_failures
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_active_and_draining_age_never_evicts_and_idle_begins_at_successful_finish():
    a, b = ControlledSocket(), ControlledSocket()
    factory = ControlledFactory(a, b)
    async with replacement_harness(factory) as env:
        first, second, third = [env.segment(order) for order in (1, 2, 3)]
        await env.engine.handle_owned_vad_event(first.start)
        await env.clock.advance_to(55)
        await env.engine.handle_owned_vad_event(first.chunk)
        assert a.pcm == first.pcm and not a.close_entered.is_set()
        await env.seal(first)
        await env.engine.handle_owned_vad_event(second.start)
        await env.clock.advance_to(110)
        await env.engine.handle_owned_vad_event(second.chunk)
        assert factory.live == [a, b]
        assert not a.close_entered.is_set() and not b.close_entered.is_set()
        await env.seal(second)
        await b.finish("B")
        await checkpoint()
        assert env.transcripts == []
        await env.clock.advance_to(165)
        await a.finish("A")
        await env.require_texts("A", "B")
        await env.start(third)
        await env.seal(third)
        await a.finish("C")
        await env.require_texts("A", "B", "C")
        assert factory.sockets == [a, b]
        assert a.pcm == first.pcm + third.pcm and b.pcm == second.pcm
        assert not a.close_entered.is_set() and not b.close_entered.is_set()


@pytest.mark.asyncio
async def test_next_same_slot_replacement_waits_for_prior_retirement_confirmation():
    old_a = ControlledSocket(hold_close=True)
    old_b = ControlledSocket()
    current_a = ControlledSocket(hold_close=True)
    current_b, newest = ControlledSocket(), ControlledSocket(ready=False)
    factory = ControlledFactory(old_a, old_b, current_a, current_b, newest)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(55)
        third, fourth, fifth = [env.segment(order) for order in (3, 4, 5)]
        for segment, socket, text in ((third, current_a, "C"), (fourth, current_b, "D")):
            await env.start(segment)
            await env.seal(segment)
            await socket.finish(text)
        await env.require_texts("A", "B", "C", "D")
        await reached(old_a.close_entered)
        await wait_until(lambda: old_b.closed)
        await env.clock.advance_to(110)
        writing = asyncio.create_task(env.start(fifth))
        try:
            await checkpoint()
            assert not writing.done()
            assert newest.task_ids == [] and newest.pcm == b""
            assert factory.live == [old_a, current_a, current_b]
            assert not current_a.close_entered.is_set()
            old_a.confirm_closed()
            await wait_until(lambda: bool(newest.task_ids))
            assert not writing.done() and newest.pcm == b""
            assert not current_a.close_entered.is_set()
            await newest.push(control(newest.task_ids[0], "task-started"))
            await writing
            await reached(current_a.close_entered)
            await env.seal(fifth)
            await newest.finish("E")
            await env.require_texts("A", "B", "C", "D", "E")
            assert old_a.pcm == first.pcm and old_b.pcm == second.pcm
            assert current_a.pcm == third.pcm and current_b.pcm == fourth.pcm
            assert newest.pcm == fifth.pcm
            assert factory.peak <= 4
            assert sum(len(socket.task_ids) for socket in factory.sockets) == 5
            current_a.confirm_closed()
            await wait_until(lambda: factory.live == [current_b, newest])
        finally:
            if not writing.done():
                writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)


@pytest.mark.asyncio
async def test_unconfirmed_retirement_remains_owned_without_stopping_healthy_recognition():
    old = ControlledSocket()
    old.reject_close = old.reject_wait_closed = True
    other, fresh = ControlledSocket(), ControlledSocket()
    factory = ControlledFactory(old, other, fresh)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(55)
        third = env.segment(3)
        await env.start(third)
        await reached(old.abort_entered)
        await reached(old.wait_closed_entered)
        assert not old.closed and fresh.pcm == third.pcm
        await env.seal(third)
        await fresh.finish("C")
        await env.require_texts("A", "B", "C")
        assert not env.terminal_failures
        initial_close_attempts = old.close_attempts
        await env.engine.stop()
        await wait_until(lambda: other.closed and fresh.closed)
        await wait_until(lambda: old.close_attempts > initial_close_attempts)
        assert not old.closed and env.engine.cleanup_debt > 0
        assert old.pcm == first.pcm and other.pcm == second.pcm
        assert fresh.pcm == third.pcm
        old.reject_close = old.reject_wait_closed = False
        old.confirm_closed()
        await env.engine.close()
        assert env.engine.cleanup_debt == 0
        assert factory.live == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "toggle_off", "configuration_replaced"])
async def test_stop_and_invalidation_cancel_both_tasks_and_teardown_active_and_retiring(operation):
    old_a = ControlledSocket(hold_close=True)
    old_b = ControlledSocket(hold_close=True)
    current_a, current_b, recovered = ControlledSocket(), ControlledSocket(), ControlledSocket()
    factory = ControlledFactory(old_a, old_b, current_a, current_b, recovered)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(55)
        third, fourth = env.segment(3), env.segment(4)
        await env.start(third)
        await env.seal(third)
        await env.start(fourth)
        await reached(old_a.close_entered)
        await reached(old_b.close_entered)
        await current_b.push(native(current_b.task_ids[0], text="revoked successor"))
        if operation == "stop":
            await env.engine.stop()
        else:
            await env.engine.abort(reason=operation)
        await reached(current_a.close_entered)
        await reached(current_b.close_entered)
        for socket, task_id in ((old_a, current_a.task_ids[0]), (old_b, current_b.task_ids[0])):
            await socket.push(native(task_id, text="revoked retired transport"))
            await socket.push(control(task_id, "task-finished"))
        old_a.confirm_closed()
        old_b.confirm_closed()
        await wait_until(lambda: not factory.live)
        await wait_until(lambda: env.engine.cleanup_debt == 0)
        await wait_until(lambda: len(env.events) == 4)
        assert [event.outcome for event in env.events] == [
            "final",
            "final",
            "cancelled",
            "cancelled",
        ]
        assert all(event.text == "" and event.text_authority == "none" for event in env.events[2:])
        assert [transcript.text for transcript in env.transcripts] == ["A", "B"]
        assert old_a.pcm == first.pcm and old_b.pcm == second.pcm
        assert current_a.pcm == third.pcm and current_b.pcm == fourth.pcm
        fifth = env.segment(5)
        await env.start(fifth)
        await env.seal(fifth)
        await recovered.finish("Recovered")
        await wait_until(lambda: len(env.transcripts) == 3)
        assert [transcript.text for transcript in env.transcripts] == ["A", "B", "Recovered"]
        assert env.transcripts[-1].source_order == 5
        assert recovered.pcm == fifth.pcm
        assert len(env.sessions) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stop", "abort"])
async def test_cancellation_during_replacement_readiness_closes_candidate_without_pcm(operation):
    old_a, old_b = ControlledSocket(), ControlledSocket()
    candidate = ControlledSocket(ready=False)
    recovered = ControlledSocket()
    factory = ControlledFactory(old_a, old_b, candidate, recovered)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(55)
        third = env.segment(3)
        writing = asyncio.create_task(env.start(third))
        try:
            await wait_until(lambda: bool(candidate.task_ids))
            assert candidate.pcm == b""
            assert not old_a.close_entered.is_set()
            if operation == "stop":
                await env.engine.stop()
            else:
                await env.engine.abort(reason="toggle_off")
            await wait_until(lambda: old_a.closed and old_b.closed and candidate.closed)
            await writing
            assert candidate.pcm == b""
            await candidate.push(control(candidate.task_ids[0], "task-started"))
            await candidate.push(native(candidate.task_ids[0], text="late cancelled readiness"))
            fourth = env.segment(4)
            await env.start(fourth)
            await env.seal(fourth)
            await recovered.finish("Recovered")
            await wait_until(lambda: len(env.transcripts) == 3)
            assert [transcript.text for transcript in env.transcripts] == ["A", "B", "Recovered"]
            assert env.transcripts[-1].source_order == 4
            assert old_a.pcm == first.pcm and old_b.pcm == second.pcm
            assert recovered.pcm == fourth.pcm
            assert not env.terminal_failures
        finally:
            if not writing.done():
                writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_temporary_websocket_closure_retries_fresh_task_before_any_pcm():
    failure = ConnectionClosedError(Close(1013, "secret-token temporary vendor reason"), None)
    failed = ControlledSocket(ready=False, readiness_failure=failure)
    recovered = ControlledSocket()
    factory = ControlledFactory(failed, recovered)
    async with replacement_harness(factory) as env:
        first = env.segment(1)
        await env.start(first)
        assert failed.closed and failed.pcm == b""
        assert failed.task_ids[0] != recovered.task_ids[0]
        assert recovered.pcm == first.pcm
        assert env.retry_delays == [0.001]
        diagnostics = failure.failure_diagnostics
        assert diagnostics.fields["ws_close_code"] == 1013
        assert diagnostics.category == "service_unavailable"
        assert diagnostics.channel == "peer"
        assert "secret-token" not in repr(diagnostics)
        assert "temporary vendor reason" not in repr(diagnostics)
        await env.seal(first)
        await recovered.finish("Recovered")
        await env.require_texts("Recovered")
        assert len(env.events) == 1 and env.events[0].outcome == "final"
        assert not env.terminal_failures
        assert sum(len(socket.task_ids) for socket in factory.sockets) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["policy", "auth"])
async def test_actual_policy_and_upgrade_auth_errors_stop_pre_pcm_and_keep_safe_metadata(kind):
    if kind == "policy":
        failure = ConnectionClosedError(Close(1008, "secret-token forbidden vendor reason"), None)
    else:
        failure = InvalidStatus(
            Response(401, "Unauthorized secret-token", Headers(), body=b"secret-token response")
        )
    failed = ControlledSocket(ready=False, readiness_failure=failure)
    unused = ControlledSocket()
    factory = ControlledFactory(failed if kind == "policy" else failure, unused)
    async with replacement_harness(factory) as env:
        first = env.segment(1)
        await env.start(first)
        await env.seal(first)
        await wait_until(lambda: len(env.events) == 1 and bool(env.terminal_failures))
        event = env.events[0]
        assert event.outcome == "failed" and event.text == ""
        assert event.failure_reason == "provider_not_ready"
        assert event.text_authority == "none"
        assert failed.pcm == b"" and factory.attempts == 1
        if kind == "policy":
            assert failed.closed and factory.sockets == [failed]
        else:
            assert factory.sockets == []
        assert unused.task_ids == [] and unused.pcm == b""
        assert env.retry_delays == []
        assert env.transcripts == []
        diagnostics = event.failure_diagnostics
        assert diagnostics is not None and diagnostics.channel == "peer"
        assert diagnostics.fields["attempts"] == 1
        if kind == "policy":
            assert diagnostics.fields["ws_close_code"] == 1008
            assert diagnostics.category == "invalid_response"
        else:
            assert diagnostics.status_code == 401
            assert diagnostics.category == "auth"
        assert diagnostics.fields["qwen_failure_kind"]
        assert "secret-token" not in repr(diagnostics)
        assert "forbidden vendor reason" not in repr(diagnostics)
        assert failure.failure_diagnostics.diagnostic_id == diagnostics.diagnostic_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native_code", "category"),
    [("InvalidApiKey", "auth"), ("CLIENT_ERROR", "invalid_response")],
)
async def test_native_rejection_followed_by_temporary_close_keeps_first_cause_and_stops(
    native_code, category
):
    class QueuedFailureSocket(ControlledSocket):
        async def send(self, value):
            await super().send(value)
            if isinstance(value, str) and json.loads(value)["header"]["action"] == "run-task":
                await self.push(
                    control(
                        self.task_ids[-1],
                        "task-failed",
                        error_code=native_code,
                        error_message="secret-token native rejection",
                    )
                )
                await self.push(
                    ConnectionClosedError(Close(1013, "secret-token subsequent closure"), None)
                )

    failed = QueuedFailureSocket(ready=False)
    unused = ControlledSocket()
    factory = ControlledFactory(failed, unused)
    async with replacement_harness(factory) as env:
        segment = env.segment(1)
        await env.start(segment)
        await env.seal(segment)
        await wait_until(lambda: len(env.events) == 1 and bool(env.terminal_failures))
        event = env.events[0]
        assert event.outcome == "failed" and event.text == "" and env.transcripts == []
        assert factory.attempts == 1 and failed.closed and failed.pcm == b""
        assert unused.task_ids == [] and unused.pcm == b""
        diagnostics = event.failure_diagnostics
        assert diagnostics.category == category
        assert diagnostics.fields["qwen_native_code"] == native_code
        assert diagnostics.fields["attempts"] == 1
        assert "ws_close_code" not in diagnostics.fields
        assert "secret-token" not in repr(diagnostics)


@pytest.mark.asyncio
async def test_each_slot_ages_from_its_own_successful_finish_not_other_slot_activity():
    old_a, old_b, fresh_a = ControlledSocket(), ControlledSocket(), ControlledSocket()
    factory = ControlledFactory(old_a, old_b, fresh_a)
    async with replacement_harness(factory) as env:
        first, second = env.segment(1), env.segment(2)
        await env.start(first)
        await env.seal(first)
        await old_a.finish("A")
        await env.require_texts("A")
        await env.start(second)
        await env.seal(second)
        await env.clock.advance_to(10)
        await old_b.finish("B")
        await env.require_texts("A", "B")
        await env.clock.advance_to(55)
        third, fourth = env.segment(3), env.segment(4)
        await env.start(third)
        await env.seal(third)
        await fresh_a.finish("C")
        await env.require_texts("A", "B", "C")
        await env.start(fourth)
        await env.seal(fourth)
        await old_b.finish("D")
        await env.require_texts("A", "B", "C", "D")
        await wait_until(lambda: old_a.closed)
        assert factory.live == [old_b, fresh_a]
        assert old_a.pcm == first.pcm and fresh_a.pcm == third.pcm
        assert old_b.pcm == second.pcm + fourth.pcm
        assert not old_b.close_entered.is_set()
        assert sum(len(socket.task_ids) for socket in factory.sockets) == 4


@pytest.mark.asyncio
async def test_actual_temporary_readiness_closures_exhaust_three_pre_pcm_attempts_without_replay():
    failures = [
        ConnectionClosedError(Close(1013, f"secret-token temporary failure {index}"), None)
        for index in range(3)
    ]
    failed = [ControlledSocket(ready=False, readiness_failure=failure) for failure in failures]
    unused = ControlledSocket()
    factory = ControlledFactory(*failed, unused)
    async with replacement_harness(factory) as env:
        first = env.segment(1)
        await env.start(first)
        await env.seal(first)
        await wait_until(lambda: len(env.events) == 1 and bool(env.terminal_failures))
        assert env.events[0].outcome == "failed"
        assert env.events[0].failure_reason == "provider_not_ready"
        assert env.events[0].text == "" and env.transcripts == []
        assert factory.attempts == 3 and factory.sockets == failed
        assert len({socket.task_ids[0] for socket in failed}) == 3
        assert all(socket.closed and socket.pcm == b"" for socket in failed)
        assert unused.task_ids == [] and unused.pcm == b""
        assert env.retry_delays == [0.001, 0.002]
        diagnostics = env.events[0].failure_diagnostics
        assert diagnostics is not None
        assert diagnostics.diagnostic_id == failures[0].failure_diagnostics.diagnostic_id
        assert diagnostics.fields["ws_close_code"] == 1013
        assert diagnostics.fields["attempts"] == 3
        assert "secret-token" not in repr(diagnostics)


@pytest.mark.asyncio
async def test_graceful_backend_stop_joins_both_retirements_after_completed_consumer_delivery():
    old_a = ControlledSocket(hold_close=True)
    old_b = ControlledSocket(hold_close=True)
    current_a, current_b = ControlledSocket(), ControlledSocket()
    factory = ControlledFactory(old_a, old_b, current_a, current_b)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(55)
        third, fourth = env.segment(3), env.segment(4)
        for segment, socket, text in ((third, current_a, "C"), (fourth, current_b, "D")):
            await env.start(segment)
            await env.seal(segment)
            await socket.finish(text)
        await env.require_texts("A", "B", "C", "D")
        await reached(old_a.close_entered)
        await reached(old_b.close_entered)
        stopping = asyncio.create_task(env.sessions[0].stop())
        try:
            await reached(current_a.close_entered)
            await reached(current_b.close_entered)
            assert not stopping.done()
            assert factory.live == [old_a, old_b]
            old_a.confirm_closed()
            old_b.confirm_closed()
            await asyncio.wait_for(stopping, timeout=1)
            assert factory.live == []
            assert [event.outcome for event in env.events] == ["final"] * 4
            assert [transcript.text for transcript in env.transcripts] == ["A", "B", "C", "D"]
            assert old_a.pcm == first.pcm and old_b.pcm == second.pcm
            assert current_a.pcm == third.pcm and current_b.pcm == fourth.pcm
        finally:
            if not stopping.done():
                stopping.cancel()
            await asyncio.gather(stopping, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_idle_replacement_readiness_retries_before_old_transport_retirement():
    old = ControlledSocket(hold_close=True)
    other = ControlledSocket()
    failure = ConnectionClosedError(Close(1013, "secret-token replacement retry"), None)
    failed = ControlledSocket(ready=False, readiness_failure=failure)
    fresh = ControlledSocket(ready=False)
    factory = ControlledFactory(old, other, failed, fresh)
    async with replacement_harness(factory) as env:
        first, second = await env.seed_pair()
        await env.clock.advance_to(55)
        third = env.segment(3)
        writing = asyncio.create_task(env.start(third))
        try:
            await wait_until(lambda: bool(fresh.task_ids))
            assert failed.closed and failed.pcm == b""
            assert not old.closed and not old.close_entered.is_set()
            assert fresh.pcm == b"" and not writing.done()
            assert failed.task_ids[0] != fresh.task_ids[0]
            await fresh.push(control(fresh.task_ids[0], "task-started"))
            await writing
            await reached(old.close_entered)
            await env.seal(third)
            await fresh.finish("C")
            await env.require_texts("A", "B", "C")
            assert old.pcm == first.pcm and other.pcm == second.pcm
            assert fresh.pcm == third.pcm
            assert env.retry_delays == [0.001] and not env.terminal_failures
            old.confirm_closed()
            await wait_until(lambda: factory.live == [other, fresh])
        finally:
            if not writing.done():
                writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)
