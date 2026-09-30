from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_normalizer import STTNormalizationError, STTScopedTurnNormalizer
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.models import (
    SpeakerAssignment,
    SpeakerAttribution,
    SpeakerKey,
    Translation,
)
from puripuly_heart.providers.stt.soniox import _SonioxSession
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness

BASELINE = "4dc46fdac8c771d3bb58e46bd44304b86fde9c47"


def settings():
    return AudioSegmentSettingsSnapshot(
        provider_id="soniox", provider_signature=("soniox",), runtime_signature=("soniox",),
        source_mode="desktop", source_language="en", expected_languages=("en",),
        target_sample_rate_hz=16000, vad_speech_threshold=0.4, vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def span(sequence, start, end, epoch=1):
    return AudioCaptureSpan(
        capture_epoch=epoch, callback_sequence=sequence, source_sample_rate_hz=16000,
        source_start_sample=start, source_end_sample=end,
        source_start_monotonic_s=start / 16000, source_end_monotonic_s=end / 16000,
        normalized_sample_rate_hz=16000, normalized_start_sample=start, normalized_end_sample=end,
    )


def token(text, speaker, start, end, final=True):
    item = dict(text=text, start_ms=start, end_ms=end, language="en", is_final=final)
    if speaker is not None:
        item["speaker"] = speaker
    return item


@dataclass
class Wire:
    origin: float
    writes: list = field(default_factory=list)
    receives: list = field(default_factory=list)
    incoming: asyncio.Queue = field(default_factory=asyncio.Queue)
    closed: bool = False

    def elapsed(self):
        return (time.perf_counter() - self.origin) * 1000

    async def send(self, payload):
        row = dict(at_ms=self.elapsed())
        if isinstance(payload, bytes):
            row.update(kind="pcm", bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                       all_zero=not any(payload))
        else:
            row.update(kind="control", payload=payload)
        self.writes.append(row)

    async def recv(self):
        payload = await self.incoming.get()
        self.receives.append(dict(at_ms=self.elapsed(), payload=json.loads(payload)))
        return payload

    async def close(self):
        self.closed = True

    def inject(self, tokens=None, **extra):
        self.incoming.put_nowait(json.dumps(dict(tokens=tokens or [], **extra)))


async def wait_for(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0)


def session(epoch, wire):
    result = _SonioxSession(
        api_key="model-free-fixture", model="stt-rt-v5", endpoint="wss://example.invalid",
        sample_rate_hz=16000, language_hints=["en"], context_terms=[],
        keepalive_interval_s=10, trailing_silence_ms=100, connect_timeout_s=1,
        enable_language_identification=True, enable_speaker_diarization=True,
        projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch),
    )
    result._ws = wire
    result._send_task = asyncio.create_task(result._send_loop())
    result._recv_task = asyncio.create_task(result._recv_loop())
    return result


@dataclass
class Translator:
    origin: float
    gates: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    completed: list = field(default_factory=list)
    active: int = 0
    peak: int = 0

    async def translate(self, *, utterance_id, text, source_language, target_language, **unused):
        self.calls.append(dict(text=text, at_ms=(time.perf_counter() - self.origin) * 1000))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if text in self.gates:
                await self.gates[text].wait()
            self.completed.append(text)
            return Translation(utterance_id=utterance_id, text="Translated " + text.strip(),
                               source_text=text, source_language=source_language,
                               target_language=target_language, channel="peer")
        finally:
            self.active -= 1

    async def close(self):
        pass


@dataclass
class Sink:
    presenter: OverlayPresenter
    origin: float
    events: list = field(default_factory=list)
    first_readable: dict = field(default_factory=dict)

    async def emit(self, event):
        at_ms = (time.perf_counter() - self.origin) * 1000
        self.events.append(dict(type=event.type, text=getattr(event, "text", ""),
                                source_text=getattr(event, "source_text", ""), at_ms=at_ms,
                                assignment=asdict(event.speaker_assignment)
                                if getattr(event, "speaker_assignment", None) else None))
        await self.presenter.emit(event)
        for block in self.presenter.snapshot().blocks:
            if block.id not in self.first_readable:
                self.first_readable[block.id] = dict(at_ms=(time.perf_counter() - self.origin) * 1000,
                                                   text=block.primary_text, style=block.speaker_style)


async def integrated(name, tokens, expected_text, expected_styles, *, translated=False,
                     repeated_context=False, padding=False, successor=False):
    origin = time.perf_counter()
    clock = SystemClock()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    sink = Sink(presenter, origin)
    translator = Translator(origin)
    if translated:
        translator.gates = {item["text"]: asyncio.Event() for item in tokens}
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None, llm=translator if translated else None, osc=osc,
        peer_translation_enabled=translated, overlay_sink=sink, clock=clock,
    )
    harness.output_runtime.activate_peer_generation(1)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    normalized = []
    children = []
    sessions = []
    wires = []
    stages = {}
    original_created = harness.translation_turns.on_child_created

    async def child_created(child):
        children.append(dict(text=child.transcript.text, ranges=child.transcript.source_text_range,
                             speaker_runs=[asdict(run) for run in child.transcript.final_speaker_runs]))
        await original_created(child)

    harness.translation_turns.on_child_created = child_created

    async def factory(_settings, epoch):
        wire = Wire(origin)
        result = session(epoch, wire)
        sessions.append(result)
        wires.append(wire)
        return result

    async def consume(event):
        at_ms = (time.perf_counter() - origin) * 1000
        if isinstance(event, STTProviderTurnUpdate):
            normalized.append(dict(kind="stable", at_ms=at_ms, text=event.text,
                                   speaker_runs=[asdict(run) for run in event.final_speaker_runs]))
            stages.setdefault("first_stable_text_ms", at_ms)
            if any(run.speaker_id for run in event.final_speaker_runs):
                stages.setdefault("first_usable_speaker_ms", at_ms)
        elif isinstance(event, STTProviderTurnTerminal):
            normalized.append(dict(kind="terminal", at_ms=at_ms, outcome=event.outcome,
                                   text=event.text, epoch_disposition=event.epoch_disposition,
                                   speaker_runs=[asdict(run) for run in event.final_speaker_runs]))
            wrong_generation = replace(event.identity.segment, activation_generation=2)
            wrong_capture_epoch = replace(event.identity.segment, capture_epoch=2)
            assert capture.admit_provider_terminal(
                replace(event, identity=replace(event.identity, segment=wrong_generation))) == ()
            assert capture.admit_provider_terminal(
                replace(event, identity=replace(event.identity, segment=wrong_capture_epoch))) == ()
        await callbacks.peer_event_handler(event)
        if isinstance(event, STTProviderTurnTerminal):
            assert capture.admit_provider_terminal(event) == ()

    engine = ScopedRecognitionEngine(session_factory=factory, event_sink=consume)
    def unavailable_capture_operation(*args, **kwargs):
        raise AssertionError("The headless probe must not open a physical capture source")

    capture = PeerCaptureSessionOwner(
        admission=object(), target_resolver=object(), provider=object(), clock=clock,
        provider_request_factory=unavailable_capture_operation,
        source_factory=unavailable_capture_operation, vad_factory=unavailable_capture_operation,
        run_audio_loop=unavailable_capture_operation, vad_sink=engine,
    )
    capture._segment_ledger = ledger
    capture._segment_ledgers.append(ledger)
    capture._publication_generations.add(1)
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    callbacks.bind_peer(harness.peer_owner)
    callbacks.bind_peer_capture(capture)
    await harness.start()
    try:
        segment_id = uuid4()
        content = span(1, 160000, 161600)
        context = span(0, 159200, 160000)
        event = SpeechStart(segment_id, np.ones(800, dtype=np.float32) * 0.1,
                            np.ones(1600, dtype=np.float32) * 0.2,
                            pre_roll_capture=(context,), chunk_capture=(content,))
        owned = ledger.observe_vad_event(event, now_monotonic_s=clock.now())
        source_map = dict(content=[asdict(content)], context=[asdict(context)],
                          coordinate_kind="modeled canonical source, not physical microphone time")
        await engine.handle_owned_vad_event(owned)
        wire, native = wires[0], sessions[0]
        initial_scope = native.speaker_session_scope
        if repeated_context:
            identity = native._event_projection.active_identity
            await native.send_turn_audio(identity, bytes(1600), payload_sequence=3,
                                         source_ranges=(context,), context_only=True)
        wire.inject([token("provisional ignored", "B", 0, 1, False)])
        await wait_for(lambda: len(wire.receives) == 1)
        assert normalized == []
        wire.inject(tokens, request_id="one-stream-request-not-a-turn-key")
        await wait_for(lambda: len([row for row in normalized if row["kind"] == "stable"]) == len(tokens))
        assert sink.events == []
        stages["source_speech_end_modeled_ms"] = 10100
        end_event = SpeechEnd(segment_id, trailing_silence_ms=0,
                              reason="delivery_deadline" if padding else "silence")
        sealed = ledger.observe_vad_event(end_event, now_monotonic_s=clock.now())
        harness.record_peer_speech_end_for_test(segment_id)
        stages["local_seal_ms"] = (time.perf_counter() - origin) * 1000
        end_task = asyncio.create_task(engine.handle_owned_vad_event(sealed))
        await wait_for(lambda: any(row.get("payload") == '{"type": "finalize"}' for row in wire.writes))
        stages["actual_finalize_write_ms"] = next(row["at_ms"] for row in wire.writes
                                                     if row.get("payload") == '{"type": "finalize"}')
        assert native._pending_finalize_requests == 1
        wire.inject([dict(text="<end>", is_final=True)])
        await wait_for(lambda: len(wire.receives) == 3)
        assert native._event_projection.active_identity is not None
        assert not end_task.done() and sink.events == []
        successor_task = None
        if successor:
            next_id = uuid4()
            next_span = span(2, 161600, 163200)
            next_owned = ledger.observe_vad_event(
                SpeechStart(next_id, np.array([], dtype=np.float32),
                            np.ones(1600, dtype=np.float32) * 0.3, chunk_capture=(next_span,)),
                now_monotonic_s=clock.now(),
            )
            before = len(wire.writes)
            stages["successor_input_attempt_ms"] = (time.perf_counter() - origin) * 1000
            successor_task = asyncio.create_task(engine.handle_owned_vad_event(next_owned))
            await asyncio.sleep(0.02)
            assert len(wire.writes) == before and not successor_task.done()
        stages["terminal_fixture_injection_ms"] = (time.perf_counter() - origin) * 1000
        wire.inject([dict(text="<fin>", is_final=True)], request_id="one-stream-request-not-a-turn-key")
        await asyncio.wait_for(end_task, 5)
        if successor_task:
            await asyncio.wait_for(successor_task, 5)
            stages["successor_pcm_write_ms"] = wire.writes[-1]["at_ms"]
            wire.inject([token("same", "A", 100, 100)])
            await wait_for(lambda: len(normalized) > len(tokens) + 1)
            next_end = ledger.observe_vad_event(SpeechEnd(next_id, trailing_silence_ms=0, reason="silence"),
                                              now_monotonic_s=clock.now())
            harness.record_peer_speech_end_for_test(next_id)
            next_end_task = asyncio.create_task(engine.handle_owned_vad_event(next_end))
            await wait_for(lambda: sum(row.get("payload") == '{"type": "finalize"}' for row in wire.writes) == 2)
            wire.inject([dict(text="<fin>", is_final=True)])
            await asyncio.wait_for(next_end_task, 5)
        if translated:
            await wait_for(lambda: len(translator.calls) == 3)
            translator.gates[tokens[-1]["text"]].set()
            await wait_for(lambda: tokens[-1]["text"] in translator.completed)
            assert sink.events == []
            translator.gates[tokens[0]["text"]].set()
            await wait_for(lambda: any(row["type"] == "translation_final" for row in sink.events))
            assert tokens[1]["text"] not in translator.completed
            stages["earlier_publication_with_later_translation_pending_ms"] = sink.events[0]["at_ms"]
            translator.gates[tokens[1]["text"]].set()
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 10)
        await asyncio.wait_for(harness.output_runtime.wait_for_peer_output_idle(), 10)
        final_events = [row for row in sink.events if row["type"] in {"peer_transcript_final", "translation_final"}]
        published = "".join(row["source_text"] if translated else row["text"] for row in final_events)
        expected = expected_text + ("same" if successor else "")
        assert published == expected, (name, published, expected)
        styles = [row["style"] for row in sink.first_readable.values()]
        assert styles == expected_styles, (name, styles, expected_styles)
        first_terminal = next(row for row in normalized if row["kind"] == "terminal")
        assert first_terminal["text"] == expected_text
        assert "".join(run["text"] for run in first_terminal["speaker_runs"]) == expected_text
        assert "".join(child["text"] for child in children) == expected
        expected_times = {
            "A_B_A": [(100, 190), (200, 290), (300, 390)],
            "A_missing_A": [(100, 190), (200, 290), (300, 390)],
            "legitimate_repeated_text_and_reuse": [(100, 100)],
            "overlap_and_zero_duration": [(100, 250), (200, 260), (260, 260)],
            "repeated_context_and_padding": [(9000, 9090)],
            "boundary_whitespace": [(100, 190), (200, 290)],
            "translated_out_of_order": [(100, 190), (200, 290), (300, 390)],
        }[name]
        assert [(run["source_start_ms"], run["source_end_ms"])
                for run in first_terminal["speaker_runs"]] == expected_times
        assert [(child["speaker_runs"][0]["source_start_ms"], child["speaker_runs"][0]["source_end_ms"])
                for child in children[:len(expected_times)]] == expected_times
        assert all(run["session_scope"] == initial_scope
                   for row in normalized for run in row["speaker_runs"])
        if name == "overlap_and_zero_duration":
            assert [run["overlaps_previous"] for run in first_terminal["speaker_runs"]] == [False, True, False]
        if repeated_context:
            assert first_terminal["speaker_runs"][0]["overlaps_previous"] is True
        stages["first_terminal_receipt_ms"] = first_terminal["at_ms"]
        stages["first_readable_publication_ms"] = min(row["at_ms"] for row in sink.first_readable.values())
        stages["finalize_to_terminal_application_ms"] = first_terminal["at_ms"] - stages["actual_finalize_write_ms"]
        stages["terminal_to_first_readable_application_ms"] = stages["first_readable_publication_ms"] - first_terminal["at_ms"]
        for child in children:
            assert "".join(run["text"] for run in child["speaker_runs"]) == child["text"]
        assert native._pending_tokens == [] and native._scoped_tokens == []
        assert native._pending_finalize_requests == 0 and native._pending_last_end_ms is None
        assert native._event_projection.active_identity is None
        assert len(sessions) == 1
        if padding:
            finalize_index = next(i for i, row in enumerate(wire.writes) if row.get("payload") == '{"type": "finalize"}')
            assert wire.writes[finalize_index - 1]["bytes"] == 6400
            assert wire.writes[finalize_index - 1]["all_zero"]
        pcm_bytes = sum(row.get("bytes", 0) for row in wire.writes)
        pieces = [
            dict(provider_start_sample=0, provider_end_sample=800, kind="context",
                 source_start_sample=159200, source_end_sample=160000),
            dict(provider_start_sample=800, provider_end_sample=2400, kind="content",
                 source_start_sample=160000, source_end_sample=161600),
        ]
        stream_end = 2400
        if repeated_context:
            pieces.append(dict(provider_start_sample=stream_end, provider_end_sample=stream_end + 800,
                               kind="replayed_context", source_start_sample=159200, source_end_sample=160000))
            stream_end += 800
        if padding:
            pieces.append(dict(provider_start_sample=stream_end, provider_end_sample=stream_end + 3200,
                               kind="S200", source_start_sample=None, source_end_sample=None))
            stream_end += 3200
        if successor:
            pieces.append(dict(provider_start_sample=stream_end, provider_end_sample=stream_end + 1600,
                               kind="successor_content", source_start_sample=161600, source_end_sample=163200))
        assert pieces[-1]["provider_end_sample"] * 2 == pcm_bytes
        return dict(name=name, status="passed", text=published, expected_text=expected,
                    source_map=source_map, provider_sample_map=pieces, stages=stages,
                    normalized=normalized, children=children,
                    first_readable=list(sink.first_readable.values()), output_events=final_events,
                    wire_writes=wire.writes, native_receives=wire.receives,
                    resources=dict(transmitted_pcm_seconds=pcm_bytes / 32000, provider_calls=0,
                                   local_transport_doubles=1, translation_double_calls=len(translator.calls),
                                   translation_double_peak=translator.peak,
                                   audio_queue_capacity=native._audio_q.maxsize,
                                   engine_retention=asdict(engine.retention_snapshot)),
                    findings=dict(text_conserved=True, pending_fin_blocks_successor=successor,
                                  first_readable_styles=styles, native_timestamps_are_literal=True))
    finally:
        for gate in translator.gates.values():
            gate.set()
        await engine.close()
        await harness.stop()
        await presenter.close()


async def lifecycle(mode):
    origin = time.perf_counter()
    wire = Wire(origin)
    native = session("fence-epoch", wire)
    from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
    identity = STTProviderTurnIdentity(AudioSegmentIdentity(1, 1, uuid4(), 1), "fence-epoch", "first")
    request = STTProviderTurnRequest(identity, settings(), "peer")
    try:
        await native.begin_turn(request)
        if mode == "source_discontinuity":
            await native.abort_turn(identity, reason="source_discontinuity")
        else:
            await native.seal_turn(identity, sealed_content_ranges=(), seal_reason="silence",
                                   observed_trailing_silence_ms=0)
            if mode == "cancelled":
                await native.abort_turn(identity, reason="cancelled")
            elif mode == "failed":
                native._handle_message(json.dumps(dict(error="fixture failure")))
            else:
                native._handle_message(json.dumps(dict(tokens=[dict(text="<fin>", is_final=True)])))
        stream = native.turn_events()
        terminal = await asyncio.wait_for(anext(stream), 1)
        assert isinstance(terminal, STTProviderTurnTerminal)
        if mode in {"empty", "late_idle_text", "late_idle_fin", "late_after_successor_active"}:
            assert terminal.outcome == "empty" and terminal.epoch_disposition == "reuse"
        else:
            assert terminal.outcome == ("failed" if mode == "failed" else "cancelled")
            assert terminal.epoch_disposition == "retire"
        if mode in {"empty", "late_after_successor_active"}:
            successor_id = replace(identity, segment=replace(identity.segment, segment_order=2, segment_id=uuid4()),
                                   provider_turn_id="second")
            await native.begin_turn(replace(request, identity=successor_id))
            injected_text = "unkeyed late old text" if mode == "late_after_successor_active" else "fresh"
            native._handle_message(json.dumps(dict(tokens=[token(injected_text, "A", 0, 0)],
                                                   request_id="same-stream-request")))
            fresh = await asyncio.wait_for(anext(stream), 1)
            assert isinstance(fresh, STTProviderTurnUpdate) and fresh.text == injected_text
            assert fresh.identity == successor_id
            await native.abort_turn(successor_id, reason="probe_cleanup")
            return dict(name=mode, status="passed", terminal=asdict(terminal),
                        successor_text=fresh.text,
                        modeled_ambiguity=mode == "late_after_successor_active",
                        conclusion="Unkeyed native text binds to active turn; no per-turn correlation guarantee"
                        if mode == "late_after_successor_active" else "Empty completion clears before successor")
        if mode.startswith("late_idle"):
            late = [token("late", "A", 0, 0)] if mode == "late_idle_text" else [dict(text="<fin>", is_final=True)]
            native._handle_message(json.dumps(dict(tokens=late)))
        ended = await asyncio.wait_for(anext(stream), 1)
        assert isinstance(ended, STTProviderEpochEnded)
        depth = native._event_projection.scoped_event_depth
        native._handle_message(json.dumps(dict(tokens=[token("late leaked text", "A", 0, 0), dict(text="<fin>", is_final=True)])))
        assert native._event_projection.scoped_event_depth == depth
        try:
            await native.begin_turn(request)
        except RuntimeError:
            pass
        else:
            raise AssertionError("retired epoch admitted new work")
        replacement = session("replacement-epoch", Wire(origin))
        try:
            assert replacement.speaker_session_scope != native.speaker_session_scope
            wrong = replace(identity, provider_epoch_id="replacement-epoch")
            normalizer = STTScopedTurnNormalizer(wrong)
            try:
                normalizer.apply_terminal(terminal)
            except STTNormalizationError as exc:
                assert exc.reason == "provider_turn_identity_mismatch"
            else:
                raise AssertionError("cross-epoch terminal accepted")
        finally:
            await replacement.close()
        return dict(name=mode, status="passed", terminal=asdict(terminal), epoch_end=asdict(ended),
                    late_output_suppressed=True, true_replacement_distinct_scope=True,
                    cross_epoch_terminal_rejected=True)
    finally:
        await native.close()


async def immutable_and_scope():
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=SystemClock())
    adapter = OverlayEventAdapter(clock=SystemClock())
    turn = uuid4()
    unknown = SpeakerAssignment(SpeakerAttribution("missing", source="soniox", speaker_scope_id="old"), (1, 1, 0))
    identified = SpeakerAssignment(SpeakerAttribution("identified", SpeakerKey("soniox", "old", "A"), "soniox", "old"), (1, 1, 0), 0)
    def event(turn_id, text, assignment):
        return adapter.translation_final(utterance_id=turn_id, channel="peer", text=text,
                                         source_text=text, source_language="en", target_language="ko",
                                         applied_context_mode="integrated", speaker_assignment=assignment)
    try:
        await presenter.emit(event(turn, "readable without ID", unknown))
        first = presenter._entries[("peer", turn)]
        visible = first.visible_since
        assert first.speaker_style == "gray"
        await presenter.emit(event(turn, "late usable ID", identified))
        assert first.speaker_style == "gray" and first.visible_since == visible
        old_turn = uuid4()
        await presenter.emit(event(old_turn, "old identified", replace(identified, scope_order=(1, 2, 0))))
        assert presenter._entries[("peer", old_turn)].speaker_style == "gold"
        new_turn = uuid4()
        new = replace(identified, attribution=SpeakerAttribution("identified", SpeakerKey("soniox", "new", "A"), "soniox", "new"), scope_order=(1, 3, 0))
        await presenter.emit(event(new_turn, "same opaque ID new connection", new))
        assert presenter._entries[("peer", new_turn)].speaker_style == "gray"
        assert presenter._entries[("peer", old_turn)].speaker_style == "gold"
        return dict(name="immutable_color_and_scope_handoff", status="passed",
                    late_identified_style="gray", readable_lifetime_unchanged=True,
                    old_scope_style="gold", new_scope_style="gray", identical_local_id_is_not_same_key=True)
    finally:
        await presenter.close()


async def engine_source_fence():
    origin = time.perf_counter()
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    observed = []
    sessions = []
    wires = []

    async def factory(_settings, epoch):
        wire = Wire(origin)
        native = session(epoch, wire)
        sessions.append(native)
        wires.append(wire)
        return native

    engine = ScopedRecognitionEngine(session_factory=factory, event_sink=observed.append)
    try:
        first_id = uuid4()
        first = ledger.observe_vad_event(
            SpeechStart(first_id, np.array([], dtype=np.float32), np.ones(1600, dtype=np.float32) * 0.1,
                        chunk_capture=(span(1, 160000, 161600),)),
            now_monotonic_s=time.monotonic(),
        )
        await engine.handle_owned_vad_event(first)
        wires[0].inject([token("before loss", "A", 0, 100)])
        await wait_for(lambda: any(isinstance(row, STTProviderTurnUpdate) for row in observed))
        source_loss = ledger.terminalize_open_for_source_loss(now_monotonic_s=time.monotonic())
        assert source_loss.outcome == "failed" and source_loss.segment.seal_reason == "source_discontinuity"
        await engine.abort(reason="source_discontinuity")
        cancellation = next(row for row in observed if isinstance(row, STTProviderTurnTerminal))
        assert cancellation.outcome == "cancelled" and cancellation.text == ""
        assert engine.retention_snapshot.retained_samples == 0
        old_count = len(observed)
        sessions[0]._handle_message(json.dumps(dict(tokens=[token("late old scope", "A", 0, 100)])))
        await asyncio.sleep(0)
        assert len(observed) == old_count
        second_id = uuid4()
        second = ledger.observe_vad_event(
            SpeechStart(second_id, np.array([], dtype=np.float32), np.ones(1600, dtype=np.float32) * 0.2,
                        chunk_capture=(span(2, 163200, 164800, epoch=2),)),
            now_monotonic_s=time.monotonic(),
        )
        await engine.handle_owned_vad_event(second)
        assert len(sessions) == 2
        assert sessions[0].speaker_session_scope != sessions[1].speaker_session_scope
        wires[1].inject([token("after loss", "A", 0, 100)])
        await wait_for(lambda: any(isinstance(row, STTProviderTurnUpdate) and row.text == "after loss"
                                  for row in observed))
        latest = next(row for row in reversed(observed) if isinstance(row, STTProviderTurnUpdate))
        assert latest.identity.segment.capture_epoch == 2
        assert latest.final_speaker_runs[0].session_scope == sessions[1].speaker_session_scope
        assert latest.identity.provider_epoch_id != cancellation.identity.provider_epoch_id
        ledger.terminalize_open_for_source_loss(now_monotonic_s=time.monotonic())
        await engine.abort(reason="probe_cleanup")
        return dict(name="engine_source_discontinuity_generation_fence", status="passed",
                    source_loss_outcome=source_loss.outcome, cancellation=asdict(cancellation),
                    retained_after_abort=0, successor_text=latest.text,
                    successor_capture_epoch=2, late_old_output_suppressed=True,
                    fresh_provider_epoch=True, fresh_speaker_scope=True)
    finally:
        await engine.close()


async def run():
    tracemalloc.start()
    origin = time.perf_counter()
    fixtures = [
        ("A_B_A", [token("Alice ", "A", 100, 190), token("Beto ", "B", 200, 290), token("Alice", "A", 300, 390)], "Alice Beto Alice", ["gold", "cyan", "gold"], {}),
        ("A_missing_A", [token("one ", "A", 100, 190), token("unknown ", None, 200, 290), token("again", "A", 300, 390)], "one unknown again", ["gold", "gray", "gold"], {}),
        ("legitimate_repeated_text_and_reuse", [token("same", "A", 100, 100)], "same", ["gold", "gold"], dict(successor=True)),
        ("overlap_and_zero_duration", [token("first ", "A", 100, 250), token("brief ", "B", 200, 260), token("yes", "A", 260, 260)], "first brief yes", ["gold", "cyan", "gold"], {}),
        ("repeated_context_and_padding", [token("echo ", "A", 9000, 9090), token("echo", "A", 9000, 9090)], "echo echo", ["gold"], dict(repeated_context=True, padding=True)),
        ("boundary_whitespace", [token("  alpha ", "A", 100, 190), token("beta  ", "B", 200, 290)], "alpha beta", ["gold", "cyan"], {}),
        ("translated_out_of_order", [token("Alice ", "A", 100, 190), token("Beto ", "B", 200, 290), token("Alice", "A", 300, 390)], "Alice Beto Alice", ["gold", "cyan", "gold"], dict(translated=True)),
    ]
    rows = []
    for name, tokens, text, styles, options in fixtures:
        rows.append(await integrated(name, tokens, text, styles, **options))
    fences = [await lifecycle(mode) for mode in ["empty", "failed", "cancelled", "source_discontinuity", "late_idle_text", "late_idle_fin", "late_after_successor_active"]]
    fences.append(await immutable_and_scope())
    fences.append(await engine_source_fence())
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    source_paths = [
        "src/puripuly_heart/providers/stt/soniox.py",
        "src/puripuly_heart/core/stt/scoped_engine.py",
        "src/puripuly_heart/core/stt/scoped_normalizer.py",
        "src/puripuly_heart/core/orchestrator/peer_translation_channel.py",
        "src/puripuly_heart/core/runtime/peer_channel.py",
        "src/puripuly_heart/core/orchestrator/translation_channel_callbacks.py",
        "src/puripuly_heart/core/orchestrator/translation_turn.py",
        "src/puripuly_heart/core/speaker_identity.py",
        "src/puripuly_heart/core/runtime/output.py",
        "src/puripuly_heart/core/overlay/presenter.py",
        "tests/helpers/translation_owners.py",
        "experiments/ami_benchmark/providers/soniox/gap/consumption_probe.py",
    ]
    return dict(schema_version=1, baseline_revision=BASELINE,
                baseline_authority="Director-supplied clean baseline; no Git inspection performed",
                command="python experiments/ami_benchmark/providers/soniox/gap/consumption_probe.py --output experiments/ami_benchmark/providers/soniox/gap/consumption_evidence.json",
                environment=dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__),
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
                fixture_input_kind="Synthetic PCM and scripted native tokens; no acoustic inference, no provider network calls",
                measurement_kind="Measured monotonic application execution around modeled source/token coordinates; injected receive policy, no microphone/VRChat/HMD",
                passed_checks=len(rows) + len(fences), fixtures=rows, fences=fences,
                resources=dict(provider_calls=0, actual_websocket_duration_s=None,
                               whole_probe_wall_s=time.perf_counter() - origin,
                               python_tracemalloc_current_bytes=current, python_tracemalloc_peak_bytes=peak,
                               allocation_scope="whole probe after imports including evidence capture, not isolated product RSS"),
                unavailable=dict(provider_stable_arrival_latency="no authorized live call or retained arrival trace",
                                 source_speech_end_to_publication="source speech is modeled, not recorded physical timing",
                                 provider_processing_latency="scripted receive, no service processing",
                                 active_websocket_duration="in-memory transport double, no socket",
                                 quality_der_wer="scripted words and speaker IDs cannot measure acoustic quality",
                                 live_pcm_memory_backlog="finite fixture workload only; no sustained live load",
                                 cost_invoice="zero paid calls; vendor billing/rate/quota not measured",
                                 continuous_candidate="not run: causal live comparison and supported ownership amendment unavailable"),
                disposition="defer; current serial finalize consumer verified, no supported quality-latency production candidate")


def main():
    parser = argparse.ArgumentParser(description="Model-free Soniox application-consumption probe; never opens a provider connection")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run())
    encoded = json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    print(json.dumps(dict(passed_checks=result["passed_checks"], disposition=result["disposition"])), file=sys.stderr)


if __name__ == "__main__":
    main()
