from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import platform
import socket
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
NETWORK_ATTEMPTS = []


def deny_network(*args, **kwargs):
    NETWORK_ATTEMPTS.append("socket connection blocked")
    raise RuntimeError("Offline probe forbids network connections")


import numpy as np
from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity, AudioCaptureSpan
from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.gpu_worker import GpuWorkerTranscription
from puripuly_heart.core.stt.backend import STTContributionConsumptionLedger, STTProviderInputTerminal, STTProviderTurnIdentity, STTProviderTurnRequest, STTProviderTurnTerminal, STTProviderTurnUpdate, STTRecognitionUnit, STTSessionProjection
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.stt.stream_input import STTStreamInputMap
from puripuly_heart.core.vad.gating import SpeechStart, SpeechEnd, create_peer_vad_gating
from puripuly_heart.providers.stt.custom import _OfflineOpenAITranscriptionSession, _StreamingOpenAIRealtimeSession
from puripuly_heart.providers.stt.deepgram import _DeepgramSDKSession, _FINALIZE, _CLOSE_STREAM
from puripuly_heart.providers.stt.elevenlabs_scribe import _ElevenLabsScribeSession
from puripuly_heart.providers.stt.gemini_transcribe import _GeminiTranscribeLiveSession
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend, _LocalGpuSTTSession
from puripuly_heart.providers.stt.local_parakeet_sherpa import LocalParakeetJapaneseSherpaSTTBackend, LocalParakeetV3SherpaSTTBackend
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend, _LocalQwenSherpaSession
from puripuly_heart.providers.stt.qwen_audio import _QwenAudioSession, QwenAudioSessionState
from puripuly_heart.providers.stt.soniox import _SonioxSession
from puripuly_heart.providers.stt.local_cpu import LocalCPUAutoSTTBackend
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
from puripuly_heart.config.provider_values import STTProviderName

UUID_ALIASES = {}


def encode(value):
    if isinstance(value, UUID):
        return UUID_ALIASES.setdefault(value, f"fixture-uuid-{len(UUID_ALIASES) + 1}")
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"samples": int(value.size), "dtype": str(value.dtype)}
    if is_dataclass(value):
        populated = {f.name: getattr(value, f.name) for f in fields(value)}
        return {"type": type(value).__name__, **{key: encode(item) for key, item in populated.items() if item is not None and not (isinstance(item, (tuple, list)) and not item)}}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def settings(provider="fixture", hangover=128):
    return AudioSegmentSettingsSnapshot(provider, (provider,), (provider,), "manual", "en", ("en",), 16000, 0.4, hangover, 500)


def request(provider, order=1):
    return STTProviderTurnRequest(STTProviderTurnIdentity(AudioSegmentIdentity(7, order, UUID(int=order), 3), "fixture-epoch", f"turn-{order}", (provider,)), settings(provider))


def span(start=0, end=5120, sequence=1, epoch=3, loss=None):
    return AudioCaptureSpan(epoch, sequence, 16000, start, end, start / 16000, end / 16000, loss, 16000, start, end)


class Wire:
    def __init__(self):
        self.sent = []
        self.handlers = {}
        self.events = asyncio.Queue()
        self.received = 0
        self.closed = False

    async def send(self, value):
        self.sent.append(json.loads(value) if isinstance(value, str) and value else value)

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)

    async def commit(self):
        self.sent.append({"commit": True})

    def on(self, event, callback):
        self.handlers[str(getattr(event, "value", event))] = callback

    def emit(self, event):
        name = event["message_type"]
        handler = self.handlers.get(name)
        if handler:
            handler(event)
        return handler is not None

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.events.get()
        self.received += 1
        return json.dumps(event)

    async def close(self):
        self.closed = True


class DeepgramWireSession(_DeepgramSDKSession):
    async def _write_thread_payload(self, payload):
        self.wire.sent.append({"type": "Finalize"} if payload is _FINALIZE else {"type": "CloseStream"} if payload is _CLOSE_STREAM else payload)


class HttpWire:
    def __init__(self, result):
        self.result = result
        self.sent = []

    async def post(self, url, **kwargs):
        self.sent.append({"url": url, "data": kwargs["data"], "wav_bytes": len(kwargs["files"]["file"][1])})
        return SimpleNamespace(status_code=200, json=lambda: self.result)

    async def aclose(self):
        pass


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def drain(session):
    events = []
    while session._event_projection.scoped_event_depth:
        events.append(await asyncio.wait_for(anext(session.turn_events()), 2))
    return events


async def make_session(case):
    projection = STTSessionProjection(mode="scoped", provider_epoch_id="fixture-epoch")
    route = case["route"]
    wire = Wire()
    if route == "soniox":
        session = _SonioxSession("offline", "stt-rt-v4", "wss://offline.invalid", 16000, [], [], 60, 0, 1, enable_speaker_diarization=case.get("speaker_enabled", True), projection=projection)
        session.speaker_session_scope = "fixture-soniox-native-scope"
        session._ws = wire
        session._send_task = asyncio.create_task(session._send_loop())
    elif route == "deepgram":
        session = DeepgramWireSession("offline", "nova-3", "en", 16000, 1, [], drain_timeout_s=60, projection=projection)
        session.wire = wire
        session._loop = asyncio.get_running_loop()
    elif route == "qwen_audio":
        session = _QwenAudioSession("offline", (), "qwen3-asr-flash", "wss://offline.invalid", 16000, 1, 1, 60, 1, 60, 0, projection=projection)
        session._ws = wire
        session._task_id = "fixture-task"
        session._state = QwenAudioSessionState.TASK_ACTIVE
    elif route == "elevenlabs":
        async def connect(options):
            wire.options = repr(options)
            return wire
        session = _ElevenLabsScribeSession("offline", "en", (), "scribe_v2_realtime", 16000, 1, keepalive_interval_s=60, scribe_connect_factory=connect, projection=projection)
        await session.start()
    elif route == "gemini":
        session = _GeminiTranscribeLiveSession("offline", [], [], "gemini-3.5-transcribe-live", 16000, 1, 0.01, projection=projection)
        session._live_session = wire
        session._send_task = asyncio.create_task(session._send_loop())
    elif route == "custom_offline":
        wire = HttpWire(case["result"])
        session = _OfflineOpenAITranscriptionSession("https://offline.invalid/v1/audio/transcriptions", "fixture-model", "offline", "en", 16000, lambda **kwargs: wire, projection=projection)
        await session.start()
    elif route == "custom_stream":
        session = _StreamingOpenAIRealtimeSession("wss://offline.invalid/v1/realtime", "fixture-model", "offline", "en", 16000, projection=projection)
        session._ws = wire
        session._recv_task = asyncio.create_task(session._receive_loop())
    else:
        raise ValueError(route)
    return session, wire


async def adapter_case(case):
    session, wire = await make_session(case)
    route = case["route"]
    req = request(route)
    events = []
    normalized = []
    consumed = []
    replay_consumed = []
    action_notes = []
    normalizers = {}
    ledger = STTContributionConsumptionLedger()
    count = case.get("audio_samples", 5120)
    ranges = (span(end=count),) if count else ()

    async def collect(action_index):
        incoming = await drain(session)
        for event in incoming:
            events.append({"after_action": action_index, "event": encode(event)})
            if isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal)):
                normalizer = normalizers.setdefault(event.identity, STTScopedTurnNormalizer(event.identity))
                result = normalizer.apply_update(event) if isinstance(event, STTProviderTurnUpdate) else normalizer.apply_terminal(event)
                if result is not None:
                    normalized.append(encode(result))
                    consumed.append(ledger.consume(result))
                    replay_consumed.append(ledger.consume(result))

    try:
        await session.begin_turn(req)
        if count:
            await session.send_turn_audio(req.identity, bytes(count * 2), payload_sequence=1, source_ranges=ranges, context_only=False)
        for index, action in enumerate(case["actions"]):
            if "begin" in action:
                req = request(route, action["begin"])
                await session.begin_turn(req)
            elif "audio" in action:
                ranges = (span(start=count, end=count * 2, sequence=2),)
                await session.send_turn_audio(req.identity, bytes(count * 2), payload_sequence=1, source_ranges=ranges, context_only=False)
            elif "seal" in action:
                await session.seal_turn(req.identity, sealed_content_ranges=ranges, seal_reason="silence", observed_trailing_silence_ms=128)
                if route == "custom_offline":
                    await session._scoped_task
            elif "finish" in action:
                if session.state is QwenAudioSessionState.FINISHING_TASK:
                    session._closing_requested = True
                await session._handle_server_message({"header": {"event": "task-finished", "task_id": "fixture-task"}})
            elif "event" in action:
                event = action["event"]
                if route == "soniox":
                    session._handle_message(json.dumps(event))
                elif route == "deepgram":
                    from deepgram.extensions.types.sockets import ListenV1ResultsEvent
                    shape = {"type": "Results", "channel_index": [0, 1], "duration": event.get("duration", 0.1), "start": event.get("start", 0), "is_final": event.get("is_final", False), "speech_final": False, "from_finalize": event.get("from_finalize", False), "channel": {"alternatives": [{"transcript": event.get("text", ""), "confidence": 1, "words": event.get("words", [])}]}, "metadata": {"request_id": "fixture-request", "model_info": {"name": "nova-3", "version": "fixture", "arch": "nova"}, "model_uuid": "fixture-model"}}
                    session._build_transcript_event(ListenV1ResultsEvent(**shape))
                elif route == "qwen_audio":
                    sentence = {"sentence_id": event["id"], "text": event["text"], "sentence_end": event["end"], "words": event.get("words", [])}
                    await session._handle_server_message({"header": {"event": "result-generated", "task_id": event.get("task_id", "fixture-task")}, "payload": {"output": {"sentence": sentence}}})
                elif route == "elevenlabs":
                    action_notes.append({"action": index, "registered_handler": wire.emit(event), "message_type": event["message_type"]})
                elif route == "gemini":
                    from google.genai import types
                    transcription = types.Transcription(text=event["final"], finished=event.get("finished"), words=event.get("words")) if "final" in event else None
                    content = types.LiveServerContent(input_transcription=transcription, interim_input_transcription=types.Transcription(text=event["interim"]) if "interim" in event else None)
                    ack = types.VoiceActivity(voice_activity_type=types.VoiceActivityType.ACTIVITY_END) if event.get("ack") else None
                    session._handle_message(types.LiveServerMessage(server_content=content, voice_activity=ack))
                elif route == "custom_stream":
                    wire.events.put_nowait(event)
                    await settle()
            await settle()
            await collect(index)
        summary = summarize([row["event"] for row in events], normalized, consumed)
        summary["retired"] = session._event_projection.retired
        summary["wire_count"] = len(wire.sent)
        summary["replay_consumed"] = "".join(replay_consumed)
        checks = {key: summary[key] == expected for key, expected in case["expect"].items()}
        checks["same_projected_contribution_replay_empty"] = summary["replay_consumed"] == ""
        return {"id": case["id"], "route": route, "status": "passed" if all(checks.values()) else "failed", "checks": checks, "summary": summary, "wire": encode(wire.sent), "events": events, "normalized": normalized, "consumed_pieces": consumed, "action_notes": action_notes}
    finally:
        for attr in ("_send_task", "_recv_task", "_queue_task", "_keepalive_task", "_scoped_drain_task", "_finish_timeout_task", "_start_timeout_task", "_scoped_final_timeout_task"):
            task = getattr(session, attr, None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        session._event_projection.close()


def summarize(events, normalized, consumed):
    terminals = [e for e in events if e["type"] == "STTProviderTurnTerminal"]
    return {"terminal_texts": [e["text"] for e in terminals], "outcomes": [e["outcome"] for e in terminals], "recognition_texts": [e["text"] for e in events if e["type"] == "STTRecognitionUnit"], "input_outcomes": [e["outcome"] for e in events if e["type"] == "STTProviderInputTerminal"], "stable_count": sum(e["type"] == "STTProviderTurnUpdate" and e["stability"] == "stable" for e in events), "provisional_count": sum(e["type"] == "STTProviderTurnUpdate" and e["stability"] == "provisional" for e in events), "speaker_runs": sum(len(e.get("final_speaker_runs", [])) for e in terminals), "consumed": "".join(consumed)}


def normalizer_case():
    identity = request("normalizer").identity
    from puripuly_heart.core.stt.backend import STTNativeProvenance
    normalizer = STTScopedTurnNormalizer(identity)
    ledger = STTContributionConsumptionLedger()
    incoming = [
        STTProviderTurnUpdate(identity, 1, "provisional", "replace", "four"),
        STTProviderTurnUpdate(identity, 2, "provisional", "replace", "no"),
        STTProviderTurnUpdate(identity, 3, "stable", "append", "no", provenance=STTNativeProvenance(native_event_id="speech-1")),
        STTProviderTurnUpdate(identity, 4, "stable", "append", " no", provenance=STTNativeProvenance(native_event_id="speech-2")),
        STTProviderTurnUpdate(identity, 5, "stable", "append", " no", provenance=STTNativeProvenance(native_event_id="speech-2")),
    ]
    outgoing = [normalizer.apply_update(event) for event in incoming]
    pieces = [ledger.consume(event) for event in outgoing if event is not None]
    terminal = normalizer.apply_terminal(STTProviderTurnTerminal(identity, "final", "no no three", text_authority="authoritative"))
    tail = ledger.consume(terminal)
    replay = ledger.consume(terminal)
    inconsistent = STTScopedTurnNormalizer(identity)
    inconsistent.apply_update(STTProviderTurnUpdate(identity, 1, "stable", "replace", "four"))
    error = None
    try:
        inconsistent.apply_terminal(STTProviderTurnTerminal(identity, "final", "no", text_authority="authoritative"))
    except RuntimeError as exc:
        error = str(exc)
    checks = {"mutable_partial_not_consumed": pieces[:2] == ["", ""], "repeated_true_speech_preserved": "".join(pieces) == "no no", "native_duplicate_suppressed": outgoing[-1] is None, "terminal_only_tail_preserved_once": tail == " three" and replay == "", "stable_correction_rejected": error == "provider_stable_prefix_inconsistent"}
    return {"id": "existing-normalizer-terminal-only-tail", "status": "passed" if all(checks.values()) else "failed", "input": encode(incoming), "projected_updates": encode(outgoing), "terminal": encode(terminal), "consumed_pieces": pieces, "terminal_tail": tail, "terminal_replay": replay, "stable_correction_error": error, "checks": checks}


class Recognizer:
    def __init__(self, result):
        self.result = result
        self.accepted_samples = []
        self.decodes = 0

    def create_stream(self):
        owner = self
        class Stream:
            result = owner.result
            def accept_waveform(self, rate, samples):
                owner.accepted_samples.append(int(samples.size))
        return Stream()

    def decode_stream(self, stream):
        self.decodes += 1


class GpuWire:
    def __init__(self):
        self.calls = []

    async def submit_pcm16(self, channel, pcm, **kwargs):
        self.calls.append({"channel": channel, "pcm_bytes": len(pcm), "language_hint": kwargs["language_hint"]})
        return GpuWorkerTranscription("no three", "en", len(pcm) / 32000, 0, 0)


def pcm_summary(samples):
    values, counts = np.unique(samples, return_counts=True)
    return {"sample_count": int(samples.size), "values_and_counts": [[round(float(value), 7), int(count)] for value, count in zip(values, counts)]}


class ContentRecognizer:
    def __init__(self):
        self.calls = []

    def create_stream(self):
        class Stream:
            result = SimpleNamespace(text="")
            def accept_waveform(self, rate, samples):
                self.samples = samples.copy()
        return Stream()

    def decode_stream(self, stream):
        self.calls.append(pcm_summary(stream.samples))
        stream.result.text = "again new" if np.max(stream.samples) > 0.05 else "again"


class ContentHttpWire(HttpWire):
    async def post(self, url, **kwargs):
        import io
        import wave
        with wave.open(io.BytesIO(kwargs["files"]["file"][1]), "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        self.sent.append(pcm_summary(samples))
        text = "again new" if np.max(samples) > 1500 else "again"
        return SimpleNamespace(status_code=200, json=lambda: {"text": text})


class ContentGpuWire(GpuWire):
    async def submit_pcm16(self, channel, pcm, **kwargs):
        samples = np.frombuffer(pcm, dtype="<i2")
        self.calls.append(pcm_summary(samples))
        return GpuWorkerTranscription("again new" if np.max(samples) > 1500 else "again", None, len(samples) / 16000, 0, 0)


async def context_cases():
    output = []
    for route in ("local_qwen", "local_parakeet_v3", "local_parakeet_ja", "local_gpu", "custom_offline", "cpu_auto"):
        projection = STTSessionProjection("scoped", "fixture-epoch")
        backend = None
        if route in ("local_qwen", "local_parakeet_v3", "local_parakeet_ja", "cpu_auto"):
            cls = {"local_qwen": LocalQwenSherpaSTTBackend, "local_parakeet_v3": LocalParakeetV3SherpaSTTBackend, "local_parakeet_ja": LocalParakeetJapaneseSherpaSTTBackend, "cpu_auto": LocalQwenSherpaSTTBackend}[route]
            delegate = cls(Path("offline-model-not-loaded"))
            wire = ContentRecognizer()
            delegate._recognizer = wire
            if route == "cpu_auto":
                backend = LocalCPUAutoSTTBackend("en")
                backend._delegate = delegate
                backend._resolved_model_id = delegate.model_id
                session = await backend.open_session(projection=projection)
            else:
                backend = delegate
                session = await backend.open_session(projection=projection)
            calls = wire.calls
        elif route == "local_gpu":
            wire = ContentGpuWire()
            backend = LocalGpuSTTBackend(wire, "peer", Path("offline-model-not-loaded"), "fixture-gpu-model", "no-device")
            session = _LocalGpuSTTSession(backend, projection)
            calls = wire.calls
        else:
            wire = ContentHttpWire({})
            session = _OfflineOpenAITranscriptionSession("https://offline.invalid/v1/audio/transcriptions", "fixture-model", "offline", "en", 16000, lambda **kwargs: wire, projection=projection)
            await session.start()
            calls = wire.sent
        old = np.full(4800, 1000, dtype="<i2").tobytes()
        new = np.full(1600, 2000, dtype="<i2").tobytes()
        terminals = []
        payloads = []
        for order in (1, 2):
            req = request(route, order)
            await session.begin_turn(req)
            parts = [(old, (span(0, 4800),), order == 2)]
            if order == 2:
                parts.append((new, (span(4800, 6400, 2),), False))
            for sequence, (pcm, ranges, context_only) in enumerate(parts, 1):
                payloads.append({"turn": order, "payload_sequence": sequence, "pcm": encode(pcm), "source_ranges": encode(ranges), "context_only": context_only})
                await session.send_turn_audio(req.identity, pcm, payload_sequence=sequence, source_ranges=ranges, context_only=context_only)
            content_ranges = (span(0, 4800),) if order == 1 else (span(4800, 6400, 2),)
            await session.seal_turn(req.identity, sealed_content_ranges=content_ranges, seal_reason="delivery_deadline" if order == 1 else "source_eof", observed_trailing_silence_ms=0)
            if route == "custom_offline":
                await session._scoped_task
            terminals.append(await asyncio.wait_for(anext(session.turn_events()), 5))
        ledger = STTContributionConsumptionLedger()
        consumed = [ledger.consume(event) for event in terminals]
        checks = {"whole_buffer_includes_context_only": [call["sample_count"] for call in calls] == [4800, 6400], "declared_content_range_does_not_trim_decode": terminals[1].text == "again new", "prior_content_can_reappear_in_distinct_terminal": [event.text for event in terminals] == ["again", "again new"] and consumed == ["again", "again new"]}
        if route == "cpu_auto":
            checks["actual_auto_returns_resolved_delegate_session"] = session.backend is delegate and backend.resolved_model_id == delegate.model_id
        output.append({"id": route + "-rollover-context-buffer", "status": "passed" if all(checks.values()) else "failed", "payloads": payloads, "declared_second_sealed_content": encode((span(4800, 6400, 2),)), "decode_inputs": calls, "terminals": encode(terminals), "consumed": consumed, "checks": checks, "inference_limit": "Transcript generated by deterministic fake from PCM markers, not a real ASR result; demonstrates buffer exposure and lack of context exclusion."})
        await session.close()
        if backend is not None and route != "local_gpu":
            await backend.close()
    return output


async def rolling_case():
    gemini, gemini_wire = await make_session({"route": "gemini"})
    deepgram, deepgram_wire = await make_session({"route": "deepgram"})
    configured = {"gemini": True}
    class BoundBackend:
        def __init__(self, session):
            self.session = session
        async def open_session(self, *, projection):
            if self.session._event_projection.projection != projection:
                raise RuntimeError("Fixture projection mismatch")
            return self.session
    rolling = RollingSTTBackend((
        RollingProviderDefinition(STTProviderName.GEMINI_TRANSCRIBE, lambda: BoundBackend(gemini), lambda: configured["gemini"]),
        RollingProviderDefinition(STTProviderName.DEEPGRAM, lambda: BoundBackend(deepgram), lambda: True),
    ))
    projection = STTSessionProjection("scoped", "fixture-epoch")
    first = await rolling.open_session(projection=projection)
    await first.begin_turn(request("rolling-gemini"))
    configured["gemini"] = False
    second = await rolling.open_session(projection=projection)
    await second.begin_turn(request("rolling-deepgram", 2))
    from google.genai import types
    gemini._handle_message(types.LiveServerMessage(server_content=types.LiveServerContent(input_transcription=types.Transcription(text="late old member"))))
    event = await asyncio.wait_for(anext(first.turn_events()), 2)
    checks = {"sessions_keep_resolved_provider": first.provider_name is STTProviderName.GEMINI_TRANSCRIBE and second.provider_name is STTProviderName.DEEPGRAM, "late_result_keeps_original_stream_scope": event.identity.stream.settings_scope == ("rolling-gemini",), "capability_is_selected_session_specific": first.independent_recognition_units and not second.independent_recognition_units}
    row = {"id": "actual-rolling-late-member-result", "status": "passed" if all(checks.values()) else "failed", "first_member": first.provider_name.value, "second_member": second.provider_name.value, "late_event": encode(event), "checks": checks, "limit": "Both member sessions share the fixture epoch deliberately; no production epoch generator, route-selection inventory, quota policy, or new capability payload is certified."}
    await first.close()
    await second.close()
    return [row]


async def local_cases():
    output = []
    for cls in (LocalQwenSherpaSTTBackend, LocalParakeetV3SherpaSTTBackend, LocalParakeetJapaneseSherpaSTTBackend):
        for timed in (False, True):
            raw = {"text": "no three"}
            if timed:
                raw.update(tokens=["no", "three"], timestamps=[0.01, 0.11], durations=[0.1, 0.2])
            recognizer = Recognizer(SimpleNamespace(**raw))
            backend = cls(Path("offline-model-not-loaded"))
            backend._recognizer = recognizer
            session = _LocalQwenSherpaSession(backend, projection=STTSessionProjection("scoped", "fixture-epoch"))
            req = request(backend.provider_id)
            await session.begin_turn(req)
            await session.send_turn_audio(req.identity, bytes(3200), payload_sequence=1, source_ranges=(span(end=1600),), context_only=False)
            before = recognizer.decodes
            await session.seal_turn(req.identity, sealed_content_ranges=(span(end=1600),), seal_reason="source_eof", observed_trailing_silence_ms=0)
            event = await asyncio.wait_for(anext(session.turn_events()), 5)
            checks = {"not_decoded_before_seal": before == 0, "one_100ms_decode": recognizer.decodes == 1 and recognizer.accepted_samples == [1600], "short_correction_preserved": event.text == "no three", "no_timing_projection": event.provenance == () and event.final_speaker_runs == ()}
            output.append({"id": f"{backend.provider_id}-{'timed' if timed else 'untimed'}-fake-runtime", "status": "passed" if all(checks.values()) else "failed", "model_id": backend.model_id, "raw_runtime_result": raw, "event": encode(event), "checks": checks, "decode_count": recognizer.decodes, "decode_samples": recognizer.accepted_samples})
            await session.close()
            await backend.close()
    wire = GpuWire()
    backend = LocalGpuSTTBackend(wire, "peer", Path("offline-model-not-loaded"), "fixture-gpu-model", "no-device", source_mode="auto")
    session = _LocalGpuSTTSession(backend, STTSessionProjection("scoped", "fixture-epoch"))
    req = request("local_gpu")
    await session.begin_turn(req)
    await session.send_turn_audio(req.identity, bytes(3200), payload_sequence=1, source_ranges=(span(end=1600),), context_only=False)
    before = len(wire.calls)
    await session.seal_turn(req.identity, sealed_content_ranges=(span(end=1600),), seal_reason="source_eof", observed_trailing_silence_ms=0)
    event = await asyncio.wait_for(anext(session.turn_events()), 5)
    checks = {"one_100ms_decode_after_seal": before == 0 and len(wire.calls) == 1 and wire.calls[0]["pcm_bytes"] == 3200, "short_correction_preserved": event.text == "no three", "language_preserved": event.final_language_runs[0].language == "en", "worker_contract_has_no_timestamps": not any("time" in f.name or "token" in f.name for f in fields(GpuWorkerTranscription))}
    output.append({"id": "local-gpu-fake-runtime", "status": "passed" if all(checks.values()) else "failed", "worker_result_fields": [f.name for f in fields(GpuWorkerTranscription)], "event": encode(event), "calls": wire.calls, "checks": checks})
    await session.close()
    return output


def mapping_case():
    discontinuity = AudioCaptureDiscontinuity("known_loss", 2.0, 4800)
    raw = AudioCaptureSpan(3, 2, 48000, 96000, 100800, 2.0, 2.1, discontinuity).with_normalized_range(sample_rate_hz=16000, start_sample=32000, end_sample=33600)
    left = raw.slice_normalized(32000, 32800)
    right = raw.slice_normalized(32800, 33600)
    mapping = STTStreamInputMap()
    first = span(0, 1600)
    pcm, prepared = mapping.prepare(bytes(3200), (first,))
    mapping.commit(prepared)
    duplicate, duplicate_map = mapping.prepare(bytes(3200), (first,))
    overlap, overlap_map = mapping.prepare(bytes(3200), (span(800, 2400, 2),))
    mapping.commit(overlap_map)
    errors = {}
    for name, ranges in {"gap": (span(3200, 4800, 3),), "epoch_change": (span(2400, 4000, 3, epoch=4),)}.items():
        try:
            mapping.prepare(bytes(3200), ranges)
        except ValueError as exc:
            errors[name] = str(exc)
    marked_contiguous = span(2400, 4000, 4, loss=discontinuity)
    _, marked_map = mapping.prepare(bytes(3200), (marked_contiguous,))
    checks = {"piecewise_resample_slice": left.source_end_sample == right.source_start_sample == 98400, "loss_only_at_left_edge": left.discontinuity_before == discontinuity and right.discontinuity_before is None, "duplicate_audio_suppressed": duplicate == b"" and duplicate_map is None, "overlap_trimmed_to_unsent_suffix": len(overlap) == 1600 and overlap_map.source_start == 1600, "gap_and_epoch_rejected": len(errors) == 2, "contiguous_loss_marker_not_rejected_by_input_map": marked_map is not None}
    return {"id": "actual-capture-span-and-stream-map", "status": "passed" if all(checks.values()) else "failed", "raw": encode(raw), "slices": encode((left, right)), "checks": checks, "errors": errors, "sent_samples_before_marked_write": mapping.sent_samples, "marked_contiguous_prepared": encode(marked_map)}


class ProbabilityEngine:
    def speech_probability(self, samples, *, sample_rate_hz):
        return 0.9 if samples[0] else 0.0

    def reset(self):
        pass


class NoInference:
    def request_prepare(self):
        raise RuntimeError("Offline probe forbids model preparation")


async def delivery_case(long):
    clock = SimpleNamespace(value=0.0)
    vad = create_peer_vad_gating(ProbabilityEngine(), sample_rate_hz=16000, ring_buffer_ms=500, hangover_ms=128)
    ledger = PeerAudioSegmentLedger(activation_generation=7, settings=settings())
    records = []
    snapshot_ends = []
    async def emit(owned):
        event = owned.event
        if isinstance(event, (SpeechStart, SpeechEnd)):
            records.append({"event": encode(event), "segment": {key: encode(getattr(owned.segment, key)) for key in ("identity", "content_sample_count", "context_sample_count", "genuine_onset", "state", "opened_at_monotonic_s", "sealed_at_monotonic_s", "seal_reason")}})
        if isinstance(event, SpeechEnd):
            snapshot_ends.append(owned.segment)
    controller = ListenDeliveryController(vad=vad, ledger=ledger, emit=emit, monotonic_clock=lambda: clock.value, smart_turn_owner=NoInference())
    try:
        total = 189 if long else 7
        for index in range(total):
            active_speech = long or index < 3
            source = span(index * 512, (index + 1) * 512, index + 1)
            clock.value = source.source_end_monotonic_s
            for event in vad.process_owned_chunk(np.full(512, float(active_speech), dtype=np.float32), (source,)):
                await controller.handle_vad_event(event)
            await controller.observe_acoustic_chunk(speech_observed=vad.last_observation_was_speech, capture=(source,))
        if long:
            start = records[-1]
            checks = {"deadline_at_first_32ms_frontier_after_6s": len(snapshot_ends) == 1 and snapshot_ends[0].seal_reason == "delivery_deadline" and abs(snapshot_ends[0].sealed_at_monotonic_s - 6.016) < 1e-8, "continuation_without_new_onset": start["event"]["type"] == "SpeechStart" and not start["segment"]["genuine_onset"], "rollover_context_300ms": start["event"]["pre_roll"]["samples"] == 4800 and start["segment"]["context_sample_count"] == 4800, "no_reclaimed_content": start["segment"]["content_sample_count"] == 512}
        else:
            checks = {"natural_short_source_sealed": len(snapshot_ends) == 1 and snapshot_ends[0].content_sample_count == 3584 and snapshot_ends[0].seal_reason == "delivery_pause", "less_than_one_second": len(snapshot_ends) == 1 and snapshot_ends[0].sealed_at_monotonic_s == 0.224, "no_model_called": True}
        return {"id": "existing-six-second-rollover" if long else "existing-224ms-source-completion", "status": "passed" if all(checks.values()) else "failed", "input": {"sample_rate": 16000, "chunk_samples": 512, "speech_chunks": total if long else 3, "silence_chunks": 0 if long else 4, "probabilities": [0.9, 0.0], "vad_start_commit_chunks": 3, "hangover_ms": 128, "delivery_profile": "off"}, "checks": checks, "boundary_records": records}
    finally:
        await controller.close()


async def run(args):
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    fixture_path = Path(__file__).with_name("fixtures.json")
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    results = []
    started = time.perf_counter()
    for case in fixtures["cases"]:
        try:
            results.append(await adapter_case(case))
        except Exception as exc:
            results.append({"id": case["id"], "route": case["route"], "status": "blocked", "error": type(exc).__name__ + ": " + str(exc)})
    for operation in (local_cases, context_cases, rolling_case):
        try:
            results.extend(await operation())
        except Exception as exc:
            results.append({"id": operation.__name__, "status": "blocked", "error": type(exc).__name__ + ": " + str(exc)})
    results.append(normalizer_case())
    results.append(mapping_case())
    results.append(await delivery_case(False))
    results.append(await delivery_case(True))
    packages = {}
    for name in ("numpy", "httpx", "websockets", "google-genai", "deepgram-sdk", "elevenlabs", "dashscope", "sherpa-onnx", "soxr", "onnxruntime"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    sources = {str(Path(module.__file__).resolve().relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in list(sys.modules.items()) if name.startswith("puripuly_heart") and getattr(module, "__file__", None)}
    report = {"schema": "issue-212-offline-results-1", "evidence_kind": "fixture observation only; no provider or model certification", "baseline_requested": "97c0ad25ea1d1d7b37a396c389b7774797a77af0", "head_observed": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "packages": packages, "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(), "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_sha256": sources, "duration_seconds": round(time.perf_counter() - started, 4), "network_attempts": NETWORK_ATTEMPTS, "results": results, "totals": {state: sum(row["status"] == state for row in results) for state in ("passed", "failed", "blocked")}, "not_run": ["All provider API calls, model inference, capture, installed app, VR/OSC and GPU worker processes", "Selected native binary/model timestamp production; fake runtime only", "Live server finality, time units, control limits, ordering, quota and cost", "Real Auto install/model selection, production Rolling epoch generation and new capability payload", "Speaker cutter or new source policy implementation", "Repository-wide tests, build, lint and formatting"]}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "totals": report["totals"], "network_attempts": NETWORK_ATTEMPTS, "duration_seconds": report["duration_seconds"]}))
    return 1 if report["totals"]["failed"] or report["totals"]["blocked"] or NETWORK_ATTEMPTS else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args)))
