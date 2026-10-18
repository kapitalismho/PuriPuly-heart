from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import issue218_experiment as local

from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.ownership import OwnedStreamInput
from puripuly_heart.core.stt.backend import (
    PermanentSTTScopedSessionError,
    STTProviderAudioBoundary,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.core.stt.recognition_units import RecognitionConsumptionLedger
from puripuly_heart.core.stt.scoped_engine import STTRecognitionWatchdogs
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.providers.stt import deepgram as deepgram_adapter
from puripuly_heart.providers.stt import gemini_transcribe as gemini_adapter

prior = local.prior
PRIVATE_VALUES = []


def safe_text(value):
    text = str(value)
    for secret in PRIVATE_VALUES:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)((?:api_key|key|token|authorization)=)[^&\s\"\']+", r"\1[redacted]", text)
    for home in (str(Path.home()), str(Path.home()).replace("\\", "/")):
        text = text.replace(home.replace("\\", "\\\\"), "<USER_HOME>").replace(home, "<USER_HOME>")
    return text


def clean(value):
    if dataclasses.is_dataclass(value):
        return clean(prior.serial(value))
    if isinstance(value, dict):
        return {
            str(k): (
                "[redacted]"
                if str(k).lower()
                in {
                    "api_key",
                    "key",
                    "token",
                    "authorization",
                    "account_id",
                    "project_id",
                    "organization_id",
                    "email",
                    "headers",
                    "session_token",
                    "new_handle",
                    "resume_handle",
                    "access_token",
                    "refresh_token",
                    "credentials",
                    "password",
                    "cookie",
                }
                else clean(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, BaseException):
        return {"exception_class": type(value).__name__, "message": safe_text(value)}
    if isinstance(value, str):
        return safe_text(value)
    return prior.serial(value)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = safe_text(json.dumps(clean(value), indent=2, ensure_ascii=False, default=str))
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")


class PrivateLogFilter(logging.Filter):
    def filter(self, record):
        record.msg = safe_text(record.getMessage())
        if record.exc_info and record.exc_info[1]:
            record.msg += " cause=" + safe_text(record.exc_info[1])
        record.exc_info = None
        record.exc_text = None
        record.args = ()
        return True


def credential_values(path):
    selected = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            match = re.match(
                r"^\s*(?:export\s+)?(GOOGLE_API_KEY2|DEEPGRAM_API_KEY)\s*=\s*(.*?)\s*$", line
            )
            if not match:
                continue
            name, value = match.groups()
            if value[:1] in ('"', "'") and value[-1:] == value[:1]:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()
            if value:
                selected[name] = value
    if not selected.get("DEEPGRAM_API_KEY") and os.environ.get("DEEPGRAM_API_KEY"):
        selected["DEEPGRAM_API_KEY"] = os.environ["DEEPGRAM_API_KEY"]
    for name in tuple(os.environ):
        if name.startswith(("GOOGLE_", "GEMINI_")) or re.search(
            r"(?:API_KEY|TOKEN|SECRET|CREDENTIAL|AUTHORIZATION)", name, re.I
        ):
            os.environ.pop(name, None)
    PRIVATE_VALUES.extend(selected.values())
    return selected


class Budget:
    def __init__(self, path):
        self.path = path
        self.counts = {
            name: {
                "remote_starts_reserved": 0,
                "audio_samples_reserved": 0,
                "audio_samples_written": 0,
                "halted": False,
            }
            for name in ("deepgram", "gemini_transcribe")
        }
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    self.apply(json.loads(line))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", newline="\n")

    def apply(self, event):
        counts = self.counts[event["route"]]
        name = event["event"]
        if name == "remote_start_reserved":
            counts["remote_starts_reserved"] += 1
        elif name == "audio_reserved":
            counts["audio_samples_reserved"] += event["samples"]
        elif name == "audio_written":
            counts["audio_samples_written"] += event["samples"]
        elif name == "provider_halted":
            counts["halted"] = True

    def record(self, route, event, **fields):
        row = {"route": route, "event": event, "at": time.monotonic(), **fields}
        self.apply(row)
        self.stream.write(json.dumps(clean(row), ensure_ascii=False) + "\n")
        self.stream.flush()

    def reserve_start(self, route, case):
        counts = self.counts[route]
        if counts["halted"] or counts["remote_starts_reserved"] >= 72:
            raise PermanentSTTScopedSessionError(
                "authorized route halted or72remote-start budget exhausted"
            )
        self.record(route, "remote_start_reserved", case=case)

    def reserve_audio(self, route, samples, case):
        counts = self.counts[route]
        if counts["halted"] or counts["audio_samples_reserved"] + samples > 300 * 16000:
            raise PermanentSTTScopedSessionError(
                "authorized route halted or300second audio budget exhausted"
            )
        self.record(route, "audio_reserved", samples=samples, case=case)

    def written(self, route, samples, case):
        self.record(route, "audio_written", samples=samples, case=case)

    def halt(self, route, reason):
        if not self.counts[route]["halted"]:
            self.record(route, "provider_halted", reason=safe_text(reason))

    def snapshot(self):
        return {
            route: {
                **counts,
                "reserved_audio_seconds": counts["audio_samples_reserved"] / 16000,
                "successfully_written_audio_seconds": counts["audio_samples_written"] / 16000,
                "max_remote_starts": 72,
                "max_audio_seconds_including_replay": 300,
            }
            for route, counts in self.counts.items()
        }

    def close(self):
        self.stream.close()


def endpoint(value):
    parts = urlsplit(str(value))
    return {
        "scheme": parts.scheme,
        "host": parts.hostname,
        "path": parts.path,
        "query_values_retained": False,
    }


@contextmanager
def instrument(route, budget, record, selected_key):
    restores = []
    trace = record["trace"]

    def replace(owner, name, value):
        restores.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    if route == "deepgram":
        cls = deepgram_adapter._DeepgramSDKSession
        original_start, original_write, original_result = (
            cls.start,
            cls._write_payload,
            cls._build_transcript_event,
        )
        original_connect = network_clients.external_websocket_connect

        def connect(url, **kwargs):
            trace.append(
                {
                    "event": "wire_connection_attempt",
                    "at": time.monotonic(),
                    "endpoint": endpoint(url),
                }
            )
            return original_connect(url, **kwargs)

        async def start(self):
            budget.reserve_start(route, record["case"])
            if self.api_key != selected_key:
                raise PermanentSTTScopedSessionError("explicit Deepgram constructor key mismatch")
            trace.append(
                {
                    "event": "credential_selection_proof",
                    "at": time.monotonic(),
                    "variable": "DEEPGRAM_API_KEY",
                    "explicit_constructor": True,
                }
            )
            try:
                await original_start(self)
            except Exception as error:
                budget.halt(route, error)
                raise

        async def write(self, payload):
            count = len(payload) // 2 if isinstance(payload, bytes) else 0
            control = (
                None
                if count
                else (
                    "Finalize"
                    if payload is deepgram_adapter._FINALIZE
                    else "CloseStream" if payload is deepgram_adapter._CLOSE_STREAM else "other"
                )
            )
            if count:
                budget.reserve_audio(route, count, record["case"])
            trace.append(
                {
                    "event": "wire_write_begin",
                    "at": time.monotonic(),
                    "samples": count,
                    "control": control,
                    "successful_sent_samples_before": record["actual_audio_samples_written"],
                }
            )
            await original_write(self, payload)
            if count:
                budget.written(route, count, record["case"])
                record["actual_audio_samples_written"] += count
            trace.append(
                {
                    "event": "wire_write_complete",
                    "at": time.monotonic(),
                    "samples": count,
                    "control": control,
                    "successful_sent_samples_after": record["actual_audio_samples_written"],
                }
            )

        def raw(self, value):
            data = (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else {"type": type(value).__name__}
            )
            trace.append({"event": "raw_wire_result", "at": time.monotonic(), "value": clean(data)})
            return original_result(self, value)

        replace(cls, "start", start)
        replace(cls, "_write_payload", write)
        replace(cls, "_build_transcript_event", raw)
        replace(network_clients, "external_websocket_connect", connect)
    else:
        cls = gemini_adapter._GeminiTranscribeLiveSession
        original_start, original_send, original_message = (
            cls.start,
            cls._send_realtime,
            cls._handle_message,
        )
        original_client = gemini_adapter._create_genai_client_sync
        from puripuly_heart.providers.stt import genai_network

        original_handshake = genai_network._LiveRedirectGuard.handshake

        async def handshake(self, *args, **kwargs):
            uri = self.protocol.uri
            host = f"[{uri.host}]" if ":" in uri.host else uri.host
            value = (
                ("wss" if uri.secure else "ws")
                + "://"
                + host
                + ":"
                + str(uri.port)
                + uri.resource_name
            )
            trace.append(
                {
                    "event": "wire_connection_attempt",
                    "at": time.monotonic(),
                    "endpoint": endpoint(value),
                }
            )
            return await original_handshake(self, *args, **kwargs)

        def create_client(api_key, http_options):
            if api_key != selected_key:
                raise PermanentSTTScopedSessionError(
                    "explicit GOOGLE_API_KEY2 constructor mismatch"
                )
            client = original_client(api_key, http_options)
            context = getattr(client, "_puripuly_live_network", None)
            trace.append(
                {
                    "event": "credential_selection_proof",
                    "at": time.monotonic(),
                    "variable": "GOOGLE_API_KEY2",
                    "explicit_constructor": True,
                    "GOOGLE_API_KEY_fallback": False,
                    "endpoint": endpoint(context[0]) if context else None,
                }
            )
            return client

        async def start(self):
            budget.reserve_start(route, record["case"])
            trace.append(
                {
                    "event": "remote_start_attempt",
                    "at": time.monotonic(),
                    "model": self.model,
                    "language_codes": self.language_codes,
                    "config": {
                        "response_modalities": ["TEXT"],
                        "transcription_mode": "VERBATIM",
                        "prefix_padding_ms": 500,
                        "silence_duration_ms": 400,
                    },
                }
            )
            try:
                await original_start(self)
            except Exception as error:
                budget.halt(route, error)
                raise

        async def send(self, **kwargs):
            audio = kwargs.get("audio")
            count = len(audio["data"]) // 2 if isinstance(audio, dict) else 0
            control = "audio_stream_end" if kwargs.get("audio_stream_end") else None
            if count:
                budget.reserve_audio(route, count, record["case"])
            trace.append(
                {
                    "event": "wire_write_begin",
                    "at": time.monotonic(),
                    "samples": count,
                    "control": control,
                    "successful_sent_samples_before": record["actual_audio_samples_written"],
                }
            )
            await original_send(self, **kwargs)
            if count:
                budget.written(route, count, record["case"])
                record["actual_audio_samples_written"] += count
            trace.append(
                {
                    "event": "wire_write_complete",
                    "at": time.monotonic(),
                    "samples": count,
                    "control": control,
                    "successful_sent_samples_after": record["actual_audio_samples_written"],
                }
            )

        def message(self, value):
            data = (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else {"type": type(value).__name__}
            )
            trace.append({"event": "raw_wire_result", "at": time.monotonic(), "value": clean(data)})
            return original_message(self, value)

        replace(cls, "start", start)
        replace(cls, "_send_realtime", send)
        replace(cls, "_handle_message", message)
        replace(gemini_adapter, "_create_genai_client_sync", create_client)
        replace(genai_network._LiveRedirectGuard, "handshake", handshake)
    try:
        yield
    finally:
        for owner, name, value in reversed(restores):
            setattr(owner, name, value)


class SessionTrace(prior.SessionTrace):
    @property
    def accepts_stream_input(self):
        return bool(getattr(self.inner, "accepts_stream_input", False))

    @property
    def independent_recognition_units(self):
        return bool(getattr(self.inner, "independent_recognition_units", False))

    async def begin_stream(self, stream):
        await self.inner.begin_stream(stream)

    def recognition_source_covers(self, ranges):
        return self.inner.recognition_source_covers(ranges)

    async def send_stream_audio(self, pcm16le, *, source_ranges):
        expected = b"".join(
            self.capture_pcm[s.normalized_start_sample * 2 : s.normalized_end_sample * 2]
            for s in source_ranges
        )
        self.trace.append(
            {
                "event": "native_stream_input",
                "at": time.monotonic(),
                "samples": len(pcm16le) // 2,
                "source_ranges": prior.serial(source_ranges),
                "matches_declared_source_pcm": pcm16le == expected,
                "pcm16_sha256": hashlib.sha256(pcm16le).hexdigest(),
            }
        )
        await self.inner.send_stream_audio(pcm16le, source_ranges=source_ranges)

    async def end_stream(self, *, reason):
        self.trace.append({"event": "native_stream_end", "at": time.monotonic(), "reason": reason})
        await self.inner.end_stream(reason=reason)


async def case_run(
    args, budget, key, name, values, mapping, translation, desktop, mode, profile, repetition
):
    route = args.route
    record = {
        "case": name,
        "mode": mode,
        "profile": profile if mode == "new" else None,
        "route": route,
        "repetition": repetition,
        "status": "running",
        "input_samples": values.size,
        "input_pcm_f32_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "assembly_scoring_only": mapping,
        "sources": [],
        "trace": [],
        "evidence": [],
        "resources": [],
        "errors": [],
        "frame_delivery_lags": [],
        "actual_audio_samples_written": 0,
        "budget_before": budget.snapshot(),
        "evidence_poll_seconds": 0.05,
    }
    trace = record["trace"]
    if route == "deepgram":
        backend = deepgram_adapter.DeepgramRealtimeSTTBackend(
            api_key=key,
            language="en",
            model="nova-3",
            sample_rate_hz=16000,
            keyterms=(),
            stream_label="peer",
        )
    else:
        backend = gemini_adapter.GeminiTranscribeSTTBackend(
            api_key=key,
            language_codes=gemini_adapter.gemini_transcribe_language_codes("en"),
            custom_vocabulary=(),
            model=gemini_adapter.GEMINI_TRANSCRIBE_STT_MODEL,
            sample_rate_hz=16000,
        )
    record["asr"] = {
        "class": type(backend).__name__,
        "model_id": backend.model,
        "sample_rate_hz": backend.sample_rate_hz,
        "language": "en",
        "keyterms": [],
        "drain_timeout_s": backend.drain_timeout_s,
        "input_strategy": (
            "continuous original stream and independent native finals; no inferred source association"
            if route == "gemini_transcribe"
            else "continuous scoped submitted input and final-text source/evidence joining; no optional speaker physical cut"
        ),
    }
    settings = prior.AudioSegmentSettingsSnapshot(
        route,
        (route, backend.model),
        ("issue218_cloud",),
        "manual",
        "en",
        ("en",),
        16000,
        0.3,
        600,
        500,
    )
    ledger = prior.PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    consumption = prior.STTContributionConsumptionLedger()
    recognition_consumption = RecognitionConsumptionLedger()
    config = prior.TranslationRuntimeConfigSnapshot(
        0,
        prior.TranslationRuntimeConfig(
            source_language="en",
            target_language="ko",
            peer_source_language="en",
            peer_target_language="ko",
            peer_translation_enabled=True,
            concurrency_limit=3,
            integrated_context_enabled=True,
            system_prompt=local.resolve_system_prompt(""),
        ),
    )
    clock = prior.SystemClock()
    presenter = local.OverlayPresenter(
        calibration=local.OverlayCalibration(), bridge=desktop, translation_enabled=True
    )
    output = local.OutputRuntime(
        chatbox=local.ChatboxGuard(),
        clock=clock,
        overlay_sink=presenter,
        routing_observer=lambda decision: trace.append(
            {"event": "routing_decision", "at": time.monotonic(), "value": prior.serial(decision)}
        ),
        activation_notice_enabled=False,
    )
    output.activate_peer_generation(1)
    diagnostics = local.TranslationLatencyDiagnosticsOwner(
        clock=clock, config_snapshot=lambda: config
    )
    projection = local.TranslationOutputProjectionOwner(
        output_runtime=output,
        ui_messages=local.TranslationUiMessageQueue(asyncio.Queue(), output),
        diagnostics=diagnostics,
        clock=clock,
    )
    peer_runtime = prior.ChannelRuntime("peer")
    requests = prior.TranslationRequestOwner(
        config_snapshot=lambda: config,
        self_runtime=prior.ChannelRuntime("self"),
        peer_runtime=peer_runtime,
        context_resolver=prior.ContextResolver(clock=clock, config_snapshot=lambda: config),
        provider_runtime=local.ProviderRuntimeHandle(name="llm", provider=translation),
        diagnostics=diagnostics,
        presentation=projection,
        clock=clock,
    )

    async def call(method, *a):
        return await getattr(peer, method)(*a)

    turns = prior.TranslationTurnLifecycleOwner(
        on_child_created=lambda *a: call("on_child_created", *a),
        on_child_started=lambda *a: call("on_child_started", *a),
        process_child=lambda *a: call("process_child", *a),
        on_child_terminal=lambda *a: call("on_child_terminal", *a),
        on_parent_closed=lambda *a: call("on_parent_closed", *a),
        on_parent_rejected=lambda *a: call("on_parent_rejected", *a),
        on_parent_admitted=lambda *a: call("on_parent_admitted", *a),
        output=SimpleNamespace(
            submit_translation_output=lambda *a: call("submit_translation_output", *a)
        ),
        config_snapshot=lambda: config,
        predecessor_wait_observer=diagnostics.record_translation_wait,
        turn_generation_observer=projection.retire_turn_generation,
    )
    peer = local.ObservedPeer(
        runtime=peer_runtime,
        config_snapshot=lambda: config,
        translation_turns=turns,
        local_asr_runtime=None,
        translation_requests=requests,
        output_projection=projection,
        diagnostics=diagnostics,
        clock=clock,
    )
    peer.record = record
    order = 0

    async def event_sink(event):
        nonlocal order
        trace.append({"event": "provider_result", "at": time.monotonic(), "value": clean(event)})
        if isinstance(event, STTRecognitionUnitTerminal):
            if (
                recognition_consumption.consume(event)
                and engine.is_current_recognition_stream(event.unit.identity.stream)
                and event.unit.identity.stream.activation_generation == ledger.activation_generation
                and event.unit.identity.stream.capture_epoch == 7
            ):
                if event.outcome == "final" and event.unit.text:
                    await peer.handle_recognition_unit(event.unit)
            return
        if isinstance(event, STTProviderAudioBoundary):
            consumption.record_boundary(event.identity, event.boundary)
            return
        if not isinstance(event, (prior.STTProviderTurnUpdate, prior.STTProviderTurnTerminal)):
            return
        if mode == "baseline" and isinstance(event, prior.STTProviderTurnUpdate):
            trace.append(
                {
                    "event": "baseline_update_admission_suppressed",
                    "at": time.monotonic(),
                    "identity": prior.serial(event.identity),
                    "source_release": prior.serial(event.source_release),
                }
            )
            return
        if isinstance(event, prior.STTProviderTurnUpdate) and event.source_release is None:
            return
        snapshot = next(s for s in ledger.snapshots if s.identity == event.identity.segment)
        if isinstance(event, prior.STTProviderTurnTerminal):
            if snapshot.sealed_at_monotonic_s is not None:
                consumption.seal_audio(event.identity)
            if event.outcome not in ("final", "degraded") or not event.text:
                consumption.retire(event.identity)
                if event.outcome not in ("empty", "final"):
                    record["errors"].append(
                        "provider terminal: " + str(event.failure_reason or event.outcome)
                    )
                return
        pieces = consumption.consume_parts(event)
        if pieces:
            order += 1
            transcript = prior.contribution_transcript(
                snapshot, event, pieces, created_at=time.monotonic(), publication_order=order
            )
            await peer.handle_stt_event(local.STTFinalEvent(transcript.utterance_id, transcript))

    sessions = []

    async def factory(_settings, epoch):
        try:
            inner = SessionTrace(
                await backend.open_session(projection=prior.STTSessionProjection("scoped", epoch)),
                trace,
                values,
            )
        except Exception as error:
            budget.halt(route, error)
            raise PermanentSTTScopedSessionError(safe_text(error)) from None
        if mode == "new":
            inner = StreamingSpeakerSession(inner)
        sessions.append(inner)
        return inner

    engine = prior.ScopedRecognitionEngine(
        factory,
        channel="peer",
        event_sink=event_sink,
        diagnostic_sink=lambda event: trace.append(
            {"event": "engine_diagnostic", "at": time.monotonic(), "value": clean(event)}
        ),
    )
    speaker = nemotron = None
    finished = asyncio.Event()
    pump = sampler = None
    received = hashlib.sha256()
    record["capture_input_samples"] = 0

    async def evidence_pump():
        while not finished.is_set():
            items = speaker.drain_evidence(2048)
            if items:
                record["evidence"].extend(prior.serial(item) for item in items)
                trace.append(
                    {"event": "evidence_delivery", "at": time.monotonic(), "count": len(items)}
                )
                await engine.observe_speaker_evidence(items)
            await asyncio.sleep(0.05)

    class Sink:
        async def handle_stream_input(self, event):
            record["capture_input_samples"] += event.chunk.size
            received.update(event.chunk.tobytes())
            if speaker:
                speaker.observe_input("issue218_cloud:" + name, event)
            if route == "gemini_transcribe":
                owned = OwnedStreamInput(
                    event=event,
                    ledger=ledger,
                    settings=ledger.settings,
                    activation_generation=ledger.activation_generation,
                )
                await engine.handle_stream_input(owned)

        async def handle_owned_vad_event(self, owned):
            if isinstance(owned.event, (prior.SpeechStart, prior.SpeechEnd)):
                trace.append({"event": "vad", "at": time.monotonic(), "value": prior.serial(owned)})
                peer._record_peer_owned_vad_event(owned)
            await engine.handle_owned_vad_event(owned)

        async def observe_source_activity(self, **kwargs):
            await engine.observe_source_activity(**kwargs)

    class Source:
        terminal_reason = "closed"

        async def frames(self):
            for sequence, offset in enumerate(range(0, values.size, 512)):
                end = min(values.size, offset + 512)
                due = started + end / 16000
                await asyncio.sleep(max(0, due - time.monotonic()))
                record["frame_delivery_lags"].append(time.monotonic() - due)
                yield prior.AudioFrameF32(
                    values[offset:end],
                    16000,
                    capture=prior.AudioCaptureSpan(
                        7, sequence, 16000, offset, end, started + offset / 16000, due
                    ),
                )

    starts = (len(desktop.events), len(desktop.snapshots), len(desktop.readbacks))
    with instrument(route, budget, record, key):
        try:
            if mode == "new":
                nemotron = prior.NemotronBackend(
                    artifact_path=args.model, runtime_dir=args.runtime, device=0, profile=profile
                )
                speaker = prior.SpeakerRuntime(nemotron)
                await speaker.start()
                if not speaker.snapshot().ready:
                    raise RuntimeError(speaker.snapshot().error)
                record["speaker_initial"] = prior.serial(speaker.snapshot())
                pump = asyncio.create_task(evidence_pump())
            await engine._ensure_session(
                settings, STTRecognitionWatchdogs(), engine._authority_generation
            )
            record["provider_ready_at"] = time.monotonic()
            record["translation_config"] = prior.serial(config)
            vad = prior.create_peer_vad_gating(
                engine=prior.SileroVadOnnx(prior.bundled_silero_vad_onnx_path()),
                sample_rate_hz=16000,
                ring_buffer_ms=500,
                speech_threshold=0.3,
                hangover_ms=600,
            )
            record["vad"] = {
                "sha256": prior.digest(prior.bundled_silero_vad_onnx_path()),
                "speech_threshold": 0.3,
                "hangover_ms": 600,
                "onset_pre_roll_ms": 500,
                "hard_rollover_pre_roll_ms": 300,
                "chunk_samples": vad.chunk_samples,
            }
            started = time.monotonic()
            record["playback_started_at"] = started
            sampler = asyncio.create_task(local.resources(record, finished, speaker, None, turns))
            await prior.run_audio_vad_loop(
                source=Source(),
                vad=vad,
                sink=Sink(),
                target_sample_rate_hz=16000,
                channel_label="peer",
                segment_ledger=ledger,
            )
            deadline = time.monotonic() + 25
            while not engine.is_at_turn_boundary or (speaker and speaker.snapshot().pending_items):
                if time.monotonic() > deadline:
                    raise TimeoutError("owned cloud/diarization drain timeout")
                await asyncio.sleep(0.01)
            await engine.wait_for_event_ingress_drain()
            await asyncio.wait_for(turns.wait_for_idle(), 90)
            await asyncio.sleep(0.1)
            record["status"] = "executed" if record["sources"] else "executed_empty"
            record["wall_seconds"] = time.monotonic() - started
            record["snapshot_capture"] = await desktop.picture(
                f"{route}_{mode}_{profile}_{repetition}_{name}"
            )
        except Exception as error:
            record["status"] = "failed"
            record["error"] = clean(error)
        finally:
            finished.set()
            if pump:
                values_pump = await asyncio.gather(pump, return_exceptions=True)
                record["errors"].extend(
                    clean(e) for e in values_pump if isinstance(e, BaseException)
                )
            if sampler:
                await sampler
            for label, operation in [
                ("engine", engine.close_backend),
                ("turns", turns.close),
                ("output", output.close),
                ("presenter", presenter.close),
            ]:
                try:
                    await operation()
                except Exception as error:
                    record["errors"].append({"cleanup": label, "error": clean(error)})
            if speaker:
                record["speaker_final"] = prior.serial(speaker.snapshot())
                await speaker.stop()
                record["nemotron_stderr"] = safe_text(
                    bytes(nemotron._stderr).decode(errors="replace")
                )
            else:
                record["speaker_final"] = None
            record["segments"] = prior.serial(ledger.snapshots)
            record["capture_received_once"] = (
                record["capture_input_samples"] == values.size
                and received.hexdigest() == record["input_pcm_f32_sha256"]
            )
            record["renderer_events"] = desktop.events[starts[0] :]
            record["renderer_snapshots"] = desktop.snapshots[starts[1] :]
            record["renderer_readbacks"] = desktop.readbacks[starts[2] :]
            record["budget_after"] = budget.snapshot()
            await desktop.replace_snapshot(local.OverlayPresentationSnapshot(revision=0, blocks=[]))
    return record


async def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GGML_VK_DISABLE_GRAPH_OPTIMIZE"] = "1"
    os.environ["NEMO_SPEECH_SCHEDULE_CAPTURE"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    keys = credential_values(args.env_file)
    handlers = logging.getLogger().handlers
    if not handlers:
        logging.basicConfig(level=logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.addFilter(PrivateLogFilter())
    budget = Budget(args.budget)
    variable = "DEEPGRAM_API_KEY" if args.route == "deepgram" else "GOOGLE_API_KEY2"
    key = keys.get(variable)
    inputs, cases = prior.input_cases(args)
    report = {
        "schema": "issue218-cloud-real-causal-e2e-1",
        "candidate": "c7e7ae0d5bfb8a5ab8637c0dbc1e40eb0f84e6f9",
        "command": sys.argv,
        "driver_sha256": prior.digest(Path(__file__)),
        "route": args.route,
        "credential_selection": {
            "variable": variable,
            "source": "<USER_HOME>/Documents/dev/puripuly_heart/.env.local targeted parser",
            "constructor_injection": "explicit api_key argument",
            "secret_value_printed_hashed_archived": False,
            "GOOGLE_API_KEY_fallback": False,
            "cloud_credential_environment_scrubbed_before_child_launch": True,
        },
        "nemotron_artifact": prior.inspect_artifact(args.model),
        "nemotron_runtime": prior.validate_runtime(args.runtime),
        "runtime_manifest_sha256": prior.digest(args.runtime / "runtime-manifest.json"),
        "official_model_card_revision": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        "inputs": inputs,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "deepgram-sdk",
                "google-genai",
                "websockets",
                "numpy",
                "onnxruntime",
                "httpx",
                "flet",
            )
        },
        "policy": {
            "trimming": "OFF",
            "padding_workaround": False,
            "threshold_tuning": False,
            "evidence_poll_seconds": 0.05,
            "concurrency": 3,
            "warmup": "actual managed Gemma owner prepare/prefill, no separate paid/cloud warmup",
            "baseline": "Deepgram suppresses update-driven early admission, retains native contributions for terminal natural/six-second completion; Gemini independent native finals remain independent in every arm, no fabricated enclosing source association or extra wait.",
            "display": "Same current production presenter/two cards/TTL8/no fixed1s wait; real Flet dispatch ACK separately from timed owned readback",
            "stream_controls": "Production natural/six-second mandatory controls unchanged; no optional diarization physical cloud cuts or invented mapping",
            "input_references": "Assembly mapping is scoring-only and never selector input",
        },
        "safety": {
            "public_approved_inputs_only": True,
            "ambient_capture": False,
            "user_settings_changed": False,
            "product_changes": False,
            "other_cloud_routes": False,
            "paid_translation": False,
            "VRChat_SteamVR_launched": False,
        },
        "budget_before": budget.snapshot(),
        "cases": [],
    }
    desktop = local.Desktop(args.output)
    translation = owner = None
    captures = []
    try:
        if not key:
            report["status"] = "blocked_missing_credential"
            report["error"] = "Authorized targeted variable unavailable: " + variable
            budget.halt(args.route, report["error"])
            return report
        if budget.counts[args.route]["halted"]:
            report["status"] = "blocked_previously_halted_route"
            return report
        translation, owner, captures = await local.gemma_start(args, report)
        if report["gemma"]["readiness"]["effective_backend"] != "vulkan":
            raise RuntimeError("Requested actual Vulkan Gemma is unavailable; no CPU substitute")
        await desktop.start()
        report["desktop"] = {
            "renderer": type(desktop.renderer).__name__,
            "window": type(desktop.renderer.window).__name__,
            "pid": desktop.pid,
        }
        arms = [("baseline", "1.04"), ("new", "1.04"), ("new", "0.64")]
        for repetition in range(args.repetitions):
            ordered = arms[repetition % 3 :] + arms[: repetition % 3]
            current = []
            for mode, profile in ordered:
                for name in args.cases.split(","):
                    if budget.counts[args.route]["halted"]:
                        break
                    values, mapping = cases[name]
                    result = await case_run(
                        args,
                        budget,
                        key,
                        name,
                        values,
                        mapping,
                        translation,
                        desktop,
                        mode,
                        profile,
                        repetition + 1,
                    )
                    report["cases"].append(result)
                    current.append(result)
                    save(args.output / "trace.json", report)
                    print(
                        json.dumps(
                            {
                                "route": args.route,
                                "mode": mode,
                                "profile": profile,
                                "repetition": repetition + 1,
                                "case": name,
                                "status": result["status"],
                                "sources": [s["source"]["raw_text"] for s in result["sources"]],
                                "audio_seconds_written": result["actual_audio_samples_written"]
                                / 16000,
                                "budget": budget.snapshot()[args.route],
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
                if budget.counts[args.route]["halted"]:
                    break
            if budget.counts[args.route]["halted"]:
                break
            if not current or not all(c["status"] == "executed" and c["sources"] for c in current):
                report["rotated_repeat_disposition"] = (
                    "Not run because initial complete working path not established; no automatic retries."
                )
                break
        report["status"] = "provider_halted" if budget.counts[args.route]["halted"] else "executed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = clean(error)
    finally:
        try:
            await desktop.close()
        except Exception as error:
            report["desktop_cleanup_error"] = clean(error)
        if translation:
            await translation.close()
        if owner:
            await owner.close()
        for capture in captures:
            await capture.scope.close()
        report["gemma_runtime_output"] = [safe_text(c.output) for c in captures]
        report["desktop_errors"] = clean(desktop.errors)
        report["budget_after"] = budget.snapshot()
        budget.close()
        save(args.output / "trace.json", report)
    return report


if __name__ == "__main__":
    home = Path.home()
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("deepgram", "gemini_transcribe"), required=True)
    parser.add_argument(
        "--env-file", type=Path, default=home / "Documents/dev/puripuly_heart/.env.local"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=home / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.gguf",
    )
    parser.add_argument("--runtime", type=Path, default=ROOT / "build/nemotron")
    parser.add_argument(
        "--gemma-model",
        type=Path,
        default=home / "AppData/Local/puripuly-heart/models/gemma-4-e4b-it-qat-ud-q4-k-xl",
    )
    parser.add_argument(
        "--gemma-runtime",
        type=Path,
        default=home / "Documents/dev/puripuly_heart/build/llama.cpp/llama.cpp-b10423",
    )
    parser.add_argument(
        "--corpus", type=Path, default=home / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1"
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=ROOT.parent
        / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification",
    )
    parser.add_argument(
        "--cases", default="clean_A_B,distinct_repeated_A_B_A,short_A_B_A,linguistic_correction"
    )
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--budget", type=Path, default=ROOT / "native/nemotron/issue218_cloud_budget.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.budget = args.budget.resolve()
    if args.output.parent not in (
        ROOT / "native/nemotron",
        ROOT / "build",
    ) or not args.output.name.startswith("issue218_cloud"):
        raise ValueError("Output must stay within assigned issue218_cloud surfaces")
    if args.budget.parent not in (
        ROOT / "native/nemotron",
        ROOT / "build",
    ) or not args.budget.name.startswith("issue218_cloud"):
        raise ValueError("Budget must stay within assigned issue218_cloud surfaces")
    result = asyncio.run(run(args))
    raise SystemExit(0 if result["status"] == "executed" else 1)
