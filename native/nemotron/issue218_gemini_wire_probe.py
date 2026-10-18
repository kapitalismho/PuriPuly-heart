"""Single-start, public-audio Gemini wire/SDK/adapter observation; no product edits."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from google.genai.live import AsyncSession
from probe import audio

from puripuly_heart.core.audio.format import (
    AudioCaptureSpan,
    AudioFrameF32,
    float32_to_pcm16le_bytes,
)
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    OwnedStreamInput,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.stt.backend import PermanentSTTScopedSessionError, STTSessionProjection
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
from puripuly_heart.core.vad.gating import create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx
from puripuly_heart.providers.stt import gemini_transcribe as adapter
from puripuly_heart.providers.stt.genai_network import _LiveRedirectGuard

ENV = Path(os.environ.get("PURIPULY_GEMINI_PROBE_ENV", ROOT / ".env.local"))
OUT = ROOT / "native/nemotron/issue218_gemini_wire_run3"
SECRET = ""
PRIVATE_NAMES = r"api_key|apiKey|key|token|authorization|headers|sessionId|session_id|sessionToken|newHandle|resumeHandle|access_token|refresh_token|credentials|cookie"


def safe(value):
    if dataclasses.is_dataclass(value):
        return {
            f.name: safe(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if not f.name.startswith("_")
        }
    if isinstance(value, dict):
        return {
            str(k): "[redacted]" if re.fullmatch(PRIVATE_NAMES, str(k), re.I) else safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, (UUID, Path)):
        return safe(str(value))
    if isinstance(value, np.ndarray):
        return {"samples": int(value.size)}
    if isinstance(value, BaseException):
        return {"exception_class": type(value).__name__, "message": safe(str(value))}
    if isinstance(value, str):
        if SECRET:
            value = value.replace(SECRET, "[redacted]")
        value = re.sub(
            r'(?i)(["\'](?:' + PRIVATE_NAMES + r')["\']\s*:\s*)["\'][^"\']*["\']',
            r'\1"[redacted]"',
            value,
        )
        value = re.sub(
            r'(?i)((?:api_key|key|token|authorization)=)[^&\s"\']+', r"\1[redacted]", value
        )
        for home in (str(Path.home()), str(Path.home()).replace("\\", "/")):
            value = value.replace(home.replace("\\", "\\\\"), "<USER_HOME>").replace(
                home, "<USER_HOME>"
            )
    return value


def dump(path, value):
    path.write_text(
        json.dumps(safe(value), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


class Trace:
    def __init__(self):
        if OUT.exists():
            raise RuntimeError(
                "Run3 evidence already exists; this driver never automatically retries"
            )
        OUT.mkdir()
        self.stream = (OUT / "trace.jsonl").open("x", encoding="utf-8", newline="\n")
        self.rows = []
        self.start_count = self.audio_reserved = self.audio_written = 0
        self.wire_pcm = bytearray()
        self.raw_for_model = {}
        self.receive_context = contextvars.ContextVar("raw_receive", default=None)
        self.control_reason = None
        self.last_eof = None
        self.start_at = None
        self.sockets = set()

    def add(self, event, **fields):
        row = safe(
            {
                "sequence": len(self.rows) + 1,
                "event": event,
                "at_monotonic_s": time.monotonic(),
                **fields,
            }
        )
        self.rows.append(row)
        self.stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.stream.flush()
        return row["sequence"]

    def counts(self):
        return {
            "remote_start_attempts": self.start_count,
            "audio_samples_reserved": self.audio_reserved,
            "audio_samples_write_returned": self.audio_written,
            "websocket_successful_audio_samples": len(self.wire_pcm) // 2,
            "reserved_audio_seconds": self.audio_reserved / 16000,
            "written_audio_seconds": self.audio_written / 16000,
        }


def credential():
    global SECRET
    selected = []
    for line in ENV.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^\s*(?:export\s+)?GOOGLE_API_KEY2\s*=\s*(.*?)\s*$", line)
        if m:
            value = m[1]
            if value[:1] in ('"', "'") and value[-1:] == value[:1]:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()
            if value:
                selected.append(value)
    for name in tuple(os.environ):
        if name.startswith(("GOOGLE_", "GEMINI_")) or re.search(
            r"API_KEY|TOKEN|SECRET|CREDENTIAL|AUTHORIZATION", name, re.I
        ):
            os.environ.pop(name, None)
    if len(selected) != 1:
        raise RuntimeError(
            "Exactly one nonempty GOOGLE_API_KEY2 definition is required; no fallback"
        )
    SECRET = selected[0]
    return SECRET


def prepare():
    with zipfile.ZipFile(ROOT / "native/nemotron/issue218_cloud_gemini_raw.zip") as archive:
        previous = json.loads(archive.read("valid/trace.json"))
    baseline = next(
        c for c in previous["cases"] if c["case"] == "clean_A_B" and c["mode"] == "baseline"
    )
    corpus = Path.home() / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1"
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["privacy"]["public_audio_only"] or manifest["privacy"]["private_user_speech"]:
        raise RuntimeError("Input corpus is not authorized public-only")
    blocks = [np.zeros(8000, np.float32)]
    sources = {}
    for piece in baseline["assembly_scoring_only"]:
        name = piece["input"]
        expected = previous["inputs"][name]
        path = corpus / expected["relative_audio_path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["verified_sha256"]:
            raise RuntimeError("Approved public clip hash mismatch")
        decoded = audio(path)
        left, right = piece["input_sample_range"]
        blocks.append(decoded[left:right])
        sources[name] = expected
    blocks.append(np.zeros(20000, np.float32))
    values = np.concatenate(blocks)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    if digest != baseline["input_pcm_f32_sha256"] or values.size != baseline["input_samples"]:
        raise RuntimeError("Exact clean_A_B assembled PCM differs from valid recording")
    if values.size > 8 * 16000:
        raise RuntimeError("Input exceeds authorized audio cap")
    old_writes = [e for e in baseline["trace"] if e["event"] == "wire_write_complete"]
    expected = {
        "assembled_samples": int(values.size),
        "assembled_f32_sha256": digest,
        "assembly_scoring_only": baseline["assembly_scoring_only"],
        "public_sources": sources,
        "valid_run_audio_samples": baseline["actual_audio_samples_written"],
        "valid_run_audio_write_sizes": [e["samples"] for e in old_writes if e["samples"]],
        "valid_run_control_frontiers": [
            e["successful_sent_samples_after"] for e in old_writes if e.get("control")
        ],
        "valid_run_final_texts": [
            e["value"]["server_content"]["input_transcription"]["text"]
            for e in baseline["trace"]
            if e["event"] == "raw_wire_result"
            and e["value"].get("server_content")
            and e["value"]["server_content"].get("input_transcription")
        ],
    }
    return values, expected


@contextmanager
def instrument(trace):
    restores = []

    def patch(owner, name, fn):
        restores.append((owner, name, getattr(owner, name)))
        setattr(owner, name, fn)

    cls = adapter._GeminiTranscribeLiveSession
    old_start, old_send, old_handle, old_end, old_seal, old_close = (
        cls.start,
        cls._send_realtime,
        cls._handle_message,
        cls.end_stream,
        cls.seal_turn,
        cls.close,
    )
    old_recv, old_ws_send, old_ws_close = (
        _LiveRedirectGuard.recv,
        _LiveRedirectGuard.send,
        _LiveRedirectGuard.close,
    )
    old_decode = AsyncSession._receive

    async def start(self):
        if trace.start_count >= 1:
            trace.add("additional_start_refused")
            raise PermanentSTTScopedSessionError("One-start probe cap; no retry")
        trace.start_count += 1
        trace.start_at = time.monotonic()
        trace.add(
            "remote_start_attempt",
            model=self.model,
            language_codes=self.language_codes,
            custom_vocabulary=self.custom_vocabulary,
            drain_timeout_s=self.drain_timeout_s,
            credential_variable="GOOGLE_API_KEY2",
            explicit_key_matches=self.api_key == SECRET,
        )
        if self.api_key != SECRET or self.model != "gemini-3.5-transcribe-live":
            raise PermanentSTTScopedSessionError("Authorized credential/model mismatch")
        await old_start(self)
        trace.add(
            "remote_start_return",
            config=self._client_resources.config.model_dump(mode="json", exclude_none=True),
        )

    async def recv(self, *args, **kwargs):
        trace.sockets.add(self)
        trace.add("websocket_recv_wait", sdk_receive_active=trace.receive_context.get() is not None)
        try:
            raw = await old_recv(self, *args, **kwargs)
        except BaseException as error:
            trace.add("websocket_recv_end", error=error)
            raise
        raw_text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        seq = trace.add(
            "raw_preparse_message",
            payload_type=type(raw).__name__,
            raw_payload=raw_text,
            redacted=safe(raw_text) != raw_text,
        )
        active = trace.receive_context.get()
        if active is not None:
            active.append(seq)
        return raw

    async def decode(self):
        received = []
        token = trace.receive_context.set(received)
        try:
            message = await old_decode(self)
            seq = trace.add(
                "sdk_decoded_message", raw_sequences=received, value=message.model_dump(mode="json")
            )
            trace.raw_for_model[id(message)] = {"sdk_sequence": seq, "raw_sequences": received}
            return message
        except BaseException as error:
            trace.add("sdk_receive_end", raw_sequences=received, error=error)
            raise
        finally:
            trace.receive_context.reset(token)

    def handle(self, message):
        link = trace.raw_for_model.pop(id(message), {})
        before = self._receipt_sequence
        trace.add(
            "adapter_handle_begin",
            **link,
            receipt_sequence_before=before,
            value=message.model_dump(mode="json"),
        )
        old_handle(self, message)
        trace.add(
            "adapter_handle_return",
            **link,
            receipt_sequence_after=self._receipt_sequence,
            stopped=self._stopped,
            protocol_failed=self._protocol_failed,
        )

    async def send(self, **kwargs):
        pcm = kwargs.get("audio", {}).get("data", b"")
        samples = len(pcm) // 2
        if trace.audio_reserved + samples > 8 * 16000:
            trace.add("audio_cap_refused", requested_samples=samples, counts=trace.counts())
            raise RuntimeError("Eight-second submitted-audio cap, including replay/context")
        trace.audio_reserved += samples
        control = "audio_stream_end" if kwargs.get("audio_stream_end") else None
        trace.add(
            "adapter_send_begin",
            samples=samples,
            pcm16_sha256=hashlib.sha256(pcm).hexdigest() if pcm else None,
            control=control,
            reason=trace.control_reason,
            counts=trace.counts(),
        )
        trace.outgoing_pcm = pcm
        try:
            await old_send(self, **kwargs)
        except BaseException as error:
            trace.add("adapter_send_failed", error=error, samples=samples)
            raise
        trace.audio_written += samples
        if control and trace.control_reason == "source_eof":
            trace.last_eof = time.monotonic()
        trace.add(
            "adapter_send_return",
            samples=samples,
            control=control,
            reason=trace.control_reason,
            counts=trace.counts(),
        )

    async def ws_send(self, message, *args, **kwargs):
        trace.sockets.add(self)
        data = json.loads(message)
        realtime = data.get("realtime_input", data.get("realtimeInput", {}))
        blob = realtime.get("audio", {})
        pcm = base64.urlsafe_b64decode(blob["data"]) if blob.get("data") else b""
        if pcm != getattr(trace, "outgoing_pcm", b""):
            raise RuntimeError("SDK wire serialization must preserve exact submitted PCM")
        if pcm:
            data = {
                "realtime_input": {
                    "audio": {
                        "mimeType": blob.get("mimeType", blob.get("mime_type")),
                        "samples": len(pcm) // 2,
                        "pcm16_sha256": hashlib.sha256(pcm).hexdigest(),
                    }
                }
            }
        trace.add("websocket_send_begin", value=data, reason=trace.control_reason)
        await old_ws_send(self, message, *args, **kwargs)
        trace.wire_pcm.extend(pcm)
        trace.add(
            "websocket_send_return",
            samples=len(pcm) // 2,
            control=(
                "audio_stream_end"
                if realtime.get("audioStreamEnd", realtime.get("audio_stream_end"))
                else None
            ),
            reason=trace.control_reason,
        )

    async def end(self, *, reason):
        trace.control_reason = reason
        trace.add("native_stream_end_begin", reason=reason)
        await old_end(self, reason=reason)
        trace.add(
            "normal_drain_return" if reason == "source_eof" else "native_stream_end_return",
            reason=reason,
            eof_observed=trace.last_eof,
            receive_task_done=self._recv_task.done() if self._recv_task else None,
        )
        trace.control_reason = None

    async def seal(self, identity, **kwargs):
        trace.control_reason = kwargs["seal_reason"]
        trace.add("native_seal_begin", identity=identity, value=kwargs)
        await old_seal(self, identity, **kwargs)
        trace.add("native_seal_return", identity=identity)
        trace.control_reason = None

    async def close(self):
        trace.add("adapter_close_begin")
        await old_close(self)
        trace.add("adapter_close_return")

    async def ws_close(self, *args, **kwargs):
        trace.add("websocket_close_begin")
        try:
            return await old_ws_close(self, *args, **kwargs)
        finally:
            trace.add(
                "websocket_close_return", close_code=self.close_code, close_reason=self.close_reason
            )

    for owner, name, fn in [
        (cls, "start", start),
        (cls, "_send_realtime", send),
        (cls, "_handle_message", handle),
        (cls, "end_stream", end),
        (cls, "seal_turn", seal),
        (cls, "close", close),
        (_LiveRedirectGuard, "recv", recv),
        (_LiveRedirectGuard, "send", ws_send),
        (_LiveRedirectGuard, "close", ws_close),
        (AsyncSession, "_receive", decode),
    ]:
        patch(owner, name, fn)
    try:
        yield
    finally:
        for owner, name, old in reversed(restores):
            setattr(owner, name, old)


class SessionTrace:
    def __init__(self, inner, trace, capture):
        self.inner, self.trace, self.capture = inner, trace, float32_to_pcm16le_bytes(capture)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def accepts_stream_input(self):
        return self.inner.accepts_stream_input

    @property
    def independent_recognition_units(self):
        return self.inner.independent_recognition_units

    async def begin_stream(self, stream):
        await self.inner.begin_stream(stream)

    def recognition_source_covers(self, ranges):
        return self.inner.recognition_source_covers(ranges)

    async def end_stream(self, *, reason):
        await self.inner.end_stream(reason=reason)

    async def turn_events(self):
        async for event in self.inner.turn_events():
            self.trace.add("normal_adapter_result", value=event)
            yield event

    async def send_turn_audio(self, identity, pcm16le, **kwargs):
        self.trace.add(
            "native_turn_audio",
            identity=identity,
            samples=len(pcm16le) // 2,
            pcm16_sha256=hashlib.sha256(pcm16le).hexdigest(),
            value=kwargs,
        )
        await self.inner.send_turn_audio(identity, pcm16le, **kwargs)

    async def send_stream_audio(self, pcm16le, *, source_ranges):
        expected = b"".join(
            self.capture[s.normalized_start_sample * 2 : s.normalized_end_sample * 2]
            for s in source_ranges
        )
        self.trace.add(
            "native_stream_audio",
            samples=len(pcm16le) // 2,
            source_ranges=source_ranges,
            matches_declared_source_pcm=pcm16le == expected,
            pcm16_sha256=hashlib.sha256(pcm16le).hexdigest(),
        )
        await self.inner.send_stream_audio(pcm16le, source_ranges=source_ranges)


async def execute(trace, values, key):
    settings = AudioSegmentSettingsSnapshot(
        "gemini_transcribe",
        ("gemini_transcribe", adapter.GEMINI_TRANSCRIBE_STT_MODEL),
        ("issue218_cloud",),
        "manual",
        "en",
        ("en",),
        16000,
        0.3,
        600,
        500,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    backend = adapter.GeminiTranscribeSTTBackend(
        api_key=key,
        language_codes=adapter.gemini_transcribe_language_codes("en"),
        custom_vocabulary=(),
        model="gemini-3.5-transcribe-live",
        sample_rate_hz=16000,
    )

    async def factory(_settings, epoch):
        return SessionTrace(
            await backend.open_session(projection=STTSessionProjection("scoped", epoch)),
            trace,
            values,
        )

    async def result(event):
        trace.add("normal_engine_result", value=event)

    engine = ScopedRecognitionEngine(
        factory,
        channel="peer",
        event_sink=result,
        diagnostic_sink=lambda e: trace.add("engine_diagnostic", value=e),
    )

    class Sink:
        async def handle_stream_input(self, event):
            await engine.handle_stream_input(
                OwnedStreamInput(
                    event=event,
                    ledger=ledger,
                    settings=ledger.settings,
                    activation_generation=ledger.activation_generation,
                )
            )

        async def handle_owned_vad_event(self, owned):
            trace.add("owned_vad_event", value=owned)
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
                trace.add(
                    "capture_frame",
                    callback_sequence=sequence,
                    start_sample=offset,
                    end_sample=end,
                    delivery_lag_s=time.monotonic() - due,
                )
                yield AudioFrameF32(
                    values[offset:end],
                    16000,
                    capture=AudioCaptureSpan(
                        7, sequence, 16000, offset, end, started + offset / 16000, due
                    ),
                )

    try:
        await engine._ensure_session(
            settings, STTRecognitionWatchdogs(), engine._authority_generation
        )
        vad = create_peer_vad_gating(
            engine=SileroVadOnnx(bundled_silero_vad_onnx_path()),
            sample_rate_hz=16000,
            ring_buffer_ms=500,
            speech_threshold=0.3,
            hangover_ms=600,
        )
        started = time.monotonic()
        trace.add(
            "playback_begin",
            samples=int(values.size),
            vad_chunk_samples=vad.chunk_samples,
            onset_pre_roll_ms=500,
            hard_rollover_pre_roll_ms=300,
            speech_threshold=0.3,
            hangover_ms=600,
        )
        await run_audio_vad_loop(
            source=Source(),
            vad=vad,
            sink=Sink(),
            target_sample_rate_hz=16000,
            channel_label="peer",
            segment_ledger=ledger,
        )
        await engine.wait_for_event_ingress_drain()
        trace.add("normal_pipeline_eof_return", turn_boundary=engine.is_at_turn_boundary)
        if trace.last_eof is None:
            raise RuntimeError("No actual source_eof audio_stream_end observed")
        observation_end = min(trace.last_eof + 9.0, trace.start_at + 28.5)
        trace.add(
            "bounded_receive_continues", until_monotonic_s=observation_end, no_additional_audio=True
        )
        await asyncio.sleep(max(0, observation_end - time.monotonic()))
        await engine.wait_for_event_ingress_drain()
        trace.add("bounded_receive_observation_end")
    finally:
        trace.add("engine_close_begin")
        close_deadline = min(
            trace.start_at + 30 if trace.start_at else time.monotonic() + 1,
            trace.last_eof + 10 if trace.last_eof else time.monotonic() + 1,
        )
        try:
            await asyncio.wait_for(
                engine.close_backend(), max(0.001, close_deadline - time.monotonic())
            )
            trace.add("engine_close_return")
        except BaseException as error:
            trace.add("engine_close_failed", error=error)
            for socket in trace.sockets:
                socket.transport.abort()
                trace.add("websocket_transport_aborted", reason="authorized_deadline_cleanup")
            raise
        trace.add("ledger_end", value=ledger.snapshots)


def summarize(trace, expected):
    raws = [r for r in trace.rows if r["event"] == "raw_preparse_message"]
    decoded = [r for r in trace.rows if r["event"] == "sdk_decoded_message"]
    wire_finals, wire_interims = [], []
    for r in raws:
        data = json.loads(r["raw_payload"])
        sc = data.get("serverContent", data.get("server_content", {}))
        for camel, snake, target in [
            ("inputTranscription", "input_transcription", wire_finals),
            ("interimInputTranscription", "interim_input_transcription", wire_interims),
        ]:
            if camel in sc or snake in sc:
                target.append(
                    {
                        "raw_sequence": r["sequence"],
                        "at_monotonic_s": r["at_monotonic_s"],
                        "value": sc.get(camel, sc.get(snake)),
                    }
                )
    sdk_finals = [
        {
            "sdk_sequence": r["sequence"],
            "raw_sequences": r["raw_sequences"],
            "value": r["value"]["server_content"]["input_transcription"],
        }
        for r in decoded
        if r["value"].get("server_content")
        and r["value"]["server_content"].get("input_transcription") is not None
    ]
    normal = [
        r
        for r in trace.rows
        if r["event"] == "normal_adapter_result" and r["value"].get("identity", {}).get("unit_id")
    ]
    engine = [
        r
        for r in trace.rows
        if r["event"] == "normal_engine_result"
        and r["value"].get("unit")
        and r["value"].get("outcome") == "final"
    ]
    wt = [r["value"].get("text", "") for r in wire_finals]
    st = [r["value"].get("text", "") for r in sdk_finals]
    nt = [r["value"]["text"] for r in normal]
    et = [r["value"]["unit"]["text"] for r in engine]
    writes = [r for r in trace.rows if r["event"] == "adapter_send_return"]
    frontier, controls = 0, []
    for r in writes:
        frontier += r["samples"]
        if r["control"]:
            controls.append(frontier)
    drains = [r for r in trace.rows if r["event"] == "normal_drain_return"]
    drain_at = drains[0]["at_monotonic_s"] if drains else None
    return {
        "counts": trace.counts(),
        "raw_messages_including_setup": len(raws),
        "sdk_messages_excluding_setup": len(decoded),
        "wire_authoritative_finals": wire_finals,
        "wire_interim_count": len(wire_interims),
        "wire_interims": wire_interims,
        "sdk_authoritative_finals": sdk_finals,
        "normal_adapter_final_texts": nt,
        "normal_engine_final_texts": et,
        "wire_sdk_adapter_engine_text_conserved_in_order": wt == st == nt == et,
        "wire_sdk_text_conserved_in_order": wt == st,
        "all_sdk_messages_paired_one_raw": all(len(r["raw_sequences"]) == 1 for r in decoded),
        "normal_drain_return_at": drain_at,
        "source_eof_send_return_at": trace.last_eof,
        "finals_after_normal_drain_return": [
            r for r in wire_finals if drain_at and r["at_monotonic_s"] > drain_at
        ],
        "messages_after_normal_drain_return": [
            r["sequence"] for r in raws if drain_at and r["at_monotonic_s"] > drain_at
        ],
        "control_audio_frontiers": controls,
        "valid_run_control_audio_frontiers": expected["valid_run_control_frontiers"],
        "control_frontiers_match_valid_run": controls == expected["valid_run_control_frontiers"],
        "audio_write_sizes_match_valid_run": [r["samples"] for r in writes if r["samples"]]
        == expected["valid_run_audio_write_sizes"],
        "wire_pcm16_sha256": hashlib.sha256(trace.wire_pcm).hexdigest(),
        "boundary_events": [
            r
            for r in trace.rows
            if r["event"]
            in {
                "remote_start_attempt",
                "remote_start_return",
                "playback_begin",
                "native_seal_begin",
                "native_seal_return",
                "native_stream_end_begin",
                "normal_drain_return",
                "normal_pipeline_eof_return",
                "bounded_receive_continues",
                "bounded_receive_observation_end",
                "engine_close_begin",
                "adapter_close_begin",
                "websocket_recv_end",
                "websocket_close_begin",
                "websocket_close_return",
                "adapter_close_return",
                "engine_close_return",
            }
        ],
    }


def main():
    logging.disable(logging.CRITICAL)
    trace = Trace()
    report = {
        "schema": "issue218-gemini-wire-1",
        "command": [".venv/Scripts/python.exe", "native/nemotron/issue218_gemini_wire_probe.py"],
        "model": "gemini-3.5-transcribe-live",
        "credential": {
            "source": "PURIPULY_GEMINI_PROBE_ENV or repository .env.local; path not recorded",
            "variable": "GOOGLE_API_KEY2",
            "fallback": False,
            "value_or_hash_recorded": False,
            "headers_recorded": False,
        },
        "limits": {
            "remote_starts": 1,
            "submitted_audio_seconds_including_context_replay": 8,
            "overall_remote_session_seconds": 30,
            "after_last_eof_seconds": 10,
            "continued_observation_target_after_eof_seconds": 9,
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("google-genai", "websockets", "numpy", "onnxruntime")
        },
        "python": sys.version.split()[0],
        "safety": {
            "other_provider_calls": 0,
            "translation_calls": 0,
            "nemotron_used": False,
            "vrchat_steamvr_used": False,
            "product_edits": False,
            "sdk_modified_on_disk": False,
            "automatic_retries": False,
        },
        "monetary_charge": "Unknown; start/audio counts are not a billing guarantee",
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "native/nemotron/issue218_gemini_wire_probe.py",
                "src/puripuly_heart/providers/stt/gemini_transcribe.py",
                ".venv/Lib/site-packages/google/genai/live.py",
                "src/puripuly_heart/core/stt/scoped_engine.py",
                "src/puripuly_heart/core/runtime/audio_vad_loop.py",
            )
        },
    }
    expected = None
    try:
        if report["dependencies"]["google-genai"] != "2.21.0":
            raise RuntimeError("Keep SDK 2.21.0; no environment mutation authorized")
        values, expected = prepare()
        report["input"] = expected
        key = credential()
        trace.add(
            "preflight_ready",
            credential_variable="GOOGLE_API_KEY2",
            assembled_samples=int(values.size),
        )
        with instrument(trace):

            async def bounded():
                await asyncio.wait_for(execute(trace, values, key), 30.0)

            asyncio.run(bounded())
        report["status"] = "executed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = safe(error)
        trace.add("probe_failed", error=error)
    finally:
        report["summary"] = summarize(trace, expected) if expected else {"counts": trace.counts()}
        dump(OUT / "report.json", report)
        (OUT / "submitted.pcm16le").write_bytes(trace.wire_pcm)
        trace.stream.close()
        archive_path = ROOT / "native/nemotron/issue218_gemini_wire_archive3.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in (
                OUT / "trace.jsonl",
                OUT / "report.json",
                OUT / "submitted.pcm16le",
                Path(__file__),
            ):
                archive.write(path, path.name)
        print(
            json.dumps(
                safe(
                    {
                        "status": report["status"],
                        "counts": trace.counts(),
                        "wire_finals": report["summary"].get("wire_authoritative_finals"),
                        "conserved": report["summary"].get(
                            "wire_sdk_adapter_engine_text_conserved_in_order"
                        ),
                        "error": report.get("error"),
                        "report": str(OUT / "report.json"),
                        "archive": str(archive_path),
                    }
                ),
                ensure_ascii=False,
            )
        )
    return 0 if report["status"] == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
