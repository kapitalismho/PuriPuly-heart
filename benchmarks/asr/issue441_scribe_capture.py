from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import json
import math
import time
import wave
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlsplit

if __package__:
    from . import issue441_cloud as cloud
    from . import issue441_records as records
    from .issue441_lexical import canonical_hash
    from .issue441_scribe_fresh import Application
    from .issue441_scribe_score import (
        ARM,
        DRAIN_S,
        MANIFEST,
        VAD,
        WIRE,
        candidate_import_root,
        make_freeze,
    )
else:
    import issue441_cloud as cloud
    import issue441_records as records
    from issue441_lexical import canonical_hash
    from issue441_scribe_fresh import Application
    from issue441_scribe_score import (
        ARM,
        DRAIN_S,
        MANIFEST,
        VAD,
        WIRE,
        candidate_import_root,
        make_freeze,
    )


RATE = cloud.RATE
PROVIDER = "elevenlabs_scribe"


def map_word(sends, session, observed, start, end):
    if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
        return None, None, [], "missing_provider_timing"
    if start < 0 or end <= start:
        return None, None, [], "invalid_provider_timing"
    pieces = []
    cursor = start
    complete = True
    for row in sends:
        if (
            row["session_id"] != session
            or row["status"] != "completed"
            or row["completed_mono_s"] > observed
            or row["transmitted_end_s"] <= start
            or row["transmitted_start_s"] >= end
        ):
            continue
        offset = row["transmitted_start_s"]
        for span in row["source_ranges"]:
            high = offset + (span["end_s"] - span["start_s"])
            lo, hi = max(start, offset), min(end, high)
            if hi > lo + 1e-10:
                piece = {
                    "start_s": span["start_s"] + (lo - offset),
                    "end_s": span["start_s"] + (hi - offset),
                }
                complete &= abs(lo - cursor) < 1e-8
                if pieces:
                    complete &= abs(piece["start_s"] - pieces[-1]["end_s"]) < 1e-8
                pieces.append(piece)
                cursor = hi
            offset = high
    if not pieces:
        return None, None, [], "unmapped_padding_or_unsent"
    if not complete or abs(cursor - end) >= 1e-8:
        return None, None, pieces, "discontinuous_or_partial_source_mapping"
    return pieces[0]["start_s"], pieces[-1]["end_s"], pieces, "mapped"


class Capture(cloud.Run):
    def __init__(self, args, recording):
        super().__init__(args, recording)
        self._receipt = None
        self.write_context = ContextVar("scribe_write_context", default=None)
        self.send_lock = asyncio.Lock()
        self.source = None
        self.data["transport_sessions"] = []
        self.metadata_observed = asyncio.Event()
        self.data["metadata"].update(
            live_provider=args.live_provider,
            measurement_session_id=self.session_id,
            source_selection="product_vad",
        )
        with wave.open(recording["audio_path"], "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
        from puripuly_heart.core.audio.format import (
            float32_to_pcm16le_bytes,
            pcm16le_bytes_to_float32,
        )

        self.normalized_pcm = float32_to_pcm16le_bytes(pcm16le_bytes_to_float32(pcm))

    def native(self, raw, *, arrival_mono, session_id):
        self._receipt = session_id, arrival_mono
        try:
            return super().native(raw, arrival_mono=arrival_mono, session_id=session_id)
        finally:
            self._receipt = None
            self.metadata_observed.set()

    def map_word(self, start, end):
        session_id, observed = self._receipt
        return map_word(self.data["sends"], session_id, observed, start, end)

    def save(self):
        self.data["connection_duration_s"] = sum(
            (row.get("closed_mono_s") or time.monotonic()) - row["connected_mono_s"]
            for row in self.data.get("transport_sessions", [])
            if row.get("connected_mono_s") is not None
        )
        super().save()

    def index_status(self):
        commits, diarized = {}, set()
        for row in self.data["events"]:
            if row["kind"] != "native":
                continue
            raw = row["raw"]
            index = raw.get("segment_index")
            if type(index) is not int:
                continue
            key = row["session_id"], index
            if raw.get("message_type") == "committed_transcript":
                commits[key] = bool(raw.get("text"))
            elif raw.get("message_type") == "diarized_transcript":
                diarized.add(key)
        missing = {key for key, nonempty in commits.items() if nonempty and key not in diarized}
        return commits, diarized, missing

    def account_audio(self):
        source_samples = self.source.released_samples if self.source is not None else 0
        ranges = []
        padding = 0
        for row in self.data["sends"]:
            if row["status"] != "completed":
                continue
            ranges.extend(
                (r["source_start_sample"], r["source_end_sample"]) for r in row["source_ranges"]
            )
            padding += row.get("generated_samples", 0)
        union = []
        for start, end in sorted(ranges):
            if union and start <= union[-1][1]:
                union[-1][1] = max(end, union[-1][1])
            else:
                union.append([start, end])
        omitted = []
        cursor = 0
        for start, end in union:
            if start > cursor:
                omitted.append([cursor, start])
            cursor = end
        if cursor < source_samples:
            omitted.append([cursor, source_samples])
        unique = sum(end - start for start, end in union)
        repeats = sum(end - start for start, end in ranges) - unique
        self.data["source_to_transmitted"] = {
            "source_samples": source_samples,
            "completed_tx_samples": self.wire_samples,
            "unique_source_samples": unique,
            "repeated_source_samples": repeats,
            "generated_samples": padding,
            "omitted_source_samples": sum(end - start for start, end in omitted),
            "omitted_source_sample_ranges": omitted,
            "covered_source_sample_ranges": union,
        }
        if unique + repeats + padding != self.wire_samples:
            self.error(RuntimeError("completed_tx_has_unaccounted_source"), "audio_accounting")


class RawWire:
    def __init__(self, run, socket, transport):
        self.run = run
        self.socket = socket
        self.closing = False
        self.transport = transport
        self.transmitted_samples = 0

    @property
    def session_id(self):
        return self.transport["session_id"]

    def __getattr__(self, name):
        return getattr(self.socket, name)

    async def recv(self):
        try:
            raw = await self.socket.recv()
        except Exception as exc:
            self.run.event(
                "transport_recv_ended",
                {"error_type": type(exc).__name__, "closing": self.closing},
                session_id=self.session_id,
            )
            if type(exc).__name__ != "ConnectionClosedOK":
                self.run.error(exc, "transport_recv")
            elif not self.closing:
                self.run.error(RuntimeError("peer_closed_before_local_close"), "transport_recv")
            raise
        observed = time.monotonic()
        try:
            parsed = json.loads(raw)
        except ValueError, RecursionError:
            self.run.event("malformed_frame", {"type": type(raw).__name__}, arrival_mono=observed)
            return raw
        if not isinstance(parsed, dict):
            self.run.event(
                "malformed_frame", {"type": type(parsed).__name__}, arrival_mono=observed
            )
            return raw
        if (
            parsed.get("message_type") == "session_started"
            and self.transport.get("session_started_mono_s") is None
        ):
            session_id = parsed.get("session_id")
            if isinstance(session_id, str) and session_id:
                self.transport.update(
                    session_id=session_id,
                    session_started_mono_s=observed,
                    effective_session_config_echo=parsed.get("config"),
                )
        if self.transport.get("session_started_mono_s") is not None:
            self.run.native(parsed, arrival_mono=observed, session_id=self.session_id)
        else:
            self.run.event(
                "transport_pre_session_frame",
                parsed,
                arrival_mono=observed,
                session_id=self.session_id,
            )
        return raw

    async def send(self, payload):
        run = self.run
        msg = json.loads(payload)
        data = base64.b64decode(msg.get("audio_base_64", ""), validate=True)
        if len(data) % 2:
            raise ValueError("raw_scribe_payload_not_pcm16")
        samples = len(data) // 2
        context = run.write_context.get()
        ranges = []
        generated_samples = 0
        kind = "keepalive"
        control = {key: value for key, value in msg.items() if key != "audio_base_64"}
        if context and context["kind"] == "audio":
            if data != context["expected_payload"] or control.get("commit") is True:
                raise ValueError("scoped_audio_does_not_match_raw_write")
            ranges = context["source_ranges"]
            expected = b"".join(
                run.normalized_pcm[2 * r["source_start_sample"] : 2 * r["source_end_sample"]]
                for r in ranges
            )
            if expected != data:
                raise ValueError("raw_write_does_not_match_mapped_normalized_pcm")
            kind = "context_audio" if context["context_only"] else "audio"
        elif control.get("commit") is True:
            if not context or context["kind"] != "seal":
                raise ValueError("scribe_commit_without_scoped_seal")
            if data and any(data):
                raise ValueError("commit_payload_has_unmapped_nonzero_audio")
            generated_samples = samples
            kind = "eof_commit" if context["seal_reason"] == "source_eof" else "controller_commit"
        elif data:
            if any(data):
                raise ValueError("unmapped_nonzero_scribe_write")
            generated_samples = samples
            kind = "generated_padding"
        fields = {
            k: v
            for k, v in (context or {}).items()
            if k not in {"kind", "expected_payload", "source_ranges"}
        }
        async with run.send_lock:
            row = await run.send(
                self.socket.send,
                payload,
                samples=samples,
                ranges=ranges,
                kind=kind,
                control=control,
                pcm_sha256=hashlib.sha256(data).hexdigest() if data else None,
                generated_samples=generated_samples,
                session_id=self.session_id,
                transmitted_start_samples=self.transmitted_samples,
                **fields,
            )
            self.transmitted_samples += samples
            return row

    async def close(self):
        self.closing = True
        self.run.event("transport_close_start", {}, session_id=self.session_id)
        try:
            await self.socket.close()
            close_code = getattr(self.socket, "close_code", None)
            if close_code != 1000:
                self.run.error(
                    RuntimeError(f"close_handshake_code:{close_code}"), "transport_close"
                )
        except Exception as exc:
            self.run.error(exc, "transport_close")
            raise
        finally:
            self.transport.update(
                closed_mono_s=time.monotonic(),
                close_code=getattr(self.socket, "close_code", None),
                completed_tx_samples=self.transmitted_samples,
            )
            self.run.metadata_observed.set()
            self.run.event(
                "transport_close_end",
                {
                    "close_code": getattr(self.socket, "close_code", None),
                    "receipt_note": "No Scribe protocol EOF acknowledgement",
                },
                session_id=self.session_id,
            )


class SourceTap:
    def __init__(self, run, engine):
        self.run = run
        self.engine = engine

    def __getattr__(self, name):
        return getattr(self.engine, name)

    async def handle_owned_vad_event(self, owned):
        self.run.row(
            "owned_vad",
            event=owned.event,
            identity=owned.segment.identity,
            settings=owned.segment.settings,
            state=owned.segment.state,
            seal_reason=owned.segment.seal_reason,
        )
        await self.engine.handle_owned_vad_event(owned)


class ScopedTap:
    def __init__(self, run, session):
        self.run = run
        self.session = session

    def __getattr__(self, name):
        return getattr(self.session, name)

    async def begin_turn(self, request):
        self.run.row("begin_turn", request=request, session_id=self.session._session_id)
        await self.session.begin_turn(request)

    async def send_turn_audio(self, identity, pcm16le, **kwargs):
        ranges = []
        for span in kwargs["source_ranges"]:
            if span.source_sample_rate_hz != RATE or span.capture_epoch != 0:
                raise ValueError("unexpected_source_provenance")
            start, end = span.source_start_sample, span.source_end_sample
            ranges.append(
                {
                    "start_s": start / RATE,
                    "end_s": end / RATE,
                    "source_start_sample": start,
                    "source_end_sample": end,
                }
            )
        token = self.run.write_context.set(
            {
                "kind": "audio",
                "identity": records.encode(identity),
                "payload_sequence": kwargs["payload_sequence"],
                "context_only": kwargs["context_only"],
                "source_ranges": ranges,
                "expected_payload": pcm16le,
            }
        )
        try:
            return await self.session.send_turn_audio(identity, pcm16le, **kwargs)
        finally:
            self.run.write_context.reset(token)

    async def seal_turn(self, identity, **kwargs):
        self.run.row("seal", identity=identity, **kwargs)
        token = self.run.write_context.set(
            {
                "kind": "seal",
                "identity": records.encode(identity),
                "seal_reason": kwargs["seal_reason"],
                "observed_trailing_silence_ms": kwargs["observed_trailing_silence_ms"],
            }
        )
        try:
            return await self.session.seal_turn(identity, **kwargs)
        finally:
            self.run.write_context.reset(token)


class LocalWire:
    close_code = 1000

    def __init__(self, attempt):
        self.inbox = asyncio.Queue()
        self.inbox.put_nowait(
            {
                "message_type": "session_started",
                "session_id": f"synthetic-{attempt}",
                "config": WIRE,
            }
        )
        self.samples = 0
        self.previous_commit = 0
        self.index = 0
        self.tasks = set()

    async def recv(self):
        return json.dumps(await self.inbox.get())

    async def send(self, frame):
        raw = json.loads(frame)
        self.samples += len(base64.b64decode(raw["audio_base_64"])) // 2
        if not raw.get("commit"):
            return
        index = self.index
        start = self.previous_commit / RATE
        end = min(self.samples / RATE, start + 0.016)
        self.previous_commit = self.samples
        self.index += 1
        self.inbox.put_nowait(
            {"message_type": "committed_transcript", "segment_index": index, "text": "Preflight."}
        )

        async def delayed():
            await asyncio.sleep(0.025)
            self.inbox.put_nowait(
                {
                    "message_type": "diarized_transcript",
                    "segment_index": index,
                    "text": "Preflight.",
                    "words": [
                        {
                            "text": "Preflight.",
                            "start": start,
                            "end": end,
                            "speaker_id": "speaker_0",
                            "type": "word",
                        }
                    ],
                }
            )

        task = asyncio.create_task(delayed())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def close(self):
        if self.tasks:
            await asyncio.gather(*self.tasks)


async def capture(run):
    from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import (
        _recognition_retention_profile,
        _recognition_watchdogs,
        _session_lifetime_enabled,
    )
    from puripuly_heart.config.runtime_resolution import STT_DEFAULT_DRAIN_TIMEOUT_S
    from puripuly_heart.core import network_clients
    from puripuly_heart.core.clock import SystemClock
    from puripuly_heart.core.peer_capture import PeerCaptureLanguageFacts
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
    from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTSessionProjection
    from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
    from puripuly_heart.core.vad.gating import create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx
    from puripuly_heart.providers.stt.elevenlabs_scribe import ElevenLabsScribeSTTBackend
    from tests.core.runtime.test_peer_capture_session import FakeProvider, make_config, make_owner

    key = (
        cloud.secret("elevenlabs_scribe_api_key")
        if run.args.live_provider
        else "synthetic-local-only"
    )
    run.secrets.append(key)

    async def websocket_factory(url, **kwargs):
        transport = {
            "attempt": len(run.data["transport_sessions"]) + 1,
            "connect_started_mono_s": time.monotonic(),
        }
        transport["session_id"] = f"unstarted:{transport['attempt']}"
        run.data["transport_sessions"].append(transport)
        query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
        run.data["metadata"]["config"] = query
        if query != WIRE:
            raise ValueError(f"product_wire_config_differs_from_frozen_profile:{query}")
        run.event(
            "connection_start",
            {"endpoint": url, "controlled_transport": not run.args.live_provider},
            session_id=transport["session_id"],
        )
        try:
            wire = (
                await network_clients.external_websocket_connect(url, **kwargs)
                if run.args.live_provider
                else LocalWire(transport["attempt"])
            )
        except Exception as exc:
            run.error(exc, "connection_setup")
            raise
        transport["connected_mono_s"] = time.monotonic()
        run.data["connected"] = True
        run.event("transport_connected", {}, session_id=transport["session_id"])
        return RawWire(run, wire, transport)

    backend = ElevenLabsScribeSTTBackend(
        api_key=key, diarize=True, websocket_factory=websocket_factory
    )
    source = cloud.Source(run, sample_limit=None if run.args.live_provider else 8 * RATE)
    run.source = source
    provider_done = asyncio.Event()
    options = SimpleNamespace(
        provider=PROVIDER,
        channel="peer",
        sample_rate_hz=RATE,
        drain_timeout_s=STT_DEFAULT_DRAIN_TIMEOUT_S,
        provider_options={},
    )
    watchdogs = _recognition_watchdogs(options)

    async def open_scoped(_settings, epoch):
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        session_id = session._session_id
        run.data["session_scopes"].append({"session_id": session_id, "provider_epoch_id": epoch})
        run.event("product_adapter_ready", {"provider_epoch_id": epoch}, session_id=session_id)
        return ScopedTap(run, session)

    clock = SystemClock()
    engine = ScopedRecognitionEngine(
        session_factory=open_scoped,
        channel="peer",
        monotonic_clock=clock.now,
        watchdog_resolver=lambda _: watchdogs,
        retention_profile_resolver=lambda settings: _recognition_retention_profile(
            options, settings
        ),
        event_drain_timeout_s=options.drain_timeout_s,
        session_lifetime_enabled=_session_lifetime_enabled(options),
        backend_close=backend.close,
    )
    run.data["metadata"]["engine_watchdogs"] = records.encode(watchdogs)
    run.data["metadata"]["engine_session_lifetime_enabled"] = engine.session_lifetime_enabled

    class CaptureProvider(FakeProvider):
        async def release(self, **kwargs):
            run.row("release", **kwargs)
            try:
                if kwargs["mode"] == "drain" and run.data["eof"]["source_reached"]:
                    run.data["drain"].update(started=True, start_s=run.now())
                    pending = tuple(engine._terminal_wait_tasks)
                    if pending:
                        _, unsettled = await asyncio.wait(
                            pending, timeout=watchdogs.final_timeout_s + watchdogs.drain_timeout_s
                        )
                        if unsettled:
                            raise RuntimeError("owned_text_settlement_did_not_finish")
                    await asyncio.wait_for(
                        engine.wait_for_event_ingress_drain(), watchdogs.drain_timeout_s
                    )
                    window = DRAIN_S if run.args.live_provider else 0.15
                    run.data["drain"].update(text_settled_s=run.now(), metadata_window_s=window)
                    deadline = time.monotonic() + window
                    while True:
                        run.metadata_observed.clear()
                        indexed, diarized, missing = run.index_status()
                        live = {
                            row["session_id"]
                            for row in run.data["transport_sessions"]
                            if row.get("session_started_mono_s") is not None
                            and not row.get("closed_mono_s")
                        }
                        remaining = deadline - time.monotonic()
                        if (
                            not any(session_id in live for session_id, _ in missing)
                            or remaining <= 0
                        ):
                            break
                        try:
                            await asyncio.wait_for(run.metadata_observed.wait(), remaining)
                        except TimeoutError:
                            pass
                    await asyncio.wait_for(
                        engine.wait_for_event_ingress_drain(), watchdogs.drain_timeout_s
                    )
                    completed_commits = [
                        row
                        for row in run.data["sends"]
                        if row["status"] == "completed"
                        and row["kind"] in {"eof_commit", "controller_commit"}
                    ]
                    native_terminals = [
                        row
                        for row in run.data["rows"]
                        if row["kind"] == "scoped_event"
                        and row["event"].get("$type", "").endswith(":STTProviderTurnTerminal")
                        and row["event"]["fields"].get("native_segment") is not None
                    ]
                    run.data["eof"]["control_written"] = any(
                        row["kind"] == "eof_commit" for row in completed_commits
                    )
                    run.data["drain"].update(
                        end_s=run.now(),
                        indexed_commits=sorted(indexed),
                        indexed_diarization=sorted(diarized),
                        nonempty_missing_diarization=sorted(missing),
                        completed_commit_writes=len(completed_commits),
                        scoped_indexed_terminal_events=len(native_terminals),
                        observed_only="Bounded experiment observation, no protocol EOF acknowledgement or complete late-event guarantee",
                    )
                    if missing:
                        run.error(
                            RuntimeError("indexed_diarization_unseen_within_observation"),
                            "bounded_drain",
                        )
                    if len(completed_commits) != len(native_terminals):
                        run.error(
                            RuntimeError("incomplete_indexed_terminals_at_drain"), "bounded_drain"
                        )
                await engine.close_backend_and_wait()
                await super().release(**kwargs)
            except Exception as exc:
                run.error(exc, "capture_release")
                raise
            finally:
                provider_done.set()

    async def audio_loop(**kwargs):
        await run_audio_vad_loop(**kwargs, channel_label="peer")

    smart = await cloud.prepare_smart(run)
    restore = cloud.install_controller_tap(run)
    silero = Path(run.args.root) / "src/puripuly_heart/data/vad/silero_vad.onnx"
    owner, *_ = make_owner(
        provider=CaptureProvider(),
        source_factory=lambda *_: source,
        sink=SourceTap(run, engine),
        smart_turn_owner=smart,
        clock=clock,
        run_audio_loop=audio_loop,
        vad_factory=lambda cfg: create_peer_vad_gating(
            SileroVadOnnx(silero),
            sample_rate_hz=RATE,
            ring_buffer_ms=cfg.vad_pre_roll_ms,
            hangover_ms=cfg.vad_hangover_ms,
        ),
    )

    application = Application(
        run.recording, clock, lambda: run.origin or run.created, journal=run.append_journal
    )
    run.data["application"] = application.trace

    async def event_sink(event):
        row_index = len(run.data["rows"])
        run.row("scoped_event", event=event)
        if isinstance(event, STTProviderTurnTerminal):
            await dispatch(event, row_index)
            if event.outcome not in {"final", "empty", "suppressed"}:
                run.error(
                    RuntimeError(event.failure_reason or event.outcome), "scoped_turn_terminal"
                )
        else:
            await dispatch(event, row_index)

    cfg = dataclasses.replace(
        make_config(
            provider_id=PROVIDER,
            language=PeerCaptureLanguageFacts(
                source_mode="auto", source_language=run.recording["language"]
            ),
        ),
        vad_speech_threshold=VAD["threshold"],
        vad_hangover_ms=VAD["hangover_ms"],
        vad_pre_roll_ms=VAD["pre_roll_ms"],
    )
    cfg = dataclasses.replace(
        cfg,
        runtime_signature=(
            *cfg.runtime_signature[:3],
            VAD["threshold"],
            VAD["hangover_ms"],
            VAD["pre_roll_ms"],
        ),
    )
    run.data["metadata"]["controller_config"] = records.encode(cfg)
    run.data["metadata"]["silero_sha256"] = cloud.digest(silero)
    try:
        await application.__aenter__()
        dispatch = application.bind_live_capture(owner, run)
        engine.bind_event_sink(event_sink)
        settings = owner._segment_settings_snapshot(cfg)
        await engine._ensure_session(
            settings, engine.watchdog_resolver(settings), engine._authority_generation
        )
        await owner.apply_intent(cfg, enabled=True)
        await asyncio.wait_for(source.closed.wait(), run.recording["duration_s"] + 30)
        await asyncio.wait_for(provider_done.wait(), 180)
        run.row(
            "segments",
            ledgers=[ledger.snapshots for ledger in owner.segment_ledgers],
            receipts=[ledger.terminal_receipts for ledger in owner.segment_ledgers],
        )
        run.row("completed", state=owner.snapshot)
        if not run.data["eof"]["source_reached"] or not run.data["drain"]["started"]:
            raise RuntimeError("capture_did_not_reach_finite_drain")
        if not run.data["session_started"]:
            raise RuntimeError("missing_product_scribe_session_ready")
    finally:
        restore()
        for cleanup in (owner.close, engine.close_backend_and_wait, smart.close):
            try:
                await cleanup()
            except Exception as exc:
                run.error(exc, f"cleanup:{cleanup.__qualname__}")
        try:
            await application.settle()
        except Exception as exc:
            run.error(exc, "application_settlement")
        finally:
            try:
                await application.__aexit__()
            except Exception as exc:
                run.error(exc, "application_cleanup")
        if run.origin is not None:
            for row in application.trace["admissions"]:
                row["t"] = row["mono"] - run.origin


def configure(args):
    args.root = args.root.resolve()
    args.arm = ARM
    args.manifest = args.root / MANIFEST
    args.freeze_sha256 = cloud.digest(args.manifest)
    args.host_contention = "Serial Scribe clips; no parallel benchmark or GUI load requested"
    args.live_provider = args.mode == "capture"
    candidate_import_root(args.root, args.candidate)
    frozen = records.load(args.frozen)
    if frozen != make_freeze(args.root):
        raise ValueError("frozen comparison differs from current inputs or method")
    args.source_method_sha256 = canonical_hash(frozen)
    args.source_method_file_sha256 = cloud.digest(args.frozen)


async def run_one(args, recording_id, output):
    settings = SimpleNamespace(**vars(args))
    settings.recording = recording_id
    settings.output = str(output)
    recording = cloud.load_recording(settings)
    run = Capture(settings, recording)
    run.data["metadata"].update(
        source_method_sha256=args.source_method_sha256,
        source_method_file_sha256=args.source_method_file_sha256,
    )
    try:
        await capture(run)
    except Exception as exc:
        run.error(exc, "runner")
    try:
        if not args.live_provider:
            raw = run.data["events"]
            commits = {
                (row["session_id"], row["raw"].get("segment_index")): row["arrival_s"]
                for row in raw
                if row["kind"] == "native"
                and row["raw"].get("message_type") == "committed_transcript"
            }
            delayed_mapped = [
                word
                for word in run.data["observations"]
                if word["kind"] == "diarized_transcript"
                and word["mapping_status"] == "mapped"
                and word["arrival_s"]
                > commits.get((word["session_id"], word["segment_index"]), math.inf)
            ]
            scoped_revisions = [
                row["event"]
                for row in run.data["rows"]
                if row["kind"] == "scoped_event"
                and row["event"].get("$type", "").endswith(":STTFinalSpeakerEvidence")
            ]
            final_receipts = [
                row
                for row in run.data["rows"]
                if row["kind"] == "final" and row["event"]["fields"]["outcome"] == "final"
            ]
            app = run.data.get("application", {})
            connected = (
                bool(app.get("snapshots"))
                and bool(app.get("requests"))
                and any(row["kind"] == "speaker_evidence" for row in app.get("admissions", []))
            )
            if not delayed_mapped or not scoped_revisions or not final_receipts or not connected:
                run.error(
                    RuntimeError(
                        "preflight_missing_mapped_late_evidence_or_authoritative_owned_final"
                    ),
                    "preflight",
                )
        run.account_audio()
    finally:
        run.finish()
    print(
        json.dumps(
            {"recording_id": recording_id, "status": run.data["status"], "output": str(output)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return run.data["status"] == "completed"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "capture"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(args)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "preflight":
        return int(not await run_one(args, "en", args.output / "en.json"))
    results = []
    for recording in records.load(args.manifest)["recordings"]:
        ident = recording["id"]
        results.append(await run_one(args, ident, args.output / f"{ident}.json"))
    return int(not all(results))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
