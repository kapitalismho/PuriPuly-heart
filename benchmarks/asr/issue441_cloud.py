from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

import issue373

RATE = 16000
ARMS = ("S0", "S1", "E0", "T1", "T2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def secret(name, env_path=None):
    import keyring

    value = keyring.get_password("puripuly-heart", name)
    if value:
        return value
    env_name = {"soniox_api_key": "SONIOX_API_KEY", "elevenlabs_scribe_api_key": "ELEVENLABS_API_KEY"}[name]
    value = os.environ.get(env_name)
    if value:
        return value
    if env_path and Path(env_path).is_file():
        for line in Path(env_path).read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == env_name:
                return value.strip().strip("\"'")
    raise RuntimeError(f"credential_missing:{name}")


def source_ranges(spans):
    return [{"start_s": span.source_start_sample / span.source_sample_rate_hz,
             "end_s": span.source_end_sample / span.source_sample_rate_hz}
            for span in spans]


class Run:
    def __init__(self, args, recording):
        self.args = args
        self.recording = recording
        self.path = Path(args.output or f".tmp/issue441/runs/{args.arm}/{recording['id']}.json")
        if self.path.exists():
            raise FileExistsError(f"refusing_to_overwrite:{self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal = self.path.with_suffix(".journal.jsonl").open("x", encoding="utf-8")
        self.created = time.monotonic()
        self.last_save = self.created
        self.origin = None
        self.wire_samples = 0
        self.session_id = uuid4().hex
        self.connection_open = None
        self.connection_closed = None
        self.secrets = []
        self.data = dict(schema_version=1, arm=args.arm, recording_id=recording["id"],
                         metadata=dict(checkout=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip(),
                                       imported_checkout=str(Path(args.root).resolve()), model="stt-rt-v6" if args.arm.startswith("S") else "scribe_v3_live",
                                       input_sha256=recording["sha256"], manifest_sha256=args.freeze_sha256,
                                       config={}, host_contention=args.host_contention,
                                       completed_send_semantics="Transport coroutine completion, not server/audio acceptance unless an explicit provider receipt says so."),
                         source_origin_mono_s=None, events=[], sends=[], observations=[], snapshots=[], rows=[], session_scopes=[],
                         connected=False, session_started=False, status="running", errors=[], sent_audio_s=0.0,
                         connection_duration_s=0.0, eof={"source_reached": False}, drain={"started": False})
        self.save()

    def now(self):
        return time.monotonic() - (self.origin if self.origin is not None else self.created)

    def append_journal(self, kind, value):
        self.journal.write(json.dumps({"journal_kind": kind, "value": value}, ensure_ascii=False) + "\n")
        self.journal.flush()

    def event(self, kind, raw, *, arrival=None):
        row = dict(seq=len(self.data["events"]), arrival_s=self.now() if arrival is None else arrival,
                   kind=kind, raw=issue373.encode(raw), session_id=self.session_id,
                   clock_basis="source_origin" if self.origin is not None else "startup_origin",
                   arrival_mono_s=time.monotonic())
        self.data["events"].append(row)
        self.append_journal("event", row)
        return row

    def start_source(self):
        if self.origin is not None:
            return
        self.origin = time.monotonic()
        self.data["source_origin_mono_s"] = self.origin
        for row in self.data["events"]:
            if row["clock_basis"] == "startup_origin":
                row["arrival_s"] = row["arrival_mono_s"] - self.origin
                row["clock_basis"] = "source_origin"
        for row in self.data["rows"]:
            row["t"] += self.created - self.origin
        for row in self.data["sends"]:
            row["start_s"] = row["start_mono_s"] - self.origin
            if row["completed_mono_s"] is not None:
                row["completed_s"] = row["completed_mono_s"] - self.origin
        self.event("source_start", {"source_origin_mono_s": self.origin})
        self.save()

    def row(self, kind, **values):
        row = {"t": self.now(), "kind": kind, **{key: issue373.encode(value) for key, value in values.items()}}
        self.data["rows"].append(row)
        self.append_journal("row", row)

    def error(self, exc, stage):
        text = str(exc)
        for value in self.secrets:
            text = text.replace(value, "[redacted]")
        row = dict(stage=stage, type=type(exc).__name__, message=text, arrival_s=self.now())
        self.data["errors"].append(row)
        self.event("execution_error", row)

    def save(self):
        self.data["sent_audio_s"] = self.wire_samples / RATE
        if self.connection_open is not None:
            self.data["connection_duration_s"] = (self.connection_closed or time.monotonic()) - self.connection_open
        temporary = self.path.with_suffix(".json.writing")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        self.last_save = time.monotonic()

    async def send(self, send, payload, *, samples=0, ranges=(), kind="audio", **extra):
        start = self.wire_samples / RATE
        ranges = list(ranges)
        row = dict(source_start_s=ranges[0]["start_s"] if ranges else None,
                   source_end_s=ranges[-1]["end_s"] if ranges else None,
                   transmitted_start_s=start, transmitted_end_s=start + samples / RATE,
                   start_s=self.now(), completed_s=None, start_mono_s=time.monotonic(), completed_mono_s=None,
                   status="attempted", kind=kind, source_ranges=ranges, session_id=self.session_id, **extra)
        self.data["sends"].append(row)
        self.append_journal("send_attempt", row)
        try:
            await send(payload)
        except BaseException as exc:
            row["status"] = "failed"
            row["completed_s"] = self.now()
            row["completed_mono_s"] = time.monotonic()
            row["error_type"] = type(exc).__name__
            self.append_journal("send_result", row)
            raise
        row["status"] = "completed"
        row["completed_s"] = self.now()
        row["completed_mono_s"] = time.monotonic()
        self.wire_samples += samples
        self.append_journal("send_result", row)
        if time.monotonic() - self.last_save >= 5:
            self.save()
        return row

    def map_word(self, start, end):
        if start is None or end is None:
            return None, None, [], "missing_provider_timing"
        pieces = []
        covered = 0.0
        for row in self.data["sends"]:
            if row["status"] != "completed" or row["transmitted_end_s"] <= row["transmitted_start_s"]:
                continue
            cursor = row["transmitted_start_s"]
            for span in row["source_ranges"]:
                count = span["end_s"] - span["start_s"]
                left, right = max(start, cursor), min(end, cursor + count)
                if right > left:
                    pieces.append(dict(start_s=span["start_s"] + left - cursor,
                                       end_s=span["start_s"] + right - cursor))
                    covered += right - left
                cursor += count
        if not pieces:
            return None, None, [], "unmapped_padding_or_unsent"
        contiguous = all(abs(a["end_s"] - b["start_s"]) < 1e-5 for a, b in zip(pieces, pieces[1:]))
        full = abs(covered - (end - start)) < 1e-4
        if not contiguous or not full:
            return None, None, pieces, "discontinuous_or_partial_source_mapping"
        return pieces[0]["start_s"], pieces[-1]["end_s"], pieces, "mapped"

    def snapshot(self, event, key, words, *, operation="replace", settled=False, kind=None, segment_index=None):
        indices = []
        for index, word in enumerate(words):
            start, end = word.get("start"), word.get("end")
            start = float(start) if isinstance(start, (float, int)) and not isinstance(start, bool) else None
            end = float(end) if isinstance(end, (float, int)) and not isinstance(end, bool) else None
            mapped_start, mapped_end, ranges, mapping = self.map_word(start, end)
            observation = dict(event_seq=event["seq"], arrival_s=event["arrival_s"], snapshot_key=key,
                               revision=event["seq"], word_index=None if word.get("_segment_text") else index,
                               evidence_granularity="segment_text" if word.get("_segment_text") else "word",
                               start_s=mapped_start, end_s=mapped_end,
                               speaker=word.get("speaker_id"), text=word.get("text", ""), settled=settled,
                               kind=kind, segment_index=segment_index, provider_start_s=start, provider_end_s=end,
                               source_ranges=ranges, mapping_status=mapping, session_id=self.session_id,
                               raw=word.get("_raw", word))
            indices.append(len(self.data["observations"]))
            self.data["observations"].append(observation)
        row = dict(event_seq=event["seq"], arrival_s=event["arrival_s"], snapshot_key=key,
                   revision=event["seq"], operation=operation, observation_indices=indices,
                   segment_index=segment_index, settled=settled, kind=kind, session_id=self.session_id)
        self.data["snapshots"].append(row)
        self.append_journal("snapshot", {**row, "observations": [self.data["observations"][i] for i in indices]})

    def native(self, raw):
        event = self.event("native", raw)
        if self.args.arm.startswith("S"):
            if "tokens" not in raw:
                return event
            provisional, final = [], []
            for token in raw.get("tokens") or []:
                if token.get("text") in ("<fin>", "<end>"):
                    continue
                word = dict(text=token.get("text", ""), speaker_id=token.get("speaker"), _raw=token)
                for field in ("start", "end"):
                    value = token.get(field + "_ms")
                    word[field] = value / 1000 if isinstance(value, (int, float)) else None
                (final if token.get("is_final") else provisional).append(word)
            self.snapshot(event, "soniox:partial", provisional, kind="soniox_provisional")
            self.snapshot(event, "soniox:final", final, operation="append", settled=True, kind="soniox_final")
        else:
            kind = raw.get("message_type", "unknown")
            segment = raw.get("segment_index")
            if kind == "session_started":
                self.data["session_started"] = True
            if kind == "partial_transcript":
                words = raw.get("words") or ([{"text": raw["text"], "_segment_text": True, "_raw": raw}] if raw.get("text") else [])
                self.snapshot(event, "scribe:partial", words, kind=kind, segment_index=segment)
            elif kind in ("committed_transcript", "committed_transcript_with_timestamps", "diarized_transcript"):
                if kind == "committed_transcript":
                    self.snapshot(event, "scribe:partial", [], operation="retract", kind=kind)
                suffix = "diarized" if kind == "diarized_transcript" else kind
                key = f"scribe:segment:{segment}:{suffix}" if isinstance(segment, int) else f"scribe:unindexed:{event['seq']}"
                words = raw.get("words") or [{"text": raw.get("text", ""), "_segment_text": True, "_raw": raw}]
                self.snapshot(event, key, words, settled=True, kind=kind, segment_index=segment)
        return event

    def finish(self):
        native = [row["raw"] for row in self.data["events"] if row["kind"] == "native"]
        self.data["counts"] = dict(
            native_commit_receipts=sum(row.get("message_type") == "committed_transcript" for row in native),
            empty_native_commit_receipts=sum(row.get("message_type") == "committed_transcript" and not row.get("text") for row in native),
            cap_commands=sum(row["kind"] == "cap_commit" and row["status"] == "completed" for row in self.data["sends"]),
            eof_commands=sum(row["kind"] in ("eof", "eof_commit") and row["status"] == "completed" for row in self.data["sends"]),
            controller_commands=sum(row["kind"] == "controller_commit" and row["status"] == "completed" for row in self.data["sends"]),
            cap_commands_latest_partial_empty=sum(row["kind"] == "cap_commit" and row["status"] == "completed"
                and row["control"].get("latest_partial") is not None and not row["control"]["latest_partial"]["text"]
                for row in self.data["sends"]))
        if self.data["errors"]:
            self.data["status"] = "partial" if self.wire_samples else "failed"
        else:
            self.data["status"] = "completed"
        self.save()
        self.journal.close()


def install_controller_tap(run):
    from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController

    originals = {}
    for name in ("_seal_locked", "_request_probe", "_receive_completion"):
        originals[name] = getattr(ListenDeliveryController, name)

    async def seal(controller, segment_id, **kwargs):
        run.event("controller_seal_attempt", dict(segment_id=str(segment_id),
                  age_s=time.monotonic() - controller._opened_at_s if controller._opened_at_s else None,
                  pause_ms=controller._observed_pause_ms(), probe_status=controller._probe_status,
                  completion=issue373.encode(controller._completion), decision=controller._completion_boundary_decision,
                  source_frontier=controller._source_frontier, **kwargs))
        result = await originals["_seal_locked"](controller, segment_id, **kwargs)
        run.event("controller_seal_result", {"segment_id": str(segment_id), "sealed": result})
        return result

    def probe(controller, generation, segment_id):
        result = originals["_request_probe"](controller, generation, segment_id)
        run.event("smart_turn_probe", dict(status=controller._probe_status,
                  request=issue373.encode(controller._smart_turn_owner.snapshot)))
        return result

    async def completion(controller, value):
        run.event("smart_turn_completion", issue373.encode(value))
        return await originals["_receive_completion"](controller, value)

    ListenDeliveryController._seal_locked = seal
    ListenDeliveryController._request_probe = probe
    ListenDeliveryController._receive_completion = completion

    def restore():
        for name, original in originals.items():
            setattr(ListenDeliveryController, name, original)
    return restore


async def prepare_smart(run):
    from puripuly_heart.core.audio.smart_turn import SmartTurnInferenceOwner, bundled_smart_turn_onnx_path

    path = bundled_smart_turn_onnx_path()
    run.data["metadata"]["smart_turn_model"] = {"filename": path.name, "sha256": digest(path)}
    smart = SmartTurnInferenceOwner(model_path=path)
    smart.request_prepare()
    await smart._prepare_task
    run.event("smart_turn_ready", issue373.encode(smart.snapshot))
    if smart.snapshot.availability != "ready":
        await smart.close()
        raise RuntimeError("real_smart_turn_not_ready")
    return smart


class Source:
    terminal_reason = None

    def __init__(self, run, *, continuous=None, chunk_samples=512):
        self.run = run
        self.continuous = continuous
        self.chunk_samples = chunk_samples
        self.closed = asyncio.Event()

    async def frames(self):
        from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32

        run = self.run
        run.start_source()
        offset = sequence = 0
        with wave.open(str(run.recording["audio_path"]), "rb") as wav:
            while payload := wav.readframes(self.chunk_samples):
                count = len(payload) // 2
                end = offset + count
                await asyncio.sleep(max(0.0, run.origin + end / RATE - time.monotonic()))
                if self.continuous:
                    await self.continuous(payload, offset / RATE, end / RATE)
                samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768
                yield AudioFrameF32(samples=samples, sample_rate_hz=RATE,
                      capture=AudioCaptureSpan(capture_epoch=0, callback_sequence=sequence,
                              source_sample_rate_hz=RATE, source_start_sample=offset, source_end_sample=end,
                              source_start_monotonic_s=run.origin + offset / RATE,
                              source_end_monotonic_s=run.origin + end / RATE))
                offset, sequence = end, sequence + 1
        run.data["eof"].update(source_reached=True, arrival_s=run.now(), source_end_s=offset / RATE)
        run.event("source_eof", run.data["eof"])

    async def close(self):
        self.closed.set()


def soniox_config(language):
    return dict(model="stt-rt-v6", audio_format="pcm_s16le", sample_rate=RATE, num_channels=1,
                enable_endpoint_detection=False, enable_language_identification=False,
                enable_speaker_diarization=True, language_hints=[language])


async def run_s0(run):
    from websockets.asyncio.client import connect

    key = secret("soniox_api_key", run.args.env)
    run.secrets.append(key)
    cfg = soniox_config(run.recording["language"])
    run.data["metadata"]["config"] = cfg
    run.event("connection_start", {"endpoint": "wss://stt-rt.soniox.com/transcribe-websocket"})
    async with connect("wss://stt-rt.soniox.com/transcribe-websocket", additional_headers={"Authorization": f"Bearer {key}"},
                       ping_interval=None, open_timeout=15, max_size=None) as ws:
        run.connection_open = time.monotonic()
        run.data["connected"] = True
        await run.send(ws.send, json.dumps(cfg), kind="configuration")
        run.event("transport_ready", {"configuration_written": True, "provider_acknowledged": False})
        finished = asyncio.Event()

        async def receive():
            try:
                async for payload in ws:
                    raw = json.loads(payload)
                    run.native(raw)
                    if any(k in raw for k in ("error", "error_code", "error_type")):
                        raise RuntimeError("soniox_server_error:" + json.dumps(raw))
                    if raw.get("finished"):
                        finished.set()
            finally:
                finished.set()

        receiver = asyncio.create_task(receive())
        try:
            async def send_audio(payload, start, end):
                if receiver.done():
                    await receiver
                    raise RuntimeError("soniox_closed_before_eof")
                await run.send(ws.send, payload, samples=len(payload) // 2,
                               ranges=[dict(start_s=start, end_s=end)])

            source = Source(run, continuous=send_audio, chunk_samples=1600)
            async for _ in source.frames():
                pass
            run.data["drain"] = dict(started=True, start_s=run.now(), timeout_s=run.args.drain_s)
            await run.send(ws.send, "", kind="eof")
            run.data["eof"]["control_written"] = True
            try:
                await asyncio.wait_for(finished.wait(), run.args.drain_s)
                if receiver.done():
                    await receiver
            except TimeoutError:
                run.data["drain"]["timed_out"] = True
                raise RuntimeError("soniox_eof_drain_timeout")
            run.data["drain"].update(end_s=run.now(), finished_received=any(e["kind"] == "native" and e["raw"].get("finished") for e in run.data["events"]))
            if not run.data["drain"]["finished_received"]:
                raise RuntimeError("soniox_closed_without_finished_receipt")
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            run.connection_closed = time.monotonic()


class Scribe:
    def __init__(self, run):
        self.run = run
        self.ws = None
        self.receiver = None
        self.lock = asyncio.Lock()
        self.ready = asyncio.Event()
        self.latest_frontier = 0.0
        self.cap_frontier = 0.0
        self.committed = set()
        self.diarized = set()
        self.last_commit_seq = None
        self.last_command = None
        self.last_native_commit = None
        self.latest_partial = None
        self.cap_frontier_evidence = {"kind": "session_start", "frontier_s": 0.0}

    async def open(self):
        from urllib.parse import urlencode
        from websockets.asyncio.client import connect

        run = self.run
        key = secret("elevenlabs_scribe_api_key", run.args.env)
        run.secrets.append(key)
        cfg = dict(model_id="scribe_v3_live", audio_format="pcm_16000", diarize="true", include_timestamps="true",
                   filter_background_audio="false", multilang_mode="true",
                   include_language_detection="true", no_verbatim="false", commit_strategy={"E0": "manual", "T1": "vad", "T2": "turn_prediction"}[run.args.arm],
                   vad_silence_threshold_secs=1.5, vad_threshold=0.4, min_speech_duration_ms=100, min_silence_duration_ms=100)
        run.data["metadata"]["config"] = cfg
        run.data["metadata"]["language_policy"] = "Automatic language detection with multilang_mode=true; language_code and secondary_languages omitted in all Scribe conditions."
        run.data["metadata"]["turn_prediction_threshold"] = 0.9
        run.data["metadata"]["cap_policy"] = "Identical T1/T2 conservative client policy: source frontier=max(last manual command frontier, received committed timed-word end), checked each 100 ms source chunk. Timed word end is not server commit boundary; trailing silence can cause earlier-than-10 s cuts after native commits. Untimed commits do not reset. Silence can produce empty cap commands every 10 s. Old delayed receipts never move the frontier backward. No exact server-consumed-frontier claim; no 4/5 s cuts."
        endpoint = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?" + urlencode(cfg)
        run.event("connection_start", {"endpoint": endpoint})
        self.ws = await connect(endpoint, additional_headers={"xi-api-key": key}, open_timeout=15, max_size=None, ping_interval=20)
        run.connection_open = time.monotonic()
        run.data["connected"] = True
        self.receiver = asyncio.create_task(self.receive())
        ready = asyncio.create_task(self.ready.wait())
        try:
            done, _ = await asyncio.wait({ready, self.receiver}, timeout=15, return_when=asyncio.FIRST_COMPLETED)
            if self.receiver in done:
                await self.receiver
                raise RuntimeError("scribe_closed_before_session_started")
            if ready not in done:
                raise RuntimeError("scribe_session_started_timeout")
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    async def receive(self):
        async for payload in self.ws:
            raw = json.loads(payload)
            event = self.run.native(raw)
            kind = raw.get("message_type", "")
            if kind == "session_started":
                self.ready.set()
                self.run.data["metadata"]["effective_session_config"] = raw.get("config")
            if kind == "partial_transcript":
                self.latest_partial = {"event_seq": event["seq"], "arrival_s": event["arrival_s"],
                                       "text": raw.get("text", ""), "turn_probability": raw.get("turn_probability")}
            if kind == "committed_transcript":
                index = raw.get("segment_index")
                if isinstance(index, int):
                    if index in self.committed:
                        self.run.event("duplicate_commit_index", {"segment_index": index})
                    self.committed.add(index)
                self.last_commit_seq = event["seq"]
                self.last_native_commit = {"event_seq": event["seq"], "arrival_s": event["arrival_s"],
                                           "segment_index": index, "text_empty": not bool(raw.get("text")),
                                           "sent_frontier_s": self.latest_frontier}
                self.run.event("native_commit_control_state", dict(segment_index=index, sent_frontier_s=self.latest_frontier,
                               cap_frontier_s=self.cap_frontier, last_command=self.last_command,
                               cause="unknown_native_or_manual_race"))
            if kind == "diarized_transcript" and isinstance(raw.get("segment_index"), int):
                self.diarized.add(raw["segment_index"])
            if kind in ("committed_transcript_with_timestamps", "diarized_transcript"):
                ends = [w["end"] for w in raw.get("words") or [] if isinstance(w.get("end"), (int, float))]
                if ends:
                    frontier = max(self.cap_frontier, min(self.latest_frontier, max(ends)))
                    if frontier > self.cap_frontier:
                        self.cap_frontier = frontier
                        self.cap_frontier_evidence = {"kind": "received_committed_word_end", "frontier_s": frontier,
                                                     "event_seq": event["seq"], "arrival_s": event["arrival_s"],
                                                     "segment_index": raw.get("segment_index"), "sent_frontier_s": self.latest_frontier}
                        self.run.event("cap_frontier_advanced", self.cap_frontier_evidence)
            if kind == "error" or kind.endswith("_error") or kind in ("auth_error", "invalid_request", "quota_exceeded", "resource_exhausted", "input_error", "transcriber_error"):
                raise RuntimeError("scribe_server_error:" + json.dumps(raw))

    def check_receiver(self):
        if self.receiver.done():
            exc = self.receiver.exception()
            if exc:
                raise exc
            raise RuntimeError("scribe_closed_before_eof")

    async def audio(self, payload, start, end):
        self.check_receiver()
        async with self.lock:
            self.check_receiver()
            message = dict(message_type="input_audio_chunk", audio_base_64=base64.b64encode(payload).decode("ascii"), sample_rate=RATE, commit=False)
            await self.run.send(self.ws.send, json.dumps(message), samples=len(payload) // 2,
                                ranges=[dict(start_s=start, end_s=end)])
            self.latest_frontier = end
            if self.run.args.arm in ("T1", "T2") and end - self.cap_frontier >= 10.0 - 1e-7:
                await self._commit_locked("cap", dict(deadline_source_s=self.cap_frontier + 10.0))

    async def _commit_locked(self, reason, details):
        self.check_receiver()
        message = dict(message_type="input_audio_chunk", audio_base_64="", sample_rate=RATE, commit=True)
        command = dict(reason=reason, source_frontier_s=self.latest_frontier, last_native_commit_seq=self.last_commit_seq,
                       chosen_age_frontier_s=self.cap_frontier, frontier_evidence=self.cap_frontier_evidence,
                       last_native_commit=self.last_native_commit, latest_partial=self.latest_partial,
                       empty_uncommitted_speech="unknown; latest partial may be stale or empty", **details)
        row = await self.run.send(self.ws.send, json.dumps(message), kind=reason + "_commit", control=command)
        self.last_command = {**command, "completed_s": row["completed_s"]}
        self.cap_frontier = self.latest_frontier
        self.cap_frontier_evidence = {"kind": "manual_command", "frontier_s": self.cap_frontier,
                                      "completed_s": row["completed_s"], "reason": reason}
        self.run.event("commit_command", self.last_command)

    async def commit(self, reason, **details):
        async with self.lock:
            await self._commit_locked(reason, details)

    async def drain(self):
        run = self.run
        run.data["drain"] = dict(started=True, start_s=run.now(), timeout_s=run.args.drain_s)
        before = set(self.committed)
        await self.commit("eof")
        run.data["eof"]["control_written"] = True
        deadline = time.monotonic() + run.args.drain_s
        await asyncio.sleep(min(3.0, run.args.drain_s))
        while time.monotonic() < deadline:
            if self.receiver.done():
                await self.receiver
                break
            if self.committed - before and self.committed <= self.diarized:
                break
            await asyncio.sleep(0.1)
        run.data["drain"].update(end_s=run.now(), committed_indices=sorted(self.committed),
                                diarized_indices=sorted(self.diarized), missing_diarized_indices=sorted(self.committed - self.diarized),
                                eof_new_commit_received=bool(self.committed - before),
                                timed_out=time.monotonic() >= deadline,
                                receipt_note="No protocol EOF acknowledgement; new commit and all indexed diarization observed, or bounded timeout. Empty EOF may legitimately have no new commit.")
        if self.committed - self.diarized:
            run.error(RuntimeError("scribe_missing_indexed_diarization_after_drain"), "drain")

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.run.connection_closed = time.monotonic()
        if self.receiver:
            result = await asyncio.gather(self.receiver, return_exceptions=True)
            if isinstance(result[0], BaseException):
                self.run.error(result[0], "scribe_receive")


async def run_e0(run, scribe):
    from puripuly_heart.core.clock import SystemClock
    from puripuly_heart.core.peer_capture import PeerCaptureLanguageFacts
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
    from puripuly_heart.core.vad.gating import SpeechEnd, create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx
    from tests.core.runtime.test_peer_capture_session import make_config, make_owner

    smart = await prepare_smart(run)
    restore = install_controller_tap(run)
    source = Source(run, continuous=scribe.audio)

    class Boundaries:
        async def handle_owned_vad_event(self, owned):
            event = owned.event
            if isinstance(event, SpeechEnd):
                run.event("controller_boundary", dict(event=issue373.encode(event), segment=issue373.encode(owned.segment),
                          sent_frontier_s=scribe.latest_frontier))
                if event.reason != "source_eof":
                    await scribe.commit("controller", seal_reason=event.reason, segment_id=str(event.utterance_id))

    async def loop(**kwargs):
        await run_audio_vad_loop(**kwargs, channel_label="peer")

    owner, *_ = make_owner(source_factory=lambda *_: source, sink=Boundaries(), smart_turn_owner=smart,
             clock=SystemClock(), run_audio_loop=loop,
             vad_factory=lambda cfg: create_peer_vad_gating(SileroVadOnnx(Path(run.args.root) / "src/puripuly_heart/data/vad/silero_vad.onnx"),
                         sample_rate_hz=RATE, ring_buffer_ms=cfg.vad_pre_roll_ms, hangover_ms=cfg.vad_hangover_ms))
    cfg = dataclasses.replace(make_config(provider_id="elevenlabs_scribe", language=PeerCaptureLanguageFacts(
                              source_mode="manual", source_language=run.recording["language"])), vad_speech_threshold=0.4)
    run.data["metadata"]["controller_config"] = issue373.encode(cfg)
    try:
        await owner.apply_intent(cfg, enabled=True)
        await asyncio.wait_for(source.closed.wait(), run.recording["duration_s"] + 60)
        run.event("controller_completed", dict(snapshot=issue373.encode(owner.snapshot),
                  ledgers=[issue373.encode(ledger.snapshots) for ledger in owner.segment_ledgers], smart_turn=issue373.encode(smart.snapshot)))
        if not run.data["eof"]["source_reached"]:
            raise RuntimeError("controller_source_terminated_before_eof")
    finally:
        await owner.close()
        await smart.close()
        restore()


async def run_scribe(run):
    scribe = Scribe(run)
    try:
        await scribe.open()
        if run.args.arm == "E0":
            await run_e0(run, scribe)
        else:
            source = Source(run, continuous=scribe.audio, chunk_samples=1600)
            async for _ in source.frames():
                pass
        await scribe.drain()
    finally:
        await scribe.close()


async def application_replay(run):
    from puripuly_heart.config.prompts import get_default_prompt
    from puripuly_heart.core.clock import FakeClock
    from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner
    from puripuly_heart.core.translation_backend import TranslationBackend
    from puripuly_heart.domain.models import Translation
    from tests.helpers.fakes import RecordingOscQueue
    from tests.helpers.translation_owners import compose_translation_test_harness

    clock = FakeClock(run.origin or run.created)
    application = dict(mode="deterministic_immediate_mt_structural_replay", requests=[], captions=[], groups=[], events=[])
    run.data["application"] = application

    class Backend(TranslationBackend):
        async def translate(self, request):
            application["requests"].append(dict(t=clock.now(), request=issue373.encode(request)))
            return Translation(utterance_id=request.utterance_id, text="[deterministic structural translation] " + request.text)

        async def close(self):
            pass

    async def emit(event):
        application["events"].append(dict(t=clock.now(), event=issue373.encode(event)))

    original_prepare, original_fragment = PeerCaptionOwner._prepare, PeerCaptionOwner._final_fragment

    def prepare(owner, caption, source, kind, config):
        application["captions"].append(dict(t=clock.now(), kind=kind, source=source, caption=issue373.encode(caption)))
        return original_prepare(owner, caption, source, kind, config)

    async def fragment(owner, transcript, assignment, **kwargs):
        previous = owner._last.utterance_id if owner._last else None
        await original_fragment(owner, transcript, assignment, **kwargs)
        application["groups"].append(dict(t=clock.now(), transcript=issue373.encode(transcript), assignment=issue373.encode(assignment),
                    previous=issue373.encode(previous), accepted_caption=issue373.encode(owner._last),
                    kind=application["captions"][-1]["kind"] if application["captions"] else None))

    PeerCaptionOwner._prepare, PeerCaptionOwner._final_fragment = prepare, fragment
    harness = compose_translation_test_harness(stt=None, llm=Backend(), osc=RecordingOscQueue(), clock=clock,
               source_language=run.recording["language"], target_language="ko" if run.recording["language"] != "ko" else "en",
               system_prompt=get_default_prompt(), peer_translation_enabled=True,
               overlay_sink=SimpleNamespace(emit=emit, active_self_overlay_metadata=lambda: None))

    async def idle():
        await harness.peer_owner.cloud_captions.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        for _ in range(12):
            await asyncio.sleep(0)

    try:
        await harness.start()
        for row in run.data["rows"]:
            if row["kind"] not in ("update", "final"):
                continue
            target = (run.origin or run.created) + row["t"]
            while True:
                due = [segment.next_check_at for segment in harness.peer_owner.cloud_captions._segments.values()
                       if not segment.closed and segment.next_check_at <= target]
                if not due:
                    break
                clock.advance(max(0, min(due) - clock.now()))
                await harness.peer_owner.cloud_captions.check_previews()
                await idle()
            clock.advance(max(0, target - clock.now()))
            event = issue373.decode(row["event"])
            if row["kind"] == "update":
                await harness.peer_owner.handle_provider_update(event)
            else:
                await harness.peer_owner.handle_provider_turn_terminal(issue373.decode(row["receipt"]), event)
            await idle()
        application.update(history=issue373.encode(harness.peer_runtime.translation_history),
                           counts=issue373.encode(harness.peer_owner.cloud_captions.request_counts),
                           clock_origin_estimate=run.origin or run.created)
    finally:
        await harness.stop()
        PeerCaptionOwner._prepare, PeerCaptionOwner._final_fragment = original_prepare, original_fragment


async def run_s1(run):
    from puripuly_heart.core.clock import SystemClock
    from puripuly_heart.core.peer_capture import PeerCaptureLanguageFacts
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
    from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTProviderTurnUpdate, STTSessionProjection, PermanentSTTScopedSessionError
    from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
    from puripuly_heart.core.vad.gating import create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx
    from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend, _SonioxSession
    from tests.core.runtime.test_peer_capture_session import FakeProvider, make_config, make_owner

    key = secret("soniox_api_key", run.args.env)
    run.secrets.append(key)
    backend = SonioxRealtimeSTTBackend(api_key=key, language_hints=[run.recording["language"]])
    run.data["metadata"]["config"] = soniox_config(run.recording["language"])
    smart = await prepare_smart(run)
    restore_controller = install_controller_tap(run)
    originals = {name: getattr(_SonioxSession, name) for name in ("start", "_handle_message", "send_turn_audio", "_selective_padding_pcm16le", "seal_turn")}
    pending = deque()

    class Wire:
        def __init__(self, ws):
            self.ws = ws

        def __getattr__(self, name):
            return getattr(self.ws, name)

        async def send(self, payload):
            if isinstance(payload, bytes):
                metadata = pending.popleft() if pending else dict(kind="unmapped_audio", ranges=[])
                return await run.send(self.ws.send, payload, samples=len(payload) // 2, **metadata)
            kind = "eof" if payload == "" else "control"
            return await run.send(self.ws.send, payload, kind=kind, control=json.loads(payload) if payload else None)

    async def start(session):
        try:
            await originals["start"](session)
        finally:
            if session._session_open_at is not None:
                run.connection_open = session._session_open_at
                run.data["connected"] = True
        session._ws = Wire(session._ws)
        run.event("transport_ready", dict(configuration_written=True, provider_acknowledged=False, speaker_session_scope=session.speaker_session_scope))

    async def native(session, payload):
        run.native(json.loads(payload))
        return await originals["_handle_message"](session, payload)

    async def audio(session, identity, pcm16le, **kwargs):
        pending.append(dict(kind="context_audio" if kwargs["context_only"] else "audio",
                            ranges=source_ranges(kwargs["source_ranges"]), context_only=kwargs["context_only"],
                            identity=issue373.encode(identity), payload_sequence=kwargs["payload_sequence"]))
        return await originals["send_turn_audio"](session, identity, pcm16le, **kwargs)

    def padding(session, **kwargs):
        payload = originals["_selective_padding_pcm16le"](session, **kwargs)
        if payload:
            pending.append(dict(kind="padding", ranges=[], generated=True, seal_reason=kwargs["seal_reason"]))
        return payload

    async def seal(session, identity, **kwargs):
        run.row("seal", identity=identity, **kwargs)
        return await originals["seal_turn"](session, identity, **kwargs)

    _SonioxSession.start, _SonioxSession._handle_message = start, native
    _SonioxSession.send_turn_audio, _SonioxSession._selective_padding_pcm16le, _SonioxSession.seal_turn = audio, padding, seal
    opened = False

    async def factory(settings, epoch):
        nonlocal opened
        if opened:
            raise PermanentSTTScopedSessionError("pilot_single_session_no_automatic_reconnect")
        opened = True
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        run.data["session_scopes"].append(dict(session_id=run.session_id, speaker_session_scope=session.speaker_session_scope, provider_epoch_id=epoch))
        run.row("session", scope=session.speaker_session_scope, epoch=epoch)
        return session

    engine = ScopedRecognitionEngine(session_factory=factory, channel="peer")
    source = Source(run)
    provider_released = asyncio.Event()

    class Provider(FakeProvider):
        async def release(self, **kwargs):
            run.row("release", **kwargs)
            await engine.wait_for_event_ingress_drain()
            await engine.close()
            run.connection_closed = time.monotonic()
            await super().release(**kwargs)
            provider_released.set()

    async def loop(**kwargs):
        await run_audio_vad_loop(**kwargs, channel_label="peer")

    owner, *_ = make_owner(provider=Provider(), source_factory=lambda *_: source, clock=SystemClock(), sink=engine,
                          smart_turn_owner=smart, run_audio_loop=loop,
                          vad_factory=lambda cfg: create_peer_vad_gating(SileroVadOnnx(Path(run.args.root) / "src/puripuly_heart/data/vad/silero_vad.onnx"),
                                      sample_rate_hz=RATE, ring_buffer_ms=cfg.vad_pre_roll_ms, hangover_ms=cfg.vad_hangover_ms))

    async def event_sink(event):
        if isinstance(event, STTProviderTurnTerminal):
            run.row("provider_terminal", event=event)
            for receipt, terminal in owner.admit_provider_terminal(event):
                run.row("final", receipt=receipt, event=terminal)
        elif isinstance(event, STTProviderTurnUpdate):
            run.row("update", event=event)
        else:
            run.row("event", event=event)

    engine.bind_event_sink(event_sink)
    cfg = dataclasses.replace(make_config(language=PeerCaptureLanguageFacts(source_mode="manual", source_language=run.recording["language"])), vad_speech_threshold=0.4)
    run.data["metadata"]["controller_config"] = issue373.encode(cfg)
    try:
        settings = owner._segment_settings_snapshot(cfg)
        await engine._ensure_session(settings, engine.watchdog_resolver(settings), engine._authority_generation)
        await owner.apply_intent(cfg, enabled=True)
        await asyncio.wait_for(source.closed.wait(), run.recording["duration_s"] + 60)
        await asyncio.wait_for(provider_released.wait(), 30)
        run.data["drain"] = dict(started=True, start_s=run.data["eof"].get("arrival_s"), end_s=run.now(),
                                mode="production_capture_dispatch_and_terminal_drain",
                                terminal_count=sum(row["kind"] == "provider_terminal" for row in run.data["rows"]),
                                transport_eof_written=any(row["kind"] == "eof" and row["status"] == "completed" for row in run.data["sends"]))
        run.row("segments", ledgers=[ledger.snapshots for ledger in owner.segment_ledgers])
        run.row("completed", state=owner.snapshot)
        if not run.data["eof"]["source_reached"]:
            raise RuntimeError("production_source_terminated_before_eof")
    finally:
        await owner.close()
        await engine.close()
        await smart.close()
        run.event("smart_turn_closed", issue373.encode(smart.snapshot))
        restore_controller()
        for name, original in originals.items():
            setattr(_SonioxSession, name, original)
        await application_replay(run)


def load_recording(args):
    path = Path(args.manifest)
    if digest(path) != args.freeze_sha256:
        raise ValueError("frozen_manifest_sha256_mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported_manifest_schema")
    recording = next(row for row in manifest["recordings"] if row["id"] == args.recording)
    if args.arm in ("T1", "T2") and not recording["diagnostic"]:
        raise ValueError("turn_arms_require_diagnostic_recording")
    audio = Path(recording["audio_path"])
    if not audio.is_absolute():
        audio = Path(args.root) / audio
    recording = {**recording, "audio_path": str(audio.resolve())}
    if digest(audio) != recording["sha256"]:
        raise ValueError("frozen_audio_sha256_mismatch")
    with wave.open(str(audio), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("source_must_be_pcm16_mono_16000")
        if abs(wav.getnframes() / RATE - recording["duration_s"]) > 1 / RATE:
            raise ValueError("source_duration_mismatch")
    return recording


async def execute(args):
    recording = load_recording(args)
    run = Run(args, recording)
    try:
        await {"S0": run_s0, "S1": run_s1, "E0": run_scribe, "T1": run_scribe, "T2": run_scribe}[args.arm](run)
    except BaseException as exc:
        run.error(exc, "run")
    finally:
        run.finish()
    print(json.dumps(dict(arm=args.arm, recording_id=args.recording, status=run.data["status"], connected=run.data["connected"],
                         sent_audio_s=run.data["sent_audio_s"], connection_duration_s=run.data["connection_duration_s"],
                         native_events=sum(row["kind"] == "native" for row in run.data["events"]), errors=run.data["errors"], output=str(run.path)), ensure_ascii=False))
    return int(run.data["status"] != "completed")


async def suite(args):
    if digest(args.manifest) != args.freeze_sha256:
        raise ValueError("frozen_manifest_sha256_mismatch")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    arms = ("S0", "S1") if args.provider == "soniox" else ("E0", "T1", "T2")
    results = []
    for arm in arms:
        for recording in manifest["recordings"]:
            if arm in ("T1", "T2") and not recording["diagnostic"]:
                continue
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--root", args.root,
                       "--manifest", args.manifest, "--freeze-sha256", args.freeze_sha256,
                       "--arm", arm, "--recording", recording["id"], "--drain-s", str(args.drain_s),
                       "--host-contention", args.host_contention]
            if args.env:
                command += ["--env", args.env]
            output = Path(f".tmp/issue441/runs/{arm}/{recording['id']}.process.log")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as log:
                process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log, cwd=args.root)
                try:
                    returncode = await asyncio.wait_for(process.wait(), recording["duration_s"] + 150)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    returncode = -1
            row = dict(arm=arm, recording_id=recording["id"], returncode=returncode, log=str(output))
            results.append(row)
            print(json.dumps(row), flush=True)
    return int(any(row["returncode"] for row in results))


async def offline(args):
    from tests.core.test_smart_turn_delivery import Harness

    outcomes = []

    async def branch(label, speech_ms, pause_ms, expected_reason, score=None):
        harness = Harness(provider_id="elevenlabs_scribe")
        await harness.open()
        if speech_ms:
            await harness.feed(speech_ms, speech=True)
        if score is None:
            await harness.feed(pause_ms, speech=False)
        else:
            await harness.feed(224, speech=False)
            request = harness.inference.requests[-1]
            await harness.complete(0, score=score, at=request.probe_frontier_monotonic_s + 0.01)
            await harness.feed(288, speech=False)
            assert not harness.vad.ends
            await harness.feed(pause_ms - 512, speech=False)
        assert len(harness.vad.ends) == 1 and harness.vad.ends[0].reason == expected_reason
        await harness.controller.close()
        outcomes.append(label)

    await branch("real_controller_ordinary_512", 0, 512, "delivery_pause")
    await branch("real_controller_incomplete_800", 0, 800, "delivery_pause", score=0.1)
    await branch("real_controller_four_second_192", 4000, 192, "delivery_pause")
    await branch("real_controller_five_second_192", 5024, 192, "delivery_pause")
    await branch("real_controller_ten_second_cap", 9984, 0, "delivery_deadline")
    harness = Harness(provider_id="elevenlabs_scribe")
    segment_id = await harness.open()
    harness.clock.value = 10.1
    assert await harness.controller._seal(segment_id, reason="delivery_pause", rollover=False)
    assert harness.vad.ends[-1].reason == "delivery_deadline"
    assert not await harness.controller._seal(segment_id, reason="delivery_deadline", rollover=True)
    await harness.controller.close()
    outcomes.append("real_controller_cap_pause_race_single_terminal")
    path = Path(args.output or ".tmp/issue441/runs/S0/offline-checks.json")
    if path.exists():
        raise FileExistsError(path)
    run_args = SimpleNamespace(**{**vars(args), "arm": "T2", "output": str(path), "freeze_sha256": "offline"})
    run = Run(run_args, dict(id="offline", sha256="synthetic", language="en"))
    try:
        run.start_source()
        completed = []

        async def send(payload):
            completed.append(payload)

        async def fail(payload):
            raise OSError("synthetic_failed_transport")

        await run.send(send, b"a", samples=RATE, ranges=[dict(start_s=2.0, end_s=3.0)])
        await run.send(send, b"b", samples=RATE, ranges=[dict(start_s=2.0, end_s=3.0)], kind="context_audio")
        await run.send(send, b"c", samples=RATE // 5, kind="padding")
        with contextlib.suppress(OSError):
            await run.send(fail, b"d", samples=RATE, ranges=[dict(start_s=9.0, end_s=10.0)])
        assert run.wire_samples == RATE * 2 + RATE // 5
        assert run.map_word(0.1, 0.5)[:2] == (2.1, 2.5)
        assert run.map_word(1.1, 1.5)[:2] == (2.1, 2.5)
        assert run.map_word(2.01, 2.1)[3] == "unmapped_padding_or_unsent"
        assert run.map_word(None, None)[3] == "missing_provider_timing"
        outcomes.append("ledger_replay_padding_failed_write")
        run.native(dict(message_type="partial_transcript", text="stale", words=[dict(text="stale", start=None, end=None, speaker_id=None)]))
        run.native(dict(message_type="partial_transcript", text="", words=[]))
        assert not run.data["snapshots"][-1]["observation_indices"]
        for index in (1, 0):
            run.native(dict(message_type="committed_transcript", text="same", segment_index=index))
        for index in (0, 1):
            run.native(dict(message_type="diarized_transcript", text="same", segment_index=index,
                            words=[dict(text="same", start=0.1, end=0.5, speaker_id="speaker_2")]))
        cutoff = len(run.data["events"])
        before = json.dumps(run.data["observations"])
        run.native(dict(message_type="speakers_merged", from_speaker_id="speaker_2", to_speaker_id="speaker_1"))
        run.native(dict(message_type="speakers_merged", from_speaker_id="speaker_1", to_speaker_id="speaker_0"))
        assert json.dumps(run.data["observations"]) == before
        assert cutoff == len(run.data["events"]) - 2
        assert {row["segment_index"] for row in run.data["snapshots"] if row["kind"] == "diarized_transcript"} == {0, 1}
        outcomes.append("indexed_interleaving_empty_revision_late_merge_immutable")
        scribe = Scribe(run)
        scribe.ws = SimpleNamespace(send=send)
        scribe.receiver = asyncio.create_task(asyncio.sleep(100))
        scribe.latest_frontier = 9.9
        scribe.cap_frontier = 0
        await scribe.audio(bytes(3200), 9.9, 10.0)
        assert run.data["sends"][-1]["kind"] == "cap_commit"
        assert scribe.cap_frontier == 10.0
        scribe.committed.add(4)
        scribe.last_commit_seq = 55
        await scribe.commit("eof")
        assert run.data["sends"][-1]["control"]["last_native_commit_seq"] == 55
        assert not any(row["kind"] == "controller_commit" for row in run.data["sends"])
        scribe.receiver.cancel()
        await asyncio.gather(scribe.receiver, return_exceptions=True)
        outcomes.append("cap_native_eof_race_journal_no_opportunity_cuts")
        await application_replay(run)
        assert not run.data["application"]["captions"]
        outcomes.append("zero_output_structural_replay_explicit_origin")
        run.data["offline_checks"] = outcomes
    except BaseException as exc:
        run.error(exc, "offline")
        raise
    finally:
        run.finish()
    print(json.dumps({"passed": outcomes, "output": str(path)}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "suite", "offline"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="benchmarks/asr/issue441-inputs.json")
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--recording")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--provider", choices=("soniox", "scribe"))
    parser.add_argument("--env")
    parser.add_argument("--output")
    parser.add_argument("--drain-s", type=float, default=12.0)
    parser.add_argument("--host-contention", default="Soniox and Scribe may run concurrently; local Nemotron and corpus automatic voice checks may contend for host CPU/GPU. Same-provider sessions serial.")
    args = parser.parse_args()
    args.root = str(Path(args.root).resolve())
    sys.path[:0] = [str(Path(args.root) / "src"), args.root]
    os.chdir(args.root)
    if args.mode != "offline" and not args.freeze_sha256:
        parser.error("--freeze-sha256 is required after Director's explicit input freeze")
    raise SystemExit(asyncio.run({"run": execute, "suite": suite, "offline": offline}[args.mode](args)))


if __name__ == "__main__":
    main()
