"""Bounded #264 R4 differentials; raw captures remain local-only."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextvars
import hashlib
import importlib.metadata
import json
import logging
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import issue218_gemini_wire_probe as prior
from google.genai import types
from google.genai.live import AsyncSession

from puripuly_heart.providers.stt import gemini_transcribe as adapter
from puripuly_heart.providers.stt.genai_network import _LiveRedirectGuard

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/nemotron"
CANDIDATE = "cc75792e9704e8609f38a3aa0794c03a1c7e3416"
EXPECTED_PCM = "1484712bf62ace51138eafb48e3525c78ce4ab49c741e0cc6eac3e1ec4cae22a"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(
        json.dumps(prior.safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def prepare():
    archive_path = BASE / "issue218_gemini_wire_archive3.zip"
    with zipfile.ZipFile(archive_path) as archive:
        raw_trace = archive.read("trace.jsonl")
        pcm = archive.read("submitted.pcm16le")
        report = json.loads(archive.read("report.json"))
    rows = [json.loads(line) for line in raw_trace.decode().splitlines()]
    origin = next(r["at_monotonic_s"] for r in rows if r["event"] == "playback_begin")
    schedule = []
    frontier = 0
    for row in rows:
        if row["event"] != "adapter_send_begin":
            continue
        size = row["samples"]
        chunk = pcm[2 * frontier : 2 * (frontier + size)]
        if size and digest(chunk) != row["pcm16_sha256"]:
            raise RuntimeError("Recorded chunk hash mismatch")
        schedule.append(
            {
                "due_s": row["at_monotonic_s"] - origin,
                "samples": size,
                "start_sample": frontier,
                "control": row["control"],
                "reason": row["reason"],
                "source_sequence": row["sequence"],
            }
        )
        frontier += size
    if frontier != 93404 or digest(pcm) != EXPECTED_PCM:
        raise RuntimeError("Approved PCM mismatch")
    if [r["start_sample"] for r in schedule if r["control"]] != [75584, 93404]:
        raise RuntimeError("Recorded control intervals mismatch")
    versions = {name: importlib.metadata.version(name) for name in report["dependencies"]}
    if versions != report["dependencies"] or sys.version.split()[0] != report["python"]:
        raise RuntimeError("Baseline dependencies changed; no call authorized by this driver")
    return (
        pcm,
        schedule,
        {
            "archive_sha256": digest(archive_path.read_bytes()),
            "trace_sha256": digest(raw_trace),
            "pcm16_sha256": digest(pcm),
            "samples": frontier,
            "dependencies": versions,
            "python": sys.version.split()[0],
        },
    )


class Trace:
    def __init__(self, folder):
        folder.mkdir()
        self.folder = folder
        self.stream = (folder / "trace.jsonl").open("x", encoding="utf-8", newline="\n")
        self.rows = []
        self.starts = self.reserved = self.written = 0
        self.wire_pcm = bytearray()
        self.sockets = set()
        self.active = contextvars.ContextVar("raw_sequence", default=None)
        self.outgoing_pcm = b""
        self.eof = self.start = None

    def add(self, event, **fields):
        row = {
            "sequence": len(self.rows) + 1,
            "event": event,
            "at_monotonic_s": time.monotonic(),
            **fields,
        }
        self.rows.append(row)
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()
        return row["sequence"]


@contextmanager
def instrument(trace):
    recv, send, decode, close = (
        _LiveRedirectGuard.recv,
        _LiveRedirectGuard.send,
        AsyncSession._receive,
        _LiveRedirectGuard.close,
    )

    async def tapped_recv(self, *args, **kwargs):
        trace.sockets.add(self)
        raw = await recv(self, *args, **kwargs)
        payload = raw.decode() if isinstance(raw, bytes) else raw
        if prior.SECRET in payload:
            raise RuntimeError("Secret appeared in inbound payload; capture refused")
        sequence = trace.add(
            "raw_preparse_message", payload_type=type(raw).__name__, raw_payload=payload
        )
        active = trace.active.get()
        if active is not None:
            active.append(sequence)
        return raw

    async def tapped_decode(self):
        sequences = []
        token = trace.active.set(sequences)
        try:
            message = await decode(self)
            trace.add(
                "sdk_decoded_message",
                raw_sequences=sequences,
                value=prior.safe(message.model_dump(mode="json")),
            )
            return message
        finally:
            trace.active.reset(token)

    async def tapped_send(self, message, *args, **kwargs):
        trace.sockets.add(self)
        value = json.loads(message)
        realtime = value.get("realtime_input", value.get("realtimeInput", {}))
        audio = realtime.get("audio", {})
        pcm = base64.urlsafe_b64decode(audio["data"]) if audio.get("data") else b""
        if pcm != trace.outgoing_pcm:
            raise RuntimeError("Wire PCM differs from requested PCM")
        if pcm:
            value = {
                "realtime_input": {
                    "audio": {
                        "samples": len(pcm) // 2,
                        "pcm16_sha256": digest(pcm),
                        "mime_type": "audio/pcm;rate=16000",
                    }
                }
            }
        trace.add("websocket_send_begin", value=prior.safe(value))
        await send(self, message, *args, **kwargs)
        trace.wire_pcm.extend(pcm)
        trace.add("websocket_send_return", samples=len(pcm) // 2)

    async def tapped_close(self, *args, **kwargs):
        trace.add("websocket_close_begin")
        result = await close(self, *args, **kwargs)
        trace.add("websocket_close_return", close_code=self.close_code)
        return result

    _LiveRedirectGuard.recv, _LiveRedirectGuard.send = tapped_recv, tapped_send
    AsyncSession._receive, _LiveRedirectGuard.close = tapped_decode, tapped_close
    try:
        yield
    finally:
        _LiveRedirectGuard.recv, _LiveRedirectGuard.send = recv, send
        AsyncSession._receive, _LiveRedirectGuard.close = decode, close


async def execute(trace, pcm, schedule, key, variant):
    resources = await asyncio.to_thread(adapter._prepare_gemini_resources_sync, key, ["en-US"], [])
    if variant == "manual":
        resources.config.realtime_input_config.automatic_activity_detection = (
            types.AutomaticActivityDetection(disabled=True)
        )
    trace.add("effective_config", value=resources.config.model_dump(mode="json", exclude_none=True))
    context = resources.client.aio.live.connect(
        model=adapter.GEMINI_TRANSCRIBE_STT_MODEL, config=resources.config
    )
    receiver = None
    entered = False
    trace.start = time.monotonic()
    trace.starts += 1
    trace.add(
        "remote_start_attempt",
        credential_variable="GOOGLE_API_KEY2",
        model=adapter.GEMINI_TRANSCRIBE_STT_MODEL,
    )
    try:
        async with asyncio.timeout_at(asyncio.get_running_loop().time() + 28.5):
            session = await context.__aenter__()
            entered = True
            trace.add("remote_start_return")

            async def receive():
                try:
                    while True:
                        async for _message in session.receive():
                            pass
                except asyncio.CancelledError:
                    trace.add("receive_cancelled_at_cleanup")
                    raise
                except Exception as error:
                    trace.add("receive_failed", exception_class=type(error).__name__)
                    raise

            receiver = asyncio.create_task(receive())

            async def transmit(control=None, chunk=b"", source=None):
                trace.outgoing_pcm = chunk
                trace.reserved += len(chunk) // 2
                trace.add(
                    "send_begin",
                    samples=len(chunk) // 2,
                    control=control,
                    submitted_frontier=trace.written,
                    source=source,
                )
                if control:
                    value = (
                        True
                        if control == "audio_stream_end"
                        else (
                            types.ActivityStart()
                            if control == "activity_start"
                            else types.ActivityEnd()
                        )
                    )
                    await session.send_realtime_input(**{control: value})
                else:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                trace.written += len(chunk) // 2
                trace.outgoing_pcm = b""
                trace.add(
                    "send_return",
                    samples=len(chunk) // 2,
                    control=control,
                    submitted_frontier=trace.written,
                    source=source,
                )

            playback = time.monotonic()
            trace.add("playback_begin")
            active = False
            for item in schedule:
                due = playback + item["due_s"]
                await asyncio.sleep(max(0, due - time.monotonic()))
                trace.add("scheduled_delivery", source=item, lag_s=time.monotonic() - due)
                if receiver.done():
                    await receiver
                    raise RuntimeError("Receiver ended before input completion")
                if item["samples"]:
                    if variant == "manual" and not active:
                        await transmit("activity_start", source=item)
                        active = True
                    start = item["start_sample"] * 2
                    await transmit(chunk=pcm[start : start + 2 * item["samples"]], source=item)
                elif variant == "manual":
                    await transmit("activity_end", source=item)
                    active = False
                elif item["reason"] == "source_eof":
                    await transmit("audio_stream_end", source=item)
                else:
                    trace.add("nonterminal_eos_omitted", submitted_frontier=trace.written)
                if item["reason"] == "source_eof":
                    trace.eof = time.monotonic()
                    trace.add("source_eof", submitted_frontier=trace.written)
            await asyncio.sleep(max(0, min(trace.eof + 9.0, trace.start + 28.0) - time.monotonic()))
            if receiver.done():
                await receiver
            trace.add("bounded_receive_observation_end", receiver_active=not receiver.done())
    finally:
        deadline = min(trace.start + 30, trace.eof + 10 if trace.eof else trace.start + 30)
        try:
            async with asyncio.timeout(max(0.001, deadline - time.monotonic())):
                if receiver:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
                if entered:
                    await context.__aexit__(None, None, None)
                await resources.client.aio.aclose()
                resources.client.close()
                await resources.async_transport.aclose()
                resources.sync_transport.close()
                trace.add("cleanup_return")
        except BaseException:
            for socket in trace.sockets:
                socket.transport.abort()
            trace.add("cleanup_deadline_abort")
            raise


def summary(trace):
    finals, interims = [], []
    for row in trace.rows:
        if row["event"] != "raw_preparse_message":
            continue
        value = json.loads(row["raw_payload"])
        content = value.get("serverContent", value.get("server_content", {}))
        for camel, snake, target in [
            ("inputTranscription", "input_transcription", finals),
            ("interimInputTranscription", "interim_input_transcription", interims),
        ]:
            if camel in content or snake in content:
                target.append(
                    {
                        "raw_sequence": row["sequence"],
                        "at_monotonic_s": row["at_monotonic_s"],
                        "value": content.get(camel, content.get(snake)),
                    }
                )
    sdk_finals = [
        r["value"]["server_content"]["input_transcription"]
        for r in trace.rows
        if r["event"] == "sdk_decoded_message"
        and r["value"].get("server_content")
        and r["value"]["server_content"].get("input_transcription") is not None
    ]
    return {
        "remote_start_attempts": trace.starts,
        "reserved_samples": trace.reserved,
        "write_returned_samples": trace.written,
        "wire_successful_samples": len(trace.wire_pcm) // 2,
        "wire_pcm16_sha256": digest(trace.wire_pcm),
        "raw_authoritative_finals": finals,
        "raw_interims": interims,
        "sdk_authoritative_finals": sdk_finals,
        "wire_sdk_final_text_conserved": [r["value"].get("text") for r in finals]
        == [r.get("text") for r in sdk_finals],
        "source_eof_monotonic_s": trace.eof,
        "boundary_events": [
            r
            for r in trace.rows
            if r["event"]
            in {
                "remote_start_attempt",
                "remote_start_return",
                "playback_begin",
                "source_eof",
                "bounded_receive_observation_end",
                "websocket_close_begin",
                "websocket_close_return",
                "cleanup_return",
                "cleanup_deadline_abort",
                "receive_failed",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--variant", choices=("auto-no-mid-eos", "manual"), required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    prior.ENV = args.credential_file
    key = prior.credential()
    pcm, schedule, source = prepare()
    config = adapter._build_live_config_sync(["en-US"], [])
    if args.variant == "manual":
        config.realtime_input_config.automatic_activity_detection = (
            types.AutomaticActivityDetection(disabled=True)
        )
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "prerequisites_ready_no_network",
                    "source": source,
                    "credential_definition_ready": True,
                    "schedule_entries": len(schedule),
                    "audio_writes": sum(bool(r["samples"]) for r in schedule),
                    "config": config.model_dump(mode="json", exclude_none=True),
                }
            )
        )
        return 0
    if args.variant == "manual":
        previous = json.loads((BASE / "issue264_gemini_auto-no-mid-eos_report.json").read_text())
        if previous["status"] != "executed" or previous.get("useful_repair_candidate") is not False:
            raise RuntimeError(
                "Manual variant requires an executed nonrepair auto variant disposition"
            )
    folder = BASE / ("issue264_gemini_" + args.variant + "_local")
    trace = Trace(folder)
    report = {
        "schema": "issue264-gemini-differential-1",
        "original_candidate": CANDIDATE,
        "variant": args.variant,
        "source": source,
        "driver_sha256": digest(Path(__file__).read_bytes()),
        "credential": "Only designated GOOGLE_API_KEY2, no fallback; no value/hash/headers recorded",
        "limits": {"starts": 1, "session_s": 30, "eof_observation_s": 9, "eof_close_cap_s": 10},
        "controlled_boundary": "Direct official SDK with production config and network policy, not adapter/engine/runtime smoke",
        "schedule": schedule,
        "status": "not_started",
    }
    try:
        with instrument(trace):
            asyncio.run(execute(trace, pcm, schedule, key, args.variant))
        report["status"] = "executed"
    except BaseException as error:
        report["status"] = "invalid_or_access_failure"
        report["exception_class"] = type(error).__name__
    finally:
        report["summary"] = summary(trace)
        trace.stream.close()
        (folder / "submitted.pcm16le").write_bytes(trace.wire_pcm)
        report["local_trace_sha256"] = digest((folder / "trace.jsonl").read_bytes())
        report["effective_config"] = next(
            (r["value"] for r in trace.rows if r["event"] == "effective_config"), None
        )
        dump(BASE / ("issue264_gemini_" + args.variant + "_report.json"), report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "variant": args.variant,
                    "starts": trace.starts,
                    "write_returned_samples": trace.written,
                    "finals": len(report["summary"]["raw_authoritative_finals"]),
                    "exception_class": report.get("exception_class"),
                }
            )
        )
    return 0 if report["status"] == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
