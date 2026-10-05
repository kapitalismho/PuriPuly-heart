"""Authorized finite native-wire observations; never imported by product code.
Run --prepare first. Connections run only with --provider; no automatic retries.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from array import array
from pathlib import Path
from urllib.parse import urlencode

from common import DIRECTORY, ROOT, pcm, safe_error, secret, settings_data

RATE = 16000
ENDPOINTS = {
    "soniox": "wss://stt-rt.soniox.com/transcribe-websocket",
    "deepgram": "wss://api.deepgram.com/v1/listen",
    "elevenlabs": "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
}
MODELS = {"soniox": "stt-rt-v5", "deepgram": "nova-3", "elevenlabs": "scribe_v2_realtime"}
KEY_NAMES = {"soniox": "soniox_api_key", "deepgram": "deepgram_api_key", "elevenlabs": "elevenlabs_scribe_api_key"}
ERROR_TYPES = {"error", "auth_error", "quota_exceeded", "commit_throttled", "rate_limited", "invalid_request", "input_error", "transcriber_error", "resource_exhausted", "unaccepted_terms", "insufficient_audio_activity"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inputs():
    audio, metadata = {}, {}
    for name in ("a", "b", "no", "correction"):
        path = DIRECTORY / "audio" / f"shared_{name}.wav"
        data = pcm(path)
        samples = array("h", data)
        active = [i for i, value in enumerate(samples) if abs(value) >= 200]
        if not active:
            raise ValueError("Synthetic source has no threshold-active samples")
        begin, end = active[0], active[-1] + 1
        padded_begin, padded_end = max(0, begin - 320), min(len(samples), end + 320)
        audio[name] = data
        audio[name + "_trim"] = data[padded_begin * 2:padded_end * 2]
        audio[name + "_active"] = data[begin * 2:end * 2]
        metadata[name] = {
            "path": str(path.relative_to(ROOT)), "wav_sha256": sha(path.read_bytes()),
            "pcm_sha256": sha(data), "samples": len(samples), "duration_s": len(samples) / RATE,
            "energy_annotation": {"absolute_pcm_threshold": 200, "active_start_sample": begin,
                                  "active_end_sample_exclusive": end, "not_forced_alignment": True},
            "trim": {"start_sample": padded_begin, "end_sample_exclusive": padded_end,
                     "duration_s": (padded_end - padded_begin) / RATE,
                     "sha256": sha(audio[name + "_trim"])},
            "active_trim": {"start_sample": begin, "end_sample_exclusive": end,
                            "duration_s": (end - begin) / RATE, "sha256": sha(audio[name + "_active"])},
        }
    return audio, metadata


def environment(metadata):
    config = settings_data().get("stt", {})
    resolved = {}
    for provider in MODELS:
        block = config.get("elevenlabs_scribe" if provider == "elevenlabs" else provider, {})
        model = block.get("model", MODELS[provider])
        endpoint = block.get("endpoint", ENDPOINTS[provider])
        if model != MODELS[provider] or endpoint != ENDPOINTS[provider]:
            raise ValueError(f"Unmapped configured route for {provider}; no calls authorized")
        resolved[provider] = {"model": model, "endpoint": endpoint}
    sources = [Path(__file__), DIRECTORY / "common.py", ROOT / "docs/asr-speaker-policy.md", ROOT / "docs/experiments/issue-212/evidence.md"]
    return {
        "python": sys.version, "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("websockets", "deepgram-sdk", "elevenlabs")},
        "source_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in sources},
        "configured_routes": resolved, "audio": metadata,
        "wire": "Raw native WebSockets using locked websockets; SDKs and packaged adapters bypassed; no product mutation",
        "sample_rate": RATE, "format": "mono PCM signed16 little endian",
        "limits": {"initial_connections_per_provider": 2, "automatic_retries": 0, "audio_ceiling_per_provider_s": 60},
    }


class Session:
    def __init__(self, provider, name, result):
        self.provider, self.name = provider, name
        self.result = result
        self.started = time.monotonic()
        self.cursor = 0
        self.events = []
        self.raw = []
        self.aliases = {}
        self.changed = asyncio.Event()
        self.failed = False
        self.closed = False
        self.client_closing = False
        self.ws = None
        self.reader = None
        result.update(name=name, events=self.events, input_map=[], controls=[], status="started")

    def sanitize(self, value, key=""):
        if key.lower() in {"api_key", "authorization", "xi-api-key", "token", "account_id", "project_id", "organization_id", "user_id", "email"}:
            return "[redacted]"
        if isinstance(value, dict):
            return {k: self.sanitize(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.sanitize(v, key) for v in value]
        if isinstance(value, str):
            if key in {"session_id", "request_id", "event_id", "item_id", "connection_id"}:
                return self.aliases.setdefault((key, value), f"{key}-{len(self.aliases) + 1}")
            return safe_error(Exception(value)).removeprefix("Exception: ")
        return value

    def log(self, kind, **values):
        self.events.append({"ordinal": len(self.events) + 1, "elapsed_s": round(time.monotonic() - self.started, 6),
                            "sent_sample_cursor": self.cursor, "kind": kind, **self.sanitize(values)})

    async def open(self, key):
        from websockets.asyncio.client import connect
        query, headers = {}, {}
        if self.provider == "deepgram":
            query = dict(model="nova-3", language="en", encoding="linear16", sample_rate=RATE, channels=1,
                         interim_results="false", punctuate="true", vad_events="false", endpointing="false")
            headers = {"Authorization": f"Token {key}"}
        elif self.provider == "elevenlabs":
            query = dict(model_id="scribe_v2_realtime", audio_format="pcm_16000", sample_rate=RATE,
                         commit_strategy="manual", language_code="en", include_timestamps="true",
                         include_language_detection="false", no_verbatim="false", filter_background_audio="false")
            headers = {"xi-api-key": key}
        self.result["options"] = query
        url = ENDPOINTS[self.provider] + ("?" + urlencode(query) if query else "")
        self.ws = await connect(url, additional_headers=headers, open_timeout=8, close_timeout=2, max_size=2**20)
        self.log("connected", endpoint=ENDPOINTS[self.provider])
        self.reader = asyncio.create_task(self.receive())
        if self.provider == "soniox":
            options = dict(model="stt-rt-v5", audio_format="pcm_s16le", sample_rate=RATE, num_channels=1,
                           enable_endpoint_detection=False, enable_language_identification=False,
                           enable_speaker_diarization=False, language_hints=["en"])
            self.result["options"] = options
            await self.ws.send(json.dumps({"api_key": key, **options}))
            self.log("configuration_written", options=options)

    async def receive(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                self.raw.append(data)
                self.log("received", payload=data)
                event_type = data.get("message_type", data.get("type", ""))
                if event_type in ERROR_TYPES or data.get("error_code") is not None or data.get("error"):
                    self.failed = True
                self.changed.set()
        except Exception as error:
            self.log("receive_error", error=safe_error(error))
            if not self.client_closing:
                self.failed = True
        finally:
            self.closed = True
            self.log("transport_closed", code=self.ws.close_code, reason=self.ws.close_reason)
            self.changed.set()

    async def audio(self, label, data, source=None):
        if self.failed or self.closed:
            raise RuntimeError("Provider error or closed socket; branch stopped")
        begin = self.cursor
        self.result["input_map"].append({"label": label, "provider_start_sample": begin,
                                         "provider_end_sample": begin + len(data) // 2,
                                         "pcm_sha256": sha(data), "source": source,
                                         "synthetic_no_source": source is None})
        started = time.monotonic()
        for offset in range(0, len(data), 3200):
            if self.failed or self.closed:
                raise RuntimeError("Provider error or closed socket; branch stopped")
            chunk = data[offset:offset + 3200]
            if self.provider == "elevenlabs":
                await self.ws.send(json.dumps({"message_type": "input_audio_chunk", "audio_base_64": base64.b64encode(chunk).decode(),
                                               "commit": False, "sample_rate": RATE}))
            else:
                await self.ws.send(chunk)
            self.cursor += len(chunk) // 2
            self.log("audio_written", label=label, sample_count=len(chunk) // 2, pcm_sha256=sha(chunk))
            await asyncio.sleep(max(0, started + (offset + len(chunk)) / 32000 - time.monotonic()))

    async def control(self, kind):
        if self.failed or self.closed:
            raise RuntimeError("Cannot write control after error/close")
        if self.provider == "soniox":
            payload = "" if kind == "end" else json.dumps({"type": "finalize"})
        elif self.provider == "deepgram":
            payload = json.dumps({"type": "CloseStream" if kind == "end" else "Finalize"})
        else:
            payload = json.dumps({"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True, "sample_rate": RATE})
        before = len(self.raw)
        await self.ws.send(payload)
        self.log("control_written", control=kind)
        self.result["controls"].append({"kind": kind, "cursor_samples": self.cursor,
                                        "elapsed_s": round(time.monotonic() - self.started, 6)})
        return before

    def completion(self, data):
        if self.provider == "soniox":
            return any(t.get("text") == "<fin>" and t.get("is_final") for t in data.get("tokens", []))
        if self.provider == "deepgram":
            return bool(data.get("from_finalize"))
        return data.get("message_type") == "committed_transcript"

    async def observe(self, before, seconds=4, completion=True):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.changed.clear()
            if completion and any(self.completion(data) for data in self.raw[before:]):
                self.log("completion_observed", after_receive_index=before)
                return True
            if self.failed or self.closed:
                break
            try:
                await asyncio.wait_for(self.changed.wait(), max(.001, end - time.monotonic()))
            except TimeoutError:
                break
        self.log("bounded_observation_ended", wait_s=seconds, completion_found=False,
                 note="Absence is censored, not a universal no-response guarantee")
        return False

    async def close(self):
        self.client_closing = True
        if self.ws is not None:
            self.log("client_close_requested")
            await self.ws.close()
        if self.reader is not None:
            self.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader
        self.result["sent_samples"] = self.cursor
        self.result["sent_audio_s"] = self.cursor / RATE
        self.result["wall_s"] = round(time.monotonic() - self.started, 6)


async def slot(session, audio, marker):
    await session.audio("slot_leading_silence", bytes(16000))
    await session.audio(f"shared_{marker}", audio[marker], f"shared_{marker}.wav:all")
    trailing = 48000 - 8000 - len(audio[marker]) // 2
    await session.audio("slot_trailing_silence", bytes(trailing * 2))


async def normal(session, audio):
    for marker in ("a", "a"):
        await slot(session, audio, marker)
        before = await session.control("finalize")
        if not await session.observe(before):
            session.result["continuation_branch"] = "not-run: no observed completion"
            if session.provider != "elevenlabs" and not session.failed and not session.closed:
                before = await session.control("end")
                await session.observe(before, completion=False)
            return
    await slot(session, audio, "b")
    if session.provider == "elevenlabs":
        before = await session.control("finalize")
        await session.observe(before)
        await session.observe(len(session.raw), seconds=2, completion=False)
    else:
        before = await session.control("end")
        await session.observe(before, completion=False)


async def passive(session, audio):
    for marker in ("a", "a", "b"):
        await slot(session, audio, marker)
    before = await session.control("end")
    await session.observe(before, completion=False)


async def short(session, audio):
    await session.audio("short_leading_silence", bytes(3200))
    await session.audio("shared_no_energy_trim", audio["no_trim"], "shared_no.wav:trim")
    await session.audio("short_trailing_silence", bytes(6400))
    before = await session.control("finalize")
    control_at = time.monotonic()
    complete = await session.observe(before)
    if session.failed or session.closed:
        return
    if session.provider == "soniox":
        if not complete:
            before = await session.control("end")
            await session.observe(before, completion=False)
            return
        await asyncio.sleep(max(0, control_at + 3 - time.monotonic()))
        for marker in ("a", "b"):
            await session.audio(f"shared_{marker}_active_trim", audio[marker + "_active"], f"shared_{marker}.wav:active_trim")
        before = await session.control("finalize")
        await session.audio("post_control_correction", audio["correction_trim"], "shared_correction.wav:trim")
        await session.observe(before)
        before = await session.control("end")
        await session.observe(before, completion=False)
    elif complete:
        await asyncio.sleep(max(0, control_at + 3 - time.monotonic()))
        before = await session.control("empty_finalize")
        await session.observe(before)
        if session.provider == "deepgram" and not session.failed and not session.closed:
            before = await session.control("end")
            await session.observe(before, completion=False)
        elif session.provider == "elevenlabs":
            await session.observe(len(session.raw), seconds=1, completion=False)
    elif session.provider == "deepgram":
        before = await session.control("end")
        await session.observe(before, completion=False)
    else:
        await session.audio("startup_discriminator_silence", bytes((38400 - session.cursor) * 2))
        complete = await session.observe(before, seconds=3)
        if not complete and not session.failed and not session.closed:
            await asyncio.sleep(max(0, control_at + 3 - time.monotonic()))
            before = await session.control("second_startup_finalize")
            await session.observe(before)


async def run(args, audio, result):
    provider = args.provider
    key = secret(KEY_NAMES[provider])
    if not key:
        result["status"] = "blocked: missing approved ASR credential"
        return
    result["provider"] = provider
    result["sessions"] = []
    for name in args.sessions:
        record = {}
        result["sessions"].append(record)
        session = Session(provider, name, record)
        try:
            async with asyncio.timeout(45):
                await session.open(key)
                await {"normal": normal, "short": short, "passive": passive}[name](session, audio)
            record["status"] = "provider-error" if session.failed else "bounded-observation-complete"
        except Exception as error:
            record["status"] = "failed-or-censored"
            session.log("session_error", error=safe_error(error))
            session.failed = True
        finally:
            await session.close()
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if session.failed:
            result["provider_branch_stopped"] = "error: no retries or further sessions"
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--provider", choices=tuple(MODELS))
    parser.add_argument("--sessions", nargs="+", choices=("normal", "short", "passive"), default=["normal", "short"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.prepare and not args.provider:
        parser.error("Explicit --provider required for real connections")
    if len(args.sessions) > 2 or len(set(args.sessions)) != len(args.sessions):
        parser.error("At most one normal and one short session per invocation")
    if "passive" in args.sessions and (args.provider != "soniox" or args.sessions != ["passive"]):
        parser.error("Approved passive discriminant is a single Soniox-only invocation")
    audio, metadata = inputs()
    result = {"revision": "TIMED-CLOUD-WIRE-1", "environment": environment(metadata),
              "argv": sys.argv[1:], "prepared_without_connections": args.prepare}
    if not args.prepare:
        asyncio.run(run(args, audio, result))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "provider": args.provider,
                      "sessions": [{"name": item["name"], "status": item["status"], "sent_audio_s": item["sent_audio_s"]}
                                   for item in result.get("sessions", [])]}))


if __name__ == "__main__":
    main()
