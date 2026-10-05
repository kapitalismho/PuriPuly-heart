"""Bounded, explicitly authorized #212 real-wire investigation; no product changes.
Run with parent locked .venv Python and --provider qwen or gemini.
No retries, fallback, mic, external TTS, billing meter or account mutation.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
import uuid
from datetime import datetime, timezone

from common import DIRECTORY, ROOT, pcm, safe_error, secret, settings_data

QMODEL = "qwen-audio-3.1-asr-flash-streaming"
GMODEL = "gemini-3.5-transcribe-live"
PLAN = {
    "Q-01": {"alternatives": "task-relative / connection-audio-relative / speech-relative timestamps", "input_control": "Same socket A with 600ms prefix, then B with 1600ms prefix; finish each once, new task IDs", "predicate": "Compare returned words to actual nonzero speech windows under all three clocks", "consequence": "Only observed task-relative mapping may support candidate T; retain per-task maps and no cross-route certification"},
    "Q-02": {"alternatives": "word/punctuation reconstruct sentence exactly / lossy or other representation", "input_control": "Retain all native sentence/word fields for A,B,short No,repeated No", "predicate": "Compare literal concatenations with exact sentence text; no strip/lowercase reconstruction", "consequence": "Preserve native exact text and separate word offsets; do not assume lexical word shape"},
    "Q-03": {"alternatives": "fixed-only words immutable / mutable / never emitted", "input_control": "Capture every partial/final fixed field", "predicate": "Any same-ID fixed prefix mutation refutes independent finality; absence or no mutation cannot establish guarantee", "consequence": "sentence_end remains only documented finality gate"},
    "Q-04": {"alternatives": "subsecond finish accepted and short content retained / protocol rejection / empty", "input_control": "Energy-bounded shared_no trim plus 80ms each side, total <1s, finish once", "predicate": "Matching task-finished and returned final text; record exact sent samples", "consequence": "Bounded short-control support only; no universal quality or floor claim"},
    "Q-05": {"alternatives": "empty task terminates / fails / times out", "input_control": "Last task sends zero audio then finish once", "predicate": "Matching task-finished versus task-failed versus 15s timeout", "consequence": "Distinguish task completion from text/empty evidence, no retries"},
    "Q-06": {"alternatives": "residual final after finish / no residual / after terminal", "input_control": "Immediate finish after serialized audio; 200ms extra observation after terminal", "predicate": "Native receipt versus finish and task-finished write/receipt timings", "consequence": "Drain to matching task-finished; later absence cannot establish universal no-tail"},
    "Q-07": {"alternatives": "unique task+sentence IDs / duplicate or conflicting final / repeat text under distinct identity", "input_control": "Fresh UUID per reused task and two distinct No occurrences in one task", "predicate": "All native IDs/order and final text are retained; no duplicate controls sent", "consequence": "Identity-based preservation; finite trace cannot certify replay/dedup universally"},
    "Q-08": {"alternatives": "continuation retained on legal same-socket next task / failure", "input_control": "B follows A after matching terminal and readiness; no application onset gate", "predicate": "Second task content and task IDs", "consequence": "Native continuation observation, not proof current adapter maps or buffers correctly"},
    "Q-09": {"alternatives": "configured route/model usable / auth, quota, unsupported failure", "input_control": "Exact Beijing workspace route and exact Beijing key only", "predicate": "Readiness/failure and returned metadata", "consequence": "No alternate paid fallback; all other endpoint rows remain not-run"},
    "Q-10": {"alternatives": "account limits/billing minima can be established / remain unknown", "input_control": "Only incidental native usage/errors from legal requests; no billing measurement or stress", "predicate": "Record returned usage without estimating charge; no universal quota claims", "consequence": "Effective concurrency/lifetime/minimum billing remain unresolved by design"},
    "G-01": {"alternatives": "final only after fence / final before fence", "input_control": "A plus 800ms silence and 2s receipt window before F1", "predicate": "Native inputTranscription receipt before F1", "consequence": "Counterexample forbids fence-required final admission"},
    "G-02": {"alternatives": "one fence exactly one final / zero,multiple,independent finals", "input_control": "A natural end then F1; B F2; No F3; second session short/repeated content", "predicate": "Retain native finals versus local fences without assigning ownership; positive finite trace cannot prove bijection", "consequence": "No FIFO pairing or final-count barrier"},
    "G-03": {"alternatives": "late final after successor input / none observed", "input_control": "Immediately send successor after each fence", "predicate": "Content-diagnostic A/B/No final receipt versus successor writes", "consequence": "Receipt cursor cannot establish acoustic source association"},
    "G-04": {"alternatives": "finals append independent text / cumulative snapshots", "input_control": "Distinct A/B plus real repeated No at new sample ranges", "predicate": "Exact native final text per receipt, not reconstructed cumulative string", "consequence": "Preserve every native final independently; equal text is not replay evidence"},
    "G-05": {"alternatives": "time/ID fields establish source association / absent or insufficient", "input_control": "Capture raw SDK-wire JSON and parsed SDK messages including optional fields", "predicate": "Inspect final fields for acoustic intervals and correlated fence/event IDs; word times unsupported by selected contract", "consequence": "No guessed coordinates from finished/activity/receipt/fence index"},
    "G-06": {"alternatives": "interim/activity/finished are final ACK / distinct events", "input_control": "Same unchanged TEXT+VERBATIM automatic VAD500/400 config", "predicate": "Retain exact field distinctions and receipt ordering", "consequence": "Only input_transcription is authoritative text, no activity ACK pairing"},
    "G-07": {"alternatives": "short input preserved / rejected or empty", "input_control": "Session2 trimmed No then immediate correction; same transport no manual activity", "predicate": "Returned independent final content and input/fence cursors", "consequence": "Bounded support only, no universal acoustic qualification"},
    "G-08": {"alternatives": "no-new-audio produces authoritative empty final / no guaranteed evidence", "input_control": "Session2 records local suppression of fence before any audio and repeated fence after last fence; no prohibited empty wire control", "predicate": "Local no-send event versus any spontaneous native empty final", "consequence": "Suppressed fence/write terminal is not empty recognition; absence is bounded only"},
    "G-09": {"alternatives": "configured default endpoint works / auth quota model unsupported", "input_control": "Effective SDK backend/base URL/API version checked before legal connection", "predicate": "Native setup or sanitized failure; no override inheritance or retry", "consequence": "No fallback; failed branch remains blocked, account limits/billing universal claims unresolved"},
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Trace:
    def __init__(self, provider: str):
        self.start = time.monotonic()
        self.provider = provider
        self.redact: list[str] = []
        self.events: list[dict] = []
        self.cursor: dict[str, int] = {}
        self.result = {
            "baseline": "0f0ba5173da51250bb5926e4a2eead5e68adff81",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "provider": provider, "plan": PLAN,
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "packages": {n: importlib.metadata.version(n) for n in ("websockets", "google-genai", "dashscope")}},
            "script_sha256": sha(script_bytes()),
            "common_sha256": sha((DIRECTORY / "common.py").read_bytes()),
            "inputs": {}, "sessions": [], "events": self.events,
            "limits": {"qwen_connections": 1, "gemini_connections": 2, "sent_audio_s_per_provider_ceiling": 60,
                       "start_timeout_s": 8, "task_finish_timeout_s": 15, "gemini_final_window_s": 3,
                       "automatic_retries": 0, "billing_meter": False},
        }

    def clean(self, value):
        text = json.dumps(value, ensure_ascii=False, default=str)
        for item in self.redact:
            text = text.replace(item, "[redacted]")
        text = re.sub(r"(https?://[^\s\"<>]*[?])[^\s\"<>]*", r"\1[redacted-query]", text)
        return json.loads(text)

    def add(self, kind: str, epoch: str, **fields):
        self.events.append(self.clean({"ordinal": len(self.events) + 1,
                                      "receipt_or_write_s": round(time.monotonic() - self.start, 6),
                                      "kind": kind, "epoch": epoch,
                                      "sent_sample_cursor": self.cursor.get(epoch, 0), **fields}))

    def save(self):
        path = DIRECTORY / f"qwen-gemini-results-{self.provider}.json"
        path.write_text(json.dumps(self.clean(self.result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def script_bytes():
    from pathlib import Path
    return Path(__file__).read_bytes()


def inputs(trace):
    clips = {}
    for name in ("shared_a", "shared_b", "shared_no", "shared_correction"):
        path = DIRECTORY / "audio" / (name + ".wav")
        data = pcm(path)
        import array
        samples = array.array("h", data)
        active = [i for i, value in enumerate(samples) if abs(value) >= 100]
        bounds = [min(active), max(active) + 1] if active else [0, 0]
        trace.result["inputs"][name] = {"file": "audio/" + path.name, "wav_sha256": sha(path.read_bytes()),
            "pcm_sha256": sha(data), "samples": len(data) // 2, "duration_s": len(data) / 32000,
            "energy_bounds_samples_threshold_100": bounds,
            "annotation": "Energy support, not phonetic alignment; synthetic Microsoft Zira shared clips"}
        clips[name] = data
    begin, end = trace.result["inputs"]["shared_no"]["energy_bounds_samples_threshold_100"]
    begin, end = max(0, begin - 1280), min(len(clips["shared_no"]) // 2, end + 1280)
    clips["short_no"] = clips["shared_no"][begin * 2:end * 2]
    trace.result["inputs"]["short_no"] = {"source": "shared_no", "slice_samples": [begin, end],
        "pcm_sha256": sha(clips["short_no"]), "samples": end - begin, "duration_s": (end - begin) / 16000,
        "trim": "Threshold100 active support plus80ms each side; no phonetic alignment claim"}
    if len(clips["short_no"]) >= 32000:
        raise ValueError("Energy-contained No is not subsecond; no clipped speech fallback")
    return clips


async def send_pcm(trace, epoch, data, label, send):
    origin = trace.cursor.get(epoch, 0)
    trace.add("input_manifest", epoch, label=label, start_sample=origin, end_sample=origin + len(data) // 2,
              pcm_sha256=sha(data), synthetic_silence=label.startswith("silence"))
    for offset in range(0, len(data), 3200):
        chunk = data[offset:offset + 3200]
        await asyncio.wait_for(send(chunk), 5)
        trace.cursor[epoch] = trace.cursor.get(epoch, 0) + len(chunk) // 2
        trace.add("audio_write", epoch, label=label, chunk_samples=len(chunk) // 2)
        await asyncio.sleep(len(chunk) / 32000)


async def qwen(trace, clips):
    from websockets.asyncio.client import connect
    from puripuly_heart.config.alibaba_connection import AlibabaRegionalSettings, resolve_alibaba_connection
    settings = settings_data()
    config = settings.get("translation", {}).get("qwen", {}).get("beijing", {})
    regional = AlibabaRegionalSettings(endpoint_mode=config.get("endpoint_mode", "legacy_shared"), api_host=config.get("api_host", ""))
    route = resolve_alibaba_connection("beijing", regional)
    if route.endpoint_mode != "workspace_dedicated":
        raise ValueError("Persisted Beijing workspace route changed; stop rather than infer alternate authority")
    trace.redact.append(route.host.split(".")[0])
    key = secret("alibaba_api_key_beijing", "ALIBABA_API_KEY_BEIJING")
    if not key:
        raise ValueError("Exact Beijing key unavailable; no legacy fallback")
    trace.redact.append(key)
    trace.result["sessions"].append({"epoch": "Q-S1", "region": "beijing", "endpoint_mode": route.endpoint_mode,
        "endpoint": trace.clean(route.websocket_url), "requested_model": QMODEL, "returned_model": None,
        "backend": "native duplex WebSocket; richer than packaged timing-dropping adapter"})
    native = []
    condition = asyncio.Condition()
    fatal = []
    async with connect(route.websocket_url, additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=8, close_timeout=3, max_size=4 * 1024 * 1024) as ws:
        async def receive():
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    task_id = event.get("header", {}).get("task_id", "Q-S1")
                    trace.add("native_receive", task_id, native=event)
                    async with condition:
                        native.append(event)
                        if event.get("header", {}).get("event") == "task-failed":
                            fatal.append("Native task-failed; branch stops")
                        condition.notify_all()
            except Exception as exc:
                trace.add("receive_error", "Q-S1", error=safe_error(exc))
                async with condition:
                    fatal.append(safe_error(exc))
                    condition.notify_all()
        receiver = asyncio.create_task(receive())
        async def wait_event(task_id, event_name, timeout):
            async def wait_inner():
                async with condition:
                    while True:
                        if fatal:
                            raise RuntimeError(fatal[0])
                        for event in native:
                            h = event.get("header", {})
                            if h.get("task_id") == task_id and h.get("event") == event_name:
                                return event
                        await condition.wait()
            return await asyncio.wait_for(wait_inner(), timeout)
        tasks = [
            ("clock_A", [("silence_prefix600", bytes(19200)), ("shared_a", clips["shared_a"])]),
            ("clock_B", [("silence_prefix1600", bytes(51200)), ("shared_b", clips["shared_b"])]),
            ("subsecond_no", [("short_no", clips["short_no"])]),
            ("repeated_no", [("short_no_1", clips["short_no"]), ("silence_gap700", bytes(22400)), ("short_no_2", clips["short_no"])]),
            ("empty", []),
        ]
        try:
            for name, pieces in tasks:
                task_id = str(uuid.uuid4())
                parameters = {"format": "pcm", "sample_rate": 16000, "semantic_punctuation_enabled": False,
                    "max_sentence_silence": 6000, "multi_threshold_mode_enabled": False, "heartbeat": True}
                request = {"header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {"task_group": "audio", "task": "asr", "function": "recognition", "model": QMODEL,
                                "parameters": parameters, "input": {}}}
                await asyncio.wait_for(ws.send(json.dumps(request)), 5)
                trace.add("run_task_write", task_id, task_name=name, native=request)
                await wait_event(task_id, "task-started", 8)
                for label, data in pieces:
                    await send_pcm(trace, task_id, data, label, ws.send)
                request = {"header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"}, "payload": {"input": {}}}
                await asyncio.wait_for(ws.send(json.dumps(request)), 5)
                trace.add("finish_task_write", task_id, task_name=name, native=request)
                await wait_event(task_id, "task-finished", 15)
                await asyncio.sleep(.2)
        finally:
            await ws.close()
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            trace.add("connection_closed", "Q-S1", close_code=ws.close_code)


async def gemini(trace, clips):
    from google import genai
    from google.genai import live, types
    from puripuly_heart.providers.stt.gemini_transcribe import _build_live_config_sync
    key = secret("gemini_transcribe_api_key", "GEMINI_TRANSCRIBE_API_KEY")
    if not key:
        raise ValueError("Gemini transcribe key unavailable")
    trace.redact.append(key)
    client = genai.Client(api_key=key)
    backend = client._api_client
    base = backend._websocket_base_url()
    if isinstance(base, bytes):
        base = base.decode()
    version = backend._http_options.api_version
    trace.result["effective_backend"] = {"vertexai": bool(backend.vertexai), "websocket_base_url": str(base), "api_version": version}
    if backend.vertexai or str(base).rstrip("/") != "wss://generativelanguage.googleapis.com" or version != "v1beta":
        raise ValueError("Uncertified Gemini backend override; no alternate call")
    original_connect = live.ws_connect
    epoch = "G-S1"
    class CaptureSocket:
        def __init__(self, ws):
            self.ws = ws
        def __getattr__(self, name):
            return getattr(self.ws, name)
        async def recv(self, *args, **kwargs):
            raw = await self.ws.recv(*args, **kwargs)
            trace.add("native_receive", epoch, native=json.loads(raw))
            return raw
        async def send(self, raw):
            # Keep exact setup/control; audio bytes already represented by manifests and hashes.
            value = json.loads(raw)
            if "setup" in value:
                trace.add("setup_write", epoch, native=value)
            await self.ws.send(raw)
    @contextlib.asynccontextmanager
    async def capture_connect(uri, **kwargs):
        trace.add("connection_attempt", epoch, endpoint=uri)
        async with original_connect(uri, **kwargs) as ws:
            yield CaptureSocket(ws)
    live.ws_connect = capture_connect
    try:
        for index in (1, 2):
            epoch = f"G-S{index}"
            config = _build_live_config_sync([], [])
            trace.result["sessions"].append({"epoch": epoch, "requested_model": GMODEL, "returned_model": None,
                "backend": "google-genai locked SDK; raw receive capture outside packaged adapter", "config": config.model_dump(mode="json", exclude_none=True)})
            async with asyncio.timeout(35):
                async with client.aio.live.connect(model=GMODEL, config=config) as session:
                    async def receive():
                        try:
                            while True:
                                async for message in session.receive():
                                    trace.add("sdk_receive", epoch, parsed=message.model_dump(mode="json", exclude_none=True))
                        except Exception as exc:
                            trace.add("receive_error", epoch, error=safe_error(exc))
                            raise
                    receiver = asyncio.create_task(receive())
                    async def send(chunk):
                        if receiver.done():
                            await receiver
                        await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
                    last_fence_cursor = 0
                    async def fence(name):
                        nonlocal last_fence_cursor
                        cursor = trace.cursor.get(epoch, 0)
                        if cursor == last_fence_cursor:
                            trace.add("fence_suppressed_no_new_audio", epoch, fence=name, note="Mirrors packaged writer; no native control sent")
                            return
                        if receiver.done():
                            await receiver
                        await session.send_realtime_input(audio_stream_end=True)
                        last_fence_cursor = cursor
                        trace.add("audio_stream_end_write", epoch, fence=name)
                    try:
                        if index == 1:
                            await send_pcm(trace, epoch, bytes(16000), "silence_prefix500", send)
                            await send_pcm(trace, epoch, clips["shared_a"], "shared_a", send)
                            await send_pcm(trace, epoch, bytes(25600), "silence_after_A800", send)
                            trace.add("prefence_observation_start", epoch, window_s=2)
                            await asyncio.sleep(2)
                            await fence("F1")
                            await send_pcm(trace, epoch, clips["shared_b"], "shared_b", send)
                            await fence("F2")
                            await send_pcm(trace, epoch, clips["short_no"], "short_no", send)
                            await fence("F3")
                        else:
                            await fence("F0-empty-suppressed")
                            await send_pcm(trace, epoch, clips["short_no"], "short_no", send)
                            await fence("F1")
                            await send_pcm(trace, epoch, clips["shared_correction"], "shared_correction", send)
                            await fence("F2")
                            await send_pcm(trace, epoch, clips["short_no"], "repeat_no_1", send)
                            await send_pcm(trace, epoch, bytes(22400), "silence_gap700", send)
                            await send_pcm(trace, epoch, clips["short_no"], "repeat_no_2", send)
                            await fence("F3")
                            await fence("F3-duplicate-suppressed")
                        trace.add("bounded_final_window_start", epoch, window_s=3)
                        await asyncio.sleep(3)
                        if receiver.done():
                            await receiver
                    finally:
                        receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)
                        trace.add("connection_closing", epoch, note="Window ends, not proof recognition complete")
    finally:
        live.ws_connect = original_connect
        await client.aio.aclose()
        client.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("qwen", "gemini"), required=True)
    args = parser.parse_args()
    trace = Trace(args.provider)
    try:
        clips = inputs(trace)
        trace.save()  # Plan, hashes and inputs exist before the first network request.
        await (qwen(trace, clips) if args.provider == "qwen" else gemini(trace, clips))
        trace.result["execution_outcome"] = "bounded plan completed"
    except Exception as exc:
        trace.result["execution_outcome"] = "branch stopped; no retry/fallback"
        trace.add("branch_error", args.provider, error=safe_error(exc))
    finally:
        trace.result["sent_audio_s"] = sum(trace.cursor.values()) / 16000
        trace.save()
    print(json.dumps({"provider": args.provider, "outcome": trace.result["execution_outcome"],
                      "sent_audio_s": trace.result["sent_audio_s"], "events": len(trace.events)}))


if __name__ == "__main__":
    asyncio.run(main())
