from __future__ import annotations

import asyncio
import collections
import json
import time
import uuid
from urllib.parse import urlencode

SAMPLE_RATE = 16000
CHUNK_MS = 40
TAIL_WAIT_S = 12.0
MID_FINALIZE_EVERY_S = 2.5
GAP_FINALIZE_OFFSET_S = 0.10
RAW_HEAD = 20
RAW_TAIL = 5

SONIOX_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_MODEL = "stt-rt-v5"
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
QWEN_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
QWEN_MODEL = "qwen-audio-3.1-asr-flash-streaming"
GEMINI_MODEL = "gemini-3.5-transcribe-live"

LANGUAGE_CODES = {
    "ko_kr": {"soniox": "ko", "deepgram": "ko", "qwen": "ko", "gemini": "ko"},
    "ja_jp": {"soniox": "ja", "deepgram": "ja", "qwen": "ja", "gemini": "ja"},
    "en_us": {"soniox": "en", "deepgram": "en", "qwen": "en", "gemini": "en-US"},
}

VARIANTS = {
    "soniox": ("natural", "finalize_gap", "finalize_mid"),
    "deepgram": ("natural", "nointerim", "finalize_gap", "finalize_mid"),
    "qwen": ("natural", "semantic"),
    "gemini": ("natural",),
}


class Recorder:
    def __init__(self) -> None:
        self.t0: float | None = None
        self.final = ""
        self.tail = ""
        self.events: list[dict] = []
        self.final_words: list[dict] = []
        self.partial_words: list[dict] = []
        self._partial_seen: set[tuple[str, float]] = set()
        self.controls: list[dict] = []
        self.notes: list[str] = []
        self.raw_head: list[dict] = []
        self.raw_tail: collections.deque[dict] = collections.deque(maxlen=RAW_TAIL)
        self.connect_s: float | None = None

    def now(self) -> float:
        return 0.0 if self.t0 is None else time.monotonic() - self.t0

    def update(self, *, final_append: str = "", tail: str | None = None) -> None:
        event: dict = {}
        if final_append:
            self.final += final_append
            event["fa"] = final_append
        if tail is not None and tail != self.tail:
            self.tail = tail
            event["tail"] = tail
        if event:
            event["t"] = round(self.now(), 4)
            self.events.append(event)

    def word(self, text: str, end_s: float | None, is_final: bool) -> None:
        if end_s is None:
            return
        entry = {"t": round(self.now(), 4), "w": text, "end_s": round(float(end_s), 3)}
        if is_final:
            self.final_words.append(entry)
            return
        key = (text, round(float(end_s), 2))
        if key not in self._partial_seen:
            self._partial_seen.add(key)
            self.partial_words.append(entry)

    def control(self, name: str, audio_s: float | None = None) -> None:
        self.controls.append({"t": round(self.now(), 4), "name": name, "audio_s": audio_s})

    def raw(self, message: object) -> None:
        entry = {"t": round(self.now(), 4), "m": message}
        if len(self.raw_head) < RAW_HEAD:
            self.raw_head.append(entry)
        else:
            self.raw_tail.append(entry)

    def export(self) -> dict:
        return {
            "connect_s": self.connect_s,
            "notes": self.notes,
            "controls": self.controls,
            "events": self.events,
            "final_words": self.final_words,
            "partial_words": self.partial_words,
            "raw_sample": self.raw_head + list(self.raw_tail),
        }


def finalize_schedule(variant: str, reference: dict) -> list[float]:
    if variant == "finalize_gap":
        return [
            s.get("speech_end_s", s["end_s"]) + GAP_FINALIZE_OFFSET_S
            for s in reference["sentences"][:-1]
        ]
    if variant == "finalize_mid":
        points, t = [], MID_FINALIZE_EVERY_S
        while t < reference["duration_s"] - 0.5:
            points.append(round(t, 3))
            t += MID_FINALIZE_EVERY_S
        return points
    return []


async def _stream_audio(rec: Recorder, pcm: bytes, send, finalize, schedule: list[float]) -> None:
    step = SAMPLE_RATE * CHUNK_MS // 1000 * 2
    pending = list(schedule)
    rec.t0 = time.monotonic()
    for offset in range(0, len(pcm), step):
        chunk = pcm[offset : offset + step]
        audio_end_s = (offset + len(chunk)) / 2 / SAMPLE_RATE
        delay = rec.t0 + audio_end_s - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        await send(chunk)
        while finalize is not None and pending and pending[0] <= audio_end_s:
            pending.pop(0)
            await finalize()
            rec.control("finalize", round(audio_end_s, 3))
    rec.control("audio_end", round(len(pcm) / 2 / SAMPLE_RATE, 3))


async def run_soniox(rec, pcm, lang, variant, schedule, key, options) -> dict:
    import websockets

    config = {
        "model": SONIOX_MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        "enable_endpoint_detection": False,
        "enable_language_identification": False,
        "enable_speaker_diarization": True,
        "language_hints": [lang],
    }
    started = time.monotonic()
    async with websockets.connect(
        SONIOX_URL,
        additional_headers={"Authorization": f"Bearer {key}"},
        ping_interval=None,
        open_timeout=20,
        max_size=None,
    ) as ws:
        rec.connect_s = round(time.monotonic() - started, 4)
        await ws.send(json.dumps(config))

        async def receive() -> None:
            async for message in ws:
                data = json.loads(message)
                rec.raw(data)
                if data.get("error_code") or data.get("error_message"):
                    raise RuntimeError(
                        f"soniox error {data.get('error_code')}: {data.get('error_message')}"
                    )
                tokens = data.get("tokens") or []
                text_tokens = [t for t in tokens if not str(t.get("text", "")).startswith("<")]
                for t in text_tokens:
                    end_ms = t.get("end_ms")
                    rec.word(
                        t["text"], None if end_ms is None else end_ms / 1000.0, bool(t["is_final"])
                    )
                if any(t.get("is_final") and t.get("text") == "<fin>" for t in tokens):
                    rec.control("fin_ack")
                rec.update(
                    final_append="".join(t["text"] for t in text_tokens if t.get("is_final")),
                    tail="".join(t["text"] for t in text_tokens if not t.get("is_final")),
                )
                if data.get("finished"):
                    return

        receiver = asyncio.create_task(receive())
        try:
            await _stream_audio(
                rec, pcm, ws.send, lambda: ws.send(json.dumps({"type": "finalize"})), schedule
            )
            await ws.send("")
        except websockets.ConnectionClosed:
            if receiver.done() and not receiver.cancelled() and receiver.exception():
                raise receiver.exception() from None
            raise
        await asyncio.wait_for(receiver, TAIL_WAIT_S)
    return {"endpoint": SONIOX_URL, "config": config}


async def run_deepgram(rec, pcm, lang, variant, schedule, key, options) -> dict:
    import websockets

    params = {
        "model": DEEPGRAM_MODEL,
        "language": lang,
        "encoding": "linear16",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "interim_results": "false" if variant == "nointerim" else "true",
        "punctuate": "true",
        "vad_events": "false",
        "endpointing": "false",
    }
    url = f"{DEEPGRAM_URL}?{urlencode(params)}"
    started = time.monotonic()
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Token {key}"}, open_timeout=20, max_size=None
    ) as ws:
        rec.connect_s = round(time.monotonic() - started, 4)

        async def receive() -> None:
            async for message in ws:
                data = json.loads(message)
                rec.raw(data)
                if data.get("type") != "Results":
                    continue
                alternative = ((data.get("channel") or {}).get("alternatives") or [{}])[0]
                text = alternative.get("transcript") or ""
                is_final = bool(data.get("is_final"))
                for w in alternative.get("words") or []:
                    rec.word(w.get("word", ""), w.get("end"), is_final)
                if data.get("from_finalize"):
                    rec.control("fin_ack")
                if is_final:
                    rec.update(final_append=f" {text}" if text else "", tail="")
                else:
                    rec.update(tail=text)

        receiver = asyncio.create_task(receive())
        await _stream_audio(
            rec, pcm, ws.send, lambda: ws.send(json.dumps({"type": "Finalize"})), schedule
        )
        await ws.send(json.dumps({"type": "CloseStream"}))
        try:
            await asyncio.wait_for(receiver, TAIL_WAIT_S)
        except TimeoutError:
            rec.notes.append("tail wait timed out")
    return {"endpoint": DEEPGRAM_URL, "config": params}


async def run_qwen(rec, pcm, lang, variant, schedule, key, options) -> dict:
    import websockets

    endpoint = options.get("qwen_endpoint") or QWEN_URL
    task_id = uuid.uuid4().hex
    parameters = {
        "format": "pcm",
        "sample_rate": SAMPLE_RATE,
        "semantic_punctuation_enabled": variant == "semantic",
        "max_sentence_silence": 800,
        "multi_threshold_mode_enabled": False,
        "heartbeat": True,
        "language_hints": [lang],
    }
    started = time.monotonic()
    async with websockets.connect(
        endpoint,
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=20,
        max_size=None,
    ) as ws:
        rec.connect_s = round(time.monotonic() - started, 4)
        await ws.send(
            json.dumps(
                {
                    "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                    "payload": {
                        "task_group": "audio",
                        "task": "asr",
                        "function": "recognition",
                        "model": QWEN_MODEL,
                        "parameters": parameters,
                        "input": {},
                    },
                }
            )
        )
        ready = asyncio.get_running_loop().create_future()

        async def receive() -> None:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                data = json.loads(message)
                rec.raw(data)
                header = data.get("header") or {}
                event = header.get("event")
                if event == "task-started" and not ready.done():
                    ready.set_result(True)
                elif event == "task-failed":
                    rec.notes.append(
                        f"task-failed {header.get('error_code')} {header.get('error_message')}"
                    )
                    if not ready.done():
                        ready.set_result(False)
                    return
                elif event == "task-finished":
                    return
                elif event == "result-generated":
                    output = (data.get("payload") or {}).get("output") or {}
                    sentence = output.get("sentence") or {}
                    if sentence.get("heartbeat"):
                        continue
                    text = sentence.get("text") or ""
                    sentence_end = bool(sentence.get("sentence_end"))
                    for w in sentence.get("words") or []:
                        end = w.get("end_time")
                        rec.word(
                            w.get("text", ""), None if end is None else end / 1000.0, sentence_end
                        )
                    if sentence_end:
                        rec.update(final_append=f" {text}", tail="")
                    else:
                        rec.update(tail=text)

        receiver = asyncio.create_task(receive())
        if await asyncio.wait_for(ready, 20):
            await _stream_audio(rec, pcm, ws.send, None, schedule)
            await ws.send(
                json.dumps(
                    {
                        "header": {
                            "action": "finish-task",
                            "task_id": task_id,
                            "streaming": "duplex",
                        },
                        "payload": {"input": {}},
                    }
                )
            )
        await asyncio.wait_for(receiver, TAIL_WAIT_S)
    return {"endpoint": endpoint, "model": QWEN_MODEL, "config": parameters}


def _seconds(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return value.total_seconds()
    text = str(value)
    try:
        return float(text[:-1]) if text.endswith("s") else float(text)
    except ValueError:
        return None


async def run_gemini(rec, pcm, lang, variant, schedule, key, options) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(
            mode="VERBATIM", language_codes=[lang]
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                prefix_padding_ms=500, silence_duration_ms=400
            ),
        ),
    )
    started = time.monotonic()
    async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
        rec.connect_s = round(time.monotonic() - started, 4)
        done = asyncio.Event()

        async def receive() -> None:
            while not done.is_set():
                received = False
                async for message in session.receive():
                    received = True
                    content = getattr(message, "server_content", None)
                    transcription = getattr(content, "input_transcription", None)
                    if transcription is None:
                        continue
                    text = transcription.text or ""
                    rec.raw({"text": text, "finished": getattr(transcription, "finished", None)})
                    for w in getattr(transcription, "words", None) or ():
                        rec.word(
                            getattr(w, "word", "") or "",
                            _seconds(getattr(w, "end_offset", None)),
                            True,
                        )
                    rec.update(final_append=text)
                if not received:
                    await asyncio.sleep(0.05)

        async def send(chunk: bytes) -> None:
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
            )

        receiver = asyncio.create_task(receive())
        await _stream_audio(rec, pcm, send, None, schedule)
        await session.send_realtime_input(audio_stream_end=True)
        await asyncio.sleep(TAIL_WAIT_S)
        done.set()
        if receiver.done() and not receiver.cancelled() and receiver.exception():
            raise receiver.exception()
        receiver.cancel()
    return {
        "model": GEMINI_MODEL,
        "config": {
            "mode": "VERBATIM",
            "language_codes": [lang],
            "prefix_padding_ms": 500,
            "silence_duration_ms": 400,
        },
    }


RUNNERS = {
    "soniox": run_soniox,
    "deepgram": run_deepgram,
    "qwen": run_qwen,
    "gemini": run_gemini,
}
