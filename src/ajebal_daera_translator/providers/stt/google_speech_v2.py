from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Protocol

from ajebal_daera_translator.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)


class GoogleSpeechV2Client(Protocol):
    def streaming_recognize(self, *, requests: Iterable[object]) -> Iterable[object]: ...


@dataclass(slots=True)
class GoogleSpeechV2Backend(STTBackend):
    recognizer: str
    endpoint: str = "speech.googleapis.com"
    sample_rate_hz: int = 16000
    language_codes: tuple[str, ...] = ("ko-KR",)
    client: GoogleSpeechV2Client | None = None

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if not self.recognizer:
            raise ValueError("recognizer must be non-empty")
        if not self.language_codes:
            raise ValueError("language_codes must be non-empty")

        from google.cloud import speech_v2  # type: ignore

        client = self.client or speech_v2.SpeechClient(client_options={"api_endpoint": self.endpoint})
        session = _GoogleSpeechV2Session(
            speech_v2=speech_v2,
            client=client,
            recognizer=self.recognizer,
            sample_rate_hz=self.sample_rate_hz,
            language_codes=self.language_codes,
        )
        await session.start()
        return session


@dataclass(slots=True)
class _GoogleSpeechV2Session(STTBackendSession):
    speech_v2: Any
    client: GoogleSpeechV2Client
    recognizer: str
    sample_rate_hz: int
    language_codes: tuple[str, ...]

    _req_q: queue.Queue[object | None] = field(init=False, repr=False)
    _event_q: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(init=False, repr=False)
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._req_q = queue.Queue()
        self._event_q = asyncio.Queue()

    async def start(self) -> None:
        initial = self._build_initial_request()
        self._req_q.put(initial)

        loop = asyncio.get_running_loop()

        def _run() -> None:
            try:
                responses = self.client.streaming_recognize(requests=_request_iter(self._req_q))
                for response in responses:
                    for event in _extract_transcript_events(response):
                        loop.call_soon_threadsafe(self._event_q.put_nowait, event)
            except BaseException as exc:
                loop.call_soon_threadsafe(self._event_q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(self._event_q.put_nowait, None)

        self._task = asyncio.create_task(asyncio.to_thread(_run))

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped:
            return
        req = self.speech_v2.StreamingRecognizeRequest(audio=pcm16le)
        self._req_q.put(req)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._req_q.put(None)

    async def close(self) -> None:
        await self.stop()
        if self._task is None:
            return
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._event_q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _build_initial_request(self) -> object:
        # Build a minimal explicit decoding config (mono PCM16LE @ sample_rate_hz).
        decoding = self.speech_v2.ExplicitDecodingConfig(
            encoding=self.speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            audio_channel_count=1,
        )
        config = self.speech_v2.RecognitionConfig(
            explicit_decoding_config=decoding,
            language_codes=list(self.language_codes),
        )

        try:
            features = self.speech_v2.StreamingRecognitionFeatures(interim_results=True)
            streaming_config = self.speech_v2.StreamingRecognitionConfig(config=config, streaming_features=features)
        except Exception:
            streaming_config = self.speech_v2.StreamingRecognitionConfig(config=config)

        return self.speech_v2.StreamingRecognizeRequest(
            recognizer=self.recognizer,
            streaming_config=streaming_config,
        )


def _request_iter(q: queue.Queue[object | None]):
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def _extract_transcript_events(response: object) -> list[STTBackendTranscriptEvent]:
    results = getattr(response, "results", None) or []
    out: list[STTBackendTranscriptEvent] = []
    for result in results:
        is_final = bool(getattr(result, "is_final", False))
        alternatives = getattr(result, "alternatives", None) or []
        if not alternatives:
            continue
        text = str(getattr(alternatives[0], "transcript", "") or "").strip()
        if not text:
            continue
        out.append(STTBackendTranscriptEvent(text=text, is_final=is_final))
    return out
