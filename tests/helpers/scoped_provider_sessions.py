from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTSessionProjection,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.deepgram import _DeepgramSDKSession
from puripuly_heart.providers.stt.elevenlabs_scribe import _ElevenLabsScribeSession
from puripuly_heart.providers.stt.gemini_transcribe import _GeminiTranscribeLiveSession
from puripuly_heart.providers.stt.soniox import _SonioxSession


def _request(provider: str, order: int = 1) -> STTProviderTurnRequest:
    settings = AudioSegmentSettingsSnapshot(
        provider_id=provider,
        provider_signature=(provider,),
        runtime_signature=(provider,),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )
    segment = AudioSegmentIdentity(
        activation_generation=1,
        segment_order=order,
        segment_id=uuid4(),
        capture_epoch=1,
    )
    return STTProviderTurnRequest(
        identity=STTProviderTurnIdentity(
            segment=segment,
            provider_epoch_id=f"epoch-{order}",
            provider_turn_id=f"turn-{order}",
        ),
        settings=settings,
    )


async def _wait(predicate) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


def _deepgram_result(text: str, *, from_finalize: bool = False, is_final: bool = True):
    from deepgram.extensions.types.sockets import ListenV1ResultsEvent
    from deepgram.extensions.types.sockets.listen_v1_results_event import (
        ListenV1Alternative,
        ListenV1Channel,
        ListenV1ModelInfo,
        ListenV1ResultsMetadata,
    )

    return ListenV1ResultsEvent(
        type="Results",
        channel_index=[0, 1],
        duration=0.1,
        start=0.0,
        is_final=is_final,
        speech_final=False,
        channel=ListenV1Channel(
            alternatives=[ListenV1Alternative(transcript=text, confidence=1.0, words=[])]
        ),
        metadata=ListenV1ResultsMetadata(
            request_id="session-request",
            model_info=ListenV1ModelInfo(name="nova-3", version="1", arch="nova"),
            model_uuid="model",
        ),
        from_finalize=from_finalize,
    )


def _deepgram_session(
    drain_timeout_s: float = 10.0,
    *,
    order: int = 1,
    epoch: str | None = None,
) -> _DeepgramSDKSession:
    session = _DeepgramSDKSession(
        api_key="k",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=[],
        drain_timeout_s=drain_timeout_s,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=epoch or f"epoch-{order}",
        ),
    )
    return session


class _FakeGeminiLive:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.queue: asyncio.Queue[object] = asyncio.Queue()
        self.closed = False

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)

    async def receive(self):
        while True:
            item = await self.queue.get()
            if isinstance(item, BaseException):
                raise item
            yield item

    async def close(self) -> None:
        self.closed = True

    def push(self, item: object) -> None:
        self.queue.put_nowait(item)


_MISSING = object()


def _gemini_message(*, final: object = _MISSING, interim: object = _MISSING, ack: bool = False):
    from google.genai import types

    content = None
    if final is not _MISSING or interim is not _MISSING:
        content = types.LiveServerContent(
            interim_input_transcription=(
                types.Transcription(text=str(interim)) if interim is not _MISSING else None
            ),
            input_transcription=(
                types.Transcription(text=str(final)) if final is not _MISSING else None
            ),
        )
    activity = None
    if ack:
        activity = types.VoiceActivity(voice_activity_type=types.VoiceActivityType.ACTIVITY_END)
    return types.LiveServerMessage(server_content=content, voice_activity=activity)


async def _gemini_session(timeout: float = 0.05, *, order: int = 1, epoch: str | None = None):
    live = _FakeGeminiLive()
    session = _GeminiTranscribeLiveSession(
        api_key="k",
        language_codes=[],
        custom_vocabulary=[],
        model="model",
        sample_rate_hz=16000,
        connect_timeout_s=10.0,
        drain_timeout_s=timeout,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=epoch or f"epoch-{order}",
        ),
    )
    session._live_session = live
    session._send_task = asyncio.create_task(session._send_loop())
    session._recv_task = asyncio.create_task(session._recv_loop())
    return session, live


class _FakeSonioxWebSocket:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.send_gate = asyncio.Event()
        self.send_gate.set()
        self.queue: asyncio.Queue[object] = asyncio.Queue()
        self.closed = False

    async def send(self, payload: object) -> None:
        self.sent.append(payload)
        await self.send_gate.wait()

    async def recv(self):
        return await self.queue.get()

    async def close(self) -> None:
        self.closed = True

    def push(self, payload: object) -> None:
        self.queue.put_nowait(payload)


def _soniox_session(*, order: int = 1, epoch: str | None = None):
    ws = _FakeSonioxWebSocket()
    session = _SonioxSession(
        api_key="k",
        model="model",
        endpoint="endpoint",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=100.0,
        trailing_silence_ms=100,
        connect_timeout_s=5.0,
        enable_language_identification=True,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=epoch or f"epoch-{order}",
        ),
    )
    session._ws = ws
    session._send_task = asyncio.create_task(session._send_loop())
    session._recv_task = asyncio.create_task(session._recv_loop())
    return session, ws


class _FakeScribeConnection:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.send_gate = asyncio.Event()
        self.send_gate.set()
        self.commit_gate = asyncio.Event()
        self.commit_gate.set()
        self.commits = 0
        self.closed = False

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)
        await self.send_gate.wait()

    async def commit(self) -> None:
        self.commits += 1
        await self.commit_gate.wait()

    async def close(self) -> None:
        self.closed = True


def _scribe_session(*, order: int = 1, epoch: str | None = None):
    connection = _FakeScribeConnection()
    session = _ElevenLabsScribeSession(
        api_key="k",
        language_code="en",
        keyterms=(),
        model="model",
        sample_rate_hz=16000,
        connect_timeout_s=10.0,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=epoch or f"epoch-{order}",
        ),
    )
    session._connection = connection
    return session, connection


def _owned_deepgram_events(
    ledger: PeerAudioSegmentLedger,
    *,
    start_sample: int,
) -> tuple[object, list[object], object]:
    segment_id = uuid4()

    def capture(sequence: int, start: int, end: int) -> AudioCaptureSpan:
        return AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=sequence,
            source_sample_rate_hz=16000,
            source_start_sample=start,
            source_end_sample=end,
            source_start_monotonic_s=start / 16000,
            source_end_monotonic_s=end / 16000,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=start,
            normalized_end_sample=end,
        )

    first_range = capture(1, start_sample, start_sample + 4)
    start = ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            np.empty((0,), dtype=np.float32),
            np.ones(4, dtype=np.float32),
            chunk_capture=(first_range,),
        ),
        now_monotonic_s=start_sample / 16000,
    )
    chunks: list[object] = []
    for index in range(1, 21):
        chunk_start = start_sample + index * 4
        chunks.append(
            ledger.observe_vad_event(
                SpeechChunk(
                    segment_id,
                    np.ones(4, dtype=np.float32),
                    chunk_capture=(capture(index + 1, chunk_start, chunk_start + 4),),
                ),
                now_monotonic_s=(chunk_start + 4) / 16000,
            )
        )
    end = ledger.observe_vad_event(
        SpeechEnd(segment_id, trailing_silence_ms=800, reason="silence"),
        now_monotonic_s=(start_sample + 84) / 16000,
    )
    return start, chunks, end
