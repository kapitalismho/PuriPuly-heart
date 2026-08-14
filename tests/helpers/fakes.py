from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from puripuly_heart.core.self_capture import (
    SelfCaptureProviderStatus,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.domain.models import OSCMessage


@dataclass(slots=True)
class FakeSender:
    sent: list[str]
    typing: list[bool]

    def __init__(self) -> None:
        self.sent = []
        self.typing = []

    def send_chatbox(self, text: str) -> None:
        self.sent.append(text)

    def send_typing(self, is_typing: bool) -> None:
        self.typing.append(is_typing)


@dataclass(slots=True)
class SpeechAwareFakeSession:
    audio: list[bytes]
    _queue: asyncio.Queue
    _seen_speech: bool = False

    def __init__(self) -> None:
        self.audio = []
        self._queue = asyncio.Queue()
        self._seen_speech = False

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio.append(pcm16le)
        is_silence = all(b == 0 for b in pcm16le)
        if not is_silence:
            self._seen_speech = True
            await self._queue.put(STTBackendTranscriptEvent(text="PARTIAL", is_final=False))
        elif self._seen_speech:
            await self._queue.put(STTBackendTranscriptEvent(text="FINAL", is_final=True))
            self._seen_speech = False

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)
        if self._seen_speech:
            await self._queue.put(STTBackendTranscriptEvent(text="FINAL", is_final=True))
            self._seen_speech = False

    async def stop(self) -> None:
        await self._queue.put(None)

    async def close(self) -> None:
        await self._queue.put(None)

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class SpeechAwareFakeBackend:
    async def open_session(self) -> SpeechAwareFakeSession:
        return SpeechAwareFakeSession()


@dataclass(slots=True)
class RecordingOscQueue:
    messages: list[OSCMessage]
    typing: list[bool]
    immediate_messages: list[str]
    process_due_calls: int
    immediate_result: bool

    def __init__(self, *, immediate_result: bool = True) -> None:
        self.messages = []
        self.typing = []
        self.immediate_messages = []
        self.process_due_calls = 0
        self.immediate_result = immediate_result

    def enqueue(self, message: OSCMessage) -> None:
        self.messages.append(message)

    def send_typing(self, is_typing: bool) -> None:
        self.typing.append(is_typing)

    def set_typing_reason(self, reason: str, active: bool) -> None:
        self.typing.append(active)

    def send_immediate(self, text: str) -> bool:
        self.immediate_messages.append(text)
        return self.immediate_result

    def process_due(self) -> None:
        self.process_due_calls += 1


class _BaseThreadStub:
    def __init__(self, target=None, name=None, daemon=None):
        _ = (name, daemon)
        self._target = target

    def join(self, timeout=None):
        _ = timeout
        return None


class TargetThread(_BaseThreadStub):
    def start(self):
        if self._target:
            self._target()


class NoopThread(_BaseThreadStub):
    def start(self):
        return None


def self_capture_snapshot(
    desired_active: bool,
    *,
    state: SelfCaptureSessionState | None = None,
) -> SelfCaptureSessionSnapshot:
    resolved_state = state or (
        SelfCaptureSessionState.RUNNING if desired_active else SelfCaptureSessionState.STOPPED
    )
    active = desired_active and resolved_state is SelfCaptureSessionState.RUNNING
    return SelfCaptureSessionSnapshot(
        state=resolved_state,
        provider_status=(
            SelfCaptureProviderStatus.READY
            if active
            else (
                SelfCaptureProviderStatus.FAILED
                if resolved_state is SelfCaptureSessionState.FAULTED
                else SelfCaptureProviderStatus.DETACHED
            )
        ),
        desired_active=desired_active,
        effective_active=active,
        generation=1,
        provider_id="deepgram" if active else None,
        runtime_signature=("runtime",) if active else None,
        failure_reason=None,
        admission_reason=None,
        has_source=active,
        has_vad=active,
        has_loop_task=active,
        cleanup_debt=0,
        closed=False,
    )


class SelfCaptureStateStub:
    def __init__(self, desired_active: bool) -> None:
        active_resource = object() if desired_active else None
        self.snapshot = self_capture_snapshot(desired_active)
        self.loop_task = active_resource
        self.source = active_resource
        self.cleanup_source = None
        self.vad = active_resource
        self.last_cleanup_exception = None

    def invalidate_intent(self) -> SelfCaptureSessionSnapshot:
        self.snapshot = self_capture_snapshot(False)
        self.loop_task = None
        self.source = None
        self.vad = None
        return self.snapshot


def samples(value: float, n: int = 512) -> np.ndarray:
    return np.full((n,), value, dtype=np.float32)
