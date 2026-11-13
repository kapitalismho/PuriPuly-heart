from __future__ import annotations

import asyncio
import contextlib
import tempfile
import unicodedata
import wave
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar, cast
from uuid import uuid4

import numpy as np

from puripuly_heart.core.asr_worker import (
    ASRWorkerClientPort,
    ASRWorkerClosedError,
    ASRWorkerProcessFactoryPort,
    ASRWorkerRequestError,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.local_asr.local_stt_assets import (
    GIGAAM_MODEL_ID,
    GRANITE_MODEL_ID,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.providers.stt.local_session import LocalOfflineSTTBackend

_T = TypeVar("_T")


@dataclass(slots=True)
class LocalCppSTTBackend(LocalOfflineSTTBackend):
    process_factory: ASRWorkerProcessFactoryPort | None = field(default=None, repr=False)
    _scope: LifecycleScope = field(init=False, repr=False)
    _resource_id: str = field(init=False, default_factory=lambda: uuid4().hex, repr=False)
    _quarantined_client: ASRWorkerClientPort | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        super(LocalCppSTTBackend, self).__post_init__()
        if self.model_id in {GRANITE_MODEL_ID, GIGAAM_MODEL_ID}:
            self.language_hint = None
        if self.stream_label not in {"self", "peer"}:
            raise ValueError("native CPU channel must be self or peer")
        if self.process_factory is None:
            raise ValueError("native CPU worker factory is required")
        self._scope = LifecycleScope(f"cpu-asr:{self.stream_label}:{id(self):x}")

    @property
    def worker_pid(self) -> int | None:
        client = cast(ASRWorkerClientPort | None, self._recognizer)
        return client.pid if client is not None else None

    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None:
        self.language_hint = (
            None if self.model_id in {GRANITE_MODEL_ID, GIGAAM_MODEL_ID} else options.language_hint
        )

    async def close(self) -> None:
        if self._quarantined_client is not None:
            await self._close_resource(self._quarantined_client)
            self._quarantined_client = None
        await super(LocalCppSTTBackend, self).close()
        assert self.process_factory is not None
        await self.process_factory.close_pending_startups()
        await self._scope.close()

    async def _settle(self, awaitable: Awaitable[_T], *, request_id: str | None = None) -> _T:
        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            draining = asyncio.ensure_future(self._cancel_and_drain(task, request_id=request_id))
            while not draining.done():
                try:
                    await asyncio.shield(draining)
                except asyncio.CancelledError:
                    continue
            draining.result()
            raise

    async def _cancel_and_drain(self, task: asyncio.Future[_T], *, request_id: str | None) -> None:
        client = cast(ASRWorkerClientPort | None, self._recognizer)
        if client is not None and request_id is not None:
            with contextlib.suppress(Exception):
                await client.cancel(request_id)
        with contextlib.suppress(Exception):
            await task

    async def _load_resource(self) -> object:
        assert self.process_factory is not None
        if self._quarantined_client is not None:
            await self._close_resource(self._quarantined_client)
            self._quarantined_client = None
        client = await self.process_factory.start(mode="persistent")
        try:
            manifest = load_local_stt_asset_manifest(self.model_id)
            await self._settle(
                client.activate(
                    model_path=self.model_dir / manifest.files[0].relative_path,
                    device_id="cpu",
                    backend="cpu",
                    channel=cast(Literal["self", "peer"], self.stream_label),
                    n_threads=2,
                )
            )
            self._resource_id = uuid4().hex
            start_lifecycle_task(self._scope, self._drain_events(client), name="events")
            return client
        except BaseException:
            self._quarantined_client = client
            await self._close_resource(client)
            self._quarantined_client = None
            raise

    async def _drain_events(self, client: ASRWorkerClientPort) -> None:
        try:
            while True:
                await client.next_event()
        except ASRWorkerClosedError:
            return

    async def _close_resource(self, resource: object) -> None:
        client = cast(ASRWorkerClientPort, resource)
        await client.close()
        if not client.is_closed:
            raise RuntimeError("provider_resource_quarantined")

    def decode_failure_retryable(self, error: BaseException | None) -> bool:
        return (
            isinstance(error, ASRWorkerRequestError)
            and error.code == "decode_failure"
            and error.attempt_started
        )

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        async with self._decode_lock:
            return await self._decode_owned(samples_f32)

    async def _decode_owned(self, samples_f32: np.ndarray) -> str:
        client = cast(ASRWorkerClientPort, await self._ensure_recognizer())
        if self._closed:
            raise RuntimeError("Local STT backend is closed")
        try:
            with tempfile.TemporaryDirectory(prefix="puripuly-cpu-asr-") as directory:
                audio_path = Path(directory) / "audio.wav"
                await run_owned_thread_call(lambda: _write_audio(audio_path, samples_f32))
                request_id = uuid4().hex
                result = await self._settle(
                    client.transcribe(
                        request_id=request_id,
                        channel=cast(Literal["self", "peer"], self.stream_label),
                        audio_path=audio_path,
                        language_hint=self.language_hint,
                    ),
                    request_id=request_id,
                )
                text = result.text.strip()
                if self.model_id == GRANITE_MODEL_ID and "<|nospeech|>" in text:
                    remainder = text.replace("<|nospeech|>", "")
                    if all(
                        char.isspace() or unicodedata.category(char).startswith("P")
                        for char in remainder
                    ):
                        return ""
                return text
        except (ASRWorkerRequestError, ASRWorkerClosedError) as exc:
            self._failed_recognizer = True
            exc.failure_resource_id = self._resource_id
            raise


def _write_audio(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())
