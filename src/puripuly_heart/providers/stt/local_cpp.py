from __future__ import annotations

import asyncio
import contextlib
import tempfile
import wave
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar, cast
from uuid import uuid4

import numpy as np

from puripuly_heart.core.asr_worker import ASRWorkerClientPort, ASRWorkerProcessFactoryPort
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.local_asr.local_stt_assets import (
    ORUKEET_MODEL_ID,
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

    def __post_init__(self) -> None:
        super(LocalCppSTTBackend, self).__post_init__()
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
        self.language_hint = None if self.model_id == ORUKEET_MODEL_ID else options.language_hint

    async def close(self) -> None:
        await super(LocalCppSTTBackend, self).close()
        assert self.process_factory is not None
        await self.process_factory.close_pending_startups()
        await self._scope.close()

    async def _settle(self, awaitable: Awaitable[_T], *, request_id: str | None = None) -> _T:
        task = start_lifecycle_task(self._scope, awaitable, name=request_id or "activation")
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            client = cast(ASRWorkerClientPort | None, self._recognizer)
            if client is not None and request_id is not None:
                with contextlib.suppress(Exception):
                    await client.cancel(request_id)
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if task.done() and not task.cancelled():
                with contextlib.suppress(Exception):
                    task.result()
            raise

    async def _load_resource(self) -> object:
        assert self.process_factory is not None
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
            start_lifecycle_task(self._scope, self._drain_events(client), name="events")
            return client
        except BaseException:
            await client.close()
            raise

    async def _drain_events(self, client: ASRWorkerClientPort) -> None:
        try:
            while True:
                await client.next_event()
        except Exception:
            return

    async def _close_resource(self, resource: object) -> None:
        await cast(ASRWorkerClientPort, resource).close()

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        client = cast(ASRWorkerClientPort, await self._ensure_recognizer())
        async with self._decode_lock:
            if self._closed:
                raise RuntimeError("Local STT backend is closed")
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
                return result.text.strip()


def _write_audio(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())
