from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from puripuly_heart.core.speaker.contracts import BackendReady, BackendUpdate, ProbabilityFrame
from puripuly_heart.runtime_layout import current_runtime_layout

from .protocol import HEADER, PROFILES, decode_header, encode


class NemotronBackend:
    def __init__(self, *, artifact_path: Path, runtime_dir: Path, device: int, profile: str):
        if not isinstance(device, int) or isinstance(device, bool) or device < 0:
            raise ValueError("explicit nonnegative Vulkan device ordinal required")
        if profile not in PROFILES:
            raise ValueError("explicit official profile '1.04' or '0.64' required")
        self.artifact_path = Path(artifact_path).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.device, self.profile = device, profile
        self._process: asyncio.subprocess.Process | None = None
        self._spawning: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr = bytearray()
        self._lock = asyncio.Lock()
        self._generation = 0
        self._request_id = 0
        self._epoch = 0
        self._cursor = 0
        self._processed = 0
        self._immutable = 0
        self._next_frame = 0
        self._ready: BackendReady | None = None

    async def start(self) -> BackendReady:
        if self._ready is not None:
            return self._ready
        if self._lock.locked():
            raise RuntimeError("Nemotron operation already in flight")
        generation = self._generation
        async with self._lock:
            layout = current_runtime_layout()
            if layout.host_kind == "pyinstaller":
                raise RuntimeError(
                    "Nemotron requires the separately packaged Python interpreter, not a frozen host executable"
                )
            environment = layout.python_child_environment()
            if layout.host_kind == "source":
                environment["PYTHONPATH"] = str(layout.app_resource_root / "src")
            environment["PYTHONNOUSERSITE"] = "1"
            environment["GGML_VK_DISABLE_GRAPH_OPTIMIZE"] = "1"
            environment["NEMO_SPEECH_SCHEDULE_CAPTURE"] = "1"
            spawning = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    str(layout.python_executable),
                    "-m",
                    "puripuly_heart.app.adapters.nemotron.worker",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                    limit=65536,
                )
            )
            self._spawning = spawning
            try:
                process = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                process = await spawning
                self._spawning = None
                if process.returncode is None:
                    process.kill()
                await process.wait()
                raise
            if generation != self._generation:
                if process.returncode is None:
                    process.kill()
                await process.wait()
                self._spawning = None
                raise asyncio.CancelledError
            self._spawning = None
            self._process = process
            self._stderr_task = asyncio.create_task(self._read_stderr(process))
            try:
                response, payload = await self._exchange(
                    "start",
                    timeout=120,
                    artifact_path=str(self.artifact_path),
                    runtime_dir=str(self.runtime_dir),
                    device=self.device,
                    profile=self.profile,
                )
                if payload:
                    raise ValueError("unexpected readiness payload")
                expected = (PROFILES[self.profile][0] + PROFILES[self.profile][1]) * 1280 + 512
                if (
                    response["tail_padding_samples"] != expected
                    or response["max_revision_samples"] != expected
                ):
                    raise ValueError("invalid native geometry readiness")
                self._epoch, self._cursor, self._processed, self._immutable, self._next_frame = (
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                self._ready = BackendReady(
                    expected, expected, tuple(tuple(p) for p in response["diagnostics"])
                )
                return self._ready
            except BaseException:
                await self._terminate()
                raise

    async def _read_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        while block := await process.stderr.read(4096):
            self._stderr.extend(block)
            if len(self._stderr) > 16384:
                del self._stderr[:-16384]

    async def _exchange(
        self, operation: str, payload: bytes = b"", *, timeout: float = 30, **fields
    ):
        process = self._process
        if (
            process is None
            or process.returncode is not None
            or process.stdin is None
            or process.stdout is None
        ):
            raise RuntimeError("Nemotron worker unavailable")
        self._request_id += 1
        ident = self._request_id
        generation = self._generation
        async with asyncio.timeout(timeout):
            process.stdin.write(
                encode({"id": ident, "op": operation, "epoch": self._epoch, **fields}, payload)
            )
            await process.stdin.drain()
            header = await process.stdout.readexactly(HEADER.size)
            metadata_size, payload_size = decode_header(header)
            response = json.loads(await process.stdout.readexactly(metadata_size))
            result = await process.stdout.readexactly(payload_size)
        if generation != self._generation:
            raise asyncio.CancelledError
        if response.get("id") != ident:
            raise ValueError("stale/mismatched Nemotron response")
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error", "Nemotron worker failed"))
        return response, result

    async def reset(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("invalid speaker epoch")
        if self._lock.locked():
            raise RuntimeError("Nemotron operation already in flight")
        async with self._lock:
            self._require_ready()
            try:
                await self._exchange("reset", epoch=epoch)
                self._epoch, self._cursor, self._processed, self._immutable, self._next_frame = (
                    epoch,
                    0,
                    0,
                    0,
                    0,
                )
            except BaseException:
                await self._terminate()
                raise

    def _require_ready(self) -> None:
        if self._ready is None or self._process is None:
            raise RuntimeError("Nemotron backend not ready")

    def _update(self, response: dict, payload: bytes) -> BackendUpdate:
        first, count = response["first_frame"], response["count"]
        processed, immutable = response["processed"], response["immutable"]
        if (
            response.get("epoch") != self._epoch
            or not isinstance(count, int)
            or not 0 <= count <= 1024
            or first != self._next_frame
        ):
            raise ValueError("invalid Nemotron result range/epoch")
        if (
            len(payload) != count * 8 * 4
            or not self._processed <= processed <= self._cursor
            or not self._immutable <= immutable <= processed
        ):
            raise ValueError("invalid Nemotron result frontier")
        if immutable != (first + count) * 160:
            raise ValueError("undrained/revisable Nemotron probability result")
        assert self._ready is not None
        if self._cursor - immutable > self._ready.max_revision_samples:
            raise ValueError("Nemotron frontend/lookahead lag exceeds readiness bound")
        values = np.frombuffer(payload, dtype="<f4").reshape(count, 8)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("invalid Nemotron probabilities")
        frames = tuple(
            ProbabilityFrame((first + i) * 160, (first + i + 1) * 160, tuple(float(p) for p in row))
            for i, row in enumerate(values)
        )
        self._processed, self._immutable, self._next_frame = processed, immutable, first + count
        return BackendUpdate(processed, immutable, frames)

    async def feed(self, pcm: np.ndarray, compact_start_sample: int) -> BackendUpdate:
        if (
            not isinstance(pcm, np.ndarray)
            or pcm.dtype != np.float32
            or pcm.ndim != 1
            or not pcm.flags.c_contiguous
            or pcm.flags.writeable
        ):
            raise ValueError("readonly contiguous mono float32 PCM required")
        if (
            not 1 <= pcm.size <= 16000
            or compact_start_sample != self._cursor
            or not np.isfinite(pcm).all()
        ):
            raise ValueError("invalid/noncontiguous/oversized Nemotron PCM")
        if self._lock.locked():
            raise RuntimeError("Nemotron operation already in flight")
        async with self._lock:
            self._require_ready()
            try:
                response, payload = await self._exchange(
                    "feed", pcm.tobytes(), start=compact_start_sample
                )
                self._cursor += pcm.size
                return self._update(response, payload)
            except BaseException:
                await self._terminate()
                raise

    async def pause(self) -> BackendUpdate:
        if self._lock.locked():
            raise RuntimeError("Nemotron operation already in flight")
        async with self._lock:
            self._require_ready()
            try:
                response, payload = await self._exchange("pause")
                return self._update(response, payload)
            except BaseException:
                await self._terminate()
                raise

    async def _terminate(self) -> None:
        process, self._process = self._process, None
        self._ready = None
        if process is not None:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()
        task, self._stderr_task = self._stderr_task, None
        if task is not None:
            await task

    async def stop(self) -> None:
        self._generation += 1
        spawning, self._spawning = self._spawning, None
        if spawning is not None:
            process = await asyncio.shield(spawning)
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()
        await self._terminate()
        self._cursor, self._processed, self._immutable, self._next_frame = 0, 0, 0, 0
