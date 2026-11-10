from __future__ import annotations

import asyncio
import hmac
import json
import math
import secrets
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from puripuly_heart.core.lifecycle import LifecycleScope

from .protocol import PhotoProviderError

_IMAGE_TIMING_STAGES = frozenset({"mask", "erase", "typeset", "roi_upload", "compose_gpu"})


def valid_wrist_calibration(value: object) -> bool:
    def rigid(matrix):
        if not isinstance(matrix, list) or len(matrix) != 3:
            return False
        if any(not isinstance(row, list) or len(row) != 4 for row in matrix):
            return False
        if any(
            type(v) not in (int, float) or not -3.402823466e38 <= v <= 3.402823466e38
            for row in matrix
            for v in row
        ):
            return False
        for i in range(3):
            for j in range(3):
                dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
                if abs(dot - int(i == j)) >= 0.01:
                    return False
        determinant = sum(
            matrix[0][i]
            * (
                matrix[1][(i + 1) % 3] * matrix[2][(i + 2) % 3]
                - matrix[1][(i + 2) % 3] * matrix[2][(i + 1) % 3]
            )
            for i in range(3)
        )
        return determinant > 0.99

    if not isinstance(value, dict) or set(value) != {"transform", "reference"}:
        return False
    reference = value["reference"]
    return (
        rigid(value["transform"])
        and isinstance(reference, dict)
        and set(reference) == {"model", "serial", "handgrip_transform"}
        and rigid(reference["handgrip_transform"])
        and all(
            isinstance(reference[key], str)
            and 0 < len(reference[key].encode("utf-8")) <= 4096
            and reference[key].isprintable()
            for key in ("model", "serial")
        )
    )


class PhotoBridge:
    def __init__(self, *, control: Callable[[dict, int], dict | None]) -> None:
        self._control = control
        self._scope = LifecycleScope("PhotoBridge")
        self._server: Server | None = None
        self._connection: ServerConnection | None = None
        self._snapshot = {"type": "photo_snapshot", "revision": 0, "photo": None}
        self._changed = asyncio.Event()
        self._commands: deque[dict] = deque()
        self._generation = 0
        self._instance_id = ""
        self._token = ""
        self._backend = "openvr"
        self.ready = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.shutdown_complete = asyncio.Event()
        self.receipts: deque[dict] = deque(maxlen=128)
        self._url = ""
        self._preparations: dict[tuple[int, str, int], asyncio.Future] = {}
        self.abort_preparation: Callable[[int], Awaitable[None]] | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handle,
            "127.0.0.1",
            0,
            max_size=16384,
            max_queue=16,
            ping_interval=10,
            ping_timeout=10,
            close_timeout=1,
            origins=[None],
        )
        port = self._server.sockets[0].getsockname()[1]
        self._url = f"ws://127.0.0.1:{port}"

    async def activate(self, generation: int, *, backend: str) -> dict:
        if backend not in ("openvr", "probe"):
            raise ValueError("unsupported photo backend")
        if self._connection is not None:
            await self._connection.close(code=1001)
            self._connection = None
        self._generation = generation
        self._instance_id = str(uuid4())
        self._token = secrets.token_urlsafe(32)
        self._backend = backend
        self.ready.clear()
        self.disconnected.clear()
        self.shutdown_complete.clear()
        self._commands.clear()
        return {
            "protocol_version": 3,
            "role": "photo",
            "bridge_url": self._url,
            "session_token": self._token,
            "instance_id": self._instance_id,
            "generation": generation,
            "backend": backend,
        }

    def publish(self, snapshot: dict) -> None:
        self._snapshot = snapshot
        self._changed.set()

    def send(self, command: dict) -> bool:
        if self._connection is None or len(self._commands) >= 16:
            return False
        if (
            command.get("type", "").startswith("photo_probe_")
            or command.get("type") == "photo_debug_preview_source"
        ) and self._backend != "probe":
            return False
        self._commands.append(command)
        self._changed.set()
        return True

    async def prepare(
        self,
        photo_id: str,
        attempt_id: int,
        source_path: Path,
        output_dir: Path,
        *,
        expected_file_id: tuple[int, int],
    ) -> dict:
        generation = self._generation
        key = (generation, photo_id, attempt_id)
        future = asyncio.get_running_loop().create_future()
        if key in self._preparations:
            raise PhotoProviderError("photo_preparation_failed", retryable=False)
        self._preparations[key] = future
        if not self.send(
            {
                "type": "photo_prepare",
                "photo_id": photo_id,
                "attempt_id": attempt_id,
                "source_path": str(source_path),
                "output_dir": str(output_dir),
                "expected_file_id": [str(item) for item in expected_file_id],
            }
        ):
            self._preparations.pop(key)
            raise PhotoProviderError("photo_preparation_failed", retryable=False)
        cancelled = False
        deadline = asyncio.get_running_loop().time() + 15
        try:
            while not future.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=max(0, deadline - asyncio.get_running_loop().time()),
                    )
                except asyncio.CancelledError:
                    cancelled = True
                except TimeoutError:
                    if self.abort_preparation is not None:
                        await self.abort_preparation(generation)
                    if not future.done():
                        future.set_exception(
                            PhotoProviderError("photo_preparation_failed", retryable=False)
                        )
            result = future.result()
            if cancelled:
                raise asyncio.CancelledError
            return result
        finally:
            self._preparations.pop(key, None)

    def renderer_stopped(self, generation: int) -> None:
        for (owner, _, _), future in self._preparations.items():
            if owner == generation and not future.done():
                future.set_exception(
                    PhotoProviderError("photo_preparation_failed", retryable=False)
                )

    def _prepared(self, value: dict, generation: int) -> None:
        future = self._preparations.get((generation, value["photo_id"], value["attempt_id"]))
        if future is None or future.done():
            return
        if value["type"] == "photo_prepare_failed":
            future.set_exception(PhotoProviderError("photo_preparation_failed", retryable=False))
        else:
            future.set_result(value)

    def _valid_auth(self, value: object) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "type",
            "protocol_version",
            "role",
            "session_token",
            "instance_id",
            "generation",
        }:
            return False
        token = value.get("session_token")
        return (
            value.get("type") == "photo_auth"
            and value.get("protocol_version") == 3
            and type(value.get("protocol_version")) is int
            and value.get("role") == "photo"
            and isinstance(token, str)
            and hmac.compare_digest(token.encode(), self._token.encode())
            and value.get("instance_id") == self._instance_id
            and type(value.get("generation")) is int
            and value.get("generation") == self._generation
            and self._generation > 0
        )

    def _valid_control(self, value: object) -> bool:
        if not isinstance(value, dict):
            return False
        kind = value.get("type")
        if not isinstance(kind, str):
            return False
        fields = {
            "photo_pull": {"photo_id": str, "event_id": str},
            "photo_hidden": {"photo_id": str},
            "photo_first_block_drawn": {"photo_id": str, "attempt_id": int, "block_id": int},
            "photo_image_timing": {
                "photo_id": str,
                "attempt_id": int,
                "block_id": int,
                "stage": str,
                "duration_ms": (int, float),
            },
            "photo_state_received": {
                "photo_id": (str, type(None)),
                "attempt_id": int,
                "revision": int,
                "block_count": int,
            },
            "photo_ready": {},
            "photo_shutdown_complete": {},
            "photo_debug_quiescent": {"request_id": str},
            "photo_debug_wrist_calibrated": {"calibration": dict},
            "photo_prepared": {
                "photo_id": str,
                "attempt_id": int,
                "source_path": str,
                "jpeg_path": str,
                "width": int,
                "height": int,
                "scale": (int, float),
                "offset_x": int,
                "offset_y": int,
                "preparation_ms": (int, float),
            },
            "photo_prepare_failed": {"photo_id": str, "attempt_id": int, "code": str},
        }.get(kind)
        if kind == "photo_prepared":
            fields.update(
                (key, (int, float)) for key in ("decode_ms", "resize_ms", "jpeg_ms") if key in value
            )
        if fields is None or set(value) != {"type", *fields}:
            return False
        for key, expected in fields.items():
            item = value[key]
            if type(item) not in (expected if isinstance(expected, tuple) else (expected,)):
                return False
            if isinstance(item, str) and (
                not item or len(item) > (32768 if key in ("source_path", "jpeg_path") else 128)
            ):
                return False
            if type(item) is int and item < 0:
                return False
        if kind == "photo_debug_quiescent" and not value["request_id"].isascii():
            return False
        if kind == "photo_debug_wrist_calibrated":
            return valid_wrist_calibration(value["calibration"])
        if kind == "photo_image_timing":
            if value["stage"] not in _IMAGE_TIMING_STAGES:
                return False
            try:
                finite = math.isfinite(value["duration_ms"])
            except OverflowError:
                return False
            if not finite or value["duration_ms"] < 0:
                return False
        return True

    def _current_image_timing(self, value: dict) -> bool:
        photo = self._snapshot.get("photo")
        if (
            photo is None
            or photo.get("photo_id") != value["photo_id"]
            or photo.get("attempt_id") != value["attempt_id"]
            or photo.get("phase") == "expired"
        ):
            return False
        if any(block["block_id"] == value["block_id"] for block in photo.get("blocks", ())):
            return True
        return value["stage"] in {"mask", "erase"} and any(
            preparation["block_id"] == value["block_id"] and preparation.get("source") is not None
            for preparation in photo.get("preparations", ())
        )

    async def _handle(self, connection: ServerConnection) -> None:
        sender: asyncio.Task | None = None
        generation = self._generation
        try:
            raw = await asyncio.wait_for(connection.recv(), timeout=3)
            if not isinstance(raw, str):
                await connection.close(code=1008)
                return
            try:
                auth = json.loads(raw)
            except ValueError, TypeError:
                await connection.close(code=1008)
                return
            if not self._valid_auth(auth) or self._connection is not None:
                await connection.close(code=1008)
                return
            generation = self._generation
            self._connection = connection
            snapshot = self._snapshot
            await connection.send(json.dumps(snapshot, separators=(",", ":")))
            sender = self._scope.create_task(
                self._send(connection, snapshot["revision"]), name=f"writer-{uuid4()}"
            )
            async for raw in connection:
                if not isinstance(raw, str) or generation != self._generation:
                    await connection.close(code=1008)
                    return
                try:
                    value = json.loads(raw)
                except ValueError, TypeError:
                    await connection.close(code=1008)
                    return
                if not self._valid_control(value):
                    await connection.close(code=1008)
                    return
                kind = value["type"]
                if kind == "photo_ready":
                    self.ready.set()
                elif kind == "photo_shutdown_complete":
                    self.shutdown_complete.set()
                elif kind == "photo_state_received":
                    self.receipts.append({**value, "generation": generation})
                elif kind in ("photo_prepared", "photo_prepare_failed"):
                    self._prepared(value, generation)
                elif kind == "photo_image_timing":
                    if self._current_image_timing(value):
                        self._control(value, generation)
                else:
                    command = self._control(value, generation)
                    if command is not None:
                        self.send(command)
        except ConnectionClosed, TimeoutError:
            pass
        finally:
            if sender is not None:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            if self._connection is connection:
                self._connection = None
                self.disconnected.set()

    async def _send(self, connection: ServerConnection, revision: int) -> None:
        try:
            while True:
                self._changed.clear()
                if self._snapshot["revision"] != revision:
                    snapshot = self._snapshot
                    await connection.send(json.dumps(snapshot, separators=(",", ":")))
                    revision = snapshot["revision"]
                while self._commands:
                    await connection.send(
                        json.dumps(self._commands.popleft(), separators=(",", ":"))
                    )
                await self._changed.wait()
        except ConnectionClosed:
            return

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self._scope.close()
        self._commands.clear()
