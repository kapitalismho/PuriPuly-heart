from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection
from websockets.exceptions import ConnectionClosed

from .diagnostics import OverlayDiagnosticsRecorder
from .manifest import normalize_overlay_logging_mode
from .protocol import OverlayPresentationSnapshot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OverlayBridge:
    session_token: str
    initial_snapshot: dict[str, object] | OverlayPresentationSnapshot | None = None
    heartbeat_interval_ms: int = 1000
    host: str = "127.0.0.1"
    port: int = 0
    overlay_instance_id: str | None = None
    diagnostics: OverlayDiagnosticsRecorder | None = None
    runtime_logging_mode: str | None = None
    desktop_runtime_controls_enabled: bool = False
    task_factory: Any | None = None

    url: str = field(init=False, default="")
    messages: asyncio.Queue[dict[str, Any]] = field(
        init=False,
        default_factory=asyncio.Queue,
    )
    _server: Server | None = field(init=False, default=None)
    _heartbeat_task: asyncio.Task[None] | None = field(init=False, default=None)
    _authenticated_connections: set[ServerConnection] = field(
        init=False,
        default_factory=set,
    )
    _snapshot: OverlayPresentationSnapshot = field(init=False)
    _snapshot_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _token_consumed: bool = field(init=False, default=False)
    _last_snapshot_revision: int = field(init=False, default=0)
    _initial_desktop_runtime_controls: list[dict[str, Any]] = field(
        init=False,
        default_factory=list,
    )

    def __post_init__(self) -> None:
        if self.initial_snapshot is None:
            self._snapshot = OverlayPresentationSnapshot()
            return
        if isinstance(self.initial_snapshot, OverlayPresentationSnapshot):
            self._snapshot = self.initial_snapshot
            self._last_snapshot_revision = self._snapshot.revision
            return
        self._snapshot = OverlayPresentationSnapshot.from_dict(self.initial_snapshot)
        self._last_snapshot_revision = self._snapshot.revision

    async def start(self) -> None:
        if self._server is not None:
            return

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=None,
        )
        socket = self._server.sockets[0]
        bound_host, bound_port = socket.getsockname()[:2]
        self.url = f"ws://{bound_host}:{bound_port}"
        self._heartbeat_task = self._create_task(
            self._run_heartbeat_loop(),
            task_name="bridge-heartbeat",
        )

    async def stop(self) -> None:
        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

        connections = list(self._authenticated_connections)
        for connection in connections:
            await connection.close()
            self._authenticated_connections.discard(connection)

        server = self._server
        if server is not None:
            server.close()
            await server.wait_closed()
            if self._server is server:
                self._server = None
        self._drain_messages()
        self._token_consumed = False
        self.url = ""

    async def replace_snapshot(self, snapshot: OverlayPresentationSnapshot) -> None:
        async with self._snapshot_lock:
            if snapshot.revision <= self._snapshot.revision:
                return
            self._snapshot = snapshot
            self._last_snapshot_revision = snapshot.revision
            await self._broadcast_json({"type": "snapshot", "payload": snapshot.to_dict()})

    async def broadcast_shutdown(self) -> None:
        await self._broadcast_json({"type": "shutdown"})

    async def broadcast_runtime_control(self, *, logging_mode: str) -> None:
        self.runtime_logging_mode = normalize_overlay_logging_mode(logging_mode)
        await self._broadcast_json(self._runtime_control_payload())

    async def broadcast_desktop_runtime_control(self, payload: Mapping[str, Any]) -> None:
        self._ensure_desktop_runtime_controls_enabled()
        await self._broadcast_json(self._desktop_runtime_control_message(payload))

    def set_initial_desktop_runtime_controls(
        self,
        sequence: Iterable[Mapping[str, Any]],
    ) -> None:
        self._ensure_desktop_runtime_controls_enabled()
        self._initial_desktop_runtime_controls = [dict(payload) for payload in sequence]

    def snapshot(self) -> OverlayPresentationSnapshot:
        return self._snapshot

    async def _handle_connection(self, connection: ServerConnection) -> None:
        authenticated = False
        connection_id = self._connection_id(connection)
        try:
            auth_payload = self._load_message(await connection.recv())
            if not self._is_valid_auth_payload(auth_payload):
                logger.warning("[OverlayBridge] Rejected overlay auth request")
                await connection.send(json.dumps({"type": "auth_error"}))
                return

            async with self._snapshot_lock:
                if not self._is_valid_auth_payload(auth_payload):
                    logger.warning("[OverlayBridge] Rejected overlay auth request after lock")
                    await connection.send(json.dumps({"type": "auth_error"}))
                    return
                self._token_consumed = True
                initial_message: dict[str, Any] = {
                    "type": "snapshot",
                    "payload": self._snapshot.to_dict(),
                }
                if self.desktop_runtime_controls_enabled:
                    startup_runtime_controls: list[dict[str, Any]] = []
                    if self.runtime_logging_mode is not None:
                        startup_runtime_controls.append(
                            dict(self._runtime_control_payload()["payload"])
                        )
                    startup_runtime_controls.extend(
                        dict(payload) for payload in self._initial_desktop_runtime_controls
                    )
                    initial_message["startup_runtime_controls"] = startup_runtime_controls
                await connection.send(json.dumps(initial_message))
                if (
                    not self.desktop_runtime_controls_enabled
                    and self.runtime_logging_mode is not None
                ):
                    await connection.send(json.dumps(self._runtime_control_payload()))
                self._authenticated_connections.add(connection)
                authenticated = True
                logger.info(
                    "[OverlayBridge] Overlay authenticated: overlay_instance_id=%s connection_id=%s revision=%s authenticated_connections=%s",
                    self.overlay_instance_id,
                    connection_id,
                    self._snapshot.revision,
                    len(self._authenticated_connections),
                )
                if self.diagnostics is not None:
                    self.diagnostics.record_bridge(
                        "connection_authenticated",
                        connection_id=connection_id,
                        authenticated_connections=len(self._authenticated_connections),
                        revision=self._snapshot.revision,
                    )

            async for raw_message in connection:
                message = self._load_message(raw_message)
                await self.messages.put(message)
        except ConnectionClosed as exc:
            close_code = self._close_code(exc)
            close_reason = self._close_reason(exc)
            logger.info(
                "[OverlayBridge] Overlay connection closed: overlay_instance_id=%s connection_id=%s code=%s reason=%s authenticated=%s authenticated_connections=%s last_snapshot_revision=%s",
                self.overlay_instance_id,
                connection_id,
                close_code,
                close_reason,
                authenticated,
                len(self._authenticated_connections),
                self._last_snapshot_revision,
            )
            if self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "connection_closed",
                    connection_id=connection_id,
                    authenticated=authenticated,
                    authenticated_connections=len(self._authenticated_connections),
                    code=close_code,
                    reason=close_reason,
                    last_snapshot_revision=self._last_snapshot_revision,
                )
            return
        finally:
            if authenticated:
                self._authenticated_connections.discard(connection)
                if self.diagnostics is not None:
                    self.diagnostics.record_bridge(
                        "connection_detached",
                        connection_id=connection_id,
                        authenticated_connections=len(self._authenticated_connections),
                        last_snapshot_revision=self._last_snapshot_revision,
                    )
            with contextlib.suppress(ConnectionClosed):
                await connection.close()

    def _is_valid_auth_payload(self, payload: dict[str, Any]) -> bool:
        return (
            payload.get("type") == "auth"
            and payload.get("session_token") == self.session_token
            and not self._token_consumed
        )

    def _load_message(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, str):
            raise ValueError("overlay bridge payload must be text JSON")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("overlay bridge payload must decode to an object")
        return data

    async def _run_heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval_ms / 1000.0)
                await self._broadcast_json({"type": "heartbeat"})
        except asyncio.CancelledError:
            raise

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
    ) -> asyncio.Task[Any]:
        if self.task_factory is not None:
            return self.task_factory(coroutine, task_name=task_name)
        return asyncio.create_task(coroutine, name=f"OverlayBridge:{task_name}")

    async def _broadcast_json(self, payload: dict[str, Any]) -> None:
        payload_type = str(payload.get("type"))
        revision: int | None = None
        block_update_ids: list[str] = []
        if payload_type == "snapshot":
            revision = payload.get("payload", {}).get("revision")  # type: ignore[assignment]
            block_update_ids = self._snapshot_block_update_ids(payload)
        if not self._authenticated_connections:
            if payload_type == "snapshot" and self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "snapshot_stored_unsent",
                    revision=revision,
                    authenticated_connections=0,
                )
            return

        message = json.dumps(payload)
        start_time = time.perf_counter()
        if payload_type == "snapshot" and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "send_start",
                revision=revision,
                authenticated_connections=len(self._authenticated_connections),
            )
        self._log_broadcast_marker(
            stage="start",
            payload_type=payload_type,
            revision=revision,
            block_update_ids=block_update_ids,
            authenticated_connections=len(self._authenticated_connections),
        )
        stale_connections: list[ServerConnection] = []
        for connection in tuple(self._authenticated_connections):
            try:
                await connection.send(message)
            except Exception as exc:
                stale_connections.append(connection)
                logger.warning(
                    "[OverlayBridge] Broadcast send failed: overlay_instance_id=%s connection_id=%s revision=%s exception_type=%s",
                    self.overlay_instance_id,
                    self._connection_id(connection),
                    revision,
                    type(exc).__name__,
                )
                if self.diagnostics is not None:
                    self.diagnostics.record_bridge(
                        "send_failure",
                        connection_id=self._connection_id(connection),
                        revision=revision,
                        exception_type=type(exc).__name__,
                        removed=True,
                    )

        for connection in stale_connections:
            self._authenticated_connections.discard(connection)

        elapsed_ms = max(0, int((time.perf_counter() - start_time) * 1000))
        if payload_type == "snapshot" and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "send_finish",
                revision=revision,
                authenticated_connections=len(self._authenticated_connections),
                stale_connections=len(stale_connections),
                elapsed_ms=elapsed_ms,
            )
        self._log_broadcast_marker(
            stage="finish",
            payload_type=payload_type,
            revision=revision,
            block_update_ids=block_update_ids,
            authenticated_connections=len(self._authenticated_connections),
            stale_connections=len(stale_connections),
            elapsed_ms=elapsed_ms,
        )

    def _snapshot_block_update_ids(self, payload: dict[str, Any]) -> list[str]:
        snapshot_payload = payload.get("payload")
        if not isinstance(snapshot_payload, dict):
            return []
        raw_blocks = snapshot_payload.get("blocks")
        if not isinstance(raw_blocks, list):
            return []
        update_ids: list[str] = []
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue
            update_id = block.get("update_id")
            if isinstance(update_id, str) and update_id:
                update_ids.append(update_id)
        return update_ids

    def _should_log_detailed_broadcast(self, payload_type: str) -> bool:
        return (
            payload_type == "snapshot"
            and normalize_overlay_logging_mode(self.runtime_logging_mode or "basic") == "detailed"
        )

    def _log_broadcast_marker(
        self,
        *,
        stage: str,
        payload_type: str,
        revision: int | None,
        block_update_ids: list[str],
        authenticated_connections: int,
        stale_connections: int | None = None,
        elapsed_ms: int | None = None,
    ) -> None:
        if not self._should_log_detailed_broadcast(payload_type):
            return
        parts = [
            "[OverlayBridge][Broadcast]",
            f"stage={stage}",
            f"overlay_instance_id={self.overlay_instance_id}",
            f"type={payload_type}",
            f"revision={revision}",
            f"authenticated_connections={authenticated_connections}",
            f"block_update_ids={block_update_ids}",
        ]
        if stale_connections is not None:
            parts.append(f"stale_connections={stale_connections}")
        if elapsed_ms is not None:
            parts.append(f"elapsed_ms={elapsed_ms}")
        logger.info(" ".join(parts))

    def _connection_id(self, connection: ServerConnection) -> str:
        return f"conn-{id(connection):x}"

    def _close_code(self, exc: ConnectionClosed) -> int | None:
        if exc.rcvd is not None:
            return exc.rcvd.code
        if exc.sent is not None:
            return exc.sent.code
        return None

    def _close_reason(self, exc: ConnectionClosed) -> str | None:
        if exc.rcvd is not None:
            return exc.rcvd.reason
        if exc.sent is not None:
            return exc.sent.reason
        return None

    def _drain_messages(self) -> None:
        while True:
            try:
                self.messages.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _runtime_control_payload(self) -> dict[str, Any]:
        return {
            "type": "runtime_control",
            "payload": {
                "logging_mode": normalize_overlay_logging_mode(self.runtime_logging_mode or "basic")
            },
        }

    def _desktop_runtime_control_message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "runtime_control",
            "payload": dict(payload),
        }

    def _ensure_desktop_runtime_controls_enabled(self) -> None:
        if not self.desktop_runtime_controls_enabled:
            raise RuntimeError("desktop runtime controls are only enabled for desktop overlays")
