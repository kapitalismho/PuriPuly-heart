from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection
from websockets.exceptions import ConnectionClosed

from puripuly_heart.core.clock import Clock, SystemClock

from .diagnostics import OverlayDiagnosticsRecorder
from .manifest import (
    OVERLAY_CONTRACT_VERSION,
    OVERLAY_EXECUTION_CONTRACT,
    OVERLAY_NATIVE_RETRY_CONTRACT,
    normalize_overlay_logging_mode,
)
from .protocol import NativeFreshRenderTargets, NativeQuietTailEpisodes, OverlayPresentationSnapshot

logger = logging.getLogger(__name__)

_SCENE_BYTE_LIMIT = 1024 * 1024
_CONTROL_BYTE_LIMIT = 4 * 1024
_CONTROL_SLOT_LIMIT = 8
_REVERSE_DIAGNOSTIC_LIMIT = 128
_REVERSE_DIAGNOSTIC_BYTE_LIMIT = 4 * 1024
_WRITE_LIMIT = 64 * 1024
_SCENE_WRITE_TIMEOUT_SECONDS = 5.0
_CONTROL_WRITE_TIMEOUT_SECONDS = 1.0
_CLOSE_TIMEOUT_SECONDS = 1.0
_DELIVERY_RECEIPT_LIMIT = 128
_REVERSE_CONTROL_TYPES = {
    "runtime_error",
    "overlay_ready",
    "shutdown_ack",
    "window_bounds_changed",
    "interaction_mode_changed",
    "reset_to_bottom_center",
}


class _BoundedReverseMessageQueue:
    def __init__(self) -> None:
        self._controls: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._diagnostics: deque[dict[str, Any]] = deque(maxlen=_REVERSE_DIAGNOSTIC_LIMIT)
        self._available = asyncio.Event()
        self._space_available = asyncio.Event()
        self._space_available.set()
        self.dropped_diagnostics = 0

    async def put(self, message: dict[str, Any]) -> None:
        payload_size = len(
            json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        message_type = str(message.get("type", ""))
        if message_type not in _REVERSE_CONTROL_TYPES:
            self.put_nowait(message)
            return
        if payload_size > _CONTROL_BYTE_LIMIT:
            raise ValueError("overlay reverse control exceeds maximum size")
        while message_type not in self._controls and len(self._controls) >= _CONTROL_SLOT_LIMIT:
            self._space_available.clear()
            if len(self._controls) < _CONTROL_SLOT_LIMIT:
                self._space_available.set()
                continue
            await self._space_available.wait()
        self._controls.pop(message_type, None)
        self._controls[message_type] = message
        self._available.set()

    def put_nowait(self, message: dict[str, Any]) -> None:
        payload_size = len(
            json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        message_type = str(message.get("type", ""))
        if message_type in _REVERSE_CONTROL_TYPES:
            if payload_size > _CONTROL_BYTE_LIMIT:
                raise asyncio.QueueFull
            if message_type not in self._controls and len(self._controls) >= _CONTROL_SLOT_LIMIT:
                raise asyncio.QueueFull
            self._controls.pop(message_type, None)
            self._controls[message_type] = message
        else:
            if payload_size > _REVERSE_DIAGNOSTIC_BYTE_LIMIT:
                self.dropped_diagnostics += 1
                return
            if len(self._diagnostics) >= _REVERSE_DIAGNOSTIC_LIMIT:
                self.dropped_diagnostics += 1
            self._diagnostics.append(message)
        self._available.set()

    async def get(self) -> dict[str, Any]:
        while True:
            try:
                return self.get_nowait()
            except asyncio.QueueEmpty:
                self._available.clear()
                if not self.empty():
                    self._available.set()
                    continue
                await self._available.wait()

    def get_nowait(self) -> dict[str, Any]:
        if self._controls:
            _, message = self._controls.popitem(last=False)
            self._space_available.set()
        elif self._diagnostics:
            message = self._diagnostics.popleft()
        else:
            raise asyncio.QueueEmpty
        if self.empty():
            self._available.clear()
        return message

    def empty(self) -> bool:
        return not self._controls and not self._diagnostics

    def qsize(self) -> int:
        return len(self._controls) + len(self._diagnostics)


@dataclass(frozen=True, slots=True)
class OverlayDeliveryReceipt:
    stage: str
    outcome: str
    scene_revision: int | None
    connection_epoch: int | None
    cause: str | None
    observed_at: float


@dataclass(frozen=True, slots=True)
class _SceneEnvelope:
    snapshot: OverlayPresentationSnapshot
    message: str
    block_expirations: Mapping[str, float | None]
    admitted_at: float


@dataclass(frozen=True, slots=True)
class _ControlEnvelope:
    key: str
    message: str
    payload_type: str


@dataclass(slots=True)
class OverlayBridge:
    session_token: str
    initial_snapshot: dict[str, object] | OverlayPresentationSnapshot | None = None
    heartbeat_interval_ms: int = 1000
    host: str = "127.0.0.1"
    port: int = 0
    overlay_instance_id: str | None = None
    runtime_generation: int = 1
    diagnostics: OverlayDiagnosticsRecorder | None = None
    runtime_logging_mode: str | None = None
    desktop_runtime_controls_enabled: bool = False
    task_factory: Any | None = None
    clock: Clock = field(default_factory=SystemClock)

    url: str = field(init=False, default="")
    messages: _BoundedReverseMessageQueue = field(
        init=False,
        default_factory=_BoundedReverseMessageQueue,
    )
    _server: Server | None = field(init=False, default=None)
    _heartbeat_task: asyncio.Task[None] | None = field(init=False, default=None)
    _writer_task: asyncio.Task[None] | None = field(init=False, default=None)
    _writer_wakeup: asyncio.Event = field(init=False, default_factory=asyncio.Event)
    _authenticated_connections: set[ServerConnection] = field(
        init=False,
        default_factory=set,
    )
    _connection_epochs: dict[ServerConnection, int] = field(init=False, default_factory=dict)
    _connection_epoch: int = field(init=False, default=0)
    _snapshot: OverlayPresentationSnapshot = field(init=False)
    _current_scene: _SceneEnvelope = field(init=False)
    _pending_scene: _SceneEnvelope | None = field(init=False, default=None)
    _active_scene: _SceneEnvelope | None = field(init=False, default=None)
    _pending_controls: OrderedDict[str, _ControlEnvelope] = field(
        init=False,
        default_factory=OrderedDict,
    )
    _replay_required: bool = field(init=False, default=False)
    _token_consumed: bool = field(init=False, default=False)
    _last_snapshot_revision: int = field(init=False, default=0)
    _initial_desktop_runtime_controls: list[dict[str, Any]] = field(
        init=False,
        default_factory=list,
    )
    _delivery_receipts: deque[OverlayDeliveryReceipt] = field(
        init=False,
        default_factory=lambda: deque(maxlen=_DELIVERY_RECEIPT_LIMIT),
    )
    _unresolved_transport_tasks: set[asyncio.Task[Any]] = field(
        init=False,
        default_factory=set,
    )
    _unresolved_connections: set[ServerConnection] = field(
        init=False,
        default_factory=set,
    )
    _unresolved_tasks_by_connection: dict[ServerConnection, set[asyncio.Task[Any]]] = field(
        init=False,
        default_factory=dict,
    )
    _connection_close_tasks: dict[ServerConnection, asyncio.Task[Any]] = field(
        init=False,
        default_factory=dict,
    )
    _health_challenges: OrderedDict[int, float] = field(
        init=False,
        default_factory=OrderedDict,
    )
    _next_health_challenge_id: int = field(init=False, default=1)
    _next_health_challenge_at: float = field(init=False, default=0.0)
    _owner_health_deadline: float | None = field(init=False, default=None)
    _native_acceptance_revision: int | None = field(init=False, default=None)
    _native_acceptance_deadline: float | None = field(init=False, default=None)
    _health_failure_reported: bool = field(init=False, default=False)
    _retirement_task: asyncio.Task[None] | None = field(init=False, default=None)
    _stop_task: asyncio.Task[None] | None = field(init=False, default=None)
    _stopped: bool = field(init=False, default=False)
    _stopping: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.initial_snapshot is None:
            self._snapshot = OverlayPresentationSnapshot()
        elif isinstance(self.initial_snapshot, OverlayPresentationSnapshot):
            self._snapshot = self.initial_snapshot
        else:
            self._snapshot = OverlayPresentationSnapshot.from_dict(self.initial_snapshot)
        self._last_snapshot_revision = self._snapshot.revision
        self._current_scene = self._make_scene_envelope(self._snapshot, {})

    @property
    def delivery_receipts(self) -> tuple[OverlayDeliveryReceipt, ...]:
        return tuple(self._delivery_receipts)

    @property
    def retained_scene_bytes(self) -> int:
        envelopes = {id(self._current_scene): self._current_scene}
        if self._active_scene is not None:
            envelopes[id(self._active_scene)] = self._active_scene
        if self._pending_scene is not None:
            envelopes[id(self._pending_scene)] = self._pending_scene
        return sum(len(envelope.message.encode("utf-8")) for envelope in envelopes.values())

    async def start(self) -> None:
        if self._server is not None:
            return
        if self._unresolved_connections:
            raise RuntimeError(
                "OverlayBridge cannot start while transport termination is unresolved"
            )
        if self._stop_task is not None and not self._stop_task.done():
            raise RuntimeError("OverlayBridge cannot start while stop is in progress")
        self._stop_task = None
        self._stopped = False
        self._stopping = False
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=None,
            compression=None,
            max_size=_SCENE_BYTE_LIMIT,
            write_limit=_WRITE_LIMIT,
            close_timeout=_CLOSE_TIMEOUT_SECONDS,
        )
        socket = self._server.sockets[0]
        bound_host, bound_port = socket.getsockname()[:2]
        self.url = f"ws://{bound_host}:{bound_port}"
        self._ensure_writer()
        self._heartbeat_task = self._create_task(
            self._run_heartbeat_loop(),
            task_name="bridge-heartbeat",
        )

    async def stop(self) -> None:
        self._stopping = True
        cleanup_task = self._stop_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(self._stop_owned(), name="OverlayBridge:stop")
            self._stop_task = cleanup_task
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup_task)
            raise

    async def _stop_owned(self) -> None:
        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

        retirement_task = self._retirement_task
        self._retirement_task = None
        if retirement_task is not None and not retirement_task.done():
            retirement_task.cancel()
            await asyncio.gather(retirement_task, return_exceptions=True)

        writer_task = self._writer_task
        self._writer_task = None
        if writer_task is not None:
            writer_task.cancel()
            await asyncio.gather(writer_task, return_exceptions=True)

        failures: list[Exception] = []
        connections = tuple(self._authenticated_connections | self._unresolved_connections)
        for connection in connections:
            failure = await self._bounded_close_connection(connection)
            if failure is not None:
                failures.append(failure)
        self._authenticated_connections.clear()
        self._connection_epochs.clear()

        server = self._server
        if server is not None:
            server.close()
            close_wait_task = asyncio.create_task(
                server.wait_closed(),
                name="OverlayBridge:server-wait-closed",
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(close_wait_task),
                    timeout=_CLOSE_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                self._retain_unresolved_task(close_wait_task)
                failures.append(exc)
            except Exception as exc:
                failures.append(exc)
            self._server = None

        pending_scene = self._pending_scene
        if pending_scene is not None:
            self._record_delivery(
                outcome="delivery_rejected",
                scene_revision=pending_scene.snapshot.revision,
                cause="bridge_stopping",
            )
        self._pending_scene = None
        self._active_scene = None
        self._pending_controls.clear()
        self._replay_required = False
        self._drain_messages()
        self._stopped = True
        self.url = ""
        failures.extend(
            RuntimeError(f"overlay transport task remains unresolved: {task.get_name()}")
            for task in tuple(self._unresolved_transport_tasks)
            if not task.done() and task not in self._connection_close_tasks.values()
        )
        if not self._unresolved_connections:
            self._token_consumed = False
        if failures:
            raise ExceptionGroup("OverlayBridge stop failed", failures)

    async def replace_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
        *,
        block_expirations: Mapping[str, float | None] | None = None,
    ) -> OverlayDeliveryReceipt:
        if self._stopping or self._stopped:
            return self._record_delivery(
                outcome="delivery_rejected",
                scene_revision=snapshot.revision,
                cause="bridge_stopped" if self._stopped else "bridge_stopping",
            )
        if snapshot.revision <= self._last_snapshot_revision:
            return self._record_delivery(
                outcome="superseded_product",
                scene_revision=snapshot.revision,
                cause="stale_revision",
            )
        self._last_snapshot_revision = snapshot.revision
        envelope = self._make_scene_envelope(snapshot, block_expirations or {})
        if len(envelope.message.encode("utf-8")) > _SCENE_BYTE_LIMIT:
            safety_snapshot = replace(snapshot, blocks=[])
            envelope = self._make_scene_envelope(safety_snapshot, {})
            self._snapshot = safety_snapshot
            self._current_scene = envelope
            self._pending_scene = envelope
            receipt = self._record_delivery(
                outcome="delivery_rejected",
                scene_revision=snapshot.revision,
                cause="scene_payload_exhausted",
            )
        else:
            self._snapshot = snapshot
            self._current_scene = envelope
            if self._pending_scene is not None:
                self._record_delivery(
                    outcome="superseded_product",
                    scene_revision=self._pending_scene.snapshot.revision,
                    cause="newer_scene",
                )
            self._pending_scene = envelope
            receipt = self._record_delivery(
                outcome="admitted",
                scene_revision=snapshot.revision,
                cause=None,
            )
        self._ensure_writer()
        self._writer_wakeup.set()
        if not self._authenticated_connections and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "snapshot_stored_unsent",
                revision=snapshot.revision,
                authenticated_connections=0,
            )
        return receipt

    async def broadcast_shutdown(self) -> None:
        self._enqueue_control("shutdown", {"type": "shutdown"}, terminal=True)

    async def broadcast_runtime_control(self, *, logging_mode: str) -> None:
        normalized_mode = normalize_overlay_logging_mode(logging_mode)
        self._enqueue_control(
            "runtime_control",
            self._runtime_control_payload(normalized_mode),
        )
        self.runtime_logging_mode = normalized_mode

    async def broadcast_desktop_runtime_control(self, payload: Mapping[str, Any]) -> None:
        self._ensure_desktop_runtime_controls_enabled()
        message = self._desktop_runtime_control_message(payload)
        key = f"runtime_control:{payload.get('command', '')}"
        self._enqueue_control(key, message)

    def set_initial_desktop_runtime_controls(
        self,
        sequence: Iterable[Mapping[str, Any]],
    ) -> None:
        self._ensure_desktop_runtime_controls_enabled()
        self._initial_desktop_runtime_controls = [dict(payload) for payload in sequence]
        self._current_scene = self._make_scene_envelope(
            self._current_scene.snapshot,
            self._current_scene.block_expirations,
        )

    def snapshot(self) -> OverlayPresentationSnapshot:
        return self._snapshot

    async def _handle_connection(self, connection: ServerConnection) -> None:
        if self._stopping or self._stopped or self._unresolved_connections:
            self._abort_connection(connection)
            return
        authenticated = False
        connection_id = self._connection_id(connection)
        epoch: int | None = None
        try:
            auth_payload = self._load_message(await connection.recv())
            if not self._is_valid_auth_payload(auth_payload):
                logger.warning("[OverlayBridge] Rejected overlay auth request")
                await asyncio.wait_for(
                    connection.send(json.dumps({"type": "auth_error"})),
                    timeout=_CONTROL_WRITE_TIMEOUT_SECONDS,
                )
                return
            self._token_consumed = True
            self._connection_epoch += 1
            epoch = self._connection_epoch
            self._connection_epochs[connection] = epoch
            self._current_scene = self._make_scene_envelope(
                self._current_scene.snapshot,
                self._current_scene.block_expirations,
            )
            now = self.clock.now()
            if self.overlay_instance_id is not None and not self.desktop_runtime_controls_enabled:
                self._next_health_challenge_at = now
                self._owner_health_deadline = now + 3.0
            self._health_failure_reported = False
            self._authenticated_connections.add(connection)
            self._replay_required = True
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
            self._ensure_writer()
            self._writer_wakeup.set()
            if not self.desktop_runtime_controls_enabled and self.runtime_logging_mode is not None:
                self._enqueue_control("runtime_control", self._runtime_control_payload())
            await asyncio.sleep(0)
            async for raw_message in connection:
                message = self._load_message(raw_message)
                try:
                    if message.get("type") == "validity_challenge":
                        self._handle_validity_challenge(message)
                        continue
                    if message.get("type") == "owner_status":
                        if self._handle_owner_status(message):
                            await self.messages.put(message)
                        continue
                    await self.messages.put(message)
                except ValueError:
                    await self._retire_connection(
                        connection,
                        epoch,
                        cause="reverse_control_oversized",
                    )
                    return
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
        finally:
            if authenticated and self._connection_epochs.get(connection) == epoch:
                self._authenticated_connections.discard(connection)
                self._connection_epochs.pop(connection, None)
                if not self._unresolved_connections:
                    self._token_consumed = False
                if self.diagnostics is not None:
                    self.diagnostics.record_bridge(
                        "connection_detached",
                        connection_id=connection_id,
                        authenticated_connections=len(self._authenticated_connections),
                        last_snapshot_revision=self._last_snapshot_revision,
                    )
                await self._bounded_close_connection(connection)

    def _is_valid_auth_payload(self, payload: dict[str, Any]) -> bool:
        if (
            payload.get("type") != "auth"
            or payload.get("session_token") != self.session_token
            or self._token_consumed
            or self._stopping
        ):
            return False
        if self.overlay_instance_id is None:
            return True
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, dict):
            return False
        if capabilities.get("execution_contract") != OVERLAY_EXECUTION_CONTRACT:
            return False
        if not self.desktop_runtime_controls_enabled and (
            capabilities.get("native_presentation_retry") != OVERLAY_NATIVE_RETRY_CONTRACT
        ):
            return False
        return (
            payload.get("contract_version") == OVERLAY_CONTRACT_VERSION
            and payload.get("overlay_instance_id") == self.overlay_instance_id
            and payload.get("runtime_generation") == self.runtime_generation
        )

    def _load_message(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, str):
            raise ValueError("overlay bridge payload must be text JSON")
        if len(payload.encode("utf-8")) > _SCENE_BYTE_LIMIT:
            raise ValueError("overlay bridge payload exceeds maximum size")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("overlay bridge payload must decode to an object")
        return data

    async def _run_heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.25)
                connection = self._current_connection()
                if connection is None:
                    continue
                now = self.clock.now()
                if self.overlay_instance_id is None or self.desktop_runtime_controls_enabled:
                    self._enqueue_control("heartbeat", {"type": "heartbeat"})
                    continue
                if now >= self._next_health_challenge_at:
                    challenge_id = self._next_health_challenge_id
                    self._next_health_challenge_id += 1
                    self._record_health_challenge(challenge_id, now)
                    self._enqueue_control(
                        "health_challenge",
                        {
                            "type": "health_challenge",
                            "challenge_id": challenge_id,
                            "overlay_instance_id": self.overlay_instance_id,
                            "runtime_generation": self.runtime_generation,
                        },
                    )
                    self._next_health_challenge_at = now + 1.0
                cause = None
                if (
                    self._native_acceptance_deadline is not None
                    and now >= self._native_acceptance_deadline
                ):
                    cause = "native_acceptance_timeout"
                elif self._owner_health_deadline is not None and now >= self._owner_health_deadline:
                    cause = "native_owner_unresponsive"
                if cause is not None and not self._health_failure_reported:
                    self._health_failure_reported = True
                    self.messages.put_nowait({"type": "runtime_error", "failure_reason": cause})
        except asyncio.CancelledError:
            raise

    def _record_health_challenge(self, challenge_id: int, issued_at: float) -> None:
        self._health_challenges[challenge_id] = issued_at
        while len(self._health_challenges) > 4:
            self._health_challenges.popitem(last=False)

    def _handle_validity_challenge(self, message: Mapping[str, Any]) -> None:
        challenge_id = message.get("challenge_id")
        if (
            not isinstance(challenge_id, int)
            or isinstance(challenge_id, bool)
            or message.get("overlay_instance_id") != self.overlay_instance_id
            or message.get("runtime_generation") != self.runtime_generation
        ):
            raise ValueError("invalid validity challenge")
        now = self.clock.now()
        blocks = []
        for block in self._current_scene.snapshot.blocks:
            expiration = self._current_scene.block_expirations.get(block.id)
            remaining_s = 3.0 if expiration is None else min(3.0, max(0.0, expiration - now))
            blocks.append(
                {
                    "id": block.id,
                    "occupant_key": block.occupant_key,
                    "remaining_s": remaining_s,
                }
            )
        self._enqueue_control(
            "validity_response",
            {
                "type": "validity_response",
                "challenge_id": challenge_id,
                "scene_revision": self._current_scene.snapshot.revision,
                "overlay_instance_id": self.overlay_instance_id,
                "runtime_generation": self.runtime_generation,
                "blocks": blocks,
            },
        )

    def _handle_owner_status(self, message: Mapping[str, Any]) -> bool:
        if (
            message.get("overlay_instance_id") != self.overlay_instance_id
            or message.get("runtime_generation") != self.runtime_generation
        ):
            raise ValueError("invalid owner status identity")
        challenge_id = message.get("health_challenge_id")
        now = self.clock.now()
        valid_response = False
        if isinstance(challenge_id, int) and not isinstance(challenge_id, bool):
            issued_at = self._health_challenges.pop(challenge_id, None)
            if issued_at is not None and now <= issued_at + 3.0:
                valid_response = True
                self._owner_health_deadline = issued_at + 3.0
                for prior in tuple(self._health_challenges):
                    if prior <= challenge_id:
                        self._health_challenges.pop(prior, None)
        applied_revision = message.get("latest_applied_revision")
        if (
            valid_response
            and isinstance(applied_revision, int)
            and not isinstance(applied_revision, bool)
            and self._native_acceptance_revision is not None
            and applied_revision >= self._native_acceptance_revision
        ):
            self._native_acceptance_revision = None
            self._native_acceptance_deadline = None
        return True

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
    ) -> asyncio.Task[Any]:
        if self.task_factory is not None:
            return self.task_factory(coroutine, task_name=task_name)
        return asyncio.create_task(coroutine, name=f"OverlayBridge:{task_name}")

    def _ensure_writer(self) -> None:
        if self._stopping:
            return
        if self._writer_task is not None and not self._writer_task.done():
            return
        self._writer_task = self._create_task(
            self._run_writer(),
            task_name="writer",
        )

    async def _run_writer(self) -> None:
        try:
            while not self._stopping:
                await self._writer_wakeup.wait()
                self._writer_wakeup.clear()
                while not self._stopping:
                    connection = self._current_connection()
                    if connection is None:
                        break
                    epoch = self._connection_epochs.get(connection)
                    if epoch is None:
                        break
                    control = self._take_control()
                    if control is not None:
                        written = await self._write_message(
                            connection,
                            epoch,
                            control.message,
                            timeout=_CONTROL_WRITE_TIMEOUT_SECONDS,
                            scene_revision=None,
                            payload_type=control.payload_type,
                        )
                        if not written:
                            break
                        continue
                    scene = self._take_scene_for_write()
                    if scene is None:
                        break
                    self._active_scene = scene
                    try:
                        current = self._current_scene
                        if scene.snapshot.revision < current.snapshot.revision:
                            self._record_delivery(
                                outcome="superseded_product",
                                scene_revision=scene.snapshot.revision,
                                cause="newer_scene",
                                connection_epoch=epoch,
                            )
                            continue
                        message = self._revalidated_scene_message(scene)
                        await self._write_message(
                            connection,
                            epoch,
                            message,
                            timeout=_SCENE_WRITE_TIMEOUT_SECONDS,
                            scene_revision=scene.snapshot.revision,
                            payload_type="snapshot",
                        )
                    finally:
                        self._active_scene = None
        except asyncio.CancelledError:
            active = self._active_scene
            if active is not None:
                self._record_delivery(
                    outcome="ambiguous",
                    scene_revision=active.snapshot.revision,
                    cause="writer_cancelled",
                )
            raise

    def _current_connection(self) -> ServerConnection | None:
        if not self._authenticated_connections:
            return None
        connection = next(iter(self._authenticated_connections))
        if connection not in self._connection_epochs:
            self._connection_epoch += 1
            self._connection_epochs[connection] = self._connection_epoch
        return connection

    def _take_control(self) -> _ControlEnvelope | None:
        shutdown = self._pending_controls.pop("shutdown", None)
        if shutdown is not None:
            return shutdown
        if not self._pending_controls:
            return None
        _, control = self._pending_controls.popitem(last=False)
        return control

    def _take_scene_for_write(self) -> _SceneEnvelope | None:
        if self._replay_required:
            self._replay_required = False
            self._pending_scene = None
            return self._current_scene
        scene = self._pending_scene
        self._pending_scene = None
        return scene

    async def _write_message(
        self,
        connection: ServerConnection,
        epoch: int,
        message: str,
        *,
        timeout: float,
        scene_revision: int | None,
        payload_type: str,
    ) -> bool:
        if self._connection_epochs.get(connection) != epoch:
            return False
        block_update_ids: list[str] = []
        if payload_type == "snapshot":
            block_update_ids = self._snapshot_block_update_ids(json.loads(message))
        start_time = time.perf_counter()
        if payload_type == "snapshot" and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "send_start",
                revision=scene_revision,
                authenticated_connections=len(self._authenticated_connections),
            )
        self._log_broadcast_marker(
            stage="start",
            payload_type=payload_type,
            revision=scene_revision,
            block_update_ids=block_update_ids,
            authenticated_connections=len(self._authenticated_connections),
        )
        send_task = asyncio.create_task(connection.send(message), name="OverlayBridge:send")
        try:
            await asyncio.wait_for(asyncio.shield(send_task), timeout=timeout)
        except asyncio.CancelledError:
            self._retain_unresolved_task(send_task, connection=connection)
            if scene_revision is not None:
                self._record_delivery(
                    outcome="ambiguous",
                    scene_revision=scene_revision,
                    cause="write_cancelled",
                    connection_epoch=epoch,
                )
            await self._retire_connection(connection, epoch, cause="write_cancelled")
            raise
        except TimeoutError:
            self._retain_unresolved_task(send_task, connection=connection)
            if scene_revision is not None:
                self._record_delivery(
                    outcome="ambiguous",
                    scene_revision=scene_revision,
                    cause="write_timeout",
                    connection_epoch=epoch,
                )
            await self._retire_connection(connection, epoch, cause="write_timeout")
            return False
        except Exception as exc:
            if scene_revision is not None:
                self._record_delivery(
                    outcome="ambiguous",
                    scene_revision=scene_revision,
                    cause=type(exc).__name__,
                    connection_epoch=epoch,
                )
            if self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "send_failure",
                    connection_id=self._connection_id(connection),
                    revision=scene_revision,
                    exception_type=type(exc).__name__,
                    removed=True,
                )
            await self._retire_connection(connection, epoch, cause=type(exc).__name__)
            elapsed_ms = max(0, int((time.perf_counter() - start_time) * 1000))
            if payload_type == "snapshot" and self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "send_finish",
                    revision=scene_revision,
                    authenticated_connections=len(self._authenticated_connections),
                    stale_connections=1,
                    elapsed_ms=elapsed_ms,
                )
            self._log_broadcast_marker(
                stage="finish",
                payload_type=payload_type,
                revision=scene_revision,
                block_update_ids=block_update_ids,
                authenticated_connections=len(self._authenticated_connections),
                stale_connections=1,
                elapsed_ms=elapsed_ms,
            )
            return False

        if scene_revision is not None:
            self._record_delivery(
                outcome="written",
                scene_revision=scene_revision,
                cause=None,
                connection_epoch=epoch,
            )
            if not self.desktop_runtime_controls_enabled:
                self._native_acceptance_revision = max(
                    scene_revision,
                    self._native_acceptance_revision or scene_revision,
                )
                if self._native_acceptance_deadline is None:
                    self._native_acceptance_deadline = self.clock.now() + 2.0
        elapsed_ms = max(0, int((time.perf_counter() - start_time) * 1000))
        if payload_type == "snapshot" and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "send_finish",
                revision=scene_revision,
                authenticated_connections=len(self._authenticated_connections),
                stale_connections=0,
                elapsed_ms=elapsed_ms,
            )
        self._log_broadcast_marker(
            stage="finish",
            payload_type=payload_type,
            revision=scene_revision,
            block_update_ids=block_update_ids,
            authenticated_connections=len(self._authenticated_connections),
            stale_connections=0,
            elapsed_ms=elapsed_ms,
        )
        return True

    async def _retire_connection(
        self,
        connection: ServerConnection,
        epoch: int | None,
        *,
        cause: str,
    ) -> None:
        if epoch is not None and self._connection_epochs.get(connection) != epoch:
            return
        self._authenticated_connections.discard(connection)
        self._connection_epochs.pop(connection, None)
        self._replay_required = True
        failure = await self._bounded_close_connection(connection)
        if failure is not None:
            self._unresolved_connections.add(connection)
            self._token_consumed = True
        elif not self._unresolved_connections:
            self._token_consumed = False
        if self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "connection_retired",
                connection_id=self._connection_id(connection),
                cause=cause,
                last_snapshot_revision=self._last_snapshot_revision,
            )

    async def _bounded_close_connection(
        self,
        connection: ServerConnection,
    ) -> Exception | None:
        close_task = self._connection_close_tasks.get(connection)
        if close_task is None:
            close_task = asyncio.create_task(
                connection.close(),
                name="OverlayBridge:close-connection",
            )
            self._connection_close_tasks[connection] = close_task
        elif not close_task.done() and connection in self._unresolved_connections:
            return TimeoutError("overlay connection close remains unresolved")
        try:
            await asyncio.wait_for(
                asyncio.shield(close_task),
                timeout=_CLOSE_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            self._abort_connection(connection)
            self._unresolved_connections.add(connection)
            self._retain_unresolved_task(close_task, connection=connection)
            return exc
        except Exception as exc:
            self._abort_connection(connection)
            self._unresolved_connections.add(connection)
            return exc
        self._connection_close_tasks.pop(connection, None)
        if not self._unresolved_tasks_by_connection.get(connection):
            self._unresolved_connections.discard(connection)
        return None

    def _abort_connection(self, connection: ServerConnection) -> None:
        transport = getattr(connection, "transport", None)
        abort = getattr(transport, "abort", None)
        if callable(abort):
            with contextlib.suppress(Exception):
                abort()

    def _retain_unresolved_task(
        self,
        task: asyncio.Task[Any],
        *,
        connection: ServerConnection | None = None,
    ) -> None:
        if task.done():
            return
        self._unresolved_transport_tasks.add(task)
        if connection is not None:
            self._unresolved_connections.add(connection)
            self._unresolved_tasks_by_connection.setdefault(connection, set()).add(task)
        task.add_done_callback(
            lambda completed: self._resolve_unresolved_task(connection, completed)
        )

    def _resolve_unresolved_task(
        self,
        connection: ServerConnection | None,
        task: asyncio.Task[Any],
    ) -> None:
        self._unresolved_transport_tasks.discard(task)
        if connection is None:
            return
        tasks = self._unresolved_tasks_by_connection.get(connection)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                self._unresolved_tasks_by_connection.pop(connection, None)
                self._unresolved_connections.discard(connection)
                self._connection_close_tasks.pop(connection, None)
                if not self._authenticated_connections and not self._unresolved_connections:
                    self._token_consumed = False

    def _request_connection_retirement(self, *, cause: str) -> None:
        task = self._retirement_task
        if task is not None and not task.done():
            return
        connection = self._current_connection()
        if connection is None:
            return
        epoch = self._connection_epochs.get(connection)
        task = self._create_task(
            self._retire_connection(connection, epoch, cause=cause),
            task_name="connection-retirement",
        )
        self._retirement_task = task
        task.add_done_callback(
            lambda completed: (
                setattr(self, "_retirement_task", None)
                if self._retirement_task is completed
                else None
            )
        )

    def _enqueue_control(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        terminal: bool = False,
    ) -> None:
        if self._stopping or self._stopped:
            raise RuntimeError("OverlayBridge is not accepting controls")
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(message.encode("utf-8")) > _CONTROL_BYTE_LIMIT:
            raise ValueError("overlay control exceeds maximum size")
        if terminal and key not in self._pending_controls:
            while len(self._pending_controls) >= _CONTROL_SLOT_LIMIT:
                evicted_key = next(
                    (candidate for candidate in self._pending_controls if candidate != "shutdown"),
                    None,
                )
                if evicted_key is None:
                    break
                self._pending_controls.pop(evicted_key)
        nonterminal_limit = _CONTROL_SLOT_LIMIT - int("shutdown" in self._pending_controls)
        if (
            key not in self._pending_controls
            and not terminal
            and len(self._pending_controls) >= nonterminal_limit
        ):
            self._request_connection_retirement(cause="control_overflow")
            raise RuntimeError("overlay control capacity exhausted")
        self._pending_controls.pop(key, None)
        self._pending_controls[key] = _ControlEnvelope(
            key=key,
            message=message,
            payload_type=str(payload.get("type", key)),
        )
        self._ensure_writer()
        self._writer_wakeup.set()

    def _make_scene_envelope(
        self,
        snapshot: OverlayPresentationSnapshot,
        block_expirations: Mapping[str, float | None],
    ) -> _SceneEnvelope:
        payload: dict[str, Any] = {
            "type": "snapshot",
            "payload": snapshot.to_dict(),
        }
        if self.desktop_runtime_controls_enabled:
            startup_runtime_controls: list[dict[str, Any]] = []
            if self.runtime_logging_mode is not None:
                startup_runtime_controls.append(dict(self._runtime_control_payload()["payload"]))
            startup_runtime_controls.extend(
                dict(control) for control in self._initial_desktop_runtime_controls
            )
            payload["startup_runtime_controls"] = startup_runtime_controls
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return _SceneEnvelope(
            snapshot=snapshot,
            message=message,
            block_expirations=dict(block_expirations),
            admitted_at=self.clock.now(),
        )

    def _revalidated_scene_message(self, envelope: _SceneEnvelope) -> str:
        now = self.clock.now()
        valid_blocks = [
            block
            for block in envelope.snapshot.blocks
            if envelope.block_expirations.get(block.id) is None
            or now < envelope.block_expirations[block.id]
        ]
        if len(valid_blocks) == len(envelope.snapshot.blocks):
            return envelope.message
        visible = {block.id for block in valid_blocks}
        targets = envelope.snapshot.native_fresh_render_targets
        self_target = targets.self if targets is not None and targets.self in visible else None
        peer_target = targets.peer if targets is not None and targets.peer in visible else None
        episodes = envelope.snapshot.native_quiet_tail_episodes
        snapshot = replace(
            envelope.snapshot,
            blocks=valid_blocks,
            native_fresh_render_targets=NativeFreshRenderTargets(
                self=self_target,
                peer=peer_target,
            ),
            native_quiet_tail_episodes=NativeQuietTailEpisodes(
                self=(episodes.self if episodes is not None and self_target is not None else None),
                peer=(episodes.peer if episodes is not None and peer_target is not None else None),
            ),
        )
        return self._make_scene_envelope(snapshot, {}).message

    def _record_delivery(
        self,
        *,
        outcome: str,
        scene_revision: int | None,
        cause: str | None,
        connection_epoch: int | None = None,
    ) -> OverlayDeliveryReceipt:
        stages = {
            "written": "transport_written",
            "ambiguous": "ambiguous_receipt",
            "superseded_product": "superseded_product",
        }
        receipt = OverlayDeliveryReceipt(
            stage=stages.get(outcome, "delivery_admitted"),
            outcome=outcome,
            scene_revision=scene_revision,
            connection_epoch=connection_epoch,
            cause=cause,
            observed_at=self.clock.now(),
        )
        self._delivery_receipts.append(receipt)
        return receipt

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

    def _runtime_control_payload(self, logging_mode: str | None = None) -> dict[str, Any]:
        return {
            "type": "runtime_control",
            "payload": {
                "logging_mode": normalize_overlay_logging_mode(
                    logging_mode or self.runtime_logging_mode or "basic"
                )
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
