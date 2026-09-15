from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection
from websockets.exceptions import ConnectionClosed

from puripuly_heart.core.clock import Clock, SystemClock

from .bridge_mailbox import OverlayBridgeMailbox, OverlayDeliveryReceipt
from .bridge_session import AuthenticatedSessionHealth
from .bridge_transport import OverlayTransportExecutor, TransportWriteCancelled
from .diagnostics import OverlayDiagnosticsRecorder
from .manifest import (
    OVERLAY_CONTRACT_VERSION,
    OVERLAY_EXECUTION_CONTRACT,
    OVERLAY_NATIVE_RETRY_CONTRACT,
)
from .protocol import OverlayPresentationSnapshot

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
    "startup_error",
    "overlay_ready",
    "desktop_first_visible",
    "shutdown_ack",
    "shutdown_complete",
    "window_bounds_changed",
    "interaction_mode_changed",
    "reset_to_bottom_center",
}
_REVERSE_CONTROL_SLOT_LIMIT = len(_REVERSE_CONTROL_TYPES)
_REVERSE_KNOWN_TYPES = frozenset(
    _REVERSE_CONTROL_TYPES
    | {
        "owner_status",
        "startup_error",
        "shutdown_complete",
        "overlay_trace",
        "overlay_event",
        "desktop_first_visible",
    }
)


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
        while (
            message_type not in self._controls
            and len(self._controls) >= _REVERSE_CONTROL_SLOT_LIMIT
        ):
            self._space_available.clear()
            if len(self._controls) < _REVERSE_CONTROL_SLOT_LIMIT:
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
            if (
                message_type not in self._controls
                and len(self._controls) >= _REVERSE_CONTROL_SLOT_LIMIT
            ):
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
    _mailbox: OverlayBridgeMailbox = field(init=False)
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
    _session_health: AuthenticatedSessionHealth = field(init=False)
    _transport_executor: OverlayTransportExecutor = field(
        init=False,
        default_factory=OverlayTransportExecutor,
    )
    _retirement_task: asyncio.Task[None] | None = field(init=False, default=None)
    _stop_task: asyncio.Task[None] | None = field(init=False, default=None)
    _stopped: bool = field(init=False, default=False)
    _stopping: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.initial_snapshot is None:
            initial_snapshot = OverlayPresentationSnapshot()
        elif isinstance(self.initial_snapshot, OverlayPresentationSnapshot):
            initial_snapshot = self.initial_snapshot
        else:
            initial_snapshot = OverlayPresentationSnapshot.from_dict(self.initial_snapshot)
        self._mailbox = OverlayBridgeMailbox(
            initial_snapshot=initial_snapshot,
            clock=self.clock,
            scene_byte_limit=_SCENE_BYTE_LIMIT,
            control_byte_limit=_CONTROL_BYTE_LIMIT,
            control_slot_limit=_CONTROL_SLOT_LIMIT,
            delivery_receipt_limit=_DELIVERY_RECEIPT_LIMIT,
        )
        self._session_health = AuthenticatedSessionHealth(
            overlay_instance_id=self.overlay_instance_id,
            runtime_generation=self.runtime_generation,
            clock=self.clock,
        )

    @property
    def delivery_receipts(self) -> tuple[OverlayDeliveryReceipt, ...]:
        return tuple(self._mailbox.delivery_receipts)

    @property
    def retained_scene_bytes(self) -> int:
        return self._mailbox.retained_scene_bytes

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
            cleanup_task = self._create_task(
                self._stop_owned(),
                task_name="stop",
                registered=False,
            )
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
            close_wait_task = self._create_task(
                server.wait_closed(),
                task_name="server-wait-closed",
                registered=False,
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

        pending_scene = self._mailbox.pending_scene
        if pending_scene is not None:
            self._mailbox.record_delivery(
                outcome="delivery_rejected",
                scene_revision=pending_scene.snapshot.revision,
                cause="bridge_stopping",
            )
        self._mailbox.pending_scene = None
        self._mailbox.active_scene = None
        self._mailbox.pending_controls.clear()
        self._mailbox.replay_required = False
        self._mailbox.startup_barrier_epoch = None
        self._drain_messages()
        self._stopped = True
        self.url = ""
        failures.extend(
            RuntimeError(f"overlay transport task remains unresolved: {task.get_name()}")
            for task in tuple(self._unresolved_transport_tasks)
            if not task.done() and task not in self._connection_close_tasks.values()
        )
        if not self._unresolved_connections:
            self._session_health.token_consumed = False
        if failures:
            raise ExceptionGroup("OverlayBridge stop failed", failures)

    async def replace_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
        *,
        block_expirations: Mapping[str, float | None] | None = None,
    ) -> OverlayDeliveryReceipt:
        if self._stopping or self._stopped:
            return self._mailbox.record_delivery(
                outcome="delivery_rejected",
                scene_revision=snapshot.revision,
                cause="bridge_stopped" if self._stopped else "bridge_stopping",
            )
        admission = self._mailbox.admit_scene(
            snapshot,
            block_expirations or {},
            startup_runtime_controls=self._startup_runtime_controls(),
        )
        if not admission.writer_required:
            return admission.receipt
        self._ensure_writer()
        self._writer_wakeup.set()
        if not self._authenticated_connections and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "snapshot_stored_unsent",
                revision=snapshot.revision,
                authenticated_connections=0,
            )
        return admission.receipt

    async def broadcast_shutdown(self) -> None:
        self._enqueue_control("shutdown", {"type": "shutdown"}, terminal=True)

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
        self._mailbox.initial_desktop_runtime_controls = [dict(payload) for payload in sequence]
        self._mailbox.rebuild_current(startup_runtime_controls=self._startup_runtime_controls())

    def snapshot(self) -> OverlayPresentationSnapshot:
        return self._mailbox.snapshot

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
            self._session_health.token_consumed = True
            self._connection_epoch += 1
            epoch = self._connection_epoch
            self._connection_epochs[connection] = epoch
            self._mailbox.startup_barrier_epoch = epoch
            self._mailbox.rebuild_current(startup_runtime_controls=self._startup_runtime_controls())
            if not self.desktop_runtime_controls_enabled:
                self._session_health.begin()
            self._authenticated_connections.add(connection)
            self._mailbox.replay_required = True
            authenticated = True
            if self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "connection_authenticated",
                    connection_id=connection_id,
                    authenticated_connections=len(self._authenticated_connections),
                    revision=self._mailbox.snapshot.revision,
                )
            self._ensure_writer()
            self._writer_wakeup.set()
            await asyncio.sleep(0)
            async for raw_message in connection:
                message = self._load_message(raw_message)
                if message.get("type") not in _REVERSE_KNOWN_TYPES:
                    await self._retire_connection(
                        connection,
                        epoch,
                        cause="unknown_reverse_message_type",
                    )
                    return
                try:
                    if message.get("type") == "owner_status":
                        await self.messages.put(self._session_health.handle_owner_status(message))
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
            if self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "connection_closed",
                    connection_id=connection_id,
                    authenticated=authenticated,
                    authenticated_connections=len(self._authenticated_connections),
                    code=close_code,
                    reason=close_reason,
                    last_snapshot_revision=self._mailbox.last_snapshot_revision,
                )
        finally:
            if epoch is not None and self._mailbox.startup_barrier_epoch == epoch:
                self._mailbox.startup_barrier_epoch = None
            if authenticated and self._connection_epochs.get(connection) == epoch:
                self._authenticated_connections.discard(connection)
                self._connection_epochs.pop(connection, None)
                if not self._unresolved_connections:
                    self._session_health.token_consumed = False
                if self.diagnostics is not None:
                    self.diagnostics.record_bridge(
                        "connection_detached",
                        connection_id=connection_id,
                        authenticated_connections=len(self._authenticated_connections),
                        last_snapshot_revision=self._mailbox.last_snapshot_revision,
                    )
                await self._bounded_close_connection(connection)

    def _is_valid_auth_payload(self, payload: dict[str, Any]) -> bool:
        return self._session_health.validate_auth(
            payload,
            session_token=self.session_token,
            token_consumed=self._session_health.token_consumed,
            stopping=self._stopping,
            desktop_runtime=self.desktop_runtime_controls_enabled,
            contract_version=OVERLAY_CONTRACT_VERSION,
            execution_contract=OVERLAY_EXECUTION_CONTRACT,
            native_retry_contract=OVERLAY_NATIVE_RETRY_CONTRACT,
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
                if self.overlay_instance_id is None or self.desktop_runtime_controls_enabled:
                    self._enqueue_control("heartbeat", {"type": "heartbeat"})
                    continue
                challenge = self._session_health.issue_challenge_if_due()
                if challenge is not None:
                    self._enqueue_control("health_challenge", challenge.payload)
                cause = self._session_health.failure_due()
                if cause is not None:
                    self.messages.put_nowait({"type": "runtime_error", "failure_reason": cause})
        except asyncio.CancelledError:
            raise

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
        registered: bool = True,
    ) -> asyncio.Task[Any]:
        if registered and self.task_factory is not None:
            return self.task_factory(coroutine, task_name=task_name)
        return asyncio.create_task(coroutine, name=f"OverlayBridge:{task_name}")

    def _create_transport_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        task_name: str,
    ) -> asyncio.Task[Any]:
        return self._create_task(
            coroutine,
            task_name=task_name,
            registered=False,
        )

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
                    control = self._mailbox.take_control(epoch=epoch)
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
                        if (
                            control.key == "shutdown"
                            and self._mailbox.startup_barrier_epoch == epoch
                        ):
                            self._mailbox.abandon_startup_scene(cause="shutdown_during_startup")
                        continue
                    scene = self._mailbox.take_scene_for_write()
                    if scene is None:
                        break
                    self._mailbox.active_scene = scene
                    try:
                        current = self._mailbox.current_scene
                        if scene.snapshot.revision < current.snapshot.revision:
                            self._mailbox.record_delivery(
                                outcome="superseded_product",
                                scene_revision=scene.snapshot.revision,
                                cause="newer_scene",
                                connection_epoch=epoch,
                            )
                            if self._mailbox.startup_barrier_epoch == epoch:
                                self._mailbox.replay_required = True
                            continue
                        message = self._mailbox.revalidated_scene_message(
                            scene, startup_runtime_controls=self._startup_runtime_controls()
                        )
                        written = await self._write_message(
                            connection,
                            epoch,
                            message,
                            timeout=_SCENE_WRITE_TIMEOUT_SECONDS,
                            scene_revision=scene.snapshot.revision,
                            payload_type="snapshot",
                        )
                        if written and self._mailbox.startup_barrier_epoch == epoch:
                            self._mailbox.startup_barrier_epoch = None
                    finally:
                        self._mailbox.active_scene = None
        except asyncio.CancelledError:
            active = self._mailbox.active_scene
            if active is not None:
                self._mailbox.record_delivery(
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
        start_time = time.perf_counter()
        if payload_type == "snapshot" and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "send_start",
                revision=scene_revision,
                authenticated_connections=len(self._authenticated_connections),
            )
        try:
            result = await self._transport_executor.write(
                connection,
                message,
                timeout=timeout,
                task_factory=self._create_transport_task,
            )
        except TransportWriteCancelled as exc:
            self._retain_unresolved_task(exc.send_task, connection=connection)
            if scene_revision is not None:
                self._mailbox.record_delivery(
                    outcome="ambiguous",
                    scene_revision=scene_revision,
                    cause="write_cancelled",
                    connection_epoch=epoch,
                )
            await self._retire_connection(connection, epoch, cause="write_cancelled")
            raise
        if result.outcome != "written":
            cause = result.cause or "write_failed"
            if not result.send_task.done():
                self._retain_unresolved_task(result.send_task, connection=connection)
            if scene_revision is not None:
                self._mailbox.record_delivery(
                    outcome="ambiguous",
                    scene_revision=scene_revision,
                    cause=cause,
                    connection_epoch=epoch,
                )
            if result.outcome == "failed" and self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "send_failure",
                    connection_id=self._connection_id(connection),
                    revision=scene_revision,
                    exception_type=cause,
                    removed=True,
                )
            await self._retire_connection(connection, epoch, cause=cause)
            elapsed_ms = max(0, int((time.perf_counter() - start_time) * 1000))
            if payload_type == "snapshot" and self.diagnostics is not None:
                self.diagnostics.record_bridge(
                    "send_finish",
                    revision=scene_revision,
                    authenticated_connections=len(self._authenticated_connections),
                    stale_connections=1,
                    elapsed_ms=elapsed_ms,
                )
            return False
        if scene_revision is not None:
            self._mailbox.record_delivery(
                outcome="written",
                scene_revision=scene_revision,
                cause=None,
                connection_epoch=epoch,
            )
            if not self.desktop_runtime_controls_enabled:
                self._session_health.record_scene_written(scene_revision)
        elapsed_ms = max(0, int((time.perf_counter() - start_time) * 1000))
        if payload_type == "snapshot" and self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "send_finish",
                revision=scene_revision,
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
        self._mailbox.replay_required = True
        if epoch is not None and self._mailbox.startup_barrier_epoch == epoch:
            self._mailbox.startup_barrier_epoch = None
        failure = await self._bounded_close_connection(connection)
        if failure is not None:
            self._unresolved_connections.add(connection)
            self._session_health.token_consumed = True
        elif not self._unresolved_connections:
            self._session_health.token_consumed = False
        if self.diagnostics is not None:
            self.diagnostics.record_bridge(
                "connection_retired",
                connection_id=self._connection_id(connection),
                cause=cause,
                last_snapshot_revision=self._mailbox.last_snapshot_revision,
            )

    async def _bounded_close_connection(
        self,
        connection: ServerConnection,
    ) -> Exception | None:
        close_task = self._connection_close_tasks.get(connection)
        if (
            close_task is not None
            and not close_task.done()
            and connection in self._unresolved_connections
        ):
            return TimeoutError("overlay connection close remains unresolved")
        result = await self._transport_executor.close(
            connection,
            timeout=_CLOSE_TIMEOUT_SECONDS,
            existing_task=close_task,
            task_factory=self._create_transport_task,
        )
        self._connection_close_tasks[connection] = result.close_task
        if result.failure is not None:
            self._unresolved_connections.add(connection)
            if isinstance(result.failure, TimeoutError):
                self._retain_unresolved_task(result.close_task, connection=connection)
            return result.failure
        self._connection_close_tasks.pop(connection, None)
        if not self._unresolved_tasks_by_connection.get(connection):
            self._unresolved_connections.discard(connection)
        return None

    def _abort_connection(self, connection: ServerConnection) -> None:
        self._transport_executor.abort(connection)

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
                    self._session_health.token_consumed = False

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
        if not self._mailbox.enqueue_control(key, payload, terminal=terminal):
            self._request_connection_retirement(cause="control_overflow")
            raise RuntimeError("overlay control capacity exhausted")
        self._ensure_writer()
        self._writer_wakeup.set()

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

    def _startup_runtime_controls(self) -> list[dict[str, Any]] | None:
        if not self.desktop_runtime_controls_enabled:
            return None
        controls: list[dict[str, Any]] = []
        controls.extend(dict(control) for control in self._mailbox.initial_desktop_runtime_controls)
        return controls

    def _desktop_runtime_control_message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "runtime_control",
            "payload": dict(payload),
        }

    def _ensure_desktop_runtime_controls_enabled(self) -> None:
        if not self.desktop_runtime_controls_enabled:
            raise RuntimeError("desktop runtime controls are only enabled for desktop overlays")
