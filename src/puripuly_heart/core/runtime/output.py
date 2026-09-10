from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import OrderedDict, deque
from collections.abc import Awaitable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID, uuid4

from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    redact_text_for_sink,
)
from puripuly_heart.core.output.models import (
    OUTPUT_ROUTE_SELF_CHATBOX,
    OUTPUT_ROUTE_SUBTITLE_OVERLAY,
    OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
    OUTPUT_ROUTING_DECISION_DENIED,
    OUTPUT_ROUTING_DECISION_PUBLISHED,
    OUTPUT_ROUTING_DECISION_SKIPPED,
    PUBLICATION_KIND_PEER_SUBTITLE,
    PUBLICATION_KIND_SELF_UTTERANCE,
    PUBLICATION_KIND_SYSTEM_DISCLOSURE,
    OutputPublicationKind,
    OutputRoute,
    OutputRoutingDecision,
    OutputRoutingDecisionStatus,
)
from puripuly_heart.core.overlay.sink import (
    OverlayApplicationReceipt,
    OverlayEvent,
    OverlayEventAdapter,
    OverlayEventUnion,
    OverlaySink,
    UtteranceClosed,
)
from puripuly_heart.domain.models import ChannelId, OSCMessage

OutputRuntimeState = Literal["open", "closing", "closed"]

SELF_SPEECH_TYPING_REASON = "self_speech_pending"
_OUTPUT_BATCH_MAX_UNSENT = 8
_OUTPUT_BATCH_MAX_BYTES = 1024 * 1024
_OUTPUT_SCOPE_MAX_BYTES = 9 * 1024 * 1024
_COMPLETED_PUBLICATION_LIMIT = 4096


@dataclass(slots=True)
class _OverlayOutputBatch:
    scope: str
    parent_id: str
    channel: ChannelId
    managed_parent: bool
    target_count: int
    turn_generation: int | None
    turn_order: int | None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    reserved_bytes: int = 0
    base_reserved_bytes: int = 0
    target_reserved_bytes: dict[int, int] = field(default_factory=dict)
    seen_targets: set[int] = field(default_factory=set)
    expected_targets: frozenset[int] = frozenset()
    completed_targets: set[int] = field(default_factory=set)
    active: bool = False
    disposition: str | None = None


class ChatboxQueue(Protocol):
    def enqueue(self, message: OSCMessage) -> None: ...
    def send_immediate(self, text: str) -> bool: ...
    def send_typing(self, is_typing: bool) -> None: ...
    def set_typing_reason(self, reason: str, active: bool) -> None: ...
    def clear_typing_reasons(self) -> None: ...
    def process_due(self) -> None: ...


class UIEventBridgePort(Protocol):
    async def run(self) -> None: ...
    async def wait_started(self) -> None: ...
    def report_overlay_state(
        self,
        state: str,
        *,
        failure_reason: str | None = None,
    ) -> None: ...
    def close(self) -> Awaitable[None] | None: ...


UIEventBridgeAdapter = UIEventBridgePort


@runtime_checkable
class ActiveSelfOverlaySinkPort(Protocol):
    def active_self_overlay_metadata(self) -> object | None: ...


@dataclass(frozen=True, slots=True)
class OutputPublicationResult:
    decision: OutputRoutingDecision
    message: OSCMessage | None = None


@dataclass(slots=True)
class OutputRuntime:
    chatbox: ChatboxQueue
    clock: Clock = field(default_factory=SystemClock)
    overlay_sink: OverlaySink | None = None
    overlay_event_adapter: OverlayEventAdapter | None = None
    flush_interval_s: float = 0.1
    diagnostics_capacity: int = 4096
    _state: OutputRuntimeState = "open"
    _chatbox_flush_task: asyncio.Task[None] | None = None
    _ui_event_bridge: UIEventBridgePort | None = None
    _ui_event_bridge_task: asyncio.Task[Any] | None = None
    _ui_event_bridge_started_wait_task: asyncio.Task[Any] | None = None
    _chatbox_typing_reasons_cleared: bool = False
    _chatbox_backlog_dropped: bool = False
    _completed_task_failures: dict[asyncio.Task[Any], Exception] = field(default_factory=dict)
    _tasks_being_collected: set[asyncio.Task[Any]] = field(default_factory=set)
    _active_delivery_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    _delivered_publications: set[tuple[OutputRoute, str]] = field(default_factory=set)
    _delivered_publication_order: OrderedDict[tuple[OutputRoute, str], tuple[str, int | None]] = (
        field(default_factory=OrderedDict)
    )
    _retired_publication_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    _adapter_sequence_namespaces: set[int] = field(default_factory=set)
    _publications_in_flight: set[tuple[OutputRoute, str]] = field(default_factory=set)
    _overlay_delivery_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _replacement_cancelled_delivery_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    _overlay_batches: dict[tuple[str, str], _OverlayOutputBatch] = field(default_factory=dict)
    _overlay_waiting: dict[str, deque[tuple[str, str]]] = field(default_factory=dict)
    _overlay_active: dict[str, tuple[str, str]] = field(default_factory=dict)
    _overlay_reserved_bytes: dict[str, int] = field(default_factory=dict)
    _terminal_overlay_batches: OrderedDict[tuple[str, str], str] = field(
        default_factory=OrderedDict
    )
    _retired_overlay_batch_order_ranges: dict[tuple[str, int], list[tuple[int, int]]] = field(
        default_factory=dict
    )
    _retired_turn_generations: dict[ChannelId, int] = field(
        default_factory=lambda: {"self": -1, "peer": -1}
    )
    _routing_decisions: deque[OutputRoutingDecision] = field(init=False)

    resource_fields = (
        "_chatbox_flush_task",
        "ChatboxPaginator._typing_reasons",
        "ChatboxPaginator._pending_pages",
        "ChatboxPaginator._pending_messages",
        "overlay_event_adapter",
        "overlay delivery tasks",
        "UIEventBridge.run task",
        "UIEventBridge startup wait task",
        "conversation adapter",
    )

    def __post_init__(self) -> None:
        if self.diagnostics_capacity < 1:
            raise ValueError("diagnostics_capacity must be positive")
        self._routing_decisions = deque(maxlen=self.diagnostics_capacity)
        if self.overlay_event_adapter is None:
            self.overlay_event_adapter = OverlayEventAdapter(clock=self.clock)

    @property
    def state(self) -> OutputRuntimeState:
        return self._state

    @property
    def is_accepting_publications(self) -> bool:
        return self._state == "open"

    @property
    def has_resources(self) -> bool:
        return (
            self._chatbox_flush_task is not None
            or self._ui_event_bridge_task is not None
            or self._ui_event_bridge_started_wait_task is not None
            or self._ui_event_bridge is not None
            or bool(self._active_delivery_tasks)
            or bool(self._overlay_batches)
            or not self._chatbox_typing_reasons_cleared
            or not self._chatbox_backlog_dropped
            or bool(self._completed_task_failures)
        )

    @property
    def chatbox_flush_task(self) -> asyncio.Task[None] | None:
        return self._chatbox_flush_task

    @property
    def ui_event_bridge_task(self) -> asyncio.Task[Any] | None:
        return self._ui_event_bridge_task

    @property
    def ui_event_bridge_started_wait_task(self) -> asyncio.Task[Any] | None:
        return self._ui_event_bridge_started_wait_task

    @property
    def routing_decisions(self) -> tuple[OutputRoutingDecision, ...]:
        return tuple(self._routing_decisions)

    def overlay_admission_snapshot(self) -> dict[str, object]:
        return {
            "active": len(self._overlay_active),
            "unsent": sum(len(waiting) for waiting in self._overlay_waiting.values()),
            "batches": len(self._overlay_batches),
            "reserved_bytes": sum(self._overlay_reserved_bytes.values()),
            "scopes": {
                scope: {
                    "active": int(scope in self._overlay_active),
                    "unsent": len(self._overlay_waiting.get(scope, ())),
                    "reserved_bytes": self._overlay_reserved_bytes.get(scope, 0),
                }
                for scope in set(self._overlay_active) | set(self._overlay_waiting)
            },
        }

    async def admit_translation_parent(
        self,
        *,
        parent_id: str,
        channel: ChannelId,
        origin: str,
        turn_generation: int,
        turn_order: int,
        retained_payload_bytes: int,
        destination_targets: Mapping[str, frozenset[int]],
    ) -> bool:
        if self._state != "open":
            return False
        if retained_payload_bytes > _OUTPUT_BATCH_MAX_BYTES:
            return False
        async with self._overlay_delivery_lock:
            if self._state != "open":
                return False
            plans: list[tuple[str, str, frozenset[int]]] = []
            for destination, targets in destination_targets.items():
                if not targets:
                    continue
                scope = self._parent_output_scope(origin, destination)
                key = (scope, parent_id)
                if key in self._overlay_batches or key in self._terminal_overlay_batches:
                    return False
                waiting = self._overlay_waiting.setdefault(scope, deque())
                if (
                    origin == "manual"
                    and scope in self._overlay_active
                    and len(waiting) >= _OUTPUT_BATCH_MAX_UNSENT
                ):
                    return False
                if (
                    self._overlay_reserved_bytes.get(scope, 0) + retained_payload_bytes
                    > _OUTPUT_SCOPE_MAX_BYTES
                ):
                    return False
                plans.append((destination, scope, targets))

            created: list[_OverlayOutputBatch] = []
            for destination, scope, targets in plans:
                waiting = self._overlay_waiting.setdefault(scope, deque())
                if scope in self._overlay_active and len(waiting) >= _OUTPUT_BATCH_MAX_UNSENT:
                    evicted_key = waiting.popleft()
                    evicted = self._overlay_batches.get(evicted_key)
                    if evicted is not None:
                        self._release_overlay_batch_locked(evicted, "output_overload")
                batch = _OverlayOutputBatch(
                    scope=scope,
                    parent_id=parent_id,
                    channel=channel,
                    managed_parent=True,
                    target_count=len(targets),
                    turn_generation=turn_generation,
                    turn_order=turn_order,
                    reserved_bytes=retained_payload_bytes,
                    base_reserved_bytes=retained_payload_bytes,
                    expected_targets=targets,
                )
                key = (scope, parent_id)
                self._overlay_batches[key] = batch
                self._overlay_reserved_bytes[scope] = (
                    self._overlay_reserved_bytes.get(scope, 0) + retained_payload_bytes
                )
                if scope not in self._overlay_active:
                    self._overlay_active[scope] = key
                    batch.active = True
                    batch.ready.set()
                else:
                    waiting.append(key)
                created.append(batch)
            return bool(created)

    async def resize_translation_parent_output(
        self,
        *,
        parent_id: str,
        origin: str,
        retained_payload_bytes: int,
        destination_indexes: Mapping[str, int],
    ) -> bool:
        async with self._overlay_delivery_lock:
            updates: list[tuple[_OverlayOutputBatch, int, int]] = []
            for destination, target_index in destination_indexes.items():
                scope = self._parent_output_scope(origin, destination)
                batch = self._overlay_batches.get((scope, parent_id))
                if batch is None or batch.disposition is not None:
                    continue
                previous = batch.target_reserved_bytes.get(target_index, 0)
                next_target_bytes = max(previous, retained_payload_bytes)
                additional = next_target_bytes - previous
                if batch.reserved_bytes + additional > _OUTPUT_BATCH_MAX_BYTES or (
                    self._overlay_reserved_bytes.get(scope, 0) + additional
                    > _OUTPUT_SCOPE_MAX_BYTES
                ):
                    for candidate in tuple(self._overlay_batches.values()):
                        if candidate.parent_id == parent_id and (
                            candidate.scope == origin or candidate.scope.startswith(f"{origin}:")
                        ):
                            self._release_overlay_batch_locked(
                                candidate,
                                "output_payload_exhausted",
                            )
                    return False
                updates.append((batch, target_index, next_target_bytes))
            for batch, target_index, next_target_bytes in updates:
                previous = batch.target_reserved_bytes.get(target_index, 0)
                additional = next_target_bytes - previous
                batch.target_reserved_bytes[target_index] = next_target_bytes
                batch.reserved_bytes += additional
                self._overlay_reserved_bytes[batch.scope] = (
                    self._overlay_reserved_bytes.get(batch.scope, 0) + additional
                )
            return True

    async def await_translation_parent(
        self,
        *,
        parent_id: str,
        origin: str,
    ) -> bool:
        async with self._overlay_delivery_lock:
            batches = tuple(
                batch
                for (scope, candidate_parent_id), batch in self._overlay_batches.items()
                if candidate_parent_id == parent_id
                and (scope == origin or scope.startswith(f"{origin}:"))
            )
        for batch in batches:
            await batch.ready.wait()
            if batch.disposition is not None:
                return False
        return bool(batches)

    async def complete_translation_parent_target(
        self,
        *,
        parent_id: str,
        origin: str,
        destination_indexes: Mapping[str, int],
    ) -> None:
        async with self._overlay_delivery_lock:
            for destination, target_index in destination_indexes.items():
                scope = self._parent_output_scope(origin, destination)
                batch = self._overlay_batches.get((scope, parent_id))
                if batch is None or batch.disposition is not None:
                    continue
                batch.completed_targets.add(target_index)
                expected = batch.expected_targets
                if (
                    expected
                    and expected.issubset(batch.completed_targets)
                    or not expected
                    and len(batch.completed_targets) >= batch.target_count
                ):
                    self._release_overlay_batch_locked(batch, "applied")

    @staticmethod
    def _parent_output_scope(origin: str, destination: str) -> str:
        return origin if destination == "overlay" else f"{origin}:{destination}"

    @property
    def has_overlay_destination(self) -> bool:
        return self.overlay_sink is not None

    @property
    def has_active_overlay_deliveries(self) -> bool:
        return bool(self._active_delivery_tasks)

    def active_self_overlay_metadata(self) -> object | None:
        overlay_sink = self.overlay_sink
        if not isinstance(overlay_sink, ActiveSelfOverlaySinkPort):
            return None
        return overlay_sink.active_self_overlay_metadata()

    async def replace_overlay_sink(
        self,
        overlay_sink: OverlaySink | None,
        *,
        expected_current: OverlaySink | None = None,
        require_match: bool = False,
    ) -> bool:
        if self._state != "open":
            raise RuntimeError("OutputRuntime is not accepting overlay destination replacement")
        async with self._overlay_delivery_lock:
            if self._state != "open":
                raise RuntimeError("OutputRuntime is not accepting overlay destination replacement")
            if require_match and self.overlay_sink is not expected_current:
                return False
            if self.overlay_sink is overlay_sink:
                return True
            await self._cancel_active_delivery_tasks_locked(replacement=True)
            self.overlay_sink = overlay_sink
            return True

    @staticmethod
    def chatbox_is_eligible(channel: ChannelId) -> bool:
        return channel == "self"

    @staticmethod
    def chatbox_is_denied(channel: ChannelId) -> bool:
        return channel == "peer"

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "OutputRuntime",
            "resource_fields": self.resource_fields,
            "stop_ingress": "stop accepting output publications",
            "shutdown_policy": (
                "chatbox: clear typing reasons on close; "
                "chatbox: drop pending pages/messages on close; "
                "overlay: cancel active delivery tasks; "
                "UI bridge: cancel task and close conversation adapter; "
                "overlay adapter: reject publications after close"
            ),
            "late_callback_rule": (
                "output after close returns denied/skipped observer decisions without user text"
            ),
        }

    async def start(self, *, auto_flush_chatbox: bool = False) -> None:
        if self._state == "closed":
            raise RuntimeError("OutputRuntime is closed; construct a new runtime to restart")
        if self._state == "closing":
            raise RuntimeError("OutputRuntime is closing; cannot start")
        self._state = "open"
        self._collect_done_task_failure(self._chatbox_flush_task)
        if auto_flush_chatbox and (
            self._chatbox_flush_task is None or self._chatbox_flush_task.done()
        ):
            self._chatbox_flush_task = self._create_task(
                self._run_chatbox_flush_loop(),
                task_name="chatbox-flush",
            )

    def start_ui_event_bridge(self, bridge: UIEventBridgePort) -> asyncio.Task[Any]:
        if self._state != "open":
            raise RuntimeError("OutputRuntime is not accepting UI event bridge work")
        if self._ui_event_bridge_task is not None:
            if not self._ui_event_bridge_task.done():
                raise RuntimeError("OutputRuntime already owns a UI event bridge task")
            self._collect_done_task_failure(self._ui_event_bridge_task)
        self._ui_event_bridge = bridge
        self._ui_event_bridge_task = self._create_task(
            bridge.run(),
            task_name="ui-event-bridge",
        )
        return self._ui_event_bridge_task

    async def wait_for_ui_event_bridge_started(self) -> None:
        bridge = self._ui_event_bridge
        bridge_task = self._ui_event_bridge_task
        if bridge is None or bridge_task is None:
            raise RuntimeError("OutputRuntime does not own a UI event bridge task")
        existing = self._ui_event_bridge_started_wait_task
        if existing is not None and not existing.done():
            raise RuntimeError("OutputRuntime already owns a UI event bridge startup waiter")
        started_task = self._create_task(
            bridge.wait_started(),
            task_name="ui-event-bridge-started-wait",
        )
        self._ui_event_bridge_started_wait_task = started_task
        try:
            done, _ = await asyncio.wait(
                {bridge_task, started_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if bridge_task in done:
                await bridge_task
                raise RuntimeError("UI Event Bridge stopped before reporting started")
            await started_task
            if bridge_task.done():
                await bridge_task
                raise RuntimeError("UI Event Bridge stopped during startup")
        finally:
            if not started_task.done():
                started_task.cancel()
            await asyncio.gather(started_task, return_exceptions=True)
            if self._ui_event_bridge_started_wait_task is started_task:
                self._ui_event_bridge_started_wait_task = None

    async def close(self) -> None:
        if self._state == "closed" and not self.has_resources:
            return

        self._state = "closing"
        failures: list[Exception] = []
        await self._cancel_chatbox_flush_task(failures)
        await self._cancel_active_delivery_tasks()
        self._clear_chatbox_typing_reasons(failures)
        self._drop_chatbox_backlog(failures)
        await self._cancel_ui_event_bridge_started_wait_task(failures)
        await self._cancel_ui_event_bridge_task(failures)
        await self._close_ui_event_bridge_adapter(failures)
        self._drain_completed_task_failures(failures)
        if failures:
            _raise_output_runtime_failures(failures)
        self._state = "closed"

    async def publish_chatbox(
        self,
        *,
        publication_id: UUID,
        channel: ChannelId,
        transcript_text: str,
        translation_text: str | None,
        include_source: bool,
        publication_kind: OutputPublicationKind | None = None,
        presentation_revision: int = 0,
        turn_generation: int | None = None,
        turn_order: int | None = None,
        target_indexes: tuple[int, ...] = (),
        target_languages: tuple[str, ...] = (),
    ) -> OutputPublicationResult:
        message = OSCMessage(
            utterance_id=publication_id,
            text=self._merge_chatbox_text(
                transcript_text=transcript_text,
                translation_text=translation_text,
                include_source=include_source,
            ),
            created_at=self.clock.now(),
            turn_generation=turn_generation,
            turn_order=turn_order,
            presentation_revision=presentation_revision,
            target_indexes=target_indexes,
            target_languages=target_languages,
        )
        publication_kind = publication_kind or (
            PUBLICATION_KIND_PEER_SUBTITLE if channel == "peer" else PUBLICATION_KIND_SELF_UTTERANCE
        )
        publication_metadata: dict[str, str | int | float | bool | None] = {
            "presentation_revision": presentation_revision,
        }
        if turn_generation is not None:
            publication_metadata.update(
                turn_generation=turn_generation,
                turn_order=turn_order,
            )
        if target_indexes:
            publication_metadata.update(
                target_indexes=",".join(str(index) for index in target_indexes),
                target_languages=",".join(target_languages),
            )
        if self._state != "open":
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(publication_id),
                publication_kind=publication_kind,
                reason=(
                    "output_runtime_closed" if self._state == "closed" else "output_runtime_closing"
                ),
                metadata={"channel": channel, "state": self._state, **publication_metadata},
            )
        if self.chatbox_is_denied(channel):
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_DENIED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(publication_id),
                publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                reason="peer_chatbox_denied",
                metadata={
                    "channel": "peer",
                    "attempted_route": OUTPUT_ROUTE_SELF_CHATBOX,
                    **publication_metadata,
                },
            )
        if not self.chatbox_is_eligible(channel):
            raise ValueError("unknown chatbox publication channel")

        publication_key = (
            OUTPUT_ROUTE_SELF_CHATBOX,
            f"{publication_id}:{presentation_revision}",
        )
        duplicate = self._duplicate_publication_result(
            publication_key=publication_key,
            publication_kind=publication_kind,
            channel=channel,
            logical_publication_id=str(publication_id),
            metadata=publication_metadata,
        )
        if duplicate is not None:
            return duplicate

        try:
            self.chatbox.enqueue(message)
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(publication_id),
                publication_kind=publication_kind,
                reason="destination_publish_failed",
                metadata={
                    "channel": channel,
                    "error_type": type(exc).__name__,
                    **publication_metadata,
                },
            )
        self._remember_delivered_publication(publication_key)
        self.set_self_chatbox_typing_reason(SELF_SPEECH_TYPING_REASON, False)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SELF_CHATBOX,
            publication_id=str(publication_id),
            publication_kind=publication_kind,
            reason=None,
            metadata={"channel": channel, **publication_metadata},
            message=message,
        )

    def set_self_chatbox_typing_reason(
        self,
        reason: str,
        active: bool,
    ) -> OutputPublicationResult:
        operation_uuid = uuid4()
        if self._state != "open":
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(operation_uuid),
                publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
                reason=(
                    "output_runtime_closed" if self._state == "closed" else "output_runtime_closing"
                ),
                metadata={"channel": "self", "state": self._state, "operation": "typing_reason"},
            )
        try:
            self.chatbox.set_typing_reason(reason, active)
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(operation_uuid),
                publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
                reason="destination_publish_failed",
                metadata={
                    "channel": "self",
                    "operation": "typing_reason",
                    "active": active,
                    "error_type": type(exc).__name__,
                },
            )
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SELF_CHATBOX,
            publication_id=str(operation_uuid),
            publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
            reason=None,
            metadata={"channel": "self", "operation": "typing_reason", "active": active},
        )

    def clear_self_chatbox_typing_reasons(
        self,
    ) -> OutputPublicationResult:
        operation_uuid = uuid4()
        if self._state != "open":
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(operation_uuid),
                publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
                reason=(
                    "output_runtime_closed" if self._state == "closed" else "output_runtime_closing"
                ),
                metadata={"channel": "self", "state": self._state, "operation": "typing_clear"},
            )
        try:
            self.chatbox.clear_typing_reasons()
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(operation_uuid),
                publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
                reason="destination_publish_failed",
                metadata={
                    "channel": "self",
                    "operation": "typing_clear",
                    "error_type": type(exc).__name__,
                },
            )
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SELF_CHATBOX,
            publication_id=str(operation_uuid),
            publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
            reason=None,
            metadata={"channel": "self", "operation": "typing_clear"},
        )

    def publish_system_disclosure_chatbox(
        self,
        *,
        text: str,
        disclosure_id: UUID | None = None,
    ) -> OutputPublicationResult:
        disclosure_uuid = disclosure_id or uuid4()
        if self._state != "open":
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason=(
                    "output_runtime_closed" if self._state == "closed" else "output_runtime_closing"
                ),
                metadata={"channel": "system", "state": self._state},
            )
        publication_key = (OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX, str(disclosure_uuid))
        duplicate = self._duplicate_publication_result(
            publication_key=publication_key,
            publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
            channel="system",
        )
        if duplicate is not None:
            return duplicate
        message = OSCMessage(
            utterance_id=disclosure_uuid,
            text=_redact_chatbox_disclosure_text(text),
            created_at=self.clock.now(),
        )
        try:
            self.chatbox.enqueue(message)
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason="destination_publish_failed",
                metadata={"channel": "system", "error_type": type(exc).__name__},
            )
        self._remember_delivered_publication(publication_key)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
            publication_id=str(disclosure_uuid),
            publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
            reason=None,
            metadata={"channel": "system"},
            message=message,
        )

    def publish_system_immediate_chatbox(
        self,
        *,
        text: str,
        disclosure_id: UUID | None = None,
    ) -> OutputPublicationResult:
        disclosure_uuid = disclosure_id or uuid4()
        if self._state != "open":
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason=(
                    "output_runtime_closed" if self._state == "closed" else "output_runtime_closing"
                ),
                metadata={"channel": "system", "state": self._state, "delivery": "immediate"},
            )
        publication_key = (OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX, str(disclosure_uuid))
        duplicate = self._duplicate_publication_result(
            publication_key=publication_key,
            publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
            channel="system",
        )
        if duplicate is not None:
            return duplicate
        safe_text = _redact_chatbox_disclosure_text(text)
        try:
            published = self.chatbox.send_immediate(safe_text)
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason="destination_publish_failed",
                metadata={
                    "channel": "system",
                    "delivery": "immediate",
                    "error_type": type(exc).__name__,
                },
            )
        if not published:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason="destination_rejected",
                metadata={"channel": "system", "delivery": "immediate"},
            )
        self._remember_delivered_publication(publication_key)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
            publication_id=str(disclosure_uuid),
            publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
            reason=None,
            metadata={"channel": "system", "delivery": "immediate"},
        )

    async def publish_overlay_event(self, event: OverlayEventUnion) -> OutputPublicationResult:
        if not isinstance(event, OverlayEvent):
            raise TypeError("event must implement the overlay event contract")
        if event.channel not in {"self", "peer"}:
            raise ValueError("overlay output requires a product channel")
        if not event.event_id.strip():
            raise ValueError("overlay output requires a publication identity")
        channel = event.channel
        publication_kind = (
            PUBLICATION_KIND_PEER_SUBTITLE if channel == "peer" else PUBLICATION_KIND_SELF_UTTERANCE
        )
        publication_id = event.event_id
        publication_scope = self._overlay_publication_scope(event)
        publication_key = (
            OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            f"{publication_scope}:{publication_id}",
        )
        batch: _OverlayOutputBatch | None = None
        overlay_sink: OverlaySink | None = None

        async with self._overlay_delivery_lock:
            if (
                event.turn_generation is None
                and event.sequence_namespace not in self._adapter_sequence_namespaces
            ):
                if len(self._adapter_sequence_namespaces) >= _COMPLETED_PUBLICATION_LIMIT:
                    return self._overlay_rejection_result(
                        event,
                        publication_kind=publication_kind,
                        reason="output_identity_capacity_exhausted",
                    )
                self._adapter_sequence_namespaces.add(event.sequence_namespace)
            rejected = self._overlay_preflight_result(
                event=event,
                publication_kind=publication_kind,
                publication_scope=publication_scope,
                publication_key=publication_key,
            )
            if rejected is not None:
                return rejected
            overlay_sink = self.overlay_sink
            if overlay_sink is None:
                return self._observe_result(
                    status=OUTPUT_ROUTING_DECISION_SKIPPED,
                    route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                    publication_id=publication_id,
                    publication_kind=publication_kind,
                    reason="destination_unconfigured",
                    metadata={"channel": channel},
                )
            batch, rejection_reason = self._register_overlay_batch_locked(event)
            if rejection_reason is not None:
                return self._overlay_rejection_result(
                    event,
                    publication_kind=publication_kind,
                    reason=rejection_reason,
                )

        if batch is None:
            raise RuntimeError("overlay batch admission did not produce an owner")
        if not batch.active:
            try:
                await batch.ready.wait()
            except asyncio.CancelledError:
                async with self._overlay_delivery_lock:
                    self._cancel_waiting_overlay_batch_locked(batch)
                raise
            if batch.disposition is not None:
                return self._overlay_rejection_result(
                    event,
                    publication_kind=publication_kind,
                    reason=batch.disposition,
                )

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("overlay publication task is unavailable")
        async with self._overlay_delivery_lock:
            if self._state != "open" or self.overlay_sink is not overlay_sink:
                self._release_overlay_batch_locked(batch, "destination_replaced")
                return self._overlay_rejection_result(
                    event,
                    publication_kind=publication_kind,
                    reason="destination_replaced",
                )
            self._publications_in_flight.add(publication_key)
            self._active_delivery_tasks.add(current_task)

        receipt: OverlayApplicationReceipt | None = None
        try:
            receipt = await overlay_sink.emit(event)
        except asyncio.CancelledError:
            receipt_lookup = getattr(overlay_sink, "application_receipt", None)
            if callable(receipt_lookup):
                receipt = receipt_lookup(publication_id)
            if receipt is None or receipt.outcome != "applied":
                if current_task in self._replacement_cancelled_delivery_tasks:
                    return self._overlay_rejection_result(
                        event,
                        publication_kind=publication_kind,
                        reason="destination_replaced",
                    )
                if self._state != "open":
                    return self._overlay_rejection_result(
                        event,
                        publication_kind=publication_kind,
                        reason="output_runtime_closing",
                    )
                async with self._overlay_delivery_lock:
                    self._release_overlay_batch_locked(batch, "cancelled_local")
                raise
        except Exception as exc:
            if (
                self._state == "open"
                and current_task not in self._replacement_cancelled_delivery_tasks
            ):
                async with self._overlay_delivery_lock:
                    self._release_overlay_batch_locked(batch, "destination_publish_failed")
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason="destination_publish_failed",
                metadata={
                    "channel": channel,
                    "error_type": type(exc).__name__,
                    "stage": "application_accepted",
                    "outcome": "not_applied",
                },
            )
        finally:
            self._active_delivery_tasks.discard(current_task)
            self._replacement_cancelled_delivery_tasks.discard(current_task)
            self._publications_in_flight.discard(publication_key)

        resolved_receipt = receipt or OverlayApplicationReceipt(
            stage="application_accepted",
            outcome="applied",
            publication_id=publication_id,
            scene_revision=None,
        )
        async with self._overlay_delivery_lock:
            self._remember_delivered_publication(
                publication_key,
                scope=publication_scope,
                sequence=event.seq,
            )
            if resolved_receipt.outcome != "applied":
                self._release_overlay_batch_locked(
                    batch,
                    resolved_receipt.cause or resolved_receipt.outcome,
                )
            elif not batch.managed_parent:
                self._release_overlay_batch_locked(batch, "applied")
            elif isinstance(event, UtteranceClosed):
                batch.completed_targets.add(event.target_index)
                if len(batch.completed_targets) >= batch.target_count:
                    self._release_overlay_batch_locked(batch, "applied")

        if resolved_receipt.outcome != "applied":
            return self._overlay_rejection_result(
                event,
                publication_kind=publication_kind,
                reason=resolved_receipt.cause or resolved_receipt.outcome,
                scene_revision=resolved_receipt.scene_revision,
            )
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason=None,
            metadata={
                "channel": channel,
                "stage": resolved_receipt.stage,
                "outcome": resolved_receipt.outcome,
                "scene_revision": resolved_receipt.scene_revision,
            },
        )

    def _overlay_preflight_result(
        self,
        *,
        event: OverlayEventUnion,
        publication_kind: OutputPublicationKind,
        publication_scope: str,
        publication_key: tuple[OutputRoute, str],
    ) -> OutputPublicationResult | None:
        if self._state != "open":
            return self._overlay_rejection_result(
                event,
                publication_kind=publication_kind,
                reason=(
                    "output_runtime_closed" if self._state == "closed" else "output_runtime_closing"
                ),
            )
        return self._duplicate_publication_result(
            publication_key=publication_key,
            publication_kind=publication_kind,
            channel=event.channel,
            logical_publication_id=event.event_id,
            scope=publication_scope,
            sequence=event.seq,
        )

    def _register_overlay_batch_locked(
        self,
        event: OverlayEventUnion,
    ) -> tuple[_OverlayOutputBatch | None, str | None]:
        scope = self._overlay_batch_scope(event)
        parent_id = str(event.parent_utterance_id or event.event_id)
        key = (scope, parent_id)
        terminal_disposition = self._terminal_overlay_batches.get(key)
        if terminal_disposition is not None:
            return None, terminal_disposition
        if event.turn_generation is not None and event.turn_generation <= (
            self._retired_turn_generations.get(event.channel, -1)
        ):
            return None, "stale_retired"
        if (
            event.turn_generation is not None
            and event.turn_order is not None
            and self._overlay_order_is_retired(
                scope,
                event.turn_generation,
                event.turn_order,
            )
        ):
            return None, "stale_retired"

        payload_bytes = max(
            event.retained_payload_bytes,
            self._overlay_event_payload_bytes(event),
        )
        if payload_bytes > _OUTPUT_BATCH_MAX_BYTES:
            return None, "output_payload_exhausted"

        batch = self._overlay_batches.get(key)
        if batch is not None:
            previous = batch.target_reserved_bytes.get(event.target_index, 0)
            next_target_bytes = max(previous, payload_bytes)
            additional_bytes = next_target_bytes - previous
            if (
                batch.reserved_bytes + additional_bytes > _OUTPUT_BATCH_MAX_BYTES
                or self._overlay_reserved_bytes.get(scope, 0) + additional_bytes
                > _OUTPUT_SCOPE_MAX_BYTES
            ):
                self._release_overlay_batch_locked(batch, "output_payload_exhausted")
                return None, "output_payload_exhausted"
            batch.target_reserved_bytes[event.target_index] = next_target_bytes
            batch.reserved_bytes += additional_bytes
            batch.seen_targets.add(event.target_index)
            self._overlay_reserved_bytes[scope] = (
                self._overlay_reserved_bytes.get(scope, 0) + additional_bytes
            )
            batch.target_count = max(batch.target_count, event.target_count)
            return batch, batch.disposition

        waiting = self._overlay_waiting.setdefault(scope, deque())
        if scope in self._overlay_active and len(waiting) >= _OUTPUT_BATCH_MAX_UNSENT:
            if event.turn_kind == "manual":
                return None, "output_overload"
            evicted_key = waiting.popleft()
            evicted = self._overlay_batches.get(evicted_key)
            if evicted is not None:
                self._release_overlay_batch_locked(evicted, "output_overload")

        if self._overlay_reserved_bytes.get(scope, 0) + payload_bytes > _OUTPUT_SCOPE_MAX_BYTES:
            return None, "output_payload_exhausted"
        batch = _OverlayOutputBatch(
            scope=scope,
            parent_id=parent_id,
            channel=event.channel,
            managed_parent=event.parent_utterance_id is not None,
            target_count=max(1, event.target_count),
            turn_generation=event.turn_generation,
            turn_order=event.turn_order,
            reserved_bytes=payload_bytes,
            base_reserved_bytes=0,
            target_reserved_bytes={event.target_index: payload_bytes},
            seen_targets={event.target_index},
            expected_targets=frozenset(range(max(1, event.target_count))),
        )
        self._overlay_batches[key] = batch
        self._overlay_reserved_bytes[scope] = (
            self._overlay_reserved_bytes.get(scope, 0) + payload_bytes
        )
        if scope not in self._overlay_active:
            self._overlay_active[scope] = key
            batch.active = True
            batch.ready.set()
            return batch, None

        waiting.append(key)
        return batch, None

    def _cancel_waiting_overlay_batch_locked(self, batch: _OverlayOutputBatch) -> None:
        if batch.active or batch.disposition is not None:
            return
        waiting = self._overlay_waiting.get(batch.scope)
        key = (batch.scope, batch.parent_id)
        if waiting is not None:
            with contextlib.suppress(ValueError):
                waiting.remove(key)
        self._release_overlay_batch_locked(batch, "cancelled_local")

    def _release_overlay_batch_locked(
        self,
        batch: _OverlayOutputBatch,
        disposition: str,
    ) -> None:
        key = (batch.scope, batch.parent_id)
        if self._overlay_batches.get(key) is not batch:
            return
        self._overlay_batches.pop(key, None)
        reserved_bytes = max(
            0,
            self._overlay_reserved_bytes.get(batch.scope, 0) - batch.reserved_bytes,
        )
        if reserved_bytes:
            self._overlay_reserved_bytes[batch.scope] = reserved_bytes
        else:
            self._overlay_reserved_bytes.pop(batch.scope, None)
        batch.disposition = disposition
        batch.ready.set()
        waiting = self._overlay_waiting.get(batch.scope)
        if waiting is not None:
            with contextlib.suppress(ValueError):
                waiting.remove(key)
        if batch.managed_parent:
            event_stub = (
                batch.turn_generation,
                batch.turn_order,
            )
            self._remember_terminal_overlay_batch_values(
                key,
                disposition,
                event_stub,
            )
        if self._overlay_active.get(batch.scope) != key:
            self._prune_output_scope(batch.scope)
            return
        self._overlay_active.pop(batch.scope, None)
        while waiting:
            successor_key = waiting.popleft()
            successor = self._overlay_batches.get(successor_key)
            if successor is None or successor.disposition is not None:
                continue
            self._overlay_active[batch.scope] = successor_key
            successor.active = True
            successor.ready.set()
            break
        self._prune_output_scope(batch.scope)

    def _prune_output_scope(self, scope: str) -> None:
        waiting = self._overlay_waiting.get(scope)
        if waiting is not None and not waiting:
            self._overlay_waiting.pop(scope, None)
        if self._overlay_reserved_bytes.get(scope) == 0:
            self._overlay_reserved_bytes.pop(scope, None)

    def _remember_terminal_overlay_batch(
        self,
        key: tuple[str, str],
        disposition: str,
        event: OverlayEventUnion,
    ) -> None:
        self._remember_terminal_overlay_batch_values(
            key,
            disposition,
            (event.turn_generation, event.turn_order),
        )

    def _remember_terminal_overlay_batch_values(
        self,
        key: tuple[str, str],
        disposition: str,
        ordering: tuple[int | None, int | None],
    ) -> None:
        self._terminal_overlay_batches.pop(key, None)
        self._terminal_overlay_batches[key] = disposition
        generation, order = ordering
        if generation is not None and order is not None:
            self._remember_retired_overlay_order(key[0], generation, order)
        while len(self._terminal_overlay_batches) > _COMPLETED_PUBLICATION_LIMIT:
            self._terminal_overlay_batches.popitem(last=False)

    def _remember_retired_overlay_order(
        self,
        scope: str,
        generation: int,
        order: int,
    ) -> None:
        key = (scope, generation)
        ranges = self._retired_overlay_batch_order_ranges.setdefault(key, [])
        start = order
        end = order
        merged: list[tuple[int, int]] = []
        inserted = False
        for existing_start, existing_end in ranges:
            if existing_end + 1 < start:
                merged.append((existing_start, existing_end))
                continue
            if end + 1 < existing_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((existing_start, existing_end))
                continue
            start = min(start, existing_start)
            end = max(end, existing_end)
        if not inserted:
            merged.append((start, end))
        self._retired_overlay_batch_order_ranges[key] = merged

    def _overlay_order_is_retired(
        self,
        scope: str,
        generation: int,
        order: int,
    ) -> bool:
        return any(
            start <= order <= end
            for start, end in self._retired_overlay_batch_order_ranges.get(
                (scope, generation),
                (),
            )
        )

    def retire_turn_generation(self, channel: ChannelId, turn_generation: int) -> None:
        previous = self._retired_turn_generations.get(channel, -1)
        if turn_generation - 1 <= previous:
            return
        self._retired_turn_generations[channel] = turn_generation - 1
        for batch in tuple(self._overlay_batches.values()):
            if batch.channel == channel and (
                batch.turn_generation is None or batch.turn_generation < turn_generation
            ):
                self._release_overlay_batch_locked(batch, "generation_retired")
        for key in tuple(self._retired_overlay_batch_order_ranges):
            if key[1] < turn_generation and (
                key[0] == channel
                or key[0] == "manual"
                or key[0].startswith(f"{channel}:")
                or key[0].startswith("manual:")
            ):
                self._retired_overlay_batch_order_ranges.pop(key, None)
        for scope in tuple(self._retired_publication_ranges):
            if not self._publication_scope_precedes_generation(
                scope,
                channel=channel,
                turn_generation=turn_generation,
            ):
                continue
            self._retired_publication_ranges.pop(scope, None)
        for publication_key, (scope, _sequence) in tuple(self._delivered_publication_order.items()):
            if self._publication_scope_precedes_generation(
                scope,
                channel=channel,
                turn_generation=turn_generation,
            ):
                self._delivered_publication_order.pop(publication_key, None)
                self._delivered_publications.discard(publication_key)

    @staticmethod
    def _publication_scope_precedes_generation(
        scope: str,
        *,
        channel: ChannelId,
        turn_generation: int,
    ) -> bool:
        prefixes = [f"{channel}:overlay:"]
        if channel == "self":
            prefixes.append("manual:overlay:")
        for prefix in prefixes:
            if not scope.startswith(prefix):
                continue
            generation_text = scope[len(prefix) :]
            return generation_text.isdigit() and int(generation_text) < turn_generation
        return False

    def _overlay_rejection_result(
        self,
        event: OverlayEventUnion,
        *,
        publication_kind: OutputPublicationKind,
        reason: str,
        scene_revision: int | None = None,
    ) -> OutputPublicationResult:
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_SKIPPED,
            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            publication_id=event.event_id,
            publication_kind=publication_kind,
            reason=reason,
            metadata={
                "channel": event.channel,
                "stage": "application_accepted",
                "outcome": "not_applied",
                "scene_revision": scene_revision,
            },
        )

    @staticmethod
    def _overlay_batch_scope(event: OverlayEventUnion) -> str:
        return event.turn_kind or str(event.channel)

    @classmethod
    def _overlay_publication_scope(cls, event: OverlayEventUnion) -> str:
        if event.turn_generation is None:
            return (
                f"{cls._overlay_batch_scope(event)}:overlay-adapter:" f"{event.sequence_namespace}"
            )
        return f"{cls._overlay_batch_scope(event)}:overlay:{event.turn_generation}"

    @staticmethod
    def _overlay_event_payload_bytes(event: OverlayEventUnion) -> int:
        total = 256
        for field_name in (
            "event_id",
            "update_id",
            "session_scope",
            "source_text_hash",
            "logical_turn_key",
            "text",
            "source_text",
            "secondary_text",
            "source_language",
            "target_language",
            "occupant_key",
        ):
            value = getattr(event, field_name, None)
            if isinstance(value, str):
                total += len(value.encode("utf-8"))
        return total

    def reject_if_closed(
        self,
        *,
        route: OutputRoute,
        publication_id: str,
        publication_kind: OutputPublicationKind,
        channel: ChannelId | str | None,
    ) -> OutputRoutingDecision | None:
        if self._state == "open":
            return None
        return self._observe_decision(
            status=OUTPUT_ROUTING_DECISION_SKIPPED,
            route=route,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason="output_runtime_closed" if self._state == "closed" else "output_runtime_closing",
            metadata={"channel": channel, "state": self._state},
        )

    async def _run_chatbox_flush_loop(self) -> None:
        try:
            while True:
                self.chatbox.process_due()
                await asyncio.sleep(self.flush_interval_s)
        except asyncio.CancelledError:
            raise

    def _create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        task_name: str,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine, name=f"OutputRuntime:{task_name}")
        task.add_done_callback(self._record_task_completion)
        return task

    async def _cancel_chatbox_flush_task(self, failures: list[Exception]) -> None:
        task = self._chatbox_flush_task
        if task is None:
            return
        await self._cancel_owned_task(task, failures)
        self._chatbox_flush_task = None

    async def _cancel_ui_event_bridge_task(self, failures: list[Exception]) -> None:
        task = self._ui_event_bridge_task
        if task is None:
            return
        await self._cancel_owned_task(task, failures)
        self._ui_event_bridge_task = None

    async def _cancel_ui_event_bridge_started_wait_task(
        self,
        failures: list[Exception],
    ) -> None:
        task = self._ui_event_bridge_started_wait_task
        if task is None:
            return
        await self._cancel_owned_task(task, failures)
        self._ui_event_bridge_started_wait_task = None

    async def _cancel_active_delivery_tasks(self) -> None:
        async with self._overlay_delivery_lock:
            await self._cancel_active_delivery_tasks_locked(replacement=False)

    async def _cancel_active_delivery_tasks_locked(self, *, replacement: bool) -> None:
        disposition = "destination_replaced" if replacement else "output_runtime_closing"
        for key, batch in tuple(self._overlay_batches.items()):
            batch.disposition = disposition
            batch.ready.set()
            if batch.managed_parent:
                self._remember_terminal_overlay_batch_values(
                    key,
                    disposition,
                    (batch.turn_generation, batch.turn_order),
                )
        self._overlay_batches.clear()
        self._overlay_waiting.clear()
        self._overlay_active.clear()
        self._overlay_reserved_bytes.clear()

        tasks = tuple(self._active_delivery_tasks)
        if replacement:
            self._replacement_cancelled_delivery_tasks.update(tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_delivery_tasks.difference_update(tasks)
        self._publications_in_flight.clear()

    async def _close_ui_event_bridge_adapter(self, failures: list[Exception]) -> None:
        bridge = self._ui_event_bridge
        if bridge is None:
            return
        try:
            result = bridge.close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            failures.append(exc)
            return
        self._ui_event_bridge = None

    def _drop_chatbox_backlog(self, failures: list[Exception]) -> None:
        if self._chatbox_backlog_dropped:
            return
        drop_pending = getattr(self.chatbox, "drop_pending", None)
        if not callable(drop_pending):
            self._chatbox_backlog_dropped = True
            return
        try:
            drop_pending()
        except Exception as exc:
            failures.append(exc)
            return
        self._chatbox_backlog_dropped = True

    def _clear_chatbox_typing_reasons(self, failures: list[Exception]) -> None:
        if self._chatbox_typing_reasons_cleared:
            return
        clear_typing_reasons = getattr(self.chatbox, "clear_typing_reasons", None)
        if not callable(clear_typing_reasons):
            self._chatbox_typing_reasons_cleared = True
            return
        try:
            clear_typing_reasons()
        except Exception as exc:
            failures.append(exc)
            return
        self._chatbox_typing_reasons_cleared = True

    async def _cancel_owned_task(
        self,
        task: asyncio.Task[Any],
        failures: list[Exception],
    ) -> None:
        stored_failure = self._completed_task_failures.pop(task, None)
        if stored_failure is not None:
            failures.append(stored_failure)
            return

        self._tasks_being_collected.add(task)
        try:
            if not task.done():
                task.cancel()
            results = await asyncio.gather(task, return_exceptions=True)
        finally:
            self._tasks_being_collected.discard(task)

        stored_failure = self._completed_task_failures.pop(task, None)
        if stored_failure is not None:
            failures.append(stored_failure)
            return
        self._append_non_cancel_failures(results, failures)

    def _record_task_completion(self, task: asyncio.Task[Any]) -> None:
        if task in self._tasks_being_collected:
            return
        self._collect_done_task_failure(task)

    def _collect_done_task_failure(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or not task.done() or task in self._completed_task_failures:
            return
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            self._completed_task_failures[task] = exc

    def _drain_completed_task_failures(self, failures: list[Exception]) -> None:
        failures.extend(self._completed_task_failures.values())
        self._completed_task_failures.clear()

    @staticmethod
    def _append_non_cancel_failures(
        results: list[object],
        failures: list[Exception],
    ) -> None:
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, Exception):
                failures.append(result)

    @staticmethod
    def _merge_chatbox_text(
        *,
        transcript_text: str,
        translation_text: str | None,
        include_source: bool,
    ) -> str:
        if translation_text is None:
            return transcript_text
        if include_source:
            return f"{transcript_text} ({translation_text})"
        return translation_text

    def _observe_result(
        self,
        *,
        status: OutputRoutingDecisionStatus,
        route: OutputRoute,
        publication_id: str,
        publication_kind: OutputPublicationKind,
        reason: str | None,
        metadata: dict[str, str | int | float | bool | None],
        message: OSCMessage | None = None,
    ) -> OutputPublicationResult:
        return OutputPublicationResult(
            decision=self._observe_decision(
                status=status,
                route=route,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason=reason,
                metadata=metadata,
            ),
            message=message,
        )

    def _duplicate_publication_result(
        self,
        *,
        publication_key: tuple[OutputRoute, str],
        publication_kind: OutputPublicationKind,
        channel: ChannelId | str | None,
        logical_publication_id: str | None = None,
        metadata: dict[str, str | int | float | bool | None] | None = None,
        scope: str | None = None,
        sequence: int | None = None,
    ) -> OutputPublicationResult | None:
        resolved_scope = scope or str(publication_key[0])
        sequence_retired = sequence is not None and any(
            start <= sequence <= end
            for start, end in self._retired_publication_ranges.get(resolved_scope, ())
        )
        if (
            publication_key not in self._delivered_publications
            and publication_key not in self._publications_in_flight
            and not sequence_retired
        ):
            return None
        route, publication_id = publication_key
        decision_metadata: dict[str, str | int | float | bool | None] = {"channel": channel}
        if metadata is not None:
            decision_metadata.update(metadata)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_SKIPPED,
            route=route,
            publication_id=logical_publication_id or publication_id,
            publication_kind=publication_kind,
            reason="duplicate_publication",
            metadata=decision_metadata,
        )

    def _remember_delivered_publication(
        self,
        publication_key: tuple[OutputRoute, str],
        *,
        scope: str | None = None,
        sequence: int | None = None,
    ) -> None:
        resolved_scope = scope or str(publication_key[0])
        completed = self._delivered_publication_order
        if publication_key in completed:
            completed.move_to_end(publication_key)
            return
        self._delivered_publications.add(publication_key)
        completed[publication_key] = (resolved_scope, sequence)
        while len(completed) > _COMPLETED_PUBLICATION_LIMIT:
            evicted_key, (evicted_scope, evicted_sequence) = completed.popitem(last=False)
            self._delivered_publications.discard(evicted_key)
            if evicted_sequence is not None:
                self._remember_retired_publication_sequence(
                    evicted_scope,
                    evicted_sequence,
                )

    def _remember_retired_publication_sequence(self, scope: str, sequence: int) -> None:
        ranges = self._retired_publication_ranges.setdefault(scope, [])
        start = sequence
        end = sequence
        merged: list[tuple[int, int]] = []
        inserted = False
        for current_start, current_end in ranges:
            if current_end + 1 < start:
                merged.append((current_start, current_end))
                continue
            if end + 1 < current_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((current_start, current_end))
                continue
            start = min(start, current_start)
            end = max(end, current_end)
        if not inserted:
            merged.append((start, end))
        self._retired_publication_ranges[scope] = merged

    def _observe_decision(
        self,
        *,
        status: OutputRoutingDecisionStatus,
        route: OutputRoute,
        publication_id: str,
        publication_kind: OutputPublicationKind,
        reason: str | None,
        metadata: dict[str, str | int | float | bool | None],
    ) -> OutputRoutingDecision:
        decision = OutputRoutingDecision(
            decision=status,
            route=route,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason=reason,
            metadata=metadata,
        )
        self._routing_decisions.append(decision)
        return decision


def _raise_output_runtime_failures(failures: list[Exception]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup("OutputRuntime close failed", failures)


def _redact_chatbox_disclosure_text(text: str) -> str:
    result = redact_text_for_sink(text, DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE)
    if result.status == DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED and result.text is not None:
        return result.text
    return DIAGNOSTIC_REDACTION_MARKER


__all__ = [
    "ChatboxQueue",
    "OutputPublicationResult",
    "OutputRuntime",
    "OutputRuntimeState",
    "UIEventBridgeAdapter",
    "UIEventBridgePort",
]
