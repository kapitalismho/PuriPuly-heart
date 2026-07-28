from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
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
    OverlayEvent,
    OverlayEventAdapter,
    OverlayEventUnion,
    OverlaySink,
)
from puripuly_heart.domain.models import ChannelId, OSCMessage

OutputRuntimeState = Literal["open", "closing", "closed"]

SELF_SPEECH_TYPING_REASON = "self_speech_pending"


class ChatboxQueue(Protocol):
    def enqueue(self, message: OSCMessage) -> None: ...
    def send_immediate(self, text: str) -> bool: ...
    def send_typing(self, is_typing: bool) -> None: ...
    def set_typing_reason(self, reason: str, active: bool) -> None: ...
    def clear_typing_reasons(self) -> None: ...
    def process_due(self) -> None: ...


class UIEventBridgeAdapter(Protocol):
    async def run(self) -> None: ...


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
    _ui_event_bridge: UIEventBridgeAdapter | None = None
    _ui_event_bridge_task: asyncio.Task[Any] | None = None
    _ui_event_bridge_started_wait_task: asyncio.Task[Any] | None = None
    _chatbox_typing_reasons_cleared: bool = False
    _chatbox_backlog_dropped: bool = False
    _completed_task_failures: dict[asyncio.Task[Any], Exception] = field(default_factory=dict)
    _tasks_being_collected: set[asyncio.Task[Any]] = field(default_factory=set)
    _active_delivery_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    _delivered_publications: set[tuple[OutputRoute, str]] = field(default_factory=set)
    _publications_in_flight: set[tuple[OutputRoute, str]] = field(default_factory=set)
    _overlay_delivery_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _replacement_cancelled_delivery_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
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

    @property
    def has_overlay_destination(self) -> bool:
        return self.overlay_sink is not None

    @property
    def has_active_overlay_deliveries(self) -> bool:
        return bool(self._active_delivery_tasks)

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

    def start_ui_event_bridge(self, bridge: UIEventBridgeAdapter) -> asyncio.Task[Any]:
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
        wait_started = getattr(bridge, "wait_started", None)
        if not callable(wait_started):
            await asyncio.sleep(0)
            if bridge_task.done():
                await bridge_task
            return
        existing = self._ui_event_bridge_started_wait_task
        if existing is not None and not existing.done():
            raise RuntimeError("OutputRuntime already owns a UI event bridge startup waiter")
        started_task = self._create_task(
            wait_started(),
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
    ) -> OutputPublicationResult:
        publication_kind = publication_kind or (
            PUBLICATION_KIND_PEER_SUBTITLE if channel == "peer" else PUBLICATION_KIND_SELF_UTTERANCE
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
                metadata={"channel": channel, "state": self._state},
            )
        if self.chatbox_is_denied(channel):
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_DENIED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(publication_id),
                publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                reason="peer_chatbox_denied",
                metadata={"channel": "peer", "attempted_route": OUTPUT_ROUTE_SELF_CHATBOX},
            )
        if not self.chatbox_is_eligible(channel):
            raise ValueError("unknown chatbox publication channel")

        publication_key = (OUTPUT_ROUTE_SELF_CHATBOX, str(publication_id))
        duplicate = self._duplicate_publication_result(
            publication_key=publication_key,
            publication_kind=publication_kind,
            channel=channel,
        )
        if duplicate is not None:
            return duplicate

        message = OSCMessage(
            utterance_id=publication_id,
            text=self._merge_chatbox_text(
                transcript_text=transcript_text,
                translation_text=translation_text,
                include_source=include_source,
            ),
            created_at=self.clock.now(),
        )
        try:
            self.chatbox.enqueue(message)
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(publication_id),
                publication_kind=publication_kind,
                reason="destination_publish_failed",
                metadata={"channel": channel, "error_type": type(exc).__name__},
            )
        self._remember_delivered_publication(publication_key)
        self.set_self_chatbox_typing_reason(SELF_SPEECH_TYPING_REASON, False)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SELF_CHATBOX,
            publication_id=str(publication_id),
            publication_kind=publication_kind,
            reason=None,
            metadata={"channel": channel},
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
        publication_key = (OUTPUT_ROUTE_SUBTITLE_OVERLAY, publication_id)
        async with self._overlay_delivery_lock:
            if self._state != "open":
                return self._observe_result(
                    status=OUTPUT_ROUTING_DECISION_SKIPPED,
                    route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                    publication_id=publication_id,
                    publication_kind=publication_kind,
                    reason=(
                        "output_runtime_closed"
                        if self._state == "closed"
                        else "output_runtime_closing"
                    ),
                    metadata={"channel": channel, "state": self._state},
                )
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
            duplicate = self._duplicate_publication_result(
                publication_key=publication_key,
                publication_kind=publication_kind,
                channel=channel,
            )
            if duplicate is not None:
                return duplicate
            self._publications_in_flight.add(publication_key)
            task = asyncio.create_task(
                overlay_sink.emit(event),
                name=f"OutputRuntime:overlay-delivery:{publication_id}",
            )
            self._active_delivery_tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            if task in self._replacement_cancelled_delivery_tasks:
                return self._observe_result(
                    status=OUTPUT_ROUTING_DECISION_SKIPPED,
                    route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                    publication_id=publication_id,
                    publication_kind=publication_kind,
                    reason="destination_replaced",
                    metadata={"channel": channel},
                )
            if self._state == "open":
                raise
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason="output_runtime_closing",
                metadata={"channel": channel, "state": self._state},
            )
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason="destination_publish_failed",
                metadata={"channel": channel, "error_type": type(exc).__name__},
            )
        finally:
            self._active_delivery_tasks.discard(task)
            self._replacement_cancelled_delivery_tasks.discard(task)
            self._publications_in_flight.discard(publication_key)

        self._remember_delivered_publication(publication_key)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason=None,
            metadata={"channel": channel},
        )

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
        tasks = tuple(self._active_delivery_tasks)
        if not tasks:
            return
        if replacement:
            self._replacement_cancelled_delivery_tasks.update(tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._active_delivery_tasks.difference_update(tasks)
        self._publications_in_flight.clear()

    async def _close_ui_event_bridge_adapter(self, failures: list[Exception]) -> None:
        bridge = self._ui_event_bridge
        if bridge is None:
            return
        close = getattr(bridge, "close", None)
        if not callable(close):
            self._ui_event_bridge = None
            return
        try:
            result = close()
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
    ) -> OutputPublicationResult | None:
        if (
            publication_key not in self._delivered_publications
            and publication_key not in self._publications_in_flight
        ):
            return None
        route, publication_id = publication_key
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_SKIPPED,
            route=route,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason="duplicate_publication",
            metadata={"channel": channel},
        )

    def _remember_delivered_publication(
        self,
        publication_key: tuple[OutputRoute, str],
    ) -> None:
        if publication_key in self._delivered_publications:
            return
        self._delivered_publications.add(publication_key)

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
]
