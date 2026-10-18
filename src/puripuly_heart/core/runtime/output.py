from __future__ import annotations

import asyncio
import inspect
import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID, uuid4

from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_CHATBOX_DISCLOSURE,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    redact_text_for_sink,
)
from puripuly_heart.core.diagnostics import (
    current_diagnostic_operation,
    emit_failure,
    safe_exception_fields,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfig
from puripuly_heart.core.output.models import (
    OUTPUT_ROUTE_CONVERSATION_FEED,
    OUTPUT_ROUTE_SELF_CHATBOX,
    OUTPUT_ROUTE_SUBTITLE_OVERLAY,
    OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
    OUTPUT_ROUTING_DECISION_DENIED,
    OUTPUT_ROUTING_DECISION_PUBLISHED,
    OUTPUT_ROUTING_DECISION_SKIPPED,
    PUBLICATION_KIND_CONVERSATION_FEED,
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
from puripuly_heart.core.runtime.output_batch import (
    COMPLETED_BATCH_LIMIT,
    OUTPUT_BATCH_MAX_UNSENT,
    DestinationBatch,
    DestinationBatchAdmission,
    OutputDestination,
)
from puripuly_heart.domain.events import SelfTurnPresentation
from puripuly_heart.domain.models import ChannelId, OSCMessage

OutputRuntimeState = Literal["open", "closing", "closed"]
SELF_SPEECH_TYPING_REASON = "self_speech_pending"
_COMPLETED_PUBLICATION_LIMIT = COMPLETED_BATCH_LIMIT
logger = logging.getLogger(__name__)


class ChatboxQueue(Protocol):
    def bind_delivery_observer(self, observer: Callable[[OSCMessage, str], None]) -> None: ...
    def enqueue_self_turn(self, message: OSCMessage, *, final: bool) -> OSCMessage | None: ...
    def promote_self_turn(self, utterance_id: UUID, revision: int) -> None: ...
    def enqueue(self, message: OSCMessage) -> OSCMessage | None: ...
    def send_immediate(self, text: str) -> tuple[bool, ErrorDiagnostics | None]: ...
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


class UiDeliveryLifecyclePort(Protocol):
    @property
    def has_resources(self) -> bool: ...

    def activate_peer_generation(self, generation: int) -> None: ...
    def retire_peer_generation(self, generation: int) -> None: ...
    def retire_self_generation(self, generation: int) -> None: ...
    def retire_destination(self) -> None: ...
    async def wait_for_idle(self) -> None: ...
    async def wait_for_peer_idle(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class ActiveSelfOverlaySinkPort(Protocol):
    def active_self_overlay_metadata(self) -> object | None: ...


@dataclass(frozen=True, slots=True)
class OutputPublicationResult:
    decision: OutputRoutingDecision
    message: OSCMessage | None = None


@dataclass(slots=True)
class _PeerOverlayBatch:
    publication_generation: int
    source_order: int
    parent_id: UUID
    events: list[
        tuple[
            OverlayEventUnion,
            tuple[OutputRoute, str],
            str,
            DestinationBatch,
        ]
    ] = field(default_factory=list)


@dataclass(slots=True)
class _SelfStreamDelivery:
    snapshot: SelfTurnPresentation
    configuration: TranslationRuntimeConfig
    offered_revision: int = -1
    offered_source_revision: int = 0
    promotion_revision: int = -1
    offered_fingerprint: tuple[object, ...] | None = None
    sent_fingerprint: tuple[object, ...] | None = None
    receipt: str | None = None
    settled: bool = False
    batch: DestinationBatch | None = None


@dataclass(slots=True)
class OutputRuntime:
    chatbox: ChatboxQueue
    clock: Clock = field(default_factory=SystemClock)
    overlay_sink: OverlaySink | None = None
    overlay_event_adapter: OverlayEventAdapter | None = None
    flush_interval_s: float = 0.1
    diagnostics_capacity: int = 4096
    routing_observer: Callable[[OutputRoutingDecision], object] | None = field(
        default=None,
        repr=False,
    )
    activation_notice_enabled: bool = True
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
    _replacement_cancelled_delivery_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    _peer_generation: int | None = None
    _retired_peer_generation: int = -1
    _latest_peer_source_order: int = -1
    _peer_identity_by_utterance: dict[UUID, tuple[int, int]] = field(default_factory=dict)
    _peer_identity_order: deque[UUID] = field(default_factory=deque)
    _peer_overlay_batches: deque[_PeerOverlayBatch] = field(default_factory=deque)
    _peer_active_batch: _PeerOverlayBatch | None = None
    _peer_overlay_worker: asyncio.Task[None] | None = None
    _peer_writer_cancel_reason: str | None = None
    _ui_delivery: UiDeliveryLifecyclePort | None = None
    _peer_overlay_delivery_observer: Callable[[OutputRoutingDecision], None] | None = None
    _self_stream_delivery_observer: Callable[[OutputRoutingDecision], None] | None = None
    _batch_admission: DestinationBatchAdmission = field(default_factory=DestinationBatchAdmission)
    _routing_decisions: deque[OutputRoutingDecision] = field(init=False)
    _diagnostic_history_evictions: int = field(init=False, default=0)
    _observer_failure_count: int = field(init=False, default=0)
    _last_observer_failure_type: str | None = field(init=False, default=None)
    _observer_failure_diagnostics: ErrorDiagnostics | None = field(init=False, default=None)
    _self_streams: OrderedDict[UUID, _SelfStreamDelivery] = field(default_factory=OrderedDict)
    _self_stream_completed: OrderedDict[UUID, None] = field(default_factory=OrderedDict)
    _self_stream_flush_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _self_stream_flush_task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False)

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
        bind_receipts = getattr(self.chatbox, "bind_delivery_observer", None)
        if bind_receipts is not None:
            bind_receipts(self._observe_self_stream_delivery)

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
            or self._peer_overlay_worker is not None
            or bool(self._peer_overlay_batches)
            or bool(self._active_delivery_tasks)
            or (self._ui_delivery is not None and self._ui_delivery.has_resources)
            or bool(self._batch_admission.batches)
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
        return self._batch_admission.snapshot()

    async def admit_translation_parent(
        self,
        *,
        parent_id: str,
        channel: ChannelId,
        origin: str,
        turn_generation: int,
        turn_order: int,
        retained_payloads: Iterable[str],
        destination_targets: Mapping[OutputDestination, frozenset[int]],
    ) -> frozenset[OutputDestination]:
        if self._state != "open":
            return frozenset()
        payloads = self._batch_admission.normalized_payloads(retained_payloads)
        async with self._batch_admission.lock:
            if self._state != "open":
                return frozenset()
            return self._batch_admission.admit_parent(
                parent_id=parent_id,
                channel=channel,
                origin=origin,
                turn_generation=turn_generation,
                turn_order=turn_order,
                payloads=payloads,
                destination_targets=destination_targets,
            )

    async def resize_translation_parent_output(
        self,
        *,
        parent_id: str,
        origin: str,
        retained_payloads: Iterable[str],
        destination_indexes: Mapping[OutputDestination, int],
    ) -> frozenset[OutputDestination]:
        payloads = self._batch_admission.normalized_payloads(retained_payloads)
        async with self._batch_admission.lock:
            return self._batch_admission.resize_parent(
                parent_id=parent_id,
                origin=origin,
                payloads=payloads,
                destinations=destination_indexes,
            )

    async def await_translation_parent(
        self,
        *,
        parent_id: str,
        origin: str,
        destinations: Iterable[OutputDestination] | None = None,
    ) -> frozenset[OutputDestination]:
        selected_destinations = frozenset(destinations) if destinations is not None else None
        async with self._batch_admission.lock:
            batches = self._batch_admission.parent_batches(
                parent_id=parent_id,
                origin=origin,
                destinations=selected_destinations,
            )
        admitted: set[OutputDestination] = set()
        for batch in batches:
            await batch.ready.wait()
            if batch.disposition is None:
                admitted.add(batch.destination)
        return frozenset(admitted)

    async def complete_translation_parent_target(
        self,
        *,
        parent_id: str,
        origin: str,
        destination_indexes: Mapping[OutputDestination, int],
        destinations: Iterable[OutputDestination] | None = None,
    ) -> None:
        selected_destinations = frozenset(destinations) if destinations is not None else None
        async with self._batch_admission.lock:
            self._batch_admission.complete_target(
                parent_id=parent_id,
                origin=origin,
                destination_indexes=destination_indexes,
                destinations=selected_destinations,
            )

    @classmethod
    def retained_payload_bytes(cls, values: Iterable[str]) -> int:
        payloads = DestinationBatchAdmission.normalized_payloads(values)
        return DestinationBatchAdmission.payload_bytes(payloads.values())

    @property
    def has_overlay_destination(self) -> bool:
        return self.overlay_sink is not None

    async def wait_for_peer_output_idle(self) -> None:
        while True:
            worker = self._peer_overlay_worker
            if worker is None:
                break
            await asyncio.gather(worker, return_exceptions=True)
        ui_delivery = self._ui_delivery
        if ui_delivery is not None:
            await ui_delivery.wait_for_peer_idle()

    @property
    def has_active_overlay_deliveries(self) -> bool:
        return bool(self._active_delivery_tasks) or self._peer_overlay_worker is not None

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
        async with self._batch_admission.lock:
            if self._state != "open":
                raise RuntimeError("OutputRuntime is not accepting overlay destination replacement")
            if require_match and self.overlay_sink is not expected_current:
                return False
            if self.overlay_sink is overlay_sink:
                return True
            await self._cancel_active_delivery_tasks_locked(replacement=True)
            self.overlay_sink = overlay_sink
            return True

    def bind_ui_delivery(self, delivery: UiDeliveryLifecyclePort) -> None:
        if self._ui_delivery is not None and self._ui_delivery is not delivery:
            raise RuntimeError("peer UI delivery owner is already bound")
        self._ui_delivery = delivery

    def bind_peer_overlay_delivery_observer(
        self,
        observer: Callable[[OutputRoutingDecision], None],
    ) -> None:
        current = self._peer_overlay_delivery_observer
        if current is not None and current is not observer:
            raise RuntimeError("peer overlay delivery observer is already bound")
        self._peer_overlay_delivery_observer = observer

    def bind_self_stream_delivery_observer(
        self, observer: Callable[[OutputRoutingDecision], None]
    ) -> None:
        self._self_stream_delivery_observer = observer

    @staticmethod
    def chatbox_is_eligible(channel: ChannelId) -> bool:
        return channel == "self"

    @staticmethod
    def chatbox_is_denied(channel: ChannelId) -> bool:
        return channel == "peer"

    def activate_peer_generation(self, generation: int) -> None:
        if generation <= self._retired_peer_generation or generation == self._peer_generation:
            return
        self._peer_generation = generation
        self._latest_peer_source_order = -1
        ui_delivery = self._ui_delivery
        if ui_delivery is not None:
            ui_delivery.activate_peer_generation(generation)

    def retire_peer_generation(self, generation: int) -> None:
        self._retired_peer_generation = max(self._retired_peer_generation, generation)
        if self._peer_generation != generation:
            return
        self._peer_generation = None
        worker = self._peer_overlay_worker
        if worker is not None and not worker.done():
            worker.cancel()
        for batch in tuple(self._peer_overlay_batches):
            self._reject_peer_batch(batch, reason="publication_generation_retired")
        self._peer_overlay_batches.clear()
        ui_delivery = self._ui_delivery
        if ui_delivery is not None:
            ui_delivery.retire_peer_generation(generation)

    def peer_publication_is_authorized(
        self,
        publication_generation: int | None,
        source_order: int | None,
    ) -> bool:
        return (
            publication_generation is not None
            and source_order is not None
            and publication_generation == self._peer_generation
        )

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

    @property
    def diagnostic_delivery_health(self) -> dict[str, object]:
        return {
            "owner_history": "retained",
            "owner_history_capacity": self.diagnostics_capacity,
            "owner_history_evictions": self._diagnostic_history_evictions,
            "observer": (
                "unavailable"
                if self.routing_observer is None
                else "degraded" if self._observer_failure_count else "available"
            ),
            "observer_failure_count": self._observer_failure_count,
            "observer_failure_diagnostic_id": (
                self._observer_failure_diagnostics.diagnostic_id
                if self._observer_failure_diagnostics is not None
                else None
            ),
            "last_observer_failure_type": self._last_observer_failure_type,
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
        if self._ui_event_bridge is not None and self._ui_event_bridge is not bridge:
            delivery = self._ui_delivery
            if delivery is not None:
                delivery.retire_destination()
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
        ui_delivery = self._ui_delivery
        if ui_delivery is not None:
            try:
                await ui_delivery.close()
            except Exception as exc:
                failures.append(exc)
                self._emit_cleanup_failure("ui_delivery_close", exc)
        self._clear_chatbox_typing_reasons(failures)
        self._drop_chatbox_backlog(failures)
        await self._cancel_ui_event_bridge_started_wait_task(failures)
        await self._cancel_ui_event_bridge_task(failures)
        await self._close_ui_event_bridge_adapter(failures)
        self._drain_completed_task_failures(failures)
        if failures:
            _raise_output_runtime_failures(failures)
        self._state = "closed"

    async def publish_self_turn(
        self, snapshot: SelfTurnPresentation, configuration: TranslationRuntimeConfig
    ) -> None:
        if snapshot.utterance_id in self._self_stream_completed:
            return
        state = self._self_streams.get(snapshot.utterance_id)
        if state is None:
            state = _SelfStreamDelivery(snapshot, configuration)
            self._self_streams[snapshot.utterance_id] = state
            async with self._batch_admission.lock:
                self._batch_admission.admit_parent(
                    parent_id=str(snapshot.utterance_id),
                    channel="self",
                    origin="self",
                    turn_generation=snapshot.turn_generation,
                    turn_order=snapshot.turn_order,
                    payloads=self._batch_admission.normalized_payloads((snapshot.source_text,)),
                    destination_targets={"chatbox": frozenset({0})},
                )
                state.batch = self._batch_admission.batches.get(
                    ("self:chatbox", str(snapshot.utterance_id))
                )
            if state.batch is None:
                state.settled = True
        elif snapshot.revision <= state.snapshot.revision:
            return
        else:
            state.snapshot = snapshot
            state.configuration = configuration
        if snapshot.status == "cancelled":
            state.settled = True
        if state.batch is not None and state.batch.disposition is None:
            payloads = (
                snapshot.source_text,
                *(
                    target.translation.text
                    for target in snapshot.targets
                    if target.translation is not None
                ),
            )
            async with self._batch_admission.lock:
                self._batch_admission.resize_parent(
                    parent_id=str(snapshot.utterance_id),
                    origin="self",
                    payloads=self._batch_admission.normalized_payloads(payloads),
                    destinations=("chatbox",),
                    replace_payloads=True,
                )
        await self._flush_self_streams()

    def _observe_self_stream_delivery(self, message: OSCMessage, status: str) -> None:
        state = self._self_streams.get(message.utterance_id)
        if state is None or message.presentation_revision != state.offered_revision:
            return
        state.receipt = status
        if status == "first_page_sent":
            state.sent_fingerprint = state.offered_fingerprint
        result = self._observe_result(
            status=(
                OUTPUT_ROUTING_DECISION_PUBLISHED
                if status == "first_page_sent"
                else OUTPUT_ROUTING_DECISION_SKIPPED
            ),
            route=OUTPUT_ROUTE_SELF_CHATBOX,
            publication_id=str(message.utterance_id),
            publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
            reason=status,
            metadata={
                "stage": "self_stream_delivery",
                "physical_ack": False,
                "presentation_revision": message.presentation_revision,
                "source_revision": state.offered_source_revision,
                "turn_generation": message.turn_generation,
                "turn_order": message.turn_order,
            },
        )
        if self._self_stream_delivery_observer is not None:
            self._self_stream_delivery_observer(result.decision)
        if self._state == "open" and (
            self._self_stream_flush_task is None or self._self_stream_flush_task.done()
        ):
            task = self._create_task(self._flush_self_streams(), task_name="self-stream-delivery")
            self._self_stream_flush_task = task
            self._active_delivery_tasks.add(task)
            task.add_done_callback(self._active_delivery_tasks.discard)

    async def _flush_self_streams(self) -> None:
        async with self._self_stream_flush_lock:
            blocked = False
            ordered = sorted(
                self._self_streams.values(),
                key=lambda state: (state.snapshot.turn_generation, state.snapshot.turn_order),
            )
            for state in ordered:
                snapshot = state.snapshot
                if state.batch is not None and state.batch.disposition is not None:
                    state.settled = True
                if state.settled:
                    continue
                ready = tuple(
                    (index, target)
                    for index, target in enumerate(snapshot.targets)
                    if target.status == "ready"
                    and target.translation is not None
                    and target.translation.text.strip()
                    and target.source_revision == snapshot.source_revision
                )
                terminal = snapshot.status != "active"
                source_only = (
                    terminal
                    and not ready
                    and any(
                        target.status == "source_only"
                        or (
                            target.status == "failed"
                            and len(snapshot.targets) == 1
                            and state.configuration.fallback_transcript_only
                        )
                        or (
                            target.status == "ready"
                            and target.translation is not None
                            and not target.translation.text.strip()
                        )
                        for target in snapshot.targets
                    )
                )
                if terminal and not ready and not source_only:
                    state.settled = True
                    continue
                if not blocked and (ready or source_only):
                    translation_text = (
                        "\n".join(
                            target.translation.text
                            for _, target in ready
                            if target.translation is not None
                        )
                        if ready
                        else None
                    )
                    indexes = tuple(index for index, _ in ready)
                    languages = tuple(target.target_language for _, target in ready)
                    payload = self._merge_chatbox_text(
                        transcript_text=snapshot.source_text,
                        translation_text=translation_text,
                        include_source=(
                            state.configuration.chatbox_include_source
                            and len(snapshot.targets) == 1
                        ),
                    )
                    fingerprint = (payload, snapshot.source_text, indexes, languages)
                    if (
                        terminal
                        and state.offered_fingerprint == fingerprint
                        and state.promotion_revision != snapshot.revision
                        and (state.sent_fingerprint == fingerprint or state.receipt is None)
                    ):
                        promote = getattr(self.chatbox, "promote_self_turn", None)
                        if promote is not None:
                            promote(snapshot.utterance_id, state.offered_revision)
                        state.promotion_revision = snapshot.revision
                        result = self._observe_result(
                            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
                            route=OUTPUT_ROUTE_SELF_CHATBOX,
                            publication_id=str(snapshot.utterance_id),
                            publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
                            reason="status_promoted",
                            metadata={
                                "stage": "self_turn_status_promoted",
                                "physical_ack": False,
                                "reused_preview_send": state.sent_fingerprint == fingerprint,
                                "queued_payload": state.receipt is None,
                                "presentation_revision": snapshot.revision,
                                "preview_revision": state.offered_revision,
                                "source_revision": snapshot.source_revision,
                            },
                        )
                        if self._self_stream_delivery_observer is not None:
                            self._self_stream_delivery_observer(result.decision)
                    if state.sent_fingerprint == fingerprint:
                        if terminal:
                            state.settled = True
                    elif state.offered_revision != snapshot.revision:
                        same_pending = (
                            state.offered_fingerprint == fingerprint and state.receipt is None
                        )
                        if not same_pending:
                            state.offered_revision = snapshot.revision
                            state.offered_source_revision = snapshot.source_revision
                            state.offered_fingerprint = fingerprint
                            state.receipt = None
                            result = await self.publish_chatbox(
                                publication_id=snapshot.utterance_id,
                                channel="self",
                                transcript_text=snapshot.source_text,
                                translation_text=translation_text,
                                include_source=(
                                    state.configuration.chatbox_include_source
                                    and len(snapshot.targets) == 1
                                ),
                                presentation_revision=snapshot.revision,
                                turn_generation=snapshot.turn_generation,
                                turn_order=snapshot.turn_order,
                                target_indexes=indexes,
                                target_languages=languages,
                                self_speech=True,
                                self_turn_final=terminal,
                            )
                            if result.decision.decision != OUTPUT_ROUTING_DECISION_PUBLISHED:
                                state.receipt = "destination_publish_failed"
                            elif not hasattr(self.chatbox, "bind_delivery_observer"):
                                state.receipt = "destination_receipt_unavailable"
                        if terminal and state.receipt is not None:
                            state.settled = True
                    elif terminal and state.receipt is not None:
                        state.settled = True
                if snapshot.source_closed and not state.settled:
                    blocked = True
            for state in ordered:
                if state.settled and state.batch is not None and state.batch.disposition is None:
                    async with self._batch_admission.lock:
                        self._batch_admission.release(state.batch, state.receipt or "applied")
                if state.settled and state.snapshot.status != "active":
                    parent_id = state.snapshot.utterance_id
                    self._self_streams.pop(parent_id, None)
                    self._self_stream_completed[parent_id] = None
            while len(self._self_stream_completed) > _COMPLETED_PUBLICATION_LIMIT:
                self._self_stream_completed.popitem(last=False)

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
        self_speech: bool = False,
        self_turn_final: bool | None = None,
        operation_id: str | None = None,
        request_id: str | None = None,
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
            self_speech=self_speech,
            operation_id=operation_id,
            request_id=request_id,
        )
        publication_kind = publication_kind or (
            PUBLICATION_KIND_PEER_SUBTITLE if channel == "peer" else PUBLICATION_KIND_SELF_UTTERANCE
        )
        publication_metadata: dict[str, str | int | float | bool | None] = {
            "presentation_revision": presentation_revision,
            "operation_id": operation_id,
            "request_id": request_id,
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
            enqueue_stream = getattr(self.chatbox, "enqueue_self_turn", None)
            evicted = (
                enqueue_stream(message, final=self_turn_final)
                if self_turn_final is not None and enqueue_stream is not None
                else self.chatbox.enqueue(message)
            )
        except Exception as exc:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(publication_id),
                publication_kind=publication_kind,
                reason="destination_publish_failed",
                metadata={
                    "channel": channel,
                    "error_type": safe_exception_fields(exc)["exception_class"],
                    **safe_exception_fields(exc),
                    **publication_metadata,
                },
            )
        self._remember_delivered_publication(publication_key)
        if evicted is not None:
            self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SELF_CHATBOX,
                publication_id=str(evicted.utterance_id),
                publication_kind=PUBLICATION_KIND_SELF_UTTERANCE,
                reason="output_overload",
                metadata={"channel": "self"},
            )
        self.set_self_chatbox_typing_reason(SELF_SPEECH_TYPING_REASON, False)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SELF_CHATBOX,
            publication_id=str(publication_id),
            publication_kind=publication_kind,
            reason=None,
            metadata={
                "channel": channel,
                "stage": "paginator_handoff",
                "outcome": "accepted",
                "physical_ack": False,
                **publication_metadata,
            },
            message=message,
        )

    def set_self_chatbox_typing_reason(
        self,
        reason: str,
        active: bool,
    ) -> None:
        if self._state != "open":
            return
        try:
            self.chatbox.set_typing_reason(reason, active)
        except Exception as exc:
            self._emit_typing_failure("typing_reason", exc)

    def clear_self_chatbox_typing_reasons(
        self,
    ) -> None:
        if self._state != "open":
            return
        try:
            self.chatbox.clear_typing_reasons()
        except Exception as exc:
            self._emit_typing_failure("typing_clear", exc)

    @staticmethod
    def _emit_typing_failure(operation: str, exception: Exception) -> None:
        origin = current_diagnostic_operation()
        diagnostics = ErrorDiagnostics(
            component="output_runtime",
            operation=operation,
            code="destination_publish_failed",
            category="lifecycle",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={
                "destination": OUTPUT_ROUTE_SELF_CHATBOX,
                **safe_exception_fields(exception),
            },
            channel="self",
            operation_id=origin["operation_id"],
            request_id=origin["request_id"],
        )
        emit_failure(logger, diagnostics, disposition="degraded", stage="typing_state")

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
        if not self.activation_notice_enabled:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason="activation_notice_disabled",
                metadata={"channel": "system"},
            )
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
                metadata={
                    "channel": "system",
                    "error_type": safe_exception_fields(exc)["exception_class"],
                    **safe_exception_fields(exc),
                },
            )
        self._remember_delivered_publication(publication_key)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
            publication_id=str(disclosure_uuid),
            publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
            reason=None,
            metadata={
                "channel": "system",
                "stage": "paginator_handoff",
                "outcome": "accepted",
                "physical_ack": False,
            },
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
        if not self.activation_notice_enabled:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason="activation_notice_disabled",
                metadata={"channel": "system", "delivery": "immediate"},
            )
        safe_text = _redact_chatbox_disclosure_text(text)
        try:
            published, diagnostics = self.chatbox.send_immediate(safe_text)
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
                    "error_type": safe_exception_fields(exc)["exception_class"],
                    **safe_exception_fields(exc),
                },
            )
        if diagnostics is not None:
            origin = current_diagnostic_operation()
            diagnostics = replace(
                diagnostics,
                fields={
                    **diagnostics.fields,
                    "publication_id": str(disclosure_uuid),
                    "destination": OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                },
                operation_id=diagnostics.operation_id or origin["operation_id"],
                request_id=diagnostics.request_id or origin["request_id"],
            )
        if not published:
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
                publication_id=str(disclosure_uuid),
                publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
                reason="destination_rejected",
                metadata={"channel": "system", "delivery": "immediate"},
                diagnostics=diagnostics,
            )
        self._remember_delivered_publication(publication_key)
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SYSTEM_DISCLOSURE_CHATBOX,
            publication_id=str(disclosure_uuid),
            publication_kind=PUBLICATION_KIND_SYSTEM_DISCLOSURE,
            reason=None,
            metadata={
                "channel": "system",
                "delivery": "immediate",
                "stage": "local_send",
                "outcome": "sent",
                "physical_ack": False,
            },
        )

    async def submit_self_overlay_event(
        self,
        event: OverlayEventUnion,
        *,
        on_failure: Callable[[], None] | None = None,
    ) -> None:
        if event.channel != "self":
            raise ValueError("Self overlay handoff requires a Self event")
        admitted = asyncio.Event()

        async def deliver() -> OutputPublicationResult:
            try:
                result = await self.publish_overlay_event(event, _admission_ready=admitted)
                if (
                    result.decision.reason == "destination_publish_failed"
                    and on_failure is not None
                ):
                    on_failure()
                return result
            finally:
                admitted.set()

        task = self._create_task(deliver(), task_name="self-overlay-delivery")
        self._active_delivery_tasks.add(task)
        task.add_done_callback(self._active_delivery_tasks.discard)
        await admitted.wait()

    async def publish_overlay_event(
        self,
        event: OverlayEventUnion,
        *,
        publication_generation: int | None = None,
        source_order: int | None = None,
        _admission_ready: asyncio.Event | None = None,
    ) -> OutputPublicationResult:
        if not isinstance(event, OverlayEvent):
            raise TypeError("event must implement the overlay event contract")
        if event.channel not in {"self", "peer"}:
            raise ValueError("overlay output requires a product channel")
        if not event.event_id.strip():
            raise ValueError("overlay output requires a publication identity")
        if event.channel == "peer" and (
            publication_generation is not None
            or source_order is not None
            or self._peer_generation is not None
        ):
            return await self._submit_peer_overlay_event(
                event,
                publication_generation=publication_generation,
                source_order=source_order,
            )
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
        batch: DestinationBatch | None = None
        overlay_sink: OverlaySink | None = None

        async with self._batch_admission.lock:
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
            payloads = self._batch_admission.normalized_payloads(
                self._overlay_event_payloads(event)
            )
            batch, rejection_reason = self._batch_admission.register_overlay_event(
                event,
                payloads=payloads,
            )
            if rejection_reason is not None:
                return self._overlay_rejection_result(
                    event,
                    publication_kind=publication_kind,
                    reason=rejection_reason,
                )
        if _admission_ready is not None:
            _admission_ready.set()

        if batch is None:
            raise RuntimeError("overlay batch admission did not produce an owner")
        if not batch.active:
            try:
                await batch.ready.wait()
            except asyncio.CancelledError:
                async with self._batch_admission.lock:
                    self._batch_admission.cancel_waiting(batch)
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
        async with self._batch_admission.lock:
            if self._state != "open" or self.overlay_sink is not overlay_sink:
                self._batch_admission.release(batch, "destination_replaced")
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
                async with self._batch_admission.lock:
                    self._batch_admission.release(batch, "cancelled_local")
                raise
        except Exception as exc:
            if (
                self._state == "open"
                and current_task not in self._replacement_cancelled_delivery_tasks
            ):
                async with self._batch_admission.lock:
                    self._batch_admission.release(batch, "destination_publish_failed")
            return self._observe_result(
                status=OUTPUT_ROUTING_DECISION_SKIPPED,
                route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason="destination_publish_failed",
                metadata={
                    "channel": channel,
                    "error_type": safe_exception_fields(exc)["exception_class"],
                    **safe_exception_fields(exc),
                    **self._event_identity_metadata(event),
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
        async with self._batch_admission.lock:
            self._remember_delivered_publication(
                publication_key,
                scope=publication_scope,
                sequence=event.seq,
            )
            if resolved_receipt.outcome != "applied":
                self._batch_admission.release(
                    batch,
                    resolved_receipt.cause or resolved_receipt.outcome,
                )
            elif not batch.managed_parent:
                self._batch_admission.release(batch, "applied")
            elif isinstance(event, UtteranceClosed):
                batch.completed_targets.add(event.target_index)
                if len(batch.completed_targets) >= batch.target_count:
                    self._batch_admission.release(batch, "applied")

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
                "physical_ack": False,
            },
        )

    async def _submit_peer_overlay_event(
        self,
        event: OverlayEventUnion,
        *,
        publication_generation: int | None,
        source_order: int | None,
    ) -> OutputPublicationResult:
        publication_id = event.event_id
        publication_scope = self._overlay_publication_scope(event)
        publication_key = (
            OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            f"{publication_scope}:{publication_id}",
        )
        parent_id = event.utterance_id
        async with self._batch_admission.lock:
            if self._state != "open":
                return self._closed_overlay_result(event)
            if self.overlay_sink is None:
                return self._unconfigured_overlay_result(event)
            if parent_id is None:
                return self._overlay_failure_result(event, "missing_peer_publication_identity")
            identity = self._resolve_peer_publication_identity(
                parent_id,
                publication_generation=publication_generation,
                source_order=source_order,
            )
            if identity is None:
                return self._overlay_failure_result(event, "missing_peer_publication_identity")
            generation, order = identity
            if generation != self._peer_generation:
                return self._overlay_failure_result(event, "publication_generation_retired")
            if order < self._latest_peer_source_order:
                return self._overlay_failure_result(event, "stale_source_order")
            if (
                event.turn_generation is None
                and event.sequence_namespace not in self._adapter_sequence_namespaces
            ):
                if len(self._adapter_sequence_namespaces) >= _COMPLETED_PUBLICATION_LIMIT:
                    return self._overlay_rejection_result(
                        event,
                        publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                        reason="output_identity_capacity_exhausted",
                    )
                self._adapter_sequence_namespaces.add(event.sequence_namespace)
            duplicate = self._duplicate_publication_result(
                publication_key=publication_key,
                publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                channel="peer",
                logical_publication_id=publication_id,
                metadata={
                    "publication_generation": generation,
                    "source_order": order,
                },
                scope=publication_scope,
            )
            if duplicate is not None:
                return duplicate
            payloads = self._batch_admission.normalized_payloads(
                self._overlay_event_payloads(event)
            )
            destination_batch, rejection_reason = self._batch_admission.register_overlay_event(
                event,
                payloads=payloads,
            )
            if rejection_reason is not None:
                return self._overlay_rejection_result(
                    event,
                    publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                    reason=rejection_reason,
                )
            if destination_batch is None:
                raise RuntimeError("overlay batch admission did not produce an owner")
            self._remember_peer_identity(parent_id, generation, order)
            self._latest_peer_source_order = max(self._latest_peer_source_order, order)
            batch = self._find_peer_batch(generation, order, parent_id)
            if batch is None:
                if len(self._peer_overlay_batches) >= OUTPUT_BATCH_MAX_UNSENT:
                    self._reject_peer_batch(
                        self._peer_overlay_batches.popleft(),
                        reason="output_overload",
                    )
                batch = _PeerOverlayBatch(generation, order, parent_id)
                self._peer_overlay_batches.append(batch)
            batch.events.append((event, publication_key, publication_scope, destination_batch))
            self._publications_in_flight.add(publication_key)
            worker = self._peer_overlay_worker
            if worker is None or worker.done():
                self._peer_overlay_worker = asyncio.create_task(
                    self._run_peer_overlay_writer(),
                    name="OutputRuntime:peer-overlay-writer",
                )
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            publication_id=publication_id,
            publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
            reason="accepted_handoff",
            metadata={
                "channel": "peer",
                "publication_generation": generation,
                "source_order": order,
                "accepted_handoff": True,
                "physical_ack": False,
                "stage": "admission_accepted",
                "outcome": "pending",
                "pending_batches": len(self._peer_overlay_batches),
            },
        )

    async def _run_peer_overlay_writer(self) -> None:
        try:
            while self._peer_overlay_batches:
                batch = self._peer_overlay_batches.popleft()
                self._peer_active_batch = batch
                index = 0
                while index < len(batch.events):
                    event, publication_key, publication_scope, destination_batch = batch.events[
                        index
                    ]
                    index += 1
                    if batch.publication_generation != self._peer_generation:
                        self._reject_peer_event(
                            event,
                            publication_key,
                            publication_scope,
                            destination_batch,
                            reason="publication_generation_retired",
                        )
                        continue
                    if not destination_batch.active:
                        try:
                            await destination_batch.ready.wait()
                        except asyncio.CancelledError:
                            cancel_reason = self._peer_writer_cancel_reason or (
                                "publication_generation_retired"
                                if batch.publication_generation != self._peer_generation
                                else "output_runtime_closing"
                            )
                            self._reject_peer_event(
                                event,
                                publication_key,
                                publication_scope,
                                destination_batch,
                                reason=cancel_reason,
                            )
                            for (
                                pending_event,
                                pending_key,
                                pending_scope,
                                pending_batch,
                            ) in batch.events[index:]:
                                self._reject_peer_event(
                                    pending_event,
                                    pending_key,
                                    pending_scope,
                                    pending_batch,
                                    reason=cancel_reason,
                                )
                            raise
                        if destination_batch.disposition is not None:
                            self._reject_peer_event(
                                event,
                                publication_key,
                                publication_scope,
                                destination_batch,
                                reason=destination_batch.disposition,
                            )
                            continue
                    sink = self.overlay_sink
                    if sink is None:
                        self._reject_peer_event(
                            event,
                            publication_key,
                            publication_scope,
                            destination_batch,
                            reason="destination_unconfigured",
                        )
                        continue
                    paced_emit = getattr(sink, "emit_peer_when_admissible", None)
                    receipt: OverlayApplicationReceipt | None = None
                    pacing_started_at: float | None = None
                    pacing_wait_reason: str | None = None

                    def observe_pacing_wait(wait_reason: str) -> None:
                        nonlocal pacing_started_at, pacing_wait_reason
                        if pacing_started_at is not None or wait_reason not in {
                            "replacement_gate",
                            "protected_rows",
                        }:
                            return
                        pacing_started_at = self.clock.now()
                        pacing_wait_reason = wait_reason
                        self._observe_decision(
                            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
                            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
                            publication_id=event.event_id,
                            publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
                            reason="logical_pacing_wait",
                            metadata={
                                "channel": "peer",
                                "publication_generation": batch.publication_generation,
                                "source_order": batch.source_order,
                                "physical_ack": False,
                                "stage": "logical_pacing",
                                "outcome": "waiting",
                                "pending_batches": len(self._peer_overlay_batches),
                                "wait_reason": wait_reason,
                            },
                        )

                    def pacing_outcome() -> dict[str, str | int | float | bool | None]:
                        metadata: dict[str, str | int | float | bool | None] = {
                            "pending_batches": len(self._peer_overlay_batches)
                        }
                        if pacing_started_at is not None:
                            metadata["handoff_wait_ms"] = int(
                                max(0.0, self.clock.now() - pacing_started_at) * 1000
                            )
                        if pacing_wait_reason is not None:
                            metadata["wait_reason"] = pacing_wait_reason
                        return metadata

                    try:
                        if callable(paced_emit):
                            try:
                                supports_wait_observer = (
                                    "on_wait" in inspect.signature(paced_emit).parameters
                                )
                            except TypeError, ValueError:
                                supports_wait_observer = False
                            if supports_wait_observer:
                                receipt = await paced_emit(event, on_wait=observe_pacing_wait)
                            else:
                                receipt = await paced_emit(event)
                        else:
                            receipt = await asyncio.wait_for(sink.emit(event), timeout=5.0)
                    except TimeoutError:
                        self._reject_peer_event(
                            event,
                            publication_key,
                            publication_scope,
                            destination_batch,
                            reason="destination_write_timeout",
                            pacing_metadata=pacing_outcome(),
                        )
                    except asyncio.CancelledError:
                        cancel_reason = self._peer_writer_cancel_reason or (
                            "publication_generation_retired"
                            if batch.publication_generation != self._peer_generation
                            else "output_runtime_closing"
                        )
                        receipt_lookup = getattr(sink, "application_receipt", None)
                        if callable(receipt_lookup):
                            receipt = receipt_lookup(event.event_id)
                        if receipt is not None and receipt.outcome == "applied":
                            self._complete_peer_event(
                                event,
                                publication_key,
                                publication_scope,
                                destination_batch,
                                batch,
                                receipt,
                                pacing_metadata=pacing_outcome(),
                            )
                        else:
                            self._reject_peer_event(
                                event,
                                publication_key,
                                publication_scope,
                                destination_batch,
                                reason=cancel_reason,
                                pacing_metadata=pacing_outcome(),
                            )
                        for (
                            pending_event,
                            pending_key,
                            pending_scope,
                            pending_batch,
                        ) in batch.events[index:]:
                            self._reject_peer_event(
                                pending_event,
                                pending_key,
                                pending_scope,
                                pending_batch,
                                reason=cancel_reason,
                            )
                        raise
                    except Exception as exc:
                        self._reject_peer_event(
                            event,
                            publication_key,
                            publication_scope,
                            destination_batch,
                            reason="destination_publish_failed",
                            error_type=safe_exception_fields(exc)["exception_class"],
                            pacing_metadata={**pacing_outcome(), **safe_exception_fields(exc)},
                        )
                    else:
                        resolved_receipt = receipt or OverlayApplicationReceipt(
                            stage="application_accepted",
                            outcome="applied",
                            publication_id=event.event_id,
                            scene_revision=None,
                        )
                        if resolved_receipt.outcome == "applied":
                            self._complete_peer_event(
                                event,
                                publication_key,
                                publication_scope,
                                destination_batch,
                                batch,
                                resolved_receipt,
                                pacing_metadata=pacing_outcome(),
                            )
                        else:
                            self._reject_peer_event(
                                event,
                                publication_key,
                                publication_scope,
                                destination_batch,
                                reason=resolved_receipt.cause or resolved_receipt.outcome,
                                scene_revision=resolved_receipt.scene_revision,
                                pacing_metadata=pacing_outcome(),
                            )
                self._peer_active_batch = None
        finally:
            self._peer_active_batch = None
            if self._peer_overlay_worker is asyncio.current_task():
                self._peer_overlay_worker = None

    def _complete_peer_event(
        self,
        event: OverlayEventUnion,
        publication_key: tuple[OutputRoute, str],
        publication_scope: str,
        destination_batch: DestinationBatch,
        peer_batch: _PeerOverlayBatch,
        receipt: OverlayApplicationReceipt,
        *,
        pacing_metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self._publications_in_flight.discard(publication_key)
        self._remember_delivered_publication(
            publication_key,
            scope=publication_scope,
        )
        if not destination_batch.managed_parent:
            self._batch_admission.release(destination_batch, "applied")
        elif isinstance(event, UtteranceClosed):
            destination_batch.completed_targets.add(event.target_index)
            if len(destination_batch.completed_targets) >= destination_batch.target_count:
                self._batch_admission.release(destination_batch, "applied")
        decision = self._observe_decision(
            status=OUTPUT_ROUTING_DECISION_PUBLISHED,
            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            publication_id=event.event_id,
            publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
            reason="application_applied",
            metadata={
                "channel": "peer",
                "utterance_id": str(event.utterance_id),
                "event_type": event.EVENT_TYPE,
                "publication_generation": peer_batch.publication_generation,
                "source_order": peer_batch.source_order,
                "accepted_handoff": True,
                "physical_ack": False,
                "stage": receipt.stage,
                "outcome": receipt.outcome,
                "scene_revision": receipt.scene_revision,
                **(pacing_metadata or {}),
            },
        )
        observer = self._peer_overlay_delivery_observer
        if observer is not None:
            observer(decision)

    def _resolve_peer_publication_identity(
        self,
        parent_id: UUID,
        *,
        publication_generation: int | None,
        source_order: int | None,
    ) -> tuple[int, int] | None:
        if publication_generation is not None and source_order is not None:
            return publication_generation, source_order
        if publication_generation is not None or source_order is not None:
            return None
        return self._peer_identity_by_utterance.get(parent_id)

    def _remember_peer_identity(self, parent_id: UUID, generation: int, order: int) -> None:
        existing = self._peer_identity_by_utterance.get(parent_id)
        if existing is not None:
            return
        self._peer_identity_by_utterance[parent_id] = (generation, order)
        self._peer_identity_order.append(parent_id)
        while len(self._peer_identity_order) > _COMPLETED_PUBLICATION_LIMIT:
            evicted = self._peer_identity_order.popleft()
            self._peer_identity_by_utterance.pop(evicted, None)

    def _find_peer_batch(
        self,
        generation: int,
        order: int,
        parent_id: UUID,
    ) -> _PeerOverlayBatch | None:
        active = self._peer_active_batch
        if active is not None and (
            active.publication_generation,
            active.source_order,
            active.parent_id,
        ) == (generation, order, parent_id):
            return active
        return next(
            (
                batch
                for batch in self._peer_overlay_batches
                if (
                    batch.publication_generation,
                    batch.source_order,
                    batch.parent_id,
                )
                == (generation, order, parent_id)
            ),
            None,
        )

    def _reject_peer_batch(self, batch: _PeerOverlayBatch, *, reason: str) -> None:
        for event, publication_key, publication_scope, destination_batch in batch.events:
            if publication_key not in self._publications_in_flight:
                continue
            self._reject_peer_event(
                event,
                publication_key,
                publication_scope,
                destination_batch,
                reason=reason,
            )

    def _reject_peer_event(
        self,
        event: OverlayEventUnion,
        publication_key: tuple[OutputRoute, str],
        publication_scope: str,
        destination_batch: DestinationBatch,
        *,
        reason: str,
        error_type: str | None = None,
        scene_revision: int | None = None,
        pacing_metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self._publications_in_flight.discard(publication_key)
        self._remember_delivered_publication(
            publication_key,
            scope=publication_scope,
        )
        self._batch_admission.release(destination_batch, reason)
        metadata: dict[str, str | int | float | bool | None] = {
            "channel": "peer",
            **self._event_identity_metadata(event),
            "utterance_id": str(event.utterance_id),
            "event_type": event.EVENT_TYPE,
            "accepted_handoff": True,
            "physical_ack": False,
            "stage": "application_accepted",
            "outcome": "not_applied",
            "scene_revision": scene_revision,
        }
        if pacing_metadata is not None:
            metadata.update(pacing_metadata)
        if error_type is not None:
            metadata["error_type"] = error_type
        decision = self._observe_decision(
            status=OUTPUT_ROUTING_DECISION_SKIPPED,
            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            publication_id=event.event_id,
            publication_kind=PUBLICATION_KIND_PEER_SUBTITLE,
            reason=reason,
            metadata=metadata,
        )
        observer = self._peer_overlay_delivery_observer
        if observer is not None:
            observer(decision)

    def _closed_overlay_result(self, event: OverlayEventUnion) -> OutputPublicationResult:
        return self._overlay_failure_result(
            event,
            "output_runtime_closed" if self._state == "closed" else "output_runtime_closing",
        )

    def _unconfigured_overlay_result(self, event: OverlayEventUnion) -> OutputPublicationResult:
        return self._overlay_failure_result(event, "destination_unconfigured")

    def _overlay_failure_result(
        self,
        event: OverlayEventUnion,
        reason: str,
        *,
        error_type: str | None = None,
    ) -> OutputPublicationResult:
        metadata: dict[str, str | int | float | bool | None] = {
            "channel": event.channel,
            **self._event_identity_metadata(event),
        }
        if error_type is not None:
            metadata["error_type"] = error_type
        return self._observe_result(
            status=OUTPUT_ROUTING_DECISION_SKIPPED,
            route=OUTPUT_ROUTE_SUBTITLE_OVERLAY,
            publication_id=event.event_id,
            publication_kind=(
                PUBLICATION_KIND_PEER_SUBTITLE
                if event.channel == "peer"
                else PUBLICATION_KIND_SELF_UTTERANCE
            ),
            reason=reason,
            metadata=metadata,
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

    def retire_turn_generation(self, channel: ChannelId, turn_generation: int) -> None:
        self._batch_admission.retire_turn_generation(channel, turn_generation)
        if channel == "self" and self._ui_delivery is not None:
            self._ui_delivery.retire_self_generation(turn_generation)
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
                **self._event_identity_metadata(event),
                "stage": "application_accepted",
                "outcome": "not_applied",
                "scene_revision": scene_revision,
            },
        )

    @classmethod
    def _overlay_publication_scope(cls, event: OverlayEventUnion) -> str:
        scope = DestinationBatchAdmission.overlay_batch_scope(event)
        if event.turn_generation is None:
            return f"{scope}:overlay-adapter:{event.sequence_namespace}"
        return f"{scope}:overlay:{event.turn_generation}"

    @staticmethod
    def _event_identity_metadata(event: OverlayEventUnion) -> dict[str, str | int | None]:
        return {
            "operation_id": event.operation_id,
            "request_id": event.request_id,
            "utterance_id": (
                str(event.parent_utterance_id or event.utterance_id)
                if event.utterance_id is not None
                else None
            ),
            "turn_generation": event.turn_generation,
        }

    @staticmethod
    def _overlay_event_payloads(event: OverlayEventUnion) -> tuple[str, ...]:
        values: list[str] = []
        for field_name in (
            "text",
            "source_text",
            "secondary_text",
            "source_language",
            "target_language",
        ):
            value = getattr(event, field_name, None)
            if isinstance(value, str) and value:
                values.append(value)
        return tuple(values)

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
        async with self._batch_admission.lock:
            await self._cancel_active_delivery_tasks_locked(replacement=False)

    async def _cancel_active_delivery_tasks_locked(self, *, replacement: bool) -> None:
        peer_worker = self._peer_overlay_worker
        self._peer_writer_cancel_reason = (
            "destination_replaced" if replacement else "output_runtime_closing"
        )
        active_peer_batch = self._peer_active_batch
        if peer_worker is not None and not peer_worker.done():
            peer_worker.cancel()
            await asyncio.gather(peer_worker, return_exceptions=True)
        if active_peer_batch is not None:
            self._reject_peer_batch(
                active_peer_batch,
                reason=self._peer_writer_cancel_reason or "output_runtime_closing",
            )
        self._peer_writer_cancel_reason = None
        self._peer_overlay_worker = None
        peer_reason = "destination_replaced" if replacement else "output_runtime_closing"
        for batch in tuple(self._peer_overlay_batches):
            self._reject_peer_batch(batch, reason=peer_reason)
        self._peer_overlay_batches.clear()
        disposition = "destination_replaced" if replacement else "output_runtime_closing"
        if replacement:
            self._batch_admission.release_destination("overlay", disposition)
        else:
            self._batch_admission.release_all(disposition)
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

    @staticmethod
    def _emit_cleanup_failure(stage: str, exception: Exception) -> None:
        emit_failure(
            logger,
            ErrorDiagnostics(
                component="output_runtime",
                operation="close",
                code="output_cleanup_failed",
                category="lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={"close_complete": False},
            ),
            stage=stage,
            exception=exception,
        )

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
            self._emit_cleanup_failure("ui_bridge_close", exc)
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
            self._emit_cleanup_failure("chatbox_backlog_close", exc)
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
            self._emit_cleanup_failure("chatbox_typing_close", exc)
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

    def record_ui_publication(
        self,
        *,
        channel: ChannelId,
        status: OutputRoutingDecisionStatus,
        publication_id: str,
        reason: str,
        publication_generation: int | None,
        source_order: int | None,
        parent_utterance_id: UUID | None,
        event_type: str,
        accepted_handoff: bool,
        ui_queue_submitted: bool,
        operation_id: str | None = None,
        request_id: str | None = None,
    ) -> OutputPublicationResult:
        return self._observe_result(
            status=status,
            route=OUTPUT_ROUTE_CONVERSATION_FEED,
            publication_id=publication_id,
            publication_kind=PUBLICATION_KIND_CONVERSATION_FEED,
            reason=reason,
            metadata={
                "channel": channel,
                "operation_id": operation_id,
                "request_id": request_id,
                "publication_generation": publication_generation,
                "source_order": source_order,
                "parent_utterance_id": (
                    str(parent_utterance_id) if parent_utterance_id is not None else None
                ),
                "event_type": event_type,
                "accepted_handoff": accepted_handoff,
                "stage": "ui_queue" if ui_queue_submitted else "admission",
                "outcome": (
                    "submitted"
                    if ui_queue_submitted
                    else "accepted" if accepted_handoff else "rejected"
                ),
                "ui_queue_submitted": ui_queue_submitted,
                "physical_ack": False,
            },
        )

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
        diagnostics: ErrorDiagnostics | None = None,
    ) -> OutputPublicationResult:
        return OutputPublicationResult(
            decision=self._observe_decision(
                status=status,
                route=route,
                publication_id=publication_id,
                publication_kind=publication_kind,
                reason=reason,
                metadata=metadata,
                diagnostics=diagnostics,
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
        diagnostics: ErrorDiagnostics | None = None,
    ) -> OutputRoutingDecision:
        if diagnostics is None and reason in {
            "destination_publish_failed",
            "destination_write_timeout",
            "output_overload",
            "output_identity_capacity_exhausted",
        }:
            fields = {
                "destination": route,
                "publication_id": metadata.get("utterance_id", publication_id),
                "publication_kind": publication_kind,
            }
            for key in (
                "exception_class",
                "cause_class_1",
                "cause_class_2",
                "cause_class_3",
                "cause_class_4",
                "errno",
                "winerror",
            ):
                if key in metadata:
                    fields[key] = metadata[key]
            if "exception_class" not in fields and "error_type" in metadata:
                fields["exception_class"] = metadata["error_type"]
            diagnostics = ErrorDiagnostics(
                component="output_runtime",
                operation="publish",
                code=reason,
                category="timeout" if reason == "destination_write_timeout" else "lifecycle",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields=fields,
                channel=(
                    metadata.get("channel") if metadata.get("channel") in {"self", "peer"} else None
                ),
                generation=metadata.get("publication_generation", metadata.get("turn_generation")),
                operation_id=metadata.get("operation_id"),
                request_id=metadata.get("request_id"),
                turn_id=metadata.get("parent_utterance_id", metadata.get("utterance_id")),
            )
        if diagnostics is not None:
            emit_failure(logger, diagnostics, disposition="degraded", stage="destination_delivery")
        decision = OutputRoutingDecision(
            decision=status,
            route=route,
            publication_id=publication_id,
            publication_kind=publication_kind,
            reason=reason,
            metadata=metadata,
            diagnostics=diagnostics,
        )
        if len(self._routing_decisions) == self.diagnostics_capacity:
            self._diagnostic_history_evictions += 1
        self._routing_decisions.append(decision)
        if self.routing_observer is not None:
            try:
                self.routing_observer(decision)
            except Exception as exc:
                self._observer_failure_count += 1
                self._last_observer_failure_type = safe_exception_fields(exc)["exception_class"]
                if self._observer_failure_count == 1:
                    self._observer_failure_diagnostics = ErrorDiagnostics(
                        component="output_runtime",
                        operation="observe",
                        code="output_observer_failed",
                        category="lifecycle",
                        visibility="diagnostic_only",
                        content_policy="metadata_only",
                        status_code=None,
                        retry_after_ms=None,
                        fields={
                            "destination": route,
                            "publication_id": metadata.get("utterance_id", publication_id),
                            "count": self._observer_failure_count,
                        },
                        cause_id=diagnostics.diagnostic_id if diagnostics else None,
                    )
                    emit_failure(
                        logger,
                        self._observer_failure_diagnostics,
                        severity="warning",
                        disposition="degraded",
                        stage="routing_observer",
                        exception=exc,
                    )
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
