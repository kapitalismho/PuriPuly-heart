from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID, uuid5

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
    TranslationRuntimeConfigSnapshotPort,
)
from puripuly_heart.core.translation_policy import (
    TranslationContextPolicy,
    TranslationRuntimePolicy,
)
from puripuly_heart.domain.models import ChannelId, FinalLanguageRun, Transcript, Translation

logger = logging.getLogger(__name__)

TranslationTurnKind = Literal["manual", "self", "peer"]
TranslationTurnOutcome = Literal["translated", "source_only", "cancelled", "failed"]


def _default_config_snapshot() -> TranslationRuntimeConfigSnapshot:
    return TranslationRuntimeConfigSnapshot(
        revision=0,
        value=TranslationRuntimeConfig(),
    )


@dataclass(frozen=True, slots=True)
class TranslationTurnRequest:
    transcript: Transcript
    source: str
    turn_kind: TranslationTurnKind
    target_languages: tuple[str, ...]
    config_snapshot: TranslationRuntimeConfigSnapshot
    precomputed_translation: Translation | None = None

    def __post_init__(self) -> None:
        if not self.transcript.is_final:
            raise ValueError("translation turn requires a final transcript")
        expected_channel: ChannelId = "peer" if self.turn_kind == "peer" else "self"
        if self.transcript.channel != expected_channel:
            raise ValueError("translation turn kind does not match transcript channel")
        normalized_targets = tuple(
            dict.fromkeys(
                language.strip() for language in self.target_languages if language.strip()
            )
        )
        if not normalized_targets:
            raise ValueError("translation turn requires at least one target language")
        if self.precomputed_translation is not None:
            if self.precomputed_translation.utterance_id != self.transcript.utterance_id:
                raise ValueError("precomputed translation identity mismatch")
            if self.precomputed_translation.channel != self.transcript.channel:
                raise ValueError("precomputed translation channel mismatch")
            if len(normalized_targets) != 1:
                raise ValueError("precomputed translation requires exactly one target language")
            nonempty_runs = tuple(
                run for run in self.transcript.final_language_runs if run.text.strip()
            )
            if len(nonempty_runs) > 1:
                raise ValueError("precomputed translation requires exactly one language run")
        object.__setattr__(self, "target_languages", normalized_targets)


@dataclass(frozen=True, slots=True)
class TranslationTurnChild:
    parent_utterance_id: UUID
    utterance_id: UUID
    sequence: int
    transcript: Transcript
    detected_language: str | None
    target_language: str
    source: str
    turn_kind: TranslationTurnKind
    context_policy: TranslationContextPolicy
    config_snapshot: TranslationRuntimeConfigSnapshot
    precomputed_translation: Translation | None = None

    @property
    def channel(self) -> ChannelId:
        return self.transcript.channel


@dataclass(frozen=True, slots=True)
class TranslationOutputSubmission:
    parent_utterance_id: UUID
    child_utterance_id: UUID
    sequence: int
    channel: ChannelId
    source: str
    source_text: str
    source_language: str | None
    target_language: str
    outcome: TranslationTurnOutcome
    config_snapshot: TranslationRuntimeConfigSnapshot
    translation: Translation | None = None
    applied_context_mode: Literal["local", "integrated"] | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if self.outcome == "translated" and self.translation is None:
            raise ValueError("translated output requires a translation")
        if self.translation is not None:
            if self.outcome != "translated":
                raise ValueError("only translated output may include a translation")
            if self.translation.utterance_id != self.child_utterance_id:
                raise ValueError("translation output child identity mismatch")
            if self.translation.channel != self.channel:
                raise ValueError("translation output channel mismatch")


@dataclass(frozen=True, slots=True)
class TranslationTurnProcessResult:
    outcome: TranslationTurnOutcome
    output: TranslationOutputSubmission | None = None

    def __post_init__(self) -> None:
        if self.output is not None and self.output.outcome != self.outcome:
            raise ValueError("translation process result outcome mismatch")


class TranslationOutputSubmissionPort(Protocol):
    async def submit_translation_output(self, submission: TranslationOutputSubmission) -> None: ...


@dataclass(slots=True)
class _TranslationTurnParent:
    parent_utterance_id: UUID
    channel: ChannelId
    children: tuple[TranslationTurnChild, ...]
    completed_child_ids: set[UUID] = field(default_factory=set)
    semantic_completed_child_ids: set[UUID] = field(default_factory=set)
    semantic_done_event: asyncio.Event = field(default_factory=asyncio.Event)
    closed_event: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False

    @property
    def child_ids(self) -> tuple[UUID, ...]:
        return tuple(child.utterance_id for child in self.children)


ChildCreated = Callable[[TranslationTurnChild], Awaitable[None]]
ChildStarted = Callable[
    [TranslationTurnChild, asyncio.Task[TranslationTurnProcessResult]], Awaitable[None]
]
ChildCancellationRequested = Callable[[], bool]
ChildProcessor = Callable[
    [TranslationTurnChild, ChildCancellationRequested],
    Awaitable[TranslationTurnProcessResult | TranslationTurnOutcome],
]
ChildTerminal = Callable[[TranslationTurnChild, TranslationTurnOutcome], Awaitable[None]]
ParentClosed = Callable[[UUID], Awaitable[None]]
ParentRejected = Callable[[UUID], Awaitable[None]]


@dataclass(slots=True)
class TranslationTurnLifecycleOwner:
    on_child_created: ChildCreated
    on_child_started: ChildStarted
    process_child: ChildProcessor
    on_child_terminal: ChildTerminal
    on_parent_closed: ParentClosed
    on_parent_rejected: ParentRejected
    predecessor_wait_observer: Callable[[str, Mapping[str, object]], None] | None = None
    output: TranslationOutputSubmissionPort | None = None
    config_snapshot: TranslationRuntimeConfigSnapshotPort = _default_config_snapshot
    policy: TranslationRuntimePolicy = field(default_factory=TranslationRuntimePolicy)
    _parents: dict[UUID, _TranslationTurnParent] = field(default_factory=dict)
    _closed_parent_ids: set[UUID] = field(default_factory=set)
    _cancelling_parent_ids: set[UUID] = field(default_factory=set)
    _parent_tasks: dict[UUID, asyncio.Task[None]] = field(default_factory=dict)
    _active_tasks: dict[UUID, asyncio.Task[TranslationTurnProcessResult]] = field(
        default_factory=dict
    )
    _channel_tails: dict[ChannelId, _TranslationTurnParent] = field(default_factory=dict)
    _scope: LifecycleScope = field(init=False)
    _blocked_channels: set[ChannelId] = field(default_factory=set)
    _accepting: bool = True
    _closed: bool = False

    def __post_init__(self) -> None:
        self._scope = LifecycleScope("translation-turns")

    @property
    def has_resources(self) -> bool:
        return bool(self._parents or self._parent_tasks or self._active_tasks)

    def is_parent_closed(self, parent_utterance_id: UUID) -> bool:
        return parent_utterance_id in self._closed_parent_ids

    def is_parent_active(self, parent_utterance_id: UUID) -> bool:
        return parent_utterance_id in self._parents

    def is_child_cancellation_requested(self, child: TranslationTurnChild) -> bool:
        return (
            self._closed
            or not self._accepting
            or child.channel in self._blocked_channels
            or child.parent_utterance_id in self._cancelling_parent_ids
        )

    def channel_ingress_open(self, channel: ChannelId) -> bool:
        return self._accepting and not self._closed and channel not in self._blocked_channels

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "TranslationTurnLifecycleOwner",
            "resource_fields": (
                "parent tasks",
                "child tasks",
                "parent/child terminal state",
            ),
            "ordering": ("same-channel LLM admission waits on predecessor semantic completion"),
            "stop_ingress": "stop accepting translation turns",
            "shutdown_policy": "cancel parent tasks, terminalize unfinished children, await scope",
            "late_callback_rule": "closed parents reject child completion and output submission",
            "fast_translation_enabled": self.policy.fast_translation_enabled,
            "context_policy": self.policy.context_policy,
        }

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("TranslationTurnLifecycleOwner is closed")

    async def open_channel_ingress(self, channel: ChannelId) -> None:
        if self._closed:
            raise RuntimeError("TranslationTurnLifecycleOwner is closed")
        self._blocked_channels.discard(channel)

    async def close_channel_ingress(self, channel: ChannelId) -> None:
        self._blocked_channels.add(channel)
        await self._cancel_channel(channel)
        self._blocked_channels.add(channel)

    async def submit(
        self,
        request: TranslationTurnRequest,
        *,
        wait_for_parent: bool = False,
    ) -> tuple[UUID, ...]:
        parent_id = request.transcript.utterance_id
        if parent_id in self._closed_parent_ids or parent_id in self._parents:
            await self._reject_parent(parent_id)
            return ()
        if (
            not self._accepting
            or self._closed
            or request.transcript.channel in self._blocked_channels
        ):
            await self._reject_parent(parent_id)
            return ()
        children = self._build_children(request)
        parent = _TranslationTurnParent(
            parent_utterance_id=parent_id,
            channel=request.transcript.channel,
            children=children,
        )
        self._parents[parent_id] = parent
        if not children:
            await self._close_parent(parent)
            return ()
        await self.start()
        for child in children:
            try:
                await self.on_child_created(child)
            except Exception:
                logger.exception("translation child creation adapter failed")
                await self._terminalize_child(child, "failed")
        if not parent.closed:
            predecessor = self._channel_tails.get(parent.channel)
            if predecessor is not None and predecessor.closed:
                predecessor = None
            self._channel_tails[parent.channel] = parent
            parent_task = start_lifecycle_task(
                self._scope,
                self._run_parent(parent, predecessor),
                name=f"parent:{parent_id}",
                eager_start=True,
            )
            if not parent_task.done():
                self._parent_tasks[parent_id] = parent_task
        if wait_for_parent:
            await parent.closed_event.wait()
        return parent.child_ids

    async def _reject_parent(self, parent_utterance_id: UUID) -> None:
        try:
            await self.on_parent_rejected(parent_utterance_id)
        except Exception:
            logger.exception("translation parent rejection adapter failed")

    async def submit_parent(
        self,
        transcript: Transcript,
        *,
        source: str,
        turn_kind: TranslationTurnKind | None = None,
        target_languages: tuple[str, ...] = ("und",),
    ) -> tuple[UUID, ...]:
        resolved_kind = turn_kind or ("peer" if transcript.channel == "peer" else "self")
        return await self.submit(
            TranslationTurnRequest(
                transcript=transcript,
                source=source,
                turn_kind=resolved_kind,
                target_languages=target_languages,
                config_snapshot=self.config_snapshot(),
            )
        )

    async def cancel_pending(self, *, channel: ChannelId | None = None) -> None:
        if self._closed:
            return
        if channel is not None:
            await self._cancel_channel(channel)
            return
        self._accepting = False
        self._request_cancellation()
        await self._scope.close()
        await self._terminalize_unfinished_parents("cancelled")
        self._parent_tasks.clear()
        self._active_tasks.clear()
        self._channel_tails.clear()
        self._accepting = True
        self._scope = LifecycleScope("translation-turns")

    async def _cancel_channel(self, channel: ChannelId) -> None:
        selected_parents = tuple(
            parent for parent in self._parents.values() if parent.channel == channel
        )
        if not selected_parents:
            return
        self._blocked_channels.add(channel)
        try:
            self._cancelling_parent_ids.update(
                parent.parent_utterance_id for parent in selected_parents
            )
            current_task = asyncio.current_task()
            tasks_to_await: list[asyncio.Task[None]] = []
            parents_to_await: list[_TranslationTurnParent] = []
            for parent in selected_parents:
                parent_task = self._parent_tasks.get(parent.parent_utterance_id)
                if parent_task is current_task:
                    continue
                if parent_task is not None and not parent_task.done():
                    parent_task.cancel()
                    tasks_to_await.append(parent_task)
                parents_to_await.append(parent)
            if tasks_to_await:
                await asyncio.gather(*tasks_to_await, return_exceptions=True)
            await self._terminalize_selected_unfinished(selected_parents, "cancelled")
            if parents_to_await:
                await asyncio.gather(*(parent.closed_event.wait() for parent in parents_to_await))
        finally:
            self._blocked_channels.discard(channel)

    async def wait_for_idle(self) -> None:
        while self._parents or self._parent_tasks or self._active_tasks:
            await asyncio.sleep(0)

    async def wait_for_parent(self, parent_utterance_id: UUID) -> None:
        parent = self._parents.get(parent_utterance_id)
        if parent is not None:
            await parent.closed_event.wait()

    async def close(self) -> None:
        if self._closed:
            return
        self._accepting = False
        self._closed = True
        self._request_cancellation()
        await self._scope.close()
        await self._terminalize_unfinished_parents("cancelled")
        self._parent_tasks.clear()
        self._active_tasks.clear()
        self._channel_tails.clear()

    def _build_children(self, request: TranslationTurnRequest) -> tuple[TranslationTurnChild, ...]:
        runs = request.transcript.final_language_runs or (
            FinalLanguageRun(text=request.transcript.text, language=""),
        )
        child_specs = [
            (run_index, target_index, run, target_language)
            for run_index, run in enumerate(runs)
            if run.text.strip()
            for target_index, target_language in enumerate(request.target_languages)
        ]
        single_identity = len(child_specs) == 1 and request.turn_kind in {"manual", "self"}
        children: list[TranslationTurnChild] = []
        for sequence, (run_index, target_index, run, target_language) in enumerate(child_specs):
            child_id = (
                request.transcript.utterance_id
                if single_identity
                else uuid5(
                    request.transcript.utterance_id,
                    f"{request.turn_kind}:{run_index}:{target_index}:{run.language}:{target_language}",
                )
            )
            children.append(
                TranslationTurnChild(
                    parent_utterance_id=request.transcript.utterance_id,
                    utterance_id=child_id,
                    sequence=sequence,
                    transcript=Transcript(
                        utterance_id=child_id,
                        text=run.text,
                        is_final=True,
                        created_at=request.transcript.created_at,
                        channel=request.transcript.channel,
                        final_language_runs=(run,),
                    ),
                    detected_language=run.language or None,
                    target_language=target_language,
                    source=request.source,
                    turn_kind=request.turn_kind,
                    context_policy=self.policy.context_policy,
                    precomputed_translation=request.precomputed_translation,
                    config_snapshot=request.config_snapshot,
                )
            )
        return tuple(children)

    async def _run_parent(
        self,
        parent: _TranslationTurnParent,
        predecessor: _TranslationTurnParent | None,
    ) -> None:
        try:
            if predecessor is not None:
                self._observe_predecessor_wait(
                    "predecessor_wait_start",
                    parent=parent,
                    predecessor=predecessor,
                )
                await predecessor.semantic_done_event.wait()
                self._observe_predecessor_wait(
                    "predecessor_wait_end",
                    parent=parent,
                    predecessor=predecessor,
                )
            for child in parent.children:
                if child.utterance_id in parent.completed_child_ids:
                    continue
                if self.is_child_cancellation_requested(child):
                    await self._terminalize_child(child, "cancelled")
                    continue
                await self._run_child(child, predecessor)
        except asyncio.CancelledError:
            await self._terminalize_parent_remaining(parent, "cancelled")
            raise
        except Exception:
            logger.exception("translation parent execution failed")
            await self._terminalize_parent_remaining(parent, "failed")
        finally:
            self._parent_tasks.pop(parent.parent_utterance_id, None)

    def _observe_predecessor_wait(
        self,
        event: str,
        *,
        parent: _TranslationTurnParent,
        predecessor: _TranslationTurnParent,
    ) -> None:
        if self.predecessor_wait_observer is None:
            return
        self.predecessor_wait_observer(
            event,
            {
                "channel": parent.channel,
                "parent_utterance_id": str(parent.parent_utterance_id),
                "predecessor_utterance_id": str(predecessor.parent_utterance_id),
                "active_parent_count": len(self._parents),
            },
        )

    async def _run_child(
        self,
        child: TranslationTurnChild,
        predecessor: _TranslationTurnParent | None,
    ) -> None:
        child_task = start_lifecycle_task(
            self._scope,
            self._execute_started_child(child, predecessor),
            name=f"child:{child.utterance_id}",
            eager_start=True,
        )
        self._active_tasks[child.utterance_id] = child_task
        try:
            await child_task
        except asyncio.CancelledError:
            await self._terminalize_child(child, "cancelled")
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
        except Exception:
            if not child_task.done():
                child_task.cancel()
                await asyncio.gather(child_task, return_exceptions=True)
            logger.exception("translation child execution adapter failed")
            await self._terminalize_child(child, "failed")
        finally:
            self._active_tasks.pop(child.utterance_id, None)

    async def _execute_started_child(
        self,
        child: TranslationTurnChild,
        predecessor: _TranslationTurnParent | None,
    ) -> TranslationTurnProcessResult:
        child_task = asyncio.current_task()
        if child_task is None:
            raise RuntimeError("translation child task is unavailable")
        await self.on_child_started(child, child_task)
        return await self._execute_child(child, predecessor)

    async def _execute_child(
        self,
        child: TranslationTurnChild,
        predecessor: _TranslationTurnParent | None,
    ) -> TranslationTurnProcessResult:
        result = await self._process_child(child)
        if self.is_child_cancellation_requested(child):
            await self._terminalize_child(child, "cancelled")
            raise asyncio.CancelledError
        self._mark_child_semantic_done(child)
        if predecessor is not None:
            await predecessor.closed_event.wait()
        if result.output is not None and self.output is not None:
            try:
                await self.output.submit_translation_output(result.output)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("translation output submission failed")
                result = TranslationTurnProcessResult("failed")
        await self._terminalize_child(child, result.outcome)
        return result

    def _mark_child_semantic_done(self, child: TranslationTurnChild) -> None:
        parent = self._parents.get(child.parent_utterance_id)
        if parent is None or parent.closed:
            return
        parent.semantic_completed_child_ids.add(child.utterance_id)
        if parent.semantic_completed_child_ids == set(parent.child_ids):
            parent.semantic_done_event.set()

    async def _process_child(self, child: TranslationTurnChild) -> TranslationTurnProcessResult:
        try:
            result = await self.process_child(
                child,
                lambda: self.is_child_cancellation_requested(child),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return TranslationTurnProcessResult("failed")
        if isinstance(result, str):
            return TranslationTurnProcessResult(result)
        return result

    def _request_cancellation(self) -> None:
        self._cancelling_parent_ids.update(self._parents)
        current_task = asyncio.current_task()
        for parent_task in tuple(self._parent_tasks.values()):
            if parent_task is not current_task and not parent_task.done():
                parent_task.cancel()

    async def _terminalize_unfinished_parents(
        self,
        outcome: TranslationTurnOutcome,
    ) -> None:
        await self._terminalize_selected_unfinished(tuple(self._parents.values()), outcome)

    async def _terminalize_selected_unfinished(
        self,
        parents: tuple[_TranslationTurnParent, ...],
        outcome: TranslationTurnOutcome,
    ) -> None:
        for parent in parents:
            await self._terminalize_parent_remaining(parent, outcome)

    async def _terminalize_parent_remaining(
        self,
        parent: _TranslationTurnParent,
        outcome: TranslationTurnOutcome,
    ) -> None:
        for child in parent.children:
            if child.utterance_id not in parent.completed_child_ids:
                await self._terminalize_child(child, outcome)

    async def _terminalize_child(
        self,
        child: TranslationTurnChild,
        outcome: TranslationTurnOutcome,
    ) -> None:
        parent = self._parents.get(child.parent_utterance_id)
        if parent is None or parent.closed or child.utterance_id in parent.completed_child_ids:
            return
        try:
            await self.on_child_terminal(child, outcome)
        except Exception:
            logger.exception("translation child terminal adapter failed")
        finally:
            parent.completed_child_ids.add(child.utterance_id)
            self._mark_child_semantic_done(child)
            if parent.completed_child_ids == set(parent.child_ids):
                await self._close_parent(parent)

    async def _close_parent(self, parent: _TranslationTurnParent) -> None:
        if parent.closed:
            return
        parent.closed = True
        self._parents.pop(parent.parent_utterance_id, None)
        if self._channel_tails.get(parent.channel) is parent:
            self._channel_tails.pop(parent.channel, None)
        self._cancelling_parent_ids.discard(parent.parent_utterance_id)
        self._closed_parent_ids.add(parent.parent_utterance_id)
        try:
            try:
                await self.on_parent_closed(parent.parent_utterance_id)
            except Exception:
                logger.exception("translation parent closure adapter failed")
        finally:
            parent.semantic_done_event.set()
            parent.closed_event.set()


__all__ = [
    "TranslationContextPolicy",
    "TranslationOutputSubmission",
    "TranslationOutputSubmissionPort",
    "TranslationRuntimePolicy",
    "TranslationTurnChild",
    "TranslationTurnKind",
    "TranslationTurnLifecycleOwner",
    "TranslationTurnOutcome",
    "TranslationTurnProcessResult",
    "TranslationTurnRequest",
]
