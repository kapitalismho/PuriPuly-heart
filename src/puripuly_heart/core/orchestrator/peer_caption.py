from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from collections.abc import Coroutine
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from uuid import UUID

from puripuly_heart.core.audio.ownership import AudioSegmentSnapshot
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime, ContextEntry
from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfigSnapshot
from puripuly_heart.core.orchestrator.translation_output_projection import (
    TranslationOutputProjectionOwner,
)
from puripuly_heart.core.orchestrator.translation_request import (
    PreparedTranslationRequest,
    TranslationProcessRequest,
    TranslationRequestPort,
    _UnmappedDetectedLanguage,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationOutputSubmission,
    _final_transcript_fragment,
    _final_transcript_segments,
    _translation_turn_child_id,
)
from puripuly_heart.core.overlay.peer_caption_layout import estimate_peer_translation_lines
from puripuly_heart.core.speaker_identity import PeerSpeakerIdentityAllocator
from puripuly_heart.domain.models import SpeakerAssignment, SpeakerAttribution, Transcript

CLOUD_PEER_PROVIDERS = frozenset({"soniox", "deepgram", "gemini_transcribe", "qwen_audio"})
PeerRequestKind = Literal["final", "preview", "repair", "continuation"]
_OPEN_WORDS = frozenset(
    "a about am an and are as at because but can could for from had has have her his how if in "
    "is its may might must my of on or our shall should so that the their they to was were what "
    "when which while who will with would your".split()
)
_KOREAN_ENDINGS = tuple(
    "면서 지만 는데 은데 어서 아서 해서 니까 으며 며 고 면 도록 려고 다가 거나 든지 은 는 이 가 을 를 에 "
    "에서 의 와 과 로 으로 도 만".split()
)
_REPLIES = {
    "en": frozenset(
        "yeah yes yep no nope okay ok right sure mhm uh-huh hmm oh wow really exactly true".split()
    ),
    "ko": frozenset("네 예 응 어 아니 아니요 아뇨 그래 그렇죠 맞아 맞아요 음 아 오".split()),
    "ja": frozenset("はい うん いいえ ええ そう".split()),
}


def preview_source(text: str) -> str:
    text = text.strip()
    boundaries = tuple(re.finditer(r"[.!?。？！,，、;；:：…—](?:[.!?…]*)", text))
    for boundary in reversed(boundaries):
        if boundary.end() >= len(text) / 2:
            return text[: boundary.end()].strip()
    return text


def join_action(
    previous: str,
    incoming: str,
    *,
    language: str,
    gap_s: float | None,
    expected_lines: float,
    provider_id: str,
    sealed: AudioSegmentSnapshot | None,
) -> Literal["merge", "continuation", "independent"]:
    language = language.lower().split("-", 1)[0]
    reply_text = incoming.lower().strip(" \t\n.!?。？！,，、:;\"'“”‘’")
    replies = _REPLIES.get(language, frozenset())
    words = re.findall(r"\w+(?:-\w+)*", reply_text)
    if gap_s is None or gap_s > 1.5 or (words and all(word in replies for word in words)):
        return "independent"
    previous = previous.rstrip()
    incoming = incoming.lstrip()
    last_word = previous.rstrip(".!?。？！,，、:;\"'“”‘’").split()[-1].lower() if previous else ""
    r0 = (
        previous.endswith(("—", "-", "...", "…", ",", "，", "、"))
        or not previous.endswith((".", "?", "!", "。", "？", "！"))
        or (language == "en" and (last_word in _OPEN_WORDS or (incoming and incoming[0].islower())))
        or (language == "ko" and last_word.endswith(_KOREAN_ENDINGS))
    )
    acoustic = sealed is not None and (
        sealed.seal_reason == "delivery_deadline"
        or (
            sealed.seal_reason == "delivery_pause"
            and sealed.delivery_pause_s is not None
            and sealed.sealed_at_monotonic_s is not None
            and sealed.sealed_at_monotonic_s - sealed.opened_at_monotonic_s >= 4.0
            and sealed.delivery_pause_s < 0.5
        )
        or sealed.smart_turn_incomplete
    )
    r1 = (
        (
            previous.endswith(("—", "–", "-", "…", "...", ",", "，", "、", ":", ";"))
            or (provider_id != "gemini_transcribe" and acoustic)
        )
        and not previous.endswith(("?", "!", "？", "！"))
        and expected_lines <= 4
    )
    if not (r0 or r1):
        return "independent"
    return "merge" if expected_lines <= 2 else "continuation"


@dataclass(slots=True)
class _Caption:
    utterance_id: UUID
    generation: int
    source_order: int
    provider_id: str
    scope: tuple[object, ...]
    source_language: str | None = None
    source: str = ""
    revision: int = 0
    shown_revision: int = 0
    context: tuple[str, ...] | None = None
    last_final_at: float | None = None
    sealed: AudioSegmentSnapshot | None = None
    superseded: tuple[UUID, ...] = ()
    speaker_assignment: SpeakerAssignment | None = None
    admitted: asyncio.Event = field(default_factory=asyncio.Event)
    predecessor: asyncio.Event | None = None
    absorbed_admissions: dict[UUID, asyncio.Event] = field(default_factory=dict)


@dataclass(slots=True)
class _Segment:
    caption: _Caption
    first_partial_at: float
    next_check_at: float
    partial: str
    config: TranslationRuntimeConfigSnapshot
    last_partial_length: int = 0
    last_preview_source: str = ""
    preview_task: asyncio.Task[None] | None = None
    timer: asyncio.Task[None] | None = None
    pending: str | None = None
    closed: bool = False


@dataclass(slots=True)
class PeerCaptionOwner:
    runtime: ChannelRuntime
    requests: TranslationRequestPort
    output: TranslationOutputProjectionOwner
    clock: Clock
    speaker_identities: PeerSpeakerIdentityAllocator = field(
        default_factory=PeerSpeakerIdentityAllocator
    )
    max_source_chars: int = 65536
    max_requests: int = 32
    request_timeout_s: float = 30.0
    _segments: dict[UUID, _Segment] = field(default_factory=dict)
    _last: _Caption | None = None
    _tail: asyncio.Event | None = None
    _completed: OrderedDict[UUID, None] = field(default_factory=OrderedDict)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set)
    _scope: LifecycleScope = field(default_factory=lambda: LifecycleScope("PeerCaptionOwner"))
    _slots: asyncio.Semaphore = field(init=False)
    _epoch: int = 0
    request_counts: dict[str, int] = field(
        default_factory=lambda: {kind: 0 for kind in ("preview", "final", "repair", "continuation")}
    )

    def __post_init__(self) -> None:
        self._slots = asyncio.Semaphore(self.max_requests)

    def _start(self, coroutine: Coroutine[Any, Any, None], name: str) -> asyncio.Task[None]:
        task = start_lifecycle_task(self._scope, coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def reset(self) -> None:
        self._epoch += 1
        for segment in self._segments.values():
            segment.closed = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._scope.close()
        await self.output.reset_peer_caption_admission()
        self._scope = LifecycleScope("PeerCaptionOwner")
        self._segments.clear()
        self._last = None
        self._tail = None
        self._completed.clear()

    async def drain(self) -> None:
        for utterance_id in tuple(self._segments):
            self.abandon(utterance_id)
        await self.wait_for_idle()

    async def wait_for_idle(self) -> None:
        while True:
            requests = tuple(task for task in self._tasks if ":timer:" not in task.get_name())
            if not requests:
                return
            results = await asyncio.gather(*requests, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    raise result

    def _new_caption(
        self, transcript: Transcript, provider_id: str, scope: tuple[object, ...]
    ) -> _Caption:
        if transcript.publication_generation is None or transcript.source_order is None:
            raise ValueError("Cloud Peer requires admitted publication identity")
        caption = _Caption(
            transcript.utterance_id,
            transcript.publication_generation,
            transcript.source_order,
            provider_id,
            scope,
            predecessor=self._tail,
        )
        self._tail = caption.admitted
        return caption

    def _context(self, config: TranslationRuntimeConfigSnapshot) -> tuple[str, ...]:
        source = self.requests.source_language_for("peer", config.value)
        target = self.requests.target_language_for("peer", config.value)
        return tuple(
            entry.text
            for entry in self.runtime.translation_history
            if self.clock.now() - entry.timestamp < 60.0
            and (
                config.value.peer_source_mode == "auto"
                or not entry.source_language
                or entry.source_language == source
            )
            and (not entry.target_language or entry.target_language == target)
            and entry.text.strip()
        )[-5:]

    def _prepare(
        self,
        caption: _Caption,
        source: str,
        kind: PeerRequestKind,
        config: TranslationRuntimeConfigSnapshot,
    ) -> PreparedTranslationRequest | None:
        if caption.context is None:
            caption.context = self._context(config)
        try:
            prepared = self.requests.prepare(
                source,
                channel="peer",
                config_snapshot=config,
                detected_language=caption.source_language,
                peer_request_kind=kind,
                peer_context_texts=caption.context,
            )
        except _UnmappedDetectedLanguage:
            return None
        return replace(prepared, provider_generation=self.requests.provider_generation)

    async def partial(
        self,
        transcript: Transcript,
        *,
        provider_id: str,
        scope: tuple[object, ...],
        config: TranslationRuntimeConfigSnapshot,
    ) -> None:
        if not self.output.output_runtime.peer_publication_is_authorized(
            transcript.publication_generation, transcript.source_order
        ):
            return
        if transcript.utterance_id in self._completed:
            return
        if len(transcript.text) > self.max_source_chars:
            raise ValueError("Peer partial text capacity exceeded")
        segment = self._segments.get(transcript.utterance_id)
        if not transcript.text.strip():
            if segment is not None and provider_id != "qwen_audio":
                segment.partial = ""
                segment.pending = None
                if segment.preview_task is not None:
                    segment.preview_task.cancel()
                await self.output.supersede_peer_caption_previews((transcript.utterance_id,))
            return
        if segment is None:
            if len(self._segments) >= 2:
                raise ValueError("Peer open segment capacity exceeded")
            started = (
                transcript.created_at if transcript.created_at is not None else self.clock.now()
            )
            segment = _Segment(
                self._new_caption(transcript, provider_id, scope),
                started,
                started + 5.0,
                transcript.text,
                config,
            )
            self._segments[transcript.utterance_id] = segment
            segment.timer = self._start(
                self._timer(segment), f"peer:timer:{transcript.utterance_id}"
            )
        segment.partial = transcript.text

    async def _timer(self, segment: _Segment) -> None:
        while not segment.closed:
            await asyncio.sleep(max(0.0, segment.next_check_at - self.clock.now()))
            await self.check_previews()

    async def check_previews(self) -> None:
        for segment in tuple(self._segments.values()):
            if segment.closed or self.clock.now() < segment.next_check_at:
                continue
            segment.next_check_at = self.clock.now() + 4.0
            source = preview_source(segment.partial)
            if len(segment.partial) <= segment.last_partial_length or len(source) <= len(
                segment.last_preview_source
            ):
                continue
            segment.last_partial_length = len(segment.partial)
            segment.last_preview_source = source
            segment.pending = source
            if segment.preview_task is None or segment.preview_task.done():
                segment.preview_task = self._start(
                    self._previews(segment), f"peer:preview:{segment.caption.utterance_id}"
                )

    async def _previews(self, segment: _Segment) -> None:
        while segment.pending is not None and not segment.closed:
            source, segment.pending = segment.pending, None
            segment.caption.revision += 1
            revision = segment.caption.revision
            prepared = self._prepare(segment.caption, source, "preview", segment.config)
            await self._slots.acquire()
            await self._execute(
                segment.caption, source, "preview", revision, segment.config, prepared, segment
            )

    def abandon(self, utterance_id: UUID) -> None:
        segment = self._segments.pop(utterance_id, None)
        if segment is None:
            return
        segment.closed = True
        segment.pending = None
        segment.caption.admitted.set()
        for task in (segment.timer, segment.preview_task):
            if task is not None:
                task.cancel()

    def _complete(self, utterance_id: UUID) -> None:
        self._completed[utterance_id] = None
        while len(self._completed) > 4096:
            self._completed.popitem(last=False)

    async def retire(self, utterance_id: UUID) -> None:
        segment = self._segments.get(utterance_id)
        self._complete(utterance_id)
        if segment is None:
            return
        caption = segment.caption
        self.abandon(utterance_id)
        caption.revision += 1
        await self.output.supersede_peer_caption_previews((utterance_id,))
        await self.output.project_peer_caption(
            TranslationOutputSubmission(
                caption.utterance_id,
                caption.utterance_id,
                0,
                "peer",
                "Peer",
                "",
                caption.source_language,
                self.requests.target_language_for("peer", segment.config.value),
                "source_only",
                segment.config,
                publication_generation=caption.generation,
                source_order=caption.source_order,
                peer_caption_revision=caption.revision,
                peer_superseded_unit_ids=caption.superseded,
                speaker_assignment=caption.speaker_assignment,
            )
        )

    async def retire_epoch(self, provider_epoch_id: str) -> None:
        for utterance_id, segment in tuple(self._segments.items()):
            if segment.caption.scope[-1:] == (provider_epoch_id,):
                await self.retire(utterance_id)

    def is_completed(self, utterance_id: UUID) -> bool:
        return utterance_id in self._completed

    async def final(
        self,
        transcript: Transcript,
        *,
        provider_id: str,
        scope: tuple[object, ...],
        config: TranslationRuntimeConfigSnapshot,
        sealed: AudioSegmentSnapshot | None = None,
    ) -> None:
        if not self.output.output_runtime.peer_publication_is_authorized(
            transcript.publication_generation, transcript.source_order
        ):
            return
        if transcript.utterance_id in self._completed:
            return
        if len(transcript.text) > self.max_source_chars:
            raise ValueError("Peer final text capacity exceeded")
        if not transcript.text.strip():
            await self.retire(transcript.utterance_id)
            return
        target = self.requests.target_language_for("peer", config.value)
        fragments = tuple(
            _final_transcript_fragment(
                transcript,
                segment,
                (
                    transcript.utterance_id
                    if sequence == 0
                    else _translation_turn_child_id(
                        transcript.utterance_id,
                        turn_kind="peer",
                        run_index=sequence,
                        target_index=0,
                        run_language=segment.language,
                        target_language=target,
                        primary_uses_parent_identity=False,
                    )
                ),
            )
            for sequence, segment in enumerate(
                segment
                for segment in _final_transcript_segments(transcript, split_speakers=True)
                if segment.text.strip()
            )
        )
        assignments = tuple(
            self.speaker_identities.observe(fragment, child_sequence=sequence)
            for sequence, fragment in enumerate(fragments)
        )
        for fragment in fragments:
            self.speaker_identities.retire(fragment.utterance_id)
        self._complete(transcript.utterance_id)
        for fragment, assignment in zip(fragments, assignments, strict=True):
            await self._final_fragment(
                fragment,
                assignment,
                provider_id=provider_id,
                scope=scope,
                config=config,
                sealed=sealed,
            )

    async def _final_fragment(
        self,
        transcript: Transcript,
        assignment: SpeakerAssignment,
        *,
        provider_id: str,
        scope: tuple[object, ...],
        config: TranslationRuntimeConfigSnapshot,
        sealed: AudioSegmentSnapshot | None,
    ) -> None:
        segment = self._segments.pop(transcript.utterance_id, None)
        if segment is not None:
            segment.closed = True
            segment.pending = None
            for task in (segment.timer, segment.preview_task):
                if task is not None:
                    task.cancel()
        caption = (
            segment.caption
            if segment is not None
            else self._new_caption(transcript, provider_id, scope)
        )
        caption.speaker_assignment = assignment
        languages = {run.language for run in transcript.final_language_runs if run.language}
        caption.source_language = next(iter(languages)) if len(languages) == 1 else None
        source = transcript.text
        kind: PeerRequestKind = "final"
        previous = self._last
        if previous is not None and previous.scope == scope and previous.provider_id == provider_id:
            separator = "" if previous.source[-1:].isspace() or source[:1].isspace() else " "
            merged = previous.source + separator + source
            gap = (
                None
                if segment is None or previous.last_final_at is None
                else segment.first_partial_at - previous.last_final_at
            )
            action = (
                join_action(
                    previous.source,
                    source,
                    language=previous.source_language
                    or self.requests.source_language_for("peer", config.value),
                    gap_s=gap,
                    expected_lines=estimate_peer_translation_lines(
                        merged,
                        self.requests.source_language_for("peer", config.value),
                        self.requests.target_language_for("peer", config.value),
                        text_scale=self.output.peer_caption_text_scale,
                    ),
                    provider_id=provider_id,
                    sealed=previous.sealed,
                )
                if len(merged) <= self.max_source_chars
                else "independent"
            )
            if action == "merge":
                previous.superseded = tuple(
                    dict.fromkeys((*previous.superseded, caption.utterance_id, *caption.superseded))
                )
                previous.absorbed_admissions[caption.utterance_id] = caption.admitted
                caption = previous
                source = merged
                kind = "repair"
                if previous.speaker_assignment is not None:
                    prior = previous.speaker_assignment
                    incoming = assignment.attribution
                    attribution = (
                        prior.attribution
                        if (
                            prior.attribution.key is not None
                            and prior.attribution.key == incoming.key
                        )
                        or (
                            prior.attribution.state == "non_diarized"
                            and incoming.state == "non_diarized"
                        )
                        else SpeakerAttribution("mixed")
                    )
                    previous.speaker_assignment = replace(
                        prior,
                        attribution=attribution,
                        source_text_range=None,
                        source_text_revision=None,
                    )
                self._tail = caption.admitted
            elif action == "continuation":
                kind = "continuation"
                current = list(self._context(config))
                if current and current[-1] == previous.source:
                    current[-1] = previous.source
                else:
                    current.append(previous.source)
                caption.context = tuple(current[-5:])
        caption.source = source
        caption.last_final_at = self.clock.now()
        caption.sealed = sealed
        caption.revision += 1
        self._last = caption
        prepared = self._prepare(caption, source, kind, config)
        self.runtime.translation_history[:] = [
            entry
            for entry in self.runtime.translation_history
            if entry.utterance_id != caption.utterance_id
        ]
        self.runtime.translation_history.append(
            ContextEntry(
                source,
                (prepared.source_language if prepared is not None else caption.source_language),
                self.requests.target_language_for("peer", config.value),
                self.clock.now(),
                "peer",
                "peer",
                caption.utterance_id,
            )
        )
        del self.runtime.translation_history[:-5]
        revision = caption.revision
        superseded = caption.superseded
        request_assignment = caption.speaker_assignment
        request_language = caption.source_language
        admission_epoch = self._epoch
        await self.output.supersede_peer_caption_previews((caption.utterance_id, *superseded))
        await self._slots.acquire()
        if admission_epoch != self._epoch:
            self._slots.release()
            return
        self._start(
            self._execute(
                caption,
                source,
                kind,
                revision,
                config,
                prepared,
                superseded=superseded,
                assignment=request_assignment,
                detected_language=request_language,
            ),
            f"peer:{kind}:{caption.utterance_id}:{revision}",
        )

    async def _execute(
        self,
        caption: _Caption,
        source: str,
        kind: PeerRequestKind,
        revision: int,
        config: TranslationRuntimeConfigSnapshot,
        prepared: PreparedTranslationRequest | None,
        segment: _Segment | None = None,
        superseded: tuple[UUID, ...] = (),
        assignment: SpeakerAssignment | None = None,
        detected_language: str | None = None,
    ) -> None:
        epoch = self._epoch
        source_language = prepared.source_language if prepared is not None else detected_language
        target_language = self.requests.target_language_for("peer", config.value)
        try:
            self.request_counts[kind] += 1
            request = TranslationProcessRequest(
                parent_utterance_id=caption.utterance_id,
                utterance_id=caption.utterance_id,
                sequence=0,
                text=source,
                channel="peer",
                source="Peer",
                target_language=target_language,
                context_policy="integrated_preferred",
                config_snapshot=config,
                detected_language=source_language if source_language != "auto" else None,
                publication_generation=caption.generation,
                source_order=caption.source_order,
                turn_kind="peer",
                expected_provider_generation=(
                    prepared.provider_generation
                    if prepared is not None
                    else self.requests.provider_generation
                ),
                peer_request_kind=kind,
                peer_caption_revision=revision,
                peer_superseded_unit_ids=superseded,
            )
            try:
                async with asyncio.timeout(self.request_timeout_s):
                    result = await self.requests.process(request, prepared=prepared)
                submission = result.output
                if submission is None:
                    raise RuntimeError("Peer translation completed without output")
            except TimeoutError:
                submission = TranslationOutputSubmission(
                    caption.utterance_id,
                    caption.utterance_id,
                    0,
                    "peer",
                    "Peer",
                    source,
                    source_language,
                    target_language,
                    "failed",
                    config,
                    failure_code="translation_timeout",
                    publication_generation=caption.generation,
                    source_order=caption.source_order,
                    peer_request_kind=kind,
                    peer_caption_revision=revision,
                    peer_superseded_unit_ids=superseded,
                    context_texts=caption.context,
                )
            submission = replace(
                submission,
                speaker_assignment=assignment,
                translation=(
                    None
                    if submission.translation is None
                    else replace(submission.translation, speaker_assignment=assignment)
                ),
            )
            if submission.failure_code == "stale_provider_completion":
                return
            if (
                epoch != self._epoch
                or (segment is not None and segment.closed)
                or revision < caption.shown_revision
            ):
                return
            if kind == "preview" and (
                submission.translation is None or revision < caption.revision
            ):
                return
            if caption.predecessor is not None and not caption.admitted.is_set():
                await caption.predecessor.wait()
            if (
                epoch != self._epoch
                or revision < caption.shown_revision
                or not self.output.output_runtime.peer_publication_is_authorized(
                    caption.generation, caption.source_order
                )
            ):
                return
            caption.shown_revision = revision
            await self.output.project_peer_caption(submission)
        finally:
            if kind != "preview" or caption.shown_revision:
                caption.admitted.set()
                for utterance_id in superseded:
                    admitted = caption.absorbed_admissions.pop(utterance_id, None)
                    if admitted is not None:
                        admitted.set()
            self._slots.release()
