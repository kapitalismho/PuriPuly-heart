from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from uuid import UUID, uuid4

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
    _FinalTranscriptSegment,
    _native_peer_source_groups,
    _source_speaker_runs,
    _translation_turn_child_id,
)
from puripuly_heart.core.overlay.peer_caption_layout import estimate_peer_translation_lines
from puripuly_heart.core.overlay.sink import PeerCaptionLineage, PeerCaptionRevision
from puripuly_heart.core.speaker_identity import (
    PeerSpeakerIdentityAllocator,
    native_peer_attribution,
)
from puripuly_heart.core.stt.backend import (
    STTFinalSpeakerRevision,
    STTNativeSegmentRef,
    STTNativeSpeakerMerge,
)
from puripuly_heart.domain.models import (
    FinalLanguageRun,
    FinalSpeakerRun,
    SpeakerAssignment,
    SpeakerAttribution,
    SpeakerKey,
    Transcript,
)

PeerRequestKind = Literal["final", "preview", "repair", "continuation"]
_SCRIBE_SOURCE = "elevenlabs_scribe"
_SCRIBE_RETENTION = 32
_NATIVE_PEER_SPEAKERS = frozenset({"soniox", _SCRIBE_SOURCE})
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
    previous_words = previous.rstrip(".!?。？！,，、:;\"'“”‘’").split()
    last_word = previous_words[-1].lower() if previous_words else ""
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
class _ScribeCommit:
    native_segment: STTNativeSegmentRef
    provider_epoch: object
    transcript: Transcript


@dataclass(frozen=True, slots=True)
class _ScribePart:
    commit: _ScribeCommit | None
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def runs(self, scope: str) -> tuple[FinalSpeakerRun, ...]:
        if self.commit is None:
            return (
                FinalSpeakerRun(
                    " ", None, scope, source=_SCRIBE_SOURCE, attribution_state="missing"
                ),
            )
        return _source_speaker_runs(self.commit.transcript, self.start, self.end)


@dataclass(slots=True, eq=False)
class _ScribeFamily:
    config: TranslationRuntimeConfigSnapshot
    lineage: PeerCaptionLineage | None
    captions: tuple[_Caption, ...] = ()


@dataclass(slots=True, eq=False)
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
    displayed_revision: int = 0
    context: tuple[str, ...] | None = None
    last_final_at: float | None = None
    sealed: AudioSegmentSnapshot | None = None
    superseded: tuple[UUID, ...] = ()
    speaker_assignment: SpeakerAssignment | None = None
    native_speakers: bool = False
    source_runs: tuple[FinalSpeakerRun, ...] = ()
    continuity_key: SpeakerKey | None = None
    scribe_continuity: SpeakerKey | None = None
    scribe_continues: UUID | None = None
    admitted: asyncio.Event = field(default_factory=asyncio.Event)
    predecessor: asyncio.Event | None = None
    absorbed_admissions: dict[UUID, asyncio.Event] = field(default_factory=dict)
    final_admissions: set[asyncio.Event] = field(default_factory=set)
    scribe_family: _ScribeFamily | None = None
    scribe_parts: tuple[_ScribePart, ...] = ()
    lineage: PeerCaptionLineage | None = None
    required_lineage_members: tuple[UUID, ...] = ()
    history_after: UUID | None = None
    history_removals: tuple[UUID, ...] = ()


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
    speaker_diarization: Callable[[str], bool] = lambda _provider_id: False
    max_source_chars: int = 65536
    max_requests: int = 32
    request_timeout_s: float = 30.0
    _segments: dict[UUID, _Segment] = field(default_factory=dict)
    _last: _Caption | None = None
    _tail: asyncio.Event | None = None
    _completed: OrderedDict[UUID, None] = field(default_factory=OrderedDict)
    _scribe_commits: OrderedDict[UUID, _ScribeCommit] = field(default_factory=OrderedDict)
    _scribe_families: OrderedDict[UUID, _ScribeFamily] = field(default_factory=OrderedDict)
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
        await self.output.reset_peer_caption_admission()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._scope.close()
        self._scope = LifecycleScope("PeerCaptionOwner")
        self._segments.clear()
        self._last = None
        self._tail = None
        self._completed.clear()
        self._scribe_commits.clear()
        self._scribe_families.clear()

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
            self._tasks.difference_update(requests)
            for result in results:
                if isinstance(result, Exception):
                    raise result

    def _new_caption(
        self,
        transcript: Transcript,
        provider_id: str,
        scope: tuple[object, ...],
        *,
        track_scribe_lineage: bool,
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
        if track_scribe_lineage:
            caption.lineage = self.output.output_runtime.open_peer_caption_lineage()
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
                segment.caption.revision += 1
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
            diarization = self.speaker_diarization(provider_id)
            caption = self._new_caption(
                transcript,
                provider_id,
                scope,
                track_scribe_lineage=provider_id == _SCRIBE_SOURCE and diarization,
            )
            if not diarization:
                caption.speaker_assignment = SpeakerAssignment(
                    SpeakerAttribution("non_diarized"),
                    scope_order=(caption.generation, caption.source_order, 0),
                )
            segment = _Segment(
                caption,
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

    def _translation_active(self) -> bool:
        return self.requests.provider_available and self.requests.translation_enabled_for("peer")

    async def check_previews(self) -> None:
        for segment in tuple(self._segments.values()):
            if segment.closed or self.clock.now() < segment.next_check_at:
                continue
            segment.next_check_at = self.clock.now() + 4.0
            if not self._translation_active():
                continue
            source = preview_source(segment.partial)
            if len(segment.partial) > segment.last_partial_length and len(source) > len(
                segment.last_preview_source
            ):
                segment.last_partial_length = len(segment.partial)
                segment.last_preview_source = source
                segment.pending = source
            if segment.pending is not None and (
                segment.preview_task is None or segment.preview_task.done()
            ):
                segment.preview_task = self._start(
                    self._previews(segment), f"peer:preview:{segment.caption.utterance_id}"
                )

    async def _previews(self, segment: _Segment) -> None:
        while segment.pending is not None and not segment.closed:
            if not self._translation_active():
                return
            source, segment.pending = segment.pending, None
            prepared = self._prepare(segment.caption, source, "preview", segment.config)
            await self._slots.acquire()
            if not self._translation_active():
                self._slots.release()
                if segment.pending is None:
                    segment.pending = source
                return
            if prepared is not None:
                prepared = replace(prepared, provider_generation=self.requests.provider_generation)
            segment.caption.revision += 1
            revision = segment.caption.revision
            await self._execute(
                segment.caption,
                source,
                "preview",
                revision,
                segment.config,
                prepared,
                segment,
                assignment=segment.caption.speaker_assignment,
                admitted=segment.caption.admitted,
                predecessor=segment.caption.predecessor,
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

    async def withdraw(self, utterance_id: UUID) -> None:
        self._complete(utterance_id)
        if utterance_id not in self._segments:
            return
        self.abandon(utterance_id)
        await self.output.supersede_peer_caption_previews((utterance_id,))

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
        for uid, family in tuple(self._scribe_families.items()):
            if family.captions and family.captions[0].scope[-1:] == (provider_epoch_id,):
                self._scribe_families.pop(uid)
        for uid, commit in tuple(self._scribe_commits.items()):
            if commit.provider_epoch == provider_epoch_id:
                self._scribe_commits.pop(uid)

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
        native_segment: STTNativeSegmentRef | None = None,
    ) -> None:
        owner_epoch = self._epoch
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
        native_speakers = provider_id in _NATIVE_PEER_SPEAKERS and any(
            run.source == provider_id for run in transcript.final_speaker_runs
        )
        scribe_commit = None
        scribe_family = None
        if native_segment is not None and provider_id == _SCRIBE_SOURCE:
            scribe_commit = _ScribeCommit(native_segment, scope[-1], transcript)
            self._scribe_commits[transcript.utterance_id] = scribe_commit
            if len(self._scribe_commits) > _SCRIBE_RETENTION:
                self._scribe_commits.popitem(last=False)
        source_groups = (
            _native_peer_source_groups(transcript)
            if native_speakers
            else _final_transcript_segments(transcript, split_speakers=True)
        )
        if not source_groups and scribe_commit is not None:
            source_groups = (
                _FinalTranscriptSegment(transcript.text, "", source_text_end=len(transcript.text)),
            )
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
                preserve_speaker_runs=native_speakers,
            )
            for sequence, segment in enumerate(
                segment for segment in source_groups if segment.text.strip()
            )
        )
        if not fragments:
            await self.retire(transcript.utterance_id)
            return
        assignments = tuple(
            self.speaker_identities.observe(
                fragment, child_sequence=sequence, native_speakers=native_speakers
            )
            for sequence, fragment in enumerate(fragments)
        )
        for fragment in fragments:
            self.speaker_identities.retire(fragment.utterance_id)
        self._complete(transcript.utterance_id)
        for fragment, assignment in zip(fragments, assignments, strict=True):
            caption = await self._final_fragment(
                fragment,
                assignment,
                provider_id=provider_id,
                scope=scope,
                config=config,
                sealed=sealed,
                native_speakers=native_speakers,
                scribe_commit=scribe_commit,
                scribe_family=scribe_family,
            )
            if owner_epoch != self._epoch:
                return
            scribe_family = caption.scribe_family
        if scribe_family is not None:
            family_id = scribe_family.captions[0].utterance_id
            self._scribe_families[family_id] = scribe_family
            self._scribe_families.move_to_end(family_id)
            if len(self._scribe_families) > _SCRIBE_RETENTION:
                self._scribe_families.popitem(last=False)

    async def _final_fragment(
        self,
        transcript: Transcript,
        assignment: SpeakerAssignment,
        *,
        provider_id: str,
        scope: tuple[object, ...],
        config: TranslationRuntimeConfigSnapshot,
        sealed: AudioSegmentSnapshot | None,
        native_speakers: bool,
        scribe_commit: _ScribeCommit | None,
        scribe_family: _ScribeFamily | None,
    ) -> _Caption:
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
            else self._new_caption(
                transcript, provider_id, scope, track_scribe_lineage=scribe_commit is not None
            )
        )
        if scribe_commit is not None:
            if scribe_family is None:
                scribe_family = _ScribeFamily(config, caption.lineage)
            caption.lineage = scribe_family.lineage
            caption.scribe_family = scribe_family
            start, end = transcript.source_text_range or (0, len(transcript.text))
            caption.scribe_parts = (_ScribePart(scribe_commit, start, end),)
        request_admission = caption.admitted
        if native_speakers and segment is not None:
            request_admission = asyncio.Event()
            self._tail = request_admission
            if scribe_commit is not None:
                caption.admitted = request_admission
        caption.speaker_assignment = assignment
        caption.native_speakers = native_speakers
        caption.source_runs = transcript.final_speaker_runs
        caption.continuity_key = assignment.attribution.key
        languages = {run.language for run in transcript.final_language_runs if run.language}
        caption.source_language = next(iter(languages)) if len(languages) == 1 else None
        source = transcript.text
        kind: PeerRequestKind = "final"
        previous = self._last
        if previous is not None and previous.scope == scope and previous.provider_id == provider_id:
            previous_language = previous.source_language or self.requests.source_language_for(
                "peer", config.value
            )
            separator = "" if previous.source[-1:].isspace() or source[:1].isspace() else " "
            merged = previous.source + separator + source
            gap = (
                None
                if segment is None or previous.last_final_at is None
                else segment.first_partial_at - previous.last_final_at
            )
            previous_key = self._resolved_continuity(previous)
            native_compatible = not (previous.native_speakers or native_speakers) or (
                previous.native_speakers
                and native_speakers
                and previous.generation == caption.generation
                and previous.source_language == caption.source_language
                and previous.speaker_assignment is not None
                and previous.speaker_assignment.attribution.state in ("identified", "missing")
                and assignment.attribution.state in ("identified", "missing")
                and previous.speaker_assignment.attribution.speaker_scope_id
                == assignment.attribution.speaker_scope_id
                and (
                    previous_key is None
                    or assignment.attribution.key is None
                    or previous_key == assignment.attribution.key
                )
            )
            scribe_compatible = scribe_commit is None or (
                previous.scribe_family is not None
                and (
                    previous.lineage is None
                    or self.output.output_runtime.peer_caption_lineage_is_live(previous.lineage)
                )
                and self._translation_active()
                and sum(len(member.source) for member in previous.scribe_family.captions)
                + len(scribe_commit.transcript.text)
                + len(separator)
                <= self.max_source_chars
            )
            action = (
                join_action(
                    previous.source,
                    source,
                    language=previous_language,
                    gap_s=gap,
                    expected_lines=estimate_peer_translation_lines(
                        merged,
                        previous_language,
                        self.requests.target_language_for("peer", config.value),
                        text_scale=self.output.peer_caption_text_scale,
                    ),
                    provider_id=provider_id,
                    sealed=previous.sealed,
                )
                if native_compatible and scribe_compatible and len(merged) <= self.max_source_chars
                else "independent"
            )
            continuity_key = previous_key or assignment.attribution.key
            if action == "merge":
                previous.superseded = tuple(
                    dict.fromkeys((*previous.superseded, caption.utterance_id, *caption.superseded))
                )
                previous.absorbed_admissions[caption.utterance_id] = caption.admitted
                if scribe_commit is not None:
                    previous.final_admissions.add(request_admission)
                if scribe_commit is not None and previous.scribe_family is not None:
                    scribe_family = previous.scribe_family
                    previous.scribe_parts += (
                        *((_ScribePart(None, 0, 1),) if separator else ()),
                        *caption.scribe_parts,
                    )
                caption = previous
                source = merged
                kind = "repair"
                if native_speakers:
                    prior = previous.speaker_assignment
                    if prior is None:
                        raise RuntimeError("Native Peer repair requires source attribution")
                    representative = prior if prior.attribution.key is not None else assignment
                    caption.speaker_assignment = replace(
                        representative,
                        scope_order=prior.scope_order,
                        source_text_range=None,
                        source_text_revision=hashlib.sha256(merged.encode("utf-8")).hexdigest()[
                            :16
                        ],
                    )
                    caption.source_runs = (
                        *previous.source_runs,
                        *(
                            (
                                FinalSpeakerRun(
                                    separator,
                                    None,
                                    assignment.attribution.speaker_scope_id,
                                    source=assignment.attribution.source,
                                    attribution_state="missing",
                                ),
                            )
                            if separator
                            else ()
                        ),
                        *transcript.final_speaker_runs,
                    )
                    caption.speaker_assignment = replace(
                        caption.speaker_assignment,
                        attribution=native_peer_attribution(merged, caption.source_runs),
                    )
                    caption.continuity_key = continuity_key
                elif previous.speaker_assignment is not None:
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
                if not native_speakers:
                    self._tail = caption.admitted
            elif action == "continuation":
                kind = "continuation"
                if scribe_commit is not None:
                    caption.scribe_continues = previous.utterance_id
                    caption.scribe_continuity = previous_key
                else:
                    caption.continuity_key = continuity_key
                current = list(self._context(config))
                if current and current[-1] == previous.source:
                    current[-1] = previous.source
                else:
                    current.append(previous.source)
                caption.context = tuple(current[-5:])
        caption.source = source
        caption.last_final_at = self.clock.now()
        if scribe_family is not None and caption not in scribe_family.captions:
            scribe_family.captions += (caption,)
        caption.sealed = sealed
        caption.revision += 1
        if caption.scribe_family is not None:
            caption.required_lineage_members = ()
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
        if admission_epoch != self._epoch:
            return caption
        if caption.scribe_family is not None:
            await self._slots.acquire()
            if admission_epoch != self._epoch:
                self._slots.release()
                return caption
            caption.final_admissions.add(request_admission)
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
                    admitted=request_admission,
                    predecessor=caption.predecessor,
                ),
                f"peer:{kind}:{caption.utterance_id}:{revision}",
            )
            return caption
        request_predecessor = caption.predecessor
        await self._slots.acquire()
        if admission_epoch != self._epoch:
            self._slots.release()
            return caption
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
                admitted=request_admission,
                predecessor=request_predecessor,
            ),
            f"peer:{kind}:{caption.utterance_id}:{revision}",
        )
        return caption

    async def speaker_evidence(
        self,
        revisions: tuple[STTFinalSpeakerRevision, ...],
        *,
        is_current: Callable[[], bool] | None,
        speaker_merge: STTNativeSpeakerMerge | None,
    ) -> None:
        if is_current is not None and not is_current():
            return
        if speaker_merge is not None and any(
            revision.native_segment.provider_session_id == speaker_merge.provider_session_id
            for revision in revisions
        ):
            self.speaker_identities.merge_speaker_ids(
                _SCRIBE_SOURCE,
                speaker_merge.provider_session_id,
                speaker_merge.from_speaker_id,
                speaker_merge.to_speaker_id,
            )
        changed: set[UUID] = set()
        for revision in revisions:
            uid = revision.identity.segment.segment_id
            commit = self._scribe_commits.get(uid)
            if (
                commit is None
                or commit.native_segment != revision.native_segment
                or commit.provider_epoch != revision.identity.provider_epoch_id
                or not self.output.output_runtime.peer_publication_is_authorized(
                    commit.transcript.publication_generation, commit.transcript.source_order
                )
                or commit.transcript.final_speaker_runs == revision.final_speaker_runs
            ):
                continue
            commit.transcript = replace(
                commit.transcript, final_speaker_runs=revision.final_speaker_runs
            )
            changed.add(uid)
        if not changed:
            return
        for family in tuple(self._scribe_families.values()):
            if is_current is not None and not is_current():
                break
            if any(
                part.commit is not None and part.commit.transcript.utterance_id in changed
                for caption in family.captions
                for part in caption.scribe_parts
            ):
                if (
                    family.lineage is not None
                    and not self.output.output_runtime.peer_caption_lineage_is_live(family.lineage)
                ):
                    continue
                await self._revise_scribe_family(family, is_current)

    async def _revise_scribe_family(
        self, family: _ScribeFamily, is_current: Callable[[], bool] | None
    ) -> None:
        old = family.captions
        root = old[0]
        ranges: list[tuple[int, int]] = []
        offset = 0
        for caption in old:
            end = offset + len(caption.source)
            ranges.append((offset, end))
            offset = end
        parts = tuple(part for caption in old for part in caption.scribe_parts)
        first_commit = next(part.commit for part in parts if part.commit is not None)
        scope = first_commit.native_segment.provider_session_id
        text = "".join(caption.source for caption in old)
        aggregate = Transcript(
            root.utterance_id,
            text,
            True,
            channel="peer",
            final_language_runs=tuple(
                FinalLanguageRun(caption.source, caption.source_language or "") for caption in old
            ),
            final_speaker_runs=tuple(run for part in parts for run in part.runs(scope)),
            publication_generation=root.generation,
            source_order=root.source_order,
        )
        live = tuple(self._scribe_member_is_live(caption) for caption in old)
        groups = self._revised_source_groups(aggregate, old, ranges, live)
        exact = {
            (start, end, caption.source_language or ""): caption
            for (start, end), caption in zip(ranges, old, strict=True)
        }
        reused = {
            caption
            for group in groups
            if (
                caption := exact.get(
                    (group.source_text_start, group.source_text_end, group.language)
                )
            )
            is not None
        }
        layout: list[tuple[_Caption, Transcript, tuple[_Caption, ...], bool]] = []
        for group in groups:
            start, end = group.source_text_start, group.source_text_end
            overlaps = tuple(
                caption
                for (left, right), caption in zip(ranges, old, strict=True)
                if left < end and right > start
            )
            caption = exact.get((start, end, group.language))
            content_changed = caption is None
            if caption is None:
                caption = next((item for item in overlaps if item not in reused), None)
                if caption is None:
                    parent = overlaps[0]
                    caption = _Caption(
                        uuid4(),
                        parent.generation,
                        parent.source_order,
                        parent.provider_id,
                        parent.scope,
                        last_final_at=parent.last_final_at,
                        sealed=parent.sealed,
                        native_speakers=True,
                        scribe_family=family,
                        lineage=family.lineage,
                    )
                reused.add(caption)
            fragment = _final_transcript_fragment(
                aggregate, group, caption.utterance_id, preserve_speaker_runs=True
            )
            if fragment.source_order != caption.source_order:
                fragment = replace(fragment, source_order=caption.source_order)
            layout.append((caption, fragment, overlaps, content_changed))
        removed = tuple(caption for caption in old if caption not in reused)
        for caption in removed:
            caption.revision += 1
            for replacement, _fragment, overlaps, _changed in layout:
                if caption in overlaps:
                    replacement.final_admissions.update(caption.final_admissions)
                    replacement.final_admissions.add(caption.admitted)
                    replacement.absorbed_admissions.update(caption.absorbed_admissions)
                    break
            caption.final_admissions.clear()
            caption.absorbed_admissions.clear()
        next_captions = tuple(caption for caption, _text, _old, _changed in layout)
        family.captions = next_captions
        tail_was_in_family = any(
            self._tail is member.admitted or self._tail in member.final_admissions for member in old
        )
        repairs: list[tuple[_Caption, PreparedTranslationRequest | None, asyncio.Event]] = []
        patches: list[_Caption] = []
        inherited_context = root.context or ()
        predecessor = root.predecessor
        previous: _Caption | None = None
        for index, (caption, fragment, overlaps, content_changed) in enumerate(layout):
            if not content_changed and caption.source_runs == fragment.final_speaker_runs:
                previous = caption
                predecessor = caption.admitted
                inherited_context = (*inherited_context, caption.source)[-5:]
                continue
            assignment = self.speaker_identities.observe(
                fragment, child_sequence=index, native_speakers=True, late=True
            )
            self.speaker_identities.retire(fragment.utterance_id)
            changed_assignment = caption.speaker_assignment is None or (
                caption.speaker_assignment.attribution,
                caption.speaker_assignment.palette_index,
                caption.speaker_assignment.palette_overflow,
            ) != (assignment.attribution, assignment.palette_index, assignment.palette_overflow)
            caption.speaker_assignment = assignment
            caption.source_runs = fragment.final_speaker_runs
            caption.continuity_key = assignment.attribution.key
            if content_changed:
                caption.scribe_continues = overlaps[0].scribe_continues
                caption.revision += 1
                caption.source = fragment.text
                caption.source_language = fragment.final_language_runs[0].language or None
                start, end = fragment.source_text_range or (0, len(fragment.text))
                caption.scribe_parts = self._slice_scribe_parts(parts, start, end)
                caption.required_lineage_members = tuple(
                    dict.fromkeys(
                        uid
                        for item in overlaps
                        for uid in (
                            (item.utterance_id,)
                            if family.lineage is not None
                            and item.utterance_id in family.lineage.admitted_members
                            else (item.utterance_id, *item.required_lineage_members)
                        )
                        if uid != caption.utterance_id
                    )
                )
                if caption.context is None:
                    caption.context = inherited_context
                caption.predecessor = predecessor
                caption.history_after = previous.utterance_id if previous is not None else None
                covered = tuple(item for item in overlaps if item in removed)
                affected = tuple(
                    uid for item in covered for uid in (item.utterance_id, *item.history_removals)
                )
                caption.history_removals = tuple(
                    dict.fromkeys((*caption.history_removals, *affected))
                )
                caption.superseded = tuple(
                    dict.fromkeys(
                        (
                            *caption.superseded,
                            *affected,
                            *(uid for item in covered for uid in item.superseded),
                        )
                    )
                )
                prepared = self._prepare(caption, caption.source, "repair", family.config)
                admission = asyncio.Event()
                caption.final_admissions.add(admission)
                repairs.append((caption, prepared, admission))
            elif (
                changed_assignment
                and caption.displayed_revision
                and caption.lineage is not None
                and self._scribe_member_is_live(caption)
            ):
                patches.append(caption)
            previous = caption
            predecessor = admission if content_changed else caption.admitted
            inherited_context = (*inherited_context, caption.source)[-5:]
        if self._last in old:
            self._last = next_captions[-1]
            if tail_was_in_family:
                self._tail = predecessor or self._last.admitted
        if repairs:
            self._revise_scribe_context(old, next_captions, family.config)
            boundaries = {
                prior.utterance_id: caption.utterance_id
                for caption, _fragment, overlaps, _changed in layout
                for prior in overlaps
            }
            for retained in self._scribe_families.values():
                for member in retained.captions:
                    if member.scribe_continues in boundaries:
                        member.scribe_continues = boundaries[member.scribe_continues]
        for caption, prepared, admission in repairs:
            self._start(
                self._execute_scribe_request(
                    caption,
                    family.config,
                    prepared,
                    kind="repair",
                    revision=caption.revision,
                    admitted=admission,
                ),
                f"peer:repair:{caption.utterance_id}:{caption.revision}",
            )
        for caption in patches:
            await self._publish_scribe_attribution(caption, family.config, is_current)

    def _resolved_continuity(self, caption: _Caption) -> SpeakerKey | None:
        key = caption.continuity_key
        preceding = caption.scribe_continues
        if key is not None or preceding is None:
            return key
        retained = {
            member.utterance_id: member
            for family in self._scribe_families.values()
            for member in family.captions
        }
        for _ in range(_SCRIBE_RETENTION):
            if preceding not in retained:
                return caption.scribe_continuity
            previous = retained[preceding]
            if previous.continuity_key is not None:
                return previous.continuity_key
            preceding = previous.scribe_continues
        return caption.scribe_continuity

    def _scribe_member_is_live(self, caption: _Caption) -> bool:
        return (
            caption.lineage is None
            or self.output.output_runtime.peer_caption_lineage_members_are_live(
                caption.lineage,
                (
                    (caption.utterance_id,)
                    if caption.utterance_id in caption.lineage.admitted_members
                    else (caption.utterance_id, *caption.required_lineage_members)
                ),
            )
        )

    @staticmethod
    def _revised_source_groups(
        aggregate: Transcript,
        old: tuple[_Caption, ...],
        ranges: list[tuple[int, int]],
        live: tuple[bool, ...],
    ) -> tuple[_FinalTranscriptSegment, ...]:
        groups: list[_FinalTranscriptSegment] = []
        index = 0
        while index < len(old):
            start, end = ranges[index]
            if not live[index]:
                groups.append(
                    _FinalTranscriptSegment(
                        old[index].source,
                        old[index].source_language or "",
                        source_text_start=start,
                        source_text_end=end,
                    )
                )
                index += 1
                continue
            stop = index + 1
            while stop < len(old) and live[stop]:
                stop += 1
            end = ranges[stop - 1][1]
            if index == 0 and stop == len(old):
                unit = aggregate
            else:
                unit = replace(
                    aggregate,
                    text=aggregate.text[start:end],
                    final_language_runs=aggregate.final_language_runs[index:stop],
                    final_speaker_runs=_source_speaker_runs(aggregate, start, end),
                )
            segments = _native_peer_source_groups(unit)
            if segments:
                groups.extend(
                    replace(
                        segment,
                        source_text_start=segment.source_text_start + start,
                        source_text_end=segment.source_text_end + start,
                    )
                    for segment in segments
                )
            else:
                groups.append(
                    _FinalTranscriptSegment(
                        unit.text,
                        old[index].source_language or "",
                        source_text_start=start,
                        source_text_end=end,
                    )
                )
            index = stop
        return tuple(groups)

    @staticmethod
    def _slice_scribe_parts(
        parts: tuple[_ScribePart, ...], start: int, end: int
    ) -> tuple[_ScribePart, ...]:
        selected = []
        offset = 0
        for part in parts:
            stop = offset + part.size
            if offset < end and stop > start:
                selected.append(
                    _ScribePart(
                        part.commit,
                        part.start + max(0, start - offset),
                        part.start + min(part.size, end - offset),
                    )
                )
            if stop >= end:
                break
            offset = stop
        return tuple(selected)

    def _revise_scribe_context(
        self,
        old: tuple[_Caption, ...],
        current: tuple[_Caption, ...],
        config: TranslationRuntimeConfigSnapshot,
    ) -> None:
        history = self.runtime.translation_history
        ids = {caption.utterance_id for caption in old}
        indexes = [index for index, entry in enumerate(history) if entry.utterance_id in ids]
        if not indexes:
            return
        previous = {entry.utterance_id: entry for entry in history if entry.utterance_id in ids}
        target = self.requests.target_language_for("peer", config.value)
        revised = [
            ContextEntry(
                caption.source,
                caption.source_language
                or (previous.get(caption.utterance_id) or history[indexes[0]]).source_language,
                target,
                (
                    caption.last_final_at
                    if caption.last_final_at is not None
                    else history[indexes[0]].timestamp
                ),
                "peer",
                "peer",
                caption.utterance_id,
            )
            for caption in current
        ]
        history[indexes[0] :] = [
            *revised,
            *(entry for entry in history[indexes[0] :] if entry.utterance_id not in ids),
        ]
        del history[:-5]

    @staticmethod
    def _settle_scribe_admission(caption: _Caption) -> None:
        caption.admitted.set()
        for admission in caption.final_admissions:
            admission.set()
        caption.final_admissions.clear()
        for admission in caption.absorbed_admissions.values():
            admission.set()
        caption.absorbed_admissions.clear()

    async def _execute_scribe_request(
        self,
        caption: _Caption,
        config: TranslationRuntimeConfigSnapshot,
        prepared: PreparedTranslationRequest | None,
        *,
        kind: PeerRequestKind,
        revision: int,
        admitted: asyncio.Event,
    ) -> None:
        epoch = self._epoch
        await self._slots.acquire()
        if (
            epoch != self._epoch
            or revision != caption.revision
            or (
                caption.lineage is not None
                and not self.output.output_runtime.peer_caption_lineage_members_are_live(
                    caption.lineage, (caption.utterance_id, *caption.required_lineage_members)
                )
            )
        ):
            self._slots.release()
            if revision == caption.revision:
                self._settle_scribe_admission(caption)
            return
        await self._execute(
            caption,
            caption.source,
            kind,
            revision,
            config,
            prepared,
            superseded=caption.superseded,
            detected_language=caption.source_language,
            admitted=admitted,
            predecessor=caption.predecessor,
        )

    async def _publish_scribe_attribution(
        self,
        caption: _Caption,
        config: TranslationRuntimeConfigSnapshot,
        is_current: Callable[[], bool] | None,
    ) -> None:
        lineage = caption.lineage
        if lineage is None:
            return
        assignment = caption.speaker_assignment
        epoch = self._epoch
        revision = caption.revision
        await self.output.project_peer_overlay(
            TranslationOutputSubmission(
                caption.utterance_id,
                caption.utterance_id,
                0,
                "peer",
                "Peer",
                "",
                caption.source_language,
                self.requests.target_language_for("peer", config.value),
                "source_only",
                config,
                publication_generation=caption.generation,
                source_order=caption.source_order,
                peer_caption_revision=caption.displayed_revision,
                speaker_assignment=assignment,
                peer_lineage_revision=PeerCaptionRevision(
                    lineage,
                    attribution_only=True,
                    is_current=lambda: (
                        epoch == self._epoch
                        and revision == caption.revision
                        and assignment == caption.speaker_assignment
                        and (is_current is None or is_current())
                    ),
                ),
            )
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
        *,
        admitted: asyncio.Event,
        predecessor: asyncio.Event | None,
    ) -> None:
        epoch = self._epoch
        scribe_plan = caption.scribe_family is not None
        required_members = caption.required_lineage_members

        def is_current() -> bool:
            return epoch == self._epoch and (not scribe_plan or caption.revision == revision)

        source_language = prepared.source_language if prepared is not None else detected_language
        target_language = self.requests.target_language_for("peer", config.value)
        try:
            if not is_current():
                return
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
            if not is_current():
                return
            if scribe_plan:
                assignment = caption.speaker_assignment
            submission = replace(
                submission,
                speaker_assignment=assignment,
                peer_history_after=caption.history_after if scribe_plan else None,
                peer_history_removals=caption.history_removals if scribe_plan else (),
                peer_lineage_revision=(
                    PeerCaptionRevision(caption.lineage, required_members, is_current=is_current)
                    if caption.lineage is not None
                    else None
                ),
                translation=(
                    None
                    if submission.translation is None
                    else replace(submission.translation, speaker_assignment=assignment)
                ),
            )
            if submission.failure_code == "stale_provider_completion":
                return
            if (
                not is_current()
                or (segment is not None and segment.closed)
                or revision < caption.displayed_revision
            ):
                return
            if kind == "preview" and (
                submission.translation is None or revision < caption.revision
            ):
                return
            if not self.output.output_runtime.register_peer_caption_identity(
                caption.utterance_id, caption.generation, caption.source_order
            ):
                return
            caption.displayed_revision = revision
            await self.output.project_peer_overlay(submission)
            if predecessor is not None and not admitted.is_set():
                await predecessor.wait()
            if (
                not is_current()
                or revision < caption.displayed_revision
                or not self.output.output_runtime.peer_publication_is_authorized(
                    caption.generation, caption.source_order
                )
            ):
                return
            caption.shown_revision = revision
            await self.output.project_peer_caption(submission, overlay_projected=True)
        finally:
            if scribe_plan:
                if kind != "preview" and revision == caption.revision:
                    self._settle_scribe_admission(caption)
            elif (
                kind != "preview" and revision >= caption.displayed_revision
            ) or caption.shown_revision:
                admitted.set()
                caption.admitted.set()
                for utterance_id in superseded:
                    absorbed = caption.absorbed_admissions.pop(utterance_id, None)
                    if absorbed is not None:
                        absorbed.set()
            self._slots.release()
