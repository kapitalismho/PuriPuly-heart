from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import AsyncIterator, Literal, Protocol, TypeVar, runtime_checkable
from uuid import UUID

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    SegmentTerminalOutcome,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun, SpeakerAttribution
from puripuly_heart.domain.recognition import (
    NativeTranscriptionEvidence,
    RecognitionStreamIdentity,
    RecognitionTextOrigin,
    RecognitionUnitIdentity,
    slice_recognition_origins,
    validate_recognition_origins,
)

MAX_SPEAKER_AUDIO_PARTS = 128


class PermanentSTTScopedSessionError(RuntimeError):
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class STTSessionProjection:
    mode: Literal["legacy", "scoped"] = "legacy"
    provider_epoch_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "scoped":
            if not self.provider_epoch_id:
                raise ValueError("scoped STT session projection requires a provider epoch")
            return
        if self.mode != "legacy":
            raise ValueError(f"unknown STT session projection mode: {self.mode!r}")
        if self.provider_epoch_id is not None:
            raise ValueError("legacy STT session projection cannot have a provider epoch")


LEGACY_STT_SESSION_PROJECTION = STTSessionProjection()


@dataclass(frozen=True, slots=True)
class STTProviderTurnIdentity:
    segment: AudioSegmentIdentity
    provider_epoch_id: str
    provider_turn_id: str
    settings_scope: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class STTProviderTurnRequest:
    identity: STTProviderTurnIdentity
    settings: AudioSegmentSettingsSnapshot
    channel: Literal["self", "peer"] = "peer"
    display_group_id: UUID | None = None
    isolate_failure: bool = False
    native_options: object | None = None
    optional_cutoff_at_monotonic_s: float | None = None
    on_native_event: Callable[[STTProviderTurnUpdate | STTProviderTurnTerminal], None] | None = (
        field(default=None, compare=False, repr=False)
    )


@dataclass(frozen=True, slots=True)
class STTSpeakerAudioPart:
    text_start: int
    text_end: int
    source_ranges: tuple[AudioCaptureSpan, ...]
    context_ranges: tuple[AudioCaptureSpan, ...]
    speaker_attribution: SpeakerAttribution
    speaker_proof_id: str | None = None


@dataclass(frozen=True, slots=True)
class STTSourceAudio:
    contribution_id: str
    source_ranges: tuple[AudioCaptureSpan, ...]
    context_ranges: tuple[AudioCaptureSpan, ...] = ()
    display_group_id: UUID | None = None
    speaker_attribution: SpeakerAttribution | None = None
    speaker_proof_id: str | None = None
    speaker_parts: tuple[STTSpeakerAudioPart, ...] = ()

    @property
    def speaker_proof_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                token
                for token in (
                    self.speaker_proof_id,
                    *(part.speaker_proof_id for part in self.speaker_parts),
                )
                if token is not None
            )
        )

    def __post_init__(self) -> None:
        if not self.contribution_id:
            raise ValueError("batch audio requires a contribution identity")


def discard_source_audio_proofs(event, evidence, *, retained=(), published=False):
    if evidence is None:
        return
    retained_tokens = {
        token
        for queued in retained
        for audio in getattr(queued, "source_audio", ())
        for token in audio.speaker_proof_ids
    }
    for audio in getattr(event, "source_audio", ()):
        for token in audio.speaker_proof_ids:
            if token not in retained_tokens:
                evidence.release_unadopted(token, published=published)


@dataclass(frozen=True, slots=True)
class STTSourceCapabilities:
    backend_id: str
    model_id: str | None = None
    policy: Literal["timed", "whole_span", "unknown"] = "unknown"
    optional_progress: bool = False


@dataclass(frozen=True, slots=True)
class STTAudioClock:
    clock_id: str
    sample_rate_hz: int
    origin_sent_sample: int | None = None
    qualified: bool = False

    def __post_init__(self) -> None:
        if not self.clock_id or self.sample_rate_hz <= 0:
            raise ValueError("invalid source clock")


@dataclass(frozen=True, slots=True)
class STTTextTiming:
    contribution_id: str
    text_start: int
    text_end: int
    semantics: Literal["word_interval", "subword_interval", "token_point", "result_span"]
    raw_start: float | None = None
    raw_end: float | None = None
    unit: Literal["s", "ms"] = "s"
    clock_id: str | None = None
    complete_boundary: bool = False
    estimated: bool = False

    def __post_init__(self) -> None:
        if not self.contribution_id or self.text_start < 0 or self.text_end < self.text_start:
            raise ValueError("invalid text timing range")


@dataclass(frozen=True, slots=True)
class STTNativeProvenance:
    native_event_id: str | None = None
    native_request_id: str | None = None
    native_item_id: str | None = None
    native_task_id: str | None = None
    barrier: str | None = None
    from_finalize: bool | None = None
    transcription: NativeTranscriptionEvidence | None = None
    capabilities: STTSourceCapabilities | None = None
    text_timing: tuple[STTTextTiming, ...] = ()
    audio_clocks: tuple[STTAudioClock, ...] = ()
    native_audio_progress: tuple[tuple[str, float], ...] = ()
    sent_until_sample: int | None = None
    control_id: str | None = None
    control_kind: str | None = None
    control_sent_sample: int | None = None
    native_model_metadata: tuple[tuple[str, str], ...] = ()
    unresolved_source_text: str | None = None


@dataclass(frozen=True, slots=True)
class STTTextContribution:
    contribution_id: str
    text_start: int
    text_end: int

    def __post_init__(self) -> None:
        if not self.contribution_id:
            raise ValueError("contribution_id must be non-empty")
        if self.text_start < 0 or self.text_end < self.text_start:
            raise ValueError("invalid contribution text range")


@dataclass(frozen=True, slots=True)
class STTSourceRelease:
    release_id: str
    text_start: int
    text_end: int
    reason: Literal["end", "manual", "drain", "boundary"]

    def __post_init__(self) -> None:
        if (
            not self.release_id
            or self.text_start < 0
            or self.text_end < self.text_start
            or self.reason not in {"end", "manual", "drain", "boundary"}
        ):
            raise ValueError("invalid source release")


@dataclass(frozen=True, slots=True)
class STTProviderTurnUpdate:
    identity: STTProviderTurnIdentity
    sequence: int
    stability: Literal["provisional", "stable"]
    assembly: Literal["append", "replace"]
    text: str
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
    provenance: STTNativeProvenance = STTNativeProvenance()
    contribution: STTTextContribution | None = None
    recognition_origins: tuple[RecognitionTextOrigin, ...] = ()
    is_current: Callable[[], bool] | None = field(default=None, compare=False, repr=False)
    source_release: STTSourceRelease | None = None
    included_contributions: tuple[STTTextContribution, ...] = ()
    source_audio: tuple[STTSourceAudio, ...] = ()


@dataclass(frozen=True, slots=True)
class STTProviderTurnTerminal:
    identity: STTProviderTurnIdentity
    outcome: SegmentTerminalOutcome
    text: str = ""
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
    text_authority: Literal["authoritative", "degraded", "none"] = "none"
    failure_reason: str | None = None
    epoch_disposition: Literal["reuse", "retire"] = "reuse"
    failure_retryable: bool = False
    recovery_pending: bool = False
    provenance: tuple[STTNativeProvenance, ...] = ()
    included_contributions: tuple[STTTextContribution, ...] = ()
    failure_diagnostics: ErrorDiagnostics | None = None
    recognition_origins: tuple[RecognitionTextOrigin, ...] = ()
    is_current: Callable[[], bool] | None = field(default=None, compare=False, repr=False)
    source_audio: tuple[STTSourceAudio, ...] = ()


_ProviderEvent = TypeVar("_ProviderEvent")


def scoped_event_for_delivery(
    event: _ProviderEvent,
) -> _ProviderEvent | STTProviderTurnTerminal | None:
    if not isinstance(event, STTProviderTurnUpdate | STTProviderTurnTerminal):
        return event
    if event.is_current is None or event.is_current():
        return event
    if isinstance(event, STTProviderTurnUpdate):
        return None
    return replace(
        event,
        outcome="cancelled",
        text="",
        final_language_runs=(),
        final_speaker_runs=(),
        text_authority="none",
        failure_reason="cancelled",
        epoch_disposition="retire",
        included_contributions=(),
        recognition_origins=(),
        source_audio=(),
        is_current=None,
    )


@dataclass(frozen=True, slots=True)
class STTAudioBoundary:
    boundary_id: str
    capture_epoch: int
    normalized_sample: int
    observed_at_monotonic_s: float
    reason: Literal["speaker", "natural", "hardcut"]
    speaker_epoch: str | None = None
    transition_start_sample: int | None = None
    transition_end_sample: int | None = None
    successor_onset_sample: int | None = None

    def __post_init__(self) -> None:
        if not self.boundary_id or self.capture_epoch < 0 or self.normalized_sample < 0:
            raise ValueError("invalid audio boundary")
        if (self.transition_start_sample is None) != (self.transition_end_sample is None):
            raise ValueError("audio transition requires both interval endpoints")
        if self.transition_start_sample is not None and (
            self.transition_start_sample < 0
            or self.transition_start_sample > self.transition_end_sample
            or self.normalized_sample != self.transition_end_sample
        ):
            raise ValueError("invalid audio transition interval")
        if self.successor_onset_sample is not None and (
            self.transition_start_sample is None
            or not self.transition_start_sample
            <= self.successor_onset_sample
            <= self.transition_end_sample
        ):
            raise ValueError("invalid successor onset")


@dataclass(frozen=True, slots=True)
class STTProviderAudioBoundary:
    identity: STTProviderTurnIdentity
    boundary: STTAudioBoundary

    def __post_init__(self) -> None:
        if self.identity.segment.capture_epoch != self.boundary.capture_epoch:
            raise ValueError("audio boundary belongs to another capture epoch")


@dataclass(frozen=True, slots=True)
class STTConsumedContribution:
    identity: STTProviderTurnIdentity
    contribution_id: str
    text_start: int
    text_end: int
    text: str


class STTContributionConsumptionLedger:
    def __init__(
        self, *, identity_capacity: int = 4096, contribution_capacity: int = 16384
    ) -> None:
        if identity_capacity < 1 or contribution_capacity < 1:
            raise ValueError("ledger capacities must be positive")
        self._identity_capacity = identity_capacity
        self._contribution_capacity = contribution_capacity
        self._consumed: dict[tuple[STTProviderTurnIdentity, str], list[tuple[int, int]]] = {}
        self._contribution_ranges: dict[tuple[STTProviderTurnIdentity, str], tuple[int, int]] = {}
        self._consumed_terminal_tails: set[STTProviderTurnIdentity] = set()
        self._known_identities: set[STTProviderTurnIdentity] = set()
        self._identity_order: deque[STTProviderTurnIdentity] = deque()
        self._retired_floor: dict[tuple[int, int], int] = {}
        self._generation_floor = -1
        self._boundaries: dict[STTProviderTurnIdentity, dict[str, STTAudioBoundary]] = {}
        self._sealed: set[STTProviderTurnIdentity] = set()
        self._releases: dict[STTProviderTurnIdentity, dict[str, STTSourceRelease]] = {}
        self._admitted_prefixes: dict[STTProviderTurnIdentity, int] = {}

    @property
    def consumed_contribution_ids(self) -> frozenset[str]:
        return frozenset(contribution_id for _identity, contribution_id in self._consumed)

    def record_boundary(
        self, identity: STTProviderTurnIdentity, boundary: STTAudioBoundary
    ) -> None:
        if boundary.capture_epoch != identity.segment.capture_epoch:
            raise ValueError("audio boundary belongs to another capture epoch")
        if not self._remember_identity(identity):
            return
        boundaries = self._boundaries.setdefault(identity, {})
        previous = boundaries.get(boundary.boundary_id)
        if previous is not None and previous != boundary:
            raise ValueError("audio boundary identity changed")
        if previous is None and len(boundaries) >= 256:
            raise ValueError("audio boundary capacity exceeded")
        boundaries[boundary.boundary_id] = boundary

    def boundaries(self, identity: STTProviderTurnIdentity) -> tuple[STTAudioBoundary, ...]:
        return tuple(self._boundaries.get(identity, {}).values())

    def seal_audio(self, identity: STTProviderTurnIdentity) -> None:
        if self._remember_identity(identity):
            self._sealed.add(identity)

    def lifecycle(self, identity: STTProviderTurnIdentity) -> tuple[bool, bool]:
        return identity in self._sealed, identity in self._consumed_terminal_tails

    def consume(self, event: STTProviderTurnUpdate | STTProviderTurnTerminal) -> str:
        return "".join(piece.text for piece in self.consume_parts(event))

    def consume_with_origins(
        self,
        event: STTProviderTurnUpdate | STTProviderTurnTerminal,
    ) -> tuple[str, tuple[RecognitionTextOrigin, ...]]:
        text = ""
        origins: tuple[RecognitionTextOrigin, ...] = ()
        for piece in self.consume_parts(event):
            offset = len(text)
            text += piece.text
            origins += tuple(
                replace(
                    origin, text_start=origin.text_start + offset, text_end=origin.text_end + offset
                )
                for origin in slice_recognition_origins(
                    event.recognition_origins, piece.text_start, piece.text_end
                )
            )
        return text, origins

    def consume_parts(
        self,
        event: STTProviderTurnUpdate | STTProviderTurnTerminal,
        *,
        text_range: tuple[int, int] | None = None,
    ) -> tuple[STTConsumedContribution, ...]:
        release = event.source_release if isinstance(event, STTProviderTurnUpdate) else None
        if release is not None and (
            not event.identity.provider_epoch_id or not event.identity.provider_turn_id
        ):
            raise ValueError("source release requires provider request ownership")
        segment = event.identity.segment
        if event.identity not in self._known_identities and (
            segment.activation_generation <= self._generation_floor
            or segment.segment_order
            <= self._retired_floor.get((segment.activation_generation, segment.capture_epoch), -1)
        ):
            return ()
        if event.identity in self._consumed_terminal_tails:
            return ()
        validate_recognition_origins(event.recognition_origins, event.text)
        if isinstance(event, STTProviderTurnUpdate):
            if event.stability != "stable":
                if release is not None:
                    raise ValueError("source release requires stable text")
                return ()
            contributions = event.included_contributions or (
                (event.contribution,) if event.contribution is not None else ()
            )
            if release is None and not contributions:
                return ()
            if release is not None:
                if release.text_end > len(event.text):
                    raise ValueError("source release exceeds assembled text")
                releases = self._releases.get(event.identity, {})
                previous = releases.get(release.release_id)
                if previous is not None and previous != release:
                    raise ValueError("source release identity changed")
                if previous is None and len(releases) >= 256:
                    raise ValueError("source release capacity exceeded")
                if len({item.contribution_id for item in contributions}) != len(contributions):
                    raise ValueError("source release contribution identities repeat")
                cursor, previous_end = release.text_start, 0
                for item in contributions:
                    if item.text_start < previous_end or item.text_end > len(event.text):
                        raise ValueError("source release contribution ranges are invalid")
                    previous_end = item.text_end
                    left, right = (
                        max(item.text_start, release.text_start),
                        min(item.text_end, release.text_end),
                    )
                    if left < right:
                        if left != cursor:
                            raise ValueError("source release leaves unresolved ownership")
                        cursor = right
                if cursor != release.text_end:
                    raise ValueError("source release leaves unresolved ownership")
        else:
            contributions = event.included_contributions
            if (
                event.text
                and not contributions
                and any(identity == event.identity for identity, _ in self._consumed)
            ):
                raise ValueError("terminal source association is missing")
            if len({item.contribution_id for item in contributions}) != len(contributions):
                raise ValueError("terminal contribution identities repeat")
            cursor = 0
            for contribution in contributions:
                if contribution.text_start < cursor or contribution.text_end > len(event.text):
                    raise ValueError("terminal contribution ranges are invalid")
                cursor = contribution.text_end
            assembled: list[STTTextContribution] = []
            cursor = 0
            for contribution in contributions:
                if contribution.text_start > cursor:
                    assembled.append(
                        STTTextContribution(
                            f"terminal:{cursor}:{contribution.text_start}",
                            cursor,
                            contribution.text_start,
                        )
                    )
                assembled.append(contribution)
                cursor = contribution.text_end
            if cursor < len(event.text):
                assembled.append(
                    STTTextContribution(
                        f"terminal:{cursor}:{len(event.text)}", cursor, len(event.text)
                    )
                )
            contributions = tuple(assembled)
        selection_start, selection_end = text_range or (
            (release.text_start, release.text_end) if release is not None else (0, len(event.text))
        )
        if release is not None and (
            selection_start < release.text_start or selection_end > release.text_end
        ):
            raise ValueError("selected source range exceeds release")
        if (
            selection_start < 0
            or selection_end > len(event.text)
            or selection_end < selection_start
        ):
            raise ValueError("invalid selected source range")
        if event.source_audio:
            audio_ids = {item.contribution_id for item in event.source_audio}
            if len(audio_ids) != len(event.source_audio):
                raise ValueError("batch audio ownership identities repeat")
            if any(
                span.capture_epoch != event.identity.segment.capture_epoch
                for item in event.source_audio
                for span in (*item.source_ranges, *item.context_ranges)
            ):
                raise ValueError("batch audio belongs to another capture epoch")
            if any(
                item.contribution_id not in audio_ids
                for item in contributions
                if item.text_start < selection_end and item.text_end > selection_start
            ):
                raise ValueError("batch contribution has unresolved audio ownership")
        prefix_end = self._admitted_prefixes.get(event.identity, 0)
        if release is not None and selection_start > prefix_end:
            raise ValueError("source release skips earlier unconsumed text")
        pieces: list[STTConsumedContribution] = []
        staged: list[tuple[tuple[STTProviderTurnIdentity, str], list[tuple[int, int]]]] = []
        for contribution in contributions:
            declared = (contribution.text_start, contribution.text_end)
            known = self._contribution_ranges.get((event.identity, contribution.contribution_id))
            if known is not None and known != declared:
                raise ValueError("contribution ownership range changed")
            if contribution.text_end > len(event.text):
                raise ValueError("contribution range exceeds assembled text")
            key = (event.identity, contribution.contribution_id)
            start = max(selection_start, contribution.text_start)
            end = min(selection_end, contribution.text_end)
            remaining = [(start, end)] if start < end else []
            for used_start, used_end in self._consumed.get(key, ()):
                next_ranges = []
                for left, right in remaining:
                    if used_end <= left or used_start >= right:
                        next_ranges.append((left, right))
                    else:
                        if left < used_start:
                            next_ranges.append((left, used_start))
                        if used_end < right:
                            next_ranges.append((used_end, right))
                remaining = next_ranges
            for left, right in remaining:
                pieces.append(
                    STTConsumedContribution(
                        event.identity,
                        contribution.contribution_id,
                        left,
                        right,
                        event.text[left:right],
                    )
                )
            staged.append((key, remaining))
        if release is not None:
            for piece in pieces:
                if piece.text_start != prefix_end:
                    raise ValueError("source release changes admitted prefix ownership")
                prefix_end = piece.text_end
        new_keys = {key for key, ranges in staged if ranges and key not in self._consumed}
        if len(self._consumed) + len(new_keys) > self._contribution_capacity:
            raise ValueError("source contribution capacity exceeded")
        commits = []
        for contribution, (key, ranges) in zip(contributions, staged):
            if ranges:
                combined = sorted((*self._consumed.get(key, ()), *ranges))
                merged: list[tuple[int, int]] = []
                for left, right in combined:
                    if merged and left <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                    else:
                        merged.append((left, right))
                if len(merged) > 256:
                    raise ValueError("source split capacity exceeded")
                commits.append((key, merged, (contribution.text_start, contribution.text_end)))
        if not self._remember_identity(event.identity):
            return ()
        for key, merged, declared in commits:
            self._consumed[key] = merged
            self._contribution_ranges[key] = declared
        if release is not None:
            self._releases.setdefault(event.identity, {})[release.release_id] = release
            self._admitted_prefixes[event.identity] = prefix_end
        if isinstance(event, STTProviderTurnTerminal) and text_range is None:
            self.retire(event.identity)
        return tuple(pieces)

    def retire(self, identity: STTProviderTurnIdentity) -> None:
        if self._remember_identity(identity):
            self._consumed_terminal_tails.add(identity)
            self._releases.pop(identity, None)
            self._admitted_prefixes.pop(identity, None)
            self._consumed = {
                key: ranges for key, ranges in self._consumed.items() if key[0] != identity
            }
            self._contribution_ranges = {
                key: span for key, span in self._contribution_ranges.items() if key[0] != identity
            }

    def retire_known(self, identity: STTProviderTurnIdentity) -> bool:
        if identity not in self._known_identities:
            return False
        self.retire(identity)
        return True

    def retire_epoch(
        self,
        provider_epoch_id: str,
        *,
        is_obsolete: Callable[[STTProviderTurnIdentity], bool],
    ) -> bool:
        identities = {
            identity
            for identity in self._known_identities
            if identity.provider_epoch_id == provider_epoch_id and is_obsolete(identity)
        }
        if not identities:
            return False
        self._consumed_terminal_tails.update(identities)
        self._consumed = {
            key: ranges for key, ranges in self._consumed.items() if key[0] not in identities
        }
        self._contribution_ranges = {
            key: span for key, span in self._contribution_ranges.items() if key[0] not in identities
        }
        self._releases = {
            identity: releases
            for identity, releases in self._releases.items()
            if identity not in identities
        }
        self._admitted_prefixes = {
            identity: prefix_end
            for identity, prefix_end in self._admitted_prefixes.items()
            if identity not in identities
        }
        return True

    def retire_all(self) -> None:
        self._consumed_terminal_tails.update(self._known_identities)
        self._consumed.clear()
        self._contribution_ranges.clear()
        self._releases.clear()
        self._admitted_prefixes.clear()

    def _remember_identity(self, identity: STTProviderTurnIdentity) -> bool:
        if identity in self._known_identities:
            return True
        segment = identity.segment
        scope = (segment.activation_generation, segment.capture_epoch)
        if segment.activation_generation <= self._generation_floor or (
            segment.segment_order <= self._retired_floor.get(scope, -1)
        ):
            return False
        expired = (
            next(
                (item for item in self._identity_order if item in self._consumed_terminal_tails),
                None,
            )
            if len(self._identity_order) >= self._identity_capacity
            else None
        )
        if len(self._identity_order) >= self._identity_capacity and expired is None:
            raise ValueError("unfinished source identity capacity exceeded")
        self._known_identities.add(identity)
        self._identity_order.append(identity)
        if expired is not None:
            self._identity_order.remove(expired)
            old = expired.segment
            old_scope = (old.activation_generation, old.capture_epoch)
            self._retired_floor[old_scope] = max(
                old.segment_order, self._retired_floor.get(old_scope, -1)
            )
            if len(self._retired_floor) > self._identity_capacity:
                generation = min(scope[0] for scope in self._retired_floor)
                self._generation_floor = max(self._generation_floor, generation)
                self._retired_floor = {
                    scope: order
                    for scope, order in self._retired_floor.items()
                    if scope[0] > self._generation_floor
                }
            self._known_identities.discard(expired)
            self._consumed_terminal_tails.discard(expired)
            self._boundaries.pop(expired, None)
            self._admitted_prefixes.pop(expired, None)
            self._sealed.discard(expired)
            self._releases.pop(expired, None)
            self._consumed = {
                key: ranges for key, ranges in self._consumed.items() if key[0] != expired
            }
            self._contribution_ranges = {
                key: span for key, span in self._contribution_ranges.items() if key[0] != expired
            }
        return True


@dataclass(frozen=True, slots=True)
class STTProviderEpochEnded:
    provider_epoch_id: str
    orderly: bool
    reason: str
    provider_turn_id: str | None = None
    failure_retryable: bool = False
    failure_diagnostics: ErrorDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class STTProviderInputTerminal:
    identity: STTProviderTurnIdentity
    outcome: Literal["submitted", "failed", "expired", "cancelled"]
    channel: Literal["self", "peer"]
    failure_reason: str | None = None
    epoch_disposition: Literal["reuse", "retire"] = "reuse"
    recovery_pending: bool = False
    failure_diagnostics: ErrorDiagnostics | None = None
    independent_recognition: bool = False


@dataclass(frozen=True, slots=True)
class STTRecognitionUnit:
    identity: RecognitionUnitIdentity
    text: str
    provenance: STTNativeProvenance = STTNativeProvenance()
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
    estimated_last_speech_at: float | None = None

    @property
    def retained_bytes(self) -> int:
        size = len(self.text.encode("utf-8"))
        native = self.provenance.transcription
        if native is not None:
            size += native.retained_bytes
        return size


@dataclass(frozen=True, slots=True)
class STTRecognitionUnitTerminal:
    unit: STTRecognitionUnit
    outcome: Literal["final", "empty", "failed", "cancelled"]
    failure_reason: str | None = None


STTProviderTurnEvent = (
    STTProviderTurnUpdate
    | STTProviderTurnTerminal
    | STTProviderEpochEnded
    | STTProviderInputTerminal
    | STTRecognitionUnit
    | STTProviderAudioBoundary
    | STTRecognitionUnitTerminal
)


@runtime_checkable
class STTScopedTurnSession(Protocol):
    async def begin_turn(self, request: STTProviderTurnRequest) -> None: ...
    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None: ...
    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None: ...
    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None: ...
    async def turn_events(self) -> AsyncIterator[STTProviderTurnEvent]: ...
    async def stop(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class STTIndependentRecognitionSession(Protocol):
    @property
    def accepts_stream_input(self) -> bool: ...
    @property
    def independent_recognition_units(self) -> bool: ...
    async def begin_stream(self, stream: RecognitionStreamIdentity) -> None: ...
    async def send_stream_audio(
        self, pcm16le: bytes, *, source_ranges: tuple[AudioCaptureSpan, ...]
    ) -> None: ...
    async def end_stream(self, *, reason: str) -> None: ...
    def recognition_source_covers(self, ranges: tuple[AudioCaptureSpan, ...]) -> bool: ...


@dataclass(frozen=True, slots=True)
class STTBackendTranscriptEvent:
    text: str
    is_final: bool
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()


class RecoverableSTTSessionError(RuntimeError):
    pass


class STTBackendSession(Protocol):
    async def send_audio(self, pcm16le: bytes) -> None: ...
    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None: ...  # Backend-specific end-of-speech handling
    async def stop(self) -> None: ...
    async def close(self) -> None: ...
    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]: ...


@runtime_checkable
class STTBackendFloat32Session(Protocol):
    async def send_audio_f32(self, samples_f32: np.ndarray) -> None: ...


class STTBackend(Protocol):
    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession: ...


@runtime_checkable
class LocalASRReconfigurableBackend(Protocol):
    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None: ...
