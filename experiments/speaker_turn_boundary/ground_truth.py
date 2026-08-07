from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransitionKind = Literal[
    "clean_handoff",
    "interruption_onset",
    "gap_speaker_change",
    "speaker_left",
    "initial_start",
    "same_speaker",
    "gap_same_speaker",
    "silence",
    "silence_start",
    "ambiguous",
    "ambiguous_adjacent",
]

POSITIVE_KINDS = frozenset({"clean_handoff", "interruption_onset", "gap_speaker_change"})


class GroundTruthValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpeakerRegion:
    audio_epoch: int
    start_sample: int
    end_sample: int
    speakers: frozenset[str]
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if self.audio_epoch < 0:
            raise GroundTruthValidationError(f"audio_epoch must be >= 0, got {self.audio_epoch}")
        if self.start_sample < 0:
            raise GroundTruthValidationError(f"start_sample must be >= 0, got {self.start_sample}")
        if self.end_sample <= self.start_sample:
            raise GroundTruthValidationError(
                "end_sample must be greater than start_sample "
                f"({self.end_sample} <= {self.start_sample})"
            )
        for speaker in self.speakers:
            if not isinstance(speaker, str) or not speaker:
                raise GroundTruthValidationError(
                    f"speaker labels must be non-empty strings, got {speaker!r}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "audio_epoch": self.audio_epoch,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "speakers": sorted(self.speakers),
            "ambiguous": self.ambiguous,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SpeakerRegion":
        return cls(
            audio_epoch=int(data["audio_epoch"]),
            start_sample=int(data["start_sample"]),
            end_sample=int(data["end_sample"]),
            speakers=frozenset(str(s) for s in data["speakers"]),
            ambiguous=bool(data["ambiguous"]),
        )


@dataclass(frozen=True, slots=True)
class SpeakerChangeGT:
    audio_epoch: int
    change_sample: int
    kind: Literal["clean_handoff", "interruption_onset", "gap_speaker_change"]
    prev_speakers: frozenset[str]
    next_speakers: frozenset[str]


@dataclass(frozen=True, slots=True)
class RegionTransition:
    audio_epoch: int
    prev_start_sample: int
    prev_speakers: frozenset[str]
    next_start_sample: int
    next_speakers: frozenset[str]
    kind: TransitionKind
    positive: bool
    ambiguous: bool


def validate_region_sequence(regions: list[SpeakerRegion]) -> None:
    if not regions:
        return
    epoch = regions[0].audio_epoch
    prev_end = regions[0].start_sample
    for index, region in enumerate(regions):
        if region.audio_epoch != epoch:
            raise GroundTruthValidationError(
                "regions must share one audio_epoch "
                f"(region {index} has {region.audio_epoch}, expected {epoch})"
            )
        if region.start_sample != prev_end:
            raise GroundTruthValidationError(
                "regions must be contiguous and ordered "
                f"(region {index} starts at {region.start_sample}, expected {prev_end})"
            )
        prev_end = region.end_sample


def classify_active_speaker_transitions(
    regions: list[SpeakerRegion],
) -> tuple[list[SpeakerChangeGT], list[RegionTransition]]:
    validate_region_sequence(regions)
    transitions: list[RegionTransition] = []
    changes: list[SpeakerChangeGT] = []
    if len(regions) < 2:
        return changes, transitions
    epoch = regions[0].audio_epoch
    first = regions[0]
    excluded = first.ambiguous
    last_active: frozenset[str] | None = None
    if not first.ambiguous and first.speakers:
        last_active = first.speakers
    gap_pending = False
    for index in range(1, len(regions)):
        prev = regions[index - 1]
        current = regions[index]
        if current.ambiguous:
            transitions.append(
                RegionTransition(
                    audio_epoch=epoch,
                    prev_start_sample=prev.start_sample,
                    prev_speakers=prev.speakers,
                    next_start_sample=current.start_sample,
                    next_speakers=current.speakers,
                    kind="ambiguous",
                    positive=False,
                    ambiguous=True,
                )
            )
            excluded = True
            last_active = None
            gap_pending = False
            continue
        if excluded:
            transitions.append(
                RegionTransition(
                    audio_epoch=epoch,
                    prev_start_sample=prev.start_sample,
                    prev_speakers=prev.speakers,
                    next_start_sample=current.start_sample,
                    next_speakers=current.speakers,
                    kind="ambiguous_adjacent",
                    positive=False,
                    ambiguous=True,
                )
            )
            excluded = False
            if current.speakers:
                last_active = current.speakers
            gap_pending = False
            continue
        if current.speakers == prev.speakers:
            if not current.speakers:
                transitions.append(
                    RegionTransition(
                        audio_epoch=epoch,
                        prev_start_sample=prev.start_sample,
                        prev_speakers=prev.speakers,
                        next_start_sample=current.start_sample,
                        next_speakers=current.speakers,
                        kind="silence",
                        positive=False,
                        ambiguous=False,
                    )
                )
                continue
            transitions.append(
                RegionTransition(
                    audio_epoch=epoch,
                    prev_start_sample=prev.start_sample,
                    prev_speakers=prev.speakers,
                    next_start_sample=current.start_sample,
                    next_speakers=current.speakers,
                    kind="same_speaker",
                    positive=False,
                    ambiguous=False,
                )
            )
            continue
        if not current.speakers:
            transitions.append(
                RegionTransition(
                    audio_epoch=epoch,
                    prev_start_sample=prev.start_sample,
                    prev_speakers=prev.speakers,
                    next_start_sample=current.start_sample,
                    next_speakers=current.speakers,
                    kind="silence_start",
                    positive=False,
                    ambiguous=False,
                )
            )
            if last_active is not None:
                gap_pending = True
            continue
        if last_active is None:
            transitions.append(
                RegionTransition(
                    audio_epoch=epoch,
                    prev_start_sample=prev.start_sample,
                    prev_speakers=prev.speakers,
                    next_start_sample=current.start_sample,
                    next_speakers=current.speakers,
                    kind="initial_start",
                    positive=False,
                    ambiguous=False,
                )
            )
            last_active = current.speakers
            gap_pending = False
            continue
        if gap_pending:
            if current.speakers == last_active:
                kind: Literal["gap_same_speaker", "gap_speaker_change"] = "gap_same_speaker"
                positive = False
            else:
                kind = "gap_speaker_change"
                positive = True
            transitions.append(
                RegionTransition(
                    audio_epoch=epoch,
                    prev_start_sample=prev.start_sample,
                    prev_speakers=prev.speakers,
                    next_start_sample=current.start_sample,
                    next_speakers=current.speakers,
                    kind=kind,
                    positive=positive,
                    ambiguous=False,
                )
            )
            if positive:
                changes.append(
                    SpeakerChangeGT(
                        audio_epoch=epoch,
                        change_sample=current.start_sample,
                        kind=kind,
                        prev_speakers=last_active,
                        next_speakers=current.speakers,
                    )
                )
            last_active = current.speakers
            gap_pending = False
            continue
        new_speakers = current.speakers - last_active
        if not new_speakers:
            kind = "speaker_left"
            positive = False
        elif not (current.speakers & last_active):
            kind = "clean_handoff"
            positive = True
        else:
            kind = "interruption_onset"
            positive = True
        transitions.append(
            RegionTransition(
                audio_epoch=epoch,
                prev_start_sample=prev.start_sample,
                prev_speakers=prev.speakers,
                next_start_sample=current.start_sample,
                next_speakers=current.speakers,
                kind=kind,
                positive=positive,
                ambiguous=False,
            )
        )
        if positive:
            changes.append(
                SpeakerChangeGT(
                    audio_epoch=epoch,
                    change_sample=current.start_sample,
                    kind=kind,
                    prev_speakers=last_active,
                    next_speakers=current.speakers,
                )
            )
        last_active = current.speakers
    return changes, transitions
