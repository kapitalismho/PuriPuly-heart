from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from uuid import uuid5

from puripuly_heart.core.audio.ownership import AudioSegmentSnapshot
from puripuly_heart.core.stt.backend import (
    STTConsumedContribution,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.domain.models import Transcript
from puripuly_heart.domain.recognition import (
    RecognitionStreamIdentity,
    RecognitionTextOrigin,
    RecognitionUnitIdentity,
    slice_recognition_origins,
    validate_recognition_origins,
)


class RecognitionConsumptionLedger:
    def __init__(self) -> None:
        self._receipt_sequences: OrderedDict[RecognitionStreamIdentity, int] = OrderedDict()
        self._retired_capture: dict[tuple[str, int], int] = {}
        self._retired_activation: dict[str, int] = {"self": -1, "peer": -1}

    def consume(self, terminal: STTRecognitionUnitTerminal) -> bool:
        identity = terminal.unit.identity
        stream = identity.stream
        if stream not in self._receipt_sequences and (
            stream.activation_generation <= self._retired_activation[stream.channel]
            or (
                stream.capture_epoch
                <= self._retired_capture.get((stream.channel, stream.activation_generation), -1)
            )
        ):
            return False
        previous = self._receipt_sequences.get(identity.stream, 0)
        if identity.receipt_sequence <= previous:
            return False
        self._receipt_sequences[identity.stream] = identity.receipt_sequence
        self._receipt_sequences.move_to_end(identity.stream)
        while len(self._receipt_sequences) > 4096:
            retired, _ = self._receipt_sequences.popitem(last=False)
            scope = (retired.channel, retired.activation_generation)
            self._retired_capture[scope] = max(
                retired.capture_epoch, self._retired_capture.get(scope, -1)
            )
            if len(self._retired_capture) > 4096:
                channel, generation = min(self._retired_capture, key=lambda item: item[1])
                self._retired_activation[channel] = max(
                    generation, self._retired_activation[channel]
                )
                self._retired_capture = {
                    scope: capture
                    for scope, capture in self._retired_capture.items()
                    if scope[0] != channel or scope[1] > self._retired_activation[channel]
                }
        return terminal.outcome in {"final", "empty"}


def recognition_transcript(
    unit: STTRecognitionUnit, *, created_at: float, publication_order: int | None = None
) -> Transcript:
    return Transcript(
        utterance_id=unit.identity.unit_id,
        text=unit.text,
        is_final=True,
        created_at=created_at,
        channel=unit.identity.stream.channel,
        final_language_runs=unit.final_language_runs,
        final_speaker_runs=unit.final_speaker_runs,
        publication_generation=(
            unit.identity.stream.activation_generation if publication_order is not None else None
        ),
        source_order=publication_order,
        recognition_origins=(
            RecognitionTextOrigin(
                unit.identity,
                0,
                len(unit.text),
                unit.provenance.transcription,
                native_provenance=(unit.provenance,),
            ),
        ),
    )


def contribution_transcript(
    snapshot: AudioSegmentSnapshot,
    event: STTProviderTurnUpdate | STTProviderTurnTerminal,
    pieces: tuple[STTConsumedContribution, ...],
    *,
    created_at: float,
    publication_order: int,
    channel: str = "peer",
) -> Transcript:
    if snapshot.identity != event.identity.segment:
        raise ValueError("contribution source request identity mismatch")
    validate_recognition_origins(event.recognition_origins, event.text)
    identity = event.identity
    text = "".join(piece.text for piece in pieces)
    unit_id = uuid5(
        snapshot.identity.segment_id,
        "|".join(
            f"{identity.provider_epoch_id}:{identity.provider_turn_id}:"
            f"{piece.contribution_id}:{piece.text_start}:{piece.text_end}"
            for piece in pieces
        ),
    )
    if isinstance(event, STTProviderTurnTerminal) and not event.included_contributions:
        unit_id = snapshot.identity.segment_id
    stream = RecognitionStreamIdentity(
        channel,
        identity.segment.activation_generation,
        identity.segment.capture_epoch,
        identity.provider_epoch_id,
        identity.settings_scope,
    )
    unit_identity = RecognitionUnitIdentity(stream, unit_id, publication_order)
    origins = []
    offset = 0
    for piece in pieces:
        audio = next(
            (item for item in event.source_audio if item.contribution_id == piece.contribution_id),
            None,
        )
        if event.source_audio and audio is None:
            raise ValueError("batch contribution has unresolved audio ownership")
        request_origin = RecognitionTextOrigin(
            unit_identity,
            offset,
            offset + len(piece.text),
            request_id=identity.provider_turn_id,
            contribution_id=piece.contribution_id,
            contribution_range=(piece.text_start, piece.text_end),
            source_ranges=audio.source_ranges if audio is not None else snapshot.content_ranges,
            context_ranges=(audio.context_ranges if audio is not None else snapshot.context_ranges),
            display_group_id=(
                audio.display_group_id if audio is not None else snapshot.display_group_id
            ),
            source_binding="request_owned",
            order_basis="capture_order",
            ownership="request",
            audio_settings=snapshot.settings,
            source_available_at_monotonic_s=created_at,
            speaker_attribution=audio.speaker_attribution if audio is not None else None,
            native_provenance=(
                (event.provenance,)
                if isinstance(event, STTProviderTurnUpdate)
                else event.provenance
            ),
        )
        if event.recognition_origins:
            for origin in slice_recognition_origins(
                event.recognition_origins, piece.text_start, piece.text_end
            ):
                if origin.source_binding == "composition":
                    origins.append(
                        replace(
                            origin,
                            text_start=offset + origin.text_start,
                            text_end=offset + origin.text_end,
                        )
                    )
                else:
                    origins.append(
                        replace(
                            request_origin,
                            identity=origin.identity,
                            text_start=offset + origin.text_start,
                            text_end=offset + origin.text_end,
                            native=origin.native,
                            source_binding=origin.source_binding,
                            order_basis=origin.order_basis,
                            contribution_range=(
                                piece.text_start + origin.text_start,
                                piece.text_start + origin.text_end,
                            ),
                        )
                    )
        else:
            origins.append(request_origin)
        offset += len(piece.text)

    def sliced_runs(runs):
        result = []
        for piece in pieces:
            cursor = 0
            for run in runs:
                end = cursor + len(run.text)
                left, right = max(cursor, piece.text_start), min(end, piece.text_end)
                if left < right:
                    result.append(replace(run, text=run.text[left - cursor : right - cursor]))
                cursor = end
        return tuple(result)

    return Transcript(
        unit_id,
        text,
        True,
        created_at=created_at,
        channel=channel,
        final_language_runs=sliced_runs(event.final_language_runs),
        final_speaker_runs=sliced_runs(event.final_speaker_runs),
        publication_generation=stream.activation_generation if channel == "peer" else None,
        source_order=publication_order if channel == "peer" else None,
        source_text_revision=f"{identity.provider_turn_id}:{publication_order}",
        recognition_origins=tuple(origins),
    )
