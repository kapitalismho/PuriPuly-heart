from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.psem_r2_policy.metrics import (
    _gt_events,
    live_parent_ledger,
    load_ami_words,
    score_live_ledger,
)
from experiments.psem_r2_policy.sortformer_live import hypothesis_at_boundary
from puripuly_heart.config.prompts import get_default_prompt
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.translation_request import render_translation_system_prompt
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnChild,
    TranslationTurnLifecycleOwner,
    TranslationTurnProcessResult,
    TranslationTurnRequest,
)
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
from puripuly_heart.domain.models import FinalLanguageRun, Transcript, Translation


def apply_observe_evidence(owner: object, payload: Mapping[str, Any]) -> str:
    method = getattr(owner, "observe_evidence", None)
    if not callable(method):
        return "source_method_absent"
    result = method(
        capture_epoch=int(payload["capture_epoch"]),
        start_sample=int(payload["start_sample"]),
        end_sample=int(payload["end_sample"]),
        available_at_monotonic_s=float(payload["available_at_monotonic_s"]),
        relation=payload["relation"],
        producer_generation=payload["producer_generation"],
        reference_generation=payload["reference_generation"],
        reference_valid=bool(payload["reference_valid"]),
    )
    return str(result)


def _unit_rows(assignment: object) -> list[dict[str, Any]]:
    units = getattr(assignment, "units", ()) or ()
    return [
        {
            "group_id": unit.group_id,
            "relation": unit.relation,
            "text": unit.text,
            "token_indexes": list(unit.token_indexes),
            "start_source_sample": unit.start_source_sample,
            "end_source_sample": unit.end_source_sample,
        }
        for unit in units
    ]


def _summary(
    *,
    assignment: object,
    children: Sequence[TranslationTurnChild],
    terminal: STTProviderTurnTerminal,
) -> dict[str, Any]:
    units = getattr(assignment, "units", ()) or ()
    return {
        "parent": str(terminal.identity.segment.segment_id),
        "assignment": getattr(assignment, "disposition", None),
        "conserved": getattr(assignment, "conserved", False),
        "n_units": len(units),
        "group_ids": [unit.group_id for unit in units],
        "token_indexes": [list(unit.token_indexes) for unit in units],
        "relations": [unit.relation for unit in units],
        "child_ids": [str(child.utterance_id) for child in children],
        "child_groups": [child.ownership_group_id for child in children],
        "child_texts": [child.transcript.text for child in children],
        "reconstructed": "".join(unit.text for unit in units),
        "n_timed": len(terminal.timed_tokens),
        "units": _unit_rows(assignment),
    }


def r2_translation_config() -> TranslationRuntimeConfig:
    return TranslationRuntimeConfig(
        source_language="en",
        target_language="ko",
        peer_source_language="en",
        peer_target_language="ko",
        peer_translation_enabled=True,
        translation_enabled=True,
        fallback_transcript_only=False,
        system_prompt=get_default_prompt(),
        context_time_window_s=0.0,
        integrated_context_time_window_s=0.0,
    )


def r2_rendered_system_prompt() -> str:
    config = r2_translation_config()
    return render_translation_system_prompt(
        config.system_prompt,
        source_language=config.peer_source_language,
        target_language=config.peer_target_language,
        input_channel="peer",
        source_specified=True,
    )


def _translation_text(result: object) -> str:
    if isinstance(result, Translation):
        return result.translated_text
    translated = getattr(result, "translated_text", None)
    if isinstance(translated, str):
        return translated
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


async def translate_assignment(
    terminal: STTProviderTurnTerminal,
    *,
    owner: PretranslationOwnershipOwner,
    events: Sequence[ProspectiveSpeakerHypothesis] = (),
    evidence: Sequence[Mapping[str, Any]] = (),
    llm: Any | None = None,
    admitted_at_monotonic_s: float,
) -> dict[str, Any]:
    evidence_status: list[str] = []
    for event in events:
        owner.observe(event)
    for item in evidence:
        evidence_status.append(apply_observe_evidence(owner, item))
    assignment = owner.assign(
        parent_utterance_id=terminal.identity.segment.segment_id,
        timed_tokens=terminal.timed_tokens,
        capture_epoch=terminal.identity.segment.capture_epoch,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
        parent_text=terminal.text,
    )
    created: list[TranslationTurnChild] = []
    outcomes: list[str] = []
    translations: list[str | None] = []
    config = r2_translation_config()
    system_prompt = r2_rendered_system_prompt()

    async def process(child: TranslationTurnChild, _cancel: object) -> TranslationTurnProcessResult:
        created.append(child)
        if llm is None:
            outcomes.append("source_only")
            translations.append(None)
            return TranslationTurnProcessResult("source_only")
        try:
            result = await llm.translate(
                utterance_id=child.utterance_id,
                text=child.transcript.text,
                system_prompt=system_prompt,
                source_language=config.peer_source_language,
                target_language=child.target_language,
                context="",
                scene_participant_count=None,
            )
        except Exception:
            outcomes.append("failed")
            translations.append(None)
            return TranslationTurnProcessResult("failed")
        translations.append(_translation_text(result))
        outcomes.append("translated")
        return TranslationTurnProcessResult("translated")

    async def noop(*_args: object, **_kwargs: object) -> None:
        return None

    turns = TranslationTurnLifecycleOwner(
        on_child_created=noop,
        on_child_started=noop,
        process_child=process,
        on_child_terminal=noop,
        on_parent_closed=noop,
        on_parent_rejected=noop,
    )
    units = (
        assignment.units if assignment.disposition == "assigned" and assignment.conserved else ()
    )
    text = terminal.text
    runs = terminal.final_language_runs or (FinalLanguageRun(text, config.peer_source_language),)
    request = TranslationTurnRequest(
        transcript=Transcript(
            utterance_id=terminal.identity.segment.segment_id,
            text=text,
            is_final=True,
            channel="peer",
            final_language_runs=runs,
            publication_generation=1,
            source_order=1,
        ),
        source="Peer",
        turn_kind="peer",
        target_languages=(config.peer_target_language,),
        config_snapshot=TranslationRuntimeConfigSnapshot(revision=0, value=config),
        ownership_units=units,
    )
    await turns.open_channel_ingress("peer")
    await turns.start()
    await turns.submit(request)
    await turns.wait_for_idle()
    await turns.close()
    summary = _summary(assignment=assignment, children=created, terminal=terminal)
    summary["observe_evidence_status"] = evidence_status
    summary["outcomes"] = outcomes
    summary["child_translations"] = translations
    summary["translated"] = (
        bool(created)
        and len(outcomes) == len(created)
        and all(outcome == "translated" for outcome in outcomes)
    )
    return summary


def r1_frozen_events(
    events: Sequence[ProspectiveSpeakerHypothesis],
    *,
    freeze_monotonic_s: float,
) -> tuple[tuple[ProspectiveSpeakerHypothesis, ...], tuple[str, ...]]:
    kept: list[ProspectiveSpeakerHypothesis] = []
    dropped: list[str] = []
    for event in events:
        if event.available_at_monotonic_s <= freeze_monotonic_s:
            kept.append(event)
        else:
            dropped.append(event.hypothesis_id)
    return tuple(kept), tuple(dropped)


def _token_pay_span(terminal: STTProviderTurnTerminal) -> tuple[int, int]:
    starts = [
        int(token.source_start_sample)
        for token in terminal.timed_tokens
        if token.source_start_sample is not None
    ]
    ends = [
        int(token.source_end_sample)
        for token in terminal.timed_tokens
        if token.source_end_sample is not None
    ]
    if not starts or not ends:
        return 0, 0
    return min(starts), max(ends)


def flush_frontier_at(
    frontiers: Sequence[Mapping[str, Any]],
    avail: float,
    pay_start: int,
    pay_end: int,
) -> int | None:
    z: int | None = None
    for item in frontiers:
        available = item.get("available_at_monotonic_s")
        sample = item.get("sample")
        if available is None or sample is None:
            continue
        if float(available) <= avail:
            z = int(sample)
    if z is None or z < pay_start or z >= pay_end:
        return None
    return z


def r1_project_diagnostic(
    terminal: STTProviderTurnTerminal,
    events: Sequence[ProspectiveSpeakerHypothesis],
    *,
    freeze_monotonic_s: float,
    frontiers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pay0, pay1 = _token_pay_span(terminal)
    frozen, dropped = r1_frozen_events(events, freeze_monotonic_s=freeze_monotonic_s)
    seals: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered = sorted(
        frozen,
        key=lambda item: (
            item.available_at_monotonic_s,
            item.estimated_transition_sample,
            item.hypothesis_id,
        ),
    )
    for event in ordered:
        boundary = int(event.estimated_transition_sample)
        avail = float(event.available_at_monotonic_s)
        event_id = event.hypothesis_id
        last_z = seals[-1]["sealed_frontier_Z"] if seals else pay0
        if not (pay0 <= boundary < pay1):
            history.append(
                {
                    "event_id": event_id,
                    "outcome": "invalid_scope",
                    "requestedX": boundary,
                    "n_seals": len(seals),
                }
            )
            continue
        if event_id in seen:
            history.append(
                {
                    "event_id": event_id,
                    "outcome": "already_separated",
                    "requestedX": boundary,
                    "n_seals": len(seals),
                }
            )
            continue
        seen.add(event_id)
        if avail > freeze_monotonic_s:
            history.append(
                {
                    "event_id": event_id,
                    "outcome": "tooLate",
                    "availability": avail,
                    "deadline": freeze_monotonic_s,
                    "requestedX": boundary,
                }
            )
            continue
        sealed_z = flush_frontier_at(frontiers, avail, pay0, pay1)
        if sealed_z is None:
            history.append(
                {
                    "event_id": event_id,
                    "outcome": "tooLate",
                    "requestedX": boundary,
                    "reason": "no accepted frontier",
                }
            )
            continue
        if any(seal["sealed_frontier_Z"] >= boundary for seal in seals):
            history.append(
                {
                    "event_id": event_id,
                    "outcome": "already_separated",
                    "requestedX": boundary,
                    "n_seals": len(seals),
                }
            )
            continue
        if not (last_z < boundary < sealed_z):
            history.append(
                {
                    "event_id": event_id,
                    "outcome": "tooLate",
                    "requestedX": boundary,
                    "sealedZ": sealed_z,
                    "last_Z": last_z,
                    "reason": "transition outside open accepted range",
                }
            )
            continue
        seals.append(
            {
                "boundary": boundary,
                "sealed_frontier_Z": sealed_z,
                "availability": avail,
                "event_id": event_id,
            }
        )
        history.append(
            {
                "event_id": event_id,
                "outcome": "applied",
                "requestedX": boundary,
                "sealedZ": sealed_z,
                "n_seals": len(seals),
                "reason": "C13 prospective seal at accepted frontier Z",
            }
        )
    return {
        "diagnostic": True,
        "translated": False,
        "assignment": "diagnostic",
        "n_units": 0,
        "group_ids": [],
        "child_groups": [],
        "child_texts": [],
        "n_seals": len(seals),
        "n_applied": sum(1 for item in history if item["outcome"] == "applied"),
        "n_too_late": sum(1 for item in history if item["outcome"] == "tooLate"),
        "seals": seals,
        "history": history,
        "frozen_dropped": list(dropped),
        "observe_evidence_status": [],
    }


def load_meeting_gt(meeting: str | None) -> dict[str, Any]:
    if not meeting:
        return {"ok": False, "reason": "no_meeting", "boundaries": (), "words": ()}
    words = load_ami_words(meeting)
    if not words:
        return {"ok": False, "reason": "missing_gt", "boundaries": (), "words": ()}
    boundaries = tuple(
        event for event in _gt_events(words) if event.get("changed") and not event.get("overlap")
    )
    return {"ok": True, "reason": None, "boundaries": boundaries, "words": tuple(words)}


def first_covering_chunk(
    at_src: int, chunks: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    covering = [
        chunk
        for chunk in chunks
        if chunk.get("start_sample") is not None
        and chunk.get("end_sample") is not None
        and int(chunk["start_sample"]) <= at_src < int(chunk["end_sample"])
    ]
    if not covering:
        return None
    return min(covering, key=lambda item: float(item["available_at_monotonic_s"]))


def charged_gt_events(
    *,
    gt_boundaries: Sequence[Mapping[str, Any]],
    native_chunks: Sequence[Mapping[str, Any]],
    capture_epoch: int,
    producer_generation: object,
    reference_generation: object,
) -> tuple[tuple[ProspectiveSpeakerHypothesis, ...], tuple[dict[str, Any], ...]]:
    used: list[ProspectiveSpeakerHypothesis] = []
    unavailable: list[dict[str, Any]] = []
    for index, boundary in enumerate(gt_boundaries):
        at_src = int(boundary["at_src"])
        match = first_covering_chunk(at_src, native_chunks)
        if match is None:
            unavailable.append({"at_src": at_src, "reason": "no_covering_chunk"})
            continue
        used.append(
            hypothesis_at_boundary(
                at_src,
                capture_epoch=capture_epoch,
                available_at_monotonic_s=float(match["available_at_monotonic_s"]),
                producer_generation=producer_generation,
                reference_generation=reference_generation,
                hypothesis_id=f"gt-control-{index}",
            )
        )
    return tuple(used), tuple(unavailable)


def control_evidence_intervals(
    words: Sequence[Mapping[str, Any]],
    native_chunks: Sequence[Mapping[str, Any]],
    *,
    parent_start_sample: int,
    parent_end_sample: int,
    admitted_at_monotonic_s: float,
    capture_epoch: int,
    producer_generation: object,
    reference_generation: object,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    scoped_words = tuple(
        word
        for word in words
        if int(word["end_src"]) > parent_start_sample
        and int(word["start_src"]) < parent_end_sample
    )
    initial_at = min(
        (
            max(int(word["start_src"]), parent_start_sample)
            for word in scoped_words
        ),
        default=None,
    )
    initial_roles = (
        {
            str(word["role"])
            for word in scoped_words
            if int(word["start_src"]) <= initial_at < int(word["end_src"])
        }
        if initial_at is not None
        else set()
    )
    anchor = next(iter(initial_roles)) if len(initial_roles) == 1 else None
    evidence: list[dict[str, Any]] = []
    for chunk in native_chunks:
        start = chunk.get("start_sample")
        end = chunk.get("end_sample")
        available = chunk.get("available_at_monotonic_s")
        if start is None or end is None or available is None:
            continue
        clipped_start = max(int(start), parent_start_sample)
        clipped_end = min(int(end), parent_end_sample)
        if clipped_end <= clipped_start or float(available) > admitted_at_monotonic_s:
            continue
        roles = {
            str(word["role"])
            for word in words
            if int(word["end_src"]) > clipped_start and int(word["start_src"]) < clipped_end
        }
        if len(roles) == 1 and anchor is not None:
            role = next(iter(roles))
            relation = "CURRENT" if role == anchor else "OTHER"
        else:
            relation = "UNKNOWN"
        evidence.append(
            {
                "capture_epoch": capture_epoch,
                "start_sample": clipped_start,
                "end_sample": clipped_end,
                "available_at_monotonic_s": float(available),
                "relation": relation,
                "producer_generation": producer_generation,
                "reference_generation": reference_generation,
                "reference_valid": True,
            }
        )
    intervals = sorted((int(item["start_sample"]), int(item["end_sample"])) for item in evidence)
    missing: list[dict[str, Any]] = []
    frontier = parent_start_sample
    for start, end in intervals:
        if start > frontier:
            missing.append(
                {
                    "start_sample": frontier,
                    "end_sample": start,
                    "reason": "no_causal_native_observation",
                }
            )
        frontier = max(frontier, end)
        if frontier >= parent_end_sample:
            break
    if frontier < parent_end_sample:
        missing.append(
            {
                "start_sample": frontier,
                "end_sample": parent_end_sample,
                "reason": "no_causal_native_observation",
            }
        )
    return tuple(evidence), tuple(missing)


def _blocked_control(
    reason: str, *, unavailable: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    return {
        "blocked": True,
        "ineligible": True,
        "translated": False,
        "assignment": "blocked",
        "reason": reason,
        "n_units": 0,
        "group_ids": [],
        "child_groups": [],
        "child_texts": [],
        "unavailable": list(unavailable),
        "zero_delay_injected": False,
        "observe_evidence_status": [],
    }


async def control_partition(
    terminal: STTProviderTurnTerminal,
    *,
    meeting: str | None,
    native_chunks: Sequence[Mapping[str, Any]],
    admitted_at_monotonic_s: float,
    producer_generation: object,
    reference_generation: object,
) -> dict[str, Any]:
    gt = load_meeting_gt(meeting)
    if not gt["ok"]:
        return _blocked_control(str(gt["reason"]))
    parent_start, parent_end = _token_pay_span(terminal)
    if parent_end <= parent_start:
        return _blocked_control("missing_parent_token_span")
    relevant_boundaries = tuple(
        boundary
        for boundary in gt["boundaries"]
        if parent_start < int(boundary["at_src"]) < parent_end
    )
    if not relevant_boundaries:
        charged: tuple[ProspectiveSpeakerHypothesis, ...] = ()
        evidence: tuple[dict[str, Any], ...] = ()
    else:
        if not native_chunks:
            return _blocked_control("missing_native_chunks")
        capture_epoch = terminal.identity.segment.capture_epoch
        charged, unavailable = charged_gt_events(
            gt_boundaries=relevant_boundaries,
            native_chunks=native_chunks,
            capture_epoch=capture_epoch,
            producer_generation=producer_generation,
            reference_generation=reference_generation,
        )
        evidence, evidence_missing = control_evidence_intervals(
            gt["words"],
            native_chunks,
            parent_start_sample=parent_start,
            parent_end_sample=parent_end,
            admitted_at_monotonic_s=admitted_at_monotonic_s,
            capture_epoch=capture_epoch,
            producer_generation=producer_generation,
            reference_generation=reference_generation,
        )
        if unavailable or evidence_missing:
            return _blocked_control(
                "insufficient_causal_native_coverage",
                unavailable=(*unavailable, *evidence_missing),
            )
    summary = await translate_assignment(
        terminal,
        owner=PretranslationOwnershipOwner(enabled=True),
        events=charged,
        evidence=evidence,
        llm=None,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
    )
    summary["translated"] = False
    summary["blocked"] = False
    summary["ineligible"] = False
    summary["unavailable"] = []
    summary["reason"] = None
    summary["causal_native_coverage"] = {
        "required": bool(relevant_boundaries),
        "observed_intervals": len(evidence),
        "relevant_gt_boundaries": len(relevant_boundaries),
    }
    summary["zero_delay_injected"] = False
    return summary


def score_arm(
    terminal: STTProviderTurnTerminal,
    units: Sequence[object],
    *,
    words: Sequence[Mapping[str, Any]] = (),
    receipts: Sequence[object] = (),
    marks: Mapping[str, float | None] | None = None,
    meeting: str | None = None,
    seal_reasons: Sequence[str] = (),
    speech_chunks: int | None = None,
    silence_chunks: int | None = None,
) -> dict[str, Any]:
    ledger = live_parent_ledger(
        parent_text=terminal.text,
        tokens=terminal.timed_tokens,
        units=units,
        receipts=receipts,
        marks=marks,
        meeting=meeting,
        seal_reasons=seal_reasons,
        speech_chunks=speech_chunks,
        silence_chunks=silence_chunks,
    )
    return {"ledger": ledger, **score_live_ledger(ledger, words=words)}


async def evaluate_protocol_arms(
    terminal: STTProviderTurnTerminal,
    *,
    r2_events: Sequence[ProspectiveSpeakerHypothesis],
    evidence: Sequence[Mapping[str, Any]] = (),
    llm: Any | None = None,
    freeze_monotonic_s: float,
    admitted_at_monotonic_s: float,
    meeting: str | None = None,
    native_chunks: Sequence[Mapping[str, Any]] = (),
    frontiers: Sequence[Mapping[str, Any]] = (),
    producer_generation: object | None = None,
    reference_generation: object | None = None,
) -> dict[str, Any]:
    producer = producer_generation if producer_generation is not None else object()
    reference = reference_generation if reference_generation is not None else object()
    r0 = await translate_assignment(
        terminal,
        owner=PretranslationOwnershipOwner(enabled=False),
        events=(),
        evidence=(),
        llm=llm,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
    )
    r2 = await translate_assignment(
        terminal,
        owner=PretranslationOwnershipOwner(enabled=True),
        events=r2_events,
        evidence=evidence,
        llm=None,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
    )
    r2["translated"] = False
    r1 = r1_project_diagnostic(
        terminal,
        r2_events,
        freeze_monotonic_s=freeze_monotonic_s,
        frontiers=frontiers,
    )
    control = await control_partition(
        terminal,
        meeting=meeting,
        native_chunks=native_chunks,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
        producer_generation=producer,
        reference_generation=reference,
    )
    return {
        "r0": r0,
        "r2": r2,
        "r1": r1,
        "control": control,
    }
