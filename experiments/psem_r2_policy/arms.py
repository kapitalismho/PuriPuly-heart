from __future__ import annotations

from typing import Any, Mapping, Sequence

from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnChild,
    TranslationTurnLifecycleOwner,
    TranslationTurnProcessResult,
    TranslationTurnRequest,
)
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
from puripuly_heart.domain.models import FinalLanguageRun, Transcript

from experiments.psem_r2_policy.sortformer_live import hypothesis_at_boundary


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

    async def process(child: TranslationTurnChild, _cancel: object) -> TranslationTurnProcessResult:
        created.append(child)
        if llm is not None:
            await llm.translate(
                text=child.transcript.text,
                source_language="en",
                target_language="ko",
            )
        return TranslationTurnProcessResult("source_only")

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
    units = assignment.units if assignment.disposition == "assigned" and assignment.conserved else ()
    text = terminal.text
    runs = terminal.final_language_runs or (FinalLanguageRun(text, "en"),)
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
        target_languages=("ko",),
        config_snapshot=TranslationRuntimeConfigSnapshot(
            revision=0, value=TranslationRuntimeConfig()
        ),
        ownership_units=units,
    )
    await turns.open_channel_ingress("peer")
    await turns.start()
    await turns.submit(request)
    await turns.wait_for_idle()
    await turns.close()
    summary = _summary(assignment=assignment, children=created, terminal=terminal)
    summary["observe_evidence_status"] = evidence_status
    summary["translated"] = llm is not None
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


def charged_gt_events(
    *,
    gt_boundaries: Sequence[Mapping[str, Any]],
    native_receipts: Sequence[Mapping[str, Any]],
    capture_epoch: int,
    producer_generation: object,
    reference_generation: object,
) -> tuple[tuple[ProspectiveSpeakerHypothesis, ...], tuple[dict[str, Any], ...]]:
    used: list[ProspectiveSpeakerHypothesis] = []
    unavailable: list[dict[str, Any]] = []
    for index, boundary in enumerate(gt_boundaries):
        at_src = int(boundary["at_src"])
        match = None
        for receipt in native_receipts:
            frontier = receipt.get("frontier")
            available = receipt.get("available_at_monotonic_s")
            if frontier is None or available is None:
                continue
            if int(frontier) >= at_src:
                if match is None or float(available) < float(match["available_at_monotonic_s"]):
                    match = receipt
        if match is None:
            unavailable.append(
                {
                    "at_src": at_src,
                    "reason": "no_native_availability",
                }
            )
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


async def evaluate_protocol_arms(
    terminal: STTProviderTurnTerminal,
    *,
    r2_events: Sequence[ProspectiveSpeakerHypothesis],
    evidence: Sequence[Mapping[str, Any]] = (),
    llm: Any | None = None,
    freeze_monotonic_s: float,
    admitted_at_monotonic_s: float,
    gt_boundaries: Sequence[Mapping[str, Any]] = (),
    native_receipts: Sequence[Mapping[str, Any]] = (),
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
        llm=llm,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
    )
    frozen_events, frozen_dropped = r1_frozen_events(
        r2_events, freeze_monotonic_s=freeze_monotonic_s
    )
    r1_evidence = tuple(
        item
        for item in evidence
        if float(item["available_at_monotonic_s"]) <= freeze_monotonic_s
    )
    r1 = await translate_assignment(
        terminal,
        owner=PretranslationOwnershipOwner(enabled=True),
        events=frozen_events,
        evidence=r1_evidence,
        llm=llm,
        admitted_at_monotonic_s=freeze_monotonic_s,
    )
    r1["frozen_dropped"] = list(frozen_dropped)
    r1["diagnostic"] = True
    control_events, control_unavailable = charged_gt_events(
        gt_boundaries=gt_boundaries,
        native_receipts=native_receipts,
        capture_epoch=terminal.identity.segment.capture_epoch,
        producer_generation=producer,
        reference_generation=reference,
    )
    control = await translate_assignment(
        terminal,
        owner=PretranslationOwnershipOwner(enabled=True),
        events=control_events,
        evidence=(),
        llm=llm,
        admitted_at_monotonic_s=admitted_at_monotonic_s,
    )
    control["unavailable"] = list(control_unavailable)
    control["zero_delay_injected"] = False
    return {
        "r0": r0,
        "r2": r2,
        "r1": r1,
        "control": control,
    }
