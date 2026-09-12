from __future__ import annotations

import hashlib
import importlib.util
import json
import mmap
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "experiments/psem_r2_policy"
RAW = {
    "ES2009a": EXP / "artifacts/dev/ES2009a/20260911T234330502940Z.json",
    "ES2009c": EXP / "artifacts/dev/ES2009c/20260912T073421006732Z.json",
    "ES2009d": EXP / "artifacts/dev/ES2009d/20260912T084839024097Z.json",
}
OLD_SOURCE = Path("C:/tmp/psem-u8-af26d1d3/tree/src/puripuly_heart/core/audio/pretranslation_ownership.py")
OLD_SOURCE_SHA256 = "7cbe23b4c3d0f5845454265b844b9ed86e3a69ce81b1ee5a7bc2d4dc2dede769"
OUTPUT = Path(__file__).with_name("causal-replay.json")
GENERATION = "reconstructed-single-run-generation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_old_module() -> Any:
    observed = sha256(OLD_SOURCE)
    if observed != OLD_SOURCE_SHA256:
        raise RuntimeError(f"old ownership source mismatch: {observed}")
    spec = importlib.util.spec_from_file_location("u13_old_pretranslation_ownership", OLD_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load old ownership source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parent_records(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            cursor = mapped.find(b' "parents": [')
            if cursor < 0:
                raise RuntimeError(f"parents array missing: {path}")
            cursor = mapped.find(b"[", cursor) + 1
            while True:
                cursor = mapped.find(b"{", cursor)
                if cursor < 0:
                    raise RuntimeError(f"unterminated parents array: {path}")
                window = 1 << 20
                while True:
                    try:
                        payload = mapped[cursor : min(cursor + window, len(mapped))].decode("utf-8")
                        record, consumed = decoder.raw_decode(payload)
                        break
                    except json.JSONDecodeError:
                        if cursor + window >= len(mapped):
                            raise
                        window *= 2
                if "parent_id" not in record:
                    raise RuntimeError(f"decoded non-parent object at {cursor}: {path}")
                yield record
                cursor += len(payload[:consumed].encode("utf-8"))
                next_nonspace = cursor
                while mapped[next_nonspace : next_nonspace + 1] in {b" ", b"\r", b"\n", b"\t"}:
                    next_nonspace += 1
                if mapped[next_nonspace : next_nonspace + 1] == b"]":
                    return
                cursor = next_nonspace + 1
        finally:
            mapped.close()


def token_objects(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    from puripuly_heart.core.stt.backend import STTTimedToken

    return tuple(
        STTTimedToken(
            text=str(row["text"]),
            language="en",
            start_ms=row.get("start_ms"),
            end_ms=row.get("end_ms"),
            timing=row.get("timing"),
            source_start_sample=row.get("source_start_sample"),
            source_end_sample=row.get("source_end_sample"),
        )
        for row in rows
    )


def event_objects(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis

    result = []
    for row in rows:
        boundary = int(row["estimated_transition_sample"])
        result.append(
            ProspectiveSpeakerHypothesis(
                hypothesis_id=str(row["hypothesis_id"]),
                revision=1,
                capture_epoch=1,
                support_start_sample=max(boundary - 1, 0),
                support_end_sample=max(boundary, 1),
                estimated_transition_sample=boundary,
                observed_frontier_sample=max(boundary, 1),
                available_at_monotonic_s=float(row["available_at_monotonic_s"]),
                producer_generation=GENERATION,
                reference_generation=GENERATION,
                producer_valid=True,
                reference_valid=True,
                retracted=False,
                local_slot=None,
            )
        )
    return tuple(result)


def evidence_objects(rows: list[dict[str, Any]], cls: Any) -> tuple[Any, ...]:
    return tuple(
        cls(
            capture_epoch=int(row["capture_epoch"]),
            start_sample=int(row["start_sample"]),
            end_sample=int(row["end_sample"]),
            available_at_monotonic_s=float(row["available_at_monotonic_s"]),
            relation=str(row["relation"]),
            producer_generation=GENERATION,
            reference_generation=GENERATION,
            reference_valid=bool(row.get("reference_valid", False)),
        )
        for row in rows
        if str(row.get("observe_evidence_status") or "observed") == "observed"
    )


def unit_rows(units: tuple[Any, ...]) -> list[dict[str, Any]]:
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


def comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "group_id",
        "relation",
        "text",
        "token_indexes",
        "start_source_sample",
        "end_source_sample",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def causal_support(
    evidence: tuple[Any, ...],
    *,
    admission: float,
    span: list[int | None],
) -> dict[str, Any]:
    if len(span) != 2 or span[0] is None or span[1] is None:
        return {"covered": False, "reason": "missing_parent_span"}
    start, end = int(span[0]), int(span[1])
    intervals = sorted(
        (max(start, item.start_sample), min(end, item.end_sample))
        for item in evidence
        if item.reference_valid
        and item.available_at_monotonic_s <= admission
        and item.end_sample > start
        and item.start_sample < end
    )
    merged: list[list[int]] = []
    for left, right in intervals:
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    gaps: list[list[int]] = []
    cursor = start
    for left, right in merged:
        if left > cursor:
            gaps.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append([cursor, end])
    return {
        "span": [start, end],
        "covered": not gaps,
        "support_union": merged,
        "gaps": gaps,
        "causally_available_intervals": len(intervals),
    }


def main() -> None:
    old = load_old_module()
    from experiments.psem_r2_policy.metrics import live_parent_ledger, load_ami_words, pair_parent_guard, score_live_ledger
    from puripuly_heart.core.audio import pretranslation_ownership as new

    cases: dict[str, Any] = {}
    all_severe: list[dict[str, Any]] = []
    repaired_witness: dict[str, Any] | None = None
    for meeting, path in RAW.items():
        words = load_ami_words(meeting)
        counts: Counter[str] = Counter()
        fidelity_mismatches: list[dict[str, Any]] = []
        severe: list[dict[str, Any]] = []
        primary = {"r0_contaminated": 0, "old_r2_contaminated": 0, "new_r2_contaminated": 0, "attributable": 0}
        fragmentation = {"old_units": 0, "new_units": 0, "old_extra_units": 0, "new_extra_units": 0}
        for parent in parent_records(path):
            counts["parents"] += 1
            token_rows = list(parent.get("tokens") or ())
            text = str(parent.get("text") or "")
            admission = (parent.get("marks") or {}).get("translation_admission")
            if not text:
                counts["skipped_empty_text"] += 1
                continue
            if not token_rows:
                counts["skipped_missing_tokens"] += 1
                continue
            if admission is None:
                counts["skipped_missing_admission"] += 1
                continue
            if parent.get("assignment") != "assigned":
                counts["skipped_not_old_assigned"] += 1
                continue
            missing_hypothesis = any(
                row.get("estimated_transition_sample") is None
                or row.get("available_at_monotonic_s") is None
                or row.get("hypothesis_id") is None
                for row in parent.get("hypotheses") or ()
            )
            missing_evidence = any(
                row.get("capture_epoch") is None
                or row.get("start_sample") is None
                or row.get("end_sample") is None
                or row.get("available_at_monotonic_s") is None
                or row.get("relation") is None
                or row.get("reference_valid") is None
                for row in parent.get("evidence") or ()
            )
            if missing_hypothesis or missing_evidence:
                counts["skipped_missing_causal_fields"] += 1
                continue

            tokens = token_objects(token_rows)
            events = event_objects(list(parent.get("hypotheses") or ()))
            old_evidence = evidence_objects(list(parent.get("evidence") or ()), old.PretranslationEvidence)
            new_evidence = evidence_objects(list(parent.get("evidence") or ()), new.PretranslationEvidence)
            old_units, old_late, _old_reasons = old.assign_ownership_units(
                tokens,
                events,
                admitted_at_monotonic_s=float(admission),
                capture_epoch=1,
                evidence=old_evidence,
            )
            new_units, new_late, new_reasons = new.assign_ownership_units(
                tokens,
                events,
                admitted_at_monotonic_s=float(admission),
                capture_epoch=1,
                evidence=new_evidence,
            )
            old_rows = unit_rows(old_units)
            new_rows = unit_rows(new_units)
            recorded_rows = list(parent.get("units") or ())
            counts["replayed"] += 1
            if comparable(old_rows) != comparable(recorded_rows):
                counts["old_fidelity_mismatch"] += 1
                if len(fidelity_mismatches) < 20:
                    fidelity_mismatches.append(
                        {
                            "parent_id": parent["parent_id"],
                            "recorded": comparable(recorded_rows),
                            "replayed": comparable(old_rows),
                            "admission": admission,
                        }
                    )
            else:
                counts["old_fidelity_match"] += 1
            if "insufficient_evidence_coverage" in new_reasons:
                counts["coverage_abstention"] += 1
            elif "no_confirmed_transition" in new_reasons:
                counts["no_transition_abstention"] += 1
            elif len(new_rows) > 1:
                counts["supported_partition"] += 1
            else:
                counts["whole_parent_no_partition"] += 1
            if old_late != new_late:
                counts["late_accounting_mismatch"] += 1
            if "".join(row["text"] for row in new_rows) != text:
                counts["new_conservation_failure"] += 1

            scored = score_live_ledger(
                live_parent_ledger(
                    parent_text=text,
                    tokens=token_rows,
                    units=new_rows,
                    receipts=parent.get("receipts") or (),
                    marks=parent.get("marks") or {},
                    meeting=meeting,
                    seal_reasons=(parent.get("seal_reason"),),
                ),
                words=words,
            )
            guard = pair_parent_guard((parent.get("r0") or {}).get("guard"), scored.get("guard"))
            if guard.get("severe"):
                item = {"meeting": meeting, "parent_id": parent["parent_id"], "failures": guard.get("failures"), "guard": guard}
                severe.append(item)
                all_severe.append(item)
            if parent["parent_id"] == "c06fe9cd-e419-47bd-8427-066896b59e18":
                repaired_witness = {
                    "meeting": meeting,
                    "parent_id": parent["parent_id"],
                    "admission": admission,
                    "span": parent.get("span"),
                    "old_units": comparable(old_rows),
                    "new_units": comparable(new_rows),
                    "new_reasons": list(new_reasons),
                    "old_late_ignored": list(old_late),
                    "new_late_ignored": list(new_late),
                    "causal_support": causal_support(
                        new_evidence,
                        admission=float(admission),
                        span=list(parent.get("span") or ()),
                    ),
                    "old_guard": parent.get("guard"),
                    "new_guard": guard,
                }
            old_frag = max(len(recorded_rows) - 1, 0)
            new_frag = max(len(new_rows) - 1, 0)
            fragmentation["old_units"] += len(recorded_rows)
            fragmentation["new_units"] += len(new_rows)
            fragmentation["old_extra_units"] += old_frag
            fragmentation["new_extra_units"] += new_frag
            r0_contamination = (parent.get("r0") or {}).get("contamination") or {}
            old_contamination = (parent.get("r2") or {}).get("contamination") or {}
            new_contamination = scored.get("contamination") or {}
            if r0_contamination.get("eligible") and old_contamination.get("eligible") and new_contamination.get("eligible"):
                primary["attributable"] += int(r0_contamination.get("attributable_chars") or 0)
                primary["r0_contaminated"] += int(r0_contamination.get("contaminated_chars") or 0)
                primary["old_r2_contaminated"] += int(old_contamination.get("contaminated_chars") or 0)
                primary["new_r2_contaminated"] += int(new_contamination.get("contaminated_chars") or 0)
                counts["primary_eligible"] += 1

        old_gain = primary["r0_contaminated"] - primary["old_r2_contaminated"]
        new_gain = primary["r0_contaminated"] - primary["new_r2_contaminated"]
        cases[meeting] = {
            "raw": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
            "counts": dict(counts),
            "old_replay_fidelity": {
                "eligible": counts["replayed"],
                "matches": counts["old_fidelity_match"],
                "mismatches": counts["old_fidelity_mismatch"],
                "examples": fidelity_mismatches,
            },
            "fragmentation": fragmentation,
            "primary_character_pool": {
                **primary,
                "old_policy_gain_chars": old_gain,
                "new_policy_gain_chars": new_gain,
                "lost_headroom_chars": old_gain - new_gain,
            },
            "new_severe_count": len(severe),
            "new_severes": severe,
        }

    payload = {
        "schema": "PSEM-R2-U13-CAUSAL-COVERAGE-REPLAY-1",
        "policy_revision": "R2-POLICY-DIRECTOR-9",
        "runtime_archive_sha256": "819465e74b847e0a46c0e0968b52d76fccc1a95fa146716411d945690e164416",
        "runtime_override": {
            "path": str(EXP / "runtime_overrides/pretranslation_ownership.py"),
            "sha256": sha256(EXP / "runtime_overrides/pretranslation_ownership.py"),
        },
        "old_runtime_source": {"path": str(OLD_SOURCE), "sha256": sha256(OLD_SOURCE)},
        "reconstruction": {
            "basis": "Recorded exact token intervals/text, evidence support intervals/arrival/reference validity/status, transition boundary/arrival/id, and actual translation admission are replayed. Immutable live_runner used capture_epoch=1 and hypothesis_at_boundary revision=1 with boundary-local support/frontier and valid producer/reference fields. A single owner/producer/reference instance served each run, so omitted generation identities are reconstructed as one per-run identity. English token language is restored from the pinned provider configuration; language does not affect partition boundaries or scored unit text.",
            "not_reconstructed": "No absent admission, token interval, evidence support/arrival/validity, or transition boundary/arrival is fabricated. Such parents are counted as skipped. Local slot is irrelevant to assignment and restored as None.",
            "limitations": "This is zero-cost offline regrouping of frozen accepted text and causal recorded arrivals, not a new live R2 translation/timing run. Recorded changed-unit translations are not reused.",
        },
        "evidence_applicability": {
            "reused": [
                "immutable accepted ASR text and token/source timing",
                "R0 ownership/scoring",
                "recorded native transition and evidence arrival times",
                "source accounting and native cadence",
            ],
            "invalidated": [
                "all old-policy R2 units and fragmentation",
                "all translations made from changed R2 units",
                "old R2 admission-to-dispatch and completion timing comparisons",
                "any claim that offline regrouping is a full live execution",
            ],
            "required_before_paid_resumption": [
                "independent review of the pinned override and causal replay",
                "new-policy live zero-cost path check with full recorded generation/support/validity fields",
                "Director decision whether zero supported partitions and complete loss of measured a/c/d headroom warrants stopping or revising the producer evidence contract",
                "if resumption is authorized, retranslate every changed R2 unit and rerun actual timing/admission comparisons; do not reuse old translations",
            ],
            "paid_enabled": False,
            "holdout_locked": True,
        },
        "cases": cases,
        "total_new_severe_count": len(all_severe),
        "all_new_severes": all_severe,
        "repaired_failure_witness": repaired_witness,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256(OUTPUT), "bytes": OUTPUT.stat().st_size, "total_new_severe_count": len(all_severe)}))


if __name__ == "__main__":
    main()
