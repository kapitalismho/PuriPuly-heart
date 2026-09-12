"""Read-only U10 offline rescore of the preserved clean ES2009a acquisition.

Streams the canonical case payload, projects parents and case-level gates into a
compact sidecar, recomputes attribution and paired guard records from the recorded
per-arm tokens/units and the validated GT word annotations, recomputes the derived
U10 decisions/coverage, verifies the recorded AMI input hashes, and writes a
separate evaluation record. The acquisition, the legacy U9 score and every
provider output are left untouched; no provider call and no HOLDOUT inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

HERE = Path(__file__).resolve().parent
POLICY = HERE.parents[1]
ROOT = POLICY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.metrics import (  # noqa: E402
    aggregate_cluster_parents,
    arm_guard_record,
    attribute_tokens,
    case_evaluation_record,
    case_execution_record,
    load_ami_words,
    pair_parent_guard,
    sequential_merge_contamination,
    u8_case_report,
    u8_phase_report,
)
from experiments.psem_r2_policy.phase import aggregate_phase  # noqa: E402

DEFAULT_CASE = POLICY / "artifacts" / "dev" / "ES2009a" / "20260911T234330502940Z.json"
LEGACY_SUMMARY = (
    POLICY / "artifacts" / "dev-supervised" / "cases" / "ES2009a" / "attempt-1" / "summary.json"
)
GATE = POLICY / "artifacts" / "dev-execution" / "cases" / "ES2009a-full" / "gate.json"
PIN = POLICY / "RUNTIME_PIN.json"
EXPANSION_AUTHORITY = (
    POLICY.parents[1]
    / "experiments"
    / "psem_e2o2_continuous_ownership"
    / "EXPANSION_AUTHORITY.json"
)
KEY = re.compile(rb'\r?\n "([^"]+)":')
BIG_LIMIT = 8_000_000
CASE_FIELDS = (
    "meeting",
    "phase",
    "capture_timing",
    "declared_source_samples",
    "sealed_segments",
    "provider_fault",
    "task_failures",
    "timing_failures",
    "budget_truncated",
    "protocol_revision",
    "u8",
    "decision",
    "aggregate",
    "n_parents",
    "execution_completed",
    "evaluation_valid",
    "operational_clean",
)
PARENT_FIELDS = (
    "index",
    "parent_id",
    "meeting",
    "cluster_id",
    "outcome",
    "terminal_outcome",
    "seal_reason",
    "status",
    "clean_completion",
    "degraded",
    "unsuccessful_source_processing",
    "accounted",
    "provenance_valid",
    "text_authority",
    "failure_reason",
    "incomplete",
    "outage",
    "text",
    "conserved",
    "span",
    "receipt",
    "sequential_target",
)
CONTAMINATION_FIELDS = (
    "eligible",
    "reason",
    "proportion",
    "attributable_chars",
    "contaminated_chars",
    "mixed_chars",
    "unaligned_chars",
    "unknown_chars",
    "coverage",
    "sequential_target",
)
GUARD_FIDELITY_FIELDS = (
    "lexical_tokens",
    "annotation_tokens",
    "excluded",
    "wrong_token_ids",
    "same_speaker_stratum",
    "span",
)
CONTAMINATION_FIDELITY_FIELDS = (
    "eligible",
    "reason",
    "attributable_chars",
    "contaminated_chars",
    "mixed_chars",
    "unaligned_chars",
)
SCORE_FIELDS = (
    "cluster_id",
    "eligible",
    "n_parents",
    "r0_chars",
    "r2_chars",
    "r0_contaminated",
    "r2_contaminated",
    "r0_proportion",
    "r2_proportion",
    "delta",
    "newly_unassigned_chars",
    "r2_unknown_chars",
    "benefit_explained_only_by_unassigned",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_normalized(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remainder = b""
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            data = remainder + chunk
            if data.endswith(b"\r"):
                remainder = b"\r"
                data = data[:-1]
            else:
                remainder = b""
            digest.update(data.replace(b"\r\n", b"\n"))
        digest.update(remainder)
    return digest.hexdigest()


def _top_level_spans(buf: Any) -> dict[str, int]:
    return {match.group(1).decode(): match.start() + 2 for match in KEY.finditer(buf)}


def _value_extent(buf: Any, key_start: int) -> tuple[int, int]:
    i = buf.find(b":", key_start) + 1
    size = len(buf)
    while i < size and buf[i : i + 1] in (b" ", b"\r", b"\n", b"\t"):
        i += 1
    if buf[i : i + 1] in (b"{", b"["):
        depth = 0
        in_str = False
        start = i
        while i < size:
            byte = buf[i]
            if in_str:
                if byte == 0x5C:
                    i += 2
                    continue
                if byte == 0x22:
                    in_str = False
                i += 1
                continue
            if byte == 0x22:
                in_str = True
            elif byte in (0x7B, 0x5B):
                depth += 1
            elif byte in (0x7D, 0x5D):
                depth -= 1
                if depth == 0:
                    return start, i + 1
            i += 1
        raise ValueError("unterminated json value")
    end = buf.find(b'\r\n "', i)
    return i, size if end == -1 else end


def _decode(buf: Any, spans: Mapping[str, int], name: str, limit: int = BIG_LIMIT) -> Any:
    start, end = _value_extent(buf, spans[name])
    if end - start > limit:
        raise ValueError(f"{name} exceeds {limit} bytes")
    raw = bytes(buf[start:end])
    if raw[0:1] not in (b"[", b"{"):
        raw = raw.rstrip(b",}] \r\n\t")
    return json.loads(raw)


def _elements(buf: Any, open_idx: int) -> Iterator[tuple[int, int]]:
    size = len(buf)
    i = open_idx + 1
    depth = 0
    in_str = False
    start = None
    while i < size:
        byte = buf[i]
        if in_str:
            if byte == 0x5C:
                i += 2
                continue
            if byte == 0x22:
                in_str = False
            i += 1
            continue
        if byte == 0x22:
            in_str = True
        elif byte in (0x7B, 0x5B):
            if depth == 0 and byte == 0x7B:
                start = i
            depth += 1
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth == 0:
                if byte == 0x5D and start is None:
                    return
                yield start, i + 1
                start = None
            elif depth < 0:
                return
        i += 1


def _arm_projection(arm: Any) -> dict[str, Any]:
    if not isinstance(arm, Mapping):
        return {}
    contamination = arm.get("contamination")
    if not isinstance(contamination, Mapping):
        contamination = arm if "eligible" in arm else {}
    ledger = arm.get("ledger") if isinstance(arm.get("ledger"), Mapping) else {}
    return {
        "contamination": {key: contamination.get(key) for key in CONTAMINATION_FIELDS},
        "recorded_guard": arm.get("guard"),
        "tokens": list(ledger.get("tokens") or ()),
        "units": list(ledger.get("units") or ()),
    }


def _arm_fidelity(
    recorded_guard: Any,
    guard: Mapping[str, Any],
    recorded_contamination: Mapping[str, Any],
    recomputed_contamination: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(recorded_guard, Mapping):
        return {"recomputed": True, "recorded_guard_present": False, "matches": True}
    guard_diff = {
        field: [recorded_guard.get(field), guard.get(field)]
        for field in GUARD_FIDELITY_FIELDS
        if recorded_guard.get(field) != guard.get(field)
    }
    contamination_diff = {
        field: [recorded_contamination.get(field), recomputed_contamination.get(field)]
        for field in CONTAMINATION_FIDELITY_FIELDS
        if recorded_contamination.get(field) != recomputed_contamination.get(field)
    }
    return {
        "recomputed": True,
        "recorded_guard_present": True,
        "guard_field_mismatches": guard_diff,
        "contamination_field_mismatches": contamination_diff,
        "matches": not guard_diff and not contamination_diff,
    }


def _recompute_parent(
    record: Mapping[str, Any],
    words: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = {arm: _arm_projection(record.get(arm)) for arm in ("r0", "r2")}
    guards: dict[str, Any] = {}
    fidelity: dict[str, Any] = {}
    for arm in ("r0", "r2"):
        payload = arms[arm]
        tokens = list(payload.get("tokens") or ())
        units = list(payload.get("units") or ())
        if not tokens and not units:
            guards[arm] = payload.get("recorded_guard")
            fidelity[arm] = {"recomputed": False, "reason": "no_recorded_ledger"}
            continue
        attributed = attribute_tokens(tokens, words)
        contamination = sequential_merge_contamination(
            units=units, attributed=attributed, words=words
        )
        guard = arm_guard_record(units=units, attributed=attributed, words=words)
        guards[arm] = guard
        fidelity[arm] = _arm_fidelity(
            payload.get("recorded_guard"),
            guard,
            payload.get("contamination") or {},
            contamination,
        )
    guard = pair_parent_guard(guards.get("r0"), guards.get("r2"))
    row = {key: record.get(key) for key in PARENT_FIELDS}
    row["guard"] = guard
    row["guard_source"] = (
        "recomputed"
        if all(
            isinstance(fidelity[arm], Mapping) and fidelity[arm].get("recomputed")
            for arm in ("r0", "r2")
        )
        else "recorded_only"
    )
    for arm in ("r0", "r2"):
        row[arm] = {
            "contamination": arms[arm].get("contamination"),
            "ledger": {
                "tokens": arms[arm].get("tokens"),
                "units": arms[arm].get("units"),
            },
        }
    return row, fidelity


def _case_from_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "meeting": projection.get("meeting"),
        "phase": projection.get("phase"),
        "parents": list(projection.get("parents") or ()),
        "capture_timing": dict(projection.get("capture_timing") or {}),
        "declared_source_samples": projection.get("declared_source_samples"),
        "sealed_segments": projection.get("sealed_segments"),
        "provider_fault": projection.get("provider_fault"),
        "task_failures": list(projection.get("task_failures") or ()),
        "timing_failures": list(projection.get("timing_failures") or ()),
        "budget_truncated": projection.get("budget_truncated"),
        "safety_failures": [],
    }


def _score_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in SCORE_FIELDS} for row in rows]


def _exclusion_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(row.get("parent_id")): str(row.get("pool_exclusion")) for row in rows}


def _case_verdicts(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    verdicts = []
    for case in cases:
        execution = case_execution_record(case)
        evaluation = case_evaluation_record(case, execution=execution)
        report = u8_case_report(case)
        verdicts.append(
            {
                "meeting": case.get("meeting"),
                "execution_completed": execution["execution_completed"],
                "execution_incomplete_reasons": execution["execution_incomplete_reasons"],
                "evaluation_valid": evaluation["evaluation_valid"],
                "evaluation_invalid_reasons": evaluation["evaluation_invalid_reasons"],
                "n_overlap_unassessable": evaluation["n_overlap_unassessable"],
                "overlap_unassessable_parents": evaluation["overlap_unassessable_parents"],
                "operational_clean": report["operational_clean"],
                "formed_parents": report["formed_parents"],
                "source_accounting": report["source_accounting"],
            }
        )
    return verdicts


def _input_integrity(gate_path: Path) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    checks = {row["check"]: row for row in gate.get("checks") or ()}
    audio_path = None
    for row in gate.get("checks") or ():
        if row.get("check") == "audio_exists":
            audio_path = Path(str(row.get("detail")))
    if audio_path is not None and audio_path.is_file():
        digest = _sha256_file(audio_path)
        audio = {
            "path": str(audio_path),
            "recomputed_sha256": digest,
            "recorded_sha256": checks.get("audio_raw_sha256", {}).get("detail"),
            "matches": digest == checks.get("audio_raw_sha256", {}).get("detail"),
            "bytes": audio_path.stat().st_size,
        }
    else:
        audio = {"path": None if audio_path is None else str(audio_path), "matches": None}
    annotations = {}
    for role in ("A", "B", "C", "D"):
        path = None
        for row in gate.get("checks") or ():
            if row.get("check") == f"xml_exists_{role}":
                path = Path(str(row.get("detail")))
        recorded = checks.get(f"xml_raw_sha256_{role}", {}).get("detail")
        if path is not None and path.is_file():
            digest = _sha256_file(path)
            annotations[role] = {
                "path": str(path),
                "recomputed_sha256": digest,
                "recorded_sha256": recorded,
                "matches": digest == recorded,
            }
        else:
            annotations[role] = {"path": None if path is None else str(path), "matches": None}
    return {
        "gate_path": str(gate_path),
        "gate_manifest_sha256": gate.get("manifest_sha256"),
        "audio_frames_recorded": checks.get("audio_frames", {}).get("detail"),
        "gt_words_recorded": gate.get("gt_words"),
        "duration_s_recorded": gate.get("duration_s"),
        "audio": audio,
        "annotations": annotations,
        "holdout_inspected": False,
    }


def _candidate_identity(policy: Path) -> dict[str, Any]:
    harness = {
        name: _sha256_file(policy / name)
        for name in ("metrics.py", "phase.py", "run.py", "RUNTIME_PIN.json")
        if (policy / name).is_file()
    }
    capsule = None
    for manifest in sorted((policy / ".capsule").glob("*/capsule_manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        overlay = {
            item.get("capsule"): item.get("sha256") for item in payload.get("overlay") or ()
        }
        if overlay.get("experiments/psem_r2_policy/metrics.py") == harness.get("metrics.py"):
            capsule = {
                "fingerprint": payload.get("fingerprint"),
                "manifest": str(manifest),
                "expected_tests": payload.get("expected_tests"),
                "runtime_archive": payload.get("runtime_archive"),
            }
            break
    pin = json.loads((policy / "RUNTIME_PIN.json").read_text(encoding="utf-8"))
    return {
        "canonical_harness_sha256": harness,
        "runtime_pin_expected_tests": pin.get("expected_tests"),
        "matching_capsule": capsule,
        "runtime_archive": pin.get("runtime_archive"),
    }


def _protocol_identity(policy: Path, expansion_path: Path) -> dict[str, Any]:
    protocol = json.loads((policy / "PROTOCOL.json").read_text(encoding="utf-8"))
    expansion = json.loads(expansion_path.read_text(encoding="utf-8"))
    return {
        "protocol_revision": protocol.get("revision"),
        "protocol_status": protocol.get("status"),
        "guard_revision": (protocol.get("guard_oracle") or {}).get("revision"),
        "expansion_authority": expansion.get("revision"),
        "expansion_status": expansion.get("status"),
        "user_selection": (
            expansion.get("subsequent_agreement", {})
            .get("overlap_agreement", {})
            .get("user_selection")
        ),
        "protocol_file_sha256": _sha256_file(policy / "PROTOCOL.json"),
        "expansion_authority_sha256": _sha256_file(expansion_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=str(DEFAULT_CASE))
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--legacy-summary", default=str(LEGACY_SUMMARY))
    parser.add_argument("--gate", default=str(GATE))
    parser.add_argument("--skip-input-verify", action="store_true")
    args = parser.parse_args(argv)

    case_path = Path(args.case)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_sha256 = _sha256_file(case_path)
    normalized_sha256 = _sha256_normalized(case_path)
    with case_path.open("rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        spans = _top_level_spans(mm)
        projection = {name: _decode(mm, spans, name) for name in CASE_FIELDS if name in spans}
        parent_array = mm.find(b"[", mm.find(b'"parents"'))
        records = [json.loads(mm[start:end]) for start, end in _elements(mm, parent_array)]
        mm.close()

    meeting = str(projection.get("meeting") or "")
    words = load_ami_words(meeting)
    parents: list[dict[str, Any]] = []
    fidelity_rows: dict[str, Any] = {}
    mismatching_parents: dict[str, Any] = {}
    recomputed_arms = 0
    for record in records:
        row, fidelity = _recompute_parent(record, words)
        parents.append(row)
        parent_id = str(row.get("parent_id"))
        for arm in ("r0", "r2"):
            entry = fidelity.get(arm)
            if not (isinstance(entry, Mapping) and entry.get("recomputed")):
                continue
            recomputed_arms += 1
            if not entry.get("matches"):
                mismatching_parents.setdefault(parent_id, {})[arm] = entry
        if not all(
            isinstance(fidelity[arm], Mapping) and fidelity[arm].get("recomputed")
            for arm in ("r0", "r2")
        ):
            fidelity_rows[parent_id] = {arm: fidelity[arm] for arm in ("r0", "r2")}

    projection["parents"] = parents
    projection["u10_recomputation"] = {
        "gt_words": len(words),
        "parents": len(parents),
        "parents_with_recomputed_guard": sum(
            1 for row in parents if row.get("guard_source") == "recomputed"
        ),
        "parents_recorded_guard_only": sorted(
            str(row.get("parent_id"))
            for row in parents
            if row.get("guard_source") != "recomputed"
        ),
    }
    case = _case_from_projection(projection)
    sidecar_path = out_dir / "ES2009a-u10-parents.json"
    sidecar_bytes = json.dumps(projection, ensure_ascii=False, indent=1).encode("utf-8")
    sidecar_path.write_bytes(sidecar_bytes)
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()

    summary = aggregate_phase(parents, cases=[case], phase=str(projection.get("phase") or "dev"))
    legacy_decision = dict(projection.get("decision") or {})
    legacy_aggregate = dict(projection.get("aggregate") or {})
    legacy_u8 = dict(projection.get("u8") or {})
    legacy_summary = (
        json.loads(Path(args.legacy_summary).read_text(encoding="utf-8"))
        if Path(args.legacy_summary).is_file()
        else {}
    )

    new_rows = summary["cluster_aggregate"]["cluster_rows"]
    legacy_rows = list(legacy_decision.get("cluster_rows") or ())
    point_unchanged = _score_rows(new_rows) == _score_rows(legacy_rows)
    new_exclusions = _exclusion_map(summary["cluster_aggregate"]["pool_exclusions"])
    legacy_exclusions = _exclusion_map(legacy_aggregate.get("pool_exclusions") or ())
    label_changes = sorted(
        parent_id
        for parent_id in set(new_exclusions) | set(legacy_exclusions)
        if new_exclusions.get(parent_id) != legacy_exclusions.get(parent_id)
    )
    bounds = summary["confirmatory"]["sensitivity"]["formed_parent_selection_bounds"]
    if not point_unchanged:
        raise RuntimeError("U10 rescore changed the primary point estimate")

    overlap_rows = [
        row
        for row in summary["u8"]["overlap_coverage"]["parents"]
        if row.get("overlap_unassessable")
    ]
    record = {
        "revision": "PSEM-R2-U10-OFFLINE-RESCORE-2",
        "kind": "offline_rescore",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "acquisition_revision": {
            "case_path": str(case_path),
            "raw_sha256": raw_sha256,
            "raw_bytes": case_path.stat().st_size,
            "legacy_normalized_sha256": normalized_sha256,
            "legacy_normalized_note": (
                "recorded case_output.sha256 in the supervised attempt-1 summary is the "
                "LF-normalized content hash, not the raw file hash"
            ),
            "recorded_protocol_revision": projection.get("protocol_revision"),
            "declared_parents": projection.get("n_parents"),
            "legacy_verdict_embedded": {
                "execution_completed": projection.get("execution_completed"),
                "evaluation_valid": projection.get("evaluation_valid"),
                "operational_clean": projection.get("operational_clean"),
                "conditional_support": projection.get("conditional_support"),
                "evaluation_invalid_reasons": legacy_u8.get("evaluation_invalid_reasons"),
                "execution_incomplete_reasons": legacy_u8.get("execution_incomplete_reasons"),
                "safety_failures": legacy_u8.get("safety_failures"),
            },
            "legacy_summary_path": str(args.legacy_summary),
            "legacy_summary_sha256": (
                _sha256_file(Path(args.legacy_summary))
                if Path(args.legacy_summary).is_file()
                else None
            ),
            "provider_calls": 0,
        },
        "evaluation_revision": {
            "evaluator": str(Path(__file__).resolve()),
            "candidate": _candidate_identity(POLICY),
            "protocol": _protocol_identity(POLICY, EXPANSION_AUTHORITY),
            "reused_recorded_arm_outputs": True,
            "recomputed_from_recorded_inputs": [
                "attribute_tokens(recorded per-arm tokens, validated GT words)",
                "sequential_merge_contamination(recorded per-arm units, recomputed attribution, GT)",
                "arm_guard_record(recorded per-arm units, recomputed attribution, GT)",
                "pair_parent_guard(recomputed r0 guard, recomputed r2 guard)",
            ],
            "reused_recorded_contamination_for_point_estimate": True,
            "recomputed_derived_fields": [
                "primary_pool_membership",
                "case_execution_record",
                "case_evaluation_record",
                "case_guard_summary",
                "operational_census",
                "cluster_rows",
                "formed_parent_selection_bounds",
                "overlap_coverage_report",
                "confirmatory_decision",
            ],
        },
        "source_identity": {
            "case_path": str(case_path),
            "raw_sha256": raw_sha256,
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sidecar_sha256,
            "sidecar_parents": len(parents),
            "projection_note": (
                "compact sidecar derived read-only from the preserved acquisition; parents carry "
                "recorded per-arm tokens/units, recorded arm contamination and the paired guard "
                "recomputed from those inputs plus the validated GT words"
            ),
        },
        "input_integrity": (
            {"skipped": True} if args.skip_input_verify else _input_integrity(Path(args.gate))
        ),
        "recomputation": {
            "gt_words_validated": len(words),
            "parents": len(parents),
            "parents_with_recomputed_guard": projection["u10_recomputation"][
                "parents_with_recomputed_guard"
            ],
            "parents_recorded_guard_only": projection["u10_recomputation"][
                "parents_recorded_guard_only"
            ],
            "annotation_tokens_recorded_in_arm_guards": sorted(
                {
                    int((row.get("guard") or {}).get("checked", {}).get("annotation_tokens") or 0)
                    for row in parents
                    if isinstance(row.get("guard"), Mapping)
                }
            ),
            "fallback_records": fidelity_rows,
            "recomputed_arms": recomputed_arms,
            "arms_matching_recorded": recomputed_arms
            - sum(len(entry) for entry in mismatching_parents.values()),
            "mismatching_parents": mismatching_parents,
            "overlap_parent_annotation_tokens": {
                row["parent_id"]: row.get("annotation_tokens") for row in overlap_rows
            },
            "overlap_parent_source_ranges": {
                row["parent_id"]: row.get("source_ranges") for row in overlap_rows
            },
        },
        "case_verdicts": _case_verdicts([case]),
        "derived": {
            "cluster_aggregate": summary["cluster_aggregate"],
            "confirmatory": summary["confirmatory"],
            "u8": summary["u8"],
            "operational_census": summary["operational_census"],
            "evaluation_valid": summary["evaluation_valid"],
            "evaluation_invalid_reasons": summary["evaluation_invalid_reasons"],
            "execution_completed": summary["execution_completed"],
            "operational_clean": summary["operational_clean"],
        },
        "comparison": {
            "point_estimate_unchanged": point_unchanged,
            "scorable_character_denominators_unchanged": point_unchanged
            and all(
                new.get("r0_chars") == old.get("r0_chars")
                and new.get("r2_chars") == old.get("r2_chars")
                for new, old in zip(_score_rows(new_rows), _score_rows(legacy_rows))
            ),
            "pool_membership_counts": {
                "legacy": {
                    "n_sequential_parents": legacy_aggregate.get("n_sequential_parents"),
                    "n_non_sequential_excluded": legacy_aggregate.get("n_non_sequential_excluded"),
                    "n_operationally_unsuccessful": legacy_aggregate.get(
                        "n_operationally_unsuccessful"
                    ),
                    "n_degraded_conditional": legacy_aggregate.get("n_degraded_conditional"),
                },
                "u10": {
                    "n_sequential_parents": summary["cluster_aggregate"]["n_sequential_parents"],
                    "n_non_sequential_excluded": summary["cluster_aggregate"][
                        "n_non_sequential_excluded"
                    ],
                    "n_operationally_unsuccessful": summary["cluster_aggregate"][
                        "n_operationally_unsuccessful"
                    ],
                    "n_degraded_conditional": summary["cluster_aggregate"][
                        "n_degraded_conditional"
                    ],
                },
            },
            "pool_exclusion_label_changes": {
                "parent_ids": label_changes,
                "legacy": {
                    parent_id: legacy_exclusions.get(parent_id) for parent_id in label_changes
                },
                "u10": {parent_id: new_exclusions.get(parent_id) for parent_id in label_changes},
            },
            "case_validity_change": {
                "legacy_evaluation_valid": legacy_summary.get("evaluation_valid"),
                "legacy_reasons": legacy_summary.get("u8_evaluation_invalid_reasons"),
                "u10_evaluation_valid": summary["evaluation_valid"],
                "u10_reasons": summary["evaluation_invalid_reasons"],
            },
            "selection_bounds": {
                "N_total": bounds["N_total"],
                "M_total": bounds["M_total"],
                "M_unscorable_total": bounds["M_unscorable_total"],
                "M_overlap_unassessable_total": bounds["M_overlap_unassessable_total"],
                "equal_cluster_mean_lower": bounds["equal_cluster_mean_lower"],
                "equal_cluster_mean_upper": bounds["equal_cluster_mean_upper"],
                "fragility": bounds["fragility"],
                "note": bounds["note"],
            },
            "legacy_decision": {
                "result": legacy_decision.get("result"),
                "pass": legacy_decision.get("pass"),
                "n_eligible_clusters": legacy_decision.get("n_eligible_clusters"),
                "cluster_mean_delta": legacy_decision.get("cluster_mean_delta"),
                "ci95": legacy_decision.get("ci95"),
            },
            "u10_decision": {
                "result": summary["confirmatory"].get("result"),
                "pass": summary["confirmatory"].get("pass"),
                "n_eligible_clusters": summary["confirmatory"].get("n_eligible_clusters"),
                "cluster_mean_delta": summary["confirmatory"].get("cluster_mean_delta"),
                "ci95": summary["confirmatory"].get("ci95"),
            },
        },
        "claim_scope": (
            "DEV single-cluster offline revision only: no confirmatory or efficacy claim, "
            "and no safety claim for measured overlap-unassessable text. HOLDOUT outcomes "
            "stay uninspected and locked."
        ),
    }
    record_path = out_dir / "ES2009a-u10-evaluation.json"
    record_path.write_bytes(json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8"))
    print(f"case={case_path.name} raw_sha256={raw_sha256}")
    print(f"sidecar={sidecar_path} sha256={sidecar_sha256}")
    print(f"record={record_path}")
    print(f"gt_words={len(words)} recomputed_guards={projection['u10_recomputation']['parents_with_recomputed_guard']}/{len(parents)}")
    print(f"evaluation_valid={summary['evaluation_valid']} reasons={summary['evaluation_invalid_reasons']}")
    print(f"point_estimate_unchanged={point_unchanged}")
    print(f"N={bounds['N_total']} M={bounds['M_total']} M_overlap={bounds['M_overlap_unassessable_total']}")
    overlap = summary["u8"]["overlap_coverage"]["overall"]
    print(
        f"overlap: formed={overlap['formed_parents']} accepted={overlap['accepted_nonempty_parents']} "
        f"qualified={overlap['qualified_parents']} chars={overlap['qualified_accepted_chars']}/{overlap['accepted_chars']}"
    )
    for row in overlap_rows:
        print(
            f"  qualified {row['parent_id']} annotation_tokens="
            f"{row['annotation_tokens']} scope=meeting_annotation_source "
            f"excluded={row['excluded_tokens']} "
            f"token_envelope={row['token_envelope']} "
            f"source_ranges={row['source_ranges']} source_interval={row['source_interval']}"
        )
    print(f"confirmatory={summary['confirmatory']['result']} pass={summary['confirmatory']['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
