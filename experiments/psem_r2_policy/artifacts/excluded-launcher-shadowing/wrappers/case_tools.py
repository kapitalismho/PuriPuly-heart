"""Per-case canonical preservation + phase aggregation for the DEV run.

--case       verify one meeting's launcher payload, write the canonical case
             output through the harness phase.write_case_output path, and emit a
             compact summary plus provisional per-case evidence.
--aggregate  hash each canonical payload before parsing, stream only the fields
             consumed by phase.aggregate_phase, and write the existing aggregate
             payload contract without retaining full case JSON objects.

The aggregate path requires the separately provisioned, pinned ijson tooling
environment. It never falls back to loading a complete canonical case.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import secrets
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import UUID, uuid5

TARGET = Path(
    r"C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls"
)


def _requested_capsule_root() -> Path | None:
    try:
        index = sys.argv.index("--capsule-root")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError):
        return None


IMPORT_ROOT = _requested_capsule_root() or TARGET
sys.path.insert(0, str(IMPORT_ROOT))
sys.path.insert(0, str(IMPORT_ROOT / "src"))

from experiments.psem_r2_policy import budget as harness_budget  # noqa: E402
from experiments.psem_r2_policy import phase as harness_phase  # noqa: E402
from experiments.psem_r2_policy.metrics import parent_content_ranges  # noqa: E402

CANONICAL_ARTIFACTS = TARGET / "experiments/psem_r2_policy/artifacts"
AGGREGATION_IMPLEMENTATION = "bounded-ijson-3.4.0.post0"

REQUIRED_CASE_FIELDS = {
    "meeting",
    "phase",
    "parents",
    "capture_timing",
    "declared_source_samples",
    "sealed_segments",
    "timing_failures",
    "task_failures",
    "provider_fault",
}
CASE_VALUE_FIELDS = REQUIRED_CASE_FIELDS - {"parents"} | {
    "severe_guard_failures",
    "safety_failures",
    "budget_truncated",
}
PARENT_SCALAR_FIELDS = {
    "parent_id",
    "meeting",
    "phase",
    "text",
    "cluster_id",
    "sequential_target",
    "status",
    "degraded",
    "clean_completion",
    "text_authority",
    "failure_reason",
    "outcome",
    "terminal_outcome",
    "seal_reason",
    "conserved",
    "incomplete",
    "outage",
    "accounted",
    "provenance_valid",
    "annotation_source",
}
PARENT_VALUE_FIELDS = {"span", "content_ranges", "marks", "latency"}
PARENT_REQUIRED_FIELDS = {
    "parent_id",
    "text",
    "cluster_id",
    "sequential_target",
    "status",
    "clean_completion",
    "text_authority",
    "outcome",
    "conserved",
    "r0",
    "r2",
    "guard",
    "accounted",
    "provenance_valid",
    "marks",
}
GUARD_VALUE_FIELDS = {"failures", "wrong_merge", "same_speaker", "checked"}
ARM_SCALAR_FIELDS = {
    "eligible",
    "proportion",
    "attributable_chars",
    "contaminated_chars",
    "unknown_chars",
    "mixed_chars",
    "unaligned_chars",
    "coverage",
    "reason",
}


def _load_ijson() -> Any:
    try:
        import ijson
    except ImportError as exc:
        raise SystemExit(
            "bounded aggregate requires ijson==3.4.0.post0; run it with the isolated "
            "artifacts/dev-bounded-aggregate/tooling-venv interpreter"
        ) from exc
    if ijson.__version__ != "3.4.0.post0":
        raise SystemExit(
            f"bounded aggregate requires ijson==3.4.0.post0, found {ijson.__version__}"
        )
    return ijson


def _consume_value(
    first_event: str,
    first_value: Any,
    events: Iterator[tuple[str, str, Any]],
    ijson: Any,
) -> Any:
    if first_event not in {"start_map", "start_array"}:
        return first_value
    builder = ijson.common.ObjectBuilder()
    builder.event(first_event, first_value)
    depth = 1
    while depth:
        _prefix, event, value = next(events)
        builder.event(event, value)
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1
    return builder.value


def _translation_evidence_state() -> dict[str, Any]:
    return {
        "parents": 0,
        "empty_or_no_asr_parents": 0,
        "arms": {
            arm: {
                "translated_true": 0,
                "translated_false": 0,
                "child_units": 0,
                "child_texts": 0,
                "nonempty_child_texts": 0,
                "translation_outputs": 0,
                "nonempty_translation_outputs": 0,
                "output_field_available_parents": 0,
                "request_records": 0,
                "admission_marks": 0,
                "completion_marks": 0,
                "statuses": Counter(),
            }
            for arm in ("r0", "r2")
        },
    }


def _observe_translation_evidence(
    evidence: dict[str, Any], relative: str, event: str, value: Any
) -> None:
    arm = relative.split(".", 1)[0]
    if arm not in evidence["arms"]:
        return
    suffix = relative.removeprefix(f"{arm}.")
    row = evidence["arms"][arm]
    if suffix == "translated" and event == "boolean":
        row["translated_true" if value else "translated_false"] += 1
    elif suffix == "child_translations" and event == "start_array":
        row["output_field_available_parents"] += 1
    elif suffix == "child_translations.item" and event == "string":
        row["translation_outputs"] += 1
        row["nonempty_translation_outputs"] += bool(value)
    elif suffix == "child_ids.item" and event == "string":
        row["child_units"] += 1
    elif suffix == "child_texts.item" and event == "string":
        row["child_texts"] += 1
        row["nonempty_child_texts"] += bool(value)
    elif suffix == "requests.item.id" and event == "string":
        row["request_records"] += 1
    elif suffix == "ledger.marks.translation_admission" and event == "number":
        row["admission_marks"] += 1
    elif suffix == "ledger.marks.translation_completion" and event == "number":
        row["completion_marks"] += 1
    elif suffix == "outcomes.item" and event == "string":
        row["statuses"][str(value)] += 1
    elif suffix.startswith("child_outcomes.") and event == "string":
        row["statuses"][str(value)] += 1


def _finish_translation_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        **evidence,
        "arms": {
            arm: {**row, "statuses": dict(row["statuses"])} for arm, row in evidence["arms"].items()
        },
        "interpretation": (
            "Counts recorded fields only. translation_outputs counts preserved output strings; "
            "a zero means that arm's consumer schema exposed no output field, not that translation "
            "semantics were inspected or that outputs were empty."
        ),
    }


def _assign_nested(target: dict[str, Any], outer: str, inner: str, value: Any) -> None:
    target.setdefault(outer, {})[inner] = value


def _validate_projected_case(case: Mapping[str, Any], present: set[str], path: Path) -> None:
    missing = sorted(REQUIRED_CASE_FIELDS - present)
    if missing:
        raise SystemExit(f"canonical case missing required fields {missing}: {path}")
    if not isinstance(case.get("meeting"), str) or not case["meeting"]:
        raise SystemExit(f"canonical case has invalid meeting: {path}")
    if case.get("phase") != "dev":
        raise SystemExit(f"canonical case has unexpected phase {case.get('phase')!r}: {path}")
    if not isinstance(case.get("parents"), list):
        raise SystemExit(f"canonical case parents is not an array: {path}")


def _validate_projected_parent(
    parent: Mapping[str, Any], present: set[str], path: Path, index: int
) -> None:
    missing = sorted(PARENT_REQUIRED_FIELDS - present)
    if missing:
        raise SystemExit(f"canonical parent {index} missing required fields {missing}: {path}")
    if not isinstance(parent.get("parent_id"), (str, int)):
        raise SystemExit(f"canonical parent {index} has invalid parent_id: {path}")
    for name in ("r0", "r2", "guard", "marks"):
        if not isinstance(parent.get(name), Mapping):
            raise SystemExit(f"canonical parent {index} has invalid {name}: {path}")


def _stream_case_projection(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ijson = _load_ijson()
    case: dict[str, Any] = {"parents": []}
    case_present: set[str] = set()
    current: dict[str, Any] | None = None
    parent_present: set[str] = set()
    parent_index = -1
    translation_evidence = _translation_evidence_state()
    with path.open("rb") as handle:
        events = iter(ijson.parse(handle, use_float=True))
        try:
            first_prefix, first_event, _first_value = next(events)
            if first_prefix != "" or first_event != "start_map":
                raise SystemExit(f"canonical case root is not an object: {path}")
            for prefix, event, value in events:
                if prefix == "" and event == "end_map":
                    break
                if prefix == "parents" and event == "start_array":
                    if "parents" in case_present:
                        raise SystemExit(f"canonical case has duplicate parents array: {path}")
                    case_present.add("parents")
                    continue
                if prefix == "parents.item" and event == "start_map":
                    if current is not None:
                        raise SystemExit(f"nested canonical parent object: {path}")
                    current = {}
                    parent_present = set()
                    parent_index += 1
                    continue
                if prefix == "parents.item" and current is None and event != "end_map":
                    raise SystemExit(
                        f"canonical parent {parent_index + 1} is not an object: {path}"
                    )
                if prefix == "parents.item" and event == "end_map":
                    if current is None:
                        raise SystemExit(f"canonical parent ended without start: {path}")
                    _validate_projected_parent(current, parent_present, path, parent_index)
                    case["parents"].append(current)
                    translation_evidence["parents"] += 1
                    translation_evidence["empty_or_no_asr_parents"] += not bool(
                        str(current.get("text") or "")
                    )
                    current = None
                    continue
                if current is not None:
                    relative = prefix.removeprefix("parents.item.")
                    _observe_translation_evidence(translation_evidence, relative, event, value)
                    if relative in {"r0", "r2", "guard"} and event == "start_map":
                        current.setdefault(relative, {})
                        parent_present.add(relative)
                    if relative in PARENT_SCALAR_FIELDS and event in {
                        "string",
                        "number",
                        "boolean",
                        "null",
                    }:
                        current[relative] = value
                        parent_present.add(relative)
                    elif relative in PARENT_VALUE_FIELDS and event != "map_key":
                        current[relative] = _consume_value(event, value, events, ijson)
                        parent_present.add(relative)
                    elif relative == "receipt.content_ranges" and event != "map_key":
                        current.setdefault("receipt", {})["content_ranges"] = _consume_value(
                            event, value, events, ijson
                        )
                    elif (
                        relative in {"r0.contamination", "r2.contamination"} and event != "map_key"
                    ):
                        outer, inner = relative.split(".", 1)
                        _assign_nested(
                            current, outer, inner, _consume_value(event, value, events, ijson)
                        )
                        parent_present.add(outer)
                    elif relative == "guard.assessed" and event in {"boolean", "null"}:
                        _assign_nested(current, "guard", "assessed", value)
                        parent_present.add("guard")
                    elif (
                        relative.startswith(("r0.", "r2."))
                        and relative.split(".", 1)[1] in ARM_SCALAR_FIELDS
                        and event in {"string", "number", "boolean", "null"}
                    ):
                        outer, inner = relative.split(".", 1)
                        _assign_nested(current, outer, inner, value)
                        parent_present.add(outer)
                    elif relative.startswith("guard.") and event != "map_key":
                        inner = relative.removeprefix("guard.")
                        if inner in GUARD_VALUE_FIELDS:
                            _assign_nested(
                                current,
                                "guard",
                                inner,
                                _consume_value(event, value, events, ijson),
                            )
                            parent_present.add("guard")
                    continue
                if prefix == "seal_lateness.violations" and event == "number":
                    case.setdefault("seal_lateness", {})["violations"] = value
                    continue
                if prefix in CASE_VALUE_FIELDS and event != "map_key":
                    case[prefix] = _consume_value(event, value, events, ijson)
                    case_present.add(prefix)
            else:
                raise SystemExit(f"canonical case ended before root object closed: {path}")
            try:
                trailing = next(events)
            except StopIteration:
                trailing = None
            if trailing is not None:
                raise SystemExit(f"canonical case has trailing JSON content: {path}")
        except (ijson.JSONError, UnicodeError, StopIteration) as exc:
            raise SystemExit(f"invalid or truncated canonical JSON: {path}: {exc}") from exc
    if current is not None:
        raise SystemExit(f"canonical case ended inside parent object: {path}")
    _validate_projected_case(case, case_present, path)
    return (
        case,
        {
            "parser": f"ijson=={ijson.__version__}",
            "backend": ijson.backend,
            "projection": AGGREGATION_IMPLEMENTATION,
        },
        _finish_translation_evidence(translation_evidence),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _outcome_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    parents = list(payload.get("parents") or ())
    outcomes = Counter(
        str((row.get("receipt") or {}).get("outcome") or row.get("outcome") or "")
        for row in parents
    )
    terminal = Counter(
        str((row.get("receipt") or {}).get("terminal_outcome") or row.get("terminal_outcome") or "")
        for row in parents
    )
    authority = Counter(
        str((row.get("receipt") or {}).get("text_authority") or row.get("text_authority") or "")
        for row in parents
    )
    failure = Counter(
        str(row.get("failure_reason") or "") for row in parents if row.get("failure_reason")
    )
    return {
        "parents": len(parents),
        "outcomes": dict(outcomes),
        "terminal_outcomes": dict(terminal),
        "text_authority": dict(authority),
        "failure_reasons": dict(failure),
        "accounted_all": all(bool(row.get("accounted", True)) for row in parents),
        "provenance_valid_all": all(bool(row.get("provenance_valid", True)) for row in parents),
        "conserved_false": sum(1 for row in parents if row.get("conserved") is False),
    }


def _compact(payload: Mapping[str, Any]) -> dict[str, Any]:
    ledger = harness_budget.BudgetLedger(harness_budget.LEDGER_PATH).snapshot()
    latency = payload.get("latency_by_operation") or {}
    seal = payload.get("seal_lateness") or {}
    u8 = payload.get("u8") or {}
    return {
        "meeting": payload.get("meeting"),
        "phase": payload.get("phase"),
        "ok": payload.get("ok"),
        "completed": payload.get("completed"),
        "execution_completed": payload.get("execution_completed"),
        "evaluation_valid": payload.get("evaluation_valid"),
        "operational_clean": payload.get("operational_clean"),
        "clean_completion": payload.get("clean_completion"),
        "conditional_support": payload.get("conditional_support"),
        "refused": payload.get("refused"),
        "paid_blocked": payload.get("paid_blocked"),
        "protocol_revision": payload.get("protocol_revision"),
        "ledger_path": payload.get("ledger_path"),
        "u8_execution_incomplete_reasons": u8.get("execution_incomplete_reasons"),
        "u8_evaluation_invalid_reasons": u8.get("evaluation_invalid_reasons"),
        "u8_safety_failures": u8.get("safety_failures"),
        "u8_operational_census": u8.get("operational_census"),
        "counts": _outcome_counts(payload),
        "capture_timing": payload.get("capture_timing"),
        "dispatch": payload.get("dispatch"),
        "seal_lateness_summary": {
            "keys": sorted(seal.keys())[:12],
            "max_lateness_s": seal.get("max_lateness_s"),
            "c5_deadline_violations": seal.get("violations"),
        },
        "latency_by_operation": latency,
        "timing_failures": payload.get("timing_failures"),
        "task_failures": payload.get("task_failures"),
        "provider_fault": payload.get("provider_fault"),
        "budget_truncated": payload.get("budget_truncated"),
        "artifact": payload.get("artifact"),
        "ledger": {
            "spent_usd": ledger.spent_usd,
            "reserved_usd": ledger.reserved_usd,
            "remaining_usd": ledger.remaining_usd,
            "phase_spent": ledger.phase_spent,
            "phase_reserved": ledger.phase_reserved,
            "credit_usd": ledger.credit_usd,
            "entries": len(ledger.entries),
        },
    }


def case_mode(args: argparse.Namespace) -> int:
    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(args.stdout)
    payload = _json_load(stdout_path)
    assert harness_phase.ARTIFACTS == CANONICAL_ARTIFACTS, harness_phase.ARTIFACTS
    assert harness_budget.LEDGER_PATH == CANONICAL_ARTIFACTS / "budget_ledger.json", (
        harness_budget.LEDGER_PATH
    )
    output = harness_phase.write_case_output("dev", args.meeting, payload)
    (case_dir / "canonical_case.json").write_text(
        json.dumps(
            {
                "meeting": args.meeting,
                "case_output": output,
                "stdout_sha256": sha256_file(stdout_path),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    meta = {}
    if args.launch_meta:
        for line in (
            Path(args.launch_meta).read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    meta = json.loads(stripped)
                    break
                except json.JSONDecodeError:
                    continue
    summary = {
        "case_output": output,
        "stdout": {
            "path": str(stdout_path),
            "bytes": stdout_path.stat().st_size,
            "sha256": sha256_file(stdout_path),
        },
        "launch": {
            "capsule_root": meta.get("capsule_root"),
            "fingerprint": meta.get("fingerprint"),
            "ledger_path": meta.get("ledger_path"),
            "case_output_dir": meta.get("case_output_dir"),
            "modules": meta.get("modules"),
            "prompt": meta.get("prompt"),
            "prompt_probe": meta.get("prompt_probe"),
        },
        **_compact(payload),
    }
    (case_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "meeting",
                    "ok",
                    "completed",
                    "execution_completed",
                    "evaluation_valid",
                    "operational_clean",
                    "counts",
                    "ledger",
                    "case_output",
                    "launch",
                )
            },
            indent=1,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


def _project_parents(case: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    parents: list[dict] = []
    latencies: list[dict] = []
    for row in list(case.get("parents") or ()):
        parents.append(
            {
                "parent_id": row.get("parent_id"),
                "meeting": row.get("meeting"),
                "phase": row.get("phase"),
                "text": row.get("text") or "",
                "cluster_id": row.get("cluster_id") or row.get("meeting"),
                "sequential_target": bool(row.get("sequential_target")),
                "status": row.get("status"),
                "degraded": bool(row.get("degraded")),
                "clean_completion": bool(row.get("clean_completion")),
                "text_authority": row.get("text_authority"),
                "failure_reason": row.get("failure_reason"),
                "outcome": row.get("outcome"),
                "terminal_outcome": row.get("terminal_outcome"),
                "seal_reason": row.get("seal_reason"),
                "conserved": row.get("conserved"),
                "span": row.get("span"),
                "content_ranges": parent_content_ranges(row),
                "r0": row.get("r0") or {},
                "r2": row.get("r2") or {},
                "guard": row.get("guard"),
                "latency": row.get("latency"),
                "annotation_source": row.get("annotation_source"),
                "incomplete": bool(row.get("incomplete")),
                "outage": bool(row.get("outage")),
                "accounted": bool(row.get("accounted", True)),
                "provenance_valid": bool(row.get("provenance_valid", True)),
            }
        )
        latencies.append(
            {
                **(row.get("latency") or row.get("marks") or {}),
                "_nontranslating_empty_parent": not bool(str(row.get("text") or "")),
            }
        )
    return parents, latencies


def _manifest_inputs(path: Path) -> list[tuple[Path, str, dict[str, Any]]]:
    manifest = _json_load(path)
    expected = list(harness_phase.declared_meetings("dev"))
    if manifest.get("protocol_revision") != harness_phase.load_protocol().get("revision"):
        raise SystemExit(f"cohort manifest protocol revision mismatch: {path}")
    if manifest.get("cohort_order") != expected:
        raise SystemExit(f"cohort manifest must contain exactly {expected}: {path}")
    indexed = {row.get("meeting"): row for row in manifest.get("cases") or ()}
    if list(indexed) != expected or len(indexed) != len(expected):
        raise SystemExit(f"cohort manifest cases must contain exactly {expected} in order: {path}")
    inputs: list[tuple[Path, str, dict[str, Any]]] = []
    for meeting in expected:
        row = indexed[meeting]
        case_output = row.get("case_output")
        if not isinstance(case_output, Mapping):
            raise SystemExit(f"cohort manifest case_output missing for {meeting}: {path}")
        case_path = TARGET / str(case_output.get("path") or "")
        digest = str(case_output.get("sha256") or "")
        if len(digest) != 64:
            raise SystemExit(f"cohort manifest case hash invalid for {meeting}: {path}")
        provenance = {
            "meeting": meeting,
            "attempt": row.get("attempt"),
            "capsule_fingerprint": row.get("capsule_fingerprint"),
            "case_output": dict(case_output),
            "stdout_sha256": (row.get("stdout") or {}).get("sha256"),
            "summary_sha256": (row.get("summary") or {}).get("sha256"),
            "cohort_manifest": {"path": str(path), "sha256": sha256_file(path)},
        }
        inputs.append((case_path, digest, provenance))
    return inputs


def _directory_inputs(raw: str) -> list[tuple[Path, str, dict[str, Any]]]:
    inputs: list[tuple[Path, str, dict[str, Any]]] = []
    for case_dir_raw in raw.split(","):
        case_dir = Path(case_dir_raw.strip())
        pointer_path = case_dir / "canonical_case.json"
        pointer = _json_load(pointer_path)
        case_output = pointer.get("case_output")
        if not isinstance(case_output, Mapping):
            raise SystemExit(f"canonical case pointer missing case_output: {pointer_path}")
        case_path = Path(str(case_output.get("path") or ""))
        digest = str(case_output.get("sha256") or "")
        if len(digest) != 64:
            raise SystemExit(f"canonical case pointer has invalid hash: {pointer_path}")
        inputs.append(
            (
                case_path,
                digest,
                {
                    "meeting": pointer.get("meeting"),
                    "case_output": dict(case_output),
                    "stdout_sha256": pointer.get("stdout_sha256"),
                },
            )
        )
    return inputs


def aggregate_mode(args: argparse.Namespace) -> int:
    parents: list[dict] = []
    marks: list[dict] = []
    cases: list[dict] = []
    provenance: list[dict] = []
    inputs = (
        _manifest_inputs(Path(args.manifest)) if args.manifest else _directory_inputs(args.dirs)
    )
    for case_path, expected_hash, case_provenance in inputs:
        observed = sha256_file(case_path)
        if observed != expected_hash:
            raise SystemExit(f"case output hash mismatch: {case_path}")
        case, implementation, translation_evidence = _stream_case_projection(case_path)
        expected_meeting = case_provenance.get("meeting")
        if expected_meeting and case.get("meeting") != expected_meeting:
            raise SystemExit(
                f"case meeting mismatch: expected {expected_meeting}, "
                f"observed {case.get('meeting')}: {case_path}"
            )
        case_parents, case_marks = _project_parents(case)
        parents.extend(case_parents)
        marks.extend(case_marks)
        cases.append(case)
        provenance.append(
            {
                **case_provenance,
                "observed_sha256": observed,
                "aggregation_implementation": implementation,
                "translation_evidence": translation_evidence,
                "session_c5_deadline_violations": {
                    "scope": "case_session",
                    "n_available": int("violations" in (case.get("seal_lateness") or {})),
                    "n_missing": int("violations" not in (case.get("seal_lateness") or {})),
                    "count": (case.get("seal_lateness") or {}).get("violations"),
                },
            }
        )
    summary = harness_phase.aggregate_phase(parents, marks=marks, cases=cases, phase="dev")
    payload = {
        "phase": "dev",
        "case_provenance": provenance,
        "n_parents": len(parents),
        "aggregate": summary,
    }
    out = Path(args.out)
    out.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    decision = summary.get("confirmatory") or {}
    compact = {
        "n_parents": len(parents),
        "execution_completed": summary.get("execution_completed"),
        "evaluation_valid": summary.get("evaluation_valid"),
        "operational_clean": summary.get("operational_clean"),
        "execution_incomplete_reasons": summary.get("execution_incomplete_reasons"),
        "evaluation_invalid_reasons": summary.get("evaluation_invalid_reasons"),
        "conditional_support": decision.get("conditional_support"),
        "decision_keys": sorted(decision.keys()),
        "n_operationally_unsuccessful": summary.get("n_operationally_unsuccessful"),
        "n_degraded_conditional": summary.get("n_degraded_conditional"),
        "n_incomplete_source_parents": summary.get("n_incomplete_source_parents"),
        "operational_census": summary.get("operational_census"),
        "coverage": (summary.get("cluster_aggregate") or {}).get("coverage"),
        "excluded_history": summary.get("excluded_history"),
        "u8_keys": sorted((summary.get("u8") or {}).keys()),
        "u8_safety_failures": (summary.get("u8") or {}).get("safety_failures"),
        "sensitivity_keys": sorted(((summary.get("u8") or {}).get("sensitivity") or {}).keys()),
        "aggregate_payload": {
            "path": str(out),
            "sha256": sha256_file(out),
            "bytes": out.stat().st_size,
        },
    }
    print(json.dumps(compact, indent=1, ensure_ascii=False, default=str))
    return 0


U15_ACQUISITION_ID = "DEV-R2-TEXT-REACQUISITION-1"
U15_MANIFEST_REVISION = "U15-TEXT-REACQUISITION-INPUT-1"
U15_JOURNAL_REVISION = "U15-TEXT-REACQUISITION-JOURNAL-1"
U15_INSPECTION_REVISION = "R2-PAIRED-TEXT-RUBRIC-2"
U15_ANALYSIS_REVISION = "U15-TEXT-REACQUISITION-ANALYSIS-3"
U17_CONTINUATION_ID = f"{U15_ACQUISITION_ID}-CONTINUATION-1"
U17_MANIFEST_REVISION = "U17-TEXT-REACQUISITION-CONTINUATION-INPUT-1"
U17_JOURNAL_REVISION = "U17-TEXT-REACQUISITION-CONTINUATION-JOURNAL-1"
U17_PRIOR_PREPARED_SHA256 = "5fce260381d61992fb1c42ce2061ed539fcb4f2753bd11cd06e7561f529e27b6"
U17_PRIOR_JOURNAL_SHA256 = "093ed6472ffe8347d33efdc8a69a98ddd800a60b45107ec5320723be72d72b9f"
U17_PRIOR_CLAIM_SHA256 = "15c305bbe561ea411b853e6a5519248c1cea0bf625a764e1b866e4cb0111b74e"
U17_EXCLUDED_REQUESTS = 240
U17_MAX_REQUESTS = 2217
U17_PRIOR_RESERVED_USD = 0.10170160000000006
U17_REMAINING_RESERVED_USD = 0.9397442
U15_COHORT_SHA256 = "93dd06414d9c1c52235a903f3ba323e235d6f299125c1f144754d31966c5c87f"
U15_MAX_REQUESTS = 2457
U15_RESERVE_CAP_USD = 1.07
U15_MODEL = "google/gemma-4-26b-a4b-it"
U15_MAX_TOKENS = 100
U15_EXPECTED_REQUESTS = {
    "ES2009a": 266,
    "ES2009c": 378,
    "ES2009d": 407,
    "ES2002b": 388,
    "EN2009d": 1018,
}
U15_NAMESPACE = UUID("6a3ba228-4039-5dd2-913b-48e96cc67d87")
U15_AUTHORITY = (
    TARGET / "experiments/psem_e2o2_continuous_ownership/EXPANSION_AUTHORITY.json"
)
U15_PROTOCOL = TARGET / "experiments/psem_r2_policy/PROTOCOL.json"
U15_BILLING = TARGET / "experiments/psem_r2_policy/BILLING_BOUNDS.json"
U15_LEDGER = TARGET / "experiments/psem_r2_policy/artifacts/budget_ledger.json"


def _canonical_json_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_capsule() -> dict[str, Any]:
    capsule = _requested_capsule_root()
    if capsule is None or capsule != IMPORT_ROOT:
        raise SystemExit("--capsule-root is required for U15 translation modes")
    manifest_path = capsule / "capsule_manifest.json"
    manifest = _json_load(manifest_path)
    expected_archive = "819465e74b847e0a46c0e0968b52d76fccc1a95fa146716411d945690e164416"
    if (manifest.get("runtime_archive") or {}).get("sha256") != expected_archive:
        raise SystemExit("capsule runtime archive identity mismatch")
    from experiments.psem_r2_policy import live_runner
    from puripuly_heart.providers.llm import openrouter

    origins = {
        "budget": str(Path(inspect.getfile(harness_budget)).resolve()),
        "live_runner": str(Path(inspect.getfile(live_runner)).resolve()),
        "openrouter": str(Path(inspect.getfile(openrouter)).resolve()),
    }
    for name, raw in origins.items():
        if capsule not in Path(raw).parents:
            raise SystemExit(f"{name} was not imported from the prepared capsule: {raw}")
    return {
        "preparation": {
            "root": str(capsule),
            "fingerprint": manifest.get("fingerprint"),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "stable": {
            "runtime_archive_sha256": expected_archive,
            "prompt": manifest.get("prompt"),
            "budget_sha256": sha256_file(Path(origins["budget"])),
            "live_runner_sha256": sha256_file(Path(origins["live_runner"])),
            "openrouter_sha256": sha256_file(Path(origins["openrouter"])),
        },
        "modules": origins,
    }


def _cohort_sources(path: Path) -> tuple[dict[str, Any], list[tuple[str, Path, str]]]:
    if sha256_file(path) != U15_COHORT_SHA256:
        raise SystemExit("U15 cohort manifest SHA-256 mismatch")
    manifest = _json_load(path)
    expected_order = list(U15_EXPECTED_REQUESTS)
    if manifest.get("cohort_order") != expected_order:
        raise SystemExit("U15 cohort order mismatch")
    rows: list[tuple[str, Path, str]] = []
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_order):
        raise SystemExit("U15 cohort cases are missing")
    for meeting, row in zip(expected_order, cases, strict=True):
        case = row.get("case_output") or {}
        raw_path = (TARGET / str(case.get("path") or "")).resolve()
        raw_sha = str(case.get("sha256") or "")
        if row.get("meeting") != meeting or row.get("attempt") != "attempt-2":
            raise SystemExit(f"unexpected U15 cohort row for {meeting}")
        if not raw_path.is_file() or sha256_file(raw_path) != raw_sha:
            raise SystemExit(f"raw case SHA-256 mismatch for {meeting}")
        rows.append((meeting, raw_path, raw_sha))
    return manifest, rows


def _request_body_and_bound(request: Mapping[str, Any]) -> tuple[str, int, float]:
    from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient

    client = HttpxOpenRouterClient(
        api_key="offline-bound",
        model=U15_MODEL,
        max_tokens=U15_MAX_TOKENS,
    )
    body = client._build_request_body(
        text=str(request["text"]),
        system_prompt=str(request["system_prompt"]),
        source_language=str(request["source_language"]),
        target_language=str(request["target_language"]),
        context=str(request["context"]),
        scene_participant_count=request.get("scene_participant_count"),
    )
    serialized = json.dumps(body, ensure_ascii=False)
    byte_count = len(serialized.encode("utf-8"))
    reserve = harness_budget.openrouter_reserve_usd(
        serialized_request=serialized, max_tokens=U15_MAX_TOKENS
    )
    return serialized, byte_count, reserve


def _prepare_parent(
    parent: Mapping[str, Any],
    *,
    meeting: str,
    parent_order: int,
    request_order: int,
    request_ids: set[str],
    child_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    parent_id = str(parent.get("parent_id") or "")
    if not parent_id or parent.get("meeting") != meeting:
        raise SystemExit(f"invalid parent identity at {meeting}:{parent_order}")
    r0 = parent.get("r0") or {}
    r2 = parent.get("r2") or {}
    r0_ids = list(r0.get("child_ids") or ())
    r0_texts = list(r0.get("child_texts") or ())
    r0_outcomes = list(r0.get("outcomes") or ())
    r0_outputs = list(r0.get("child_translations") or ())
    if not (len(r0_ids) == len(r0_texts) == len(r0_outcomes) == len(r0_outputs)):
        raise SystemExit(f"unaligned retained R0 evidence at {meeting}:{parent_order}")
    retained_r0 = [
        {
            "child_id": str(child_id),
            "text": str(text),
            "status": str(status),
            "response": response if isinstance(response, str) else None,
        }
        for child_id, text, status, response in zip(
            r0_ids, r0_texts, r0_outcomes, r0_outputs, strict=True
        )
    ]
    r2_ids = list(r2.get("child_ids") or ())
    r2_texts = list(r2.get("child_texts") or ())
    r2_groups = list(r2.get("child_groups") or ())
    statuses = dict(r2.get("child_outcomes") or {})
    original_requests = list(r2.get("requests") or ())
    if not (len(r2_ids) == len(r2_texts) == len(r2_groups)):
        raise SystemExit(f"unaligned R2 child evidence at {meeting}:{parent_order}")
    by_child: dict[str, Mapping[str, Any]] = {}
    for request in original_requests:
        child_id = str(request.get("utterance_id") or "")
        if child_id in by_child:
            raise SystemExit(f"duplicate request child at {meeting}:{parent_order}")
        by_child[child_id] = request
    prepared_requests: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for child_order, (child_id_raw, text_raw, group_raw) in enumerate(
        zip(r2_ids, r2_texts, r2_groups, strict=True)
    ):
        child_id = str(child_id_raw)
        if not child_id or child_id in child_ids:
            raise SystemExit(f"duplicate or empty R2 child ID: {child_id!r}")
        child_ids.add(child_id)
        status = str(statuses.get(child_id) or "")
        request = by_child.pop(child_id, None)
        child = {
            "child_order": child_order,
            "original_child_id": child_id,
            "ownership_group_id": str(group_raw),
            "text": str(text_raw),
            "original_status": status,
            "original_request_id": None if request is None else str(request.get("id") or ""),
        }
        children.append(child)
        if status == "source_only":
            if request is not None:
                raise SystemExit(f"source_only child has a request: {child_id}")
            continue
        if request is None:
            raise SystemExit(f"submitted R2 child has no request: {child_id}")
        original_id = str(request.get("id") or "")
        if not original_id or original_id in request_ids:
            raise SystemExit(f"duplicate or empty original request ID: {original_id!r}")
        request_ids.add(original_id)
        immutable = {
            "text": str(request.get("text") or ""),
            "system_prompt": str(request.get("system_prompt") or ""),
            "source_language": request.get("source_language"),
            "target_language": request.get("target_language"),
            "context": request.get("context"),
            "scene_participant_count": request.get("scene_participant_count"),
        }
        if (
            immutable["text"] != child["text"]
            or immutable["source_language"] != "en"
            or immutable["target_language"] != "ko"
            or immutable["context"] != ""
            or immutable["scene_participant_count"] is not None
        ):
            raise SystemExit(f"original request configuration drift: {original_id}")
        serialized, byte_count, reserve = _request_body_and_bound(immutable)
        if int(request.get("bytes") or -1) != byte_count:
            raise SystemExit(f"original request byte bound mismatch: {original_id}")
        new_utterance_id = str(uuid5(U15_NAMESPACE, f"{U15_ACQUISITION_ID}:{original_id}:{child_id}"))
        if new_utterance_id == child_id:
            raise SystemExit(f"new and original utterance IDs collide: {child_id}")
        prepared_requests.append(
            {
                "ordered_index": request_order,
                "meeting": meeting,
                "parent_order": parent_order,
                "parent_id": parent_id,
                **child,
                "new_utterance_id": new_utterance_id,
                **immutable,
                "request_body_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                "bytes": byte_count,
                "reserve_usd": reserve,
            }
        )
        request_order += 1
    if by_child:
        raise SystemExit(f"request IDs outside R2 children at {meeting}:{parent_order}")
    return (
        {
            "ordered_index": parent_order,
            "meeting": meeting,
            "parent_id": parent_id,
            "accepted_text": str(parent.get("text") or ""),
            "operational_status": parent.get("status"),
            "terminal_outcome": parent.get("terminal_outcome"),
            "failure_reason": parent.get("failure_reason"),
            "retained_r0": retained_r0,
            "r2_children": children,
        },
        prepared_requests,
        request_order,
    )


def translation_prepare_mode(args: argparse.Namespace) -> int:
    capsule = _require_capsule()
    cohort_path = Path(args.manifest).resolve()
    cohort, sources = _cohort_sources(cohort_path)
    protocol = _json_load(U15_PROTOCOL)
    rubric = ((protocol.get("measurements") or {}).get("translation_rubric") or {})
    contract = protocol.get("u15_translation_reacquisition") or {}
    if (
        protocol.get("revision") != "R2-POLICY-DIRECTOR-13"
        or rubric.get("revision") != U15_INSPECTION_REVISION
        or contract.get("acquisition_id") != U15_ACQUISITION_ID
        or int(contract.get("maximum_requests") or 0) != U15_MAX_REQUESTS
        or abs(float(contract.get("additional_reservation_cap_usd") or 0) - U15_RESERVE_CAP_USD)
        > 1e-12
    ):
        raise SystemExit("frozen U15 protocol/rubric contract mismatch")
    parents: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    parent_ids: set[str] = set()
    child_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    per_meeting: Counter[str] = Counter()
    original_statuses: Counter[str] = Counter()
    r0_statuses: Counter[str] = Counter()
    request_order = 0
    for meeting, raw_path, _raw_sha in sources:
        ijson = _load_ijson()
        with raw_path.open("rb") as handle:
            for local_order, parent in enumerate(ijson.items(handle, "parents.item", use_float=True)):
                prepared_parent, rows, request_order = _prepare_parent(
                    parent,
                    meeting=meeting,
                    parent_order=len(parents),
                    request_order=request_order,
                    request_ids=request_ids,
                    child_ids=child_ids,
                )
                if prepared_parent["parent_id"] in parent_ids:
                    raise SystemExit(f"duplicate parent ID: {prepared_parent['parent_id']}")
                parent_ids.add(prepared_parent["parent_id"])
                parents.append(prepared_parent)
                for row in rows:
                    prompt_hashes.add(hashlib.sha256(row["system_prompt"].encode("utf-8")).hexdigest())
                    per_meeting[meeting] += 1
                    original_statuses[row["original_status"]] += 1
                for row in prepared_parent["r2_children"]:
                    if row["original_status"] == "source_only":
                        original_statuses["source_only"] += 1
                for row in prepared_parent["retained_r0"]:
                    r0_statuses[row["status"]] += 1
                requests.extend(rows)
    reserve_sum = sum(float(row["reserve_usd"]) for row in requests)
    max_bytes = max((int(row["bytes"]) for row in requests), default=0)
    conservative = U15_MAX_REQUESTS * harness_budget.openrouter_reserve_usd(
        serialized_request=b"x" * max_bytes, max_tokens=U15_MAX_TOKENS
    )
    if (
        len(parents) != 2482
        or sum(bool(row["accepted_text"]) for row in parents) != 2367
        or len(child_ids) != 2459
        or len(requests) != U15_MAX_REQUESTS
        or dict(per_meeting) != U15_EXPECTED_REQUESTS
        or original_statuses != Counter({"translated": 2455, "failed": 2, "source_only": 2})
        or r0_statuses != Counter({"translated": 2363, "failed": 4})
        or len(prompt_hashes) != 1
        or max_bytes != 4010
        or reserve_sum > U15_RESERVE_CAP_USD + 1e-12
        or conservative > U15_RESERVE_CAP_USD + 1e-12
    ):
        raise SystemExit("U15 full-cohort census or reserve proof mismatch")
    payload = {
        "revision": U15_MANIFEST_REVISION,
        "acquisition_id": U15_ACQUISITION_ID,
        "protocol": {
            "path": U15_PROTOCOL.relative_to(TARGET).as_posix(),
            "revision": protocol["revision"],
            "sha256": sha256_file(U15_PROTOCOL),
        },
        "rubric": {
            "revision": rubric["revision"],
            "sha256": _canonical_json_sha(rubric),
        },
        "cohort_manifest": {
            "path": cohort_path.relative_to(TARGET).as_posix(),
            "sha256": U15_COHORT_SHA256,
        },
        "raw_inputs": [
            {
                "meeting": meeting,
                "path": raw.relative_to(TARGET).as_posix(),
                "sha256": digest,
            }
            for meeting, raw, digest in sources
        ],
        "capsule": capsule,
        "implementation": {
            "path": Path(__file__).resolve().relative_to(TARGET).as_posix(),
            "sha256": sha256_file(Path(__file__)),
            "parser": f"ijson=={_load_ijson().__version__}",
            "backend": _load_ijson().backend,
        },
        "configuration": {
            "model": U15_MODEL,
            "max_tokens": U15_MAX_TOKENS,
            "source_language": "en",
            "target_language": "ko",
            "context": "",
            "scene_participant_count": None,
            "system_prompt_sha256": next(iter(prompt_hashes)),
        },
        "census": {
            "parents": len(parents),
            "nonempty_parents": sum(bool(row["accepted_text"]) for row in parents),
            "r2_children": len(child_ids),
            "requests": len(requests),
            "requests_by_meeting": dict(per_meeting),
            "original_r2_statuses": dict(original_statuses),
            "retained_r0_statuses": dict(r0_statuses),
            "source_only_not_called": original_statuses["source_only"],
        },
        "reserve_proof": {
            "exact_sum_usd": reserve_sum,
            "maximum_request_bytes": max_bytes,
            "conservative_all_max_usd": conservative,
            "additional_cap_usd": U15_RESERVE_CAP_USD,
        },
        "parents": parents,
        "requests": requests,
    }
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"prepared input already exists: {out}")
    _atomic_json(out, payload)
    print(
        json.dumps(
            {
                "status": "prepared",
                "path": str(out),
                "sha256": sha256_file(out),
                "requests": len(requests),
                "parents": len(parents),
                "exact_reserve_usd": reserve_sum,
                "conservative_reserve_usd": conservative,
                "credential_presence": __import__(
                    "experiments.psem_r2_policy.credentials", fromlist=["credential_presence"]
                ).credential_presence(),
                "capsule_modules": capsule["modules"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_prepared(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    prepared = _json_load(path)
    requests = prepared.get("requests")
    parents = prepared.get("parents")
    if (
        prepared.get("revision") != U15_MANIFEST_REVISION
        or prepared.get("acquisition_id") != U15_ACQUISITION_ID
        or (prepared.get("census") or {}).get("requests") != U15_MAX_REQUESTS
        or not isinstance(requests, list)
        or len(requests) != U15_MAX_REQUESTS
        or not isinstance(parents, list)
        or len(parents) != 2482
    ):
        raise SystemExit("prepared U15 input manifest is malformed")
    indexes = [row.get("ordered_index") for row in requests]
    original_ids = [row.get("original_request_id") for row in requests]
    child_ids = [row.get("original_child_id") for row in requests]
    new_ids = [row.get("new_utterance_id") for row in requests]
    if indexes != list(range(U15_MAX_REQUESTS)):
        raise SystemExit("prepared U15 request order is malformed")
    if any(
        len(set(values)) != U15_MAX_REQUESTS or any(not isinstance(value, str) for value in values)
        for values in (original_ids, child_ids, new_ids)
    ):
        raise SystemExit("prepared U15 input has duplicate or missing request linkage")
    prompt_hashes: set[str] = set()
    exact = 0.0
    for row in requests:
        if (
            row.get("source_language") != "en"
            or row.get("target_language") != "ko"
            or row.get("context") != ""
            or row.get("scene_participant_count") is not None
            or row["new_utterance_id"] == row["original_child_id"]
        ):
            raise SystemExit(f"prepared request configuration drift: {row.get('ordered_index')}")
        serialized, byte_count, reserve = _request_body_and_bound(row)
        prompt_hashes.add(hashlib.sha256(row["system_prompt"].encode("utf-8")).hexdigest())
        if (
            row.get("request_body_sha256")
            != hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            or int(row.get("bytes") or -1) != byte_count
            or abs(float(row.get("reserve_usd") or -1) - reserve) > 1e-15
        ):
            raise SystemExit(f"prepared request body/bound drift: {row.get('ordered_index')}")
        exact += reserve
    configured = prepared.get("configuration") or {}
    proof = prepared.get("reserve_proof") or {}
    if (
        configured
        != {
            "model": U15_MODEL,
            "max_tokens": U15_MAX_TOKENS,
            "source_language": "en",
            "target_language": "ko",
            "context": "",
            "scene_participant_count": None,
            "system_prompt_sha256": next(iter(prompt_hashes)) if len(prompt_hashes) == 1 else None,
        }
        or abs(float(proof.get("exact_sum_usd") or -1) - exact) > 1e-12
        or int(proof.get("maximum_request_bytes") or -1) != max(int(row["bytes"]) for row in requests)
        or float(proof.get("additional_cap_usd") or -1) != U15_RESERVE_CAP_USD
    ):
        raise SystemExit("prepared U15 configuration or reserve proof is malformed")
    return prepared, digest


def _preflight_execution(
    *,
    prepared: Mapping[str, Any],
    prepared_sha: str,
    capsule: Mapping[str, Any],
    ledger_path: Path,
    authority_path: Path,
    billing_path: Path,
    verification: bool,
    enforce_activation: bool,
) -> None:
    if (prepared.get("capsule") or {}).get("stable") != capsule.get("stable"):
        raise SystemExit("prepared stable runtime identity no longer matches actual capsule")
    if sha256_file(Path(__file__)) != (prepared.get("implementation") or {}).get("sha256"):
        raise SystemExit("U15 acquisition implementation changed after input preparation")
    if sha256_file(U15_PROTOCOL) != (prepared.get("protocol") or {}).get("sha256"):
        raise SystemExit("protocol changed after U15 input preparation")
    rubric = ((_json_load(U15_PROTOCOL).get("measurements") or {}).get("translation_rubric") or {})
    if _canonical_json_sha(rubric) != (prepared.get("rubric") or {}).get("sha256"):
        raise SystemExit("translation rubric changed after U15 input preparation")
    if not verification and ledger_path.resolve() != U15_LEDGER.resolve():
        raise SystemExit("paid U15 execution requires the canonical ledger")
    ledger_state = _json_load(ledger_path)
    if (
        float(ledger_state.get("cap_usd") or 0) != 5.25
        or ledger_state.get("phase_caps_usd")
        != {"dev": 3.0, "holdout": 2.25, "contingency": 0.0}
    ):
        raise SystemExit("canonical ledger caps do not match the U15 allocation")
    authority = _json_load(authority_path)
    contract = ((authority.get("subsequent_agreement") or {}).get("u15_translation_reacquisition") or {})
    billing = _json_load(billing_path)
    go = billing.get("u15_translation_reacquisition_go") or {}
    if (
        billing.get("budget_defensible") is not True
        or (billing.get("openrouter") or {}).get("defensible") is not True
    ):
        raise SystemExit("U15 paid execution is blocked by indefensible OpenRouter pricing")
    if (not verification or enforce_activation) and (
        contract.get("acquisition_id") != U15_ACQUISITION_ID
        or contract.get("prepared_input_manifest_sha256") != prepared_sha
        or billing.get("paid_ready") is not True
        or go
        != {
            "acquisition_id": U15_ACQUISITION_ID,
            "prepared_input_manifest_sha256": prepared_sha,
            "enabled": True,
        }
    ):
        raise SystemExit("Director U15 paid approval is absent or does not match prepared input")
    exact = float((prepared.get("reserve_proof") or {}).get("exact_sum_usd") or 0)
    snapshot = harness_budget.BudgetLedger(ledger_path).snapshot()
    if exact > U15_RESERVE_CAP_USD + 1e-12:
        raise SystemExit("U15 additional reservation cap would be exceeded")
    if snapshot.phase_spent["dev"] + snapshot.phase_reserved["dev"] + exact > 3.0 + 1e-12:
        raise SystemExit("DEV budget would be exceeded before the U15 run")
    if snapshot.spent_usd + snapshot.reserved_usd + exact > 5.25 + 1e-12:
        raise SystemExit("global budget would be exceeded before the U15 run")


def _append_journal(handle: Any, event: Mapping[str, Any]) -> None:
    handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _acquisition_claim_path(
    *, ledger_path: Path, verification: bool, acquisition_id: str
) -> Path:
    if verification:
        return ledger_path.with_name(f".{ledger_path.name}.{acquisition_id}.claim.json")
    return (
        TARGET
        / "experiments/psem_r2_policy/artifacts/dev-text-reacquisition"
        / f"{acquisition_id}.claim.json"
    )


def _acquire_acquisition_claim(
    *,
    path: Path,
    acquisition_id: str,
    prepared_sha: str,
    journal_path: Path,
    ledger_path: Path,
    capsule: Mapping[str, Any],
    verification: bool,
) -> dict[str, Any]:
    payload = {
        "revision": (
            "U17-TEXT-REACQUISITION-CONTINUATION-CLAIM-1"
            if acquisition_id == U17_CONTINUATION_ID
            else "U15-TEXT-REACQUISITION-CLAIM-1"
        ),
        "acquisition_id": acquisition_id,
        "prepared_input_manifest_sha256": prepared_sha,
        "journal_path": str(journal_path),
        "ledger_path": str(ledger_path),
        "runtime": dict(capsule),
        "verification_namespace": verification,
        "claimed_wall_utc": datetime.now(timezone.utc).isoformat(),
        "claimed_monotonic_s": time.monotonic(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise SystemExit(
            f"acquisition already claimed; automatic retry/resume is forbidden: {path}"
        ) from exc
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata": payload,
    }


async def _translation_acquire(args: argparse.Namespace) -> int:
    capsule = _require_capsule()
    continuation = args.mode == "translation-continue-acquire"
    prepared_path = Path(args.input).resolve()
    if continuation:
        prepared, prepared_sha, _root_prepared = _load_continuation(prepared_path)
    else:
        prepared, prepared_sha = _load_prepared(prepared_path)
    acquisition_id = U17_CONTINUATION_ID if continuation else U15_ACQUISITION_ID
    request_limit = len(prepared["requests"])
    batch_reserve_cap = (
        U15_RESERVE_CAP_USD - U17_PRIOR_RESERVED_USD
        if continuation
        else U15_RESERVE_CAP_USD
    )
    journal_path = Path(args.journal).resolve()
    if journal_path.exists():
        raise SystemExit(f"acquisition journal already exists; refusing any network: {journal_path}")
    verification = bool(args.verification_script)
    ledger_path = Path(args.ledger).resolve() if args.ledger else U15_LEDGER.resolve()
    authority_path = Path(args.authority).resolve() if args.authority else U15_AUTHORITY.resolve()
    billing_path = Path(args.billing).resolve() if args.billing else U15_BILLING.resolve()
    claim_path = _acquisition_claim_path(
        ledger_path=ledger_path,
        verification=verification,
        acquisition_id=acquisition_id,
    ).resolve()
    protected_paths = {
        prepared_path,
        ledger_path,
        authority_path,
        billing_path,
        claim_path,
    }
    if journal_path in protected_paths or len(protected_paths) != 5:
        raise SystemExit("acquisition input, gates, ledger, claim, and journal paths must be distinct")
    if verification and os.environ.get("PSEM_U15_ZERO_COST_VERIFY") != "1":
        raise SystemExit("verification transport requires PSEM_U15_ZERO_COST_VERIFY=1")
    if verification and (not args.ledger or ledger_path == U15_LEDGER.resolve()):
        raise SystemExit("verification transport requires an explicit noncanonical ledger")
    if args.verify_paid_gates and not verification:
        raise SystemExit("--verify-paid-gates requires a zero-cost verification transport")
    if continuation:
        _preflight_continuation(
            prepared=prepared,
            prepared_sha=prepared_sha,
            capsule=capsule,
            ledger_path=ledger_path,
            authority_path=authority_path,
            billing_path=billing_path,
            verification=verification,
            enforce_activation=bool(args.verify_paid_gates),
        )
    else:
        _preflight_execution(
            prepared=prepared,
            prepared_sha=prepared_sha,
            capsule=capsule,
            ledger_path=ledger_path,
            authority_path=authority_path,
            billing_path=billing_path,
            verification=verification,
            enforce_activation=bool(args.verify_paid_gates),
        )
    secrets = __import__(
        "experiments.psem_r2_policy.credentials", fromlist=["load_runtime_secrets"]
    ).load_runtime_secrets()
    if not verification and not secrets.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is absent")
    claim = _acquire_acquisition_claim(
        path=claim_path,
        acquisition_id=acquisition_id,
        prepared_sha=prepared_sha,
        journal_path=journal_path,
        ledger_path=ledger_path,
        capsule=capsule,
        verification=verification,
    )
    print(
        json.dumps(
            {
                "status": (
                    "READY_U17_TEXT_REACQUISITION_CONTINUATION"
                    if continuation
                    else "READY_U15_TEXT_REACQUISITION"
                ),
                "requests": request_limit,
                "reserve_usd": (prepared["reserve_proof"])["exact_sum_usd"],
                "paid": not verification,
                "openrouter_credential_present": bool(secrets.get("OPENROUTER_API_KEY")),
                "claim_path": str(claim_path),
            }
        ),
        flush=True,
    )
    from experiments.psem_r2_policy.live_runner import BudgetedOpenRouter, _safe_translation_error
    from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient, OpenRouterLLMProvider

    inner_client = None
    if verification:
        script = _json_load(Path(args.verification_script))
        scripted = iter(script.get("responses") or ())
        if script.get("transport") == "provider":
            class VerificationClient:
                async def translate(self, **_kwargs: Any) -> str:
                    try:
                        row = next(scripted)
                    except StopIteration as exc:
                        raise RuntimeError("verification script exhausted") from exc
                    if int(row.get("status", 200)) != 200:
                        raise RuntimeError("zero-cost verification failure")
                    return str(row.get("text", ""))

                async def close(self) -> None:
                    return None

            inner_client = VerificationClient()
        else:
            import httpx

            def handler(_request: Any) -> Any:
                try:
                    row = next(scripted)
                except StopIteration:
                    return httpx.Response(599, json={"error": "verification script exhausted"})
                status = int(row.get("status", 200))
                if status != 200:
                    return httpx.Response(status, json={"error": "zero-cost verification"})
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": row.get("text", "")},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )

            inner_client = HttpxOpenRouterClient(
                api_key="zero-cost-verification", model=U15_MODEL, max_tokens=U15_MAX_TOKENS
            )
            inner_client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenRouterLLMProvider(
        api_key=secrets.get("OPENROUTER_API_KEY") or "zero-cost-verification",
        model=U15_MODEL,
        max_tokens=U15_MAX_TOKENS,
        client=inner_client,
    )
    budgeted = BudgetedOpenRouter(
        provider,
        ledger=harness_budget.BudgetLedger(ledger_path),
        phase="dev",
        network=not verification,
        clock=time.monotonic,
        clock_scope="system_monotonic",
    )
    budgeted.arm = "u15"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    attempted = 0
    failed = False
    reserved = 0.0
    with journal_path.open("x", encoding="utf-8") as journal:
        _append_journal(
            journal,
            {
                "event": "header",
                "revision": U17_JOURNAL_REVISION if continuation else U15_JOURNAL_REVISION,
                "acquisition_id": acquisition_id,
                "prepared_input": {"path": str(prepared_path), "sha256": prepared_sha},
                "protocol": prepared["protocol"],
                "rubric": prepared["rubric"],
                "capsule": capsule,
                "authority": {"path": str(authority_path), "sha256": sha256_file(authority_path)},
                "billing": {"path": str(billing_path), "sha256": sha256_file(billing_path)},
                "ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
                "acquisition_claim": claim,
                "started_wall_utc": datetime.now(timezone.utc).isoformat(),
                "started_monotonic_s": time.monotonic(),
                "verification_transport": verification,
            },
        )
        for row in prepared["requests"]:
            reserve = float(row["reserve_usd"])
            if attempted >= request_limit or reserved + reserve > batch_reserve_cap + 1e-12:
                _append_journal(
                    journal,
                    {
                        "event": "stopped",
                        "reason": "batch_request_or_cumulative_acquisition_cap",
                        "attempted": attempted,
                        "remaining": request_limit - attempted,
                    },
                )
                failed = True
                break
            _append_journal(
                journal,
                {
                    "event": "started",
                    "ordered_index": row["ordered_index"],
                    "original_request_id": row["original_request_id"],
                    "original_child_id": row["original_child_id"],
                    "new_utterance_id": row["new_utterance_id"],
                    "parent_id": row["parent_id"],
                    "meeting": row["meeting"],
                    "request_body_sha256": row["request_body_sha256"],
                    "reserve_usd": reserve,
                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_s": time.monotonic(),
                },
            )
            before = len(budgeted.requests)
            error: dict[str, Any] | None = None
            caught_error: Exception | None = None
            response: str | None = None
            try:
                result = await budgeted.translate(
                    utterance_id=UUID(row["new_utterance_id"]),
                    text=row["text"],
                    system_prompt=row["system_prompt"],
                    source_language=row["source_language"],
                    target_language=row["target_language"],
                    context=row["context"],
                    scene_participant_count=row["scene_participant_count"],
                )
                response = result.translated_text
                status = "translated"
            except Exception as exc:
                caught_error = exc
                failed = True
            actual = budgeted.requests[-1] if len(budgeted.requests) > before else None
            dispatched = actual is not None and actual.get("outcome") is not None
            if failed:
                if caught_error is None:
                    raise RuntimeError("caught acquisition failure is missing")
                if dispatched and actual.get("outcome") == "translated":
                    error = _safe_translation_error(caught_error)
                    error["message"] = "post-provider finalization failed"
                    error["phase"] = "post_provider_finalization"
                    error["category"] = "execution_failure"
                    status = "failed"
                elif dispatched:
                    error = actual.get("error")
                    status = "failed"
                else:
                    error = _safe_translation_error(caught_error)
                    error["phase"] = "pre_dispatch_reservation"
                    if isinstance(caught_error, harness_budget.BudgetError):
                        error["category"] = "budget_refusal"
                        status = "budget_refused"
                    else:
                        error["category"] = "execution_failure"
                        status = "failed"
            if actual is not None and actual.get("id") == row["original_request_id"]:
                raise RuntimeError("new and original request IDs collided")
            if dispatched:
                attempted += 1
                reserved += reserve
            _append_journal(
                journal,
                {
                    "event": "completed",
                    "ordered_index": row["ordered_index"],
                    "original_request_id": row["original_request_id"],
                    "original_child_id": row["original_child_id"],
                    "new_utterance_id": row["new_utterance_id"],
                    "parent_id": row["parent_id"],
                    "meeting": row["meeting"],
                    "status": status,
                    "new_response": response,
                    "error": error,
                    "budgeted_request": actual,
                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_s": time.monotonic(),
                },
            )
            print(
                json.dumps(
                    {
                        "status": status,
                        "completed": attempted,
                        "remaining": request_limit - attempted,
                        "reserved_usd": reserved,
                    }
                ),
                flush=True,
            )
            if failed:
                break
        _append_journal(
            journal,
            {
                "event": "summary",
                "complete": attempted == request_limit and not failed,
                "attempted": attempted,
                "remaining_not_attempted": request_limit - attempted,
                "reserved_usd": reserved,
                "finished_wall_utc": datetime.now(timezone.utc).isoformat(),
                "finished_monotonic_s": time.monotonic(),
            },
        )
    await budgeted.close()
    if inner_client is not None:
        await inner_client.close()
    return 0 if attempted == request_limit and not failed else 1


def translation_acquire_mode(args: argparse.Namespace) -> int:
    return asyncio.run(_translation_acquire(args))


def _journal_state(
    path: Path,
    *,
    prepared: Mapping[str, Any],
    prepared_sha: str,
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
    dict[str, Any],
]:
    requests = list(prepared["requests"])
    acquisition_id = str(prepared["acquisition_id"])
    journal_revision = (
        U17_JOURNAL_REVISION
        if acquisition_id == U17_CONTINUATION_ID
        else U15_JOURNAL_REVISION
    )
    header: dict[str, Any] | None = None
    completed: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] | None = None
    active_index: int | None = None
    next_index = 0
    terminal = False
    request_ids: set[str] = set()
    dispatched_reserve = 0.0
    dispatched_count = 0
    anomalies: list[dict[str, Any]] = []
    quarantine_requires_summary = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid acquisition journal line {line_number}: {exc}") from exc
            kind = event.get("event")
            if quarantine_requires_summary and kind != "summary":
                raise SystemExit(
                    "post-provider finalization quarantine must be followed by summary"
                )
            if kind == "header":
                claim = event.get("acquisition_claim") or {}
                claim_meta = claim.get("metadata") or {}
                claim_path = Path(str(claim.get("path") or ""))
                if (
                    line_number != 1
                    or header is not None
                    or event.get("revision") != journal_revision
                    or event.get("acquisition_id") != acquisition_id
                    or (event.get("prepared_input") or {}).get("sha256") != prepared_sha
                    or (event.get("capsule") or {}).get("stable")
                    != (prepared.get("capsule") or {}).get("stable")
                    or claim_meta.get("acquisition_id") != acquisition_id
                    or claim_meta.get("prepared_input_manifest_sha256") != prepared_sha
                    or claim_meta.get("journal_path") != str(path.resolve())
                    or not claim_path.is_file()
                    or sha256_file(claim_path) != claim.get("sha256")
                ):
                    raise SystemExit("acquisition journal header/claim does not match prepared input")
                header = event
                continue
            if header is None or summary is not None:
                raise SystemExit(f"invalid acquisition journal lifecycle at line {line_number}")
            if kind == "started":
                if terminal or active_index is not None or next_index >= len(requests):
                    raise SystemExit(f"invalid started lifecycle at line {line_number}")
                row = requests[next_index]
                expected = {
                    "ordered_index": next_index,
                    "original_request_id": row["original_request_id"],
                    "original_child_id": row["original_child_id"],
                    "new_utterance_id": row["new_utterance_id"],
                    "parent_id": row["parent_id"],
                    "meeting": row["meeting"],
                    "request_body_sha256": row["request_body_sha256"],
                }
                if any(event.get(key) != value for key, value in expected.items()) or abs(
                    float(event.get("reserve_usd", -1)) - float(row["reserve_usd"])
                ) > 1e-15:
                    raise SystemExit(f"started request linkage mismatch at line {line_number}")
                if not isinstance(event.get("wall_utc"), str) or not isinstance(
                    event.get("monotonic_s"), (int, float)
                ):
                    raise SystemExit(f"started timestamps missing at line {line_number}")
                active_index = next_index
                next_index += 1
                continue
            if kind == "completed":
                if active_index is None or event.get("ordered_index") != active_index:
                    raise SystemExit(f"completed request lifecycle mismatch at line {line_number}")
                row = requests[active_index]
                expected = {
                    "original_request_id": row["original_request_id"],
                    "original_child_id": row["original_child_id"],
                    "new_utterance_id": row["new_utterance_id"],
                    "parent_id": row["parent_id"],
                    "meeting": row["meeting"],
                }
                if any(event.get(key) != value for key, value in expected.items()):
                    raise SystemExit(f"completed request linkage mismatch at line {line_number}")
                actual = event.get("budgeted_request")
                status = event.get("status")
                if not isinstance(actual, dict):
                    raise SystemExit(f"completed budget record missing at line {line_number}")
                actual_id = actual.get("id")
                if (
                    not isinstance(actual_id, str)
                    or actual_id == row["original_request_id"]
                    or actual_id in request_ids
                    or actual.get("arm") != "u15"
                    or actual.get("utterance_id") != row["new_utterance_id"]
                    or actual.get("text") != row["text"]
                    or actual.get("system_prompt") != row["system_prompt"]
                    or actual.get("source_language") != row["source_language"]
                    or actual.get("target_language") != row["target_language"]
                    or actual.get("context") != row["context"]
                    or actual.get("scene_participant_count") != row["scene_participant_count"]
                    or int(actual.get("bytes", -1)) != int(row["bytes"])
                    or abs(float(actual.get("usd", -1)) - float(row["reserve_usd"])) > 1e-15
                ):
                    raise SystemExit(f"completed budget lineage mismatch at line {line_number}")
                request_ids.add(actual_id)
                if status == "translated":
                    consistent = (
                        actual.get("outcome") == "translated"
                        and isinstance(event.get("new_response"), str)
                        and actual.get("translated_text") == event.get("new_response")
                        and event.get("error") is None
                        and actual.get("error") is None
                    )
                elif status == "failed":
                    normal_failure = (
                        actual.get("outcome") == "failed"
                        and event.get("new_response") is None
                        and isinstance(event.get("error"), dict)
                        and actual.get("error") == event.get("error")
                        and actual.get("translated_text") is None
                    )
                    reported_error = event.get("error")
                    post_provider_failure = (
                        actual.get("outcome") == "translated"
                        and isinstance(actual.get("translated_text"), str)
                        and actual.get("error") is None
                        and event.get("new_response") is None
                        and (
                            reported_error is None
                            or (
                                isinstance(reported_error, dict)
                                and reported_error.get("phase")
                                == "post_provider_finalization"
                                and reported_error.get("category") == "execution_failure"
                                and isinstance(reported_error.get("type"), str)
                                and bool(reported_error.get("type"))
                                and reported_error.get("message")
                                == "post-provider finalization failed"
                            )
                        )
                    )
                    pre_dispatch_failure = (
                        actual.get("outcome") is None
                        and actual.get("translated_text") is None
                        and actual.get("error") is None
                        and event.get("new_response") is None
                        and isinstance(reported_error, dict)
                        and reported_error.get("phase") == "pre_dispatch_reservation"
                        and reported_error.get("category") == "execution_failure"
                        and isinstance(reported_error.get("type"), str)
                        and bool(reported_error.get("type"))
                    )
                    consistent = normal_failure or post_provider_failure or pre_dispatch_failure
                    if post_provider_failure or pre_dispatch_failure:
                        anomaly_type = (
                            "post_provider_finalization_failure"
                            if post_provider_failure
                            else "pre_dispatch_reservation_failure"
                        )
                        anomalies.append(
                            {
                                "type": anomaly_type,
                                "ordered_index": active_index,
                                "new_request_id": actual_id,
                                "error_missing": reported_error is None,
                                "reported_error": reported_error,
                                "source_journal_line": line_number,
                            }
                        )
                        quarantine_requires_summary = True
                    terminal = True
                elif status == "budget_refused":
                    reported_error = event.get("error")
                    legacy_refusal = (
                        isinstance(reported_error, dict)
                        and reported_error.get("type") == "BudgetError"
                        and reported_error.get("phase") is None
                    )
                    explicit_refusal = (
                        isinstance(reported_error, dict)
                        and reported_error.get("phase") == "pre_dispatch_reservation"
                        and reported_error.get("category") == "budget_refusal"
                        and isinstance(reported_error.get("type"), str)
                        and bool(reported_error.get("type"))
                    )
                    consistent = (
                        actual.get("outcome") is None
                        and actual.get("translated_text") is None
                        and actual.get("error") is None
                        and event.get("new_response") is None
                        and (legacy_refusal or explicit_refusal)
                    )
                    terminal = True
                else:
                    consistent = False
                if not consistent:
                    raise SystemExit(f"completed outcome mismatch at line {line_number}")
                if not isinstance(event.get("wall_utc"), str) or not isinstance(
                    event.get("monotonic_s"), (int, float)
                ):
                    raise SystemExit(f"completed timestamps missing at line {line_number}")
                completed[active_index] = event
                if actual.get("outcome") in {"translated", "failed"}:
                    dispatched_count += 1
                    dispatched_reserve += float(row["reserve_usd"])
                active_index = None
                continue
            if kind == "stopped":
                if active_index is not None or terminal:
                    raise SystemExit(f"invalid stop lifecycle at line {line_number}")
                terminal = True
                continue
            if kind == "summary":
                if active_index is not None:
                    raise SystemExit("summary cannot classify an in-flight request")
                actually_complete = (
                    len(completed) == len(requests)
                    and all(row.get("status") == "translated" for row in completed.values())
                )
                if (
                    event.get("complete") is not actually_complete
                    or int(event.get("attempted", -1)) != dispatched_count
                    or int(event.get("remaining_not_attempted", -1))
                    != len(requests) - dispatched_count
                    or abs(float(event.get("reserved_usd", -1)) - dispatched_reserve) > 1e-12
                ):
                    raise SystemExit("acquisition journal summary is inconsistent")
                quarantine_requires_summary = False
                summary = event
                terminal = True
                continue
            raise SystemExit(f"unknown acquisition journal event at line {line_number}: {kind!r}")
    if header is None:
        raise SystemExit("acquisition journal header is missing")
    if quarantine_requires_summary:
        raise SystemExit("post-provider finalization quarantine summary is missing")
    return (
        header,
        completed,
        summary,
        anomalies,
        {
            "started_ordered_indexes": [
                int(requests[position]["ordered_index"]) for position in range(next_index)
            ],
            "dispatched_count": dispatched_count,
            "dispatched_reserve_usd": dispatched_reserve,
        },
    )


def _continuation_root_and_prior(
    continuation: Mapping[str, Any],
) -> tuple[Path, Path]:
    root_path = (TARGET / str((continuation.get("root_prepared_input") or {}).get("path") or "")).resolve()
    prior_journal_path = (
        TARGET / str((continuation.get("prior_evidence") or {}).get("journal_path") or "")
    ).resolve()
    if root_path == prior_journal_path:
        raise SystemExit("continuation root input and prior journal paths collide")
    return root_path, prior_journal_path


def _validated_prior_execution(
    root_path: Path, prior_journal_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root_prepared, root_sha = _load_prepared(root_path)
    if root_sha != U17_PRIOR_PREPARED_SHA256:
        raise SystemExit("continuation prior prepared input identity mismatch")
    if sha256_file(prior_journal_path) != U17_PRIOR_JOURNAL_SHA256:
        raise SystemExit("continuation prior journal identity mismatch")
    header, completed, summary, anomalies, lifecycle = _journal_state(
        prior_journal_path,
        prepared=root_prepared,
        prepared_sha=root_sha,
    )
    claim = header["acquisition_claim"]
    if claim.get("sha256") != U17_PRIOR_CLAIM_SHA256:
        raise SystemExit("continuation prior claim identity mismatch")
    started = lifecycle["started_ordered_indexes"]
    if (
        started != list(range(U17_EXCLUDED_REQUESTS))
        or len(completed) != U17_EXCLUDED_REQUESTS
        or summary is None
        or summary.get("complete") is not False
        or int(summary.get("attempted", -1)) != U17_EXCLUDED_REQUESTS
        or int(summary.get("remaining_not_attempted", -1)) != U17_MAX_REQUESTS
        or abs(float(summary.get("reserved_usd", -1)) - U17_PRIOR_RESERVED_USD) > 1e-12
        or len(anomalies) != 1
        or anomalies[0].get("type") != "post_provider_finalization_failure"
    ):
        raise SystemExit("continuation prior execution lifecycle mismatch")
    return root_prepared, header, completed, summary, anomalies, lifecycle


def translation_continuation_prepare_mode(args: argparse.Namespace) -> int:
    capsule = _require_capsule()
    root_path = Path(args.input).resolve()
    prior_journal_path = Path(args.journal).resolve()
    root, header, _completed, summary, anomalies, lifecycle = _validated_prior_execution(
        root_path, prior_journal_path
    )
    protocol = _json_load(U15_PROTOCOL)
    rubric = ((protocol.get("measurements") or {}).get("translation_rubric") or {})
    authority = _json_load(U15_AUTHORITY)
    contract = (
        (authority.get("subsequent_agreement") or {}).get("u17_translation_continuation")
        or {}
    )
    if (
        protocol.get("revision") != "R2-POLICY-DIRECTOR-14"
        or rubric.get("revision") != U15_INSPECTION_REVISION
        or contract.get("acquisition_id") != U17_CONTINUATION_ID
        or contract.get("parent_acquisition_id") != U15_ACQUISITION_ID
        or contract.get("prior_prepared_input_sha256") != U17_PRIOR_PREPARED_SHA256
        or contract.get("prior_journal_sha256") != U17_PRIOR_JOURNAL_SHA256
        or contract.get("prior_claim_sha256") != U17_PRIOR_CLAIM_SHA256
        or int(contract.get("excluded_prior_requests", -1)) != U17_EXCLUDED_REQUESTS
        or int(contract.get("maximum_new_requests", -1)) != U17_MAX_REQUESTS
        or int(contract.get("maximum_combined_requests", -1)) != U15_MAX_REQUESTS
    ):
        raise SystemExit("frozen U17 continuation authority/protocol mismatch")
    started = set(lifecycle["started_ordered_indexes"])
    requests: list[dict[str, Any]] = []
    for root_row in root["requests"]:
        root_index = int(root_row["ordered_index"])
        if root_index in started:
            continue
        row = dict(root_row)
        row["root_ordered_index"] = root_index
        row["ordered_index"] = len(requests)
        requests.append(row)
    exact = sum(float(row["reserve_usd"]) for row in requests)
    cumulative = U17_PRIOR_RESERVED_USD + exact
    if (
        len(requests) != U17_MAX_REQUESTS
        or abs(exact - U17_REMAINING_RESERVED_USD) > 1e-12
        or abs(cumulative - float((root["reserve_proof"])["exact_sum_usd"])) > 1e-12
        or cumulative > U15_RESERVE_CAP_USD + 1e-12
    ):
        raise SystemExit("U17 continuation request census or cumulative reserve mismatch")
    payload = {
        "revision": U17_MANIFEST_REVISION,
        "acquisition_id": U17_CONTINUATION_ID,
        "parent_acquisition_id": U15_ACQUISITION_ID,
        "root_prepared_input": {
            "path": root_path.relative_to(TARGET).as_posix(),
            "sha256": U17_PRIOR_PREPARED_SHA256,
        },
        "prior_evidence": {
            "journal_path": prior_journal_path.relative_to(TARGET).as_posix(),
            "journal_sha256": U17_PRIOR_JOURNAL_SHA256,
            "claim_path": Path(str(header["acquisition_claim"]["path"]))
            .resolve()
            .relative_to(TARGET)
            .as_posix(),
            "claim_sha256": U17_PRIOR_CLAIM_SHA256,
            "execution_implementation": root["implementation"],
            "started_requests": U17_EXCLUDED_REQUESTS,
            "summary": summary,
            "anomalies": anomalies,
        },
        "protocol": {
            "path": U15_PROTOCOL.relative_to(TARGET).as_posix(),
            "revision": protocol["revision"],
            "sha256": sha256_file(U15_PROTOCOL),
        },
        "rubric": {
            "revision": rubric["revision"],
            "sha256": _canonical_json_sha(rubric),
        },
        "capsule": capsule,
        "implementation": {
            "path": Path(__file__).resolve().relative_to(TARGET).as_posix(),
            "sha256": sha256_file(Path(__file__)),
            "parser": f"ijson=={_load_ijson().__version__}",
            "backend": _load_ijson().backend,
        },
        "configuration": root["configuration"],
        "census": {
            "root_parents": len(root["parents"]),
            "root_requests": len(root["requests"]),
            "excluded_started_requests": U17_EXCLUDED_REQUESTS,
            "requests": len(requests),
        },
        "reserve_proof": {
            "exact_sum_usd": exact,
            "prior_reserved_usd": U17_PRIOR_RESERVED_USD,
            "cumulative_exact_sum_usd": cumulative,
            "cumulative_cap_usd": U15_RESERVE_CAP_USD,
            "remaining_cap_usd": U15_RESERVE_CAP_USD - U17_PRIOR_RESERVED_USD,
            "maximum_request_bytes": max(int(row["bytes"]) for row in requests),
        },
        "requests": requests,
    }
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"continuation prepared input already exists: {out}")
    protected = {
        root_path,
        prior_journal_path,
        Path(str(header["acquisition_claim"]["path"])).resolve(),
    }
    if out in protected:
        raise SystemExit("continuation prepared output collides with immutable prior evidence")
    _atomic_json(out, payload)
    print(
        json.dumps(
            {
                "status": "continuation_prepared",
                "path": str(out),
                "sha256": sha256_file(out),
                "requests": len(requests),
                "excluded_prior_requests": U17_EXCLUDED_REQUESTS,
                "exact_reserve_usd": exact,
                "cumulative_reserve_usd": cumulative,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_continuation(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    digest = sha256_file(path)
    continuation = _json_load(path)
    root_path, prior_journal_path = _continuation_root_and_prior(continuation)
    root, header, _completed, summary, anomalies, lifecycle = _validated_prior_execution(
        root_path, prior_journal_path
    )
    requests = continuation.get("requests")
    if (
        continuation.get("revision") != U17_MANIFEST_REVISION
        or continuation.get("acquisition_id") != U17_CONTINUATION_ID
        or continuation.get("parent_acquisition_id") != U15_ACQUISITION_ID
        or (continuation.get("root_prepared_input") or {}).get("sha256")
        != U17_PRIOR_PREPARED_SHA256
        or (continuation.get("prior_evidence") or {}).get("journal_sha256")
        != U17_PRIOR_JOURNAL_SHA256
        or (continuation.get("prior_evidence") or {}).get("claim_sha256")
        != U17_PRIOR_CLAIM_SHA256
        or not isinstance(requests, list)
        or len(requests) != U17_MAX_REQUESTS
    ):
        raise SystemExit("prepared U17 continuation manifest is malformed")
    started = set(lifecycle["started_ordered_indexes"])
    expected_root_rows = [
        row for row in root["requests"] if int(row["ordered_index"]) not in started
    ]
    for continuation_index, (row, root_row) in enumerate(
        zip(requests, expected_root_rows, strict=True)
    ):
        expected = dict(root_row)
        expected["root_ordered_index"] = int(root_row["ordered_index"])
        expected["ordered_index"] = continuation_index
        if row != expected:
            raise SystemExit(
                f"continuation request differs from immutable root request: {continuation_index}"
            )
    proof = continuation.get("reserve_proof") or {}
    exact = sum(float(row["reserve_usd"]) for row in requests)
    if (
        abs(exact - U17_REMAINING_RESERVED_USD) > 1e-12
        or abs(float(proof.get("exact_sum_usd", -1)) - exact) > 1e-12
        or abs(float(proof.get("prior_reserved_usd", -1)) - U17_PRIOR_RESERVED_USD)
        > 1e-12
        or abs(
            float(proof.get("cumulative_exact_sum_usd", -1))
            - float((root["reserve_proof"])["exact_sum_usd"])
        )
        > 1e-12
        or float(proof.get("cumulative_cap_usd", -1)) != U15_RESERVE_CAP_USD
        or continuation.get("configuration") != root["configuration"]
        or (continuation.get("capsule") or {}).get("stable")
        != (root.get("capsule") or {}).get("stable")
        or (continuation.get("prior_evidence") or {}).get("summary") != summary
        or (continuation.get("prior_evidence") or {}).get("anomalies") != anomalies
        or (continuation.get("prior_evidence") or {}).get("execution_implementation")
        != root["implementation"]
        or header["acquisition_claim"].get("sha256") != U17_PRIOR_CLAIM_SHA256
    ):
        raise SystemExit("prepared U17 continuation proof is malformed")
    return continuation, digest, root


def _preflight_continuation(
    *,
    prepared: Mapping[str, Any],
    prepared_sha: str,
    capsule: Mapping[str, Any],
    ledger_path: Path,
    authority_path: Path,
    billing_path: Path,
    verification: bool,
    enforce_activation: bool,
) -> None:
    if (prepared.get("capsule") or {}).get("stable") != capsule.get("stable"):
        raise SystemExit("continuation stable runtime identity no longer matches actual capsule")
    if sha256_file(Path(__file__)) != (prepared.get("implementation") or {}).get("sha256"):
        raise SystemExit("U17 continuation implementation changed after input preparation")
    if sha256_file(U15_PROTOCOL) != (prepared.get("protocol") or {}).get("sha256"):
        raise SystemExit("protocol changed after U17 continuation preparation")
    rubric = ((_json_load(U15_PROTOCOL).get("measurements") or {}).get("translation_rubric") or {})
    if _canonical_json_sha(rubric) != (prepared.get("rubric") or {}).get("sha256"):
        raise SystemExit("translation rubric changed after U17 continuation preparation")
    if not verification and ledger_path.resolve() != U15_LEDGER.resolve():
        raise SystemExit("paid U17 continuation requires the canonical ledger")
    ledger_state = _json_load(ledger_path)
    if (
        float(ledger_state.get("cap_usd") or 0) != 5.25
        or ledger_state.get("phase_caps_usd")
        != {"dev": 3.0, "holdout": 2.25, "contingency": 0.0}
    ):
        raise SystemExit("canonical ledger caps do not match the U17 allocation")
    authority = _json_load(authority_path)
    contract = (
        (authority.get("subsequent_agreement") or {}).get("u17_translation_continuation")
        or {}
    )
    billing = _json_load(billing_path)
    go = billing.get("u17_translation_continuation_go") or {}
    if (
        billing.get("budget_defensible") is not True
        or (billing.get("openrouter") or {}).get("defensible") is not True
    ):
        raise SystemExit("U17 continuation is blocked by indefensible OpenRouter pricing")
    if (
        contract.get("acquisition_id") != U17_CONTINUATION_ID
        or contract.get("parent_acquisition_id") != U15_ACQUISITION_ID
        or contract.get("prior_prepared_input_sha256") != U17_PRIOR_PREPARED_SHA256
        or contract.get("prior_journal_sha256") != U17_PRIOR_JOURNAL_SHA256
        or contract.get("prior_claim_sha256") != U17_PRIOR_CLAIM_SHA256
        or int(contract.get("excluded_prior_requests", -1)) != U17_EXCLUDED_REQUESTS
        or int(contract.get("maximum_new_requests", -1)) != U17_MAX_REQUESTS
    ):
        raise SystemExit("Director U17 continuation authority does not match prior execution")
    if (not verification or enforce_activation) and (
        billing.get("paid_ready") is not True
        or go
        != {
            "acquisition_id": U17_CONTINUATION_ID,
            "prepared_input_manifest_sha256": prepared_sha,
            "enabled": True,
        }
    ):
        raise SystemExit("Director U17 continuation paid approval is absent or mismatched")
    proof = prepared.get("reserve_proof") or {}
    exact = float(proof.get("exact_sum_usd") or 0)
    prior = float(proof.get("prior_reserved_usd") or 0)
    if (
        abs(prior - U17_PRIOR_RESERVED_USD) > 1e-12
        or prior + exact > U15_RESERVE_CAP_USD + 1e-12
    ):
        raise SystemExit("U17 cumulative acquisition cap would be exceeded")
    snapshot = harness_budget.BudgetLedger(ledger_path).snapshot()
    if snapshot.phase_spent["dev"] + snapshot.phase_reserved["dev"] + exact > 3.0 + 1e-12:
        raise SystemExit("DEV budget would be exceeded before the U17 continuation")
    if snapshot.spent_usd + snapshot.reserved_usd + exact > 5.25 + 1e-12:
        raise SystemExit("global budget would be exceeded before the U17 continuation")


def translation_inspect_mode(args: argparse.Namespace) -> int:
    capsule = _require_capsule()
    combined = args.mode == "translation-combined-inspect"
    prepared_path = Path(args.input).resolve()
    prepared, prepared_sha = _load_prepared(prepared_path)
    if (prepared.get("capsule") or {}).get("stable") != capsule.get("stable"):
        raise SystemExit("inspection stable runtime does not match prepared input")
    journal_path = Path(args.journal).resolve()
    header, completed, summary, anomalies, root_lifecycle = _journal_state(
        journal_path, prepared=prepared, prepared_sha=prepared_sha
    )
    execution_batches = [
        {
            "acquisition_id": U15_ACQUISITION_ID,
            "protocol": prepared["protocol"],
            "rubric": prepared["rubric"],
            "prepared_input_sha256": prepared_sha,
            "implementation": prepared["implementation"],
            "journal_sha256": sha256_file(journal_path),
            "acquisition_claim": header["acquisition_claim"],
            "capsule": header["capsule"],
        }
    ]
    journal_evidence: dict[str, Any] = {
        U15_ACQUISITION_ID: {
            "journal_sha256": sha256_file(journal_path),
            "summary": summary,
        }
    }
    protected = {
        prepared_path,
        journal_path,
        Path(str((header["acquisition_claim"])["path"])).resolve(),
    }
    if combined:
        continuation_path = Path(args.continuation_input).resolve()
        continuation_journal_path = Path(args.continuation_journal).resolve()
        continuation, continuation_sha, continuation_root = _load_continuation(
            continuation_path
        )
        if continuation_root != prepared:
            raise SystemExit("combined inspection root prepared input mismatch")
        (
            continuation_header,
            continuation_completed,
            continuation_summary,
            continuation_anomalies,
            _continuation_lifecycle,
        ) = _journal_state(
            continuation_journal_path,
            prepared=continuation,
            prepared_sha=continuation_sha,
        )
        prior_started_ids = {
            prepared["requests"][index]["original_request_id"]
            for index in root_lifecycle["started_ordered_indexes"]
        }
        continuation_ids = {
            row["original_request_id"] for row in continuation["requests"]
        }
        if (
            len(continuation_ids) != U17_MAX_REQUESTS
            or prior_started_ids & continuation_ids
        ):
            raise SystemExit("combined inspection request overlap detected")
        for continuation_index, event in continuation_completed.items():
            root_index = int(
                continuation["requests"][continuation_index]["root_ordered_index"]
            )
            if root_index in completed:
                raise SystemExit("combined inspection duplicate root request outcome")
            completed[root_index] = event
        anomalies = [
            {**row, "acquisition_id": U15_ACQUISITION_ID} for row in anomalies
        ] + [
            {**row, "acquisition_id": U17_CONTINUATION_ID}
            for row in continuation_anomalies
        ]
        summary = {
            U15_ACQUISITION_ID: summary,
            U17_CONTINUATION_ID: continuation_summary,
        }
        execution_batches.append(
            {
                "acquisition_id": U17_CONTINUATION_ID,
                "protocol": continuation["protocol"],
                "rubric": continuation["rubric"],
                "prepared_input_sha256": continuation_sha,
                "implementation": continuation["implementation"],
                "journal_sha256": sha256_file(continuation_journal_path),
                "acquisition_claim": continuation_header["acquisition_claim"],
                "capsule": continuation_header["capsule"],
            }
        )
        journal_evidence[U17_CONTINUATION_ID] = {
            "journal_sha256": sha256_file(continuation_journal_path),
            "summary": continuation_summary,
        }
        protected.update(
            {
                continuation_path,
                continuation_journal_path,
                Path(
                    str((continuation_header["acquisition_claim"])["path"])
                ).resolve(),
            }
        )
    analysis_protocol = _json_load(U15_PROTOCOL)
    analysis_rubric = (
        (analysis_protocol.get("measurements") or {}).get("translation_rubric") or {}
    )
    if (
        analysis_protocol.get("revision") != "R2-POLICY-DIRECTOR-14"
        or analysis_rubric.get("revision") != U15_INSPECTION_REVISION
    ):
        raise SystemExit("current analysis protocol/rubric identity mismatch")
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for request in prepared["requests"]:
        by_parent.setdefault(request["parent_id"], []).append(request)
    rng = secrets.SystemRandom()
    inspection: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for ordinal, parent in enumerate(prepared["parents"]):
        rows = by_parent.get(parent["parent_id"], [])
        new_events = [completed.get(int(row["ordered_index"])) for row in rows]
        old_ok = bool(parent["retained_r0"]) and all(
            row["status"] == "translated" and isinstance(row["response"], str)
            for row in parent["retained_r0"]
        )
        new_ok = bool(rows) and all(
            event is not None
            and event.get("status") == "translated"
            and isinstance(event.get("new_response"), str)
            for event in new_events
        )
        source_only = any(
            row["original_status"] == "source_only" for row in parent["r2_children"]
        )
        reason = None
        if not parent["accepted_text"]:
            reason = "empty_source_parent"
        elif source_only:
            reason = "source_only_child_not_called"
        elif not old_ok:
            reason = "retained_r0_response_unavailable"
        elif not new_ok:
            if any(event and event.get("status") == "failed" for event in new_events):
                reason = "new_response_failed"
            elif any(event and event.get("status") == "budget_refused" for event in new_events):
                reason = "new_response_budget_refused"
            else:
                reason = "new_response_unattempted_or_inflight"
        r0_joined = "".join(
            row["response"] for row in parent["retained_r0"] if isinstance(row["response"], str)
        )
        u15_joined = "".join(
            event["new_response"]
            for event in new_events
            if event is not None and isinstance(event.get("new_response"), str)
        )
        swapped = bool(rng.getrandbits(1))
        candidates = [u15_joined, r0_joined] if swapped else [r0_joined, u15_joined]
        inspection_id = f"U15-{ordinal:04d}"
        inspection.append(
            {
                "inspection_id": inspection_id,
                "source_text": parent["accepted_text"],
                "candidate_A": candidates[0] if reason is None else None,
                "candidate_B": candidates[1] if reason is None else None,
                "complete_pair_available": reason is None,
                "unavailable_reason": reason,
                "ratings": None,
            }
        )
        key.append(
            {
                "inspection_id": inspection_id,
                "parent_id": parent["parent_id"],
                "candidate_A": "u15_r2" if swapped else "retained_r0",
                "candidate_B": "retained_r0" if swapped else "u15_r2",
                "acquisition_ids": (
                    [U15_ACQUISITION_ID, U17_CONTINUATION_ID]
                    if combined
                    else [U15_ACQUISITION_ID]
                ),
            }
        )
    reader_path = Path(__file__).resolve()
    analysis_identity = {
        "analysis_protocol": {
            "path": U15_PROTOCOL.relative_to(TARGET).as_posix(),
            "revision": analysis_protocol["revision"],
            "sha256": sha256_file(U15_PROTOCOL),
        },
        "analysis_rubric": {
            "revision": analysis_rubric["revision"],
            "sha256": _canonical_json_sha(analysis_rubric),
        },
        "revision": U15_ANALYSIS_REVISION,
        "reader": {
            "path": reader_path.relative_to(TARGET).as_posix(),
            "sha256": sha256_file(reader_path),
        },
        "execution_batches": execution_batches,
    }
    view = {
        "revision": U15_INSPECTION_REVISION,
        "blinding": "fresh_private_randomness; realized mapping retained only in separate arm key",
        "input_manifests": {
            row["acquisition_id"]: row["prepared_input_sha256"]
            for row in execution_batches
        },
        "journal_evidence": journal_evidence,
        "journal_summary": summary,
        "analysis_identity": analysis_identity,
        "anomalies": anomalies,
        "ratings_created": False,
        "census": {
            "parents": len(inspection),
            "complete_pairs": sum(row["complete_pair_available"] for row in inspection),
            "unavailable_reasons": dict(
                Counter(
                    row["unavailable_reason"]
                    for row in inspection
                    if row["unavailable_reason"] is not None
                )
            ),
        },
        "records": inspection,
    }
    key_payload = {
        "revision": f"{U15_INSPECTION_REVISION}-ARM-KEY",
        "input_manifests": {
            row["acquisition_id"]: row["prepared_input_sha256"]
            for row in execution_batches
        },
        "mapping": key,
        "analysis_identity": analysis_identity,
    }
    out = Path(args.out).resolve()
    key_out = Path(args.key_out).resolve()
    protected = set(protected)
    if out == key_out or out in protected or key_out in protected:
        raise SystemExit("inspection view, arm key, input, journal, and claim paths must differ")
    if out.exists() or key_out.exists():
        raise SystemExit("inspection output or arm key already exists")
    _atomic_json(out, view)
    _atomic_json(key_out, key_payload)
    print(
        json.dumps(
            {
                "status": "inspection_prepared",
                "view_sha256": sha256_file(out),
                "key_sha256": sha256_file(key_out),
                **view["census"],
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    case_parser = sub.add_parser("case")
    case_parser.add_argument("--meeting", required=True)
    case_parser.add_argument("--stdout", required=True)
    case_parser.add_argument("--case-dir", required=True)
    case_parser.add_argument("--launch-meta")
    agg_parser = sub.add_parser("aggregate")
    aggregate_inputs = agg_parser.add_mutually_exclusive_group(required=True)
    aggregate_inputs.add_argument("--dirs")
    aggregate_inputs.add_argument("--manifest")
    agg_parser.add_argument("--out", required=True)
    prepare_parser = sub.add_parser("translation-prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--capsule-root", required=True)
    prepare_parser.add_argument("--out", required=True)
    continuation_prepare_parser = sub.add_parser("translation-continuation-prepare")
    continuation_prepare_parser.add_argument("--input", required=True)
    continuation_prepare_parser.add_argument("--capsule-root", required=True)
    continuation_prepare_parser.add_argument("--journal", required=True)
    continuation_prepare_parser.add_argument("--out", required=True)
    acquire_parser = sub.add_parser("translation-acquire")
    acquire_parser.add_argument("--input", required=True)
    acquire_parser.add_argument("--capsule-root", required=True)
    acquire_parser.add_argument("--journal", required=True)
    acquire_parser.add_argument("--ledger")
    acquire_parser.add_argument("--authority")
    acquire_parser.add_argument("--billing")
    acquire_parser.add_argument("--verification-script")
    acquire_parser.add_argument("--verify-paid-gates", action="store_true")
    continuation_acquire_parser = sub.add_parser("translation-continue-acquire")
    continuation_acquire_parser.add_argument("--input", required=True)
    continuation_acquire_parser.add_argument("--capsule-root", required=True)
    continuation_acquire_parser.add_argument("--journal", required=True)
    continuation_acquire_parser.add_argument("--ledger")
    continuation_acquire_parser.add_argument("--authority")
    continuation_acquire_parser.add_argument("--billing")
    continuation_acquire_parser.add_argument("--verification-script")
    continuation_acquire_parser.add_argument("--verify-paid-gates", action="store_true")
    inspect_parser = sub.add_parser("translation-inspect")
    inspect_parser.add_argument("--input", required=True)
    inspect_parser.add_argument("--capsule-root", required=True)
    inspect_parser.add_argument("--journal", required=True)
    inspect_parser.add_argument("--out", required=True)
    inspect_parser.add_argument("--key-out", required=True)
    combined_inspect_parser = sub.add_parser("translation-combined-inspect")
    combined_inspect_parser.add_argument("--input", required=True)
    combined_inspect_parser.add_argument("--capsule-root", required=True)
    combined_inspect_parser.add_argument("--journal", required=True)
    combined_inspect_parser.add_argument("--continuation-input", required=True)
    combined_inspect_parser.add_argument("--continuation-journal", required=True)
    combined_inspect_parser.add_argument("--out", required=True)
    combined_inspect_parser.add_argument("--key-out", required=True)
    args = parser.parse_args()
    if args.mode == "case":
        return case_mode(args)
    if args.mode == "aggregate":
        return aggregate_mode(args)
    if args.mode == "translation-prepare":
        return translation_prepare_mode(args)
    if args.mode == "translation-continuation-prepare":
        return translation_continuation_prepare_mode(args)
    if args.mode in {"translation-acquire", "translation-continue-acquire"}:
        return translation_acquire_mode(args)
    return translation_inspect_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
