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
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

TARGET = Path(
    r"C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls"
)
sys.path.insert(0, str(TARGET / "src"))
sys.path.insert(0, str(TARGET))

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
    "text",
    "cluster_id",
    "sequential_target",
    "status",
    "degraded",
    "clean_completion",
    "text_authority",
    "failure_reason",
    "outcome",
    "seal_reason",
    "conserved",
    "incomplete",
    "outage",
    "accounted",
    "provenance_valid",
}
PARENT_VALUE_FIELDS = {"span", "content_ranges", "marks"}
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
            arm: {**row, "statuses": dict(row["statuses"])}
            for arm, row in evidence["arms"].items()
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
        raise SystemExit(
            f"canonical parent {index} missing required fields {missing}: {path}"
        )
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
                    case_present.add("parents")
                    continue
                if prefix == "parents.item" and event == "start_map":
                    if current is not None:
                        raise SystemExit(f"nested canonical parent object: {path}")
                    current = {}
                    parent_present = set()
                    parent_index += 1
                    continue
                if prefix == "parents.item" and event == "end_map":
                    if current is None:
                        raise SystemExit(f"canonical parent ended without start: {path}")
                    _validate_projected_parent(current, parent_present, path, parent_index)
                    case["parents"].append(current)
                    translation_evidence["parents"] += 1
                    translation_evidence["empty_or_no_asr_parents"] += (
                        not bool(str(current.get("text") or ""))
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
                    elif relative in {"r0.contamination", "r2.contamination"} and event != "map_key":
                        outer, inner = relative.split(".", 1)
                        _assign_nested(
                            current, outer, inner, _consume_value(event, value, events, ijson)
                        )
                        parent_present.add(outer)
                    elif relative == "guard.assessed" and event in {"boolean", "null"}:
                        _assign_nested(current, "guard", "assessed", value)
                        parent_present.add("guard")
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
    outcomes = Counter(str((row.get("receipt") or {}).get("outcome") or row.get("outcome") or "") for row in parents)
    terminal = Counter(
        str((row.get("receipt") or {}).get("terminal_outcome") or row.get("terminal_outcome") or "")
        for row in parents
    )
    authority = Counter(str((row.get("receipt") or {}).get("text_authority") or row.get("text_authority") or "") for row in parents)
    failure = Counter(str(row.get("failure_reason") or "") for row in parents if row.get("failure_reason"))
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
            "c5_deadline_violations": seal.get("c5_deadline_violations"),
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
    assert harness_budget.LEDGER_PATH == CANONICAL_ARTIFACTS / "budget_ledger.json", harness_budget.LEDGER_PATH
    output = harness_phase.write_case_output("dev", args.meeting, payload)
    (case_dir / "canonical_case.json").write_text(
        json.dumps({"meeting": args.meeting, "case_output": output, "stdout_sha256": sha256_file(stdout_path)}, indent=1) + "\n",
        encoding="utf-8",
    )
    meta = {}
    if args.launch_meta:
        for line in Path(args.launch_meta).read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    meta = json.loads(stripped)
                    break
                except json.JSONDecodeError:
                    continue
    summary = {
        "case_output": output,
        "stdout": {"path": str(stdout_path), "bytes": stdout_path.stat().st_size, "sha256": sha256_file(stdout_path)},
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
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("meeting", "ok", "completed", "execution_completed", "evaluation_valid", "operational_clean", "counts", "ledger", "case_output", "launch")}, indent=1, ensure_ascii=False, default=str))
    return 0


def _project_parents(case: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    parents: list[dict] = []
    marks: list[dict] = []
    meeting = case.get("meeting")
    for row in list(case.get("parents") or ()):
        parents.append(
            {
                "parent_id": row.get("parent_id"),
                "meeting": meeting,
                "text": row.get("text") or "",
                "cluster_id": row.get("cluster_id") or meeting,
                "sequential_target": bool(row.get("sequential_target")),
                "status": row.get("status"),
                "degraded": bool(row.get("degraded")),
                "clean_completion": bool(row.get("clean_completion")),
                "text_authority": row.get("text_authority"),
                "failure_reason": row.get("failure_reason"),
                "outcome": row.get("outcome"),
                "seal_reason": row.get("seal_reason"),
                "conserved": row.get("conserved"),
                "span": row.get("span"),
                "content_ranges": parent_content_ranges(row),
                "r0": row.get("r0") or {},
                "r2": row.get("r2") or {},
                "guard": row.get("guard"),
                "incomplete": bool(row.get("incomplete")),
                "outage": bool(row.get("outage")),
                "accounted": bool(row.get("accounted", True)),
                "provenance_valid": bool(row.get("provenance_valid", True)),
            }
        )
        marks.append(row.get("marks") or {})
    return parents, marks


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
        _manifest_inputs(Path(args.manifest))
        if args.manifest
        else _directory_inputs(args.dirs)
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
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
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
        "aggregate_payload": {"path": str(out), "sha256": sha256_file(out), "bytes": out.stat().st_size},
    }
    print(json.dumps(compact, indent=1, ensure_ascii=False, default=str))
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
    args = parser.parse_args()
    if args.mode == "case":
        return case_mode(args)
    return aggregate_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
