"""Per-case canonical preservation + phase aggregation for the DEV run.

--case       verify one meeting's launcher payload, write the canonical case
             output through the harness phase.write_case_output path, and emit a
             compact summary plus provisional per-case evidence.
--aggregate  rebuild the exact run.py parent projection from the five canonical
             case payloads, call the harness phase.aggregate_phase, and write the
             full aggregate payload plus a compact summary.

No behavior patching: only the existing harness entry points are called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

TARGET = Path(
    r"C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls"
)
sys.path.insert(0, str(TARGET / "src"))
sys.path.insert(0, str(TARGET))

from experiments.psem_r2_policy import budget as harness_budget  # noqa: E402
from experiments.psem_r2_policy import phase as harness_phase  # noqa: E402

CANONICAL_ARTIFACTS = TARGET / "experiments/psem_r2_policy/artifacts"


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
                "r0": row.get("r0") or {},
                "r2": row.get("r2") or {},
                "incomplete": bool(row.get("incomplete")),
                "outage": bool(row.get("outage")),
                "accounted": bool(row.get("accounted", True)),
                "provenance_valid": bool(row.get("provenance_valid", True)),
            }
        )
        marks.append(row.get("marks") or {})
    return parents, marks


def aggregate_mode(args: argparse.Namespace) -> int:
    parents: list[dict] = []
    marks: list[dict] = []
    cases: list[dict] = []
    provenance: list[dict] = []
    for case_dir_raw in args.dirs.split(","):
        case_dir = Path(case_dir_raw.strip())
        pointer = _json_load(case_dir / "canonical_case.json")
        case_path = Path(pointer["case_output"]["path"])
        observed = sha256_file(case_path)
        if observed != pointer["case_output"]["sha256"]:
            raise SystemExit(f"case output hash mismatch: {case_path}")
        case = _json_load(case_path)
        case_parents, case_marks = _project_parents(case)
        parents.extend(case_parents)
        marks.extend(case_marks)
        cases.append(case)
        provenance.append(
            {
                "meeting": case.get("meeting"),
                "case_output": pointer["case_output"],
                "stdout_sha256": pointer.get("stdout_sha256"),
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
    agg_parser.add_argument("--dirs", required=True)
    agg_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "case":
        return case_mode(args)
    return aggregate_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
