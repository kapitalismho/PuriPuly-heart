from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.budget import LEDGER_PATH, BudgetLedger
from experiments.psem_r2_policy.live_runner import LIVE_ROUTE
from experiments.psem_r2_policy.metrics import u8_case_report
from experiments.psem_r2_policy.phase import (
    aggregate_phase,
    load_protocol,
    resolve_meetings,
    write_case_output,
)
from experiments.psem_r2_policy.pipeline import (
    refuse_paid_if_disabled,
    run_intercepted_deepgram_path,
    run_paid_live,
    run_synthetic_path,
)


def _ok_split(result: dict) -> bool:
    parents = list(result.get("parents") or ())
    if result.get("network") is not False or len(parents) != 1:
        return False
    parent = parents[0]
    return (
        parent["text"] == "Hello there"
        and parent["conserved"] is True
        and parent["group_ids"] == ["CURRENT-0", "OTHER-1"]
        and (parent["r2"] or {}).get("child_groups") == ["CURRENT-0", "OTHER-1"]
    )


async def execute(
    *,
    paid: bool,
    offline_replay: bool,
    smoke: bool,
    wav: str | None,
    phase: str | None,
    meeting: str | None,
) -> dict:
    protocol = load_protocol()
    if paid or phase:
        selected_phase = phase or "dev"
        meetings: tuple[str, ...]
        if meeting:
            try:
                meetings = resolve_meetings(selected_phase, meeting)
            except ValueError as exc:
                return {
                    "ok": False,
                    "completed": False,
                    "refused": True,
                    "paid_blocked": True,
                    "network": False,
                    "runner_called": False,
                    "reason": str(exc),
                    "protocol_revision": protocol.get("revision"),
                }
        elif paid:
            return {
                "ok": False,
                "completed": False,
                "refused": True,
                "paid_blocked": True,
                "network": False,
                "runner_called": False,
                "reason": "--paid/--phase requires a declared --phase and --meeting",
                "protocol_revision": protocol.get("revision"),
            }
        else:
            try:
                meetings = resolve_meetings(selected_phase, None)
            except ValueError as exc:
                return {
                    "ok": False,
                    "completed": False,
                    "refused": True,
                    "paid_blocked": True,
                    "network": False,
                    "runner_called": False,
                    "reason": str(exc),
                    "protocol_revision": protocol.get("revision"),
                }
        first = refuse_paid_if_disabled(
            phase=selected_phase,
            meeting=meetings[0],
            wav_path=wav,
        )
        if first is not None:
            first["protocol_revision"] = protocol.get("revision")
            first["meetings"] = list(meetings)
            return first
        ledger = BudgetLedger(LEDGER_PATH)
        if paid:
            paid_result = await run_paid_live(
                wav,
                budget=ledger,
                phase=selected_phase,
                meeting=meetings[0],
            )
            paid_result["asr"] = LIVE_ROUTE["asr_provider"]
            paid_result["asr_model"] = LIVE_ROUTE["asr_model"]
            paid_result["translation"] = LIVE_ROUTE["translation"]
            paid_result["u8"] = u8_case_report(paid_result)
            paid_result["execution_completed"] = bool(paid_result["u8"]["execution_completed"])
            paid_result["evaluation_valid"] = bool(paid_result["u8"]["evaluation_valid"])
            paid_result["operational_clean"] = bool(paid_result["u8"]["operational_clean"])
            paid_result["conditional_support"] = bool(
                (paid_result.get("decision") or {}).get("pass")
            )
            paid_result["ok"] = bool(
                paid_result["execution_completed"] and paid_result["evaluation_valid"]
            )
            paid_result["clean_completion"] = bool(paid_result.get("clean_completion"))
            paid_result["protocol_revision"] = protocol.get("revision")
            paid_result["ledger_path"] = str(ledger.path)
            return paid_result
        parents: list[dict] = []
        cases: list[dict] = []
        outputs = []
        marks = []
        for item in meetings:
            blocked = refuse_paid_if_disabled(phase=selected_phase, meeting=item, wav_path=None)
            if blocked is not None:
                blocked["protocol_revision"] = protocol.get("revision")
                blocked["meetings"] = list(meetings)
                return blocked
            case = await run_paid_live(
                None,
                budget=ledger,
                phase=selected_phase,
                meeting=item,
            )
            case["meeting"] = item
            if case.get("refused") or case.get("paid_blocked"):
                case["protocol_revision"] = protocol.get("revision")
                return case
            outputs.append(write_case_output(selected_phase, item, case))
            cases.append(case)
            case_parents = list(case.get("parents") or ())
            for row in case_parents:
                parents.append(
                    {
                        "parent_id": row.get("parent_id"),
                        "meeting": item,
                        "text": row.get("text") or "",
                        "cluster_id": row.get("cluster_id") or item,
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
                        "guard": row.get("guard"),
                        "incomplete": bool(row.get("incomplete")),
                        "outage": bool(row.get("outage")),
                        "accounted": bool(row.get("accounted", True)),
                        "provenance_valid": bool(row.get("provenance_valid", True)),
                    }
                )
                marks.append(row.get("marks") or {})
        summary = aggregate_phase(parents, marks=marks, cases=cases, phase=selected_phase)
        execution_completed = bool(summary["execution_completed"])
        evaluation_valid = bool(summary["evaluation_valid"])
        clean_completion = bool(parents) and all(
            bool(row.get("clean_completion")) for row in parents
        )
        return {
            "ok": execution_completed and evaluation_valid,
            "completed": execution_completed,
            "clean_completion": clean_completion,
            "operational_clean": summary["operational_clean"],
            "execution_completed": execution_completed,
            "evaluation_valid": evaluation_valid,
            "execution_incomplete_reasons": summary["execution_incomplete_reasons"],
            "evaluation_invalid_reasons": summary["evaluation_invalid_reasons"],
            "conditional_support": summary["confirmatory"]["conditional_support"],
            "network": True,
            "phase": selected_phase,
            "meetings": list(meetings),
            "outputs": outputs,
            "confirmatory": summary["confirmatory"],
            "cluster_aggregate": summary["cluster_aggregate"],
            "u8": summary["u8"],
            "operational_census": summary["operational_census"],
            "excluded_history": summary["excluded_history"],
            "n_operationally_unsuccessful": summary["n_operationally_unsuccessful"],
            "n_degraded_conditional": summary["n_degraded_conditional"],
            "n_incomplete_source_parents": summary["n_incomplete_source_parents"],
            "unsuccessful_parents": summary["unsuccessful_parents"],
            "degraded_parents": summary["degraded_parents"],
            "pool_exclusions": summary["pool_exclusions"],
            "sensitivity": summary["confirmatory"]["sensitivity"],
            "latency_by_operation": summary["latency_by_operation"],
            "protocol_revision": protocol.get("revision"),
            "ledger_path": str(ledger.path),
            "refused": False,
            "paid_blocked": False,
        }
    if offline_replay:
        synthetic = await run_synthetic_path()
        return {
            "ok": _ok_split(synthetic),
            "network": False,
            "mode": "offline_replay",
            "asr": LIVE_ROUTE["asr_provider"],
            "path": synthetic["path"],
            "synthetic": synthetic,
            "protocol_revision": protocol.get("revision"),
        }
    if not smoke:
        return {
            "ok": False,
            "completed": False,
            "refused": True,
            "network": False,
            "reason": "specify --smoke, --offline-replay, or --paid with declared --phase and --meeting",
            "protocol_revision": protocol.get("revision"),
        }
    intercept = await run_intercepted_deepgram_path()
    return {
        "ok": _ok_split(intercept) and (intercept["parents"][0]["n_timed"] == 2),
        "network": False,
        "mode": "smoke",
        "asr": LIVE_ROUTE["asr_provider"],
        "asr_model": LIVE_ROUTE["asr_model"],
        "translation": LIVE_ROUTE["translation"],
        "path": intercept["path"],
        "intercept": intercept,
        "protocol_revision": protocol.get("revision"),
        "protocol_asr": protocol.get("providers", {}).get("asr"),
        "live_methods": intercept.get("methods"),
        "open_session_calls": intercept.get("open_session_calls"),
        "r0": intercept.get("r0"),
        "r2": intercept.get("r2"),
        "r1": intercept.get("r1"),
        "control": intercept.get("control"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--wav")
    parser.add_argument("--offline-replay", nargs="?", const="generic")
    parser.add_argument("--phase", choices=["dev", "holdout"])
    parser.add_argument("--meeting")
    args = parser.parse_args(argv)
    payload = asyncio.run(
        execute(
            paid=args.paid,
            offline_replay=bool(args.offline_replay),
            smoke=args.smoke,
            wav=args.wav,
            phase=args.phase,
            meeting=args.meeting,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    if payload.get("refused") or payload.get("paid_blocked"):
        return 1
    if args.paid or args.phase:
        return 0 if payload.get("ok") and payload.get("completed") else 1
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
