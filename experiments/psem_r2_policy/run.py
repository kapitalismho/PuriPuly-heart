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

from experiments.psem_r2_policy.live_runner import LIVE_ROUTE, ami_wav_path
from experiments.psem_r2_policy.phase import (
    aggregate_phase,
    holdout_unlock_error,
    load_protocol,
    resolve_meetings,
    write_case_output,
)
from experiments.psem_r2_policy.pipeline import (
    run_intercepted_deepgram_path,
    run_paid_live,
    run_synthetic_path,
)


def _ok_split(result: dict) -> bool:
    enabled = result["enabled"]
    return (
        result["network"] is False
        and result["text"] == "Hello there"
        and enabled["conserved"] is True
        and enabled["group_ids"] == ["CURRENT-0", "OTHER-1"]
        and enabled["child_groups"] == ["CURRENT-0", "OTHER-1"]
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
    if paid:
        paid_result = await run_paid_live(wav, phase=phase or "dev")
        paid_result["asr"] = LIVE_ROUTE["asr_provider"]
        paid_result["asr_model"] = LIVE_ROUTE["asr_model"]
        paid_result["translation"] = LIVE_ROUTE["translation"]
        paid_result["protocol_revision"] = protocol.get("revision")
        return paid_result
    if phase:
        if phase == "holdout":
            locked = holdout_unlock_error()
            if locked is not None:
                return {
                    "ok": False,
                    "network": False,
                    "phase": phase,
                    "reason": locked,
                    "confirmatory": {
                        "result": "Inconclusive due to sample, timing, alignment, runtime or budget gap",
                        "pass": False,
                        "n_eligible_clusters": 0,
                    },
                    "protocol_revision": protocol.get("revision"),
                }
        meetings = resolve_meetings(phase, meeting)
        parents: list[dict] = []
        outputs = []
        marks = []
        for item in meetings:
            path = ami_wav_path(item)
            case = await run_paid_live(str(path), phase=phase)
            case["meeting"] = item
            outputs.append(write_case_output(phase, item, case))
            if case.get("incomplete") or case.get("outage"):
                parents.append({"incomplete": True, "meeting": item, "cluster_id": item})
                continue
            if case.get("r0") and case.get("r2"):
                parents.append(
                    {
                        "meeting": item,
                        "cluster_id": case.get("cluster_id"),
                        "sequential_target": bool(
                            (case.get("metrics") or {}).get("sequential_target")
                        ),
                        "r0": case["r0"],
                        "r2": case["r2"],
                        "incomplete": False,
                    }
                )
            marks.append(case.get("marks") or {})
        summary = aggregate_phase(parents, marks=marks)
        return {
            "ok": False,
            "network": False,
            "phase": phase,
            "meetings": list(meetings),
            "outputs": outputs,
            "reason": "phase network execution remains Director gated",
            "confirmatory": summary["confirmatory"],
            "cluster_aggregate": summary["cluster_aggregate"],
            "latency_by_operation": summary["latency_by_operation"],
            "protocol_revision": protocol.get("revision"),
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
    intercept = await run_intercepted_deepgram_path()
    return {
        "ok": _ok_split(intercept) and intercept["n_timed"] == 2,
        "network": False,
        "mode": "smoke" if smoke else "intercept",
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
    if args.paid:
        executor = payload.get("executor") or payload.get("paid_executor")
        return 0 if executor == "run_continuous_wav" and payload.get("wav_path") else 1
    if args.phase == "holdout" and not payload.get("ok"):
        return 1
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
