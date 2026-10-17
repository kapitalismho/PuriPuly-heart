from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

import pytest

from tests.fixtures.self_grace_233.experiment import POLICIES, Case, Step, run_case

HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_cases(schedule, inputs, delay):
    events = schedule["events"]
    origin = schedule["origin_monotonic_s"]
    cases = []
    for case_name in dict.fromkeys(entry["case"] for entry in inputs["episodes"]):
        labels = [entry["label"] for entry in inputs["episodes"] if entry["case"] == case_name]
        mapping = dict(zip(labels, ("A", "B"), strict=True))
        selected = [event for event in events if event.get("label") in mapping]
        first = next(event for event in selected if event["event"] == "vad_start")
        base = first["source"]["opened_at_monotonic_s"] - origin
        steps = []
        for event in selected:
            acoustic = mapping[event["label"]]
            arrival = event["at_s"] - base
            kind = event["event"]
            if kind in ("vad_start", "vad_chunk", "vad_end"):
                source = event["source"]
                source_at = (
                    source["sealed_at_monotonic_s"]
                    if kind == "vad_end"
                    else source["content_ranges"][-1]["source_start_monotonic_s"]
                )
                steps.append(
                    Step(
                        at=arrival,
                        event=kind.removeprefix("vad_"),
                        acoustic=acoustic,
                        source_at=source_at - origin - base,
                    )
                )
            elif kind == "asr_terminal":
                payload = event["payload"]
                steps.append(
                    Step(
                        at=arrival,
                        event="terminal",
                        acoustic=acoustic,
                        text=payload["text"],
                        outcome=payload["outcome"],
                    )
                )
            else:
                raise RuntimeError("This bridge only accepts the observed final-only route")
        cases.append(
            Case(
                name=f"recorded_audio_{case_name}_{delay:g}",
                steps=tuple(steps),
                llm_delay=delay,
                until=max(step.at for step in steps) + delay + 2.0,
                overlap=True,
                route="recorded_local_parakeet_v3_final_only_replay",
                expected_defect=True,
            )
        )
    return cases


async def execute(args):
    schedule_path = HERE / args.schedule
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    if sha256(HERE / "inputs.json") != schedule["inputs_manifest_sha256"]:
        raise RuntimeError("Recorded schedule and frozen input set differ")
    if "source_at" not in Step.__dataclass_fields__:
        raise RuntimeError(
            "Shared deterministic fixture must expose Step.source_at before clock-preserving replay"
        )
    results = []
    cases = []
    for delay in args.llm_delay:
        for case in make_cases(schedule, inputs, delay):
            cases.append(asdict(case))
            for policy in POLICIES:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    results.append(await run_case(case, policy, monkeypatch))
    report = {
        "schema": "self-grace-233/real-asr-schedule-policy-replay/v1",
        "schedule": schedule_path.name,
        "schedule_sha256": sha256(schedule_path),
        "inputs_manifest_sha256": schedule["inputs_manifest_sha256"],
        "fixture_sha256": sha256(ROOT / "tests/fixtures/self_grace_233/experiment.py"),
        "command": [sys.executable, *sys.argv],
        "cases": cases,
        "results": results,
        "limits": [
            "This imports the existing deterministic fixture; it does not build a second pipeline",
            "Observed real local CPU ASR arrival times and texts are fixed for all policies",
            "VAD source timestamp and handler receipt remain distinct",
            "Provider identities are consistently remapped to the shared fixture acoustic A/B identities",
            "Fixture audio chunks are modeled ingress, not fresh speech decode or literal audio replay",
            "No translation-quality, live-capture, fresh-provider, OSC or HMD improvement claim",
            "Only LLM delay is varied between comparison groups; provider admission/output clocks are fake and coordinated with timer wakeups",
            "Shared fixture profiles and sinks differ from the production probe; read their recorded effective profiles",
        ],
    }
    (HERE / args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": args.report,
                "samples": len(results),
                "rows": [
                    {
                        key: row[key]
                        for key in (
                            "case",
                            "policy",
                            "source_inputs",
                            "commits",
                            "premature_commits",
                            "grace_only_wait_ms",
                            "metrics",
                        )
                    }
                    for row in results
                ],
            }
        )
    )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default="baseline_translation_asr_schedule.json")
    parser.add_argument("--report", default="recorded_schedule_comparison.json")
    parser.add_argument("--llm-delay", type=float, nargs="+", default=[0.05, 1.0])
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.schedule, args.report)) or any(
        delay < 0 for delay in args.llm_delay
    ):
        parser.error("Inputs and report must be task-local filenames and LLM delays nonnegative")
    return asyncio.run(execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
