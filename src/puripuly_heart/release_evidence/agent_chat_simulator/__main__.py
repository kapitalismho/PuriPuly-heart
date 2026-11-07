from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid7

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from puripuly_heart import __version__
from puripuly_heart.app.adapters.agent_relay_auth import AgentRelayTokenOwner
from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import AgentServiceError
from puripuly_heart.core.managed_identity import ManagedIdentityBundle, encode_base64url

from .execution import (
    DEFAULT_DIALOGUE_CONCURRENCY,
    DEFAULT_JUDGE_CONCURRENCY,
    AttemptJob,
    AttemptScheduler,
)
from .judging import (
    CRITERIA,
    JUDGE_REVISION,
    apply_calibration,
    calibration_export,
    compare_calibration,
    judge,
)
from .models import JUDGE_EFFORT, JUDGE_MODEL, SubscriptionConnectGate
from .relay_context import load_relay_context
from .runner import (
    ControlledModels,
    controlled_transport,
    invocation_evidence,
    report_header,
    run_attempt,
    run_case,
    write_json,
)
from .scenarios import SCENARIOS, select_scenarios


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def probe_completed(result: dict) -> bool:
    roles = result.get("roles", {})
    return all(
        isinstance(roles.get(role), dict)
        and roles[role].get("status") == "completed"
        and roles[role].get("value") == {"ok": True}
        for role in ("user", "judge")
    )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be greater than zero")
    return number


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Issue #332 bounded synthetic Agent Chat simulator. No GUI or microphone. Controlled smoke is never live evaluation evidence."
    )
    commands = command.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help=f"Default: all {len(SCENARIOS)} cases, exactly 3 logical runs each"
    )
    run.add_argument("--case", action="append", dest="cases")
    run.add_argument("--split", choices=["development", "heldout"])
    run.add_argument("--relay", default=os.environ.get("PURIPULY_SIMULATOR_RELAY"))
    run.add_argument("--relay-build", default=os.environ.get("PURIPULY_SIMULATOR_RELAY_BUILD"))
    run.add_argument(
        "--relay-context",
        default=os.environ.get("PURIPULY_SIMULATOR_RELAY_CONTEXT"),
        help="Required trusted-context manifest exported offline from this exact emitted relay build",
    )
    run.add_argument(
        "--broker",
        default=os.environ.get("PURIPULY_SIMULATOR_BROKER"),
        help="Isolated broker origin; actual signing/refresh with a fresh memory-only synthetic identity per attempt",
    )
    run.add_argument(
        "--isolated-staging",
        action="store_true",
        help="Confirm relay token/installation/D1/R2 and support destination are isolated synthetic staging resources, local or remote",
    )
    run.add_argument(
        "--sign-in",
        action="store_true",
        help="Reuse benchmark-only OS-keyring ChatGPT login, signing in if none exists, before same-model API fallback",
    )
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument(
        "--subscription-only",
        action="store_true",
        help="Use benchmark-only OS-keyring ChatGPT login; never use an API key",
    )
    run.add_argument(
        "--reauthenticate", action="store_true", help="Explicitly renew benchmark ChatGPT login"
    )
    run.add_argument("--output", default="agent-chat-simulator-results.json")
    run.add_argument(
        "--dialogue-concurrency",
        type=positive_int,
        default=DEFAULT_DIALOGUE_CONCURRENCY,
        help="Concurrent dialogues per model arm (default: %(default)s)",
    )
    run.add_argument(
        "--judge-concurrency",
        type=positive_int,
        default=DEFAULT_JUDGE_CONCURRENCY,
        help="Concurrent judge completions per model arm (default: %(default)s)",
    )
    run.add_argument(
        "--calibration-samples",
        help="Export first20 without overwriting an existing frozen selection",
    )
    smoke = commands.add_parser(
        "smoke",
        help="No-network controlled responses through actual owner/adapters/HTTP transport; not live passes",
    )
    smoke.add_argument("--output", default="agent-chat-simulator-smoke.json")
    smoke.add_argument("--case", action="append", dest="cases")
    smoke.add_argument(
        "--relay",
        help="Optional isolated emitted relay with controlled upstream; wiring-only, never a live scenario pass",
    )
    smoke.add_argument(
        "--token-url",
        help="Synthetic loopback token issuer for emitted relay smoke; token is never recorded",
    )
    smoke.add_argument("--relay-build")
    smoke.add_argument(
        "--relay-context",
        help="Optional emitted-context manifest; controlled smoke never creates live evidence",
    )
    probe = commands.add_parser(
        "probe", help=f"Probe fixed Terra-medium/Sol-{JUDGE_EFFORT} access, without scenarios"
    )
    probe.add_argument("--sign-in", action="store_true")
    probe.add_argument("--api-key-env", default="OPENAI_API_KEY")
    probe.add_argument(
        "--subscription-only",
        action="store_true",
        help="Use benchmark-only OS-keyring ChatGPT login; never use an API key",
    )
    probe.add_argument(
        "--reauthenticate", action="store_true", help="Explicitly renew benchmark ChatGPT login"
    )
    probe.add_argument("--output", default="agent-chat-model-access.json")
    export = commands.add_parser(
        "calibration-export", help="Freeze the first20 conversations for actual owner judgment"
    )
    export.add_argument("--report", required=True)
    export.add_argument("--output", required=True)
    compare = commands.add_parser(
        "calibration-compare",
        help="Compare same20 owner verdicts criterion by criterion; no inferred scores",
    )
    compare.add_argument("--samples", required=True)
    compare.add_argument("--owner", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument(
        "--report", help="Attach calibration to its original report without rerunning conversations"
    )
    compare.add_argument(
        "--previous",
        help="Previous same20 comparison; a changed judge revision is required before a second unreliable criterion becomes advisory",
    )
    rejudge = commands.add_parser(
        "calibration-rejudge",
        help="Rejudge the SAME frozen20 after judge wording/model revisions; never creates examples",
    )
    rejudge.add_argument("--samples", required=True)
    rejudge.add_argument("--sign-in", action="store_true")
    rejudge.add_argument("--api-key-env", default="OPENAI_API_KEY")
    rejudge.add_argument(
        "--subscription-only",
        action="store_true",
        help="Use benchmark-only OS-keyring ChatGPT login; never use an API key",
    )
    rejudge.add_argument(
        "--reauthenticate", action="store_true", help="Explicitly renew benchmark ChatGPT login"
    )
    rejudge.add_argument("--output", required=True)
    render = commands.add_parser(
        "render-html",
        help="Render one campaign's saved benchmark JSON as standalone offline Korean HTML",
    )
    render.add_argument("--report", action="extend", nargs="+", required=True, dest="reports")
    render.add_argument("--output", required=True)
    return command


def make_models(args, *, subscription_gate: SubscriptionConnectGate | None = None):
    from .models import SimulatorModels

    return SimulatorModels(
        api_key="" if args.subscription_only else os.environ.get(args.api_key_env, ""),
        sign_in=args.sign_in or args.subscription_only,
        reauthenticate=args.reauthenticate,
        subscription_gate=subscription_gate,
    )


def blocked_cases(cases, missing: list[str]) -> list[dict]:
    return [
        {
            "id": case.id,
            "category": case.category,
            "split": case.split,
            "status": "blocked",
            "successes": 0,
            "logical_runs": 3,
            "advisory_failures": {},
            "runs": [
                {
                    "id": f"{case.id}/run-{number}",
                    "status": "blocked",
                    "attempts": [],
                    "missing_prerequisites": missing,
                }
                for number in range(1, 4)
            ],
        }
        for case in cases
    ]


def save_samples(path: str, report: dict) -> None:
    if Path(path).exists():
        raise ValueError(
            "Calibration sample file already exists; retain the same IDs and use calibration-rejudge, not a new export."
        )
    write_json(path, calibration_export(report))


async def execute(args) -> int:
    if args.command == "render-html":
        from .html_report import render_html

        result = render_html(args.reports, args.output)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "calibration-export":
        save_samples(args.output, load_json(args.report))
        print(json.dumps({"output": args.output, "status": "exported"}))
        return 0
    if args.command == "calibration-compare":
        samples = load_json(args.samples)
        result = compare_calibration(
            samples, load_json(args.owner), load_json(args.previous) if args.previous else None
        )
        if args.report:
            report = load_json(args.report)
            frozen = calibration_export(report)
            if [(item["id"], item["digest"]) for item in frozen["conversations"]] != [
                (item["id"], item["digest"]) for item in samples["conversations"]
            ]:
                result = {
                    "status": "blocked",
                    "reason": "Calibration samples do not belong to this report's first20.",
                }
            apply_calibration(report, samples, result)
            all_cases = len(report["cases"]) == len(SCENARIOS) and all(
                case["status"] == "passed" for case in report["cases"]
            )
            report["live_pass"] = (
                result["status"] == "passed" and all_cases and not report.get("controlled_smoke")
            )
            report["status"] = "passed" if report["live_pass"] else "blocked_or_failed"
            write_json(args.report, report)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "passed" else 2
    relay_contexts = None
    if args.command == "run" or (args.command == "smoke" and args.relay_context):
        relay_contexts = load_relay_context(args.relay_context, relay_build=args.relay_build)
    if args.command == "smoke":
        if args.relay:
            if not args.token_url or not args.relay_build:
                raise ValueError(
                    "Emitted wiring smoke requires synthetic token URL and actual relay build identity"
                )
            for url in (args.relay, args.token_url):
                parsed = urlsplit(url)
                if (
                    parsed.scheme != "http"
                    or parsed.hostname not in {"127.0.0.1", "::1"}
                    or parsed.username
                    or parsed.password
                ):
                    raise ValueError("Controlled emitted smoke permits only explicit loopback HTTP")

            class LoopbackToken:
                def __init__(self):
                    self.rejected_token = None

                async def get_token(self):
                    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
                        response = await client.get(args.token_url)
                        response.raise_for_status()
                        token = response.json()["token"]
                        if token == self.rejected_token:
                            raise AgentServiceError("service_authentication_failed")
                        return token

                def invalidate(self, token):
                    self.rejected_token = token

                async def close(self):
                    return None

            def emitted(_app):
                return AgentRelayTransport(args.relay, LoopbackToken(), allow_loopback_http=True)

            result = await run_attempt(
                select_scenarios(["recognition-capture-off"])[0],
                ControlledModels(),
                emitted,
                identity="emitted-wiring",
                controlled=True,
                relay_contexts=relay_contexts,
            )
            methods = [
                call["method"]
                for call in result["relay_calls"]
                if call.get("status") == "completed"
            ]
            context_calls = {
                item["call_id"]
                for item in result["archive"]
                if item.get("type") == "function_call" and item.get("name") == "get_app_context"
            }
            wired = (
                "classify" in methods
                and methods.count("respond") >= 2
                and any(
                    item.get("type") == "function_call_output"
                    and item.get("call_id") in context_calls
                    and json.loads(item["output"]).get("status") == "observed"
                    for item in result["archive"]
                )
            )
            report = report_header(
                relay=args.relay,
                build=args.relay_build,
                controlled=True,
                relay_contexts=relay_contexts,
            )
            report.update(planned_scenarios=1, planned_runs=1)
            report.update(
                status="controlled_wiring_completed" if wired else "controlled_wiring_failed",
                attempt=result,
                limits_note="Actual owner, adapters, HTTP, authenticated emitted Worker and local context tool roundtrip; upstream/user/judge controlled. No live scenario pass or new support-destination assertion.",
            )
            write_json(args.output, report)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "live_pass": False,
                        "methods": methods,
                        "output": args.output,
                    }
                )
            )
            return 0 if wired else 1
        report = report_header(
            relay="controlled HTTPX transport; no network or emitted Worker",
            build=None,
            controlled=True,
            relay_contexts=relay_contexts,
        )
        cases = select_scenarios(args.cases)
        report.update(planned_scenarios=len(cases), planned_runs=len(cases))
        models = ControlledModels()
        for case in cases:
            result = await run_attempt(
                case,
                models,
                controlled_transport,
                identity=f"smoke/{case.id}",
                controlled=True,
                relay_contexts=relay_contexts,
            )
            report["cases"].append({"id": case.id, "status": result["status"], "attempt": result})
        report["status"] = (
            "controlled_smoke_completed"
            if all(case["status"] == "passed" for case in report["cases"])
            else "controlled_smoke_failed"
        )
        report["limits_note"] = (
            "One deterministic smoke per selected boundary, not the three-run live evaluation. Provider, D, M, user, judge and support receiver are controlled. Actual owner/context/tools/transport/bundle creation execute. No emitted relay, hardware, private settings, real key or network is used."
        )
        write_json(args.output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "live_pass": False,
                    "cases": [
                        {"id": case["id"], "status": case["status"]} for case in report["cases"]
                    ],
                    "output": args.output,
                }
            )
        )
        return 0 if report["status"] == "controlled_smoke_completed" else 1
    models = make_models(args)
    try:
        if args.command == "probe":
            try:
                result = await models.probe()
                report = {
                    "status": "completed" if probe_completed(result) else "failed",
                    "probe": result,
                    "attempts": deepcopy(models.attempts),
                }
            except Exception as error:
                report = {
                    "status": "blocked",
                    "code": getattr(error, "code", type(error).__name__),
                    "evidence": getattr(error, "evidence", None),
                    "attempts": deepcopy(models.attempts),
                }
            write_json(args.output, report)
            print(json.dumps({"status": report["status"], "output": args.output}))
            return 0 if report["status"] == "completed" else 2
        if args.command == "calibration-rejudge":
            if (
                Path(args.output).resolve() == Path(args.samples).resolve()
                or Path(args.output).exists()
            ):
                raise ValueError("Rejudgment requires a new output; frozen samples stay unchanged")
            frozen = load_json(args.samples)
            if len(frozen.get("conversations", [])) != 20:
                raise ValueError("The frozen same20 conversations are required")
            samples = {
                "format": "puri-agent-calibration-2",
                "judge_revision": JUDGE_REVISION,
                "judge_model": JUDGE_MODEL,
                "judge_effort": JUDGE_EFFORT,
                "conversations": frozen["conversations"],
                "status": "incomplete",
                "access_attempts": [],
            }
            for sample in samples["conversations"]:
                sample["evidence"]["judgment"] = {
                    "status": "pending",
                    "revision": JUDGE_REVISION,
                    "model": JUDGE_MODEL,
                    "effort": JUDGE_EFFORT,
                }
                sample["rejudgment_calls"] = []
                sample["rejudgment_access_attempts"] = []
                if Counter(
                    item.get("name") for item in sample.get("owner_criteria", [])
                ) != Counter(CRITERIA):
                    sample["owner_criteria"] = [
                        {"name": name, "verdict": None} for name in CRITERIA
                    ]
            write_json(args.output, samples)
            try:
                await models.prepare()
            except (Exception, asyncio.CancelledError) as error:
                samples.update(
                    status="blocked",
                    code=(
                        "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else getattr(error, "code", type(error).__name__)
                    ),
                    evidence=getattr(error, "evidence", None),
                )
            else:
                for sample in samples["conversations"]:
                    evidence = sample["evidence"]
                    access_start = len(models.attempts)
                    try:
                        evidence["judgment"] = await judge(
                            models,
                            {
                                "scenario": evidence["scenario"],
                                "controlled": evidence.get("controlled", False),
                                "transcript": evidence["transcript"],
                                "actions": evidence["actions"],
                                "archive": evidence["archive"],
                                "events": evidence["events"],
                                **invocation_evidence(
                                    evidence["relay_calls"],
                                    evidence["events"],
                                    relay_contexts=evidence.get("relay_contexts"),
                                ),
                                "code": evidence["code"],
                            },
                            sample["rejudgment_calls"],
                        )
                    except asyncio.CancelledError:
                        evidence["judgment"].update(status="cancelled", code="cancelled")
                        samples["code"] = "cancelled"
                        break
                    except Exception as error:
                        evidence["judgment"].update(
                            status="blocked",
                            code=getattr(error, "code", type(error).__name__),
                            evidence=getattr(error, "evidence", None),
                        )
                    finally:
                        sample["rejudgment_access_attempts"] = deepcopy(
                            models.attempts[access_start:]
                        )
                        samples["access_attempts"] = deepcopy(models.attempts)
                        write_json(args.output, samples)
                if all(
                    sample["evidence"]["judgment"]["status"] == "valid"
                    for sample in samples["conversations"]
                ):
                    samples["status"] = "rejudged_same20"
            samples["access_attempts"] = deepcopy(models.attempts)
            write_json(args.output, samples)
            print(json.dumps({"status": samples["status"], "output": args.output}))
            return 0 if samples["status"] == "rejudged_same20" else 2
        cases = select_scenarios(args.cases, args.split)
        report = report_header(
            relay=args.relay,
            build=args.relay_build,
            controlled=False,
            relay_contexts=relay_contexts,
        )
        report.update(planned_scenarios=len(cases), planned_runs=len(cases) * 3)
        completed_attempts = {}
        case_results = {}
        report["execution_policy"] = {
            "dialogue_concurrency": args.dialogue_concurrency,
            "judge_concurrency": args.judge_concurrency,
            "dialogue_order": "sequential within each independent conversation",
            "judgment": "separate bounded queue",
        }
        report["selection"] = (
            "full" if len(cases) == len(SCENARIOS) else "targeted_not_full_release_evidence"
        )
        report["broker_origin"] = args.broker
        report["identity_policy"] = (
            "Fresh in-memory UUIDv7/Ed25519 synthetic installation per attempt; actual AgentRelayTokenOwner signs and refreshes for the same identity within that attempt."
        )
        missing = []
        if not args.relay:
            missing.append(
                "isolated staging relay origin (local authenticated workerd is permitted)"
            )
        if not args.relay_build:
            missing.append("actual running relay bundle/build identity")
        if not args.isolated_staging:
            missing.append("explicit isolated synthetic staging/support destination confirmation")
        if not args.broker:
            missing.append(
                "isolated broker origin for signed synthetic-installation tokens and refresh"
            )
        if missing:
            report["cases"] = blocked_cases(cases, missing)
            report["status"] = "blocked"
            report["missing_prerequisites"] = missing
        else:
            try:
                await models.prepare()
            except Exception as error:
                missing.append(
                    f"Terra-medium/Sol-{JUDGE_EFFORT} access: "
                    + getattr(error, "code", type(error).__name__)
                )
                report["access_failure"] = getattr(error, "evidence", None)
            if missing:
                report["cases"] = blocked_cases(cases, missing)
                report["status"] = "blocked"
            else:
                journal_path = Path(args.output).with_suffix(".attempts.jsonl")
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                with journal_path.open("x", encoding="utf-8"):
                    pass
                report["attempt_journal"] = str(journal_path)
                worker_args = argparse.Namespace(**vars(args))
                worker_args.sign_in = args.sign_in or args.reauthenticate
                worker_args.reauthenticate = False
                subscription_gate = SubscriptionConnectGate()

                def factory(app):
                    private_key = Ed25519PrivateKey.generate()
                    bundle = ManagedIdentityBundle(
                        installation_id=str(uuid7()),
                        device_public_key=encode_base64url(
                            private_key.public_key().public_bytes_raw()
                        ),
                        _private_key=private_key,
                    )
                    app.synthetic_installation_id = bundle.installation_id

                    async def identity():
                        return bundle

                    broker = urlsplit(args.broker)
                    auth = AgentRelayTokenOwner(
                        lambda: args.broker,
                        identity,
                        app_version=__version__,
                        allow_loopback_http=broker.scheme == "http"
                        and broker.hostname in {"127.0.0.1", "::1"},
                    )
                    parsed = urlsplit(args.relay)
                    return AgentRelayTransport(
                        args.relay,
                        auth,
                        allow_loopback_http=parsed.scheme == "http"
                        and parsed.hostname in {"127.0.0.1", "::1"},
                    )

                jobs = []
                for case in cases:
                    for repetition in range(1, 4):
                        identity = f"{case.id}/run-{repetition}/attempt-1"

                        async def attempt(scoped_models, case=case, identity=identity):
                            return await run_attempt(
                                case,
                                scoped_models,
                                factory,
                                identity=identity,
                                relay_contexts=relay_contexts,
                            )

                        jobs.append(AttemptJob(identity, case.id, attempt))

                async def checkpoint(job, result):
                    with journal_path.open("a", encoding="utf-8") as journal:
                        journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                        journal.flush()
                        os.fsync(journal.fileno())
                    completed_attempts[job.identity] = result
                    case = next(case for case in cases if case.id == job.group)
                    runs = [
                        {
                            "id": f"{case.id}/run-{repetition}",
                            "status": saved["status"],
                            "attempts": [saved],
                        }
                        for repetition in range(1, 4)
                        if (
                            saved := completed_attempts.get(f"{case.id}/run-{repetition}/attempt-1")
                        )
                        is not None
                    ]

                    async def saved_attempt(scenario, identity):
                        return completed_attempts[identity]

                    case_results[case.id] = (
                        await run_case(case, saved_attempt)
                        if len(runs) == 3
                        else {
                            "id": case.id,
                            "category": case.category,
                            "split": case.split,
                            "status": "incomplete",
                            "logical_runs": 3,
                            "runs": runs,
                        }
                    )
                    report["cases"] = [
                        case_results[case.id] for case in cases if case.id in case_results
                    ]
                    report["access_attempts"] = models.attempts + [
                        call
                        for attempt in completed_attempts.values()
                        for call in attempt.get("access_attempts", [])
                    ]
                    write_json(args.output, report)

                scheduler = AttemptScheduler(
                    lambda job: make_models(worker_args, subscription_gate=subscription_gate),
                    dialogue_concurrency=args.dialogue_concurrency,
                    judge_concurrency=args.judge_concurrency,
                )
                await scheduler.run(jobs, checkpoint=checkpoint)
                report["scenario_status"] = (
                    "passed"
                    if all(case["status"] == "passed" for case in report["cases"])
                    else (
                        "harness_error"
                        if any(case["status"] == "harness_error" for case in report["cases"])
                        else "failed_or_blocked"
                    )
                )
                report["status"] = (
                    "blocked_calibration"
                    if report["scenario_status"] == "passed"
                    else report["scenario_status"]
                )
        report["access_attempts"] = models.attempts + [
            call
            for attempt in completed_attempts.values()
            for call in attempt.get("access_attempts", [])
        ]
        report["summary"] = {
            "planned_scenarios": report["planned_scenarios"],
            "saved_scenarios": report["saved_scenarios"],
            "planned_runs": report["planned_runs"],
            "saved_runs": report["saved_runs"],
            "case_statuses": {
                status: sum(case["status"] == status for case in report["cases"])
                for status in ("passed", "failed", "blocked", "harness_error")
            },
            "cost_usd": None,
            "cost_basis": "Unknown until all role usage and valid rates are available; never treated as zero.",
        }
        write_json(args.output, report)
        if args.calibration_samples:
            save_samples(args.calibration_samples, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "summary": report["summary"],
                    "output": args.output,
                    "calibration": report["calibration"]["status"],
                }
            )
        )
        return 2
    finally:
        await models.close()


def main() -> int:
    args = parser().parse_args()
    try:
        return asyncio.run(execute(args))
    except (ValueError, OSError) as error:
        result = {"status": "blocked", "code": type(error).__name__}
        if args.command == "render-html":
            result["detail"] = str(error)
        print(json.dumps(result, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
