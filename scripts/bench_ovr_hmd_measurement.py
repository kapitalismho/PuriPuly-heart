from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_LOCK = Path("C:/pph206-hmd-kit/active-live.lock")
ARMS = {
    "baseline": "8666bb57935b7c6da0c3c8aeaec9d116762d9f3d",
    "candidate": "c501b83350d4c39129cf61dd3e582b6ebbea04a5",
}
SCHEMA = "issue206-hmd-kit-v1"
RUN_SCHEMA = "issue206-hmd-run-v1"
OBSERVATION_SCHEMA = "issue206-hmd-observation-v1"
SOURCE_EXE_SHA256 = "5a4c14864dd2c8662bcfdb77c1076812f6a58fa45ad998c56d66e2b3880184ee"
VENDORED_DLL_SHA256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"
EXPECTED_STARTUP_CONTRACT = {
    "app_version": "2.7.0",
    "contract_version": 15,
    "execution_contract": {"revision": "r2", "version": 1},
    "native_presentation_retry": {"ownership": "exclusive", "version": 1},
    "speaker_identity_presentation": {"version": 2, "policy": "immutable_first_readable_style"},
}
SCENARIOS = (
    "independent",
    "stable",
    "final_only",
    "stable_burst",
    "resume",
    "mixed_active",
    "finalized_eviction",
    "sticky",
    "expiry",
    "clear_off",
    "restart_reconnect",
    "sustained",
)
ANCHORS = ("head_locked", "spatial_locked")
PHYSICAL_RESULTS = (
    "no_issue",
    "stale",
    "missing",
    "wrong_text",
    "placement",
    "flicker",
    "discomfort",
)


class MeasurementError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tree_identity(root: Path) -> dict:
    files = {
        p.relative_to(root).as_posix(): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    return {
        "files": files,
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def _interpreter_identity() -> dict:
    dependencies = sorted(
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    )
    return {
        "executable": str(Path(sys.executable).resolve()),
        "sha256": _sha256(Path(sys.executable)),
        "version": platform.python_version(),
        "dependencies": dependencies,
    }


def _check_startup_contract(executable: Path) -> dict:
    try:
        result = subprocess.run(
            [str(executable), "--check-startup-contract"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        contract = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise MeasurementError(f"startup contract inspection failed: {exc}") from exc
    if result.returncode != 0 or contract != EXPECTED_STARTUP_CONTRACT:
        raise MeasurementError("startup contract mismatch")
    return {"exit_code": result.returncode, "stdout": contract, "vr_initialization": False}


def prepare_session(executable: Path, stage: Path | None = None) -> Path:
    executable = executable.resolve()
    if not executable.is_file() or _sha256(executable) != SOURCE_EXE_SHA256:
        raise MeasurementError("native executable hash mismatch or missing")
    startup = _check_startup_contract(executable)
    stage = (
        stage
        or Path("C:/pph206-hmd-kit") / (_utc_now().replace(":", "") + "-" + secrets.token_hex(4))
    ).resolve()
    stage.mkdir(parents=True, exist_ok=False)
    source = stage / "source"
    source.mkdir()
    identities = {}
    for arm, revision in ARMS.items():
        archive = source / f"{arm}.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", revision],
            cwd=ROOT,
            check=True,
        )
        destination = source / arm
        destination.mkdir()
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")
        identities[arm] = {
            "revision": revision,
            "archive_sha256": _sha256(archive),
            "tree": _tree_identity(destination),
            "lock_sha256": _sha256(destination / "uv.lock"),
        }
    control = stage / "control"
    control.mkdir()
    for name in ("bench_ovr_hmd_measurement.py", "hmd_issue206_scenarios.py"):
        shutil.copy2(ROOT / "scripts" / name, control / name)
    native = stage / "runtime" / "run"
    native.mkdir(parents=True)
    shutil.copy2(executable, native / "PuriPulyHeartOverlay.exe")
    candidate = source / "candidate"
    shutil.copytree(candidate / "third_party" / "openvr", stage / "runtime" / "openvr-provenance")
    shutil.copytree(
        candidate / "third_party" / "noto-sans-cjk", stage / "runtime" / "noto-provenance"
    )
    shutil.copy2(
        candidate / "third_party" / "openvr" / "win64" / "openvr_api.dll", native / "openvr_api.dll"
    )
    shutil.copytree(
        candidate / "src" / "puripuly_heart" / "data", native / "puripuly_heart" / "data"
    )
    if _sha256(native / "openvr_api.dll") != VENDORED_DLL_SHA256:
        raise MeasurementError("vendored OpenVR DLL mismatch")
    staged_startup = _check_startup_contract(native / "PuriPulyHeartOverlay.exe")
    payload = {
        "schema": SCHEMA,
        "prepared_at": _utc_now(),
        "session": stage.name,
        "preparation_baseline": ARMS["candidate"],
        "sources": identities,
        "control": _tree_identity(control),
        "runtime": _tree_identity(stage / "runtime"),
        "interpreter": _interpreter_identity(),
        "startup": startup,
        "staged_startup": staged_startup,
        "native": {
            "sha256": SOURCE_EXE_SHA256,
            "shared": True,
            "production_equivalence": "pinned delta is test-only; complete native tree hashes differ",
            "build_receipt": "docs/issue-206-verification.md release artifact",
            "profile": "p05",
            "handoff": "off",
        },
        "scenarios": list(SCENARIOS),
        "anchors": list(ANCHORS),
        "preregistration": {
            "one_scenario_per_launch": True,
            "automatic_live_next": False,
            "baseline_ui_block_expected": ["independent"],
            "source_route_comparison_seconds": 1.0,
            "offline_comparison_seconds": 0.2,
            "synthetic_provider_only": True,
            "render_ack_or_source_only_dwell": False,
            "scoped_grace_ms": 400,
            "peer_gate_seconds": 1.0,
            "idle_ttl_seconds": 8.0,
            "short_stable_updates": 4,
            "sustained_cadence_seconds": 0.5,
            "sustained_default_seconds": 10,
            "sustained_max_seconds": 60,
            "sustained_max_updates": 120,
            "run_default_timeout_seconds": 90,
            "run_max_timeout_seconds": 120,
            "stop_poll_seconds": 0.1,
            "native_retry_bounds": {
                "cadence_ms": 100,
                "deadline_ms": 500,
                "stream_max": 4,
                "final_max": 5,
                "readiness_no_progress_ms": 2000,
            },
        },
        "boundary": "developer owner-composition harness, not installed application proof",
        "physical_hmd": "not_observed",
        "privacy": "synthetic public text only; no settings/audio/secrets/OSC sends",
    }
    _write_json(stage / "preparation.json", payload)
    (stage / "preparation.sha256").write_text(_sha256(stage / "preparation.json"), encoding="ascii")
    load_prepared_stage(stage)
    return stage


def load_prepared_stage(stage: Path) -> dict:
    stage = stage.resolve()
    try:
        report = stage / "preparation.json"
        if _sha256(report) != (stage / "preparation.sha256").read_text(encoding="ascii").strip():
            raise MeasurementError("preparation identity mismatch")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if (
            _sha256(Path(__file__).resolve())
            != payload["control"]["files"]["bench_ovr_hmd_measurement.py"]
        ):
            raise MeasurementError("launching harness identity mismatch; use the staged control")
        if payload["schema"] != SCHEMA or payload["interpreter"] != json.loads(
            json.dumps(_interpreter_identity())
        ):
            raise MeasurementError("schema or interpreter/dependency identity mismatch")
        for arm, revision in ARMS.items():
            identity = payload["sources"][arm]
            if (
                identity["revision"] != revision
                or _tree_identity(stage / "source" / arm) != identity["tree"]
                or _sha256(stage / "source" / f"{arm}.tar") != identity["archive_sha256"]
                or _sha256(stage / "source" / arm / "uv.lock") != identity["lock_sha256"]
            ):
                raise MeasurementError(f"source identity mismatch: {arm}")
        for directory in ("control", "runtime"):
            if _tree_identity(stage / directory) != payload[directory]:
                raise MeasurementError(f"{directory} identity mismatch")
        if payload["startup"]["stdout"] != EXPECTED_STARTUP_CONTRACT:
            raise MeasurementError("startup contract metadata mismatch")
        if _sha256(stage / "runtime/run/PuriPulyHeartOverlay.exe") != SOURCE_EXE_SHA256:
            raise MeasurementError("native identity mismatch")
        return payload
    except (OSError, ValueError, KeyError) as exc:
        raise MeasurementError(f"invalid prepared stage: {exc}") from exc


def inspect_process_names() -> set[str]:
    if os.name != "nt":
        raise MeasurementError("live process inspection unavailable")
    completed = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10, check=False
    )
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    if completed.returncode or not rows or any(len(row) < 5 for row in rows):
        raise MeasurementError("process inspection failed; live blocked")
    return {row[0].lower() for row in rows}


def validate_live_guard(process_names: set[str], *, confirmed_hmd_ready: bool) -> None:
    if not confirmed_hmd_ready:
        raise MeasurementError("live requires --confirm-hmd-ready")
    names = {name.lower() for name in process_names}
    if {"puripulyheartoverlay.exe", "puripuly-heart.exe", "puripulyheart.exe"} & names:
        raise MeasurementError("preexisting overlay/app already running; not stopped")
    if not {"vrserver.exe", "vrcompositor.exe"} <= names:
        raise MeasurementError("SteamVR not ready; start it yourself")


def request_stop(stage: Path, run_id: str) -> Path:
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id
    ):
        raise MeasurementError("invalid run ID")
    run = stage.resolve() / "runs" / run_id
    if not (run / "owned.json").is_file():
        raise MeasurementError("unknown owned run")
    path = run / "stop.request"
    path.write_text(_utc_now(), encoding="ascii")
    return path


def validate_sustained_guard(stage: Path, arm: str, anchor: str) -> None:
    successful = set()
    for path in (stage / "runs").glob("*/report.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        observation = path.with_name("observation.json")
        if (
            run.get("mode") != "live"
            or run.get("arm") != arm
            or run.get("anchor") != anchor
            or run.get("software", {}).get("outcome") != "pass"
            or not observation.is_file()
        ):
            continue
        observed = json.loads(observation.read_text(encoding="utf-8"))
        if observed.get("result") == "no_issue" and observed.get("run_report_sha256") == _sha256(
            path
        ):
            successful.add(run["scenario"])
    if not set(SCENARIOS[:-1]) <= successful:
        raise MeasurementError(
            "sustained live requires successful observed short checks for this arm/anchor"
        )


def environment_inventory() -> dict:
    result = {
        "os": platform.platform(),
        "gpu_driver": "unknown",
        "SteamVR_version": "unknown",
        "VRChat_version": "unknown",
    }
    if os.name != "nt":
        return result
    command = (
        "$g=Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion;"
        "$v=Get-Process vrserver -ErrorAction SilentlyContinue | ForEach-Object {$_.FileVersionInfo.FileVersion};"
        "$c=Get-Process VRChat -ErrorAction SilentlyContinue | ForEach-Object {$_.FileVersionInfo.FileVersion};"
        "@{gpu_driver=@($g);SteamVR_version=@($v);VRChat_version=@($c)} | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            parsed = json.loads(completed.stdout)
            result.update({key: value or "unknown" for key, value in parsed.items()})
    except OSError, ValueError, subprocess.SubprocessError:
        pass
    return result


def _record_pre_runtime_abort(run, run_id, arm, scenario, anchor, live, exc, phase, confirmed):
    _write_json(
        run / "report.json",
        {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "arm": arm,
            "scenario": scenario,
            "anchor": anchor,
            "mode": "live" if live else "offline_dry_run",
            "software": {
                "outcome": "failed",
                "failure_reason": (
                    "cancelled"
                    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                    else str(exc)
                ),
                "phase": phase,
                "cleanup": "complete" if confirmed else "unconfirmed",
                "native_startup": "not_started" if confirmed else "unknown",
            },
            "physical_hmd": {"result": "not_observed", "api_success_is_not_physical_pass": True},
        },
    )
    (run / "report.sha256").write_text(_sha256(run / "report.json"), encoding="ascii")


def run_owned_worker(args):
    run = args.stage / "runs" / args.run_id
    lock = LIVE_LOCK if args.live else args.stage / "active.lock"
    if (
        not (run / "owned.json").is_file()
        or not lock.is_file()
        or json.loads(lock.read_text(encoding="utf-8")).get("run_id") != args.run_id
    ):
        raise MeasurementError("run-correlated controller ownership required")
    runtime_entered = False
    try:
        load_prepared_stage(args.stage)
        source = args.stage.resolve() / "source" / args.arm
        sys.path[:0] = [str(source / "src"), str(source)]
        sys.modules["bench_ovr_hmd_measurement"] = sys.modules[__name__]
        from hmd_issue206_scenarios import run_measurement

        runtime_entered = True
        asyncio.run(run_measurement(args))
    except BaseException as exc:
        if not runtime_entered:
            _record_pre_runtime_abort(
                run,
                args.run_id,
                args.arm,
                args.scenario,
                args.anchor,
                args.live,
                exc,
                "worker_bootstrap",
                True,
            )
        raise


def launch_arm(
    stage: Path,
    *,
    arm: str,
    scenario: str,
    anchor: str,
    live: bool = False,
    confirmed_hmd_ready: bool = False,
    duration: float = 10.0,
    timeout: float = 90.0,
    device: str = "unknown",
    firmware: str = "unknown",
    connection: str = "unknown",
) -> Path:
    if arm not in ARMS or scenario not in SCENARIOS or anchor not in ANCHORS:
        raise MeasurementError("unknown arm/scenario/anchor")
    if (
        not math.isfinite(duration)
        or not 1 <= duration <= 60
        or not math.isfinite(timeout)
        or not 1 <= timeout <= 120
    ):
        raise MeasurementError("duration 1..60 and timeout 1..120 seconds required")
    if live:
        if not confirmed_hmd_ready:
            raise MeasurementError("live requires --confirm-hmd-ready")
        validate_live_guard(inspect_process_names(), confirmed_hmd_ready=True)
    stage = stage.resolve()
    load_prepared_stage(stage)
    if live and scenario == "sustained":
        validate_sustained_guard(stage, arm, anchor)
    run_id = f"{arm}-{scenario}-{secrets.token_hex(4)}"
    run = stage / "runs" / run_id
    run.mkdir(parents=True)
    print(f"Run ID: {run_id}\nRun report: {run / 'report.json'}", flush=True)
    lock = LIVE_LOCK if live else stage / "active.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise MeasurementError(
            "concurrent/stale owned run guard: inspect active.lock; never delete while owned runtime exists"
        ) from exc
    child = None
    spawn_attempted = False
    confirmed_no_worker = True
    try:
        os.close(descriptor)
        _write_json(lock, {"run_id": run_id, "controller_pid": os.getpid()})
        _write_json(
            run / "owned.json",
            {"run_id": run_id, "controller_pid": os.getpid(), "native_pid": "not_started"},
        )
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith("PURIPULY_") or key in ("PYTHONPATH", "PYTHONHOME"):
                env.pop(key)
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(stage / "source" / arm / "src")
                + os.pathsep
                + str(stage / "source" / arm),
                "LOCALAPPDATA": str(run / "localappdata"),
                "APPDATA": str(run / "appdata"),
                "TEMP": str(run / "tmp"),
                "TMP": str(run / "tmp"),
                "PURIPULY_OVERLAY_QUIET_TAIL_PROFILE": "p05",
                "PURIPULY_OVERLAY_HANDOFF_EXPERIMENT": "off",
            }
        )
        for directory in ("localappdata", "appdata", "tmp", "logs"):
            (run / directory).mkdir()
        args = [
            sys.executable,
            "-B",
            str(stage / "control" / "bench_ovr_hmd_measurement.py"),
            "internal-run",
            "--stage",
            str(stage),
            "--run-id",
            run_id,
            "--arm",
            arm,
            "--scenario",
            scenario,
            "--anchor",
            anchor,
            "--duration",
            str(duration),
            "--timeout",
            str(timeout),
            "--device",
            device,
            "--firmware",
            firmware,
            "--connection",
            connection,
        ]
        if live:
            args += ["--live", "--confirm-hmd-ready"]
        spawn_attempted = True
        confirmed_no_worker = False
        child = subprocess.Popen(args, env=env, cwd=run)
        try:
            child.wait(timeout=timeout + 30)
        except KeyboardInterrupt, subprocess.TimeoutExpired:
            request_stop(stage, run_id)
            child.wait(timeout=25)
            raise MeasurementError("interrupted/timeout; stop requested and owned worker exited")
        if child.returncode:
            raise MeasurementError(
                f"owned arm exited {child.returncode}; inspect {run / 'report.json'}"
            )
    except BaseException as exc:
        if child is None:
            confirmed_no_worker = not spawn_attempted or isinstance(exc, OSError)
            _record_pre_runtime_abort(
                run,
                run_id,
                arm,
                scenario,
                anchor,
                live,
                exc,
                "controller_before_spawn" if confirmed_no_worker else "spawn_unconfirmed",
                confirmed_no_worker,
            )
        raise
    finally:
        if child is None and confirmed_no_worker:
            lock.unlink(missing_ok=True)
        elif child is not None and child.poll() is not None and (run / "report.json").is_file():
            receipt = json.loads((run / "report.json").read_text(encoding="utf-8"))
            if (
                receipt.get("schema") == RUN_SCHEMA
                and receipt.get("run_id") == run_id
                and receipt.get("software", {}).get("cleanup") == "complete"
            ):
                lock.unlink(missing_ok=True)
    return run / "report.json"


def record_observation(run_report: Path, *, result: str, note: str, uncertainty: str) -> Path:
    if result not in PHYSICAL_RESULTS or not note.strip() or not uncertainty.strip():
        raise MeasurementError("observation requires supported result, note and uncertainty")
    run = json.loads(run_report.read_text(encoding="utf-8"))
    if run.get("schema") != RUN_SCHEMA or run.get("mode") != "live":
        raise MeasurementError("offline/unknown run cannot receive HMD observation")
    outcome = run["software"]["outcome"]
    path = run_report.with_name("observation.json")
    if path.exists():
        raise MeasurementError("observation already recorded; preserve original evidence")
    _write_json(
        path,
        {
            "schema": OBSERVATION_SCHEMA,
            "run_id": run["run_id"],
            "recorded_at": _utc_now(),
            "run_report_sha256": _sha256(run_report),
            "result": result,
            "qualitative_note": note,
            "uncertainty": uncertainty,
            "latency_claim": "not_measured",
            "software_run_outcome": outcome,
            "overall_outcome": "failed" if outcome != "pass" else "observation_recorded",
            "new_run_performed": False,
            "cannot_upgrade_failed_software_run": True,
        },
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opt-in issue206 isolated developer HMD kit; no automatic VR launch."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--executable", type=Path, default=Path("C:/pph206-native/release/PuriPulyHeartOverlay.exe")
    )
    prepare.add_argument("--stage", type=Path)
    inspect = sub.add_parser("inspect", aliases=["preflight"])
    inspect.add_argument("--stage", type=Path, required=True)
    for name in ("dry-run", "live", "internal-run"):
        command = sub.add_parser(name)
        command.add_argument("--stage", type=Path, required=True)
        command.add_argument("--arm", choices=ARMS, required=True)
        command.add_argument("--scenario", choices=SCENARIOS, required=True)
        command.add_argument("--anchor", choices=ANCHORS, default="head_locked")
        command.add_argument("--duration", type=float, default=10)
        command.add_argument("--timeout", type=float, default=90)
        command.add_argument("--confirm-hmd-ready", action="store_true")
        for field in ("device", "firmware", "connection"):
            command.add_argument("--" + field, default="unknown")
        if name == "internal-run":
            command.add_argument("--run-id", required=True)
            command.add_argument("--live", action="store_true")
    stop = sub.add_parser("stop")
    stop.add_argument("--stage", type=Path, required=True)
    stop.add_argument("--run-id", required=True)
    observe = sub.add_parser("observe")
    observe.add_argument("--run-report", type=Path, required=True)
    observe.add_argument("--result", choices=PHYSICAL_RESULTS, required=True)
    observe.add_argument("--note", required=True)
    observe.add_argument("--uncertainty", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            print(prepare_session(args.executable, args.stage))
        elif args.command in ("inspect", "preflight"):
            payload = load_prepared_stage(args.stage)
            try:
                names = inspect_process_names()
                readiness = {
                    "SteamVR_processes_present": {"vrserver.exe", "vrcompositor.exe"} <= names,
                    "preexisting_native_overlay": "puripulyheartoverlay.exe" in names,
                }
            except MeasurementError, OSError, subprocess.SubprocessError:
                readiness = "unknown; live launch will fail closed"
            print(
                json.dumps(
                    {
                        "identity": "verified",
                        "session": payload["session"],
                        "physical": "not_observed",
                        "vr_initialized": False,
                        "live_process_prerequisites": readiness,
                        "environment": environment_inventory(),
                    }
                )
            )
        elif args.command == "stop":
            print(request_stop(args.stage, args.run_id))
        elif args.command == "observe":
            print(
                record_observation(
                    args.run_report,
                    result=args.result,
                    note=args.note,
                    uncertainty=args.uncertainty,
                )
            )
        elif args.command == "internal-run":
            run_owned_worker(args)
        else:
            print(
                launch_arm(
                    args.stage,
                    arm=args.arm,
                    scenario=args.scenario,
                    anchor=args.anchor,
                    live=args.command == "live",
                    confirmed_hmd_ready=args.confirm_hmd_ready,
                    duration=args.duration,
                    timeout=args.timeout,
                    device=args.device,
                    firmware=args.firmware,
                    connection=args.connection,
                )
            )
        return 0
    except (MeasurementError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
