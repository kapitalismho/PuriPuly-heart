from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 900
MAX_TIMEOUT_SECONDS = 86400
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "INTERRUPTED"}


class ControlError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path, *, blocking: bool) -> Iterator[None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
    except OSError as exc:
        raise ControlError(f"cannot open lock {path}: {exc}") from exc
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(handle.fileno(), mode, 1)
        else:
            import fcntl

            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), flags)
    except OSError as exc:
        handle.close()
        raise ControlError(f"lock is already held: {path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ControlError(f"file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlError(f"JSON root must be an object: {path}")
    return value


def validate_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ControlError(f"{label} must be a non-empty path string")
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ControlError(f"{label} must stay beneath the run root")
    return value


def reject_unknown_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ControlError(f"{label} has unknown keys: {', '.join(unknown)}")


def validate_plan(value: dict[str, Any]) -> dict[str, Any]:
    reject_unknown_keys(value, {"schema_version", "stages"}, "plan")
    if type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ControlError(f"plan schema_version must be integer {SCHEMA_VERSION}")
    stages = value.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ControlError("plan stages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(stages):
        if not isinstance(raw, dict):
            raise ControlError(f"stage {index} must be an object")
        reject_unknown_keys(
            raw,
            {"id", "argv", "timeout_seconds", "required_outputs"},
            f"stage {index}",
        )
        stage_id = raw.get("id")
        if not isinstance(stage_id, str) or not stage_id or stage_id in seen:
            raise ControlError(f"stage {index} id must be non-empty and unique")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in stage_id
        ):
            raise ControlError(f"stage id has unsupported characters: {stage_id}")
        seen.add(stage_id)
        argv = raw.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item and "\x00" not in item for item in argv)
        ):
            raise ControlError(f"stage {stage_id} argv must contain non-empty strings")
        timeout = raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 1 <= timeout <= MAX_TIMEOUT_SECONDS
        ):
            raise ControlError(
                f"stage {stage_id} timeout_seconds must be an integer from 1 to {MAX_TIMEOUT_SECONDS}"
            )
        outputs = raw.get("required_outputs", [])
        if not isinstance(outputs, list):
            raise ControlError(f"stage {stage_id} required_outputs must be a list")
        normalized.append(
            {
                "id": stage_id,
                "argv": list(argv),
                "timeout_seconds": timeout,
                "required_outputs": [
                    validate_relative_path(item, f"stage {stage_id} output") for item in outputs
                ],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "stages": normalized}


def workflow_root() -> Path:
    return Path(__file__).resolve().parent


def default_plan_path() -> Path:
    return workflow_root() / "stage_plan.json"


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def substitutions(run_root: Path) -> dict[str, str]:
    root = workflow_root()
    return {
        "{workflow_root}": str(root),
        "{workflow_root_wsl}": windows_path_to_wsl(root),
        "{run_root}": str(run_root.resolve()),
        "{python}": sys.executable,
    }


def expand(text: str, values: dict[str, str]) -> str:
    for token, value in values.items():
        text = text.replace(token, value)
    return text


def paths(state_root: Path) -> dict[str, Path]:
    return {
        "root": state_root,
        "state": state_root / "state.json",
        "plan": state_root / "frozen_plan.json",
        "control_lock": state_root / "control.lock",
        "driver_lock": state_root / "driver.lock",
        "logs": state_root / "logs",
    }


def next_stage_id(state: dict[str, Any], plan: dict[str, Any]) -> str | None:
    completed = set(state["completed_stages"])
    for stage in plan["stages"]:
        if stage["id"] not in completed:
            return stage["id"]
    return None


def public_state(state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": state["run_id"],
        "status": state["status"],
        "current_stage": state["current_stage"],
        "completed_stages": state["completed_stages"],
        "next_stage": next_stage_id(state, plan),
        "pause_requested": state["pause_requested"],
        "plan_sha256": state["plan_sha256"],
        "last_error": state["last_error"],
        "last_recorded_child": state.get("last_recorded_child"),
        "interruption_warning": state.get("interruption_warning"),
    }


def load_bound(
    paths_by_name: dict[str, Path], requested_plan: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_object(paths_by_name["state"])
    frozen = validate_plan(load_object(paths_by_name["plan"]))
    frozen_digest = digest(frozen)
    if frozen_digest != state.get("plan_sha256"):
        raise ControlError("frozen plan identity differs from stored run state")
    if requested_plan is not None:
        supplied = validate_plan(load_object(requested_plan))
        if digest(supplied) != frozen_digest:
            raise ControlError("requested plan identity differs from the frozen plan for this run")
    expected_commands = {
        stage["id"]: digest(
            {
                "argv": stage["argv"],
                "timeout_seconds": stage["timeout_seconds"],
                "required_outputs": stage["required_outputs"],
            }
        )
        for stage in frozen["stages"]
    }
    if state.get("command_identities") != expected_commands:
        raise ControlError("stored command identities do not match the frozen plan")
    return state, frozen


def command_identities(plan: dict[str, Any]) -> dict[str, str]:
    return {
        stage["id"]: digest(
            {
                "argv": stage["argv"],
                "timeout_seconds": stage["timeout_seconds"],
                "required_outputs": stage["required_outputs"],
            }
        )
        for stage in plan["stages"]
    }


def initial_state(plan: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "created_at": now,
        "updated_at": now,
        "status": "READY",
        "pause_requested": False,
        "current_stage": None,
        "completed_stages": [],
        "plan_sha256": digest(plan),
        "command_identities": command_identities(plan),
        "stage_outcomes": [],
        "last_error": None,
        "last_recorded_child": None,
        "interruption_warning": None,
    }


def initialize(state_root: Path, plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    locations = paths(state_root)
    try:
        state_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ControlError(f"cannot create state root {state_root}: {exc}") from exc
    try:
        with file_lock(locations["control_lock"], blocking=True):
            requested = validate_plan(load_object(plan_path))
            state_exists = locations["state"].exists()
            frozen_exists = locations["plan"].exists()
            names = {item.name for item in state_root.iterdir()}
            if not state_exists and not frozen_exists:
                unexpected = sorted(names - {"control.lock", "driver.lock"})
                if unexpected:
                    raise ControlError(
                        "uninitialized state root has unexpected content: " + ", ".join(unexpected)
                    )
                state = initial_state(requested)
                atomic_json(locations["plan"], requested)
                atomic_json(locations["state"], state)
                return state, requested
            if not state_exists and frozen_exists:
                allowed = {"control.lock", "driver.lock", "frozen_plan.json"}
                unexpected = sorted(names - allowed)
                if unexpected:
                    raise ControlError(
                        "partial initialization has unexpected content: " + ", ".join(unexpected)
                    )
                frozen = validate_plan(load_object(locations["plan"]))
                if digest(requested) != digest(frozen):
                    raise ControlError(
                        "requested plan identity differs from recoverable frozen initialization"
                    )
                state = initial_state(frozen)
                atomic_json(locations["state"], state)
                return state, frozen
            raise ControlError(f"run state already exists or is inconsistent: {state_root}")
    except ControlError:
        raise
    except OSError as exc:
        raise ControlError(f"cannot initialize state root {state_root}: {exc}") from exc


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(path, state)


def receipt(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return {"path": str(path), "size_bytes": size, "sha256": hasher.hexdigest()}


def stop_timed_out_child(child: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False, capture_output=True
        )
    else:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    child.wait()


def execute_stage(
    stage: dict[str, Any],
    state_root: Path,
    locations: dict[str, Path],
    state: dict[str, Any],
) -> None:
    stage_id = stage["id"]
    values = substitutions(state_root)
    argv = [expand(item, values) for item in stage["argv"]]
    attempt_number = 1 + sum(1 for item in state["stage_outcomes"] if item["stage_id"] == stage_id)
    stdout_path = locations["logs"] / f"{stage_id}.attempt-{attempt_number}.stdout.log"
    stderr_path = locations["logs"] / f"{stage_id}.attempt-{attempt_number}.stderr.log"
    started_at = utc_now()
    timed_out = False
    launch_error: str | None = None
    with file_lock(locations["control_lock"], blocking=True):
        current, _ = load_bound(locations)
        if current["pause_requested"]:
            current["status"] = "PAUSED"
            current["current_stage"] = None
            save_state(locations["state"], current)
            state.clear()
            state.update(current)
            return
        current["status"] = "LAUNCHING"
        current["current_stage"] = stage_id
        current["last_error"] = None
        save_state(locations["state"], current)
        locations["logs"].mkdir(parents=True, exist_ok=True)
        stdout = stdout_path.open("wb")
        try:
            stderr = stderr_path.open("wb")
        except BaseException:
            stdout.close()
            raise
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            child = subprocess.Popen(
                argv,
                cwd=workflow_root(),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=os.name != "nt",
                creationflags=creationflags,
            )
        except OSError as exc:
            child = None
            launch_error = f"stage could not start: {exc}"
        if child is not None:
            current["status"] = "RUNNING"
            current["child"] = {
                "pid": child.pid,
                "stage_id": stage_id,
                "started_at": started_at,
            }
            save_state(locations["state"], current)
            state.clear()
            state.update(current)
    try:
        if child is None:
            return_code = None
        else:
            try:
                return_code = child.wait(timeout=stage["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_timed_out_child(child)
                return_code = child.returncode
    finally:
        stdout.close()
        stderr.close()
    finished_at = utc_now()
    outputs: list[dict[str, Any]] = []
    output_error: str | None = None
    if return_code == 0 and not timed_out:
        for relative in stage["required_outputs"]:
            output = (state_root / relative).resolve()
            try:
                output.relative_to(state_root.resolve())
            except ValueError:
                output_error = f"required output escaped run root: {relative}"
                break
            if not output.is_file():
                output_error = f"required output missing: {relative}"
                break
            try:
                outputs.append(receipt(output))
            except OSError as exc:
                output_error = f"required output cannot be read: {relative}: {exc}"
                break
    error = launch_error or ("stage timed out" if timed_out else output_error)
    if return_code not in {0, None} and error is None:
        error = f"stage exited with code {return_code}"
    outcome = {
        "stage_id": stage_id,
        "attempt": attempt_number,
        "command_sha256": state["command_identities"][stage_id],
        "started_at": started_at,
        "finished_at": finished_at,
        "return_code": return_code,
        "timed_out": timed_out,
        "status": "SUCCEEDED" if error is None else "FAILED",
        "stdout": receipt(stdout_path),
        "stderr": receipt(stderr_path),
        "outputs": outputs,
        "error": error,
    }
    with file_lock(locations["control_lock"], blocking=True):
        current, _ = load_bound(locations)
        current.pop("child", None)
        current["stage_outcomes"].append(outcome)
        current["current_stage"] = None
        if error is None:
            current["completed_stages"].append(stage_id)
            current["status"] = "PAUSED" if current["pause_requested"] else "READY"
        else:
            current["status"] = "FAILED"
            current["last_error"] = error
        save_state(locations["state"], current)
        state.clear()
        state.update(current)


def drive(state_root: Path, plan_path: Path, *, resume: bool) -> dict[str, Any]:
    locations = paths(state_root)
    with file_lock(locations["driver_lock"], blocking=False):
        with file_lock(locations["control_lock"], blocking=True):
            state, plan = load_bound(locations, plan_path)
            if state["status"] in {"RUNNING", "LAUNCHING"}:
                recorded_child = state.pop("child", None)
                if recorded_child is None:
                    recorded_child = {
                        "pid": None,
                        "stage_id": state["current_stage"],
                        "started_at": None,
                    }
                state["status"] = "INTERRUPTED"
                state["current_stage"] = None
                state["last_recorded_child"] = recorded_child
                state["interruption_warning"] = (
                    "the recorded child may still be running; its outputs are uncommitted and must not "
                    "be trusted until manually reconciled"
                )
                state["last_error"] = (
                    "prior driver ended with an unknown in-flight child outcome; implicit retry is refused"
                )
                save_state(locations["state"], state)
                raise ControlError(f"{state['last_error']}; {state['interruption_warning']}")
            if state["status"] == "FAILED":
                raise ControlError(
                    "failed stage blocks continuation; inspect retained logs and state"
                )
            if state["status"] == "INTERRUPTED":
                raise ControlError(state["last_error"])
            if resume:
                state["pause_requested"] = False
                state["status"] = "READY"
                save_state(locations["state"], state)
            elif state["status"] != "READY":
                raise ControlError(f"run cannot start from status {state['status']}")
        by_id = {stage["id"]: stage for stage in plan["stages"]}
        while True:
            with file_lock(locations["control_lock"], blocking=True):
                state, plan = load_bound(locations)
                next_id = next_stage_id(state, plan)
                if next_id is None:
                    state["status"] = "COMPLETED"
                    state["pause_requested"] = False
                    state["current_stage"] = None
                    save_state(locations["state"], state)
                    return public_state(state, plan)
                if state["pause_requested"]:
                    state["status"] = "PAUSED"
                    state["current_stage"] = None
                    save_state(locations["state"], state)
                    return public_state(state, plan)
            execute_stage(by_id[next_id], state_root, locations, state)
            if state["status"] in {"PAUSED", "FAILED"}:
                return public_state(state, plan)


def pause(state_root: Path, plan_path: Path | None = None) -> dict[str, Any]:
    locations = paths(state_root)
    with file_lock(locations["control_lock"], blocking=True):
        state, plan = load_bound(locations)
        if state["status"] == "COMPLETED":
            return public_state(state, plan)
        if state["status"] in {"FAILED", "INTERRUPTED"}:
            return public_state(state, plan)
        state["pause_requested"] = True
        if state["status"] == "READY":
            state["status"] = "PAUSED"
        save_state(locations["state"], state)
        return public_state(state, plan)


def status(state_root: Path, plan_path: Path | None = None) -> dict[str, Any]:
    locations = paths(state_root)
    with file_lock(locations["control_lock"], blocking=True):
        state, plan = load_bound(locations)
        return public_state(state, plan)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("run", "pause", "status", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--state-root", type=Path, required=True)
        command.add_argument("--plan", type=Path, default=default_plan_path())
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "run":
            initialize(args.state_root, args.plan)
            result = drive(args.state_root, args.plan, resume=False)
        elif args.command == "pause":
            result = pause(args.state_root, args.plan)
        elif args.command == "resume":
            result = drive(args.state_root, args.plan, resume=True)
        else:
            result = status(args.state_root, args.plan)
        print(json.dumps(result, sort_keys=True))
        return 1 if result["status"] == "FAILED" else 0
    except ControlError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
