from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HEARTBEAT_INTERVAL_SECONDS = 15.0
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
GIT_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
DECISION_KEYS = {
    "schema_version",
    "artifact_role",
    "run_id",
    "config_sha256",
    "gate_id",
    "action",
    "rationale",
    "created_at",
}
ARCHIVED_DECISION_KEYS = DECISION_KEYS | {"consumed_at"}
STATUS_VALUES = {
    "STARTING",
    "RUNNING",
    "WAITING_FOR_DECISION",
    "COMPLETED",
    "ERROR",
}


class ControlPlaneError(RuntimeError):
    pass


class DeadlineExceededError(ControlPlaneError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_absolute_deadline(value: object) -> datetime:
    if not isinstance(value, str):
        raise ControlPlaneError("absolute_deadline_utc must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ControlPlaneError("absolute_deadline_utc must be a canonical UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ControlPlaneError("absolute_deadline_utc must be a canonical UTC timestamp")
    normalized = parsed.astimezone(UTC)
    if parsed.utcoffset().total_seconds() != 0 or normalized.isoformat() != value:
        raise ControlPlaneError("absolute_deadline_utc must be a canonical UTC timestamp")
    return normalized


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(path, canonical_bytes(value))


def atomic_create_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ControlPlaneError(f"refusing to overwrite existing file: {path}") from exc
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path, value: object) -> None:
    atomic_create_bytes(path, canonical_bytes(value))


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlPlaneError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ControlPlaneError(f"JSON value must be an object: {path}")
    return value


def append_event(path: Path, event: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(payload + "\n")
        output.flush()
        os.fsync(output.fileno())


def append_event_once(path: Path, event: Mapping[str, object]) -> None:
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or SHA256_PATTERN.fullmatch(event_id) is None:
        raise ControlPlaneError("idempotent event requires an exact event_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b""
    if path.exists():
        with path.open("r+b") as stream:
            payload = stream.read()
            if payload and not payload.endswith(b"\n"):
                boundary = payload.rfind(b"\n") + 1
                stream.seek(boundary)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                payload = payload[:boundary]
        for raw_line in payload.splitlines():
            try:
                existing = json.loads(raw_line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ControlPlaneError("events log contains a malformed record") from exc
            if isinstance(existing, dict) and existing.get("event_id") == event_id:
                if existing != dict(event):
                    raise ControlPlaneError("events log contains a conflicting event_id")
                return
    append_event(path, event)


def contains_runpod_api_key(value: object) -> bool:
    if isinstance(value, str):
        return "RUNPOD_API_KEY" in value.upper()
    if isinstance(value, Mapping):
        return any(
            contains_runpod_api_key(key) or contains_runpod_api_key(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(contains_runpod_api_key(item) for item in value)
    return False


def reject_runpod_api_key_environment() -> None:
    if "RUNPOD_API_KEY" in os.environ:
        raise ControlPlaneError("RUNPOD_API_KEY must not be present in the Pod environment")


def artifact_path(run_root: Path, raw: object, field: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ControlPlaneError(f"{field} must be a nonempty run-relative path")
    resolved = (run_root / raw).resolve()
    if not resolved.is_relative_to(run_root.resolve()):
        raise ControlPlaneError(f"{field} escapes the run root")
    return resolved


def require_string_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ControlPlaneError(f"{field} must be a nonempty string array")
    return list(value)


def validate_artifacts(value: object, field: str, require_hash: bool) -> None:
    if not isinstance(value, list):
        raise ControlPlaneError(f"{field} must be an array")
    paths: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ControlPlaneError(f"{field}[{index}] must be an object")
        path = row.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ControlPlaneError(f"{field}[{index}].path must be run-relative")
        if path in paths:
            raise ControlPlaneError(f"{field} contains duplicate path {path}")
        paths.add(path)
        expected = row.get("sha256")
        if require_hash and (
            not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None
        ):
            raise ControlPlaneError(f"{field}[{index}].sha256 must be exact")
        if expected is not None and (
            not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None
        ):
            raise ControlPlaneError(f"{field}[{index}].sha256 is invalid")


def validate_config(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 2:
        raise ControlPlaneError("run config schema_version must be 2")
    parse_absolute_deadline(value.get("absolute_deadline_utc"))
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ControlPlaneError("run_id is invalid")
    candidate_git_head = value.get("candidate_git_head")
    if (
        not isinstance(candidate_git_head, str)
        or GIT_HEAD_PATTERN.fullmatch(candidate_git_head) is None
    ):
        raise ControlPlaneError(
            "candidate_git_head must be exactly 40 lowercase hexadecimal characters"
        )
    for field in ("persistent_root", "repository_root"):
        raw = value.get(field)
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ControlPlaneError(f"{field} must be an absolute path")
    phases = value.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ControlPlaneError("phases must be a nonempty array")
    phase_ids: list[str] = []
    gate_ids: set[str] = set()
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ControlPlaneError(f"phases[{index}] must be an object")
        phase_id = phase.get("id")
        if not isinstance(phase_id, str) or RUN_ID_PATTERN.fullmatch(phase_id) is None:
            raise ControlPlaneError(f"phases[{index}].id is invalid")
        if phase_id in phase_ids:
            raise ControlPlaneError(f"duplicate phase id: {phase_id}")
        phase_ids.append(phase_id)
        require_string_list(phase.get("argv"), f"phases[{index}].argv")
        environment = phase.get("environment", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in environment.items()
        ):
            raise ControlPlaneError(f"phases[{index}].environment must contain strings")
        validate_artifacts(
            phase.get("required_inputs", []), f"phases[{index}].required_inputs", True
        )
        validate_artifacts(
            phase.get("required_outputs", []), f"phases[{index}].required_outputs", False
        )
        gate = phase.get("decision_gate_after")
        if gate is not None:
            gate_id = gate.get("id") if isinstance(gate, dict) else None
            if (
                not isinstance(gate_id, str)
                or RUN_ID_PATTERN.fullmatch(gate_id) is None
                or gate_id in gate_ids
            ):
                raise ControlPlaneError(f"phases[{index}].decision_gate_after is invalid")
            gate_ids.add(gate_id)
            actions = gate.get("actions")
            if not isinstance(actions, dict) or not actions:
                raise ControlPlaneError(f"phases[{index}] gate actions must be nonempty")
            if any(
                not isinstance(action, str)
                or not action
                or (target is not None and not isinstance(target, str))
                for action, target in actions.items()
            ):
                raise ControlPlaneError(f"phases[{index}] gate actions are invalid")
    known = set(phase_ids)
    first_phase = value.get("first_phase", phase_ids[0])
    if first_phase not in known:
        raise ControlPlaneError("first_phase does not name a configured phase")
    for phase in phases:
        next_phase = phase.get("next_phase")
        if next_phase is not None and next_phase not in known:
            raise ControlPlaneError(f"unknown next_phase: {next_phase}")
        gate = phase.get("decision_gate_after")
        if gate is not None:
            for target in gate["actions"].values():
                if target is not None and target not in known:
                    raise ControlPlaneError(f"gate action names unknown phase: {target}")
    if contains_runpod_api_key(value):
        raise ControlPlaneError("RUNPOD_API_KEY must not appear in Pod run configuration")


def run_root_for(config: Mapping[str, object]) -> Path:
    return (
        Path(str(config["persistent_root"])).resolve()
        / "issue-107"
        / "runs"
        / str(config["run_id"])
    )


def expand(value: str, config: Mapping[str, object], run_root: Path) -> str:
    return value.replace("{run_root}", str(run_root)).replace(
        "{repository_root}", str(config["repository_root"])
    )


def state_paths(run_root: Path) -> dict[str, Path]:
    control = run_root / "control"
    return {
        "control": control,
        "config": control / "run_config.json",
        "state": control / "state.json",
        "heartbeat": control / "heartbeat.json",
        "events": control / "events.jsonl",
        "decision": control / "decision.json",
        "decisions": control / "decisions",
        "phase_complete": control / "phase-complete",
        "lock": control / "runner.lock",
        "logs": run_root / "logs",
    }


def initialize(config_source: Path) -> tuple[dict[str, Any], Path, dict[str, Path], dict[str, Any]]:
    config = load_json_object(config_source)
    validate_config(config)
    run_root = run_root_for(config)
    paths = state_paths(run_root)
    for name in ("control", "decisions", "phase_complete", "logs"):
        paths[name].mkdir(parents=True, exist_ok=True)
    config_payload = canonical_bytes(config)
    config_sha256 = sha256_bytes(config_payload)
    if paths["config"].exists():
        if paths["config"].read_bytes() != config_payload:
            raise ControlPlaneError("durable run config differs from the requested config")
    else:
        atomic_write_bytes(paths["config"], config_payload)
    if paths["state"].exists():
        state = load_json_object(paths["state"])
        if state.get("config_sha256") != config_sha256:
            raise ControlPlaneError("state is bound to a different run config")
    else:
        now = utc_now()
        phases = config["phases"]
        state = {
            "schema_version": 1,
            "run_id": config["run_id"],
            "config_sha256": config_sha256,
            "status": "STARTING",
            "active_phase": None,
            "next_phase": config.get("first_phase", phases[0]["id"]),
            "completed_phases": [],
            "waiting_gate": None,
            "error": None,
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        atomic_write_json(paths["state"], state)
        append_event(
            paths["events"],
            {"at": now, "event": "run_initialized", "config_sha256": config_sha256},
        )
    return config, run_root, paths, state


def acquire_lock(path: Path, replace_stale: bool) -> str:
    if replace_stale and path.exists():
        path.unlink()
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ControlPlaneError("runner lock already exists") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump({"pid": os.getpid(), "token": token, "created_at": utc_now()}, output)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return token


def release_lock(path: Path, token: str) -> None:
    try:
        current = load_json_object(path)
    except ControlPlaneError:
        return
    if current.get("token") == token:
        path.unlink(missing_ok=True)


def write_state(paths: Mapping[str, Path], state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    if state.get("status") not in STATUS_VALUES:
        raise ControlPlaneError("attempted to write an invalid control-plane status")
    atomic_write_json(paths["state"], state)


def raise_if_heartbeat_failed(errors: queue.SimpleQueue[BaseException]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    raise ControlPlaneError("heartbeat write failed") from error


def heartbeat_loop(
    paths: Mapping[str, Path],
    stop: threading.Event,
    ready: threading.Event,
    errors: queue.SimpleQueue[BaseException],
    writer: Callable[[Path, object], None],
    interval_seconds: float,
) -> None:
    sequence = 0
    while True:
        try:
            state = load_json_object(paths["state"])
            heartbeat = {
                "schema_version": 1,
                "run_id": state["run_id"],
                "config_sha256": state["config_sha256"],
                "status": state["status"],
                "active_phase": state.get("active_phase"),
                "sequence": sequence,
                "pid": os.getpid(),
                "updated_at": utc_now(),
            }
            writer(paths["heartbeat"], heartbeat)
            ready.set()
        except BaseException as exc:
            errors.put(exc)
            ready.set()
            stop.set()
            return
        sequence += 1
        if stop.wait(interval_seconds):
            return


def kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0 and process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        kill_process_group(process)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    grace_deadline = time.monotonic() + 5.0
    while time.monotonic() < grace_deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def verify_inputs(run_root: Path, rows: object) -> list[dict[str, object]]:
    result = []
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        path = artifact_path(run_root, row.get("path"), f"required_inputs[{index}].path")
        if not path.is_file():
            raise ControlPlaneError(f"required input is absent: {path}")
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise ControlPlaneError(f"required input hash mismatch: {path}")
        result.append({"path": str(path), "sha256": observed, "size": path.stat().st_size})
    return result


def prepare_outputs(run_root: Path, rows: object) -> list[tuple[Path, str | None]]:
    result = []
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        path = artifact_path(run_root, row.get("path"), f"required_outputs[{index}].path")
        if path.exists():
            raise ControlPlaneError(f"required output already exists before phase start: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        result.append((path, row.get("sha256")))
    return result


def verify_outputs(rows: list[tuple[Path, str | None]]) -> list[dict[str, object]]:
    result = []
    for path, expected in rows:
        if not path.is_file():
            raise ControlPlaneError(f"required output is absent after phase completion: {path}")
        observed = sha256_file(path)
        if expected is not None and observed != expected:
            raise ControlPlaneError(f"required output hash mismatch: {path}")
        result.append({"path": str(path), "sha256": observed, "size": path.stat().st_size})
    return result


def phase_by_id(config: Mapping[str, object], phase_id: str) -> dict[str, Any]:
    for phase in config["phases"]:
        if phase["id"] == phase_id:
            return phase
    raise ControlPlaneError(f"phase is not configured: {phase_id}")


def default_next_phase(config: Mapping[str, object], phase_id: str) -> str | None:
    phase_ids = [phase["id"] for phase in config["phases"]]
    index = phase_ids.index(phase_id)
    return phase_ids[index + 1] if index + 1 < len(phase_ids) else None


def run_phase(
    config: dict[str, Any],
    run_root: Path,
    paths: dict[str, Path],
    state: dict[str, Any],
    phase: dict[str, Any],
    heartbeat_errors: queue.SimpleQueue[BaseException],
) -> None:
    raise_if_heartbeat_failed(heartbeat_errors)
    deadline = parse_absolute_deadline(config.get("absolute_deadline_utc"))
    if datetime.now(UTC) >= deadline:
        raise DeadlineExceededError("immutable billing deadline reached before phase start")
    phase_id = phase["id"]
    if phase_id in state["completed_phases"]:
        raise ControlPlaneError(f"refusing to execute completed phase again: {phase_id}")
    inputs = verify_inputs(run_root, phase.get("required_inputs", []))
    outputs = prepare_outputs(run_root, phase.get("required_outputs", []))
    argv = [expand(item, config, run_root) for item in phase["argv"]]
    working_directory = Path(
        expand(phase.get("cwd", "{repository_root}"), config, run_root)
    ).resolve()
    if not working_directory.is_dir():
        raise ControlPlaneError(f"phase working directory is absent: {working_directory}")
    environment = os.environ.copy()
    if "RUNPOD_API_KEY" in environment:
        raise ControlPlaneError("RUNPOD_API_KEY must not be present in the Pod environment")
    environment.update(
        {
            key: expand(value, config, run_root)
            for key, value in phase.get("environment", {}).items()
        }
    )
    stdout_path = paths["logs"] / f"{phase_id}.stdout.log"
    stderr_path = paths["logs"] / f"{phase_id}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise ControlPlaneError(f"phase log already exists: {phase_id}")
    state.update(
        {
            "status": "RUNNING",
            "active_phase": phase_id,
            "next_phase": phase_id,
            "waiting_gate": None,
            "error": None,
        }
    )
    write_state(paths, state)
    started_at = utc_now()
    append_event(
        paths["events"],
        {
            "at": started_at,
            "event": "phase_started",
            "phase_id": phase_id,
            "argv_sha256": sha256_bytes(canonical_bytes(argv)),
        },
    )
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        try:
            while process.poll() is None:
                raise_if_heartbeat_failed(heartbeat_errors)
                remaining_seconds = (deadline - datetime.now(UTC)).total_seconds()
                if remaining_seconds <= 0:
                    kill_process_group(process)
                    raise DeadlineExceededError(
                        "immutable billing deadline reached during active phase"
                    )
                try:
                    process.wait(timeout=min(0.1, remaining_seconds))
                except subprocess.TimeoutExpired:
                    pass
            if datetime.now(UTC) >= deadline:
                kill_process_group(process)
                raise DeadlineExceededError(
                    "phase completion occurred at or after the immutable billing deadline"
                )
            raise_if_heartbeat_failed(heartbeat_errors)
        except BaseException:
            terminate_process_group(process)
            raise
        return_code = process.returncode
    if return_code is None:
        raise ControlPlaneError(f"phase {phase_id} process state is unavailable")
    if return_code != 0:
        raise ControlPlaneError(f"phase {phase_id} exited with code {return_code}")
    verified_outputs = verify_outputs(outputs)
    raise_if_heartbeat_failed(heartbeat_errors)
    marker = {
        "schema_version": 1,
        "artifact_role": "detached_phase_completion",
        "run_id": config["run_id"],
        "config_sha256": state["config_sha256"],
        "phase_id": phase_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "return_code": return_code,
        "argv_sha256": sha256_bytes(canonical_bytes(argv)),
        "inputs": inputs,
        "outputs": verified_outputs,
        "stdout": {
            "path": str(stdout_path),
            "sha256": sha256_file(stdout_path),
            "size": stdout_path.stat().st_size,
        },
        "stderr": {
            "path": str(stderr_path),
            "sha256": sha256_file(stderr_path),
            "size": stderr_path.stat().st_size,
        },
    }
    atomic_write_json(paths["phase_complete"] / f"{phase_id}.json", marker)
    state["completed_phases"].append(phase_id)
    append_event(
        paths["events"],
        {"at": utc_now(), "event": "phase_completed", "phase_id": phase_id},
    )
    gate = phase.get("decision_gate_after")
    if gate is not None:
        state.update(
            {
                "status": "WAITING_FOR_DECISION",
                "active_phase": None,
                "next_phase": None,
                "waiting_gate": {
                    "id": gate["id"],
                    "after_phase": phase_id,
                    "actions": sorted(gate["actions"]),
                    "entered_at": utc_now(),
                },
            }
        )
        write_state(paths, state)
        append_event(
            paths["events"],
            {"at": utc_now(), "event": "decision_gate_entered", "gate_id": gate["id"]},
        )
        return
    next_phase = phase.get("next_phase", default_next_phase(config, phase_id))
    if next_phase is None:
        state.update(
            {
                "status": "COMPLETED",
                "active_phase": None,
                "next_phase": None,
                "waiting_gate": None,
                "completed_at": utc_now(),
            }
        )
    else:
        state.update(
            {
                "status": "STARTING",
                "active_phase": None,
                "next_phase": next_phase,
                "waiting_gate": None,
            }
        )
    write_state(paths, state)


def decision_archive_path(paths: Mapping[str, Path], gate_id: str) -> Path:
    if RUN_ID_PATTERN.fullmatch(gate_id) is None:
        raise ControlPlaneError("decision gate id is not safe for archival")
    return paths["decisions"] / f"{gate_id}-decision.json"


def validate_operator_decision(
    decision: dict[str, Any],
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    gate: Mapping[str, Any],
    configured_gate: Mapping[str, Any],
) -> None:
    if set(decision) != DECISION_KEYS:
        raise ControlPlaneError("decision schema is invalid")
    created_at = decision.get("created_at")
    try:
        created = parse_absolute_deadline(created_at)
    except ControlPlaneError as exc:
        raise ControlPlaneError("decision timestamp is invalid") from exc
    if (
        decision.get("schema_version") != 1
        or decision.get("artifact_role") != "detached_operator_decision"
        or decision.get("run_id") != config["run_id"]
        or decision.get("config_sha256") != state["config_sha256"]
        or decision.get("gate_id") != gate["id"]
        or decision.get("action") not in configured_gate["actions"]
        or not isinstance(decision.get("rationale"), str)
        or not decision["rationale"].strip()
        or created > datetime.now(UTC)
        or contains_runpod_api_key(decision)
    ):
        raise ControlPlaneError("decision does not match the active gate")


def operator_decision_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if not DECISION_KEYS <= set(value):
        raise ControlPlaneError("archived decision schema is invalid")
    return {key: value[key] for key in DECISION_KEYS}


def validate_archived_decision(
    archived: dict[str, Any],
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    gate: Mapping[str, Any],
    configured_gate: Mapping[str, Any],
) -> dict[str, Any]:
    if set(archived) != ARCHIVED_DECISION_KEYS:
        raise ControlPlaneError("archived decision schema is invalid")
    decision = operator_decision_payload(archived)
    validate_operator_decision(decision, config, state, gate, configured_gate)
    try:
        consumed = parse_absolute_deadline(archived.get("consumed_at"))
        created = parse_absolute_deadline(decision["created_at"])
    except ControlPlaneError as exc:
        raise ControlPlaneError("archived decision timestamp is invalid") from exc
    if consumed < created or consumed > datetime.now(UTC):
        raise ControlPlaneError("archived decision timestamp is invalid")
    return decision


def decision_event(
    archived: Mapping[str, Any], event: str, archive_name: str | None = None
) -> dict[str, object]:
    decision = operator_decision_payload(archived)
    decision_sha256 = sha256_bytes(canonical_bytes(decision))
    at = archived["consumed_at"] if event == "decision_consumed" else decision["created_at"]
    value: dict[str, object] = {
        "at": at,
        "event": event,
        "gate_id": decision["gate_id"],
        "action": decision["action"],
        "decision_sha256": decision_sha256,
    }
    if archive_name is not None:
        value["decision_file"] = archive_name
    value["event_id"] = sha256_bytes(canonical_bytes(value))
    return value


def consume_decision(config: dict[str, Any], paths: dict[str, Path], state: dict[str, Any]) -> bool:
    if state["status"] != "WAITING_FOR_DECISION":
        return True
    gate = state.get("waiting_gate")
    if not isinstance(gate, dict):
        raise ControlPlaneError("waiting state has no gate")
    phase = phase_by_id(config, gate["after_phase"])
    configured_gate = phase["decision_gate_after"]
    archive = decision_archive_path(paths, gate["id"])
    candidates = sorted(paths["decisions"].glob(f"{gate['id']}-*.json"))
    if any(candidate != archive for candidate in candidates):
        raise ControlPlaneError("conflicting archived decisions exist for the active gate")
    live_exists = paths["decision"].is_file()
    archive_exists = archive.is_file()
    if not live_exists and not archive_exists:
        return False
    live_decision = None
    if live_exists:
        live_decision = load_json_object(paths["decision"])
        validate_operator_decision(live_decision, config, state, gate, configured_gate)
    if archive_exists:
        archived = load_json_object(archive)
        decision = validate_archived_decision(archived, config, state, gate, configured_gate)
        if live_decision is not None and live_decision != decision:
            raise ControlPlaneError("live and archived decisions conflict")
    else:
        if live_decision is None:
            raise ControlPlaneError("decision journal is unavailable")
        archived = {**live_decision, "consumed_at": utc_now()}
        atomic_create_json(archive, archived)
        decision = validate_archived_decision(archived, config, state, gate, configured_gate)
    if paths["decision"].exists():
        current_live = load_json_object(paths["decision"])
        if current_live != decision:
            raise ControlPlaneError("live and archived decisions conflict")
        paths["decision"].unlink()
        fsync_directory(paths["control"])
    append_event_once(paths["events"], decision_event(archived, "decision_recorded"))
    append_event_once(
        paths["events"],
        decision_event(archived, "decision_consumed", archive.name),
    )
    action = decision["action"]
    target = configured_gate["actions"][action]
    if target is None:
        state.update(
            {
                "status": "COMPLETED",
                "active_phase": None,
                "next_phase": None,
                "waiting_gate": None,
                "completed_at": utc_now(),
            }
        )
    else:
        state.update(
            {
                "status": "STARTING",
                "active_phase": None,
                "next_phase": target,
                "waiting_gate": None,
            }
        )
    write_state(paths, state)
    return True


def execute(
    config_source: Path,
    replace_stale_lock: bool = False,
    *,
    heartbeat_writer: Callable[[Path, object], None] = atomic_write_json,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    reject_runpod_api_key_environment()
    if heartbeat_interval_seconds <= 0:
        raise ControlPlaneError("heartbeat interval must be positive")
    config, run_root, paths, state = initialize(config_source)
    token = acquire_lock(paths["lock"], replace_stale_lock)
    heartbeat_stop = threading.Event()
    heartbeat_ready = threading.Event()
    heartbeat_errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()
    heartbeat = threading.Thread(
        target=heartbeat_loop,
        args=(
            paths,
            heartbeat_stop,
            heartbeat_ready,
            heartbeat_errors,
            heartbeat_writer,
            heartbeat_interval_seconds,
        ),
        daemon=True,
    )
    heartbeat.start()
    try:
        if not heartbeat_ready.wait(timeout=5.0):
            raise ControlPlaneError("heartbeat did not initialize")
        raise_if_heartbeat_failed(heartbeat_errors)
        state = load_json_object(paths["state"])
        if state["status"] == "COMPLETED":
            return state
        if state["status"] == "ERROR":
            raise ControlPlaneError("run is in ERROR state")
        if state["status"] == "RUNNING":
            raise ControlPlaneError("interrupted RUNNING phase requires a new run id")
        if not consume_decision(config, paths, state):
            raise_if_heartbeat_failed(heartbeat_errors)
            return load_json_object(paths["state"])
        state = load_json_object(paths["state"])
        while state["status"] not in {"WAITING_FOR_DECISION", "COMPLETED"}:
            raise_if_heartbeat_failed(heartbeat_errors)
            phase_id = state.get("next_phase")
            if not isinstance(phase_id, str):
                raise ControlPlaneError("nonterminal state has no next phase")
            run_phase(
                config,
                run_root,
                paths,
                state,
                phase_by_id(config, phase_id),
                heartbeat_errors,
            )
            state = load_json_object(paths["state"])
        raise_if_heartbeat_failed(heartbeat_errors)
        return state
    except BaseException as exc:
        state = load_json_object(paths["state"])
        state.update(
            {
                "status": "ERROR",
                "active_phase": None,
                "next_phase": None,
                "waiting_gate": None,
                "error": {"type": type(exc).__name__, "message": str(exc), "at": utc_now()},
            }
        )
        write_state(paths, state)
        append_event(
            paths["events"],
            {"at": utc_now(), "event": "run_error", "error_type": type(exc).__name__},
        )
        raise
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2.0)
        release_lock(paths["lock"], token)


def write_decision(run_root: Path, gate_id: str, action: str, rationale: str) -> dict[str, Any]:
    reject_runpod_api_key_environment()
    paths = state_paths(run_root.resolve())
    token = acquire_lock(paths["lock"], False)
    try:
        state = load_json_object(paths["state"])
        config = load_json_object(paths["config"])
        validate_config(config)
        config_sha256 = sha256_bytes(canonical_bytes(config))
        if state.get("config_sha256") != config_sha256:
            raise ControlPlaneError("state is bound to a different run config")
        gate = state.get("waiting_gate")
        if state.get("status") != "WAITING_FOR_DECISION" or not isinstance(gate, dict):
            raise ControlPlaneError("run is not waiting for a decision")
        if gate.get("id") != gate_id or action not in gate.get("actions", []):
            raise ControlPlaneError("decision is not allowed at the active gate")
        if not rationale.strip():
            raise ControlPlaneError("decision rationale is required")
        archive = decision_archive_path(paths, gate_id)
        archived = sorted(paths["decisions"].glob(f"{gate_id}-*.json"))
        if paths["decision"].exists() or archive.exists() or archived:
            raise ControlPlaneError("a decision already exists for the active gate")
        decision = {
            "schema_version": 1,
            "artifact_role": "detached_operator_decision",
            "run_id": state["run_id"],
            "config_sha256": state["config_sha256"],
            "gate_id": gate_id,
            "action": action,
            "rationale": rationale,
            "created_at": utc_now(),
        }
        phase = phase_by_id(config, gate["after_phase"])
        validate_operator_decision(
            decision,
            config,
            state,
            gate,
            phase["decision_gate_after"],
        )
        atomic_create_json(paths["decision"], decision)
        append_event_once(paths["events"], decision_event(decision, "decision_recorded"))
        return decision
    finally:
        release_lock(paths["lock"], token)


def self_test_config(
    persistent_root: Path, run_id: str, phases: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "persistent_root": str(persistent_root),
        "repository_root": str(Path.cwd().resolve()),
        "candidate_git_head": "0" * 40,
        "absolute_deadline_utc": "2999-01-01T00:00:00+00:00",
        "phases": phases,
    }


def expect_control_plane_error(action: Callable[[], object], expected_message: str) -> str:
    try:
        action()
    except ControlPlaneError as exc:
        message = str(exc)
        if expected_message not in message:
            raise ControlPlaneError(f"self-test received unexpected error: {message}") from exc
        return message
    raise ControlPlaneError(f"self-test expected rejection: {expected_message}")


def _self_test_without_runpod_api_key() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="issue-107-control-plane-") as temporary:
        persistent_root = Path(temporary).resolve()
        run_id = "self-test"
        config = self_test_config(
            persistent_root,
            run_id,
            [
                {
                    "id": "first",
                    "argv": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path(r'{run_root}/receipts/first.txt').write_text('first', encoding='utf-8')",
                    ],
                    "required_outputs": [{"path": "receipts/first.txt"}],
                    "decision_gate_after": {
                        "id": "after-first",
                        "actions": {"continue": "second", "stop": None},
                    },
                },
                {
                    "id": "second",
                    "argv": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path(r'{run_root}/receipts/second.txt').write_text('second', encoding='utf-8')",
                    ],
                    "required_outputs": [{"path": "receipts/second.txt"}],
                },
            ],
        )
        config_path = persistent_root / "config.json"
        atomic_write_json(config_path, config)
        first = execute(config_path)
        if first["status"] != "WAITING_FOR_DECISION":
            raise ControlPlaneError("self-test did not enter its decision gate")
        run_root = run_root_for(config)
        write_decision(run_root, "after-first", "continue", "exercise resume path")
        final = execute(config_path)
        if final["status"] != "COMPLETED" or final["completed_phases"] != [
            "first",
            "second",
        ]:
            raise ControlPlaneError("self-test did not complete both phases")
        return {
            "first_status": first["status"],
            "final_status": final["status"],
            "completed_phases": final["completed_phases"],
        }


def _self_test_failure_paths() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="issue-107-control-failures-") as temporary:
        persistent_root = Path(temporary).resolve()
        configured_key = self_test_config(
            persistent_root,
            "configured-key",
            [
                {
                    "id": "phase",
                    "argv": [sys.executable, "-c", "pass"],
                    "environment": {"RUNPOD_API_KEY": "forbidden"},
                }
            ],
        )
        expect_control_plane_error(
            lambda: validate_config(configured_key),
            "RUNPOD_API_KEY must not appear",
        )

        inherited_key = self_test_config(
            persistent_root,
            "inherited-key",
            [{"id": "phase", "argv": [sys.executable, "-c", "pass"]}],
        )
        inherited_path = persistent_root / "inherited-key.json"
        atomic_write_json(inherited_path, inherited_key)

        completed_key = self_test_config(
            persistent_root,
            "completed-key",
            [{"id": "phase", "argv": [sys.executable, "-c", "pass"]}],
        )
        completed_path = persistent_root / "completed-key.json"
        atomic_write_json(completed_path, completed_key)
        if execute(completed_path)["status"] != "COMPLETED":
            raise ControlPlaneError("API-key terminal-state fixture did not complete")

        waiting_key = self_test_config(
            persistent_root,
            "waiting-key",
            [
                {
                    "id": "phase",
                    "argv": [sys.executable, "-c", "pass"],
                    "decision_gate_after": {
                        "id": "waiting-gate",
                        "actions": {"stop": None},
                    },
                }
            ],
        )
        waiting_path = persistent_root / "waiting-key.json"
        atomic_write_json(waiting_path, waiting_key)
        if execute(waiting_path)["status"] != "WAITING_FOR_DECISION":
            raise ControlPlaneError("API-key waiting-state fixture did not reach its gate")

        os.environ["RUNPOD_API_KEY"] = "self-test-secret"
        try:
            for candidate_path in (inherited_path, completed_path, waiting_path):
                expect_control_plane_error(
                    lambda candidate_path=candidate_path: execute(candidate_path),
                    "RUNPOD_API_KEY must not be present",
                )
        finally:
            os.environ.pop("RUNPOD_API_KEY", None)
        if run_root_for(inherited_key).exists():
            raise ControlPlaneError("inherited API key rejection mutated a new run")
        if (
            load_json_object(state_paths(run_root_for(completed_key))["state"])["status"]
            != "COMPLETED"
        ):
            raise ControlPlaneError("terminal API key rejection mutated completed state")
        if (
            load_json_object(state_paths(run_root_for(waiting_key))["state"])["status"]
            != "WAITING_FOR_DECISION"
        ):
            raise ControlPlaneError("waiting API key rejection mutated waiting state")

        stale_output = self_test_config(
            persistent_root,
            "stale-output",
            [
                {
                    "id": "phase",
                    "argv": [sys.executable, "-c", "pass"],
                    "required_outputs": [{"path": "receipts/output.txt"}],
                }
            ],
        )
        stale_path = persistent_root / "stale-output.json"
        atomic_write_json(stale_path, stale_output)
        replayed_output = run_root_for(stale_output) / "receipts" / "output.txt"
        replayed_output.parent.mkdir(parents=True, exist_ok=True)
        replayed_output.write_text("replayed", encoding="utf-8")
        expect_control_plane_error(
            lambda: execute(stale_path),
            "required output already exists before phase start",
        )

        interrupted = self_test_config(
            persistent_root,
            "interrupted-running",
            [{"id": "phase", "argv": [sys.executable, "-c", "pass"]}],
        )
        interrupted_path = persistent_root / "interrupted-running.json"
        atomic_write_json(interrupted_path, interrupted)
        _, _, interrupted_paths, interrupted_state = initialize(interrupted_path)
        interrupted_state.update(
            {"status": "RUNNING", "active_phase": "phase", "next_phase": "phase"}
        )
        write_state(interrupted_paths, interrupted_state)
        expect_control_plane_error(
            lambda: execute(interrupted_path),
            "interrupted RUNNING phase requires a new run id",
        )
        if load_json_object(interrupted_paths["state"])["status"] != "ERROR":
            raise ControlPlaneError("interrupted RUNNING rejection was not durable")

        heartbeat_failure = self_test_config(
            persistent_root,
            "heartbeat-failure",
            [
                {
                    "id": "phase",
                    "argv": [
                        sys.executable,
                        "-c",
                        "import time; from pathlib import Path; Path(r'{run_root}/child-started.txt').write_text('started', encoding='utf-8'); time.sleep(30); Path(r'{run_root}/child-finished.txt').write_text('finished', encoding='utf-8')",
                    ],
                    "required_outputs": [{"path": "child-finished.txt"}],
                }
            ],
        )
        heartbeat_path = persistent_root / "heartbeat-failure.json"
        atomic_write_json(heartbeat_path, heartbeat_failure)
        heartbeat_root = run_root_for(heartbeat_failure)
        started_path = heartbeat_root / "child-started.txt"
        finished_path = heartbeat_root / "child-finished.txt"

        def failing_heartbeat_writer(path: Path, value: object) -> None:
            if path.name == "heartbeat.json" and started_path.is_file():
                raise OSError("injected heartbeat write failure")
            atomic_write_json(path, value)

        started_at = time.monotonic()
        expect_control_plane_error(
            lambda: execute(
                heartbeat_path,
                heartbeat_writer=failing_heartbeat_writer,
                heartbeat_interval_seconds=0.01,
            ),
            "heartbeat write failed",
        )
        elapsed = time.monotonic() - started_at
        heartbeat_state = load_json_object(state_paths(heartbeat_root)["state"])
        if (
            elapsed >= 15.0
            or not started_path.is_file()
            or finished_path.exists()
            or heartbeat_state["status"] != "ERROR"
        ):
            raise ControlPlaneError("heartbeat failure did not terminate the child fail-closed")
        return {
            "configured_api_key_rejected": True,
            "inherited_api_key_rejected_before_mutation": True,
            "completed_state_api_key_rejected_without_mutation": True,
            "waiting_state_api_key_rejected_without_mutation": True,
            "stale_output_rejected": True,
            "interrupted_running_rejected": True,
            "heartbeat_failure_killed_child": True,
            "heartbeat_failure_recorded_error": True,
            "heartbeat_failure_elapsed_seconds": round(elapsed, 3),
        }


def self_test() -> dict[str, object]:
    api_key = os.environ.pop("RUNPOD_API_KEY", None)
    try:
        happy_path = _self_test_without_runpod_api_key()
        failure_paths = _self_test_failure_paths()
        return {"passed": True, **happy_path, "failure_paths": failure_paths}
    finally:
        if api_key is not None:
            os.environ["RUNPOD_API_KEY"] = api_key


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--replace-stale-lock", action="store_true")
    decision_parser = commands.add_parser("decide")
    decision_parser.add_argument("--run-root", type=Path, required=True)
    decision_parser.add_argument("--gate", required=True)
    decision_parser.add_argument("--action", required=True)
    decision_parser.add_argument("--rationale", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-root", type=Path, required=True)
    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command != "self-test":
        reject_runpod_api_key_environment()
    if args.command == "run":
        result = execute(args.config, args.replace_stale_lock)
    elif args.command == "decide":
        result = write_decision(args.run_root, args.gate, args.action, args.rationale)
    elif args.command == "status":
        result = load_json_object(state_paths(args.run_root.resolve())["state"])
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
