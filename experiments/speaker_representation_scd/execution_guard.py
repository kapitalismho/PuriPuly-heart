from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import psutil

from experiments.speaker_representation_scd.provenance import (
    self_sha256_valid,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.windows_job import (
    MAX_JOB_MEMORY_BYTES,
    WindowsMemoryJob,
)

LEGACY_ARGUMENT = "experiments/speaker_turn_boundary"
LEGACY_MODULE = "experiments.speaker_turn_boundary"
POTENTIAL_HOSTS = {
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "python.exe",
    "pythonw.exe",
    "uv.exe",
}
MAX_RESIDENT_RAM_GIB = 24.0
MAX_ACTION_SECONDS = 24 * 60 * 60
MAX_CUMULATIVE_SECONDS = 96 * 60 * 60


class ExecutionGuardError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerExecution:
    execution_id: str
    requested_argv: tuple[str, ...]
    expected_receipt_relative_path: str | None


def strict_legacy_scan() -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    matches: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name"]):
        pid = int(process.info["pid"])
        name = str(process.info.get("name") or "")
        if name.lower() not in POTENTIAL_HOSTS:
            continue
        try:
            args = [str(value) for value in process.cmdline()]
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, psutil.ZombieProcess) as exc:
            failures.append({"pid": pid, "name": name, "reason": type(exc).__name__})
            continue
        module = None
        if "-m" in args:
            index = args.index("-m")
            if index + 1 < len(args):
                module = args[index + 1]
        script_match = any(LEGACY_ARGUMENT in arg.replace("\\", "/") for arg in args)
        module_match = isinstance(module, str) and module.startswith(LEGACY_MODULE)
        if script_match or module_match:
            matches.append({"pid": pid, "name": name, "module": module})
    return (
        tuple(sorted(matches, key=lambda item: item["pid"])),
        tuple(sorted(failures, key=lambda item: item["pid"])),
    )


def _exclusive_json(path: Path, document: dict[str, Any]) -> None:
    payload = with_self_sha256(document)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _usage_documents(control_root: Path) -> tuple[tuple[Path, dict[str, Any]], ...]:
    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((control_root / "usage").glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutionGuardError(f"cannot validate prior usage receipt {path}: {exc}") from exc
        if not isinstance(document, dict) or not self_sha256_valid(document):
            raise ExecutionGuardError(f"invalid prior usage receipt: {path}")
        if (
            document.get("schema_version") != 1
            or document.get("artifact_role") != "r1_resource_usage"
        ):
            raise ExecutionGuardError(f"invalid prior usage receipt contract: {path}")
        execution_id = document.get("execution_id")
        elapsed = document.get("elapsed_seconds")
        if (
            not isinstance(execution_id, str)
            or len(execution_id) != 32
            or any(character not in "0123456789abcdef" for character in execution_id)
            or not isinstance(document.get("action"), str)
            or document.get("status") not in {"completed", "aborted"}
            or not isinstance(elapsed, (int, float))
            or elapsed < 0
        ):
            raise ExecutionGuardError(f"invalid prior usage receipt fields: {path}")
        documents.append((path, document))
    return tuple(documents)


def _prior_usage_seconds(control_root: Path) -> float:
    total = 0.0
    for _, document in _usage_documents(control_root):
        elapsed = document["elapsed_seconds"]
        total += float(elapsed)
    return total


def _process_tree_rss(root_pids: Iterable[int]) -> tuple[int, tuple[int, ...]]:
    processes: dict[int, psutil.Process] = {}
    failures: set[int] = set()
    for pid in root_pids:
        try:
            root = psutil.Process(int(pid))
        except psutil.NoSuchProcess:
            continue
        processes[root.pid] = root
        try:
            for child in root.children(recursive=True):
                processes[child.pid] = child
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            failures.add(root.pid)
    total = 0
    for pid, process in processes.items():
        try:
            total += int(process.memory_info().rss)
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, psutil.ZombieProcess):
            failures.add(pid)
    return total, tuple(sorted(failures))


def terminate_process_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        children = root.children(recursive=True)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        children = []
    for process in reversed(children):
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    try:
        root.terminate()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    _, alive = psutil.wait_procs(children + [root], timeout=5)
    for process in alive:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass


@dataclass
class ExecutionLease:
    cache_root: Path
    action: str
    requested_argv: tuple[str, ...]
    expected_receipt: Path | None = None
    max_action_seconds: float = MAX_ACTION_SECONDS

    def __post_init__(self) -> None:
        self.cache_root = self.cache_root.resolve()
        self.control_root = self.cache_root / "control"
        self.lock_path = self.control_root / "r1_execution.lock"
        self.token = secrets.token_hex(16)
        self.expected_receipt_relative_path: str | None = None
        if self.expected_receipt is not None:
            self.expected_receipt = self.expected_receipt.resolve()
            try:
                relative = self.expected_receipt.relative_to(self.cache_root)
            except ValueError as exc:
                raise ExecutionGuardError(
                    "the action receipt must remain under the external cache root"
                ) from exc
            self.expected_receipt_relative_path = relative.as_posix()
        self.started_monotonic = time.monotonic()
        self.started_at_utc = datetime.now(UTC).isoformat()
        self.prior_usage_seconds = 0.0
        self.peak_rss_bytes = 0
        self.authoritative_peak_job_memory_bytes: int | None = None
        self.enforced_job_memory_limit_bytes: int | None = None
        self.job_memory_headroom_bytes: int | None = None
        self.preassignment_commit_bytes: int | None = None
        self.action_receipt: dict[str, Any] | None = None
        self._entered = False
        self._status = "aborted"
        self._failure_reason: str | None = "lease exited without completion"

    def __enter__(self) -> ExecutionLease:
        self.control_root.mkdir(parents=True, exist_ok=True)
        self.prior_usage_seconds = _prior_usage_seconds(self.control_root)
        if self.prior_usage_seconds >= MAX_CUMULATIVE_SECONDS:
            raise ExecutionGuardError("the 96-hour cumulative R1 ceiling is exhausted")
        matches, failures = strict_legacy_scan()
        if failures:
            raise ExecutionGuardError(f"legacy process inspection failed: {failures}")
        if matches:
            raise ExecutionGuardError(f"legacy contention detected: {matches}")
        lock = {
            "schema_version": 1,
            "artifact_role": "r1_execution_lease",
            "pid": os.getpid(),
            "action": self.action,
            "requested_argv": list(self.requested_argv),
            "expected_action_receipt_relative_path": (self.expected_receipt_relative_path),
            "started_at_utc": self.started_at_utc,
            "token": self.token,
        }
        try:
            _exclusive_json(self.lock_path, lock)
        except FileExistsError as exc:
            raise ExecutionGuardError(
                f"another R1 execution lease already exists: {self.lock_path}"
            ) from exc
        self._entered = True
        try:
            self.check((os.getpid(),))
        except BaseException:
            self.lock_path.unlink(missing_ok=True)
            self._entered = False
            raise
        return self

    def worker_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["SRSCD_EXECUTION_LEASE_TOKEN"] = self.token
        environment["SRSCD_REQUESTED_ARGV"] = json.dumps(list(self.requested_argv))
        if self.expected_receipt_relative_path is not None:
            environment["SRSCD_EXPECTED_ACTION_RECEIPT"] = self.expected_receipt_relative_path
        else:
            environment.pop("SRSCD_EXPECTED_ACTION_RECEIPT", None)
        return environment

    def check(self, root_pids: Iterable[int]) -> None:
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed > self.max_action_seconds:
            raise ExecutionGuardError(
                f"action exceeded its wall ceiling: {elapsed:.3f} > {self.max_action_seconds:.3f}"
            )
        if self.prior_usage_seconds + elapsed > MAX_CUMULATIVE_SECONDS:
            raise ExecutionGuardError("action exceeded the 96-hour cumulative R1 ceiling")
        matches, failures = strict_legacy_scan()
        if failures:
            raise ExecutionGuardError(f"legacy process inspection failed: {failures}")
        if matches:
            raise ExecutionGuardError(f"legacy contention detected: {matches}")
        rss_bytes, memory_failures = _process_tree_rss(root_pids)
        if memory_failures:
            raise ExecutionGuardError(
                f"R1 process memory inspection failed for PIDs: {memory_failures}"
            )
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        peak_gib = self.peak_rss_bytes / (1024**3)
        if peak_gib > MAX_RESIDENT_RAM_GIB:
            raise ExecutionGuardError(f"R1 process tree exceeded 24 GiB: {peak_gib:.6f}")

    def complete(self) -> None:
        accounting = (
            self.authoritative_peak_job_memory_bytes,
            self.enforced_job_memory_limit_bytes,
            self.job_memory_headroom_bytes,
            self.preassignment_commit_bytes,
        )
        if any(value is None for value in accounting):
            raise ExecutionGuardError(
                "R1 completion requires authoritative Windows Job memory accounting"
            )
        assert all(value is not None for value in accounting)
        if self.authoritative_peak_job_memory_bytes > MAX_JOB_MEMORY_BYTES:
            raise ExecutionGuardError("authoritative Windows Job memory exceeded 24 GiB")
        if sum(accounting[1:]) != MAX_JOB_MEMORY_BYTES:
            raise ExecutionGuardError("Windows Job memory boundary identity differs")
        if self.expected_receipt_relative_path is not None and self.action_receipt is None:
            raise ExecutionGuardError("R1 completion requires an exact action receipt binding")
        self._status = "completed"
        self._failure_reason = None

    def bind_action_receipt(self) -> None:
        if self.expected_receipt is None or self.expected_receipt_relative_path is None:
            raise ExecutionGuardError("the R1 action has no expected receipt")
        try:
            document = json.loads(self.expected_receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutionGuardError(f"cannot validate the R1 action receipt: {exc}") from exc
        if not isinstance(document, dict) or not self_sha256_valid(document):
            raise ExecutionGuardError("the R1 action receipt self identity is invalid")
        binding = document.get("supervision_binding")
        expected_binding = {
            "execution_id": self.token,
            "expected_receipt_relative_path": self.expected_receipt_relative_path,
            "authority": "requires_completed_usage_attestation",
        }
        if binding != expected_binding:
            raise ExecutionGuardError("the R1 action receipt lease binding differs")
        self.action_receipt = {
            "relative_path": self.expected_receipt_relative_path,
            "sha256": sha256_file(self.expected_receipt),
            "self_sha256": document["self_sha256"],
            "execution_id": self.token,
        }

    def record_authoritative_job_memory(
        self,
        *,
        peak_bytes: int,
        enforced_limit_bytes: int,
        headroom_bytes: int,
        preassignment_commit_bytes: int,
    ) -> None:
        values = (
            int(peak_bytes),
            int(enforced_limit_bytes),
            int(headroom_bytes),
            int(preassignment_commit_bytes),
        )
        if any(value < 0 for value in values):
            raise ExecutionGuardError("authoritative Windows Job memory peak is invalid")
        self.authoritative_peak_job_memory_bytes = max(
            self.authoritative_peak_job_memory_bytes or 0,
            values[0],
        )
        self.enforced_job_memory_limit_bytes = values[1]
        self.job_memory_headroom_bytes = values[2]
        self.preassignment_commit_bytes = values[3]

    def fail(self, reason: str) -> None:
        self._status = "aborted"
        self._failure_reason = reason

    def __exit__(self, exc_type, exc, _traceback) -> bool:
        if exc is not None:
            self.fail(f"{type(exc).__name__}: {exc}")
        elapsed = time.monotonic() - self.started_monotonic
        receipt = {
            "schema_version": 1,
            "artifact_role": "r1_resource_usage",
            "execution_id": self.token,
            "action": self.action,
            "requested_argv": list(self.requested_argv),
            "expected_action_receipt_relative_path": (self.expected_receipt_relative_path),
            "action_receipt": self.action_receipt,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "prior_usage_seconds": self.prior_usage_seconds,
            "cumulative_usage_seconds": self.prior_usage_seconds + elapsed,
            "peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "hard_memory_boundary": {
                "mechanism": "windows_job_object_job_memory",
                "contract_ceiling_bytes": MAX_JOB_MEMORY_BYTES,
                "enforced_job_memory_limit_bytes": (self.enforced_job_memory_limit_bytes),
                "reserved_headroom_bytes": self.job_memory_headroom_bytes,
                "preassignment_commit_bytes": self.preassignment_commit_bytes,
                "authoritative_peak_job_memory_bytes": (self.authoritative_peak_job_memory_bytes),
                "applied": self.authoritative_peak_job_memory_bytes is not None,
            },
            "status": self._status,
            "failure_reason": self._failure_reason,
        }
        usage_path = self.control_root / "usage" / f"{self.token}.json"
        if self._entered:
            _exclusive_json(usage_path, receipt)
            try:
                lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if lock.get("token") != self.token or not self_sha256_valid(lock):
                    raise ExecutionGuardError("R1 execution lease identity changed")
                self.lock_path.unlink()
            except BaseException as cleanup_error:
                if exc is None:
                    raise cleanup_error
        return False


def validate_worker_execution(
    cache_root: Path,
    expected_receipt: Path | None = None,
) -> WorkerExecution:
    token = os.environ.get("SRSCD_EXECUTION_LEASE_TOKEN")
    if not token:
        raise ExecutionGuardError("R1 worker requires a supervised execution lease")
    path = cache_root.resolve() / "control" / "r1_execution.lock"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionGuardError(f"cannot validate the R1 execution lease: {exc}") from exc
    if not isinstance(document, dict) or not self_sha256_valid(document):
        raise ExecutionGuardError("R1 execution lease self identity is invalid")
    if document.get("token") != token:
        raise ExecutionGuardError("R1 execution lease token differs")
    parent_pid = int(document.get("pid", -1))
    if parent_pid != os.getppid() or not psutil.pid_exists(parent_pid):
        raise ExecutionGuardError("R1 worker parent does not own the execution lease")
    requested = os.environ.get("SRSCD_REQUESTED_ARGV")
    try:
        values = json.loads(requested or "[]")
    except json.JSONDecodeError as exc:
        raise ExecutionGuardError("R1 requested argv is invalid") from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ExecutionGuardError("R1 requested argv is invalid")
    relative = document.get("expected_action_receipt_relative_path")
    if relative is not None and not isinstance(relative, str):
        raise ExecutionGuardError("R1 expected receipt binding is invalid")
    environment_relative = os.environ.get("SRSCD_EXPECTED_ACTION_RECEIPT")
    if environment_relative != relative:
        raise ExecutionGuardError("R1 expected receipt environment differs")
    if expected_receipt is not None:
        try:
            expected_relative = (
                expected_receipt.resolve().relative_to(cache_root.resolve()).as_posix()
            )
        except ValueError as exc:
            raise ExecutionGuardError(
                "R1 expected receipt must remain under the external cache root"
            ) from exc
        if relative != expected_relative:
            raise ExecutionGuardError("R1 worker receipt path differs from its lease")
    return WorkerExecution(
        execution_id=token,
        requested_argv=tuple(values),
        expected_receipt_relative_path=relative,
    )


def validate_worker_lease(cache_root: Path) -> tuple[str, ...]:
    return validate_worker_execution(cache_root).requested_argv


def _receipt_relative_path(cache_root: Path, receipt: Path) -> str:
    try:
        return receipt.resolve().relative_to(cache_root.resolve()).as_posix()
    except ValueError as exc:
        raise ExecutionGuardError(
            "the R1 action receipt must remain under the external cache root"
        ) from exc


def action_receipt_is_authoritative(
    cache_root: Path,
    receipt: Path,
    expected_action: str,
) -> bool:
    relative = _receipt_relative_path(cache_root, receipt)
    completed: list[dict[str, Any]] = []
    for _, usage in _usage_documents(cache_root.resolve() / "control"):
        claim = usage.get("action_receipt")
        if (
            usage.get("status") == "completed"
            and isinstance(claim, dict)
            and claim.get("relative_path") == relative
        ):
            completed.append(usage)
    if not completed:
        return False
    if len(completed) != 1:
        raise ExecutionGuardError("the R1 action receipt has multiple completed usage attestations")
    usage = completed[0]
    if usage.get("action") != expected_action or usage.get("failure_reason") is not None:
        raise ExecutionGuardError("the completed R1 usage action identity differs")
    if usage.get("expected_action_receipt_relative_path") != relative:
        raise ExecutionGuardError("the completed R1 expected receipt path differs")
    boundary = usage.get("hard_memory_boundary")
    if not isinstance(boundary, dict):
        raise ExecutionGuardError("the completed R1 usage lacks hard memory accounting")
    accounting = (
        boundary.get("enforced_job_memory_limit_bytes"),
        boundary.get("reserved_headroom_bytes"),
        boundary.get("preassignment_commit_bytes"),
    )
    peak = boundary.get("authoritative_peak_job_memory_bytes")
    if (
        boundary.get("mechanism") != "windows_job_object_job_memory"
        or boundary.get("contract_ceiling_bytes") != MAX_JOB_MEMORY_BYTES
        or boundary.get("applied") is not True
        or not all(isinstance(value, int) and value >= 0 for value in accounting)
        or not isinstance(peak, int)
        or peak < 0
        or peak > MAX_JOB_MEMORY_BYTES
        or sum(accounting) != MAX_JOB_MEMORY_BYTES
    ):
        raise ExecutionGuardError("the completed R1 hard memory attestation is invalid")
    try:
        document = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionGuardError(f"cannot load the completed R1 receipt: {exc}") from exc
    if not isinstance(document, dict) or not self_sha256_valid(document):
        raise ExecutionGuardError("the completed R1 action receipt self identity is invalid")
    execution_id = usage.get("execution_id")
    if (
        not isinstance(execution_id, str)
        or len(execution_id) != 32
        or any(character not in "0123456789abcdef" for character in execution_id)
    ):
        raise ExecutionGuardError("the completed R1 execution ID is invalid")
    claim = usage["action_receipt"]
    expected_claim = {
        "relative_path": relative,
        "sha256": sha256_file(receipt),
        "self_sha256": document["self_sha256"],
        "execution_id": execution_id,
    }
    if claim != expected_claim:
        raise ExecutionGuardError("the completed R1 action receipt hash binding differs")
    expected_binding = {
        "execution_id": execution_id,
        "expected_receipt_relative_path": relative,
        "authority": "requires_completed_usage_attestation",
    }
    if document.get("supervision_binding") != expected_binding:
        raise ExecutionGuardError("the completed R1 action receipt lease binding differs")
    return True


def load_completed_action_receipt(
    cache_root: Path,
    receipt: Path,
    expected_action: str,
) -> dict[str, Any]:
    if not receipt.is_file():
        raise ExecutionGuardError(f"completed R1 action receipt is missing: {receipt}")
    if not action_receipt_is_authoritative(cache_root, receipt, expected_action):
        raise ExecutionGuardError(
            f"R1 action receipt lacks a completed usage attestation: {receipt}"
        )
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def quarantine_orphan_action_receipt(cache_root: Path, receipt: Path) -> Path:
    relative = _receipt_relative_path(cache_root, receipt)
    byte_sha256 = sha256_file(receipt)
    orphan_root = cache_root.resolve() / "control" / "orphans"
    orphan_root.mkdir(parents=True, exist_ok=True)
    target = orphan_root / f"{time.time_ns()}-{receipt.name}"
    receipt.rename(target)
    metadata = {
        "schema_version": 1,
        "artifact_role": "r1_orphan_action_receipt",
        "quarantined_at_utc": datetime.now(UTC).isoformat(),
        "original_relative_path": relative,
        "quarantined_relative_path": target.relative_to(cache_root.resolve()).as_posix(),
        "sha256": byte_sha256,
        "reason": "no_unique_completed_usage_attestation",
    }
    _exclusive_json(target.with_name(f"{target.name}.metadata.json"), metadata)
    return target


def run_supervised(
    lease: ExecutionLease,
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    poll_seconds: float = 0.25,
) -> None:
    job = WindowsMemoryJob(MAX_JOB_MEMORY_BYTES)
    process: subprocess.Popen | None = None
    try:
        process = job.launch(command, cwd=cwd, environment=environment)
        while True:
            returncode = process.poll()
            active_processes = job.active_processes()
            if returncode is not None and returncode != 0:
                raise ExecutionGuardError(f"supervised R1 worker exited with code {returncode}")
            if returncode == 0 and active_processes > 0:
                raise ExecutionGuardError(
                    "supervised R1 worker exited while descendants remained active"
                )
            if returncode is not None and active_processes == 0:
                break
            lease.check((os.getpid(), process.pid))
            time.sleep(poll_seconds)
        lease.check((os.getpid(),))
    finally:
        try:
            if (
                job.effective_job_memory_limit_bytes is None
                or job.preassignment_commit_bytes is None
            ):
                raise ExecutionGuardError("Windows Job memory accounting was not initialized")
            lease.record_authoritative_job_memory(
                peak_bytes=job.peak_memory_bytes(),
                enforced_limit_bytes=job.effective_job_memory_limit_bytes,
                headroom_bytes=job.headroom_bytes,
                preassignment_commit_bytes=job.preassignment_commit_bytes,
            )
        finally:
            try:
                if job.active_processes() > 0:
                    job.terminate()
            finally:
                job.close()
