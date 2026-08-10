from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from experiments.speaker_representation_scd import execution_guard, r1_execute
from experiments.speaker_representation_scd.execution_guard import (
    MAX_CUMULATIVE_SECONDS,
    MAX_RESIDENT_RAM_GIB,
    ExecutionGuardError,
    ExecutionLease,
    load_completed_action_receipt,
    run_supervised,
    validate_worker_lease,
)
from experiments.speaker_representation_scd.provenance import (
    self_sha256_valid,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import REPOSITORY_ROOT
from experiments.speaker_representation_scd.windows_job import (
    MAX_JOB_MEMORY_BYTES,
    WindowsMemoryJob,
)


def test_execution_lease_is_exclusive_and_records_usage(tmp_path: Path) -> None:
    with ExecutionLease(tmp_path, "smoke", ("r1_execute", "smoke")) as lease:
        with pytest.raises(ExecutionGuardError, match="already exists"):
            with ExecutionLease(tmp_path, "models", ("r1_execute", "models")):
                pass
        run_supervised(
            lease,
            [sys.executable, "-c", "value = bytearray(1024 * 1024)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            poll_seconds=0.01,
        )
        lease.complete()
    assert not (tmp_path / "control" / "r1_execution.lock").exists()
    receipts = list((tmp_path / "control" / "usage").glob("*.json"))
    assert len(receipts) == 1
    document = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert self_sha256_valid(document)
    assert document["status"] == "completed"
    boundary = document["hard_memory_boundary"]
    assert boundary["applied"] is True
    assert boundary["mechanism"] == "windows_job_object_job_memory"
    assert 0 < boundary["authoritative_peak_job_memory_bytes"]
    assert boundary["authoritative_peak_job_memory_bytes"] <= boundary["contract_ceiling_bytes"]
    assert (
        sum(
            boundary[key]
            for key in (
                "enforced_job_memory_limit_bytes",
                "reserved_headroom_bytes",
                "preassignment_commit_bytes",
            )
        )
        == boundary["contract_ceiling_bytes"]
    )


def test_windows_job_hard_memory_limit_rejects_excess_allocation(
    tmp_path: Path,
) -> None:
    ceiling = 128 * 1024**2
    allocation_program = (
        "import sys\n"
        "values = []\n"
        "try:\n"
        "    while True:\n"
        "        values.append(bytearray(4 * 1024 * 1024))\n"
        "except MemoryError:\n"
        "    sys.exit(23)\n"
    )
    driver_program = (
        "import subprocess, sys\n"
        f"result = subprocess.run([sys.executable, '-c', {allocation_program!r}])\n"
        "sys.exit(result.returncode)\n"
    )
    job = WindowsMemoryJob(ceiling)
    try:
        process = job.launch(
            [sys.executable, "-c", driver_program],
            cwd=tmp_path,
            environment=os.environ.copy(),
        )
        assert process.wait(timeout=30) == 23
        deadline = time.monotonic() + 5
        while job.active_processes() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.active_processes() == 0
        assert 0 < job.peak_memory_bytes() <= ceiling
        assert job.effective_job_memory_limit_bytes is not None
        assert job.preassignment_commit_bytes is not None
        assert (
            job.effective_job_memory_limit_bytes
            + job.headroom_bytes
            + job.preassignment_commit_bytes
            == ceiling
        )
    finally:
        if job.active_processes() > 0:
            job.terminate()
        job.close()


def test_execution_lease_rejects_memory_before_work_and_removes_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution_guard,
        "_process_tree_rss",
        lambda _pids: (int((MAX_RESIDENT_RAM_GIB + 1) * 1024**3), ()),
    )
    with pytest.raises(ExecutionGuardError, match="exceeded 24 GiB"):
        with ExecutionLease(tmp_path, "smoke", ("r1_execute", "smoke")):
            pass
    assert not (tmp_path / "control" / "r1_execution.lock").exists()


def test_execution_lease_detects_memory_growth_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = ExecutionLease(tmp_path, "smoke", ("r1_execute", "smoke"))
    lease.__enter__()
    monkeypatch.setattr(
        execution_guard,
        "_process_tree_rss",
        lambda _pids: (int((MAX_RESIDENT_RAM_GIB + 1) * 1024**3), ()),
    )
    with pytest.raises(ExecutionGuardError, match="exceeded 24 GiB"):
        lease.check((os.getpid(),))
    lease.fail("memory ceiling")
    lease.__exit__(None, None, None)


def test_execution_lease_detects_legacy_process_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = ExecutionLease(tmp_path, "models", ("r1_execute", "models"))
    lease.__enter__()
    monkeypatch.setattr(
        execution_guard,
        "strict_legacy_scan",
        lambda: (({"pid": 77, "name": "python.exe", "module": "legacy"},), ()),
    )
    with pytest.raises(ExecutionGuardError, match="legacy contention"):
        lease.check((os.getpid(),))
    lease.fail("legacy contention")
    lease.__exit__(None, None, None)


def test_execution_lease_treats_process_inspection_failure_as_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution_guard,
        "strict_legacy_scan",
        lambda: ((), ({"pid": 91, "name": "python.exe", "reason": "AccessDenied"},)),
    )
    with pytest.raises(ExecutionGuardError, match="inspection failed"):
        with ExecutionLease(tmp_path, "models", ("r1_execute", "models")):
            pass


def test_execution_lease_enforces_action_wall_ceiling(tmp_path: Path) -> None:
    lease = ExecutionLease(
        tmp_path,
        "smoke",
        ("r1_execute", "smoke"),
        max_action_seconds=0.5,
    )
    lease.__enter__()
    lease.started_monotonic -= 1
    with pytest.raises(ExecutionGuardError, match="wall ceiling"):
        lease.check((os.getpid(),))
    lease.fail("wall ceiling")
    lease.__exit__(None, None, None)
    assert not (tmp_path / "control" / "r1_execution.lock").exists()


def test_direct_worker_without_supervisor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SRSCD_EXECUTION_LEASE_TOKEN", raising=False)
    with pytest.raises(ExecutionGuardError, match="supervised execution lease"):
        validate_worker_lease(tmp_path)


def test_live_supervised_worker_accepts_the_windows_venv_launcher_ancestor(
    tmp_path: Path,
) -> None:
    environment_key = "SRSCD_TEST_CACHE_ROOT"
    worker = (
        "import os; "
        "from pathlib import Path; "
        "from experiments.speaker_representation_scd.execution_guard import validate_worker_lease; "
        f"validate_worker_lease(Path(os.environ['{environment_key}']))"
    )
    with ExecutionLease(tmp_path, "smoke", ("r1_execute", "smoke")) as lease:
        environment = lease.worker_environment()
        environment[environment_key] = str(tmp_path)
        run_supervised(
            lease,
            [sys.executable, "-c", worker],
            cwd=REPOSITORY_ROOT,
            environment=environment,
            poll_seconds=0.01,
        )
        lease.complete()


def test_supervisor_terminates_worker_when_legacy_contention_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def scan():
        nonlocal calls
        calls += 1
        if calls >= 4:
            return (({"pid": 88, "name": "python.exe", "module": "legacy"},), ())
        return (), ()

    monkeypatch.setattr(execution_guard, "strict_legacy_scan", scan)
    with ExecutionLease(tmp_path, "smoke", ("r1_execute", "smoke")) as lease:
        with pytest.raises(ExecutionGuardError, match="legacy contention"):
            run_supervised(
                lease,
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=tmp_path,
                environment=os.environ.copy(),
                poll_seconds=0.01,
            )
        lease.fail("legacy contention")


def test_cumulative_wall_ledger_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "control" / "usage" / "prior.json"
    path.parent.mkdir(parents=True)
    payload = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r1_resource_usage",
            "execution_id": "0" * 32,
            "action": "models",
            "status": "aborted",
            "elapsed_seconds": MAX_CUMULATIVE_SECONDS,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExecutionGuardError, match="96-hour cumulative"):
        with ExecutionLease(tmp_path, "models", ("r1_execute", "models")):
            pass


def test_aborted_child_receipt_is_not_authoritative_and_retry_quarantines_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "manifests" / "r1_environment_sync.json"
    monkeypatch.setattr(r1_execute, "validated_cache_root", lambda _action: tmp_path)
    monkeypatch.setattr(r1_execute, "_worker_command", lambda *_args: ["worker"])

    def write_child_receipt(lease: ExecutionLease) -> None:
        assert lease.expected_receipt is not None
        payload = with_self_sha256(
            {
                "schema_version": 1,
                "artifact_role": "test_action_receipt",
                "supervision_binding": {
                    "execution_id": lease.token,
                    "expected_receipt_relative_path": (lease.expected_receipt_relative_path),
                    "authority": "requires_completed_usage_attestation",
                },
            }
        )
        lease.expected_receipt.parent.mkdir(parents=True, exist_ok=True)
        lease.expected_receipt.write_text(json.dumps(payload), encoding="utf-8")

    def aborted_supervisor(lease: ExecutionLease, *_args, **_kwargs) -> None:
        write_child_receipt(lease)
        raise ExecutionGuardError("final supervisor check failed")

    monkeypatch.setattr(r1_execute, "run_supervised", aborted_supervisor)
    with pytest.raises(ExecutionGuardError, match="final supervisor check failed"):
        r1_execute.execute("sync-environment", None, ("r1_execute", "sync-environment"))
    assert receipt.is_file()
    with pytest.raises(ExecutionGuardError, match="lacks a completed usage attestation"):
        load_completed_action_receipt(tmp_path, receipt, "sync-environment")

    def completed_supervisor(lease: ExecutionLease, *_args, **_kwargs) -> None:
        write_child_receipt(lease)
        lease.record_authoritative_job_memory(
            peak_bytes=1024,
            enforced_limit_bytes=MAX_JOB_MEMORY_BYTES - 1024**3,
            headroom_bytes=1024**3,
            preassignment_commit_bytes=0,
        )

    monkeypatch.setattr(r1_execute, "run_supervised", completed_supervisor)
    result = r1_execute.execute(
        "sync-environment",
        None,
        ("r1_execute", "sync-environment"),
    )
    assert result == receipt
    assert (
        load_completed_action_receipt(
            tmp_path,
            receipt,
            "sync-environment",
        )["artifact_role"]
        == "test_action_receipt"
    )
    orphaned = list((tmp_path / "control" / "orphans").glob("*-r1_environment_sync.json"))
    assert len(orphaned) == 1
    assert len(list((tmp_path / "control" / "orphans").glob("*.metadata.json"))) == 1

    def unexpected_supervisor(*_args, **_kwargs):
        raise AssertionError("authoritative evidence reached worker execution")

    monkeypatch.setattr(r1_execute, "run_supervised", unexpected_supervisor)
    with pytest.raises(ExecutionGuardError, match="completed evidence"):
        r1_execute.execute(
            "sync-environment",
            None,
            ("r1_execute", "sync-environment"),
        )
    mutated = json.loads(receipt.read_text(encoding="utf-8"))
    mutated["tampered"] = True
    receipt.write_text(json.dumps(with_self_sha256(mutated)), encoding="utf-8")
    with pytest.raises(ExecutionGuardError, match="hash binding differs"):
        load_completed_action_receipt(tmp_path, receipt, "sync-environment")
