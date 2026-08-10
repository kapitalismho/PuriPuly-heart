from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from experiments.speaker_representation_scd.execution_guard import (
    load_completed_action_receipt,
    validate_worker_execution,
    validate_worker_lease,
)
from experiments.speaker_representation_scd.provenance import (
    load_json,
    self_sha256_valid,
    sha256_bytes,
    sha256_file,
    verify_file_identity,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import (
    EXPERIMENT_ROOT,
    GATE_PATH,
    SOURCE_REGISTRY_PATH,
    R1GateError,
    validated_cache_root,
)
from experiments.speaker_representation_scd.run_provenance import run_provenance
from experiments.speaker_representation_scd.windows_job import MAX_JOB_MEMORY_BYTES

EXPECTED_RUNTIME = {
    "huggingface-hub": "0.31.4",
    "matplotlib": "3.10.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "psutil": "7.0.0",
    "pyarrow": "20.0.0",
    "pyyaml": "6.0.2",
    "safetensors": "0.5.3",
    "scikit-learn": "1.6.1",
    "soundfile": "0.13.1",
    "torch": "2.7.1+cpu",
    "torchaudio": "2.7.1+cpu",
    "transformers": "4.52.3",
}


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise R1GateError(f"refusing to overwrite an existing R1 artifact: {path}")
    payload = with_self_sha256(document)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _verify_files(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        path = root / row["path"]
        errors.extend(verify_file_identity(path, row["sha256"], row.get("size_bytes")))
        if path.is_file():
            verified.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if errors:
        raise R1GateError("; ".join(errors))
    return verified


def _verify_existing_files(root: Path, rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for row in rows:
        path = root / row["path"]
        if path.exists():
            errors.extend(verify_file_identity(path, row["sha256"], row.get("size_bytes")))
    if errors:
        raise R1GateError("; ".join(errors))


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    errors: list[str] = []
    for distribution, expected in EXPECTED_RUNTIME.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing distribution: {distribution}")
            continue
        versions[distribution] = actual
        if actual != expected:
            errors.append(f"{distribution}: {actual} != {expected}")
    if sys.version_info[:3] != (3, 12, 10):
        errors.append(f"python: {sys.version.split()[0]} != 3.12.10")
    if errors:
        raise R1GateError("; ".join(errors))
    return versions


def sync_environment(cache_root: Path, requested_argv: tuple[str, ...]) -> dict[str, Any]:
    validated_cache_root("environment_sync")
    receipt_path = cache_root / "manifests" / "r1_environment_sync.json"
    execution = validate_worker_execution(cache_root, receipt_path)
    if execution.requested_argv != requested_argv:
        raise R1GateError("R1 environment worker invocation differs from its lease")
    if receipt_path.exists():
        raise R1GateError(f"refusing to rerun environment sync: {receipt_path}")
    cache_root.mkdir(parents=True, exist_ok=True)
    project = EXPERIMENT_ROOT / "environment"
    uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    if uv_version != "uv 0.9.17 (2b5d65e61 2025-12-09)":
        raise R1GateError(f"uv version mismatch: {uv_version}")
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(cache_root / "uv")
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    _run(
        [
            "uv",
            "sync",
            "--project",
            str(project),
            "--frozen",
            "--no-python-downloads",
            "--python",
            str(EXPERIMENT_ROOT.parents[1] / ".venv" / "Scripts" / "python.exe"),
        ],
        cwd=EXPERIMENT_ROOT.parents[1],
        env=environment,
    )
    python = project / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise R1GateError(f"research interpreter was not created: {python}")
    output = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.metadata,json,sys; names="
            + repr(list(EXPECTED_RUNTIME))
            + "; print(json.dumps({'python':sys.version.split()[0],'packages':{n:importlib.metadata.version(n) for n in names}},sort_keys=True))",
        ],
        cwd=EXPERIMENT_ROOT.parents[1],
        text=True,
    )
    observed = json.loads(output)
    observed["uv"] = uv_version
    expected = {
        "python": "3.12.10",
        "uv": "uv 0.9.17 (2b5d65e61 2025-12-09)",
        "packages": EXPECTED_RUNTIME,
    }
    if observed != expected:
        raise R1GateError(f"locked environment mismatch: {observed!r}")
    observed["executable"] = str(python.resolve())
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    receipt = {
        "schema_version": 1,
        "artifact_role": "r1_environment_sync_receipt",
        "experiment_id": "speaker_representation_scd_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "supervision_binding": {
            "execution_id": execution.execution_id,
            "expected_receipt_relative_path": (execution.expected_receipt_relative_path),
            "authority": "requires_completed_usage_attestation",
        },
        "r1_gate_sha256": sha256_file(gate_path),
        "r1_gate_self_sha256": gate["self_sha256"],
        "execution_code_manifest_sha256": gate["execution_code"]["manifest_sha256"],
        "environment_contract": gate["environment"],
        "observed_environment": observed,
        "run_provenance": run_provenance(
            EXPERIMENT_ROOT.parents[1],
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=False,
        ),
    }
    _write_json(receipt_path, receipt)
    return load_json(receipt_path)


def _acquire_huggingface(model: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    target = cache_root / "models" / model["model_id"] / model["revision"]
    target.parent.mkdir(parents=True, exist_ok=True)
    _verify_existing_files(target, model["required_files"])
    snapshot_download(
        repo_id=model["repository"].removeprefix("https://huggingface.co/"),
        revision=model["revision"],
        allow_patterns=[row["path"] for row in model["required_files"]],
        local_dir=target,
        cache_dir=cache_root / "huggingface",
        max_workers=1,
    )
    files = _verify_files(target, model["required_files"])
    return {
        "model_id": model["model_id"],
        "repository": model["repository"],
        "revision": model["revision"],
        "root": str(target.resolve()),
        "files": files,
    }


def _git_output_lines(target: Path, *args: str) -> tuple[str, ...]:
    output = subprocess.check_output(["git", *args], cwd=target, text=True)
    return tuple(line for line in output.splitlines() if line)


def _is_exact_empty_no_checkout(target: Path) -> bool:
    if any(entry.name != ".git" for entry in target.iterdir()):
        return False
    head_paths = tuple(sorted(_git_output_lines(target, "ls-tree", "-r", "--name-only", "HEAD")))
    if not head_paths:
        return False
    index_paths = _git_output_lines(target, "ls-files")
    unstaged_paths = _git_output_lines(target, "diff", "--name-only")
    untracked_paths = _git_output_lines(target, "ls-files", "--others", "--exclude-standard")
    cached_paths = tuple(sorted(_git_output_lines(target, "diff", "--cached", "--name-only")))
    deleted_paths = tuple(
        sorted(
            _git_output_lines(
                target,
                "diff",
                "--cached",
                "--diff-filter=D",
                "--name-only",
            )
        )
    )
    return (
        not index_paths
        and not unstaged_paths
        and not untracked_paths
        and cached_paths == head_paths
        and deleted_paths == head_paths
    )


def _active_git_processes() -> tuple[dict[str, Any], ...]:
    matches: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name"]):
        name = str(process.info.get("name") or "").lower()
        if name not in {"git.exe", "git-lfs.exe"}:
            continue
        try:
            command = [str(value) for value in process.cmdline()]
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, psutil.ZombieProcess) as exc:
            failures.append(
                {
                    "pid": int(process.info["pid"]),
                    "name": name,
                    "reason": type(exc).__name__,
                }
            )
            continue
        matches.append(
            {
                "pid": int(process.info["pid"]),
                "name": name,
                "command": command,
            }
        )
    if failures:
        raise R1GateError(f"Git process inspection failed: {failures}")
    return tuple(sorted(matches, key=lambda row: row["pid"]))


def _usage_document(path: Path) -> dict[str, Any]:
    try:
        document = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise R1GateError(f"cannot validate R1 usage receipt {path}: {exc}") from exc
    execution_id = document.get("execution_id")
    elapsed = document.get("elapsed_seconds")
    if (
        not self_sha256_valid(document)
        or document.get("schema_version") != 1
        or document.get("artifact_role") != "r1_resource_usage"
        or not isinstance(execution_id, str)
        or len(execution_id) != 32
        or any(character not in "0123456789abcdef" for character in execution_id)
        or path.name != f"{execution_id}.json"
        or not isinstance(document.get("action"), str)
        or document.get("status") not in {"completed", "aborted"}
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
    ):
        raise R1GateError(f"invalid R1 usage receipt contract: {path}")
    return document


def _aborted_model_usage(path: Path) -> dict[str, Any]:
    document = _usage_document(path)
    if (
        document.get("action") != "models"
        or document.get("status") != "aborted"
        or document.get("action_receipt") is not None
        or document.get("expected_action_receipt_relative_path")
        != "manifests/r1_model_acquisition.json"
        or not isinstance(document.get("failure_reason"), str)
        or not document["failure_reason"]
    ):
        raise R1GateError(f"usage receipt is not an aborted model action: {path}")
    boundary = document.get("hard_memory_boundary")
    if not isinstance(boundary, dict):
        raise R1GateError(f"aborted R1 usage lacks hard Job accounting: {path}")
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
        or sum(accounting) != MAX_JOB_MEMORY_BYTES
        or not isinstance(peak, int)
        or peak < 0
        or peak > MAX_JOB_MEMORY_BYTES
    ):
        raise R1GateError(f"aborted R1 usage has invalid hard Job accounting: {path}")
    try:
        started = datetime.fromisoformat(str(document["started_at_utc"]))
        completed = datetime.fromisoformat(str(document["completed_at_utc"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise R1GateError(f"aborted R1 usage timestamps are invalid: {path}") from exc
    if started.tzinfo is None or completed.tzinfo is None or completed < started:
        raise R1GateError(f"aborted R1 usage timestamps are invalid: {path}")
    return document


def _aborted_usage_for_git_lock(cache_root: Path, lock_mtime: datetime) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    usage_root = cache_root / "control" / "usage"
    for path in sorted(usage_root.glob("*.json")):
        document = _usage_document(path)
        if document.get("action") != "models" or document.get("status") != "aborted":
            continue
        document = _aborted_model_usage(path)
        started = datetime.fromisoformat(document["started_at_utc"])
        completed = datetime.fromisoformat(document["completed_at_utc"])
        if started <= lock_mtime <= completed:
            matches.append({"path": path, "document": document})
    if len(matches) != 1:
        raise R1GateError(
            f"stale Git lock does not map to exactly one aborted model action: {len(matches)}"
        )
    return matches[0]


def _recovery_receipt_path(cache_root: Path, relative_lock: str, source_id: str) -> Path:
    path_hash = sha256_bytes(relative_lock.encode("utf-8"))[:16]
    return cache_root / "control" / "recoveries" / f"{source_id}-{path_hash}.json"


def _load_recovery_receipt(path: Path, cache_root: Path) -> dict[str, Any]:
    try:
        document = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise R1GateError(f"cannot validate Git lock recovery receipt {path}: {exc}") from exc
    if (
        not self_sha256_valid(document)
        or document.get("schema_version") != 1
        or document.get("artifact_role") != "r1_git_index_lock_recovery_authorization"
        or document.get("expected_action_receipt_relative_path")
        != "manifests/r1_model_acquisition.json"
    ):
        raise R1GateError(f"invalid Git lock recovery receipt contract: {path}")
    source = document.get("source_usage")
    lock = document.get("lock")
    authorizer = document.get("recovery_authorized_execution_id")
    if not isinstance(source, dict) or not isinstance(lock, dict):
        raise R1GateError(f"invalid Git lock recovery receipt fields: {path}")
    relative_source = source.get("relative_path")
    relative_lock = lock.get("relative_path")
    try:
        created = datetime.fromisoformat(str(document["created_at_utc"]))
        lock_time = datetime.fromisoformat(str(lock["last_write_time_utc"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise R1GateError(f"invalid Git lock recovery timestamps: {path}") from exc
    if (
        created.tzinfo is None
        or lock_time.tzinfo is None
        or set(lock) != {"relative_path", "sha256", "size_bytes", "last_write_time_utc"}
        or not isinstance(relative_lock, str)
        or not relative_lock
        or Path(relative_lock).is_absolute()
        or not relative_lock.endswith("/.git/index.lock")
        or not isinstance(lock.get("sha256"), str)
        or len(lock["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in lock["sha256"])
        or lock.get("size_bytes") != 0
        or set(source) != {"relative_path", "sha256", "self_sha256", "execution_id"}
        or not isinstance(relative_source, str)
        or not relative_source
        or Path(relative_source).is_absolute()
    ):
        raise R1GateError(f"invalid Git lock recovery receipt fields: {path}")
    resolved_cache = cache_root.resolve()
    source_path = cache_root / relative_source
    try:
        normalized_source = source_path.resolve().relative_to(resolved_cache).as_posix()
        (cache_root / relative_lock).resolve().relative_to(resolved_cache)
    except ValueError as exc:
        raise R1GateError(f"Git lock recovery receipt escapes the external cache: {path}") from exc
    source_document = _aborted_model_usage(source_path)
    expected_source = {
        "relative_path": normalized_source,
        "sha256": sha256_file(source_path),
        "self_sha256": source_document["self_sha256"],
        "execution_id": source_document["execution_id"],
    }
    expected_path = _recovery_receipt_path(
        cache_root, str(lock.get("relative_path")), source_document["execution_id"]
    )
    if source != expected_source or path.resolve() != expected_path.resolve():
        raise R1GateError(f"Git lock recovery receipt source identity differs: {path}")
    if (
        not isinstance(authorizer, str)
        or len(authorizer) != 32
        or any(character not in "0123456789abcdef" for character in authorizer)
    ):
        raise R1GateError(f"Git lock recovery authorizer identity is invalid: {path}")
    return document


def _recovery_reference(path: Path, cache_root: Path, document: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(cache_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "self_sha256": document["self_sha256"],
        "source_execution_id": document["source_usage"]["execution_id"],
        "source_usage_self_sha256": document["source_usage"]["self_sha256"],
        "recovery_authorized_execution_id": document["recovery_authorized_execution_id"],
    }


def _existing_recoveries(
    cache_root: Path, relative_lock: str, current_execution_id: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((cache_root / "control" / "recoveries").glob("*.json")):
        document = _load_recovery_receipt(path, cache_root)
        if document["lock"].get("relative_path") != relative_lock:
            continue
        authorizer = document["recovery_authorized_execution_id"]
        if authorizer != current_execution_id:
            _aborted_model_usage(cache_root / "control" / "usage" / f"{authorizer}.json")
        records.append(_recovery_reference(path, cache_root, document))
    return records


def _recover_stale_git_index_lock(
    target: Path, cache_root: Path, current_execution_id: str
) -> list[dict[str, Any]]:
    if (
        len(current_execution_id) != 32
        or any(character not in "0123456789abcdef" for character in current_execution_id)
    ):
        raise R1GateError("current R1 execution identity is invalid for Git lock recovery")
    resolved_cache = cache_root.resolve()
    try:
        resolved_target = target.resolve()
        relative_target = resolved_target.relative_to(resolved_cache)
    except ValueError as exc:
        raise R1GateError("Git lock recovery target is outside the external cache") from exc
    relative_lock = (relative_target / ".git" / "index.lock").as_posix()
    lock = target / ".git" / "index.lock"
    recoveries = _existing_recoveries(cache_root, relative_lock, current_execution_id)
    if not lock.exists() and not lock.is_symlink():
        return recoveries
    if lock.is_symlink() or not lock.is_file():
        raise R1GateError(f"Git index lock is not the expected zero-byte file: {lock}")
    lock_stat = lock.stat()
    if lock_stat.st_size != 0:
        raise R1GateError(f"Git index lock is not the expected zero-byte file: {lock}")
    if (cache_root / "manifests" / "r1_model_acquisition.json").exists():
        raise R1GateError("Git lock recovery is forbidden after model acquisition completion")
    active = _active_git_processes()
    if active:
        raise R1GateError(f"Git lock recovery found active Git processes: {active}")
    if not _is_exact_empty_no_checkout(target):
        raise R1GateError("Git lock recovery target is not the exact empty no-checkout state")
    lock_mtime = datetime.fromtimestamp(lock_stat.st_mtime, UTC)
    source = _aborted_usage_for_git_lock(cache_root, lock_mtime)
    source_path = source["path"]
    source_document = source["document"]
    receipt_path = _recovery_receipt_path(
        cache_root, relative_lock, source_document["execution_id"]
    )
    lock_identity = {
        "relative_path": relative_lock,
        "sha256": sha256_file(lock),
        "size_bytes": lock_stat.st_size,
        "last_write_time_utc": lock_mtime.isoformat(),
    }
    source_identity = {
        "relative_path": source_path.resolve().relative_to(resolved_cache).as_posix(),
        "sha256": sha256_file(source_path),
        "self_sha256": source_document["self_sha256"],
        "execution_id": source_document["execution_id"],
    }
    if receipt_path.exists():
        receipt = _load_recovery_receipt(receipt_path, cache_root)
        if receipt.get("lock") != lock_identity or receipt.get("source_usage") != source_identity:
            raise R1GateError("existing Git lock recovery receipt identity differs")
    else:
        _write_json(
            receipt_path,
            {
                "schema_version": 1,
                "artifact_role": "r1_git_index_lock_recovery_authorization",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "recovery_authorized_execution_id": current_execution_id,
                "expected_action_receipt_relative_path": (
                    "manifests/r1_model_acquisition.json"
                ),
                "lock": lock_identity,
                "source_usage": source_identity,
            },
        )
        receipt = _load_recovery_receipt(receipt_path, cache_root)
    if (
        lock.stat().st_size != lock_identity["size_bytes"]
        or sha256_file(lock) != lock_identity["sha256"]
        or datetime.fromtimestamp(lock.stat().st_mtime, UTC).isoformat()
        != lock_identity["last_write_time_utc"]
    ):
        raise R1GateError("Git index lock changed during recovery authorization")
    lock.unlink()
    return _existing_recoveries(cache_root, relative_lock, current_execution_id)


def _git_checkout(
    repository: str,
    revision: str,
    target: Path,
    *,
    cache_root: Path,
    current_execution_id: str,
    lfs_file: str | None = None,
) -> list[dict[str, Any]]:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        _run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(target)],
            env=environment,
        )
    if not (target / ".git").is_dir():
        raise R1GateError(f"existing acquisition target is not a Git checkout: {target}")
    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=target, text=True
    ).strip()
    if origin.rstrip("/") != repository.rstrip("/"):
        raise R1GateError(f"Git acquisition target has another origin: {target}")
    recoveries = _recover_stale_git_index_lock(target, cache_root, current_execution_id)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=target, text=True)
    if status.strip() and not _is_exact_empty_no_checkout(target):
        raise R1GateError(f"Git acquisition target is dirty before checkout: {target}")
    _run(["git", "checkout", "--detach", revision], cwd=target)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    if actual != revision:
        raise R1GateError(f"Git revision mismatch for {target}: {actual} != {revision}")
    if lfs_file is not None:
        _run(["git", "lfs", "pull", "--include", lfs_file, "--exclude", ""], cwd=target)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=target, text=True)
    if status.strip():
        raise R1GateError(f"Git acquisition target is dirty: {target}")
    return recoveries


def _acquire_eres(
    registry: dict[str, Any], cache_root: Path, current_execution_id: str
) -> dict[str, Any]:
    model = registry["eres2netv2"]
    checkpoint_root = cache_root / "models" / model["model_id"] / model["checkpoint_revision"]
    checkpoint_lock_recoveries = _git_checkout(
        model["checkpoint_repository"],
        model["checkpoint_revision"],
        checkpoint_root,
        cache_root=cache_root,
        current_execution_id=current_execution_id,
        lfs_file=model["checkpoint_file"]["path"],
    )
    checkpoint_files = _verify_files(
        checkpoint_root,
        [model["checkpoint_file"], model["checkpoint_config"]],
    )
    source_root = cache_root / "sources" / "3d-speaker" / model["source_revision"]
    source_lock_recoveries = _git_checkout(
        model["source_repository"],
        model["source_revision"],
        source_root,
        cache_root=cache_root,
        current_execution_id=current_execution_id,
    )
    source_files = _verify_files(source_root, model["source_files"] + [model["source_license"]])
    return {
        "model_id": model["model_id"],
        "checkpoint_repository": model["checkpoint_repository"],
        "checkpoint_revision": model["checkpoint_revision"],
        "checkpoint_root": str(checkpoint_root.resolve()),
        "checkpoint_files": checkpoint_files,
        "source_repository": model["source_repository"],
        "source_revision": model["source_revision"],
        "source_root": str(source_root.resolve()),
        "source_files": source_files,
        "stale_git_lock_recoveries": checkpoint_lock_recoveries + source_lock_recoveries,
    }


def _environment_receipt_gate_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    environment = receipt.get("environment_contract")
    if not isinstance(environment, dict):
        raise R1GateError("R1 environment sync receipt lacks its environment contract")
    pyproject = environment.get("pyproject")
    lock = environment.get("lock")
    if not isinstance(pyproject, dict) or not isinstance(lock, dict):
        raise R1GateError("R1 environment sync receipt environment identity is invalid")
    return {
        "r1_gate_sha256": receipt.get("r1_gate_sha256"),
        "r1_gate_self_sha256": receipt.get("r1_gate_self_sha256"),
        "execution_code_manifest_sha256": receipt.get("execution_code_manifest_sha256"),
        "environment_pyproject_sha256": pyproject.get("sha256"),
        "environment_lock_sha256": lock.get("sha256"),
    }


def _validate_environment_receipt_gate(
    receipt: dict[str, Any], gate: dict[str, Any], gate_path: Path
) -> None:
    if receipt.get("environment_contract") != gate.get("environment"):
        raise R1GateError("R1 environment sync receipt uses another environment contract")
    identity = _environment_receipt_gate_identity(receipt)
    current = {
        "r1_gate_sha256": sha256_file(gate_path),
        "r1_gate_self_sha256": gate.get("self_sha256"),
        "execution_code_manifest_sha256": gate.get("execution_code", {}).get(
            "manifest_sha256"
        ),
        "environment_pyproject_sha256": gate.get("environment", {})
        .get("pyproject", {})
        .get("sha256"),
        "environment_lock_sha256": gate.get("environment", {}).get("lock", {}).get("sha256"),
    }
    if identity == current:
        return
    predecessors = gate.get("receipt_compatibility", {}).get(
        "environment_sync_predecessors"
    )
    if not isinstance(predecessors, list) or identity not in predecessors:
        raise R1GateError("R1 environment sync receipt uses another acquisition gate")


def acquire_models(
    cache_root: Path,
    selected: set[str] | None,
    requested_argv: tuple[str, ...],
) -> dict[str, Any]:
    validated_cache_root("model_artifact_download")
    _runtime_versions()
    receipt_path = cache_root / "manifests" / "r1_model_acquisition.json"
    execution = validate_worker_execution(cache_root, receipt_path)
    if execution.requested_argv != requested_argv:
        raise R1GateError("R1 acquisition worker invocation differs from its lease")
    if receipt_path.exists():
        raise R1GateError(f"refusing to rerun model acquisition: {receipt_path}")
    registry_path = EXPERIMENT_ROOT / SOURCE_REGISTRY_PATH
    registry = load_json(registry_path)
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    sync_path = cache_root / "manifests" / "r1_environment_sync.json"
    sync_receipt = load_completed_action_receipt(
        cache_root,
        sync_path,
        "sync-environment",
    )
    _validate_environment_receipt_gate(sync_receipt, gate, gate_path)
    available = {model["model_id"] for model in registry["models"]} | {
        registry["eres2netv2"]["model_id"]
    }
    requested = available if selected is None else selected
    unknown = requested - available
    if unknown:
        raise R1GateError(f"unknown model IDs: {sorted(unknown)}")
    expected_bytes = sum(
        row["size_bytes"]
        for model in registry["models"]
        if model["model_id"] in requested
        for row in model["required_files"]
    )
    if registry["eres2netv2"]["model_id"] in requested:
        expected_bytes += registry["eres2netv2"]["checkpoint_file"]["size_bytes"]
        expected_bytes += registry["eres2netv2"]["checkpoint_config"]["size_bytes"]
    if expected_bytes > 25 * 1024**3:
        raise R1GateError("requested source artifacts exceed the 25 GiB ceiling")
    records: list[dict[str, Any]] = []
    for model in registry["models"]:
        if model["model_id"] in requested:
            records.append(_acquire_huggingface(model, cache_root))
    if registry["eres2netv2"]["model_id"] in requested:
        records.append(_acquire_eres(registry, cache_root, execution.execution_id))
    r0_registry = load_json(EXPERIMENT_ROOT / "models" / "registry.json")
    license_by_model = {model["model_id"]: model["license_id"] for model in r0_registry["models"]}
    model_contracts = []
    for model in registry["models"]:
        if model["model_id"] in requested:
            model_contracts.append(
                {
                    "model_id": model["model_id"],
                    "repository": model["repository"],
                    "revision": model["revision"],
                    "loader_class": model["loader_class"],
                    "processor": model["processor"],
                    "frontend": registry["ssl_frontend"],
                    "trust_remote_code": model["trust_remote_code"],
                    "license": license_by_model[model["model_id"]],
                }
            )
    if registry["eres2netv2"]["model_id"] in requested:
        eres = registry["eres2netv2"]
        model_contracts.append(
            {
                "model_id": eres["model_id"],
                "checkpoint_repository": eres["checkpoint_repository"],
                "checkpoint_revision": eres["checkpoint_revision"],
                "source_repository": eres["source_repository"],
                "source_revision": eres["source_revision"],
                "frontend": eres["official_frontend"],
                "taps": eres["taps"],
                "license": license_by_model[eres["model_id"]],
            }
        )
    receipt = {
        "schema_version": 1,
        "artifact_role": "r1_model_acquisition_receipt",
        "experiment_id": "speaker_representation_scd_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "supervision_binding": {
            "execution_id": execution.execution_id,
            "expected_receipt_relative_path": (execution.expected_receipt_relative_path),
            "authority": "requires_completed_usage_attestation",
        },
        "r1_gate_sha256": sha256_file(gate_path),
        "r1_gate_self_sha256": gate["self_sha256"],
        "execution_code_manifest_sha256": gate["execution_code"]["manifest_sha256"],
        "environment_sync_receipt_sha256": sha256_file(sync_path),
        "environment_sync_receipt_self_sha256": sync_receipt["self_sha256"],
        "source_registry_sha256": sha256_file(registry_path),
        "runtime": {
            "python": sys.version.split()[0],
            "executable": str(Path(sys.executable).resolve()),
            "packages": _runtime_versions(),
        },
        "invocation": {
            "action": "models",
            "selected_model_ids": sorted(requested),
        },
        "run_provenance": run_provenance(
            EXPERIMENT_ROOT.parents[1],
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=False,
        ),
        "expected_download_bytes": expected_bytes,
        "model_contracts": model_contracts,
        "models": records,
        "corpus_downloaded": False,
    }
    _write_json(receipt_path, receipt)
    return load_json(receipt_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sync-environment")
    subparsers.add_parser("models")
    args = parser.parse_args(argv)
    if not args.worker:
        raise R1GateError(
            "direct R1 workers are disabled; use experiments.speaker_representation_scd.r1_execute"
        )
    action = "environment_sync" if args.command == "sync-environment" else "model_artifact_download"
    cache_root = validated_cache_root(action)
    requested_argv = validate_worker_lease(cache_root)
    if args.command == "sync-environment":
        result = sync_environment(cache_root, requested_argv)
    else:
        result = acquire_models(cache_root, None, requested_argv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
