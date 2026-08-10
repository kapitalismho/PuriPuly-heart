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

from experiments.speaker_representation_scd.execution_guard import (
    load_completed_action_receipt,
    validate_worker_execution,
    validate_worker_lease,
)
from experiments.speaker_representation_scd.provenance import (
    load_json,
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


def _git_checkout(
    repository: str, revision: str, target: Path, *, lfs_file: str | None = None
) -> None:
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


def _acquire_eres(registry: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    model = registry["eres2netv2"]
    checkpoint_root = cache_root / "models" / model["model_id"] / model["checkpoint_revision"]
    _git_checkout(
        model["checkpoint_repository"],
        model["checkpoint_revision"],
        checkpoint_root,
        lfs_file=model["checkpoint_file"]["path"],
    )
    checkpoint_files = _verify_files(
        checkpoint_root,
        [model["checkpoint_file"], model["checkpoint_config"]],
    )
    source_root = cache_root / "sources" / "3d-speaker" / model["source_revision"]
    _git_checkout(model["source_repository"], model["source_revision"], source_root)
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
        records.append(_acquire_eres(registry, cache_root))
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
