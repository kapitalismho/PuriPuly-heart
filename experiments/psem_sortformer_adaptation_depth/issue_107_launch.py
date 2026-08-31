from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.psem_sortformer_adaptation_depth.authority_registry import authority_registry_root
from experiments.psem_sortformer_adaptation_depth.detached_phase_runner import (
    GIT_HEAD_PATTERN,
    canonical_bytes,
    fsync_directory,
    load_json_object,
    sha256_bytes,
    sha256_file,
    validate_config,
)

IMAGE_IDENTITY = "sha256:14acbef50fa15281bded1d3fbbcd8029091aeba0692d5647255aa5b90eff8ca7"
RUN_MODULE = "experiments.psem_sortformer_adaptation_depth.run"
LAUNCH_MODULE = "experiments.psem_sortformer_adaptation_depth.issue_107_launch"
RUNTIME_VALIDATOR = Path("/opt/psem/validate-runtime.py")
STORAGE_ROOT = Path("/workspace")
MINIMUM_STORAGE_CAPACITY_BYTES = 30_000_000_000
MINIMUM_FREE_STORAGE_BYTES = 8 * 1024**3
DEV_ROLE = "PSEM-STRATEGY-DEV"


class LaunchError(RuntimeError):
    pass


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path.resolve()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _candidate_git_head(value: str) -> str:
    if GIT_HEAD_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "candidate Git head must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def _aware_datetime(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset")
    return parsed


def _require_exact_candidate_repository(args: argparse.Namespace) -> None:
    repository_root = args.repository_root.resolve()
    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise LaunchError("failed to inspect the candidate Git repository") from exc
    observed_head = head_result.stdout.strip()
    if observed_head != args.candidate_git_head:
        raise LaunchError("repository HEAD differs from the authorized candidate Git head")
    if status_result.stdout:
        raise LaunchError("candidate Git repository must be clean")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bound_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LaunchError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise LaunchError(f"refusing to overwrite existing output: {path}") from exc
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_json(path: Path, value: object) -> None:
    _atomic_create_bytes(path, canonical_bytes(value))


def _atomic_create_bound_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    bound = _bound_payload(value)
    _atomic_create_json(path, bound)
    return bound


def _run_root(args: argparse.Namespace) -> Path:
    return args.persistent_root / "issue-107" / "runs" / args.run_id


def _summary_relative(phase_id: str) -> str:
    return f"receipts/{phase_id}-summary.json"


def _format_float(value: float) -> str:
    return format(value, ".17g")


def _format_datetime(value: datetime) -> str:
    return value.isoformat()


def _common_phase_argv(args: argparse.Namespace, phase_id: str) -> list[str]:
    return [
        "--run-id",
        args.run_id,
        "--persistent-root",
        str(args.persistent_root),
        "--repository-root",
        str(args.repository_root),
        "--candidate-git-head",
        args.candidate_git_head,
        "--checkpoint",
        str(args.checkpoint),
        "--corpus-root",
        str(args.corpus_root),
        "--reference-root",
        str(args.reference_root),
        "--nemo-checkout",
        str(args.nemo_checkout),
        "--image-identity",
        IMAGE_IDENTITY,
        "--hourly-price-usd",
        _format_float(args.hourly_price_usd),
        "--hourly-price-source",
        args.hourly_price_source,
        "--billing-started-at",
        _format_datetime(args.billing_started_at),
        "--max-runtime-hours",
        _format_float(args.max_runtime_hours),
        "--minimum-free-bytes",
        str(args.minimum_free_bytes),
        "--phase-summary",
        f"{{run_root}}/{_summary_relative(phase_id)}",
    ]


def _phase(
    args: argparse.Namespace,
    phase_id: str,
    command: str,
    *,
    extra_argv: Sequence[str] = (),
    next_phase: str | None = None,
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": phase_id,
        "argv": [
            sys.executable,
            "-m",
            LAUNCH_MODULE,
            command,
            *extra_argv,
            *_common_phase_argv(args, phase_id),
        ],
        "cwd": "{repository_root}",
        "environment": {
            "PSEM_CONTAINER_IMAGE_IDENTITY": IMAGE_IDENTITY,
            "PSEM_SORTFORMER_NEMO_PATH": str(args.checkpoint),
            "PSEM_CORPUS_ROOT": str(args.corpus_root),
            "PSEM_REFERENCE_ROOT": str(args.reference_root),
            "PSEM_ADAPTATION_OUTPUT_ROOT": "{run_root}/output",
            "PSEM_PROTOCOL_REGISTRY_ROOT": "{run_root}/protocol-registry",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "required_inputs": [],
        "required_outputs": [{"path": _summary_relative(phase_id)}],
        "next_phase": next_phase,
    }
    if gate is not None:
        value["decision_gate_after"] = dict(gate)
    return value


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    phases = [
        _phase(args, "bootstrap-f0", "bootstrap-f0", next_phase="h-head-material-and-dev"),
        _phase(
            args,
            "h-head-material-and-dev",
            "run-arm",
            extra_argv=("--arm", "H-HEAD"),
            next_phase="t2-top-material-and-dev",
        ),
        _phase(
            args,
            "t2-top-material-and-dev",
            "run-arm",
            extra_argv=("--arm", "T2-TOP"),
            gate={
                "id": "after-h-t2-dev",
                "actions": {
                    "open_ta": "ta-all-temporal-material-and-dev",
                    "select_candidate": None,
                    "stop": None,
                },
            },
        ),
        _phase(
            args,
            "ta-all-temporal-material-and-dev",
            "run-ta",
            gate={
                "id": "after-ta-dev",
                "actions": {"select_candidate": None, "stop": None},
            },
        ),
    ]
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "persistent_root": str(args.persistent_root),
        "repository_root": str(args.repository_root),
        "candidate_git_head": args.candidate_git_head,
        "first_phase": "bootstrap-f0",
        "phases": phases,
    }


def _validate_common(args: argparse.Namespace) -> None:
    candidate_git_head = getattr(args, "candidate_git_head", None)
    if (
        not isinstance(candidate_git_head, str)
        or GIT_HEAD_PATTERN.fullmatch(candidate_git_head) is None
    ):
        raise LaunchError("candidate Git head must be exactly 40 lowercase hexadecimal characters")
    if args.image_identity != IMAGE_IDENTITY:
        raise LaunchError("image identity differs from the authorized digest")
    if not args.hourly_price_source.strip():
        raise LaunchError("hourly price source must be nonempty")
    if args.minimum_free_bytes < MINIMUM_FREE_STORAGE_BYTES:
        raise LaunchError("minimum free bytes must be at least 8 GiB")
    projected_cost = args.hourly_price_usd * args.max_runtime_hours
    if not math.isfinite(projected_cost) or projected_cost > 30.0:
        raise LaunchError("projected deadline cost exceeds USD 30")
    _cost_seconds(args)
    run_root = _run_root(args).resolve()
    repository_root = args.repository_root.resolve()
    if run_root == repository_root or run_root.is_relative_to(repository_root):
        raise LaunchError("run root must be outside the repository")


def _reject_sensitive_environment() -> None:
    forbidden = [name for name in ("RUNPOD_API_KEY", "PSEM_ALLOW_EVAL") if name in os.environ]
    if forbidden:
        raise LaunchError(f"forbidden inherited environment variable: {forbidden[0]}")


def _validate_environment(args: argparse.Namespace, run_root: Path) -> None:
    expected = {
        "PSEM_CONTAINER_IMAGE_IDENTITY": IMAGE_IDENTITY,
        "PSEM_SORTFORMER_NEMO_PATH": str(args.checkpoint),
        "PSEM_CORPUS_ROOT": str(args.corpus_root),
        "PSEM_REFERENCE_ROOT": str(args.reference_root),
        "PSEM_ADAPTATION_OUTPUT_ROOT": str(run_root / "output"),
        "PSEM_PROTOCOL_REGISTRY_ROOT": str(run_root / "protocol-registry"),
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise LaunchError(f"runtime environment mismatch: {name}")


def _load_runtime_context(
    args: argparse.Namespace, phase_id: str
) -> tuple[Path, str, dict[str, Any]]:
    _reject_sensitive_environment()
    _validate_common(args)
    _require_exact_candidate_repository(args)
    if args.persistent_root.resolve() != STORAGE_ROOT.resolve():
        raise LaunchError("runtime persistent root must be /workspace")
    run_root = _run_root(args).resolve()
    expected_summary = run_root / _summary_relative(phase_id)
    if args.phase_summary.resolve() != expected_summary:
        raise LaunchError("phase summary path differs from the configured output")
    config_path = run_root / "control" / "run_config.json"
    config = load_json_object(config_path)
    validate_config(config)
    expected_config = _build_config(args)
    if config != expected_config:
        raise LaunchError("durable run config differs from the exact launch graph")
    config_sha256 = sha256_bytes(canonical_bytes(config))
    state = load_json_object(run_root / "control" / "state.json")
    if (
        state.get("run_id") != args.run_id
        or state.get("config_sha256") != config_sha256
        or state.get("status") != "RUNNING"
        or state.get("active_phase") != phase_id
    ):
        raise LaunchError("detached state does not authorize the active phase")
    _validate_environment(args, run_root)
    return run_root, config_sha256, config


def _storage_evidence(label: str, minimum_free_bytes: int) -> dict[str, int | str]:
    usage = shutil.disk_usage(STORAGE_ROOT)
    evidence: dict[str, int | str] = {
        "label": label,
        "captured_at": datetime.now(UTC).isoformat(),
        "path": str(STORAGE_ROOT),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_total_bytes": MINIMUM_STORAGE_CAPACITY_BYTES,
        "required_free_bytes": minimum_free_bytes,
    }
    if usage.total < MINIMUM_STORAGE_CAPACITY_BYTES:
        raise LaunchError("/workspace total capacity is below 30000000000 bytes")
    if usage.free < minimum_free_bytes:
        raise LaunchError("/workspace free space is below the configured minimum")
    return evidence


def _artifact_record(path: Path, run_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LaunchError(f"required artifact is absent: {path}")
    resolved = path.resolve()
    authority_root = authority_registry_root().resolve()
    if resolved.is_relative_to(run_root):
        scope = "run_root"
        relative_path = resolved.relative_to(run_root).as_posix()
    elif resolved.is_relative_to(authority_root):
        scope = "authority_registry"
        relative_path = resolved.relative_to(authority_root).as_posix()
    else:
        raise LaunchError(f"artifact escapes authorized roots: {path}")
    return {
        "path": str(resolved),
        "scope": scope,
        "relative_path": relative_path,
        "sha256": sha256_file(resolved),
        "size": resolved.stat().st_size,
    }


def _snapshot_scientific(run_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    roots = [run_root / name for name in ("receipts", "output", "protocol-registry")]
    roots.append(authority_registry_root().resolve())
    for root in roots:
        if not root.exists():
            continue
        if not root.is_dir():
            raise LaunchError(f"scientific root is not a directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                record = _artifact_record(path, run_root)
                records[record["path"]] = record
    return records


def _generated_artifacts(
    before: Mapping[str, Mapping[str, Any]], run_root: Path
) -> list[dict[str, Any]]:
    after = _snapshot_scientific(run_root)
    return [
        after[path] for path in sorted(after) if path not in before or after[path] != before[path]
    ]


def _require_absent(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            raise LaunchError(f"refusing preexisting scientific output: {path}")


def _command_record(argv: Sequence[str]) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    completed = subprocess.run(list(argv), check=False, shell=False)
    completed_at = datetime.now(UTC).isoformat()
    record = {
        "argv": list(argv),
        "argv_sha256": sha256_bytes(canonical_bytes(list(argv))),
        "started_at": started_at,
        "completed_at": completed_at,
        "return_code": completed.returncode,
    }
    if completed.returncode != 0:
        raise LaunchError(
            f"subprocess exited with code {completed.returncode}: {json.dumps(list(argv))}"
        )
    return record


def _run_command(arguments: Sequence[str]) -> dict[str, Any]:
    return _command_record([sys.executable, "-m", RUN_MODULE, *arguments])


def _runtime_validator_command(receipt: Path) -> list[str]:
    return [
        sys.executable,
        str(RUNTIME_VALIDATOR),
        "--mode",
        "runtime",
        "--expected-image-identity",
        IMAGE_IDENTITY,
        "--receipt",
        str(receipt),
    ]


def _common_receipts(run_root: Path) -> dict[str, Path]:
    receipts = run_root / "receipts"
    return {
        "runtime_validation": receipts / "runtime-validation.json",
        "preflight": receipts / "runtime-preflight.json",
        "dependency_lock": receipts / "dependency-lock.json",
        "sampling_manifest": receipts / "sampling-manifest.jsonl",
        "sampling_validation": receipts / "sampling-validation.json",
        "class_weights": receipts / "class-weights.json",
        "lineage_authorization": receipts / "lineage-authorization.json",
        "lineage": receipts / "lineage.json",
        "validated_lineage": receipts / "validated-lineage.json",
        "runtime_identity": receipts / "runtime-identity.json",
        "f0_prediction": receipts / "f0-dev-prediction-set.json",
        "f0_result": receipts / "f0-dev-result.json",
        "staged_after_f0": receipts / "staged-after-f0.json",
    }


def _arm_slug(arm: str) -> str:
    return {
        "H-HEAD": "h-head",
        "T2-TOP": "t2-top",
        "TA-ALL-TEMPORAL": "ta-all-temporal",
    }[arm]


def _arm_paths(run_root: Path, arm: str) -> dict[str, Path]:
    slug = _arm_slug(arm)
    receipts = run_root / "receipts"
    return {
        "canary": receipts / f"{slug}-canary.json",
        "smoke": receipts / f"{slug}-smoke.json",
        "cost": receipts / f"{slug}-cost.json",
        "material_inputs": receipts / f"{slug}-material-inputs.json",
        "material_bundle": receipts / f"{slug}-material-bundle.json",
        "material_gate": receipts / f"{slug}-material-gate.json",
        "training": receipts / f"{slug}-training.json",
        "checkpoint_receipt": receipts / f"{slug}-checkpoint-receipt.json",
        "prediction": receipts / f"{slug}-dev-prediction-set.json",
        "dev_result": receipts / f"{slug}-dev-result.json",
        "staged": receipts
        / {
            "H-HEAD": "staged-after-h.json",
            "T2-TOP": "staged-after-t2.json",
            "TA-ALL-TEMPORAL": "staged-after-ta.json",
        }[arm],
        "checkpoint": run_root / "output" / "checkpoints" / arm / "7301" / "step-256.pt",
        "checkpoint_arm_root": run_root / "output" / "checkpoints" / arm,
        "prediction_arm_root": run_root / "output" / "predictions" / DEV_ROLE / arm,
    }


def _arm_priors(run_root: Path, arm: str) -> tuple[Path, list[Path]]:
    common = _common_receipts(run_root)
    h = _arm_paths(run_root, "H-HEAD")
    if arm == "H-HEAD":
        return common["staged_after_f0"], [common["f0_result"]]
    if arm == "T2-TOP":
        return h["staged"], [common["f0_result"], h["dev_result"]]
    t2 = _arm_paths(run_root, "T2-TOP")
    return t2["staged"], [common["f0_result"], h["dev_result"], t2["dev_result"]]


def _cost_seconds(args: argparse.Namespace) -> tuple[float, float]:
    now = datetime.now(UTC)
    started = args.billing_started_at.astimezone(UTC)
    authorized_seconds = args.max_runtime_hours * 3600.0
    deadline = started + timedelta(seconds=authorized_seconds)
    if now < started:
        raise LaunchError("billing start is in the future")
    if now > deadline:
        raise LaunchError("authorized billing deadline has passed")
    actual = (now - started).total_seconds()
    projected = authorized_seconds - actual
    if args.hourly_price_usd * (actual + projected) / 3600.0 > 30.0:
        raise LaunchError("current projected billing horizon exceeds USD 30")
    return actual, projected


def _cost_command(args: argparse.Namespace, output: Path, action: str) -> dict[str, Any]:
    actual, projected = _cost_seconds(args)
    return _run_command(
        [
            "cost-receipt",
            "--hourly-price-usd",
            _format_float(args.hourly_price_usd),
            "--hourly-price-source",
            args.hourly_price_source,
            "--actual-gpu-seconds",
            _format_float(actual),
            "--projected-remaining-gpu-seconds",
            _format_float(projected),
            "--command",
            action,
            "--output",
            str(output),
        ]
    )


def _phase_summary(
    *,
    args: argparse.Namespace,
    phase_id: str,
    run_root: Path,
    config_sha256: str,
    started_at: str,
    storage: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]],
    before: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = _generated_artifacts(before, run_root)
    value = {
        "schema_version": 1,
        "artifact_role": "issue_107_launch_phase_summary",
        "run_id": args.run_id,
        "config_sha256": config_sha256,
        "phase_id": phase_id,
        "image_identity": IMAGE_IDENTITY,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "storage_evidence": [dict(snapshot) for snapshot in storage],
        "inputs": list(inputs),
        "commands": [dict(command) for command in commands],
        "artifacts": artifacts,
    }
    return _atomic_create_bound_json(args.phase_summary, value)


def write_config(args: argparse.Namespace) -> dict[str, Any]:
    _validate_common(args)
    _require_exact_candidate_repository(args)
    output = args.output.resolve()
    repository_root = args.repository_root.resolve()
    if output == repository_root or output.is_relative_to(repository_root):
        raise LaunchError("run config output must be outside the repository")
    config = _build_config(args)
    validate_config(config)
    _atomic_create_json(output, config)
    return {
        "output": str(output),
        "config_sha256": sha256_bytes(canonical_bytes(config)),
        "run_root": str(_run_root(args).resolve()),
        "phases": [phase["id"] for phase in config["phases"]],
    }


def bootstrap_f0(args: argparse.Namespace) -> dict[str, Any]:
    phase_id = "bootstrap-f0"
    started_at = datetime.now(UTC).isoformat()
    run_root, config_sha256, _ = _load_runtime_context(args, phase_id)
    storage = [_storage_evidence("phase_start", args.minimum_free_bytes)]
    roots = [run_root / name for name in ("receipts", "output", "protocol-registry")]
    for root in roots:
        if root.exists():
            if not root.is_dir() or any(root.iterdir()):
                raise LaunchError(f"refusing preexisting scientific outputs: {root}")
        else:
            root.mkdir(parents=False)
    paths = _common_receipts(run_root)
    _require_absent([*paths.values(), args.phase_summary])
    before = _snapshot_scientific(run_root)
    commands: list[dict[str, Any]] = []
    commands.append(_command_record(_runtime_validator_command(paths["runtime_validation"])))
    commands.append(
        _run_command(
            [
                "preflight",
                "--checkpoint",
                str(args.checkpoint),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output-root",
                str(run_root / "output"),
                "--protocol-registry-root",
                str(run_root / "protocol-registry"),
                "--receipt-output",
                str(paths["preflight"]),
            ]
        )
    )
    commands.append(_run_command(["dependency-lock", "--output", str(paths["dependency_lock"])]))
    commands.append(
        _run_command(
            [
                "sampling-manifest",
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output",
                str(paths["sampling_manifest"]),
            ]
        )
    )
    commands.append(
        _run_command(
            [
                "validate-sampling-manifest",
                "--manifest",
                str(paths["sampling_manifest"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output",
                str(paths["sampling_validation"]),
            ]
        )
    )
    commands.append(
        _run_command(
            [
                "class-weights",
                "--manifest",
                str(paths["sampling_manifest"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output",
                str(paths["class_weights"]),
            ]
        )
    )
    commands.append(
        _run_command(["lineage-authorization", "--output", str(paths["lineage_authorization"])])
    )
    _cost_seconds(args)
    storage.append(_storage_evidence("before_build_lineage", args.minimum_free_bytes))
    commands.append(
        _run_command(
            [
                "build-lineage",
                "--checkpoint",
                str(args.checkpoint),
                "--nemo-checkout",
                str(args.nemo_checkout),
                "--dependency-lock",
                str(paths["dependency_lock"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output-root",
                str(run_root / "output"),
                "--authorization",
                str(paths["lineage_authorization"]),
                "--device",
                "cuda",
                "--output",
                str(paths["lineage"]),
                "--runtime-identity-output",
                str(paths["runtime_identity"]),
            ]
        )
    )
    commands.append(
        _run_command(
            [
                "validate-lineage",
                str(paths["lineage"]),
                "--runtime-identity",
                str(paths["runtime_identity"]),
                "--output",
                str(paths["validated_lineage"]),
            ]
        )
    )
    _cost_seconds(args)
    storage.append(_storage_evidence("before_f0_inference", args.minimum_free_bytes))
    commands.append(
        _run_command(
            [
                "infer",
                "--checkpoint",
                str(args.checkpoint),
                "--nemo-checkout",
                str(args.nemo_checkout),
                "--dependency-lock",
                str(paths["dependency_lock"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output-root",
                str(run_root / "output"),
                "--protocol-registry-root",
                str(run_root / "protocol-registry"),
                "--device",
                "cuda",
                "--role",
                DEV_ROLE,
                "--arm",
                "F0-FROZEN-FLOAT",
                "--output",
                str(paths["f0_prediction"]),
            ]
        )
    )
    commands.append(
        _run_command(
            [
                "evaluate",
                str(paths["f0_prediction"]),
                "--output",
                str(paths["f0_result"]),
            ]
        )
    )
    commands.append(
        _run_command(
            [
                "stage-init",
                str(paths["f0_result"]),
                "--output",
                str(paths["staged_after_f0"]),
            ]
        )
    )
    inputs = [_artifact_record(run_root / "control" / "run_config.json", run_root)]
    return _phase_summary(
        args=args,
        phase_id=phase_id,
        run_root=run_root,
        config_sha256=config_sha256,
        started_at=started_at,
        storage=storage,
        commands=commands,
        inputs=inputs,
        before=before,
    )


def _arm_input_paths(run_root: Path, arm: str) -> list[Path]:
    common = _common_receipts(run_root)
    staged, prior_results = _arm_priors(run_root, arm)
    return [
        common["preflight"],
        common["dependency_lock"],
        common["sampling_manifest"],
        common["sampling_validation"],
        common["class_weights"],
        common["validated_lineage"],
        common["runtime_identity"],
        staged,
        *prior_results,
    ]


def _material_inputs(
    *,
    arm: str,
    common: Mapping[str, Path],
    paths: Mapping[str, Path],
    staged: Path,
    prior_results: Sequence[Path],
    ta_authorization: Path | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "issue_107_material_inputs",
        "arm": arm,
        "seed": 7301,
        "preflight_receipt": str(common["preflight"]),
        "sampling_validation": str(common["sampling_validation"]),
        "class_weight_receipt": str(common["class_weights"]),
        "lineage_receipt": str(common["validated_lineage"]),
        "runtime_identity": str(common["runtime_identity"]),
        "runtime_evidence": str(paths["canary"]),
        "short_smoke_receipt": str(paths["smoke"]),
        "cost_receipt": str(paths["cost"]),
        "staged_execution_receipt": str(staged),
        "staged_dev_results": [str(path) for path in prior_results],
    }
    if ta_authorization is not None:
        value["ta_open_authorization"] = str(ta_authorization)
    return value


def _run_arm_sequence(
    *,
    args: argparse.Namespace,
    arm: str,
    run_root: Path,
    storage: list[dict[str, int | str]],
    ta_authorization: Path | None = None,
    create_cost: bool = True,
) -> list[dict[str, Any]]:
    common = _common_receipts(run_root)
    paths = _arm_paths(run_root, arm)
    staged, prior_results = _arm_priors(run_root, arm)
    commands: list[dict[str, Any]] = []
    ta_args = (
        ["--ta-open-authorization", str(ta_authorization)] if ta_authorization is not None else []
    )
    _cost_seconds(args)
    storage.append(_storage_evidence("before_canary", args.minimum_free_bytes))
    commands.append(
        _run_command(
            [
                "canary-arm",
                "--checkpoint",
                str(args.checkpoint),
                "--nemo-checkout",
                str(args.nemo_checkout),
                "--dependency-lock",
                str(common["dependency_lock"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--manifest",
                str(common["sampling_manifest"]),
                "--arm",
                arm,
                "--device",
                "cuda",
                *ta_args,
                "--output",
                str(paths["canary"]),
            ]
        )
    )
    _cost_seconds(args)
    storage.append(_storage_evidence("before_smoke", args.minimum_free_bytes))
    commands.append(
        _run_command(
            [
                "smoke-arm",
                "--checkpoint",
                str(args.checkpoint),
                "--nemo-checkout",
                str(args.nemo_checkout),
                "--dependency-lock",
                str(common["dependency_lock"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--manifest",
                str(common["sampling_manifest"]),
                "--class-weights",
                str(common["class_weights"]),
                "--arm",
                arm,
                "--device",
                "cuda",
                *ta_args,
                "--output",
                str(paths["smoke"]),
            ]
        )
    )
    if create_cost:
        commands.append(
            _cost_command(args, paths["cost"], f"Issue #107 {arm} authorized billing horizon")
        )
    _atomic_create_bound_json(
        paths["material_inputs"],
        _material_inputs(
            arm=arm,
            common=common,
            paths=paths,
            staged=staged,
            prior_results=prior_results,
            ta_authorization=ta_authorization,
        ),
    )
    commands.append(
        _run_command(
            [
                "assemble-material-bundle",
                str(paths["material_inputs"]),
                "--output",
                str(paths["material_bundle"]),
            ]
        )
    )
    storage.append(
        _storage_evidence("before_material_validation_and_training", args.minimum_free_bytes)
    )
    commands.append(
        _run_command(
            [
                "validate-material-gate",
                str(paths["material_bundle"]),
                "--manifest",
                str(common["sampling_manifest"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output",
                str(paths["material_gate"]),
            ]
        )
    )
    _cost_seconds(args)
    commands.append(
        _run_command(
            [
                "train-arm",
                "--checkpoint",
                str(args.checkpoint),
                "--nemo-checkout",
                str(args.nemo_checkout),
                "--dependency-lock",
                str(common["dependency_lock"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--manifest",
                str(common["sampling_manifest"]),
                "--class-weights",
                str(common["class_weights"]),
                "--material-gate",
                str(paths["material_gate"]),
                "--output-root",
                str(run_root / "output"),
                "--device",
                "cuda",
                "--training-output",
                str(paths["training"]),
                "--checkpoint-receipt-output",
                str(paths["checkpoint_receipt"]),
            ]
        )
    )
    _cost_seconds(args)
    storage.append(_storage_evidence("before_dev_inference", args.minimum_free_bytes))
    commands.append(
        _run_command(
            [
                "infer",
                "--checkpoint",
                str(args.checkpoint),
                "--nemo-checkout",
                str(args.nemo_checkout),
                "--dependency-lock",
                str(common["dependency_lock"]),
                "--corpus-root",
                str(args.corpus_root),
                "--reference-root",
                str(args.reference_root),
                "--output-root",
                str(run_root / "output"),
                "--protocol-registry-root",
                str(run_root / "protocol-registry"),
                "--device",
                "cuda",
                "--role",
                DEV_ROLE,
                "--arm",
                arm,
                "--seed",
                "7301",
                "--trained-checkpoint",
                str(paths["checkpoint"]),
                "--trained-checkpoint-receipt",
                str(paths["checkpoint_receipt"]),
                "--output",
                str(paths["prediction"]),
            ]
        )
    )
    commands.append(
        _run_command(
            [
                "evaluate",
                str(paths["prediction"]),
                "--output",
                str(paths["dev_result"]),
            ]
        )
    )
    stage_arguments = [
        "stage-append",
        str(staged),
        str(paths["dev_result"]),
    ]
    for prior in prior_results:
        stage_arguments.extend(["--prior-result", str(prior)])
    stage_arguments.extend(["--output", str(paths["staged"])])
    commands.append(_run_command(stage_arguments))
    return commands


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    phase_id = {
        "H-HEAD": "h-head-material-and-dev",
        "T2-TOP": "t2-top-material-and-dev",
    }[args.arm]
    started_at = datetime.now(UTC).isoformat()
    run_root, config_sha256, _ = _load_runtime_context(args, phase_id)
    storage = [_storage_evidence("phase_start", args.minimum_free_bytes)]
    paths = _arm_paths(run_root, args.arm)
    planned = [
        paths[key]
        for key in (
            "canary",
            "smoke",
            "cost",
            "material_inputs",
            "material_bundle",
            "material_gate",
            "training",
            "checkpoint_receipt",
            "prediction",
            "dev_result",
            "staged",
            "checkpoint_arm_root",
            "prediction_arm_root",
        )
    ]
    _require_absent([*planned, args.phase_summary])
    input_paths = _arm_input_paths(run_root, args.arm)
    inputs = [_artifact_record(path, run_root) for path in input_paths]
    before = _snapshot_scientific(run_root)
    commands = _run_arm_sequence(
        args=args,
        arm=args.arm,
        run_root=run_root,
        storage=storage,
    )
    return _phase_summary(
        args=args,
        phase_id=phase_id,
        run_root=run_root,
        config_sha256=config_sha256,
        started_at=started_at,
        storage=storage,
        commands=commands,
        inputs=inputs,
        before=before,
    )


def _load_open_ta_decision(
    run_root: Path, run_id: str, config_sha256: str
) -> tuple[Path, dict[str, Any]]:
    decision_dir = run_root / "control" / "decisions"
    candidates = sorted(decision_dir.glob("after-h-t2-dev-*.json"))
    if len(candidates) != 1:
        raise LaunchError("exactly one archived after-h-t2-dev decision is required")
    if (run_root / "control" / "decision.json").exists():
        raise LaunchError("an unconsumed detached decision is present")
    path = candidates[0]
    decision = load_json_object(path)
    expected_keys = {
        "schema_version",
        "artifact_role",
        "run_id",
        "config_sha256",
        "gate_id",
        "action",
        "rationale",
        "created_at",
    }
    if set(decision) != expected_keys:
        raise LaunchError("archived detached decision schema is invalid")
    try:
        created_at = _aware_datetime(str(decision["created_at"]))
    except argparse.ArgumentTypeError as exc:
        raise LaunchError("archived detached decision timestamp is invalid") from exc
    if (
        decision["schema_version"] != 1
        or decision["artifact_role"] != "detached_operator_decision"
        or decision["run_id"] != run_id
        or decision["config_sha256"] != config_sha256
        or decision["gate_id"] != "after-h-t2-dev"
        or decision["action"] != "open_ta"
        or not isinstance(decision["rationale"], str)
        or not decision["rationale"].strip()
        or created_at > datetime.now(created_at.tzinfo)
    ):
        raise LaunchError("archived detached open_ta decision is not authorized")
    return path, decision


def run_ta(args: argparse.Namespace) -> dict[str, Any]:
    phase_id = "ta-all-temporal-material-and-dev"
    started_at = datetime.now(UTC).isoformat()
    run_root, config_sha256, _ = _load_runtime_context(args, phase_id)
    storage = [_storage_evidence("phase_start", args.minimum_free_bytes)]
    decision_archive, decision = _load_open_ta_decision(run_root, args.run_id, config_sha256)
    paths = _arm_paths(run_root, "TA-ALL-TEMPORAL")
    scientific_decision = run_root / "receipts" / "open-ta-dev-decision.json"
    ta_authorization = run_root / "receipts" / "open-ta-authorization.json"
    planned = [
        scientific_decision,
        ta_authorization,
        *[
            paths[key]
            for key in (
                "canary",
                "smoke",
                "cost",
                "material_inputs",
                "material_bundle",
                "material_gate",
                "training",
                "checkpoint_receipt",
                "prediction",
                "dev_result",
                "staged",
                "checkpoint_arm_root",
                "prediction_arm_root",
            )
        ],
    ]
    _require_absent([*planned, args.phase_summary])
    input_paths = _arm_input_paths(run_root, "TA-ALL-TEMPORAL")
    inputs = [_artifact_record(path, run_root) for path in [decision_archive, *input_paths]]
    before = _snapshot_scientific(run_root)
    prior_results = _arm_priors(run_root, "TA-ALL-TEMPORAL")[1]
    commands: list[dict[str, Any]] = []
    decision_arguments = [
        "dev-decision",
        "--decision",
        "open_ta",
        "--selected-arm",
        "TA-ALL-TEMPORAL",
        "--rationale",
        decision["rationale"],
    ]
    for result in prior_results:
        decision_arguments.extend(["--dev-result", str(result)])
    decision_arguments.extend(["--output", str(scientific_decision)])
    commands.append(_run_command(decision_arguments))
    commands.append(
        _cost_command(
            args,
            paths["cost"],
            "Issue #107 TA-ALL-TEMPORAL authorized billing horizon",
        )
    )
    open_arguments = [
        "open-ta",
        str(scientific_decision),
        str(paths["cost"]),
        str(_arm_priors(run_root, "TA-ALL-TEMPORAL")[0]),
    ]
    for result in prior_results:
        open_arguments.extend(["--dev-result", str(result)])
    open_arguments.extend(["--output", str(ta_authorization)])
    commands.append(_run_command(open_arguments))
    commands.extend(
        _run_arm_sequence(
            args=args,
            arm="TA-ALL-TEMPORAL",
            run_root=run_root,
            storage=storage,
            ta_authorization=ta_authorization,
            create_cost=False,
        )
    )
    return _phase_summary(
        args=args,
        phase_id=phase_id,
        run_root=run_root,
        config_sha256=config_sha256,
        started_at=started_at,
        storage=storage,
        commands=commands,
        inputs=inputs,
        before=before,
    )


def _add_common_arguments(parser: argparse.ArgumentParser, *, phase: bool) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--persistent-root", type=_absolute_path, required=True)
    parser.add_argument("--repository-root", type=_absolute_path, required=True)
    parser.add_argument("--candidate-git-head", type=_candidate_git_head, required=True)
    parser.add_argument("--checkpoint", type=_absolute_path, required=True)
    parser.add_argument("--corpus-root", type=_absolute_path, required=True)
    parser.add_argument("--reference-root", type=_absolute_path, required=True)
    parser.add_argument("--nemo-checkout", type=_absolute_path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--hourly-price-usd", type=_positive_float, required=True)
    parser.add_argument("--hourly-price-source", required=True)
    parser.add_argument("--billing-started-at", type=_aware_datetime, required=True)
    parser.add_argument("--max-runtime-hours", type=_positive_float, required=True)
    parser.add_argument("--minimum-free-bytes", type=_positive_int, required=True)
    if phase:
        parser.add_argument("--phase-summary", type=_absolute_path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser("write-config")
    _add_common_arguments(config_parser, phase=False)
    config_parser.add_argument("--output", type=_absolute_path, required=True)
    bootstrap_parser = commands.add_parser("bootstrap-f0")
    _add_common_arguments(bootstrap_parser, phase=True)
    arm_parser = commands.add_parser("run-arm")
    arm_parser.add_argument("--arm", choices=("H-HEAD", "T2-TOP"), required=True)
    _add_common_arguments(arm_parser, phase=True)
    ta_parser = commands.add_parser("run-ta")
    _add_common_arguments(ta_parser, phase=True)
    args = parser.parse_args(argv)
    if args.command == "write-config":
        result = write_config(args)
    elif args.command == "bootstrap-f0":
        result = bootstrap_f0(args)
    elif args.command == "run-arm":
        result = run_arm(args)
    else:
        result = run_ta(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
