from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.psem_training_strategy_gate.data.dataset_freeze import (
    DatasetFreezeError,
    validate_checked_dataset_freeze,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    open_reference_checkout,
)
from experiments.psem_training_strategy_gate.runtime_contract import (
    RUNTIME_ARTIFACT_PATHS,
    RUNTIME_CHECK_ARTIFACT_PATHS,
    RUNTIME_CHECK_IDS,
    RUNTIME_RECEIPT_ROLES,
    RuntimeEvidenceError,
    runtime_artifact_checks,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = EXPERIMENT_ROOT / "contract.json"
CONFIG_PATH = EXPERIMENT_ROOT / "config.json"
DATA_DIR = EXPERIMENT_ROOT / "data" / "v2"
SOURCE_MANIFEST_PATH = DATA_DIR / "source_manifest.jsonl"
SOURCE_REGISTRY_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "speaker_representation_scd"
    / "models"
    / "source_registry.json"
)
LABEL_GENERATOR_PATH = EXPERIMENT_ROOT / "data" / "label_contract.py"
EXPERIMENT_ID = "psem_training_strategy_gate_v1"
CONTRACT_VERSION = "issue-76-v1"
AUTHORITY = {
    "ref": "https://github.com/kapitalismho/PuriPuly-heart/issues/76",
    "sha256": "48fbea67633ce4876a94a2901332f8455dccb8275c408fd11ca244bc3f6181bb",
}
EXPECTED_CONTRACT_CANONICAL_SHA256 = (
    "4b4f6a9dfbdf3c9c0c7ce85b210cc1a405d309b8d88b829d9655e0005642e1d0"
)
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "3faf132c4df56e77651583fe3de292d52e14bab2fa2e5b2a2e177235c5fb28d2"
)
EXPECTED_DATASET_FREEZE_SHA256 = "bc7e63bb201c2a33a9b2d69b2364fed8f03839278098f0bd175d6833b330a41e"
EXPECTED_DATASET_PREFLIGHT_SHA256 = (
    "79c4f4d188381288ccefcd2e4dcbbf6b17c86936119af538292dd04f379f4531"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = "76d5a6640ffabbc3cf91c25f5a94284f9869ad266e621ee06f48a987d5d7c6de"
EXPECTED_SOURCE_REGISTRY_SHA256 = "0cc07fa30294a2ae9a4d30ef4bcb7201f6b03d700ce815139975678704d182bb"
EXPECTED_LABEL_GENERATOR_SHA256 = "91829073e8bb85104c59b7750fe80e0be97797eab7289c44ca88fa90291e1423"
EXPECTED_MODEL_IDENTITY = {
    "model_id": "wavlm-base-plus",
    "loader_class": "transformers.WavLMModel",
    "repository": "https://huggingface.co/microsoft/wavlm-base-plus",
    "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
}
EXPECTED_REFERENCE = {
    "repository": "https://github.com/nttcslab-sp/diar-forced-alignment",
    "commit": "9527b7c64846fb38316a610f32e9d3466bd6d8b7",
}
EXPECTED_SOURCE_COUNT = 93
OUTPUT_RELATIVE_PATH = Path("results") / EXPERIMENT_ID
DEFAULT_OUTPUT_ROOT = (
    Path(tempfile.gettempdir()).resolve() / "puripuly-heart-research" / EXPERIMENT_ID
)
RUNTIME_RECEIPTS = RUNTIME_RECEIPT_ROLES
EXPECTED_CHECK_IDS = (
    "contract.issue_76_controls_exact",
    "paths.roots_safe",
    "dataset.freeze_file_identity",
    "dataset.preflight_file_identity",
    "dataset.freeze_current_and_ready",
    "labels.generator_identity",
    "model.source_registry_identity",
    "model.wavlm_registry_entry_exact",
    "model.wavlm_checkpoint_files_exact",
    "sources.manifest_identity",
    "sources.bound_waveforms_resolve",
    "sources.byte_identity_verification_enabled",
    "sources.forced_alignment_reference_exact",
    *(f"runtime_receipt.{name}" for name in RUNTIME_RECEIPTS),
    "git.candidate_is_clean",
)
RECEIPT_KEYS = {
    "schema_version",
    "artifact_role",
    "experiment_id",
    "contract_version",
    "generated_at",
    "authority",
    "binding",
    "git",
    "paths",
    "checks",
    "failed_checks",
    "ready_for_material_run",
    "payload_sha256",
}
BINDING_KEYS = {
    "experiment_id",
    "contract_sha256",
    "config_sha256",
    "dataset_freeze_sha256",
    "source_manifest_sha256",
    "source_registry_sha256",
    "label_generator_sha256",
    "git_commit",
}
PATH_KEYS = {"cache_root", "corpus_root", "reference_root", "output_root"}


class ExperimentPreflightError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreflightPaths:
    cache_root: Path | None
    corpus_root: Path | None
    reference_root: Path | None
    output_root: Path | None
    errors: tuple[str, ...] = ()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256_or_none(path: Path) -> str | None:
    try:
        return sha256_file(path) if path.is_file() else None
    except OSError:
        return None


def _mapping_or_empty(path: Path) -> Mapping[str, Any]:
    try:
        value = load_json(path)
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(
    check_id: str,
    passed: bool,
    *,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "expected": expected,
        "observed": observed,
    }


def _failed_group(check_ids: Sequence[str], exc: Exception) -> list[dict[str, Any]]:
    observed = {"error_type": type(exc).__name__, "error": str(exc)}
    return [
        _check(check_id, False, expected="check completes and passes", observed=observed)
        for check_id in check_ids
    ]


def _run_group(
    check_ids: Sequence[str],
    producer: Callable[[], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        rows = producer()
        if tuple(row.get("id") for row in rows) != tuple(check_ids):
            raise ValueError("preflight check group returned an incomplete check inventory")
        return rows
    except Exception as exc:
        return _failed_group(check_ids, exc)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_paths = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty_paths), "dirty_paths": dirty_paths}


def _safe_git_state() -> dict[str, Any]:
    try:
        return _git_state()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "commit": None,
            "dirty": True,
            "dirty_paths": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _resolve_value(
    name: str,
    value: str | Path | None,
    errors: list[str],
) -> Path | None:
    if value is None or not str(value).strip():
        return None
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return None


def _unsafe_write_root(path: Path) -> bool:
    return path == Path(path.anchor) or path.is_relative_to(REPOSITORY_ROOT)


def resolve_paths(
    *,
    cache_root: str | Path | None = None,
    corpus_root: str | Path | None = None,
    reference_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> PreflightPaths:
    errors: list[str] = []
    cache = _resolve_value("cache_root", cache_root or os.environ.get("SRSCD_CACHE_ROOT"), errors)
    corpus = _resolve_value(
        "corpus_root", corpus_root or os.environ.get("PSEM_CORPUS_ROOT"), errors
    )
    reference = _resolve_value(
        "reference_root",
        reference_root or os.environ.get("PSEM_REFERENCE_ROOT"),
        errors,
    )
    output = _resolve_value("output_root", output_root, errors)
    if output is None:
        output = (
            (cache / OUTPUT_RELATIVE_PATH).resolve() if cache is not None else DEFAULT_OUTPUT_ROOT
        )
    if _unsafe_write_root(output):
        errors.append(f"output_root is not a safe external write root: {output}")
        output = DEFAULT_OUTPUT_ROOT
    return PreflightPaths(cache, corpus, reference, output, tuple(errors))


def _static_contract_checks(
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = {
        "contract_canonical_sha256": EXPECTED_CONTRACT_CANONICAL_SHA256,
        "config_canonical_sha256": EXPECTED_CONFIG_CANONICAL_SHA256,
        "authority": AUTHORITY,
        "experiment_id": EXPERIMENT_ID,
        "contract_version": CONTRACT_VERSION,
        "pretrained_checkpoint": {
            **EXPECTED_MODEL_IDENTITY,
            "source_registry_path": (
                "experiments/speaker_representation_scd/models/source_registry.json"
            ),
            "source_registry_sha256": EXPECTED_SOURCE_REGISTRY_SHA256,
        },
    }
    observed = {
        "contract_canonical_sha256": canonical_sha256(contract),
        "config_canonical_sha256": canonical_sha256(config),
        "authority": contract.get("authority"),
        "experiment_id": config.get("experiment_id"),
        "contract_version": config.get("contract_version"),
        "pretrained_checkpoint": config.get("pretrained_checkpoint"),
    }
    return [
        _check(
            "contract.issue_76_controls_exact",
            observed == expected,
            expected=expected,
            observed=observed,
        )
    ]


def _path_checks(paths: PreflightPaths) -> list[dict[str, Any]]:
    observed: dict[str, Any] = {"errors": list(paths.errors), "roots": {}}
    failures = list(paths.errors)
    for name in PATH_KEYS:
        path = getattr(paths, name)
        safe = path is not None
        if path is not None:
            try:
                safe = (
                    path.is_absolute()
                    and path == path.resolve()
                    and path != Path(path.anchor)
                    and not path.is_relative_to(REPOSITORY_ROOT)
                )
            except (OSError, RuntimeError, ValueError):
                safe = False
        observed["roots"][name] = {
            "path": str(path) if path is not None else None,
            "safe": safe,
        }
        if not safe:
            failures.append(name)
    return [
        _check(
            "paths.roots_safe",
            not failures,
            expected="all four resolved roots are non-root paths outside the repository",
            observed=observed,
        )
    ]


def _dataset_checks() -> list[dict[str, Any]]:
    freeze_path = DATA_DIR / "dataset_freeze.json"
    preflight_path = DATA_DIR / "preflight_report.json"
    freeze_digest = _file_sha256_or_none(freeze_path)
    preflight_digest = _file_sha256_or_none(preflight_path)
    checks = [
        _check(
            "dataset.freeze_file_identity",
            freeze_digest == EXPECTED_DATASET_FREEZE_SHA256,
            expected=EXPECTED_DATASET_FREEZE_SHA256,
            observed=freeze_digest,
        ),
        _check(
            "dataset.preflight_file_identity",
            preflight_digest == EXPECTED_DATASET_PREFLIGHT_SHA256,
            expected=EXPECTED_DATASET_PREFLIGHT_SHA256,
            observed=preflight_digest,
        ),
    ]
    expected_ready = {
        "id": "PSEM-STRATEGY-DATA-v2",
        "status": "frozen",
        "ready_for_issue_76": True,
        "check_count": 59,
    }
    try:
        freeze = validate_checked_dataset_freeze(DATA_DIR)
        observed_ready = {
            "id": freeze.get("dataset_freeze_id"),
            "status": freeze.get("freeze_status"),
            "ready_for_issue_76": freeze.get("preflight_binding", {}).get("ready_for_issue_76"),
            "check_count": freeze.get("preflight_binding", {}).get("check_count"),
        }
    except (DatasetFreezeError, OSError, TypeError, ValueError) as exc:
        observed_ready = {"error_type": type(exc).__name__, "error": str(exc)}
    checks.append(
        _check(
            "dataset.freeze_current_and_ready",
            observed_ready == expected_ready,
            expected=expected_ready,
            observed=observed_ready,
        )
    )
    return checks


def _model_checks(cache_root: Path | None) -> list[dict[str, Any]]:
    registry_sha256 = _file_sha256_or_none(SOURCE_REGISTRY_PATH)
    checks = [
        _check(
            "model.source_registry_identity",
            registry_sha256 == EXPECTED_SOURCE_REGISTRY_SHA256,
            expected=EXPECTED_SOURCE_REGISTRY_SHA256,
            observed=registry_sha256,
        )
    ]
    identity: Mapping[str, Any] | None = None
    try:
        registry = load_json(SOURCE_REGISTRY_PATH)
        models = registry.get("models") if isinstance(registry, dict) else None
        rows = (
            [
                row
                for row in models
                if isinstance(row, dict) and row.get("model_id") == "wavlm-base-plus"
            ]
            if isinstance(models, list)
            else []
        )
        identity = rows[0] if len(rows) == 1 else None
    except (OSError, TypeError, ValueError):
        identity = None
    observed_identity = (
        {key: identity.get(key) for key in EXPECTED_MODEL_IDENTITY}
        if identity is not None
        else None
    )
    checks.append(
        _check(
            "model.wavlm_registry_entry_exact",
            observed_identity == EXPECTED_MODEL_IDENTITY,
            expected=EXPECTED_MODEL_IDENTITY,
            observed=observed_identity,
        )
    )
    required_files = identity.get("required_files") if identity is not None else None
    verified_files: list[dict[str, Any]] = []
    failures: list[str] = []
    if cache_root is None or identity is None or not isinstance(required_files, list):
        failures.append("cache root or pinned registry identity is unavailable")
    else:
        cache = cache_root.resolve()
        model_root = (
            cache
            / "models"
            / EXPECTED_MODEL_IDENTITY["model_id"]
            / EXPECTED_MODEL_IDENTITY["revision"]
        ).resolve()
        if not model_root.is_relative_to(cache):
            failures.append("model root escapes cache root")
        else:
            for expected in required_files:
                if not isinstance(expected, dict):
                    failures.append("malformed required-file registry row")
                    continue
                relative = Path(str(expected.get("path", "")))
                if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                    failures.append(f"unsafe model path: {relative}")
                    continue
                path = (model_root / relative).resolve()
                if not path.is_relative_to(model_root) or not path.is_file():
                    failures.append(str(path))
                    continue
                actual = {
                    "path": relative.as_posix(),
                    "sha256": _file_sha256_or_none(path),
                    "size_bytes": path.stat().st_size,
                }
                verified_files.append(actual)
                if actual["sha256"] != expected.get("sha256") or actual[
                    "size_bytes"
                ] != expected.get("size_bytes"):
                    failures.append(str(path))
    checks.append(
        _check(
            "model.wavlm_checkpoint_files_exact",
            not failures
            and isinstance(required_files, list)
            and len(verified_files) == len(required_files),
            expected=(required_files if isinstance(required_files, list) else "pinned model files"),
            observed={"verified_files": verified_files, "failures": failures},
        )
    )
    return checks


def _source_rows() -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("source manifest row must be an object")
        rows.append(value)
    return rows


def _source_checks(
    corpus_root: Path | None,
    reference_root: Path | None,
    *,
    verify_source_bytes: bool,
) -> list[dict[str, Any]]:
    manifest_sha256 = _file_sha256_or_none(SOURCE_MANIFEST_PATH)
    checks = [
        _check(
            "sources.manifest_identity",
            manifest_sha256 == EXPECTED_SOURCE_MANIFEST_SHA256,
            expected=EXPECTED_SOURCE_MANIFEST_SHA256,
            observed=manifest_sha256,
        )
    ]
    verified = 0
    failures: list[str] = []
    try:
        source_rows = _source_rows()
    except (OSError, TypeError, ValueError) as exc:
        source_rows = []
        failures.append(f"{type(exc).__name__}: {exc}")
    if corpus_root is None or not corpus_root.is_dir():
        failures.append("PSEM_CORPUS_ROOT is unavailable")
    else:
        root = corpus_root.resolve()
        for index, row in enumerate(source_rows):
            source_id = str(row.get("source_id", f"row-{index}"))
            try:
                relative = Path(str(row["audio_ref"]))
                path = (root / relative).resolve()
                valid = (
                    not relative.is_absolute()
                    and ".." not in relative.parts
                    and path.is_relative_to(root)
                    and path.is_file()
                )
                if valid and verify_source_bytes:
                    valid = (
                        path.stat().st_size == row["waveform_size_bytes"]
                        and sha256_file(path) == row["waveform_sha256"]
                    )
            except (KeyError, OSError, TypeError, ValueError):
                valid = False
            if valid:
                verified += 1
            else:
                failures.append(source_id)
    checks.append(
        _check(
            "sources.bound_waveforms_resolve",
            len(source_rows) == EXPECTED_SOURCE_COUNT
            and verified == EXPECTED_SOURCE_COUNT
            and not failures,
            expected={
                "source_count": EXPECTED_SOURCE_COUNT,
                "byte_identity_verified": True,
            },
            observed={
                "source_count": len(source_rows),
                "verified": verified,
                "byte_identity_verified": verify_source_bytes,
                "failures": failures,
            },
        )
    )
    checks.append(
        _check(
            "sources.byte_identity_verification_enabled",
            verify_source_bytes,
            expected=True,
            observed=verify_source_bytes,
        )
    )
    try:
        checkout = open_reference_checkout(reference_root) if reference_root is not None else None
        observed_reference = dict(checkout.provenance) if checkout is not None else None
    except Exception as exc:
        observed_reference = {"error_type": type(exc).__name__, "error": str(exc)}
    reference_passed = (
        isinstance(observed_reference, dict)
        and observed_reference.get("repository") == EXPECTED_REFERENCE["repository"]
        and observed_reference.get("commit") == EXPECTED_REFERENCE["commit"]
    )
    checks.append(
        _check(
            "sources.forced_alignment_reference_exact",
            reference_passed,
            expected=EXPECTED_REFERENCE,
            observed=observed_reference,
        )
    )
    return checks


def _runtime_receipt_valid(
    receipt: Mapping[str, Any],
    receipt_name: str,
    artifact_role: str,
    binding: Mapping[str, Any],
    output_root: Path | None,
    corpus_root: Path | None = None,
    reference_root: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    payload = dict(receipt)
    digest = payload.pop("payload_sha256", None)
    rows = receipt.get("checks")
    check_ids = (
        [row.get("id") for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    )
    checks_valid = (
        isinstance(rows, list)
        and tuple(check_ids) == RUNTIME_CHECK_IDS[receipt_name]
        and all(
            set(row) == {"id", "passed", "expected", "observed"}
            and isinstance(row.get("id"), str)
            and bool(row["id"])
            and row.get("passed") is True
            for row in rows
            if isinstance(row, dict)
        )
    )
    try:
        generated_at = datetime.fromisoformat(str(receipt.get("generated_at")))
        timestamp_valid = generated_at.tzinfo is not None
    except ValueError:
        timestamp_valid = False
    details = receipt.get("details")
    descriptors = details.get("artifacts") if isinstance(details, Mapping) else None
    expected_relatives = tuple(Path(value) for value in RUNTIME_ARTIFACT_PATHS[receipt_name])
    details_schema_valid = (
        isinstance(details, Mapping)
        and set(details) == {"artifacts"}
        and isinstance(descriptors, list)
        and len(descriptors) == len(expected_relatives)
    )
    if not details_schema_valid:
        descriptors = []
    artifact_rows = []
    for descriptor, relative in zip(descriptors, expected_relatives):
        valid = False
        try:
            path = Path(str(descriptor["path"]))
            resolved = path.resolve()
            expected_path = (output_root / relative).resolve() if output_root is not None else None
            expected_keys = {"path", "sha256", "size_bytes"}
            if relative.suffix == ".json":
                expected_keys.add("canonical_sha256")
            valid = (
                output_root is not None
                and str(resolved) == str(path)
                and resolved == expected_path
                and set(descriptor) == expected_keys
                and isinstance(descriptor["size_bytes"], int)
                and not isinstance(descriptor["size_bytes"], bool)
                and descriptor["size_bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", str(descriptor["sha256"])) is not None
                and resolved.is_file()
                and resolved.stat().st_size == descriptor["size_bytes"]
                and sha256_file(resolved) == descriptor["sha256"]
            )
            if valid and relative.suffix == ".json":
                value = load_json(resolved)
                valid = (
                    re.fullmatch(r"[0-9a-f]{64}", str(descriptor["canonical_sha256"])) is not None
                    and canonical_sha256(value) == descriptor["canonical_sha256"]
                )
        except (KeyError, OSError, TypeError, ValueError):
            resolved = None
            valid = False
        artifact_rows.append(
            {"path": str(resolved) if resolved is not None else None, "valid": valid}
        )
    expected_artifacts = (
        {
            str((output_root / relative).resolve())
            for relative in RUNTIME_ARTIFACT_PATHS[receipt_name]
        }
        if output_root is not None
        else set()
    )
    artifacts_valid = {row["path"] for row in artifact_rows} == expected_artifacts and all(
        row["valid"] for row in artifact_rows
    )
    try:
        check_artifact_path = (
            output_root / RUNTIME_CHECK_ARTIFACT_PATHS[receipt_name]
            if output_root is not None
            else None
        )
        check_artifact = load_json(check_artifact_path) if check_artifact_path is not None else None
        validation_context: dict[str, Any] = {}
        if receipt_name in {"model_graphs", "gradient_canary", "weight_update_canary"}:
            parameter_artifact = load_json(output_root / "audits" / "parameter_inventory.json")
            validation_context["parameter_inventory"] = parameter_artifact
        if receipt_name == "arm_comparability":
            if corpus_root is None or reference_root is None:
                raise RuntimeEvidenceError("comparability validation roots are unavailable")
            from experiments.psem_training_strategy_gate.audit import comparability_provenance
            from experiments.psem_training_strategy_gate.augmentation import (
                augmentation_decision,
                validate_augmentation_decision,
            )
            from experiments.psem_training_strategy_gate.sampling import (
                TRAIN_ROLE,
                iter_rows,
                load_runtime_sessions,
                load_waveform_window,
                target_for_row,
            )

            row_id = check_artifact.get("row_id")
            matching_rows = [
                value
                for value in iter_rows(output_root / "manifests" / "sampling_manifest.jsonl")
                if value.get("row_id") == row_id
            ]
            if len(matching_rows) != 1:
                raise RuntimeEvidenceError("comparability row is absent or duplicated")
            row = matching_rows[0]
            if check_artifact.get("source_id") != row.get("source_id") or check_artifact.get(
                "boundary_sample"
            ) != row.get("boundary_sample"):
                raise RuntimeEvidenceError(
                    "comparability artifact is not bound to its sampling row"
                )
            decision = row.get("augmentation")
            validate_augmentation_decision(decision)
            if decision != augmentation_decision(str(row_id)):
                raise RuntimeEvidenceError("comparability augmentation is not row-bound")
            sessions = load_runtime_sessions(
                corpus_root,
                reference_root,
                roles=(TRAIN_ROLE,),
            )
            session = sessions[str(row["source_id"])]
            target = target_for_row(row, session)
            raw_waveform = load_waveform_window(row, session, corpus_root)
            validation_context["comparability"] = comparability_provenance(
                row,
                session,
                raw_waveform,
                target,
            )
        expected_semantic_checks = (
            runtime_artifact_checks(
                receipt_name,
                check_artifact,
                validation_context=validation_context,
            )
            if isinstance(check_artifact, Mapping)
            else None
        )
        semantic_checks_valid = rows == expected_semantic_checks
        if receipt_name in {"sampling_manifest", "augmentation_manifest"}:
            sampling_path = output_root / "manifests" / "sampling_manifest.jsonl"
            sampling_sha256 = sha256_file(sampling_path)
            if receipt_name == "sampling_manifest":
                if corpus_root is None or reference_root is None:
                    raise RuntimeEvidenceError("sampling validation roots are unavailable")
                from experiments.psem_training_strategy_gate.sampling import (
                    TRAIN_ROLE,
                    load_runtime_sessions,
                    validate_sampling_manifest,
                )

                sessions = load_runtime_sessions(
                    corpus_root,
                    reference_root,
                    roles=(TRAIN_ROLE,),
                )
                observed_sampling = validate_sampling_manifest(sampling_path, sessions)
                semantic_checks_valid = (
                    semantic_checks_valid
                    and check_artifact.get("manifest_sha256") == sampling_sha256
                    and all(
                        check_artifact.get(field) == value
                        for field, value in observed_sampling.items()
                    )
                )
            else:
                semantic_checks_valid = (
                    semantic_checks_valid
                    and check_artifact.get("sampling_manifest_sha256") == sampling_sha256
                )
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        semantic_checks_valid = False
    observed = {
        "top_level_schema_exact": set(receipt)
        == {
            "schema_version",
            "artifact_role",
            "status",
            "generated_at",
            "binding",
            "checks",
            "details",
            "payload_sha256",
        },
        "schema_version": receipt.get("schema_version"),
        "artifact_role": receipt.get("artifact_role"),
        "status": receipt.get("status"),
        "binding": receipt.get("binding"),
        "check_ids": check_ids,
        "checks_valid": checks_valid,
        "semantic_checks_valid": semantic_checks_valid,
        "details_present": isinstance(details, dict) and bool(details),
        "details_schema_valid": details_schema_valid,
        "timestamp_valid": timestamp_valid,
        "artifacts": artifact_rows,
        "artifacts_valid": artifacts_valid,
        "payload_sha256_valid": (
            isinstance(digest, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
            and digest == canonical_sha256(payload)
        ),
    }
    passed = (
        observed["top_level_schema_exact"] is True
        and observed["schema_version"] == 1
        and observed["artifact_role"] == artifact_role
        and observed["status"] == "pass"
        and observed["binding"] == dict(binding)
        and observed["checks_valid"] is True
        and observed["semantic_checks_valid"] is True
        and observed["details_present"] is True
        and observed["details_schema_valid"] is True
        and observed["timestamp_valid"] is True
        and observed["artifacts_valid"] is True
        and observed["payload_sha256_valid"] is True
    )
    return passed, observed


def _receipt_checks(
    output_root: Path | None,
    binding: Mapping[str, Any],
    corpus_root: Path | None = None,
    reference_root: Path | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, artifact_role in RUNTIME_RECEIPTS.items():
        path = output_root / "preflight" / f"{name}.json" if output_root is not None else None
        try:
            value = load_json(path) if path is not None else None
            receipt = value if isinstance(value, dict) else None
        except (OSError, TypeError, ValueError):
            receipt = None
        if receipt is None:
            passed = False
            observed = None
        else:
            passed, observed = _runtime_receipt_valid(
                receipt,
                name,
                artifact_role,
                binding,
                output_root,
                corpus_root,
                reference_root,
            )
        checks.append(
            _check(
                f"runtime_receipt.{name}",
                passed,
                expected={
                    "schema_version": 1,
                    "artifact_role": artifact_role,
                    "status": "pass",
                    "binding": dict(binding),
                    "checks": "nonempty, unique, complete, and all passing",
                    "check_ids": list(RUNTIME_CHECK_IDS[name]),
                    "artifacts": list(RUNTIME_ARTIFACT_PATHS[name]),
                    "payload_sha256_valid": True,
                },
                observed=observed,
            )
        )
    return checks


def _binding(git: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": _file_sha256_or_none(CONTRACT_PATH),
        "config_sha256": _file_sha256_or_none(CONFIG_PATH),
        "dataset_freeze_sha256": _file_sha256_or_none(DATA_DIR / "dataset_freeze.json"),
        "source_manifest_sha256": _file_sha256_or_none(SOURCE_MANIFEST_PATH),
        "source_registry_sha256": _file_sha256_or_none(SOURCE_REGISTRY_PATH),
        "label_generator_sha256": _file_sha256_or_none(LABEL_GENERATOR_PATH),
        "git_commit": git.get("commit"),
    }


def build_preflight(
    paths: PreflightPaths,
    *,
    verify_source_bytes: bool = True,
) -> dict[str, Any]:
    contract = _mapping_or_empty(CONTRACT_PATH)
    config = _mapping_or_empty(CONFIG_PATH)
    git = _safe_git_state()
    binding = _binding(git)
    checks: list[dict[str, Any]] = []
    checks.extend(
        _run_group(
            ("contract.issue_76_controls_exact",),
            lambda: _static_contract_checks(contract, config),
        )
    )
    checks.extend(_run_group(("paths.roots_safe",), lambda: _path_checks(paths)))
    checks.extend(
        _run_group(
            (
                "dataset.freeze_file_identity",
                "dataset.preflight_file_identity",
                "dataset.freeze_current_and_ready",
            ),
            _dataset_checks,
        )
    )
    checks.extend(
        _run_group(
            ("labels.generator_identity",),
            lambda: [
                _check(
                    "labels.generator_identity",
                    _file_sha256_or_none(LABEL_GENERATOR_PATH) == EXPECTED_LABEL_GENERATOR_SHA256,
                    expected=EXPECTED_LABEL_GENERATOR_SHA256,
                    observed=_file_sha256_or_none(LABEL_GENERATOR_PATH),
                )
            ],
        )
    )
    checks.extend(
        _run_group(
            (
                "model.source_registry_identity",
                "model.wavlm_registry_entry_exact",
                "model.wavlm_checkpoint_files_exact",
            ),
            lambda: _model_checks(paths.cache_root),
        )
    )
    checks.extend(
        _run_group(
            (
                "sources.manifest_identity",
                "sources.bound_waveforms_resolve",
                "sources.byte_identity_verification_enabled",
                "sources.forced_alignment_reference_exact",
            ),
            lambda: _source_checks(
                paths.corpus_root,
                paths.reference_root,
                verify_source_bytes=verify_source_bytes,
            ),
        )
    )
    runtime_ids = tuple(f"runtime_receipt.{name}" for name in RUNTIME_RECEIPTS)
    checks.extend(
        _run_group(
            runtime_ids,
            lambda: _receipt_checks(
                paths.output_root,
                binding,
                paths.corpus_root,
                paths.reference_root,
            ),
        )
    )
    checks.extend(
        _run_group(
            ("git.candidate_is_clean",),
            lambda: [
                _check(
                    "git.candidate_is_clean",
                    git.get("dirty") is False,
                    expected={"dirty": False},
                    observed=git,
                )
            ],
        )
    )
    failed = [row["id"] for row in checks if not row["passed"]]
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_experiment_preflight",
        "experiment_id": EXPERIMENT_ID,
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "authority": AUTHORITY,
        "binding": binding,
        "git": git,
        "paths": {
            "cache_root": str(paths.cache_root) if paths.cache_root is not None else None,
            "corpus_root": str(paths.corpus_root) if paths.corpus_root is not None else None,
            "reference_root": (
                str(paths.reference_root) if paths.reference_root is not None else None
            ),
            "output_root": str(paths.output_root) if paths.output_root is not None else None,
        },
        "checks": checks,
        "failed_checks": failed,
        "ready_for_material_run": not failed,
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def write_preflight(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_receipt_shape(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path | None,
) -> None:
    if set(receipt) != RECEIPT_KEYS:
        raise ExperimentPreflightError("experiment preflight receipt schema is incomplete")
    payload = dict(receipt)
    digest = payload.pop("payload_sha256")
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or digest != canonical_sha256(payload)
    ):
        raise ExperimentPreflightError("experiment preflight receipt digest is invalid")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_role") != "psem_experiment_preflight"
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("contract_version") != CONTRACT_VERSION
        or receipt.get("authority") != AUTHORITY
    ):
        raise ExperimentPreflightError("experiment preflight receipt identity is invalid")
    try:
        generated_at = datetime.fromisoformat(str(receipt.get("generated_at")))
    except ValueError as exc:
        raise ExperimentPreflightError("experiment preflight timestamp is invalid") from exc
    if generated_at.tzinfo is None:
        raise ExperimentPreflightError("experiment preflight timestamp is not timezone-aware")
    binding = receipt.get("binding")
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
        raise ExperimentPreflightError("experiment preflight binding is incomplete")
    git = receipt.get("git")
    if (
        not isinstance(git, dict)
        or set(git) != {"commit", "dirty", "dirty_paths"}
        or git.get("dirty") is not False
        or git.get("dirty_paths") != []
        or not re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit")))
    ):
        raise ExperimentPreflightError("experiment preflight Git state is invalid")
    if (
        binding["experiment_id"] != EXPERIMENT_ID
        or binding["git_commit"] != git["commit"]
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(binding[key]))
            for key in BINDING_KEYS - {"experiment_id", "git_commit"}
        )
    ):
        raise ExperimentPreflightError("experiment preflight binding identity is invalid")
    paths = receipt.get("paths")
    if not isinstance(paths, dict) or set(paths) != PATH_KEYS:
        raise ExperimentPreflightError("experiment preflight paths are incomplete")
    for value in paths.values():
        if not isinstance(value, str) or not value or str(Path(value).resolve()) != value:
            raise ExperimentPreflightError("experiment preflight path is not canonical")
    rows = receipt.get("checks")
    if (
        not isinstance(rows, list)
        or tuple(row.get("id") for row in rows if isinstance(row, dict)) != EXPECTED_CHECK_IDS
        or any(
            not isinstance(row, dict)
            or set(row) != {"id", "passed", "expected", "observed"}
            or row.get("passed") is not True
            for row in rows
        )
    ):
        raise ExperimentPreflightError("experiment preflight check inventory is incomplete")
    if receipt.get("ready_for_material_run") is not True or receipt.get("failed_checks") != []:
        raise ExperimentPreflightError("material work is blocked by experiment preflight")
    if receipt_path is not None:
        expected_path = Path(paths["output_root"]) / "preflight" / "experiment_receipt.json"
        if receipt_path.resolve() != expected_path.resolve():
            raise ExperimentPreflightError("experiment preflight receipt path is invalid")


def require_passing_preflight(path: Path) -> dict[str, Any]:
    try:
        value = load_json(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ExperimentPreflightError("passing experiment preflight receipt is required") from exc
    if not isinstance(value, dict):
        raise ExperimentPreflightError("experiment preflight receipt must be an object")
    receipt = value
    _validate_receipt_shape(receipt, receipt_path=path)
    recorded_paths = receipt["paths"]
    paths = resolve_paths(
        cache_root=recorded_paths["cache_root"],
        corpus_root=recorded_paths["corpus_root"],
        reference_root=recorded_paths["reference_root"],
        output_root=recorded_paths["output_root"],
    )
    current = build_preflight(paths, verify_source_bytes=True)
    try:
        _validate_receipt_shape(current, receipt_path=None)
    except ExperimentPreflightError as exc:
        raise ExperimentPreflightError(
            "material work is blocked because current preflight revalidation failed"
        ) from exc
    stable_current = {
        key: value
        for key, value in current.items()
        if key not in {"generated_at", "payload_sha256"}
    }
    stable_receipt = {
        key: value
        for key, value in receipt.items()
        if key not in {"generated_at", "payload_sha256"}
    }
    if stable_current != stable_receipt:
        raise ExperimentPreflightError("experiment preflight receipt is stale")
    return receipt
