from __future__ import annotations

import hashlib
import ntpath
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

DISTRIBUTION_EVIDENCE_SCHEMA = "puripuly-heart/windows-process-distribution/v4"
PRODUCTION_APP_ID = "{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"
APPROVED_ALTERNATE_APP_ID = "{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}"
CLIENT_KEYS = ("vrchat", "discord_stable", "discord_ptb", "discord_canary")
ClientStatus = Literal["passed", "waived", "unavailable", "not_run"]


@dataclass(frozen=True, slots=True)
class ManualClientCell:
    status: ClientStatus
    result: str | None
    multilevel_ancestry_risk: bool
    waiver_authority: str | None = None
    waiver_reason: str | None = None


def validate_installer_isolation(*, app_id: str, install_dir: Path, workspace_root: Path) -> None:
    normalized_app_id = app_id.strip().strip("{").strip("}").casefold()
    normalized_production = PRODUCTION_APP_ID.strip("{").strip("}").casefold()
    if normalized_app_id == normalized_production or not normalized_app_id:
        raise ValueError("installer smoke requires a non-production alternate AppId")
    resolved_install = install_dir.resolve()
    resolved_workspace = workspace_root.resolve()
    if resolved_install == resolved_workspace or resolved_workspace in resolved_install.parents:
        raise ValueError("installer smoke directory must be outside the workspace")
    normalized = ntpath.normcase(str(resolved_install))
    if normalized.endswith(ntpath.normcase(r"Program Files\PuriPulyHeart")):
        raise ValueError("installer smoke directory must not use the production install path")


def validate_runtime_report(report: dict[str, object], *, expected_root: Path) -> None:
    required = {
        "schema",
        "status",
        "proctap_version",
        "proctap_module",
        "runtime_native_module",
        "runtime_native_sha256",
        "artifact_native_module",
        "artifact_native_sha256",
        "runtime_loaded_modules",
        "runtime_executable",
        "runtime_executable_sha256",
        "native_process_specific",
        "capture_started",
        "capture_closed",
        "device_fallback_used",
        "credentials_used",
        "network_used",
    }
    if set(report) != required or report.get("schema") != (
        "puripuly-heart/process-capture-packaged-smoke/v2"
    ):
        raise ValueError("invalid process-capture runtime report schema")
    if (
        report.get("status") != "passed"
        or report.get("proctap_version") != "1.1.1"
        or any(
            report.get(name) is not True
            for name in ("native_process_specific", "capture_started", "capture_closed")
        )
        or any(
            report.get(name) is not False
            for name in ("device_fallback_used", "credentials_used", "network_used")
        )
    ):
        raise ValueError("process-capture runtime report did not pass strict validation")
    expected = expected_root.resolve()
    native_path = Path(str(report["artifact_native_module"])).resolve()
    if (
        native_path.parent != expected / "site-packages" / "proctap"
        or Path(str(report["runtime_native_module"])).resolve() != native_path
        or report["runtime_native_sha256"] != report["artifact_native_sha256"]
    ):
        raise ValueError("reported ProcTap native module does not match the native payload")
    executable = Path(str(report["runtime_executable"])).resolve()
    if executable != expected / "python.exe":
        raise ValueError("reported runtime executable does not match the native payload")
    modules = report["runtime_loaded_modules"]
    if not isinstance(modules, dict) or set(modules) != {
        "platform",
        "process_source",
        "proctap",
        "proctap_native",
    }:
        raise ValueError("invalid native runtime module provenance")
    for record in modules.values():
        if (
            not isinstance(record, dict)
            or set(record) != {"origin", "artifact", "sha256"}
            or not Path(str(record["origin"])).resolve().is_relative_to(expected)
            or not Path(str(record["artifact"])).resolve().is_relative_to(expected)
        ):
            raise ValueError("runtime module originated outside the expected artifact root")
    associations = [
        (native_path, report["artifact_native_sha256"]),
        (executable, report["runtime_executable_sha256"]),
        *((Path(str(record["artifact"])), record["sha256"]) for record in modules.values()),
    ]
    for path, digest in associations:
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError("reported runtime artifact hash does not match")


def build_installer_provenance(
    *,
    release_installer_path: Path,
    isolated_installer_path: Path,
    isolated_install_log_path: Path,
    alternate_app_id: str,
    workspace_root: Path,
) -> dict[str, object]:
    if alternate_app_id != APPROVED_ALTERNATE_APP_ID:
        raise ValueError("isolated installer provenance requires the approved alternate AppId")
    release = release_installer_path.resolve()
    isolated = isolated_installer_path.resolve()
    log = isolated_install_log_path.resolve()
    workspace = workspace_root.resolve()
    expected_release_dir = workspace / "installer_output"
    if expected_release_dir != release.parent:
        raise ValueError("release installer must come from the workspace release output")
    if workspace == isolated or workspace in isolated.parents:
        raise ValueError("isolated installer artifact must be outside the workspace")
    for path in (release, isolated, log):
        if not path.is_file():
            raise ValueError("installer provenance input is missing")
    release_hash = _sha256(release)
    isolated_hash = _sha256(isolated)
    if release_hash == isolated_hash:
        raise ValueError("release and isolated installer artifacts must be distinct")
    isolated_timestamp = isolated.stat().st_mtime
    log_timestamp = log.stat().st_mtime
    if log_timestamp < isolated_timestamp:
        raise ValueError("isolated installer log predates the installer artifact")
    value = {
        "release_installer_path": str(release),
        "release_installer_sha256": release_hash,
        "isolated_installer_path": str(isolated),
        "isolated_installer_sha256": isolated_hash,
        "isolated_installer_last_write_utc": _utc_timestamp(isolated_timestamp),
        "isolated_install_log_path": str(log),
        "isolated_install_log_sha256": _sha256(log),
        "isolated_install_log_last_write_utc": _utc_timestamp(log_timestamp),
    }
    validate_installer_provenance(value, validate_files=True)
    return value


def validate_installer_provenance(value: object, *, validate_files: bool = False) -> None:
    required = {
        "release_installer_path",
        "release_installer_sha256",
        "isolated_installer_path",
        "isolated_installer_sha256",
        "isolated_installer_last_write_utc",
        "isolated_install_log_path",
        "isolated_install_log_sha256",
        "isolated_install_log_last_write_utc",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("invalid installer provenance schema")
    release_hash = value["release_installer_sha256"]
    isolated_hash = value["isolated_installer_sha256"]
    log_hash = value["isolated_install_log_sha256"]
    if not all(
        isinstance(item, str)
        and len(item) == 64
        and all(character in "0123456789abcdef" for character in item)
        for item in (release_hash, isolated_hash, log_hash)
    ):
        raise ValueError("invalid installer provenance hash")
    if release_hash == isolated_hash:
        raise ValueError("release and isolated installer hashes cannot be swapped or equal")
    if validate_files:
        file_hashes = (
            (Path(str(value["release_installer_path"])), release_hash),
            (Path(str(value["isolated_installer_path"])), isolated_hash),
            (Path(str(value["isolated_install_log_path"])), log_hash),
        )
        for path, expected_hash in file_hashes:
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError("installer provenance file/hash association is invalid")
    isolated_time = datetime.fromisoformat(str(value["isolated_installer_last_write_utc"]))
    log_time = datetime.fromisoformat(str(value["isolated_install_log_last_write_utc"]))
    if log_time < isolated_time:
        raise ValueError("isolated installer log timestamp is not associated with the artifact")


def validate_manual_matrix(matrix: dict[str, ManualClientCell]) -> None:
    if tuple(matrix) != CLIENT_KEYS:
        raise ValueError("manual client matrix keys or order are invalid")
    for cell in matrix.values():
        if cell.status == "passed" and not cell.result:
            raise ValueError("passed manual client cells require a result")
        if cell.status == "waived" and (
            cell.result != "not_tested"
            or cell.waiver_authority != "acceptance_authority"
            or not cell.waiver_reason
        ):
            raise ValueError("waived manual client cells require explicit acceptance authority")
        if cell.status not in {"passed", "waived"} and cell.result is not None:
            raise ValueError("unavailable or not-run cells cannot claim a result")
        if cell.status != "waived" and (
            cell.waiver_authority is not None or cell.waiver_reason is not None
        ):
            raise ValueError("only waived manual client cells can include waiver facts")
        if not cell.multilevel_ancestry_risk:
            raise ValueError("manual matrix must carry the multilevel ancestry risk")


def validate_distribution_evidence(evidence: dict[str, object]) -> None:
    required = {
        "schema",
        "status",
        "classification",
        "supported_target",
        "packaged",
        "installed",
        "installer_isolation",
        "manual_matrix",
        "manual_matrix_complete",
        "manual_matrix_status",
        "technical_status",
        "overlay",
        "workflow",
        "commands",
    }
    if set(evidence) != required or evidence.get("schema") != DISTRIBUTION_EVIDENCE_SCHEMA:
        raise ValueError("invalid Windows process distribution evidence schema")
    if evidence.get("status") not in {
        "passed",
        "closed_with_waiver",
        "partial",
        "failed",
        "blocked",
    }:
        raise ValueError("invalid Windows process distribution evidence status")
    installer_isolation = evidence.get("installer_isolation")
    if not isinstance(installer_isolation, dict):
        raise ValueError("invalid installer isolation evidence")
    validate_installer_provenance(installer_isolation.get("installer_provenance"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
