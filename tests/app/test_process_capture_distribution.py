from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.windows_process_distribution import (
    APPROVED_ALTERNATE_APP_ID,
    CLIENT_KEYS,
    DISTRIBUTION_EVIDENCE_SCHEMA,
    PRODUCTION_APP_ID,
    ManualClientCell,
    build_installer_provenance,
    validate_distribution_evidence,
    validate_installer_isolation,
    validate_installer_provenance,
    validate_manual_matrix,
    validate_runtime_report,
)
from tests.helpers.paths import REPO_ROOT as ROOT


def _provenance_fixture(tmp_path: Path) -> dict[str, object]:
    release = tmp_path / "release.exe"
    isolated = tmp_path / "isolated.exe"
    log = tmp_path / "install.log"
    release.write_bytes(b"release")
    isolated.write_bytes(b"isolated")
    log.write_bytes(b"log")
    return {
        "release_installer_path": str(release),
        "release_installer_sha256": hashlib.sha256(b"release").hexdigest(),
        "isolated_installer_path": str(isolated),
        "isolated_installer_sha256": hashlib.sha256(b"isolated").hexdigest(),
        "isolated_installer_last_write_utc": "2026-07-10T18:13:25Z",
        "isolated_install_log_path": str(log),
        "isolated_install_log_sha256": hashlib.sha256(b"log").hexdigest(),
        "isolated_install_log_last_write_utc": "2026-07-10T18:13:33Z",
    }


def test_build_spec_collects_pinned_proctap_hidden_imports_and_native_binary() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert 'collect_dynamic_libs("proctap", destdir="proctap")' in spec
    assert 'get_module_file_attribute("proctap._native")' in spec
    assert 'collect_submodules("proctap")' in spec
    assert '"proctap", "proctap._native", "proctap.backends.windows"' in spec
    assert "Pinned ProcTap package did not provide a packageable _native extension" in spec


def test_release_workflow_runs_packaged_installed_strict_smoke_and_alternate_installer() -> None:
    script = (ROOT / "scripts/ci/build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert script.count("Invoke-ProcessCaptureRuntimeSmokeCheck") >= 4
    assert '"/DMyAppId=$InstallerTestAppId"' in script
    assert '$InstallerTestAppId = "{{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}"' in script
    assert '"/DSkipLocalSttProvisioning=1"' in script
    assert 'ArgumentList @("process-capture-runtime-check")' not in script
    assert "PURIPULY_HEART_RELEASE_PROCESS_CAPTURE_SMOKE" in script
    assert '"/DProcessCaptureSmokeArtifactRoot=$processCaptureSmokeArtifactRoot"' in script
    assert "native_process_specific" in script
    assert "device_fallback_used" in script


def test_installer_smoke_skip_is_compile_time_only_and_production_default_is_unchanged() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "#ifdef SkipLocalSttProvisioning" in script
    assert "Local STT provisioning skipped for isolated installer smoke." in script
    assert f'#define MyAppId "{{{PRODUCTION_APP_ID}}}"' not in script
    assert '#define MyAppId "{{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"' in script
    assert "#ifdef ProcessCaptureSmokeArtifactRoot" in script
    assert 'DestDir: "{app}\\process-capture-smoke"' in script


def test_production_build_and_cli_omit_release_only_smoke_helper() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")
    main_source = (ROOT / "src/puripuly_heart/main.py").read_text(encoding="utf-8")
    helper_spec = (ROOT / "scripts/release/process-capture-runtime-smoke.spec").read_text(
        encoding="utf-8"
    )

    assert 'os.environ.get("PURIPULY_HEART_RELEASE_PROCESS_CAPTURE_SMOKE") == "1"' in spec
    assert "noarchive=release_smoke" in spec
    assert "process-capture-runtime-check" not in main_source
    assert "must not collect duplicate production modules" in helper_spec


def test_installer_isolation_rejects_production_identity_and_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    install = tmp_path / "isolated-install"
    workspace.mkdir()

    validate_installer_isolation(
        app_id="{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}",
        install_dir=install,
        workspace_root=workspace,
    )
    with pytest.raises(ValueError, match="alternate AppId"):
        validate_installer_isolation(
            app_id=PRODUCTION_APP_ID,
            install_dir=install,
            workspace_root=workspace,
        )
    with pytest.raises(ValueError, match="outside the workspace"):
        validate_installer_isolation(
            app_id="{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}",
            install_dir=workspace / "install",
            workspace_root=workspace,
        )


def test_runtime_report_requires_native_hash_strict_mode_and_no_fallback(tmp_path: Path) -> None:
    native = tmp_path / "proctap" / "_native.cp312-win_amd64.pyd"
    native.parent.mkdir()
    native.write_bytes(b"native")
    helper = tmp_path / "PuriPulyHeartProcessCaptureSmoke.exe"
    helper.write_bytes(b"helper")
    report = {
        "schema": "puripuly-heart/process-capture-packaged-smoke/v1",
        "status": "passed",
        "proctap_version": "1.1.1",
        "proctap_module": str(tmp_path / "proctap" / "__init__.py"),
        "runtime_native_module": str(native),
        "runtime_native_sha256": hashlib.sha256(b"native").hexdigest(),
        "artifact_native_module": str(native),
        "artifact_native_sha256": hashlib.sha256(b"native").hexdigest(),
        "helper_executable": str(helper),
        "helper_executable_sha256": hashlib.sha256(b"helper").hexdigest(),
        "native_process_specific": True,
        "capture_started": True,
        "device_fallback_used": False,
        "credentials_used": False,
        "network_used": False,
        "release_only_helper": True,
    }

    validate_runtime_report(report, expected_root=tmp_path)
    with pytest.raises(ValueError, match="strict validation"):
        validate_runtime_report({**report, "device_fallback_used": True}, expected_root=tmp_path)


def test_manual_matrix_cannot_claim_unavailable_clients_and_carries_risk() -> None:
    matrix = {
        key: ManualClientCell(
            status="unavailable",
            result=None,
            multilevel_ancestry_risk=True,
        )
        for key in CLIENT_KEYS
    }

    validate_manual_matrix(matrix)
    matrix["vrchat"] = ManualClientCell(
        status="unavailable", result="passed", multilevel_ancestry_risk=True
    )
    with pytest.raises(ValueError, match="cannot claim"):
        validate_manual_matrix(matrix)


def test_distribution_evidence_schema_is_strict(tmp_path: Path) -> None:
    evidence = {
        "schema": DISTRIBUTION_EVIDENCE_SCHEMA,
        "status": "passed",
        "classification": None,
        "supported_target": {},
        "packaged": {},
        "installed": {},
        "installer_isolation": {"installer_provenance": _provenance_fixture(tmp_path)},
        "manual_matrix": {},
        "manual_matrix_complete": False,
        "manual_matrix_status": "waived",
        "technical_status": "passed",
        "overlay": {},
        "workflow": {},
        "release_only_smoke": {},
        "commands": [],
    }

    validate_distribution_evidence(evidence)
    with pytest.raises(ValueError, match="schema"):
        validate_distribution_evidence({**evidence, "extra": json.loads("true")})


def test_installer_provenance_generator_distinguishes_release_and_isolated_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    release_dir = workspace / "installer_output"
    isolated_dir = tmp_path / "isolated"
    release_dir.mkdir(parents=True)
    isolated_dir.mkdir()
    release = release_dir / "setup.exe"
    isolated = isolated_dir / "setup.exe"
    log = isolated_dir / "install.log"
    release.write_bytes(b"release-identity")
    isolated.write_bytes(b"alternate-identity")
    log.write_bytes(b"installed alternate")

    provenance = build_installer_provenance(
        release_installer_path=release,
        isolated_installer_path=isolated,
        isolated_install_log_path=log,
        alternate_app_id=APPROVED_ALTERNATE_APP_ID,
        workspace_root=workspace,
    )

    assert provenance["release_installer_sha256"] == hashlib.sha256(b"release-identity").hexdigest()
    assert (
        provenance["isolated_installer_sha256"] == hashlib.sha256(b"alternate-identity").hexdigest()
    )
    swapped = {
        **provenance,
        "release_installer_sha256": provenance["isolated_installer_sha256"],
        "isolated_installer_sha256": provenance["release_installer_sha256"],
    }
    with pytest.raises(ValueError, match="file/hash association"):
        validate_installer_provenance(swapped, validate_files=True)

    with pytest.raises(ValueError, match="workspace release output"):
        build_installer_provenance(
            release_installer_path=isolated,
            isolated_installer_path=release,
            isolated_install_log_path=log,
            alternate_app_id=APPROVED_ALTERNATE_APP_ID,
            workspace_root=workspace,
        )


def test_release_workflow_uses_short_overlay_target_and_verifies_current_version() -> None:
    script = (ROOT / "scripts/ci/build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "PURIPULY_HEART_RELEASE_BUILD_ROOT" in script
    assert 'Join-Path $env:TEMP "PuriPulyHeart-ReleaseBuild-$AppVersion"' in script
    assert "$overlayReleasePath --version" in script
    assert 'throw "Rust overlay version mismatch: expected $AppVersion' in script
