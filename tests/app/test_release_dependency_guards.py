from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from puripuly_heart.core.local_qwen_runtime import LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR

from puripuly_heart.core.overlay.openvr_vendor import (
    OPENVR_VENDOR_DLL_URL,
    OPENVR_VENDOR_LICENSE_URL,
    OPENVR_VENDOR_REPOSITORY_REF,
)
from tests.helpers.paths import REPO_ROOT as ROOT

PINNED_PYTHON_VERSION = 'PYTHON_VERSION: "3.12.10"'
PINNED_UV_VERSION = 'UV_VERSION: "0.9.17"'
PINNED_INNOSETUP_VERSION = 'INNOSETUP_VERSION: "6.6.1"'
SHARED_SETUP_ACTION = "./.github/actions/setup-uv-environment"
PINNED_SOXR_SPECIFIER = "soxr==1.1.0"
SOXR_RELEASE_INPUTS_SCRIPT = "scripts/ci/prepare-soxr-release-inputs.ps1"
FLET_RUNTIME_PREPARATION_SCRIPT = "scripts/ci/prepare-flet-runtime.ps1"
FLET_RUNTIME_VERSION = "0.86.1"
FLET_RUNTIME_SHA256 = "2cf0865b31bd0e394a24a6c2d270e084cf9dad9c711e0b5d0cf9fa9bfac31e14"
SOXR_RELEASE_INPUTS_MANIFEST_PATH = "build/soxr-release-inputs/manifest.json"
SOXR_PACKAGED_RUNTIME_RELATIVE_DIR = "soxr"
SOXR_COMPLIANCE_BUNDLE_RELATIVE_DIR = "third_party\\soxr"
SOXR_SOURCE_BUNDLE_NAME = "PuriPulyHeart-soxr-third-party-source-bundle.zip"
SOXR_LICENSE_TEXT_RELATIVE_PATH = "src/puripuly_heart/data/licenses/COPYING.LGPL-2.1.txt"
OPENVR_VENDOR_DLL_RELATIVE_PATH = "third_party/openvr/win64/openvr_api.dll"
OPENVR_VENDOR_SHA256_RELATIVE_PATH = "third_party/openvr/win64/openvr_api.dll.sha256"
OPENVR_VENDOR_LICENSE_RELATIVE_PATH = "third_party/openvr/LICENSE"
OPENVR_VENDOR_README_RELATIVE_PATH = "third_party/openvr/README.md"
OPENVR_VENDOR_DLL_SHA256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"
OPENVR_VENDOR_SHA256_LINE = f"{OPENVR_VENDOR_DLL_SHA256} *openvr_api.dll"
PINNED_LIBSOXR_SOURCE_URL = (
    "https://sourceforge.net/projects/soxr/files/soxr-0.1.3-Source.tar.xz/download"
)
PINNED_LIBSOXR_SOURCE_SHA256 = "b111c15fdc8c029989330ff559184198c161100a59312f5dc19ddeb9b5a15889"
README_INSTALLER_BUILD_COMMAND = "ISCC installer.iss"
FULL_WINDOWS_RELEASE_SCRIPT = "scripts/ci/build-release-artifacts.ps1"
BUILD_SPEC_DIRECT_PACKAGING_HEADER = (
    "Direct Windows PyInstaller packaging (executable-only / manual installer packaging):"
)
BUILD_SPEC_DIRECT_PACKAGING_CAVEAT = (
    "This direct path is not the release-complete compliance-packaging path and requires "
    "the staged overlay executable at build/overlay/PuriPulyHeartOverlay.exe plus the "
    "vendored OpenVR bundle under third_party/openvr/ (enforced below)."
)
BUILD_SPEC_FULL_RELEASE_HEADER = "Full release-complete compliance packaging requires scripts/ci/prepare-soxr-release-inputs.ps1 before scripts/ci/build-release-artifacts.ps1:"
OPENVR_NOTICE_HEADER = "OpenVR client binding library (openvr_api.dll)"
OPENVR_NOTICE_SOURCE_BUNDLE_RELATIVE_DIR = "third_party\\openvr\\"
OPENVR_NOTICE_APP_PRIVATE_EXPLANATION = (
    "This application bundles the Windows x64 OpenVR client binding library as an app-private "
    "dependency for the packaged overlay/runtime path."
)
OPENVR_NOTICE_PACKAGED_BUILD_EXPLANATION = (
    "Installed builds load this DLL from the application's own tree. The vendored bundle pinned "
    f"from {OPENVR_VENDOR_REPOSITORY_REF} under {OPENVR_NOTICE_SOURCE_BUNDLE_RELATIVE_DIR} is the "
    "packaging source for that app-private DLL, so packaged and installed builds do not depend on "
    "a shared SteamVR system copy."
)
OPENVR_NOTICE_NEXT_SECTION_HEADER = "Noto Sans CJK Medium TTC"


def _slice_section(text: str, start_marker: str, end_marker: str | None = None) -> str:
    start_index = text.index(start_marker)

    if end_marker is None:
        return text[start_index:]

    end_index = text.index(end_marker, start_index)
    return text[start_index:end_index]


def _workflow_job_block(workflow: str, job_name: str) -> str:
    lines = workflow.splitlines()
    start_index = lines.index(f"  {job_name}:")
    end_index = len(lines)

    for index in range(start_index + 1, len(lines)):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            end_index = index
            break

    return "\n".join(lines[start_index:end_index])


def _expected_openvr_notice_section() -> str:
    openvr_license_text = (
        (ROOT / OPENVR_VENDOR_LICENSE_RELATIVE_PATH).read_text(encoding="utf-8").strip()
    )

    return (
        f"{OPENVR_NOTICE_HEADER}\n"
        f"Upstream pin: {OPENVR_VENDOR_REPOSITORY_REF}\n"
        f"DLL source: {OPENVR_VENDOR_DLL_URL}\n"
        f"LICENSE source: {OPENVR_VENDOR_LICENSE_URL}\n"
        "License: BSD-3-Clause\n"
        "Bundled runtime: openvr_api.dll\n"
        f"Packaging source bundle: {OPENVR_NOTICE_SOURCE_BUNDLE_RELATIVE_DIR}\n"
        "Packaging source files: LICENSE ; README.md ; win64\\openvr_api.dll ; "
        "win64\\openvr_api.dll.sha256\n\n"
        f"{OPENVR_NOTICE_APP_PRIVATE_EXPLANATION}\n\n"
        f"{OPENVR_NOTICE_PACKAGED_BUILD_EXPLANATION}\n\n"
        "----\n\n"
        "BSD 3-Clause License\n\n"
        f"{openvr_license_text}\n\n"
        "----"
    )


def test_pyproject_caps_deepgram_sdk_below_v6() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "deepgram-sdk>=5.3.4,<6.0.0" in pyproject["project"]["dependencies"]


def test_pyproject_includes_sherpa_onnx_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "sherpa-onnx>=1.13.4" in pyproject["project"]["dependencies"]


def test_pyproject_pins_soxr_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert PINNED_SOXR_SPECIFIER in pyproject["project"]["dependencies"]


def test_pyproject_build_extra_covers_python_soxr_no_build_isolation_backend() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    build_extra = pyproject["project"]["optional-dependencies"]["build"]

    assert "scikit-build-core>=0.10" in build_extra
    assert "nanobind>=2" in build_extra
    assert "setuptools_scm[toml]>=6.2" in build_extra


def test_uv_lock_pins_sherpa_onnx_version() -> None:
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    match = re.search(
        r'\[\[package\]\]\s+name = "sherpa-onnx"\s+version = "([^"]+)"',
        uv_lock,
        re.MULTILINE,
    )

    assert match is not None
    assert match.group(1) == "1.13.4"


def test_uv_lock_pins_soxr_version() -> None:
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    match = re.search(
        r'\[\[package\]\]\s+name = "soxr"\s+version = "([^"]+)"',
        uv_lock,
        re.MULTILINE,
    )

    assert match is not None
    assert match.group(1) == "1.1.0"


def test_uv_lock_includes_python_soxr_build_backend_packages() -> None:
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert 'name = "scikit-build-core"' in uv_lock
    assert 'name = "nanobind"' in uv_lock
    assert 'name = "setuptools-scm"' in uv_lock


def test_release_workflow_uses_frozen_lockfile_sync() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert SHARED_SETUP_ACTION in workflow
    assert 'python -m pip install -e ".[build]"' not in workflow


def test_release_workflow_uses_tag_name_as_release_title() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "name: ${{ github.ref_name }}" in workflow
    assert "name: PuriPuly Heart v${{ needs.verify-version.outputs.version }}" not in workflow


def test_release_template_omits_license_and_notices_boilerplate() -> None:
    template = (ROOT / ".github" / "release-template.md").read_text(encoding="utf-8")

    assert "License and notices" not in template
    assert "Project license:" not in template
    assert "Third-party notices:" not in template
    assert "Previously published MIT versions remain available under MIT." not in template
    assert "----" not in template


def test_release_template_uses_expected_star_request_copy() -> None:
    template = (ROOT / ".github" / "release-template.md").read_text(encoding="utf-8")

    expected_copy = (
        "**If PuriPuly has made your world a bit wider,\n"
        "please consider hitting the ⭐ Star button at the top of the GitHub page.\n"
        "It would mean a lot**"
    )

    assert expected_copy in template
    assert "it would mean a lot" not in template


def test_workflows_pin_exact_python_and_uv_versions() -> None:
    for workflow_path in (
        ROOT / ".github" / "workflows" / "pr-ci.yml",
        ROOT / ".github" / "workflows" / "release.yml",
    ):
        workflow = workflow_path.read_text(encoding="utf-8")
        assert PINNED_PYTHON_VERSION in workflow
        assert PINNED_UV_VERSION in workflow
        assert SHARED_SETUP_ACTION in workflow


def test_release_workflow_pins_innosetup_and_build_installer_without_slow_smoke_script() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "release.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    assert PINNED_INNOSETUP_VERSION in workflow
    assert "scripts/ci/build-release-artifacts.ps1" not in workflow
    assert "cargo build" in workflow
    assert "PyInstaller" in workflow
    assert "ISCC.exe" in workflow
    assert "DisplayVersion" in workflow
    assert "Inno Setup version mismatch" in workflow
    assert "--allow-downgrade" in workflow
    assert "--force" in workflow


def test_shared_windows_build_script_runs_packaged_smoke_test() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "Start-Process" in script
    assert '"--clean"' in script
    assert '"--version"' in script
    assert "_multiarray_umath" in script
    assert 'Get-ChildItem -Path $distDir -Filter "_multiarray_umath*.pyd" -Recurse' in script
    assert "osc-send" not in script
    assert "Remove-Item -Recurse -Force $pyInstallerBuildDir" in script
    assert "Remove-Item -Recurse -Force $distDir" in script
    assert '"innosetup"' in script
    assert '"--version=$InnoSetupVersion"' in script
    assert "DisplayVersion" in script
    assert "Get-Command choco" in script


def test_build_spec_collects_all_registered_moved_module_alias_targets() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert "from puripuly_heart._compat import moved_module_alias_targets" in spec
    for parent_name in (
        "puripuly_heart.app",
        "puripuly_heart.app.adapters",
        "puripuly_heart.app.services",
        "puripuly_heart.core",
    ):
        assert f'"{parent_name}"' in spec
    assert "import_module(moved_module_alias_parent)" in spec
    assert "moved_module_hiddenimports = list(moved_module_alias_targets())" in spec
    assert "] + moved_module_hiddenimports + collect_submodules" in spec


def test_shared_windows_build_script_smoke_tests_gui_startup_for_each_layout() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert '@("gui-startup-check")' in script
    assert 'Invoke-GuiStartupSmokeCheck -ExePath $exePath -Label "Packaged"' in script
    assert 'Invoke-GuiStartupSmokeCheck -ExePath $installedExePath -Label "Installed"' in script
    assert 'Invoke-GuiStartupSmokeCheck -ExePath $installedExePath -Label "Reinstalled"' in script


def test_shared_windows_build_script_reads_soxr_runtime_report_files_for_packaged_and_installed_smoke() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH" in script
    assert '$soxrRuntimeReportDir = Join-Path $PWD "build/soxr-runtime-smoke"' in script
    assert "$packagedSoxrRuntimeReportPath" in script
    assert "$installedSoxrRuntimeReportPath" in script
    assert "$reinstalledSoxrRuntimeReportPath" in script
    assert "ConvertFrom-Json" in script
    assert "$reportedImportedExtensionPath" in script
    assert "$reportedLoadedSoxrDllPath" in script


def test_shared_windows_build_script_uses_alternate_app_id_and_isolated_installer_smoke_dir() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "$InstallerTestAppId" in script
    assert (
        '$InstallerSmokeDir = Join-Path $env:LOCALAPPDATA "Programs\\PuriPulyHeart-LocalSTT-Test"'
        in script
    )
    assert "$InstallerSmokeDir" in script
    assert '"/CURRENTUSER"' in script
    assert '"/VERYSILENT"' in script
    assert '"/SUPPRESSMSGBOXES"' in script
    assert '"/DIR=$InstallerSmokeDir"' in script


def test_shared_windows_build_script_builds_release_installer_without_alternate_app_id() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert 'Invoke-ExternalProcess -FilePath $isccPath -ArgumentList @("installer.iss")' in script


def test_shared_windows_build_script_uses_separate_smoke_installer_build_with_alternate_app_id() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "$InstallerSmokeBuildDir" in script
    assert "$InstallerSmokeAppDataRoot" in script
    assert '"/DMyAppId=$InstallerTestAppId"' in script
    assert '"/O$InstallerSmokeBuildDir"' in script
    assert "$smokeInstallerPath" in script


def test_shared_windows_build_script_overrides_local_stt_appdata_for_smoke_and_checks_log() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT" in script
    assert "$InstallerSmokeLogPath" in script
    assert '"/LOG=$InstallerSmokeLogPath"' in script
    assert "Local STT provisioning completed successfully." in script


def test_shared_windows_build_script_checks_packaged_http_extension_example() -> None:
    script = (ROOT / "scripts/ci/build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert (
        '$packagedTranslationExamplesDir = Join-Path $distDir "examples\\http_extensions"' in script
    )
    assert (
        "$packagedMyMemoryExamplePath = Join-Path $packagedTranslationExamplesDir "
        '"mymemory.json"'
    ) in script
    assert "Packaged MyMemory example not found" in script
    assert "-PathType Leaf" in script


def test_build_spec_bundles_vendored_openvr_runtime_dll() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert (
        "from puripuly_heart.core.overlay.openvr_vendor import "
        "collect_vendored_openvr_runtime_binaries" in spec
    )
    assert "runtime_binaries += collect_vendored_openvr_runtime_binaries()" in spec
    assert "SteamVR\\bin\\win64\\openvr_api.dll" not in spec


def test_build_spec_bundles_http_extension_examples() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert 'HTTP_EXTENSION_EXAMPLES_SOURCE_DIR = Path("examples/http_extensions").resolve()' in spec
    assert 'HTTP_EXTENSION_EXAMPLES_PACKAGED_DIR = Path("examples/http_extensions")' in spec
    assert (
        "(\n"
        "        str(HTTP_EXTENSION_EXAMPLES_SOURCE_DIR),\n"
        "        HTTP_EXTENSION_EXAMPLES_PACKAGED_DIR.as_posix(),\n"
        "    ),"
    ) in spec
    assert (ROOT / "examples/http_extensions/mymemory.json").is_file()


def test_vendored_openvr_bundle_files_exist_and_sha256_line_is_exact() -> None:
    dll_path = ROOT / OPENVR_VENDOR_DLL_RELATIVE_PATH
    sha256_path = ROOT / OPENVR_VENDOR_SHA256_RELATIVE_PATH
    license_path = ROOT / OPENVR_VENDOR_LICENSE_RELATIVE_PATH
    readme_path = ROOT / OPENVR_VENDOR_README_RELATIVE_PATH

    assert dll_path.is_file()
    assert sha256_path.is_file()
    assert license_path.is_file()
    assert readme_path.is_file()
    assert sha256_path.read_text(encoding="utf-8") == f"{OPENVR_VENDOR_SHA256_LINE}\n"


def test_shared_windows_build_script_parses_in_powershell() -> None:
    script_path = ROOT / "scripts" / "ci" / "build-release-artifacts.ps1"
    powershell_path = shutil.which("pwsh") or shutil.which("powershell.exe")
    if powershell_path is None:
        pytest.skip("PowerShell executable not available")

    escaped_script_path = str(script_path).replace("'", "''")
    parse_command = (
        "$tokens = $null; "
        "$errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped_script_path}', [ref]$tokens, [ref]$errors"
        ") > $null; "
        "if ($errors.Count -ne 0) { "
        '$errors | ForEach-Object { "{0}:{1}: {2}" -f $_.Extent.StartLineNumber, $_.Extent.StartColumnNumber, $_.Message }; '
        "exit 1; "
        "}"
    )
    completed = subprocess.run(
        [powershell_path, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", parse_command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "PowerShell failed to parse scripts/ci/build-release-artifacts.ps1\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )


def test_shared_windows_build_script_uses_vendored_openvr_bundle_and_hash_helpers() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert (
        '$openVrVendorDllPath = Join-Path $PWD "third_party/openvr/win64/openvr_api.dll"' in script
    )
    assert (
        '$openVrVendorSha256Path = Join-Path $PWD "third_party/openvr/win64/openvr_api.dll.sha256"'
        in script
    )
    assert f'$PinnedOpenVrVendorDllSha256 = "{OPENVR_VENDOR_DLL_SHA256}"' in script
    assert "function Get-PinnedSha256FromFile" in script
    assert "function Assert-FileSha256Equals" in script
    assert (
        "$pinnedOpenVrVendorDllSha256FromFile = Get-PinnedSha256FromFile -Path $openVrVendorSha256Path"
        in script
    )
    assert "if ($pinnedOpenVrVendorDllSha256FromFile -ne $PinnedOpenVrVendorDllSha256) {" in script
    assert "SteamVR\\bin\\win64\\openvr_api.dll" not in script
    assert (
        "Copy-Item -Path $openVrRuntimeDllPath -Destination $packagedOverlayDllPath -Force"
        not in script
    )


def test_shared_windows_build_script_hash_checks_openvr_dll_at_all_release_stages() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert (
        "Copy-Item -Path $openVrVendorDllPath -Destination $overlayBundledDllPath -Force" in script
    )
    assert (
        "Assert-FileSha256Equals -Path $overlayBundledDllPath -ExpectedSha256 "
        '$PinnedOpenVrVendorDllSha256 -Label "Staged OpenVR runtime DLL"' in script
    )
    assert (
        "Assert-FileSha256Equals -Path $packagedOverlayDllPath -ExpectedSha256 "
        '$PinnedOpenVrVendorDllSha256 -Label "Packaged OpenVR runtime DLL"' in script
    )
    assert '$installedOpenVrDllPath = Join-Path $InstallerSmokeDir "openvr_api.dll"' in script
    assert (
        "Assert-FileSha256Equals -Path $installedOpenVrDllPath -ExpectedSha256 "
        '$PinnedOpenVrVendorDllSha256 -Label "Installed OpenVR runtime DLL"' in script
    )
    assert (
        "Assert-FileSha256Equals -Path $installedOpenVrDllPath -ExpectedSha256 "
        '$PinnedOpenVrVendorDllSha256 -Label "Reinstalled OpenVR runtime DLL"' in script
    )
    assert (
        "Copy-Item -Path $openVrVendorDllPath -Destination $packagedOverlayDllPath -Force"
        not in script
    )


def test_shared_windows_build_script_reinstall_smoke_restores_deliberately_mutated_openvr_dll() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "[System.IO.File]::WriteAllBytes($installedOpenVrDllPath" in script
    assert "$mutatedInstalledOpenVrDllHash = Get-FileSha256 -Path $installedOpenVrDllPath" in script
    assert "Failed to mutate installed OpenVR runtime DLL before reinstall smoke" in script
    assert "$reinstalledOpenVrDllHash = Get-FileSha256 -Path $installedOpenVrDllPath" in script
    assert "Installed OpenVR runtime DLL reinstall smoke failed to restore pinned hash" in script
    assert script.index(
        "$mutatedInstalledOpenVrDllHash = Get-FileSha256 -Path $installedOpenVrDllPath"
    ) < script.index("$installerReinstallSmoke = Start-Process -FilePath $smokeInstallerPath")


def test_installer_script_documents_vendored_openvr_install_from_packaged_tree_without_steamvr_lookup() -> (
    None
):
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert (
        "; Vendored OpenVR runtime DLL comes from dist\\PuriPulyHeart\\openvr_api.dll in the packaged tree built by build.spec."
        in script
    )
    assert "; Installer build/install never resolves SteamVR paths for openvr_api.dll." in script
    assert (
        'Source: "{#MyPackagedAppDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs'
        in script
    )
    assert "Steam\\steamapps\\common\\SteamVR\\bin\\win64\\openvr_api.dll" not in script


def test_build_spec_local_qwen_runtime_dlls() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix() == "_runtime/local_qwen"
    assert (
        "from puripuly_heart.core.local_qwen_runtime import "
        "LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR" in spec
    )
    assert re.search(
        r'collect_dynamic_libs\(\s*"onnxruntime",\s*'
        r"destdir=LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR\.as_posix\(\)\s*\)",
        spec,
    )
    assert "binaries=runtime_binaries" in spec


def test_build_spec_header_distinguishes_direct_packaging_from_full_release_complete_path() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")
    direct_section = _slice_section(
        spec, BUILD_SPEC_DIRECT_PACKAGING_HEADER, BUILD_SPEC_FULL_RELEASE_HEADER
    )
    full_release_section = _slice_section(spec, BUILD_SPEC_FULL_RELEASE_HEADER)

    assert BUILD_SPEC_DIRECT_PACKAGING_HEADER in spec
    assert BUILD_SPEC_DIRECT_PACKAGING_CAVEAT in direct_section
    assert BUILD_SPEC_FULL_RELEASE_HEADER in spec
    assert FULL_WINDOWS_RELEASE_SCRIPT in spec
    assert SOXR_RELEASE_INPUTS_SCRIPT in direct_section
    assert "pyinstaller build.spec" in direct_section
    assert README_INSTALLER_BUILD_COMMAND in direct_section
    assert direct_section.index(SOXR_RELEASE_INPUTS_SCRIPT) < direct_section.index(
        "pyinstaller build.spec"
    )
    assert direct_section.index("pyinstaller build.spec") < direct_section.index(
        README_INSTALLER_BUILD_COMMAND
    )
    assert SOXR_RELEASE_INPUTS_SCRIPT in full_release_section
    assert FULL_WINDOWS_RELEASE_SCRIPT in full_release_section
    assert full_release_section.index(SOXR_RELEASE_INPUTS_SCRIPT) < full_release_section.index(
        FULL_WINDOWS_RELEASE_SCRIPT
    )


def test_shared_windows_build_script_checks_packaged_local_qwen_runtime_dir_for_onnx_dlls() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "function Resolve-PackagedLocalQwenRuntimeDir" in script
    assert 'Join-Path $DistDir "_runtime\\local_qwen"' in script
    assert 'Join-Path $DistDir "_internal\\_runtime\\local_qwen"' in script
    assert "if (Test-Path $packagedLocalQwenRuntimeDir) {" in script
    assert "return $packagedLocalQwenRuntimeDir" in script
    assert "return $packagedLocalQwenFallbackRuntimeDir" in script
    assert (
        "$packagedLocalQwenRuntimeDir = Resolve-PackagedLocalQwenRuntimeDir -DistDir $distDir"
        in script
    )
    assert (
        '$packagedOnnxRuntimeDllPath = Join-Path $packagedLocalQwenRuntimeDir "onnxruntime.dll"'
        in script
    )
    assert (
        "$packagedOnnxRuntimeProvidersSharedDllPath = Join-Path "
        '$packagedLocalQwenRuntimeDir "onnxruntime_providers_shared.dll"' in script
    )
    assert "if (-not (Test-Path $packagedOnnxRuntimeDllPath)) {" in script
    assert "if (-not (Test-Path $packagedOnnxRuntimeProvidersSharedDllPath)) {" in script


def test_build_spec_numpy_runtime_guard_narrow() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert '"numpy._core._multiarray_umath"' in spec
    assert 'collect_dynamic_libs("numpy")' not in spec
    assert 'collect_submodules("numpy")' not in spec


def test_build_spec_guards_required_runtime_dependency_imports_and_assets() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    for hiddenimport in (
        '"flet"',
        '"flet_desktop"',
        '"sherpa_onnx"',
        '"sherpa_onnx.lib._sherpa_onnx"',
        '"sherpa_onnx.offline_recognizer"',
        '"onnxruntime"',
        '"proctap"',
        '"cryptography"',
        '"sounddevice"',
        '"soxr"',
    ):
        assert hiddenimport in spec

    assert 'collect_data_files("flet_desktop")' in spec
    assert 'collect_dynamic_libs("sherpa_onnx", destdir="sherpa_onnx/lib")' in spec
    assert "runtime_binaries = collect_dynamic_libs(" in spec
    assert '"onnxruntime", destdir=LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()' in spec
    assert 'get_module_file_attribute("proctap._native")' in spec
    assert 'collect_dynamic_libs("proctap"' in spec
    assert '"soxr.dll"' in spec


def test_huggingface_xet_dependencies_and_windows_packaging_are_pinned_and_guarded() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")
    main = (ROOT / "src" / "puripuly_heart" / "main.py").read_text(encoding="utf-8")

    assert '"huggingface-hub==1.26.0"' in project
    assert '"hf-xet==1.5.2"' in project
    assert 'collect_data_files("huggingface_hub")' in spec
    assert 'collect_submodules("huggingface_hub")' in spec
    assert 'get_module_file_attribute("hf_xet.hf_xet")' in spec
    assert 'runtime_binaries += [(str(hf_xet_native_extension), "hf_xet")]' in spec
    assert '"hf_xet.hf_xet"' in spec
    assert "required_huggingface_hiddenimports" in spec
    assert '"hf-xet-runtime-check"' in main
    assert 'huggingface_hub.__version__ != "1.26.0"' in main


def test_huggingface_xet_apache_notices_cover_pinned_runtime_packages() -> None:
    notices = (ROOT / "src" / "puripuly_heart" / "data" / "THIRD_PARTY_NOTICES.txt").read_text(
        encoding="utf-8"
    )

    assert "huggingface-hub 1.26.0: Apache-2.0" in notices
    assert "hf-xet 1.5.2 and its Windows native extension: Apache-2.0" in notices
    assert "HF_XET_HIGH_PERFORMANCE by default" in notices


def test_prepare_soxr_release_inputs_script_builds_system_linked_runtime_and_source_bundle() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert PINNED_SOXR_SPECIFIER in script
    assert PINNED_LIBSOXR_SOURCE_SHA256 in script
    assert "USE_SYSTEM_LIBSOXR=ON" in script
    assert '"soxr.dll"' in script
    assert SOXR_RELEASE_INPUTS_MANIFEST_PATH in script
    assert SOXR_SOURCE_BUNDLE_NAME in script
    assert "Compress-Archive" in script
    assert "--no-build-isolation" in script
    assert '"-DCMAKE_SHARED_LIBRARY_PREFIX=lib"' not in script
    assert '"-DCMAKE_IMPORT_LIBRARY_PREFIX=lib"' not in script


def test_prepare_soxr_release_inputs_script_uses_pinned_libsoxr_hash_and_repo_controlled_wheel_build() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert f'$LibsoxrSourceUrl = "{PINNED_LIBSOXR_SOURCE_URL}"' in script
    assert f'$expectedLibsoxrSourceSha256 = "{PINNED_LIBSOXR_SOURCE_SHA256}"' in script
    assert "$libsoxrSourceSha256 -ne $expectedLibsoxrSourceSha256" in script
    assert "python-soxr source wheel using the prepared project environment" in script
    assert '"--no-build-isolation"' in script


def test_prepare_soxr_release_inputs_script_downloads_libsoxr_with_curl_follow_redirects() -> None:
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert 'Resolve-CommandPath -Name "curl.exe"' in script
    assert "Invoke-External -FilePath $curlCommand -ArgumentList @(" in script
    assert '"-L"' in script
    assert "$LibsoxrSourceUrl" in script
    assert "$libsoxrSourcePath" in script


def test_prepare_soxr_release_inputs_script_sets_cmake_policy_minimum_for_libsoxr_013() -> None:
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert '"-DCMAKE_POLICY_VERSION_MINIMUM=3.5"' in script


def test_prepare_soxr_release_inputs_script_renames_windows_soxr_outputs_to_packaged_soxr_names() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert '$libsoxrBuiltDllPath = Join-Path $libsoxrInstallDir "bin\\soxr.dll"' in script
    assert '$libsoxrImportLibPath = Join-Path $libsoxrInstallDir "lib\\soxr.lib"' in script
    assert '$stagedSoxrDllPath = Join-Path $runtimeStageDir "soxr.dll"' in script
    assert "Copy-Item -Path $libsoxrBuiltDllPath -Destination $stagedSoxrDllPath -Force" in script
    assert '"-DCMAKE_SHARED_LIBRARY_PREFIX=lib"' not in script
    assert '"-DCMAKE_IMPORT_LIBRARY_PREFIX=lib"' not in script


def test_prepare_soxr_release_inputs_script_bootstraps_pip_before_wheel_build() -> None:
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert '"ensurepip"' in script
    assert script.index('"ensurepip"') < script.index('"pip"')


def test_prepare_soxr_release_inputs_script_passes_nanobind_cmake_dir_to_python_soxr_build() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert "$nanobindCmakeDir" in script
    assert "nanobind-config.cmake" in script
    assert '"--config-settings=cmake.define.nanobind_DIR=$nanobindCmakeDir"' in script


def test_prepare_soxr_release_inputs_script_adds_built_libsoxr_bin_to_path_before_wheel_build() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert '$libsoxrBinDir = Join-Path $libsoxrInstallDir "bin"' in script
    assert '$env:PATH = "$libsoxrBinDir;$env:PATH"' in script


def test_prepare_soxr_release_inputs_script_temporarily_stages_soxr_dll_beside_build_python_and_removes_it() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert '$wheelBuildSoxrDllPath = Join-Path $projectEnvironmentScripts "soxr.dll"' in script
    assert (
        "$wheelBuildSoxrDllBackupPath = Join-Path $ReleaseInputsRoot "
        '"wheel-build-python-soxr.dll.backup"' in script
    )
    assert "$hadExistingWheelBuildSoxrDll = Test-Path $wheelBuildSoxrDllPath" in script
    assert script.index("try {") < script.index(
        "Move-Item -Path $wheelBuildSoxrDllPath -Destination $wheelBuildSoxrDllBackupPath -Force"
    )
    assert script.index(
        "Move-Item -Path $wheelBuildSoxrDllPath -Destination $wheelBuildSoxrDllBackupPath -Force"
    ) < script.index(
        "Copy-Item -Path $libsoxrBuiltDllPath -Destination $wheelBuildSoxrDllPath -Force"
    )
    assert (
        "Move-Item -Path $wheelBuildSoxrDllPath -Destination $wheelBuildSoxrDllBackupPath -Force"
        in script
    )
    assert (
        "Copy-Item -Path $libsoxrBuiltDllPath -Destination $wheelBuildSoxrDllPath -Force" in script
    )
    assert "$wheelBuildError = $null" in script
    assert "catch {" in script
    assert "$wheelBuildError = $_" in script
    assert "finally {" in script
    assert "Remove-Item -Path $wheelBuildSoxrDllPath -Force -ErrorAction SilentlyContinue" in script
    assert (
        "if ($hadExistingWheelBuildSoxrDll -and (Test-Path $wheelBuildSoxrDllBackupPath)) {"
        in script
    )
    assert (
        "Move-Item -Path $wheelBuildSoxrDllBackupPath -Destination $wheelBuildSoxrDllPath -Force"
        in script
    )
    assert "$wheelBuildCleanupActionError = $null" in script
    assert "$wheelBuildCleanupIssues = @()" in script
    assert "if ($null -ne $wheelBuildCleanupActionError) {" in script
    assert script.index("finally {") < script.index("$wheelBuildCleanupIssues = @()")
    assert "if ($wheelBuildCleanupIssues.Count -gt 0) {" in script
    assert "$PSCmdlet.ThrowTerminatingError($wheelBuildError)" in script


def test_prepare_soxr_release_inputs_script_skips_python_soxr_stub_generation_in_release_input_build() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert '$soxrExtractRoot = Join-Path $ReleaseInputsRoot "soxr-src"' in script
    assert '$soxrCMakeListsPath = Join-Path $soxrSourceRoot.FullName "CMakeLists.txt"' in script
    assert "nanobind_add_stub(soxr_ext_stub" in script
    assert "if (NOT CMAKE_CROSSCOMPILING)" in script
    assert "if (FALSE) # release-input wheel build disables stub generation" in script


def test_prepare_soxr_release_inputs_script_uses_uri_relative_path_helper_for_manifest_paths() -> (
    None
):
    script = (ROOT / SOXR_RELEASE_INPUTS_SCRIPT).read_text(encoding="utf-8")

    assert "[System.Uri]::new" in script
    assert "[System.IO.Path]::GetRelativePath" not in script


def test_build_spec_uses_prepared_soxr_release_inputs_and_guards_packaged_layout() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert SOXR_RELEASE_INPUTS_SCRIPT in spec
    assert SOXR_RELEASE_INPUTS_MANIFEST_PATH in spec
    assert 'read_text(encoding="utf-8-sig")' in spec
    assert 'contents_directory="."' in spec
    assert SOXR_PACKAGED_RUNTIME_RELATIVE_DIR in spec
    assert '"soxr"' in spec
    assert '"soxr.soxr_ext"' in spec
    assert "soxr.dll" in spec
    assert 'collect_dynamic_libs("soxr")' not in spec


def test_flet_runtime_preparation_and_build_spec_pin_the_official_windows_archive() -> None:
    script = (ROOT / FLET_RUNTIME_PREPARATION_SCRIPT).read_text(encoding="utf-8")
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert f'$FletVersion = "{FLET_RUNTIME_VERSION}"' in script
    assert f'$ExpectedSha256 = "{FLET_RUNTIME_SHA256}"' in script
    assert "/releases/download/v$FletVersion/flet-windows.zip" in script
    assert FLET_RUNTIME_PREPARATION_SCRIPT in spec
    assert FLET_RUNTIME_SHA256 in spec
    assert "import flet_cli.__pyinstaller.config as flet_pyinstaller_hook_config" in spec
    assert (
        "flet_pyinstaller_hook_config.temp_bin_dir = "
        "str(FLET_WINDOWS_RUNTIME_ARCHIVE_PATH.parent)" in spec
    )
    assert '(str(FLET_WINDOWS_RUNTIME_ARCHIVE_PATH), "flet_desktop/app")' not in spec


def test_release_workflow_prepares_flet_runtime_before_pyinstaller() -> None:
    release_workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    release_script = (ROOT / FULL_WINDOWS_RELEASE_SCRIPT).read_text(encoding="utf-8")

    job_block = _workflow_job_block(release_workflow, "build-installer")
    assert FLET_RUNTIME_PREPARATION_SCRIPT in job_block
    assert job_block.index(FLET_RUNTIME_PREPARATION_SCRIPT) < job_block.index("PyInstaller")
    assert "prepare-flet-runtime.ps1" in release_script
    assert release_script.index("prepare-flet-runtime.ps1") < release_script.index('"PyInstaller"')


def test_build_spec_deduplicates_any_root_level_auto_collected_soxr_dll() -> None:
    spec = (ROOT / "build.spec").read_text(encoding="utf-8")

    assert "def normalize_soxr_runtime_binaries(binaries):" in spec
    assert 'normalized_destination_name = destination_name.replace("\\\\", "/")' in spec
    assert 'normalized_destination_name == "soxr.dll"' in spec
    assert "Path(source_path).resolve() == sibling_dll_path" not in spec
    assert "normalize_soxr_runtime_binaries(a.binaries)" in spec
    assert spec.index("a = Analysis(") < spec.index("normalize_soxr_runtime_binaries(a.binaries)")


def test_release_workflow_prepares_soxr_release_inputs_before_build_and_publishes_source_bundle_without_installer_sha256() -> (
    None
):
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    job_block = _workflow_job_block(workflow, "build-installer")

    assert SOXR_RELEASE_INPUTS_SCRIPT in job_block
    assert "scripts/ci/build-release-artifacts.ps1" not in job_block
    assert job_block.index(SOXR_RELEASE_INPUTS_SCRIPT) < job_block.index("PyInstaller")
    assert "native/gpu_worker/Cargo.toml" in job_block
    assert "PuriPulyHeartGpuWorker.exe" in job_block
    assert job_block.index("native/gpu_worker/Cargo.toml") < job_block.index("PyInstaller")
    assert "VULKAN_SDK_SHA256" in workflow
    assert "copy_only=1" in job_block
    assert "SPIRV-HeadersConfig.cmake" in job_block
    assert SOXR_SOURCE_BUNDLE_NAME in workflow
    assert (
        "release-artifacts/installer_output/"
        "PuriPulyHeart-Setup-${{ needs.verify-version.outputs.version }}.exe" in workflow
    )
    assert (
        "release-artifacts/build/soxr-release-inputs/"
        "PuriPulyHeart-soxr-third-party-source-bundle.zip" in workflow
    )
    assert (
        "PuriPulyHeart-Setup-${{ needs.verify-version.outputs.version }}.exe.sha256" not in workflow
    )
    assert '"$installer.sha256"' not in workflow
    assert "Get-FileHash -Path $installer -Algorithm SHA256" not in workflow


def test_broker_direct_deploy_syncs_discord_oauth_secrets() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-broker-direct.yml").read_text(
        encoding="utf-8"
    )

    production_to_runtime_secret_names = {
        "DISCORD_CLIENT_ID_PRODUCTION": "DISCORD_CLIENT_ID",
        "DISCORD_CLIENT_SECRET_PRODUCTION": "DISCORD_CLIENT_SECRET",
        "DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION": "DISCORD_REDIRECT_URI_ALLOWLIST",
        "DISCORD_USER_REF_SECRET_PRODUCTION": "DISCORD_USER_REF_SECRET",
    }
    for production_name, runtime_name in production_to_runtime_secret_names.items():
        assert f"{production_name}: ${{{{ secrets.{production_name} }}}}" in workflow
        assert f'if [[ -z "${{{production_name}//[[:space:]]/}}" ]]; then' in workflow
        assert f"{production_name} is required and must not be blank." in workflow
        assert f"printf '%s' \"${production_name}\"" in workflow
        assert f"pnpm exec wrangler secret put {runtime_name} --config" in workflow

    assert (
        "CONFIRM_PRODUCTION_DEPLOY: ${{ github.event.inputs.confirm_production_deploy }}"
        in workflow
    )
    assert 'if [[ "$CONFIRM_PRODUCTION_DEPLOY" !=' in workflow
    assert 'if [[ "${{ github.event.inputs.confirm_production_deploy }}"' not in workflow
    for port in (62187, 62188, 62189):
        assert f"http://127.0.0.1:{port}/discord/callback" in workflow
    assert "DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION must include" in workflow


def test_broker_d1_cleanup_supports_installation_id_target() -> None:
    workflow = (ROOT / ".github" / "workflows" / "maintenance-broker-d1-cleanup.yml").read_text(
        encoding="utf-8"
    )

    assert "installation_id:" in workflow
    assert "description: Target installation id" in workflow
    assert "hardware_hash:" in workflow
    assert "required: false" in workflow
    assert "hardware_hash_salt_version:" in workflow
    assert "installation_id and hardware hash inputs are mutually exclusive" in workflow
    assert (
        "either installation_id or both hardware_hash and hardware_hash_salt_version are required"
        in workflow
    )
    assert "TARGET_MODE: ${{ github.event.inputs.mode }}" in workflow
    assert "TARGET_INSTALLATION_ID: ${{ github.event.inputs.installation_id }}" in workflow
    assert 'installation_id="${{ github.event.inputs.installation_id }}"' not in workflow

    assert "SELECT {sql_literal(installation_id)} AS installation_id" in workflow
    assert "COALESCE(e.discord_user_ref, di.discord_user_ref) AS discord_user_ref" in workflow
    assert "LEFT JOIN discord_identities di" in workflow
    assert "e.discord_issue_status" in workflow
    assert 'json.loads(path.read_text(encoding="utf-8"))' in workflow
    assert 'd1_rows(Path(".maintenance-inspect.json"))' in workflow
    assert "def sql_literal(value: str) -> str:" in workflow
    assert "DELETE FROM broker_request_events WHERE installation_id IN" in workflow
    assert "DELETE FROM discord_oauth_sessions WHERE installation_id IN" in workflow
    assert "DELETE FROM installations WHERE installation_id IN" in workflow
    assert "DELETE FROM discord_identities " in workflow
    assert "WHERE discord_user_ref IN" in workflow


def test_lgpl_text_file_exists_for_bundled_soxr_compliance_bundle() -> None:
    lgpl_text_path = ROOT / SOXR_LICENSE_TEXT_RELATIVE_PATH

    assert lgpl_text_path.is_file()

    lgpl_text = lgpl_text_path.read_text(encoding="utf-8")
    lgpl_lines = lgpl_text.splitlines()

    assert lgpl_lines[:2] == [
        "GNU LESSER GENERAL PUBLIC LICENSE",
        "Version 2.1, February 1999",
    ]
    assert "TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION" in lgpl_text
    assert "END OF TERMS AND CONDITIONS" in lgpl_text


def test_third_party_notices_cover_soxr_runtime_and_installed_compliance_bundle() -> None:
    notices = (ROOT / "src" / "puripuly_heart" / "data" / "THIRD_PARTY_NOTICES.txt").read_text(
        encoding="utf-8"
    )

    assert "Python-SoXR" in notices
    assert "libsoxr" in notices
    assert "soxr.dll" in notices
    assert SOXR_SOURCE_BUNDLE_NAME in notices
    assert "COPYING.LGPL-2.1.txt" in notices
    assert "third_party\\soxr\\" in notices
    assert "Installed releases include an LGPL compliance bundle under" in notices
    assert "exact python-soxr 1.1.0 and libsoxr 0.1.3 source archives used to build" in notices
    assert "{app}" not in notices


def test_third_party_notices_cover_vendored_openvr_bundle_and_bsd_terms() -> None:
    notices = (ROOT / "src" / "puripuly_heart" / "data" / "THIRD_PARTY_NOTICES.txt").read_text(
        encoding="utf-8"
    )
    openvr_notice_section = _slice_section(
        notices, OPENVR_NOTICE_HEADER, OPENVR_NOTICE_NEXT_SECTION_HEADER
    ).strip()

    assert openvr_notice_section == _expected_openvr_notice_section()
    assert "{app}" not in openvr_notice_section


def test_shared_windows_build_script_runs_local_qwen_runtime_check_smoke() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert (
        'Start-Process -FilePath $exePath -ArgumentList @("local-qwen-runtime-check") -Wait -PassThru'
        in script
    )
    assert "Local Qwen runtime smoke test failed" in script


def test_shared_windows_build_script_resolves_packaged_local_qwen_runtime_dir_after_pyinstaller_build() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert "function Resolve-PackagedLocalQwenRuntimeDir" in script
    assert (
        "$packagedLocalQwenRuntimeDir = Resolve-PackagedLocalQwenRuntimeDir -DistDir $distDir"
        in script
    )
    assert 'Join-Path $DistDir "_runtime\\local_qwen"' in script
    assert 'Join-Path $DistDir "_internal\\_runtime\\local_qwen"' in script
    assert script.index("Invoke-ExternalProcess -FilePath $pythonCommand") < script.index(
        "$packagedLocalQwenRuntimeDir = Resolve-PackagedLocalQwenRuntimeDir -DistDir $distDir"
    )


def test_shared_windows_build_script_runs_soxr_runtime_check_smoke() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert (
        "Invoke-SoxrRuntimeSmokeCheck -ExePath $exePath -ReportPath $packagedSoxrRuntimeReportPath "
        "-ExpectedExtensionPath $packagedSoxrExtensionPath -ExpectedSoxrDllPath $packagedSoxrDllPath "
        '-Label "Packaged"' in script
    )
    assert "soxr runtime smoke test failed" in script
    assert script.index("$localQwenRuntimeSmokeTest = Start-Process") < script.index(
        "Invoke-SoxrRuntimeSmokeCheck -ExePath $exePath"
    )
    assert script.index("Invoke-SoxrRuntimeSmokeCheck -ExePath $exePath") < script.index(
        "Smoke-testing packaged overlay executable"
    )


def test_shared_windows_build_script_guards_packaged_soxr_dll_layout_and_source_bundle_contents() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert '$packagedSoxrRuntimeDir = Join-Path $distDir "soxr"' in script
    assert '$packagedSoxrDllPath = Join-Path $packagedSoxrRuntimeDir "soxr.dll"' in script
    assert (
        '$packagedSoxrDlls = @(Get-ChildItem -Path $distDir -Filter "soxr.dll" -Recurse -File '
        "-ErrorAction SilentlyContinue)" in script
    )
    assert "if ($packagedSoxrDlls.Count -ne 1) {" in script
    assert "$packagedSoxrDlls[0].FullName" in script
    assert (
        '$stalePackagedLibsoxrDlls = @(Get-ChildItem -Path $distDir -Filter "libsoxr.dll" -Recurse '
        "-File -ErrorAction SilentlyContinue)" in script
    )
    assert "if ($stalePackagedLibsoxrDlls.Count -ne 0) {" in script
    assert (
        '$soxrReleaseInputsManifestPath = Join-Path $PWD "build/soxr-release-inputs/manifest.json"'
        in script
    )
    assert "ConvertFrom-Json" in script
    assert "[System.IO.Compression.ZipFile]::OpenRead($soxrSourceBundlePath)" in script
    assert '$sourceBundleArchive.GetEntry("manifest.json")' in script
    assert "$sourceBundleManifest.sources" in script


def test_shared_windows_build_script_stages_and_reinstalls_soxr_compliance_bundle() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert (
        '$soxrLicenseTextPath = Join-Path $PWD "src\\puripuly_heart\\data\\licenses\\COPYING.LGPL-2.1.txt"'
        in script
    )
    assert '$packagedSoxrComplianceDir = Join-Path $distDir "third_party\\soxr"' in script
    assert (
        '$packagedSoxrLicensePath = Join-Path $packagedSoxrComplianceDir "COPYING.LGPL-2.1.txt"'
        in script
    )
    assert (
        "$packagedSoxrSourceBundlePath = Join-Path $packagedSoxrComplianceDir "
        "([System.IO.Path]::GetFileName($soxrSourceBundlePath))" in script
    )
    assert (
        '$installedSoxrComplianceDir = Join-Path $InstallerSmokeDir "third_party\\soxr"' in script
    )
    assert (
        '$installedSoxrLicensePath = Join-Path $installedSoxrComplianceDir "COPYING.LGPL-2.1.txt"'
        in script
    )
    assert (
        "$installedSoxrSourceBundlePath = Join-Path $installedSoxrComplianceDir "
        "([System.IO.Path]::GetFileName($soxrSourceBundlePath))" in script
    )
    assert "$expectedInstalledSoxrLicenseHash" in script
    assert "$expectedInstalledSoxrSourceBundleHash" in script
    assert "$reinstalledSoxrLicenseHash" in script
    assert "$reinstalledSoxrSourceBundleHash" in script
    assert (
        "Installed soxr LGPL license text reinstall smoke failed to restore bundled hash" in script
    )
    assert "Installed soxr source bundle reinstall smoke failed to restore bundled hash" in script


def test_shared_windows_build_script_runs_installed_app_soxr_runtime_check_after_installer_smoke() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert '$installedExePath = Join-Path $InstallerSmokeDir "PuriPulyHeart.exe"' in script
    assert '$installedSoxrDllPath = Join-Path $InstallerSmokeDir "soxr\\soxr.dll"' in script
    assert "$installedSoxrExtensionPath" in script
    assert (
        '$installedLegacySoxrDllPath = Join-Path $InstallerSmokeDir "soxr\\libsoxr.dll"' in script
    )
    assert (
        "Invoke-SoxrRuntimeSmokeCheck -ExePath $installedExePath -ReportPath $installedSoxrRuntimeReportPath "
        "-ExpectedExtensionPath $installedSoxrExtensionPath -ExpectedSoxrDllPath $installedSoxrDllPath "
        '-Label "Installed"' in script
    )
    assert "$Label soxr runtime smoke test failed" in script
    assert (
        "Installed app still contains stale legacy soxr runtime DLL after installer smoke" in script
    )
    assert script.index("if ($installerSmoke.ExitCode -ne 0) {") < script.index(
        "Invoke-SoxrRuntimeSmokeCheck -ExePath $installedExePath -ReportPath $installedSoxrRuntimeReportPath "
        "-ExpectedExtensionPath $installedSoxrExtensionPath -ExpectedSoxrDllPath $installedSoxrDllPath "
        '-Label "Installed"'
    )


def test_shared_windows_build_script_reinstall_smoke_restores_official_soxr_runtime_and_compliance_bundle() -> (
    None
):
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert (
        '$InstallerReinstallSmokeLogPath = Join-Path $env:TEMP "PuriPulyHeart-LocalSTT-Test-reinstall.log"'
        in script
    )
    assert "$expectedInstalledSoxrDllHash" in script
    assert "[System.IO.File]::WriteAllBytes($installedSoxrDllPath" in script
    assert "[System.IO.File]::WriteAllBytes($installedLegacySoxrDllPath" in script
    assert (
        "$mutatedInstalledSoxrDllHash = (Get-FileHash -Path $installedSoxrDllPath -Algorithm SHA256).Hash"
        in script
    )
    assert '"/LOG=$InstallerReinstallSmokeLogPath"' in script
    assert "$reinstalledSoxrDllHash" in script
    assert "Installed soxr runtime DLL reinstall smoke failed to restore bundled hash" in script
    assert "Installed stale legacy soxr runtime DLL was not removed by reinstall smoke" in script
    assert script.count("Invoke-SoxrRuntimeSmokeCheck -ExePath $installedExePath") >= 2


def test_shared_windows_build_script_reinstall_smoke_removes_seeded_root_level_soxr_dll() -> None:
    script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(encoding="utf-8")

    assert '$legacyRootLevelSoxrDllPath = Join-Path $InstallerSmokeDir "soxr.dll"' in script
    assert "[System.IO.File]::WriteAllBytes($legacyRootLevelSoxrDllPath" in script
    assert "if (Test-Path $legacyRootLevelSoxrDllPath) {" in script


def test_shared_setup_action_installs_pinned_uv_and_uses_frozen_sync() -> None:
    action = (ROOT / ".github" / "actions" / "setup-uv-environment" / "action.yml").read_text(
        encoding="utf-8"
    )

    assert "uses: actions/setup-python@v5" in action
    assert "cache-dependency-path: uv.lock" in action
    assert '"uv==${{ inputs.uv-version }}"' in action
    assert "uv sync ${{ inputs.sync-args }} --frozen" in action


def test_windows_gpu_worker_native_sources_compile_as_utf8() -> None:
    cargo_config = (ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8")

    assert "CXXFLAGS_x86_64_pc_windows_msvc" in cargo_config
    assert 'value = "/utf-8"' in cargo_config
    assert "force = true" in cargo_config


def test_installer_script_guards_against_repo_checkout_installs() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "#ifndef MyAppId" in script
    assert "AppId={#MyAppId}" in script
    assert r"DefaultDirName={autopf}\{#MyAppDirName}" in script
    assert "function DirectoryLooksLikeRepositoryCheckout(Path: String): Boolean;" in script
    assert "DirExists(AddBackslash(ProbePath) + '.git')" in script
    assert "FileExists(AddBackslash(ProbePath) + 'pyproject.toml')" in script
    assert "FileExists(AddBackslash(ProbePath) + 'AGENTS.md')" in script
    assert "procedure ResetSuspiciousInstallDir();" in script
    assert "if DirectoryLooksLikeRepositoryCheckout(CandidateDir) then begin" in script
    assert "Resetting suspicious install dir inside a repository checkout:" in script
    assert "WizardForm.DirEdit.Text := DefaultDir;" in script
    assert r"DefaultDir := ExpandConstant('{autopf}\{#MyAppDirName}');" in script
    assert "procedure InitializeWizard();" in script
    assert "function PrepareToInstall(var NeedsRestart: Boolean): String;" in script


def test_installer_script_copies_full_packaged_app_tree_without_legacy_internal_subdir_assumption() -> (
    None
):
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert (
        'Source: "{#MyPackagedAppDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs'
        in script
    )
    assert 'Excludes: "{#MyAppExeName},{#MyOverlayExeName},{#MyGpuWorkerExeName}"' in script
    assert (
        'Source: "{#MyPackagedAppDir}\\{#MyGpuWorkerExeName}"; DestDir: "{app}"; Flags: ignoreversion'
        in script
    )
    assert 'Source: "{#MyPackagedAppDir}\\_internal\\*"' not in script


def test_installer_script_uses_root_level_local_stt_manifest_path_for_packaged_layout() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert (
        '#define LocalSttManifestRelativePath "puripuly_heart\\data\\models\\qwen3-asr-0.6b-int8-sherpa.manifest.json"'
        in script
    )
    assert (
        '#define LocalSttManifestRelativePath "_internal\\puripuly_heart\\data\\models\\qwen3-asr-0.6b-int8-sherpa.manifest.json"'
        not in script
    )
    assert (
        '#define ParakeetV3ManifestRelativePath "puripuly_heart\\data\\models\\parakeet-tdt-0.6b-v3-int8-sherpa.manifest.json"'
        in script
    )
    assert (
        '#define ParakeetJapaneseManifestRelativePath "puripuly_heart\\data\\models\\parakeet-tdt-ctc-0.6b-ja-int8-sherpa.manifest.json"'
        in script
    )


def test_installer_script_guards_against_temporary_install_dirs() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "function DirectoryLooksLikeTemporaryLocation(Path: String): Boolean;" in script
    assert "TempRoot := RemoveBackslashUnlessRoot(GetEnv('TEMP'));" in script
    assert "TempRoot := RemoveBackslashUnlessRoot(GetEnv('TMP'));" in script
    assert (
        r"TempRoot := RemoveBackslashUnlessRoot(ExpandConstant('{localappdata}\Temp'));" in script
    )
    assert r"TempRoot := RemoveBackslashUnlessRoot(ExpandConstant('{tmp}'));" in script
    assert r"TempRoot := RemoveBackslashUnlessRoot(ExpandConstant('{win}\Temp'));" in script
    assert "if DirectoryLooksLikeTemporaryLocation(CandidateDir) then begin" in script


def test_installer_script_path_prefix_helper_handles_drive_root_boundaries() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert r"(NormalizedRoot[Length(NormalizedRoot)] = '\')" in script


def test_installer_script_uses_inno_managed_local_stt_download() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "LocalSttSourcePage" not in script
    assert "LocalSttSizeLabel" not in script
    assert "InitializeLocalSttWizardPage" not in script
    assert "ActivateLocalSttPage" not in script
    assert "CurPageChanged" in script
    assert "WizardForm.ReadyMemo.Text" in script
    assert "LocalSttRedownloadSize" in script
    assert "WizardIsTaskSelected('redownloadasr')" in script
    assert "LocalSttReinstallCheckBox" not in script
    assert "LocalSttReinstall=" not in script
    assert "CreateDownloadPage" in script
    assert "ASR model" in script
    assert "Local Speech Model" not in script
    assert "local speech model" not in script
    assert "DownloadTemporaryFile" in script
    assert "DownloadPage.Add" not in script
    assert "DownloadPage.Download" not in script
    assert "GetSHA256OfFile" in script
    assert "FileSize64" in script
    assert "GetSpaceOnDisk64" in script
    assert "ValidateLocalSttInstalledManifest" in script
    assert "selected_revision" in script
    assert "DownloadAndInstallLocalSttSource('huggingface'" in script
    assert "DownloadAndInstallLocalSttSource('modelscope'" in script
    assert "function RunLocalSttModelInstall(): Boolean;" in script
    assert "if not RunLocalSttModelInstall() then begin" in script
    assert "continuing app install without bundled ASR model" in script
    assert "Result := ExpandConstant('{cm:LocalSttDownloadFailed}');" not in script
    assert "CurStepChanged" not in script
    assert "GetSelectedLocalSttSource" not in script
    assert "LocalSttSourceComboBox" not in script
    assert "install-local-stt-model.ps1" not in script
    assert "SW_HIDE" not in script


def test_installer_script_uses_concise_local_stt_user_copy_across_locales() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    expected_messages = {
        "english.LocalSttDownloadSize": "ASR models will be downloaded.%nInstallation requires %1 of disk space.",
        "english.LocalSttNoDownload": "All ASR models are installed.",
        "english.LocalSttDownloadDescription": "",
        "english.AsrModelsGroup": "ASR Models",
        "english.RedownloadAsrTask": "Re-download all ASR models",
        "english.LocalSttRedownloadSize": "Re-downloading ASR models.%nInstallation requires %1 of disk space.",
        "korean.LocalSttDownloadSize": "ASR 모델을 다운로드합니다.%n설치에 필요한 용량은 %1입니다.",
        "korean.LocalSttNoDownload": "ASR 모델이 모두 설치되어있습니다.",
        "korean.LocalSttDownloadDescription": "",
        "korean.AsrModelsGroup": "ASR 모델",
        "korean.RedownloadAsrTask": "ASR 모델 전체 재다운로드",
        "korean.LocalSttRedownloadSize": "ASR 모델을 재다운로드합니다.%n설치에 필요한 용량은 %1입니다.",
        "japanese.LocalSttDownloadSize": "ASRモデルをダウンロードします。%nインストールには%1の空き容量が必要です。",
        "japanese.LocalSttNoDownload": "ASRモデルはすべてインストール済みです。",
        "japanese.LocalSttDownloadDescription": "",
        "japanese.AsrModelsGroup": "ASRモデル",
        "japanese.RedownloadAsrTask": "ASRモデルをすべて再ダウンロード",
        "japanese.LocalSttRedownloadSize": "ASRモデルを再ダウンロードします。%nインストールには%1の空き容量が必要です。",
        "chinesesimplified.LocalSttDownloadSize": "将下载 ASR 模型。%n安装需要 %1 的空间。",
        "chinesesimplified.LocalSttNoDownload": "ASR 模型均已安装。",
        "chinesesimplified.LocalSttDownloadDescription": "",
        "chinesesimplified.AsrModelsGroup": "ASR 模型",
        "chinesesimplified.RedownloadAsrTask": "重新下载所有 ASR 模型",
        "chinesesimplified.LocalSttRedownloadSize": "重新下载 ASR 模型。%n安装需要 %1 的空间。",
        "chinesetraditional.LocalSttDownloadSize": "將下載 ASR 模型。%n安裝需要 %1 的空間。",
        "chinesetraditional.LocalSttNoDownload": "ASR 模型均已安裝。",
        "chinesetraditional.LocalSttDownloadDescription": "",
        "chinesetraditional.AsrModelsGroup": "ASR 模型",
        "chinesetraditional.RedownloadAsrTask": "重新下載所有 ASR 模型",
        "chinesetraditional.LocalSttRedownloadSize": "重新下載 ASR 模型。%n安裝需要 %1 的空間。",
    }

    for key, value in expected_messages.items():
        assert f"{key}={value}\n" in script

    assert "LocalSttPageTitle" not in script
    assert "LocalSttChecking" not in script
    assert "LocalSttPageDescription" not in script
    assert "Hugging Face is tried first; ModelScope is used automatically if needed." not in script
    assert "먼저 Hugging Face를 시도하고, 필요하면 ModelScope로 자동 전환합니다." not in script


def test_installer_reports_required_model_bytes_and_keeps_one_continuous_progress_page() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "QwenDownloadSize = 987664355;" in script
    assert "ParakeetV3DownloadSize = 670478772;" in script
    assert "ParakeetJapaneseDownloadSize = 655571161;" in script
    assert "LocalSttRequiredDownloadBytes" in script
    assert "procedure CompleteLocalSttDownloadSegment" in script
    assert script.count("DownloadPage.Show;") == 1
    assert script.count("DownloadPage.Hide;") == 1
    assert "DownloadPage.SetProgress" in script
    assert "LocalSttDisplayedDownloadBytes" in script


def test_installer_uses_manifest_check_for_existing_models_and_checks_both_storage_locations() -> (
    None
):
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "ValidateParakeetV3InstalledManifest(ParakeetV3Dir)" in script
    assert "FileExists(AddBackslash(ParakeetV3Dir) + 'encoder.int8.onnx')" in script
    assert "ValidateParakeetJapaneseInstalledManifest(ParakeetJapaneseDir)" in script
    assert "FileExists(AddBackslash(ParakeetJapaneseDir) + 'model.int8.onnx')" in script
    assert "ValidateLocalSttInstalledManifest(QwenDir)" in script
    assert "FileExists(AddBackslash(QwenDir) + 'decoder.int8.onnx')" in script
    assert "WizardIsTaskSelected('redownloadasr')" in script
    assert "ReDownloadSelected" in script
    assert "EnsureExistingLocalSttInstall" not in script
    assert "EnsureExistingParakeetV3Install" not in script
    assert "EnsureExistingParakeetJapaneseInstall" not in script
    assert "GetSHA256OfFile" in script
    assert "FileSize64" in script
    assert "ValidateLocalSttInstalledManifest" in script
    assert "GetSpaceOnDisk64(TempPath" in script
    assert "GetSpaceOnDisk64(ModelPath" in script
    assert "LocalSttRequiredDownloadBytes * 2" in script


def test_installer_script_embeds_local_stt_manifest_assets_for_inno_download() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "HuggingFaceLocalSttUrl" in script
    assert "ModelScopeLocalSttUrl" in script
    for manifest_path in sorted((ROOT / "src/puripuly_heart/data/models").glob("*.manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["engine"] == "transcribe.cpp-vulkan":
            assert manifest["model_id"] not in script
            continue
        assert manifest["model_id"] in script
        assert manifest["install_dirname"] in script
        for source in manifest["sources"].values():
            assert source["revision"] in script
        for asset in manifest["files"]:
            assert asset["relative_path"] in script
            assert asset["sha256"] in script
            assert str(asset["size_bytes"]) in script


def test_installer_attempts_all_required_cpu_models_and_continues_on_partial_failure() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "function RunRequiredCpuLocalSttModelInstalls(): Boolean;" in script
    parakeet_v3_call = "ParakeetV3Installed := RunParakeetV3LocalSttModelInstall();"
    parakeet_ja_call = "ParakeetJapaneseInstalled := RunParakeetJapaneseLocalSttModelInstall();"
    qwen_call = "if not RunLocalSttModelInstall() then begin"
    assert script.index(parakeet_v3_call) < script.index(parakeet_ja_call) < script.index(qwen_call)
    assert (
        "Result := ParakeetV3Installed and ParakeetJapaneseInstalled and QwenInstalled;" in script
    )
    assert "if not RunRequiredCpuLocalSttModelInstalls() then begin" in script
    assert "continuing app install without bundled ASR model" in script


def test_installer_script_supports_local_stt_appdata_override_for_smoke_runs() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT" in script
    assert "GetEnv('PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT')" in script


def test_installer_script_deletes_managed_default_vad_cache_on_install() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert "[InstallDelete]" in script
    assert 'Type: files; Name: "{localappdata}\\puripuly-heart\\silero_vad.onnx"' in script


def test_installer_script_deletes_root_level_and_nested_legacy_soxr_dlls_on_install() -> None:
    script = (ROOT / "installer.iss").read_text(encoding="utf-8")

    assert 'Type: files; Name: "{app}\\soxr.dll"' in script
    assert 'Type: files; Name: "{app}\\soxr\\libsoxr.dll"' in script


def test_local_stt_installer_script_uses_manifest_validation_and_atomic_promotion() -> None:
    script = (ROOT / "scripts" / "installer" / "install-local-stt-model.ps1").read_text(
        encoding="utf-8"
    )

    assert "Get-FileHash" in script
    assert "Invoke-WebRequest" in script
    assert "installed-manifest.json" in script
    assert "huggingface" in script.lower()
    assert "modelscope" in script.lower()
    assert "Move-Item" in script
    assert "selectedSource" in script or "SelectedSource" in script


def test_local_stt_installer_script_writes_bomless_manifest_and_treats_invalid_install_as_recoverable() -> (
    None
):
    script = (ROOT / "scripts" / "installer" / "install-local-stt-model.ps1").read_text(
        encoding="utf-8"
    )

    assert "UTF8Encoding($false)" in script
    assert "WriteAllText" in script
    assert "catch {" in script
    assert "return $false" in script


def test_local_stt_installer_script_attempts_backup_restore_on_promotion_failure() -> None:
    script = (ROOT / "scripts" / "installer" / "install-local-stt-model.ps1").read_text(
        encoding="utf-8"
    )

    assert '$backupDir = "$InstallDir.backup"' in script
    assert "Move-Item -Path $backupDir -Destination $InstallDir -Force" in script


def test_chinese_installer_language_files_use_matching_message_keys() -> None:
    pattern = re.compile(r"^([A-Za-z][A-Za-z0-9]*)=")

    def extract_keys(path: Path) -> set[str]:
        keys: set[str] = set()
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = pattern.match(line)
            if match:
                keys.add(match.group(1))
        return keys

    simplified = extract_keys(ROOT / "installer" / "Languages" / "ChineseSimplified.isl")
    traditional = extract_keys(ROOT / "installer" / "Languages" / "ChineseTraditional.isl")

    assert traditional == simplified
