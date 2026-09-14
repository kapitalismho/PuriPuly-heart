from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

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
FLET_RUNTIME_PREPARATION_SCRIPT = "scripts/ci/prepare-flet-runtime.ps1"
FLET_RUNTIME_VERSION = "0.86.1"
FLET_RUNTIME_SHA256 = "2cf0865b31bd0e394a24a6c2d270e084cf9dad9c711e0b5d0cf9fa9bfac31e14"
SOXR_LICENSE_TEXT_RELATIVE_PATH = "src/puripuly_heart/data/licenses/COPYING.LGPL-2.1.txt"
OPENVR_VENDOR_DLL_RELATIVE_PATH = "third_party/openvr/win64/openvr_api.dll"
OPENVR_VENDOR_SHA256_RELATIVE_PATH = "third_party/openvr/win64/openvr_api.dll.sha256"
OPENVR_VENDOR_LICENSE_RELATIVE_PATH = "third_party/openvr/LICENSE"
OPENVR_VENDOR_README_RELATIVE_PATH = "third_party/openvr/README.md"
OPENVR_VENDOR_DLL_SHA256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"
OPENVR_VENDOR_SHA256_LINE = f"{OPENVR_VENDOR_DLL_SHA256} *openvr_api.dll"
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


def test_third_party_notices_cover_vendored_openvr_bundle_and_bsd_terms() -> None:
    notices = (ROOT / "src" / "puripuly_heart" / "data" / "THIRD_PARTY_NOTICES.txt").read_text(
        encoding="utf-8"
    )
    openvr_notice_section = _slice_section(
        notices, OPENVR_NOTICE_HEADER, OPENVR_NOTICE_NEXT_SECTION_HEADER
    ).strip()

    assert openvr_notice_section == _expected_openvr_notice_section()
    assert "{app}" not in openvr_notice_section


def test_shared_setup_action_installs_pinned_uv_and_uses_frozen_sync() -> None:
    action = (ROOT / ".github" / "actions" / "setup-uv-environment" / "action.yml").read_text(
        encoding="utf-8"
    )

    assert "uses: actions/setup-python@v7" in action
    assert "cache-dependency-path: uv.lock" in action
    assert '"uv==${{ inputs.uv-version }}"' in action
    assert "uv sync ${{ inputs.sync-args }} --frozen" in action


def test_windows_gpu_worker_native_sources_compile_as_utf8() -> None:
    cargo_config = (ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8")

    assert "CXXFLAGS_x86_64_pc_windows_msvc" in cargo_config
    assert 'value = "/utf-8"' in cargo_config
    assert "force = true" in cargo_config


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
