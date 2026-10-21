from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from puripuly_heart.core.overlay.openvr_vendor import (
    OPENVR_VENDOR_DLL_URL,
    OPENVR_VENDOR_LICENSE_URL,
    OPENVR_VENDOR_REPOSITORY_REF,
)
from tests.helpers.paths import REPO_ROOT as ROOT

PINNED_SOXR_SPECIFIER = "soxr==1.1.0"
FLET_RUNTIME_PREPARATION_SCRIPT = "scripts/ci/prepare-flet-runtime.ps1"
FLET_RUNTIME_VERSION = "1.0.0"
FLET_RUNTIME_SHA256 = "758f21506fbb9ad180bd93c7460a2ca55630401c6026a9bc9e2273444014491d"
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


def test_pyproject_pins_soxr_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert PINNED_SOXR_SPECIFIER in pyproject["project"]["dependencies"]


def test_production_export_includes_locked_httpx_socks_transport(tmp_path: Path) -> None:
    from packaging.requirements import Requirement

    from puripuly_heart.release_evidence.native_distribution import filter_requirements

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    httpx = next(
        Requirement(value)
        for value in pyproject["project"]["dependencies"]
        if Requirement(value).name == "httpx"
    )
    assert httpx.extras == {"socks"}
    assert str(httpx.specifier) == ">=0.27"
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}
    assert packages["httpx"]["optional-dependencies"]["socks"] == [{"name": "socksio"}]
    assert {"name": "httpx", "extra": ["socks"]} in packages["puripuly-heart"]["dependencies"]
    assert packages["socksio"]["wheels"]
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to exercise the native production dependency export")
    exported = tmp_path / "exported.txt"
    subprocess.run(
        [
            uv,
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            str(exported),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    filtered = tmp_path / "native-requirements.txt"
    filter_requirements(exported, filtered)
    requirements = {
        Requirement(line.split(" \\", 1)[0]).name: Requirement(line.split(" \\", 1)[0])
        for line in filtered.read_text(encoding="utf-8").splitlines()
        if line and not line[0].isspace() and not line.startswith(("#", "--"))
    }
    assert "httpx" in requirements
    assert str(requirements["socksio"].specifier) == f"=={packages['socksio']['version']}"


def test_pyproject_build_extra_covers_python_soxr_no_build_isolation_backend() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    build_extra = pyproject["project"]["optional-dependencies"]["build"]

    assert "scikit-build-core>=0.10" in build_extra
    assert "nanobind>=2" in build_extra
    assert "setuptools_scm[toml]>=6.2" in build_extra


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


@pytest.mark.parametrize(
    ("use_pr_mode", "stale"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["release-valid", "release-stale", "pr-valid", "pr-stale"],
)
def test_shared_setup_action_preserves_release_default_and_rejects_stale_pr_lock(
    tmp_path: Path, use_pr_mode: bool, stale: bool
) -> None:
    action = yaml.safe_load(
        (ROOT / ".github/actions/setup-uv-environment/action.yml").read_text(encoding="utf-8")
    )
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-ci.yml").read_text(encoding="utf-8"))
    setup_python = next(step for step in action["runs"]["steps"] if "uses" in step)
    assert setup_python["uses"] == "actions/setup-python@v7"
    assert setup_python["with"]["cache-dependency-path"] == "uv.lock"
    inputs = {
        "sync-args": "--extra dev",
        "uv-version": workflow["env"]["UV_VERSION"],
        "lock-mode": action["inputs"]["lock-mode"]["default"],
    }
    if use_pr_mode:
        pr_setup = next(
            step
            for step in workflow["jobs"]["python-style"]["steps"]
            if step.get("uses") == "./.github/actions/setup-uv-environment"
        )
        inputs["lock-mode"] = pr_setup["with"]["lock-mode"]
        inputs["sync-args"] = pr_setup["with"]["sync-args"]
    commands = [
        shlex.split(
            re.sub(
                r"\$\{\{\s*inputs\.([a-z-]+)\s*\}\}",
                lambda match: inputs[match[1]],
                line,
            )
        )
        for step in action["runs"]["steps"]
        for line in step.get("run", "").splitlines()
        if line.strip()
    ]
    installer = next(command for command in commands if command[:3] == ["python", "-m", "pip"])
    assert "install" in installer
    assert f"uv=={inputs['uv-version']}" in installer
    sync_commands = [command for command in commands if command[:2] == ["uv", "sync"]]
    assert len(sync_commands) == 1
    uv = shutil.which("uv")
    assert uv is not None, "The pinned uv tool is required for the setup invocation contract"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("UV_") and key.upper() != "VIRTUAL_ENV"
    }
    environment.update(
        UV_PROJECT_ENVIRONMENT=str(tmp_path / ".venv"),
        UV_CACHE_DIR=str(tmp_path / "cache"),
        UV_NO_CONFIG="1",
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
        UV_PYTHON=sys.executable,
    )

    def invoke(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [uv, *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        print(
            json.dumps(
                {
                    "arguments": arguments,
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        )
        return completed

    version = invoke(["--version"])
    assert version.returncode == 0
    assert version.stdout.split()[1] == inputs["uv-version"]
    project = tmp_path / "pyproject.toml"
    metadata = (
        '[project]\nname = "ci-lock-contract"\nversion = "0.0.0"\n'
        'requires-python = ">=3.14,<3.15"\n[project.optional-dependencies]\ndev = []\n'
    )
    project.write_text(metadata, encoding="utf-8")
    locked = invoke(["lock"])
    assert locked.returncode == 0, locked.stderr
    lock = tmp_path / "uv.lock"
    original_lock = lock.read_bytes()
    if stale:
        project.write_text(
            metadata.replace('version = "0.0.0"', 'version = "0.0.1"'), encoding="utf-8"
        )
    synced = invoke(sync_commands[0][1:])
    assert (synced.returncode == 0) is (not use_pr_mode or not stale), synced.stderr
    assert lock.read_bytes() == original_lock
    pr_run = next(
        step["run"]
        for step in workflow["jobs"]["python-style"]["steps"]
        if step.get("run", "").startswith("uv run ")
    )
    run_prefix = shlex.split(pr_run)[:3]
    invoked = invoke(
        [
            *run_prefix[1:],
            "python",
            "-c",
            "from pathlib import Path; Path('command-executed').write_text('yes')",
        ]
    )
    assert (invoked.returncode == 0) is (not stale), invoked.stderr
    assert (tmp_path / "command-executed").exists() is (not stale)
    assert lock.read_bytes() == original_lock
    if stale:
        assert "lockfile" in invoked.stderr and "--locked" in invoked.stderr


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
