from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import psutil
import pytest

from tests.helpers.paths import REPO_ROOT

POWERSHELL = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(
    os.name != "nt" or POWERSHELL is None, reason="Windows PowerShell 7 is required"
)


@pytest.fixture
def runner(tmp_path: Path) -> Path:
    script = tmp_path / "run.ps1"
    script.write_text(
        """param([string]$Source, [string]$Python, [string]$ArgumentsFile, [int]$TimeoutSeconds)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$function = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Invoke-External'
}, $true)
. ([scriptblock]::Create($function.Extent.Text))
$arguments = @(Get-Content -LiteralPath $ArgumentsFile -Raw -Encoding utf8 | ConvertFrom-Json)
Invoke-External -FilePath $Python -ArgumentList $arguments -TimeoutSeconds $TimeoutSeconds
""",
        encoding="utf-8",
    )
    return script


def _invoke(runner: Path, arguments: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    arguments_file = runner.with_suffix(".json")
    arguments_file.write_text(json.dumps(arguments), encoding="utf-8")
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(runner),
            "-Source",
            str(REPO_ROOT / "scripts/ci/prepare-soxr-release-inputs.ps1"),
            "-Python",
            sys.executable,
            "-ArgumentsFile",
            str(arguments_file),
            "-TimeoutSeconds",
            str(timeout),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def _archive(path: Path, mode: str, name: str) -> None:
    payload = b"cmake_minimum_required(VERSION 3.5)\n"
    with tarfile.open(path, mode) as archive:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))


@pytest.mark.parametrize(("suffix", "mode"), [("tar.gz", "w:gz"), ("tar.xz", "w:xz")])
def test_extracts_source_archive_with_unicode_and_spaces(
    tmp_path: Path, runner: Path, suffix: str, mode: str
) -> None:
    archive = tmp_path / f"소스 archive.{suffix}"
    destination = tmp_path / "출력 with spaces"
    _archive(archive, mode, "source/CMakeLists.txt")
    completed = _invoke(
        runner,
        ["-I", "-m", "tarfile", "--filter", "data", "--extract", str(archive), str(destination)],
    )
    assert completed.returncode == 0, completed.stderr
    assert (destination / "source/CMakeLists.txt").read_bytes() == (
        b"cmake_minimum_required(VERSION 3.5)\n"
    )


def test_rejects_archive_path_traversal(tmp_path: Path, runner: Path) -> None:
    archive = tmp_path / "unsafe.tar.xz"
    destination = tmp_path / "output"
    _archive(archive, "w:xz", "../escaped.txt")
    completed = _invoke(
        runner,
        ["-I", "-m", "tarfile", "--filter", "data", "--extract", str(archive), str(destination)],
    )
    assert completed.returncode != 0
    assert not (tmp_path / "escaped.txt").exists()


def test_timeout_stops_process_tree(tmp_path: Path, runner: Path) -> None:
    parent_pid_file = tmp_path / "parent.pid"
    child_pid_file = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "subprocess.Popen([sys.executable, '-I', sys.argv[2], sys.argv[3]])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    try:
        completed = _invoke(
            runner,
            ["-I", str(parent), str(parent_pid_file), str(child), str(child_pid_file)],
            timeout=3,
        )
        assert completed.returncode != 0
        for pid_file in (parent_pid_file, child_pid_file):
            pid = int(pid_file.read_text())
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue
            process.wait(timeout=5)
    finally:
        for pid_file in (parent_pid_file, child_pid_file):
            if pid_file.exists():
                try:
                    psutil.Process(int(pid_file.read_text())).kill()
                except psutil.NoSuchProcess:
                    pass


@pytest.fixture
def isolated_runner(tmp_path: Path) -> Path:
    script = tmp_path / "run-isolated.ps1"
    script.write_text(
        """param([string]$Source, [string]$Python, [string]$ArgumentsFile, [string]$StateRoot, [int]$TimeoutSeconds)
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$function = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Invoke-IsolatedProcess'
}, $true)
. ([scriptblock]::Create($function.Extent.Text))
$arguments = @(Get-Content -LiteralPath $ArgumentsFile -Raw -Encoding utf8 | ConvertFrom-Json)
$result = Invoke-IsolatedProcess -FilePath $Python -ArgumentList $arguments -StateRoot $StateRoot -TimeoutSeconds $TimeoutSeconds
$result | ConvertTo-Json -Compress
""",
        encoding="utf-8",
    )
    return script


def _invoke_isolated(
    runner: Path, state: Path, arguments: list[str], timeout: int = 15
) -> subprocess.CompletedProcess:
    arguments_file = runner.with_suffix(".json")
    arguments_file.write_text(json.dumps(arguments), encoding="utf-8")
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(runner),
            "-Source",
            str(REPO_ROOT / "scripts/ci/build-native-release-artifacts.ps1"),
            "-Python",
            sys.executable,
            "-ArgumentsFile",
            str(arguments_file),
            "-StateRoot",
            str(state),
            "-TimeoutSeconds",
            str(timeout),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def test_release_startup_uses_only_synthetic_state(
    tmp_path: Path, isolated_runner: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PURIPULY_TEST_PARENT_STATE", "must not escape")
    state = tmp_path / "설정 with spaces"
    completed = _invoke_isolated(
        isolated_runner,
        state,
        [
            "-c",
            "import json, os; print(json.dumps({"
            "'cwd': os.getcwd(), 'home': os.environ['USERPROFILE'],"
            "'appdata': os.environ['APPDATA'], 'local': os.environ['LOCALAPPDATA'],"
            "'inherited': os.environ.get('PURIPULY_TEST_PARENT_STATE')}))",
        ],
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    state_report = json.loads(result["Output"])
    assert Path(state_report["cwd"]) == state
    assert Path(state_report["home"]) == state / "profile"
    assert Path(state_report["appdata"]) == state / "profile/appdata"
    assert Path(state_report["local"]) == state / "profile/localappdata"
    assert state_report["inherited"] is None


def test_release_startup_failure_preserves_diagnostics(
    tmp_path: Path, isolated_runner: Path
) -> None:
    completed = _invoke_isolated(
        isolated_runner,
        tmp_path / "state",
        [
            "-c",
            "import sys; print('startup output'); print('startup error', file=sys.stderr); sys.exit(7)",
        ],
    )
    assert completed.returncode != 0
    assert "exit 7" in completed.stderr
    assert "startup output" in completed.stderr
    assert "startup error" in completed.stderr


def test_release_startup_timeout_stops_the_process(tmp_path: Path, isolated_runner: Path) -> None:
    pid_file = tmp_path / "startup.pid"
    try:
        completed = _invoke_isolated(
            isolated_runner,
            tmp_path / "state",
            [
                "-c",
                "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
                str(pid_file),
            ],
            timeout=2,
        )
        assert completed.returncode != 0
        assert "timed out after 2 seconds" in completed.stderr
        pid = int(pid_file.read_text())
        assert not psutil.pid_exists(pid)
    finally:
        if pid_file.exists():
            try:
                psutil.Process(int(pid_file.read_text())).kill()
            except psutil.NoSuchProcess:
                pass
