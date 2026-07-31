from __future__ import annotations

import codecs
import hashlib
import http.server
import json
import shutil
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT as ROOT

POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
SCRIPT_PATH = ROOT / "scripts" / "installer" / "install-local-stt-model.ps1"


class _QuietHttpHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        _ = format, args


@pytest.fixture()
def windows_temp_dir() -> Path:
    base_dir = ROOT / ".pytest_cache"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="local-stt-installer-", dir=base_dir))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture()
def file_server(windows_temp_dir: Path):
    source_dir = windows_temp_dir / "source"
    source_dir.mkdir()

    handler = lambda *args, **kwargs: _QuietHttpHandler(  # noqa: E731
        *args, directory=str(source_dir), **kwargs
    )
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield source_dir, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _to_windows_path(path: Path) -> str:
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def _run_installer_script(
    *, manifest_path: Path, app_data_root: Path, selected_source: str
) -> None:
    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _to_windows_path(SCRIPT_PATH),
            "-ManifestPath",
            _to_windows_path(manifest_path),
            "-AppDataRoot",
            _to_windows_path(app_data_root),
            "-SelectedSource",
            selected_source,
        ],
        capture_output=True,
        text=False,
        check=True,
    )
    assert completed.returncode == 0, completed.stderr


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_manifest(
    *,
    base_url: str,
    model_bytes: bytes,
    token_bytes: bytes,
    missing_secondary: bool = False,
) -> dict[str, object]:
    files = [
        {
            "relative_path": "model.int8.onnx",
            "sha256": _sha256_bytes(model_bytes),
        },
        {
            "relative_path": "tokens.txt",
            "sha256": _sha256_bytes(token_bytes),
        },
    ]
    if missing_secondary:
        files.append(
            {
                "relative_path": "missing.txt",
                "sha256": "aaaabbbbccccddddeeeeffff0000111122223333444455556666777788889999",
            }
        )

    return {
        "manifest_version": 1,
        "installed_manifest_version": 1,
        "model_id": "qwen3-asr-0.6b-int8-sherpa",
        "engine": "sherpa-onnx",
        "upstream_repo": "example/repo",
        "install_dirname": "qwen3-asr-0.6b-int8-sherpa",
        "installed_manifest_filename": "installed-manifest.json",
        "sources": {
            "huggingface": {
                "name": "huggingface",
                "revision": "hf-rev-1",
                "download_url_template": f"{base_url}/{{path}}",
            },
            "modelscope": {
                "name": "modelscope",
                "revision": "ms-rev-1",
                "download_url_template": f"{base_url}/{{path}}",
            },
        },
        "files": files,
    }


def test_installer_script_recovers_invalid_existing_install_and_writes_bomless_manifest(
    windows_temp_dir: Path,
    file_server,
) -> None:
    if not POWERSHELL.exists():
        pytest.skip("Windows PowerShell is not available")

    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokens.txt").write_bytes(token_bytes)

    manifest_path = windows_temp_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                base_url=base_url,
                model_bytes=model_bytes,
                token_bytes=token_bytes,
            )
        ),
        encoding="utf-8",
    )

    app_data_root = windows_temp_dir / "appdata"
    install_dir = app_data_root / "models" / "qwen3-asr-0.6b-int8-sherpa"
    install_dir.mkdir(parents=True)
    (install_dir / "installed-manifest.json").write_text("{invalid", encoding="utf-8")

    _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source="huggingface",
    )

    installed_manifest = install_dir / "installed-manifest.json"
    assert installed_manifest.exists()
    assert not installed_manifest.read_bytes().startswith(codecs.BOM_UTF8)
    assert (
        json.loads(installed_manifest.read_text(encoding="utf-8"))["selected_source"]
        == "huggingface"
    )
    assert (install_dir / "model.int8.onnx").read_bytes() == b"model-bytes"
    assert (install_dir / "tokens.txt").read_bytes() == b"token-bytes"


def test_installer_script_preserves_existing_install_until_staging_succeeds(
    windows_temp_dir: Path,
    file_server,
) -> None:
    if not POWERSHELL.exists():
        pytest.skip("Windows PowerShell is not available")

    source_dir, base_url = file_server
    model_bytes = b"new-model-bytes"
    token_bytes = b"new-token-bytes"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokens.txt").write_bytes(token_bytes)

    manifest_path = windows_temp_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _build_manifest(
                base_url=base_url,
                model_bytes=model_bytes,
                token_bytes=token_bytes,
                missing_secondary=True,
            )
        ),
        encoding="utf-8",
    )

    app_data_root = windows_temp_dir / "appdata"
    install_dir = app_data_root / "models" / "qwen3-asr-0.6b-int8-sherpa"
    install_dir.mkdir(parents=True)
    (install_dir / "model.int8.onnx").write_bytes(b"old-model-bytes")
    (install_dir / "tokens.txt").write_bytes(b"old-token-bytes")
    (install_dir / "installed-manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "model_id": "qwen3-asr-0.6b-int8-sherpa",
                "engine": "sherpa-onnx",
                "install_dirname": "qwen3-asr-0.6b-int8-sherpa",
                "selected_source": "huggingface",
                "selected_revision": "hf-rev-1",
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _to_windows_path(SCRIPT_PATH),
            "-ManifestPath",
            _to_windows_path(manifest_path),
            "-AppDataRoot",
            _to_windows_path(app_data_root),
            "-SelectedSource",
            "huggingface",
            "-Reinstall",
        ],
        capture_output=True,
        text=False,
    )

    assert completed.returncode != 0
    assert (install_dir / "model.int8.onnx").read_bytes() == b"old-model-bytes"
    assert (install_dir / "tokens.txt").read_bytes() == b"old-token-bytes"
    assert (
        json.loads((install_dir / "installed-manifest.json").read_text(encoding="utf-8"))[
            "selected_revision"
        ]
        == "hf-rev-1"
    )


def test_promote_staging_install_restores_backup_when_destination_move_fails(
    windows_temp_dir: Path,
) -> None:
    if not POWERSHELL.exists():
        pytest.skip("Windows PowerShell is not available")

    install_dir = windows_temp_dir / "install"
    install_dir.mkdir()
    (install_dir / "existing.txt").write_text("old", encoding="utf-8")
    missing_staging_dir = windows_temp_dir / "missing-staging"
    backup_dir = Path(str(install_dir) + ".backup")

    script_text = SCRIPT_PATH.read_text(encoding="utf-8")
    functions_only = script_text.split("$manifest = Read-JsonObject -Path $ManifestPath", 1)[0]
    functions_only = functions_only.split(
        '$DefaultInstalledManifestFilename = "installed-manifest.json"\n', 1
    )[1]
    command = "\n".join(
        [
            '$ErrorActionPreference = "Stop"',
            "Set-StrictMode -Version Latest",
            '$DefaultInstalledManifestFilename = "installed-manifest.json"',
            functions_only,
            (
                "try { "
                f"Promote-StagingInstall -StagingDir '{_to_windows_path(missing_staging_dir)}' "
                f"-InstallDir '{_to_windows_path(install_dir)}'; "
                "exit 0 "
                "} catch { exit 1 }"
            ),
        ]
    )

    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=False,
    )

    assert completed.returncode != 0
    assert (install_dir / "existing.txt").read_text(encoding="utf-8") == "old"
    assert not backup_dir.exists()
