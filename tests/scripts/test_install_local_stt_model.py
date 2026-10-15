from __future__ import annotations

import codecs
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT as ROOT

POWERSHELL = Path(
    shutil.which("powershell")
    or shutil.which("pwsh")
    or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
)
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

    requests: list[str] = []

    class RecordingHandler(_QuietHttpHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            super().do_GET()

    def handler(*args, **kwargs):
        return RecordingHandler(*args, directory=str(source_dir), **kwargs)

    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield source_dir, f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _to_windows_path(path: Path) -> str:
    if os.name == "nt" or not str(POWERSHELL).endswith(".exe"):
        return str(path.resolve())
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def _run_installer_script(
    *,
    manifest_path: Path,
    app_data_root: Path,
    selected_source: str,
    reinstall: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    command = [
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
    ]
    if reinstall:
        command.append("-Reinstall")
    return subprocess.run(command, capture_output=True, text=False, check=check)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_manifest(
    *,
    base_url: str,
    model_bytes: bytes,
    token_bytes: bytes,
    missing_secondary: bool = False,
    source_urls: dict[str, str] | None = None,
) -> dict[str, object]:
    files = [
        {
            "relative_path": "model.int8.onnx",
            "size_bytes": len(model_bytes),
            "sha256": _sha256_bytes(model_bytes),
        },
        {
            "relative_path": "tokens.txt",
            "size_bytes": len(token_bytes),
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

    if source_urls is None:
        source_urls = {"huggingface": base_url, "modelscope": base_url}

    return {
        "manifest_version": 1,
        "installed_manifest_version": 1,
        "model_id": "qwen3-asr-0.6b-int8-sherpa",
        "engine": "sherpa-onnx",
        "upstream_repo": "example/repo",
        "install_dirname": "qwen3-asr-0.6b-int8-sherpa",
        "installed_manifest_filename": "installed-manifest.json",
        "sources": {
            name: {
                "name": name,
                "revision": "ms-rev-1" if name == "modelscope" else "hf-rev-1",
                "download_url_template": f"{url}/{{path}}",
            }
            for name, url in source_urls.items()
        },
        "files": files,
    }


def test_installer_script_recovers_invalid_existing_install_and_writes_bomless_manifest(
    windows_temp_dir: Path,
    file_server,
) -> None:
    if not POWERSHELL.exists():
        pytest.skip("Windows PowerShell is not available")

    source_dir, base_url, _ = file_server
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

    source_dir, base_url, _ = file_server
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

    completed = _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source="huggingface",
        reinstall=True,
        check=False,
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
    assert not list(install_dir.parent.glob("*.staging-*"))
    assert not Path(str(install_dir) + ".backup").exists()


@pytest.fixture()
def mirror_install_case(windows_temp_dir: Path, file_server):
    if not POWERSHELL.exists():
        pytest.skip("PowerShell is not available")

    source_dir, base_url, requests = file_server
    model_bytes = b"verified-model"
    token_bytes = b"verified-tokens"
    manifest = _build_manifest(
        base_url=base_url,
        model_bytes=model_bytes,
        token_bytes=token_bytes,
        source_urls={
            "huggingface": f"{base_url}/huggingface",
            "hf_mirror": f"{base_url}/hf_mirror",
        },
    )
    model_id = "parakeet-tdt-ctc-0.6b-ja-int8-sherpa"
    manifest["model_id"] = model_id
    manifest["install_dirname"] = model_id
    manifest["files"][0]["source_path_overrides"] = {"hf_mirror": "nested/mirrored-model.onnx"}
    for source_name in ("huggingface", "hf_mirror"):
        source_root = source_dir / source_name
        source_root.mkdir()
        remote_path = (
            "nested/mirrored-model.onnx" if source_name == "hf_mirror" else "model.int8.onnx"
        )
        model_path = source_root / remote_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(model_bytes)
        (source_root / "tokens.txt").write_bytes(token_bytes)

    manifest_path = windows_temp_dir / "mirror-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    app_data_root = windows_temp_dir / "mirror-appdata"
    install_dir = app_data_root / "models" / model_id
    return manifest_path, app_data_root, install_dir, source_dir, requests


@pytest.mark.parametrize("selected_source", ["huggingface", "hf_mirror"])
def test_installer_script_installs_preferred_http_source(
    mirror_install_case, selected_source: str
) -> None:
    manifest_path, app_data_root, install_dir, _, requests = mirror_install_case

    _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source=selected_source,
    )

    assert (install_dir / "model.int8.onnx").read_bytes() == b"verified-model"
    assert (install_dir / "tokens.txt").read_bytes() == b"verified-tokens"
    installed_bytes = (install_dir / "installed-manifest.json").read_bytes()
    assert not installed_bytes.startswith(codecs.BOM_UTF8)
    installed = json.loads(installed_bytes)
    assert installed["selected_source"] == selected_source
    assert installed["selected_revision"] == "hf-rev-1"
    remote_model = (
        "nested/mirrored-model.onnx" if selected_source == "hf_mirror" else "model.int8.onnx"
    )
    assert requests == [
        f"/{selected_source}/{remote_model}",
        f"/{selected_source}/tokens.txt",
    ]
    assert not list(install_dir.parent.glob("*.staging-*"))
    assert not Path(str(install_dir) + ".backup").exists()


@pytest.mark.parametrize("selected_source", ["huggingface", "hf_mirror"])
@pytest.mark.parametrize("failure", ["http", "checksum"])
def test_installer_script_falls_back_after_transfer_or_checksum_failure(
    mirror_install_case, selected_source: str, failure: str
) -> None:
    manifest_path, app_data_root, install_dir, source_dir, requests = mirror_install_case
    tokens_path = source_dir / selected_source / "tokens.txt"
    if failure == "http":
        tokens_path.unlink()
    else:
        tokens_path.write_bytes(b"corrupt--tokens")
    alternate_source = "huggingface" if selected_source == "hf_mirror" else "hf_mirror"

    _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source=selected_source,
    )

    assert (install_dir / "model.int8.onnx").read_bytes() == b"verified-model"
    assert (install_dir / "tokens.txt").read_bytes() == b"verified-tokens"
    installed = json.loads((install_dir / "installed-manifest.json").read_bytes())
    assert installed["selected_source"] == alternate_source
    assert installed["selected_revision"] == "hf-rev-1"
    assert requests[0].startswith(f"/{selected_source}/")
    assert requests[-1] == f"/{alternate_source}/tokens.txt"
    assert len(requests) == 4
    assert not list(install_dir.parent.glob("*.staging-*"))
    assert not Path(str(install_dir) + ".backup").exists()


@pytest.mark.parametrize("installed_source", ["huggingface", "hf_mirror"])
def test_installer_script_reuses_valid_install_from_either_registered_source(
    mirror_install_case, installed_source: str
) -> None:
    manifest_path, app_data_root, install_dir, source_dir, requests = mirror_install_case
    _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source=installed_source,
    )
    installed_bytes = (install_dir / "installed-manifest.json").read_bytes()
    shutil.rmtree(source_dir)
    requests.clear()
    preferred_source = "huggingface" if installed_source == "hf_mirror" else "hf_mirror"

    _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source=preferred_source,
    )

    assert requests == []
    assert (install_dir / "installed-manifest.json").read_bytes() == installed_bytes
    assert (install_dir / "model.int8.onnx").read_bytes() == b"verified-model"
    assert (install_dir / "tokens.txt").read_bytes() == b"verified-tokens"
    assert not list(install_dir.parent.glob("*.staging-*"))


def test_installer_script_preserves_install_when_both_mirror_sources_fail(
    mirror_install_case,
) -> None:
    manifest_path, app_data_root, install_dir, source_dir, requests = mirror_install_case
    _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source="huggingface",
    )
    installed_bytes = (install_dir / "installed-manifest.json").read_bytes()
    (install_dir / "user-data.txt").write_bytes(b"keep-user-data")
    (source_dir / "hf_mirror" / "tokens.txt").write_bytes(b"corrupt--tokens")
    (source_dir / "huggingface" / "tokens.txt").unlink()
    requests.clear()

    completed = _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source="hf_mirror",
        reinstall=True,
        check=False,
    )

    assert completed.returncode != 0
    assert requests[0].startswith("/hf_mirror/")
    assert requests[-1] == "/huggingface/tokens.txt"
    assert (install_dir / "installed-manifest.json").read_bytes() == installed_bytes
    assert (install_dir / "model.int8.onnx").read_bytes() == b"verified-model"
    assert (install_dir / "tokens.txt").read_bytes() == b"verified-tokens"
    assert (install_dir / "user-data.txt").read_bytes() == b"keep-user-data"
    assert not list(install_dir.parent.glob("*.staging-*"))
    assert not Path(str(install_dir) + ".backup").exists()


def test_installer_script_does_not_fall_back_after_invalid_source_configuration(
    mirror_install_case,
) -> None:
    manifest_path, app_data_root, install_dir, _, requests = mirror_install_case
    manifest = json.loads(manifest_path.read_bytes())
    del manifest["sources"]["hf_mirror"]["download_url_template"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = _run_installer_script(
        manifest_path=manifest_path,
        app_data_root=app_data_root,
        selected_source="hf_mirror",
        check=False,
    )

    assert completed.returncode != 0
    assert requests == []
    assert not install_dir.exists()
    assert not list(install_dir.parent.glob("*.staging-*"))
