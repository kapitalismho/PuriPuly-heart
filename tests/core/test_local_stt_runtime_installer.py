from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import shutil
import socketserver
import tempfile
import threading
from pathlib import Path

import pytest

from puripuly_heart.core.local_stt_assets import (
    InstalledLocalSTTManifest,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
    inspect_local_stt_install_state,
)
from puripuly_heart.core.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallCancelled,
    LocalSTTRuntimeInstallError,
    RuntimeLocalSTTStatusUpdate,
    cleanup_local_stt_install_residue,
    ensure_local_stt_installed,
)


class _QuietHttpHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        _ = format, args


@pytest.fixture()
def temp_dir() -> Path:
    base_dir = Path(tempfile.mkdtemp(prefix="local-stt-runtime-installer-"))
    try:
        yield base_dir
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_cleanup_restores_backup_and_removes_only_known_staging(tmp_path: Path) -> None:
    backup = tmp_path / "known-model.backup"
    staging = tmp_path / "known-model.staging-abcd"
    unrelated = tmp_path / "other-model.staging-abcd"
    backup.mkdir()
    staging.mkdir()
    unrelated.mkdir()
    (backup / "model.bin").write_bytes(b"ready")

    reconciled = cleanup_local_stt_install_residue(
        model_root=tmp_path,
        install_dirnames=("known-model",),
    )

    assert {path.name for path in reconciled} == {backup.name, staging.name}
    assert (tmp_path / "known-model" / "model.bin").read_bytes() == b"ready"
    assert not backup.exists()
    assert not staging.exists()
    assert unrelated.exists()


def test_cleanup_rejects_unsafe_install_directory_names(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()

    with pytest.raises(ValueError, match="safe directory names"):
        cleanup_local_stt_install_residue(
            model_root=tmp_path,
            install_dirnames=("../sentinel",),
        )

    assert sentinel.exists()


@pytest.fixture()
def file_server(temp_dir: Path):
    source_dir = temp_dir / "source"
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_manifest(
    *,
    huggingface_url: str,
    modelscope_url: str,
    model_bytes: bytes,
    token_bytes: bytes,
    model_id: str = "qwen3-asr-0.6b-int8-sherpa",
) -> LocalSTTAssetManifest:
    return LocalSTTAssetManifest(
        manifest_version=1,
        installed_manifest_version=1,
        model_id=model_id,
        engine="sherpa-onnx",
        upstream_repo="example/repo",
        install_dirname=model_id,
        sources={
            "huggingface": LocalSTTAssetSource(
                name="huggingface",
                revision="hf-rev-1",
                download_url_template=f"{huggingface_url}/{{path}}",
            ),
            "modelscope": LocalSTTAssetSource(
                name="modelscope",
                revision="ms-rev-1",
                download_url_template=f"{modelscope_url}/{{path}}",
            ),
        },
        files=(
            LocalSTTAssetFile(
                relative_path="model.int8.onnx",
                sha256=_sha256_bytes(model_bytes),
                size_bytes=len(model_bytes),
                source_path_overrides={"modelscope": "modelscope/model.int8.onnx"},
            ),
            LocalSTTAssetFile(
                relative_path="tokenizer/tokens.txt",
                sha256=_sha256_bytes(token_bytes),
                size_bytes=len(token_bytes),
            ),
        ),
    )


def _write_manifest_file(path: Path, payload: InstalledLocalSTTManifest) -> None:
    path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_downloads_preferred_source_successfully(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/unused",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    installed = await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
        on_status=statuses.append,
    )

    assert installed.selected_source == "huggingface"
    assert statuses[0] == RuntimeLocalSTTStatusUpdate(status="downloading", percent=0)
    assert statuses[-1] == RuntimeLocalSTTStatusUpdate(status="ready", percent=None)
    state = inspect_local_stt_install_state(
        (temp_dir / "appdata" / "models") / manifest.install_dirname, manifest=manifest
    )
    assert state.status == "ready"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_falls_back_once_when_preferred_source_fails(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "modelscope").mkdir()
    (source_dir / "modelscope" / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/missing",
        modelscope_url=base_url,
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )

    installed = await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
    )

    assert installed.selected_source == "modelscope"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_uses_source_specific_remote_paths(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "modelscope").mkdir()
    (source_dir / "modelscope" / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/missing",
        modelscope_url=base_url,
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )

    installed = await ensure_local_stt_installed(
        preferred_source="modelscope",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
    )

    assert installed.selected_source == "modelscope"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_raises_after_both_sources_fail(
    temp_dir: Path,
    file_server,
) -> None:
    _source_dir, base_url = file_server
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/missing-a",
        modelscope_url=f"{base_url}/missing-b",
        model_bytes=b"model-bytes",
        token_bytes=b"token-bytes",
    )
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    with pytest.raises(LocalSTTRuntimeInstallError):
        await ensure_local_stt_installed(
            preferred_source="huggingface",
            model_root=temp_dir / "appdata" / "models",
            manifest=manifest,
            on_status=statuses.append,
        )

    assert statuses[0] == RuntimeLocalSTTStatusUpdate(status="downloading", percent=0)
    assert statuses[-1] == RuntimeLocalSTTStatusUpdate(status="download_failed", percent=None)
    assert not ((temp_dir / "appdata" / "models") / manifest.install_dirname).exists()


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_reports_monotonic_overall_progress(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"m" * 70
    token_bytes = b"t" * 30
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/unused",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
        on_status=statuses.append,
    )

    percents = [status.percent for status in statuses if status.status == "downloading"]
    assert percents
    assert percents == sorted(percents)
    assert percents[0] == 0
    assert percents[-1] < 100


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_aborts_when_cancelled_before_download(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    (source_dir / "model.int8.onnx").write_bytes(b"model-bytes")
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(b"token-bytes")
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/unused",
        model_bytes=b"model-bytes",
        token_bytes=b"token-bytes",
    )
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(LocalSTTRuntimeInstallCancelled):
        await ensure_local_stt_installed(
            preferred_source="huggingface",
            model_root=temp_dir / "appdata" / "models",
            manifest=manifest,
            cancel_event=cancel_event,
        )

    assert not ((temp_dir / "appdata" / "models") / manifest.install_dirname).exists()


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_rejects_checksum_mismatch_before_promotion(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    (source_dir / "model.int8.onnx").write_bytes(b"corrupted-model")
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(b"token-bytes")
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/missing",
        model_bytes=b"model-bytes",
        token_bytes=b"token-bytes",
    )
    model_root = temp_dir / "appdata" / "models"
    install_dir = model_root / manifest.install_dirname
    install_dir.mkdir(parents=True)
    (install_dir / "model.int8.onnx").write_bytes(b"old-model")
    (install_dir / "tokenizer").mkdir()
    (install_dir / "tokenizer" / "tokens.txt").write_bytes(b"old-token")
    _write_manifest_file(
        install_dir / "installed-manifest.json",
        InstalledLocalSTTManifest(
            manifest_version=manifest.installed_manifest_version,
            model_id=manifest.model_id,
            engine=manifest.engine,
            install_dirname=manifest.install_dirname,
            selected_source="huggingface",
            selected_revision="old-rev",
        ),
    )

    with pytest.raises(LocalSTTRuntimeInstallError):
        await ensure_local_stt_installed(
            preferred_source="huggingface",
            model_root=model_root,
            manifest=manifest,
        )

    assert (install_dir / "model.int8.onnx").read_bytes() == b"old-model"
    assert (install_dir / "tokenizer" / "tokens.txt").read_bytes() == b"old-token"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_redownloads_structurally_valid_but_checksum_corrupt_install(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"repaired-model"
    token_bytes = b"repaired-token"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/missing",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    model_root = temp_dir / "appdata" / "models"
    install_dir = model_root / manifest.install_dirname
    install_dir.mkdir(parents=True)
    (install_dir / "model.int8.onnx").write_bytes(b"corrupt-existing-model")
    (install_dir / "tokenizer").mkdir()
    (install_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    _write_manifest_file(
        install_dir / "installed-manifest.json",
        InstalledLocalSTTManifest(
            manifest_version=manifest.installed_manifest_version,
            model_id=manifest.model_id,
            engine=manifest.engine,
            install_dirname=manifest.install_dirname,
            selected_source="huggingface",
            selected_revision=manifest.sources["huggingface"].revision,
        ),
    )

    # Cheap runtime inspection intentionally treats checksum-corrupt installs as ready.
    state = inspect_local_stt_install_state(install_dir, manifest=manifest)
    assert state.status == "ready"

    installed = await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=model_root,
        manifest=manifest,
    )

    assert installed.selected_source == "huggingface"
    assert (install_dir / "model.int8.onnx").read_bytes() == model_bytes
    assert (install_dir / "tokenizer" / "tokens.txt").read_bytes() == token_bytes


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_recovers_invalid_existing_install(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"new-model"
    token_bytes = b"new-token"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/missing",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    model_root = temp_dir / "appdata" / "models"
    install_dir = model_root / manifest.install_dirname
    install_dir.mkdir(parents=True)
    (install_dir / "model.int8.onnx").write_bytes(b"broken")
    (install_dir / "installed-manifest.json").write_text("{invalid", encoding="utf-8")

    installed = await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=model_root,
        manifest=manifest,
    )

    assert installed.selected_source == "huggingface"
    assert (install_dir / "model.int8.onnx").read_bytes() == model_bytes
    assert (install_dir / "tokenizer" / "tokens.txt").read_bytes() == token_bytes


@pytest.mark.asyncio
async def test_independent_cpu_model_installs_survive_one_model_download_failure(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_ids = ("parakeet-v3", "parakeet-ja", "qwen")
    manifests: list[LocalSTTAssetManifest] = []
    for model_id in model_ids:
        model_bytes = f"{model_id}-model".encode()
        token_bytes = f"{model_id}-tokens".encode()
        model_source = source_dir / model_id
        if model_id != "parakeet-ja":
            model_source.mkdir()
            (model_source / "model.int8.onnx").write_bytes(model_bytes)
            (model_source / "tokenizer").mkdir()
            (model_source / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
        manifests.append(
            _build_manifest(
                huggingface_url=f"{base_url}/{model_id}",
                modelscope_url=f"{base_url}/{model_id}",
                model_bytes=model_bytes,
                token_bytes=token_bytes,
                model_id=model_id,
            )
        )
    model_root = temp_dir / "appdata" / "models"

    results = await asyncio.gather(
        *(
            ensure_local_stt_installed(
                model_id=manifest.model_id,
                preferred_source="huggingface",
                model_root=model_root,
                manifest=manifest,
            )
            for manifest in manifests
        ),
        return_exceptions=True,
    )

    assert isinstance(results[0], InstalledLocalSTTManifest)
    assert isinstance(results[1], LocalSTTRuntimeInstallError)
    assert isinstance(results[2], InstalledLocalSTTManifest)
    assert (
        inspect_local_stt_install_state(
            model_root / manifests[0].install_dirname,
            manifest=manifests[0],
            verify_checksums=True,
        ).status
        == "ready"
    )
    assert (
        inspect_local_stt_install_state(
            model_root / manifests[1].install_dirname,
            manifest=manifests[1],
            verify_checksums=True,
        ).status
        == "missing"
    )
    assert (
        inspect_local_stt_install_state(
            model_root / manifests[2].install_dirname,
            manifest=manifests[2],
            verify_checksums=True,
        ).status
        == "ready"
    )


@pytest.mark.asyncio
async def test_runtime_install_rejects_cross_model_manifest_target() -> None:
    manifest = _build_manifest(
        huggingface_url="https://example.invalid",
        modelscope_url="https://example.invalid",
        model_bytes=b"model",
        token_bytes=b"tokens",
        model_id="model-a",
    )

    with pytest.raises(LocalSTTRuntimeInstallError, match="does not match"):
        await ensure_local_stt_installed(model_id="model-b", manifest=manifest)
