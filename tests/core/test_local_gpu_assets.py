from __future__ import annotations

import hashlib
import json
from pathlib import Path

from puripuly_heart.core.local_asr.local_gpu_assets import (
    LOCAL_QWEN_GPU_ENGINE,
    LOCAL_QWEN_GPU_MODEL_FILENAME,
    inspect_local_gpu_install,
    load_local_gpu_asset_manifest,
    local_gpu_model_path,
)
from puripuly_heart.core.local_asr.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
)


def _fixture_manifest(payload: bytes) -> LocalSTTAssetManifest:
    return LocalSTTAssetManifest(
        manifest_version=1,
        installed_manifest_version=1,
        model_id=LOCAL_QWEN_GPU_MODEL_ID,
        engine=LOCAL_QWEN_GPU_ENGINE,
        upstream_repo="fixture@revision",
        install_dirname="gpu-fixture",
        sources={
            "fixture": LocalSTTAssetSource(
                name="fixture",
                revision="revision",
                repo_id="fixture/repo",
                download_url_template="https://example.invalid/{relative_path}",
            )
        },
        files=(
            LocalSTTAssetFile(
                relative_path=LOCAL_QWEN_GPU_MODEL_FILENAME,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            ),
        ),
    )


def _write_installed_contract(root: Path, manifest: LocalSTTAssetManifest) -> Path:
    install_root = root / manifest.install_dirname
    install_root.mkdir(parents=True)
    installed = {
        "manifest_version": manifest.installed_manifest_version,
        "model_id": manifest.model_id,
        "engine": manifest.engine,
        "install_dirname": manifest.install_dirname,
        "selected_source": "fixture",
        "selected_revision": "revision",
    }
    (install_root / manifest.installed_manifest_filename).write_text(
        json.dumps(installed),
        encoding="utf-8",
    )
    return install_root


def test_gpu_manifest_pins_one_strict_vulkan_q6_k_asset() -> None:
    manifest = load_local_gpu_asset_manifest()

    assert manifest.model_id == LOCAL_QWEN_GPU_MODEL_ID
    assert manifest.engine == LOCAL_QWEN_GPU_ENGINE
    assert manifest.sources["huggingface"].revision == ("92282af1610a2db19d66f2bef1e260f5deca782d")
    assert [(item.relative_path, item.size_bytes, item.sha256) for item in manifest.files] == [
        (
            LOCAL_QWEN_GPU_MODEL_FILENAME,
            1_692_554_208,
            "c75a961b7134a6c952d89797865cb0d0376876185aee04ef6d12c31c2952e4e1",
        )
    ]
    source = manifest.sources["huggingface"]
    assert source.download_url_template.format(path=manifest.files[0].relative_path) == (
        "https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/"
        "92282af1610a2db19d66f2bef1e260f5deca782d/Qwen3-ASR-1.7B-Q6_K.gguf"
    )


def test_gpu_install_is_not_inspected_before_explicit_opt_in(tmp_path: Path) -> None:
    snapshot = inspect_local_gpu_install(explicit_opt_in=False, model_root=tmp_path)

    assert snapshot.status == "not_requested"
    assert not snapshot.activation_allowed
    assert snapshot.state is None
    assert list(tmp_path.iterdir()) == []


def test_gpu_install_requires_its_own_exact_asset_contract(tmp_path: Path) -> None:
    payload = b"gpu-model-fixture"
    manifest = _fixture_manifest(payload)

    missing = inspect_local_gpu_install(
        explicit_opt_in=True,
        model_root=tmp_path,
        manifest=manifest,
    )
    assert missing.status == "missing"
    assert not missing.activation_allowed

    install_root = _write_installed_contract(tmp_path, manifest)
    model_path = install_root / LOCAL_QWEN_GPU_MODEL_FILENAME
    model_path.write_bytes(b"wrong")
    invalid = inspect_local_gpu_install(
        explicit_opt_in=True,
        model_root=tmp_path,
        manifest=manifest,
    )
    assert invalid.status == "invalid"
    assert not invalid.activation_allowed

    model_path.write_bytes(payload)
    ready = inspect_local_gpu_install(
        explicit_opt_in=True,
        model_root=tmp_path,
        manifest=manifest,
    )
    assert ready.status == "ready"
    assert ready.activation_allowed
    assert local_gpu_model_path(tmp_path, manifest=manifest) == model_path
