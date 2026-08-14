from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from puripuly_heart.core.local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    InstalledLocalSTTManifest,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
    LocalSTTInstallState,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_catalog import (
    PARAKEET_JAPANESE_SUPPORTED_LANGUAGE_CODES,
    PARAKEET_V3_SUPPORTED_LANGUAGE_CODES,
    QWEN_06B_SUPPORTED_LANGUAGE_CODES,
    LocalCPUInstallSnapshot,
    LocalCPUModelInstall,
    LocalSTTUnsupportedLanguageError,
    inspect_required_cpu_model_installs,
    local_cpu_model_supports_language,
    resolve_cpu_auto_model,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(model_id: str, payload: bytes) -> LocalSTTAssetManifest:
    return LocalSTTAssetManifest(
        manifest_version=1,
        installed_manifest_version=1,
        model_id=model_id,
        engine="sherpa-onnx",
        upstream_repo=f"test/{model_id}",
        install_dirname=model_id,
        sources={
            "test": LocalSTTAssetSource(
                name="test",
                revision=f"{model_id}-revision",
            )
        },
        files=(
            LocalSTTAssetFile(
                relative_path="model.bin",
                sha256=_sha256(payload),
                size_bytes=len(payload),
            ),
        ),
    )


def _write_install(root: Path, manifest: LocalSTTAssetManifest, payload: bytes) -> None:
    model_dir = root / manifest.install_dirname
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(payload)
    installed = InstalledLocalSTTManifest(
        manifest_version=manifest.installed_manifest_version,
        model_id=manifest.model_id,
        engine=manifest.engine,
        install_dirname=manifest.install_dirname,
        selected_source="test",
        selected_revision=manifest.sources["test"].revision,
    )
    (model_dir / manifest.installed_manifest_filename).write_text(
        json.dumps(installed.to_dict()),
        encoding="utf-8",
    )


def test_required_cpu_manifests_are_independent_and_pinned() -> None:
    manifests = {
        model_id: load_local_stt_asset_manifest(model_id)
        for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
    }

    assert set(manifests) == {
        PARAKEET_V3_MODEL_ID,
        PARAKEET_JAPANESE_MODEL_ID,
        LOCAL_STT_MODEL_ID,
    }
    assert len({manifest.install_dirname for manifest in manifests.values()}) == 3
    assert all(manifest.files for manifest in manifests.values())
    assert all(manifest.sources for manifest in manifests.values())
    assert all(
        len(source.revision) == 40
        for manifest in manifests.values()
        for source in manifest.sources.values()
    )
    assert all(
        len(asset.sha256) == 64 and asset.size_bytes is not None
        for manifest in manifests.values()
        for asset in manifest.files
    )


def test_cpu_auto_routes_every_pinned_capability_boundary() -> None:
    assert PARAKEET_JAPANESE_SUPPORTED_LANGUAGE_CODES == {"ja"}
    assert resolve_cpu_auto_model("ja") == PARAKEET_JAPANESE_MODEL_ID
    for language_code in PARAKEET_V3_SUPPORTED_LANGUAGE_CODES:
        assert resolve_cpu_auto_model(language_code) == PARAKEET_V3_MODEL_ID
    for language_code in QWEN_06B_SUPPORTED_LANGUAGE_CODES - {
        "ja",
        *PARAKEET_V3_SUPPORTED_LANGUAGE_CODES,
    }:
        assert resolve_cpu_auto_model(language_code) == LOCAL_STT_MODEL_ID


@pytest.mark.parametrize("language_code", ["ca", "no", "sr", "", "unknown"])
def test_cpu_auto_rejects_codes_outside_pinned_capability_union(language_code: str) -> None:
    with pytest.raises(LocalSTTUnsupportedLanguageError):
        resolve_cpu_auto_model(language_code)


def test_cpu_capability_normalizes_configured_regional_codes_without_expanding_union() -> None:
    assert resolve_cpu_auto_model("ja-JP") == PARAKEET_JAPANESE_MODEL_ID
    assert resolve_cpu_auto_model("EN_us") == PARAKEET_V3_MODEL_ID
    assert resolve_cpu_auto_model("zh-CN") == LOCAL_STT_MODEL_ID
    assert local_cpu_model_supports_language(PARAKEET_JAPANESE_MODEL_ID, "ja-JP") is True
    assert local_cpu_model_supports_language(PARAKEET_JAPANESE_MODEL_ID, "ko-KR") is False


def test_cpu_auto_gate_requires_three_independently_checksum_valid_installs(
    tmp_path: Path,
) -> None:
    payloads = {
        PARAKEET_V3_MODEL_ID: b"parakeet-v3",
        PARAKEET_JAPANESE_MODEL_ID: b"parakeet-ja",
        LOCAL_STT_MODEL_ID: b"qwen",
    }
    manifests = {model_id: _manifest(model_id, payload) for model_id, payload in payloads.items()}
    for model_id, payload in payloads.items():
        _write_install(tmp_path, manifests[model_id], payload)

    ready = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)

    assert ready.cpu_auto_available is True
    assert all(ready.state_for(model_id).status == "ready" for model_id in payloads)

    japanese_model = tmp_path / PARAKEET_JAPANESE_MODEL_ID / "model.bin"
    japanese_model.write_bytes(b"corrupt")
    partial = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)

    assert partial.cpu_auto_available is False
    assert partial.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert partial.state_for(PARAKEET_V3_MODEL_ID).status == "ready"
    assert partial.state_for(LOCAL_STT_MODEL_ID).status == "ready"


def test_one_model_installed_manifest_cannot_satisfy_another_model(
    tmp_path: Path,
) -> None:
    manifests = {
        model_id: _manifest(model_id, model_id.encode())
        for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
    }
    for model_id, manifest in manifests.items():
        _write_install(tmp_path, manifest, model_id.encode())

    japanese_dir = tmp_path / PARAKEET_JAPANESE_MODEL_ID
    parakeet_v3_installed = (tmp_path / PARAKEET_V3_MODEL_ID / "installed-manifest.json").read_text(
        encoding="utf-8"
    )
    (japanese_dir / "installed-manifest.json").write_text(
        parakeet_v3_installed,
        encoding="utf-8",
    )

    snapshot = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)

    assert snapshot.cpu_auto_available is False
    assert snapshot.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert snapshot.state_for(PARAKEET_V3_MODEL_ID).status == "ready"


def test_one_manifest_authority_cannot_be_injected_for_all_required_models(
    tmp_path: Path,
) -> None:
    parakeet_v3 = _manifest(PARAKEET_V3_MODEL_ID, b"parakeet-v3")
    _write_install(tmp_path, parakeet_v3, b"parakeet-v3")
    manifests = {model_id: parakeet_v3 for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS}

    snapshot = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)

    assert snapshot.cpu_auto_available is False
    assert snapshot.state_for(PARAKEET_V3_MODEL_ID).status == "ready"
    assert snapshot.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert snapshot.state_for(LOCAL_STT_MODEL_ID).status == "invalid"


def test_cpu_auto_snapshot_gate_requires_exact_unique_model_identities() -> None:
    installed = InstalledLocalSTTManifest(
        manifest_version=1,
        model_id=PARAKEET_V3_MODEL_ID,
        engine="sherpa-onnx",
        install_dirname=PARAKEET_V3_MODEL_ID,
        selected_source="test",
        selected_revision="revision",
    )
    duplicate = LocalCPUModelInstall(
        model_id=PARAKEET_V3_MODEL_ID,
        state=LocalSTTInstallState(status="ready", installed_manifest=installed),
    )

    snapshot = LocalCPUInstallSnapshot(models=(duplicate, duplicate, duplicate))

    assert snapshot.cpu_auto_available is False
