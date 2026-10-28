from __future__ import annotations

import codecs
import hashlib
import json
from pathlib import Path

import pytest

from puripuly_heart.core.local_asr import local_stt_assets as local_stt_assets_module
from puripuly_heart.core.local_asr.local_stt_assets import (
    ORUKEET_MODEL_ID,
    InstalledLocalSTTManifest,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
    LocalSTTInstallState,
    LocalSTTManifestInvalidError,
    default_local_stt_installed_manifest_path,
    default_local_stt_model_dir,
    default_local_stt_model_root,
    default_local_stt_source_for_locale,
    inspect_local_stt_install_state,
    load_local_stt_asset_manifest,
    validate_local_stt_install,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _test_manifest() -> LocalSTTAssetManifest:
    return LocalSTTAssetManifest(
        manifest_version=1,
        installed_manifest_version=1,
        model_id="test-model",
        engine="sherpa-onnx",
        upstream_repo="test/repository",
        install_dirname="test-model",
        sources={
            "huggingface": LocalSTTAssetSource(
                name="huggingface",
                revision="hf-rev-1",
            ),
            "modelscope": LocalSTTAssetSource(
                name="modelscope",
                revision="ms-rev-1",
            ),
        },
        files=(
            LocalSTTAssetFile(
                relative_path="model.int8.onnx",
                sha256=_sha256_bytes(b"model-bytes"),
            ),
            LocalSTTAssetFile(
                relative_path="tokens.txt",
                sha256=_sha256_bytes(b"token-bytes"),
            ),
        ),
    )


def _write_valid_install(root: Path, manifest: LocalSTTAssetManifest) -> None:
    (root / "model.int8.onnx").write_bytes(b"model-bytes")
    (root / "tokens.txt").write_bytes(b"token-bytes")
    installed = InstalledLocalSTTManifest(
        manifest_version=manifest.installed_manifest_version,
        model_id=manifest.model_id,
        engine=manifest.engine,
        install_dirname=manifest.install_dirname,
        selected_source="huggingface",
        selected_revision=manifest.sources["huggingface"].revision,
    )
    default_local_stt_installed_manifest_path(root).write_text(
        json.dumps(installed.to_dict(), indent=2),
        encoding="utf-8",
    )


def test_default_local_stt_paths_use_user_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    from puripuly_heart.config import paths

    def _fake_user_config_dir(*, app_dir_name: str = paths.APP_DIR_NAME) -> Path:
        return Path("/tmp") / app_dir_name

    monkeypatch.setattr(paths, "user_config_dir", _fake_user_config_dir)

    assert default_local_stt_model_root() == Path("/tmp") / paths.APP_DIR_NAME / "models"
    model_dir = default_local_stt_model_dir(ORUKEET_MODEL_ID)
    assert model_dir == default_local_stt_model_root() / ORUKEET_MODEL_ID
    assert default_local_stt_installed_manifest_path(model_dir) == (
        model_dir / "installed-manifest.json"
    )


@pytest.mark.parametrize(
    "retired_model_id",
    ["qwen3-asr-0.6b-int8-sherpa", "parakeet-tdt-0.6b-v3-int8-sherpa"],
)
def test_retired_models_are_not_discoverable_or_downloadable(retired_model_id: str) -> None:
    with pytest.raises(LocalSTTManifestInvalidError, match="unknown local STT model_id"):
        load_local_stt_asset_manifest(retired_model_id)
    with pytest.raises(LocalSTTManifestInvalidError, match="unknown local STT model_id"):
        default_local_stt_model_dir(retired_model_id)


def test_validate_local_stt_install_accepts_matching_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)

    installed = validate_local_stt_install(install_dir, manifest=manifest)

    assert installed.selected_source == "huggingface"
    assert installed.selected_revision == "hf-rev-1"


def test_validate_local_stt_runtime_ready_accepts_matching_manifest_without_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)

    def fail_if_hashed(_path: Path) -> str:
        raise AssertionError("runtime validator must not hash files")

    monkeypatch.setattr(local_stt_assets_module, "_sha256_file", fail_if_hashed)

    installed = local_stt_assets_module.validate_local_stt_runtime_ready(
        install_dir,
        manifest=manifest,
    )

    assert installed.selected_source == "huggingface"
    assert installed.selected_revision == "hf-rev-1"


def test_validate_local_stt_install_rejects_missing_required_file(tmp_path: Path) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)
    (install_dir / "tokens.txt").unlink()

    with pytest.raises(LocalSTTManifestInvalidError, match="missing required model file"):
        validate_local_stt_install(install_dir, manifest=manifest)


def test_validate_local_stt_runtime_ready_rejects_missing_required_file(tmp_path: Path) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)
    (install_dir / "tokens.txt").unlink()

    with pytest.raises(LocalSTTManifestInvalidError, match="missing required model file"):
        local_stt_assets_module.validate_local_stt_runtime_ready(
            install_dir,
            manifest=manifest,
        )


def test_validate_local_stt_install_rejects_stale_revision(tmp_path: Path) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)

    stale_manifest = InstalledLocalSTTManifest(
        manifest_version=manifest.installed_manifest_version,
        model_id=manifest.model_id,
        engine=manifest.engine,
        install_dirname=manifest.install_dirname,
        selected_source="huggingface",
        selected_revision="old-revision",
    )
    default_local_stt_installed_manifest_path(install_dir).write_text(
        json.dumps(stale_manifest.to_dict(), indent=2),
        encoding="utf-8",
    )

    with pytest.raises(LocalSTTManifestInvalidError, match="stale installed manifest revision"):
        validate_local_stt_install(install_dir, manifest=manifest)


def test_validate_local_stt_install_rejects_checksum_mismatch(tmp_path: Path) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)
    (install_dir / "model.int8.onnx").write_bytes(b"corrupted")

    with pytest.raises(LocalSTTManifestInvalidError, match="checksum mismatch"):
        validate_local_stt_install(install_dir, manifest=manifest)


def test_validate_local_stt_install_accepts_bom_prefixed_installed_manifest(
    tmp_path: Path,
) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)

    installed_manifest_path = default_local_stt_installed_manifest_path(install_dir)
    payload = installed_manifest_path.read_text(encoding="utf-8")
    installed_manifest_path.write_bytes(codecs.BOM_UTF8 + payload.encode("utf-8"))

    installed = validate_local_stt_install(install_dir, manifest=manifest)

    assert installed.selected_source == "huggingface"


def test_validate_local_stt_install_wraps_invalid_json_manifest(tmp_path: Path) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    (install_dir / "model.int8.onnx").write_bytes(b"model-bytes")
    (install_dir / "tokens.txt").write_bytes(b"token-bytes")
    default_local_stt_installed_manifest_path(install_dir).write_text("{invalid", encoding="utf-8")

    with pytest.raises(LocalSTTManifestInvalidError, match="invalid local STT installed manifest"):
        validate_local_stt_install(install_dir, manifest=manifest)


def test_validate_local_stt_install_wraps_missing_manifest_fields(tmp_path: Path) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    (install_dir / "model.int8.onnx").write_bytes(b"model-bytes")
    (install_dir / "tokens.txt").write_bytes(b"token-bytes")
    default_local_stt_installed_manifest_path(install_dir).write_text(
        json.dumps(
            {
                "manifest_version": manifest.installed_manifest_version,
                "model_id": manifest.model_id,
                "engine": manifest.engine,
                "install_dirname": manifest.install_dirname,
                "selected_source": "huggingface",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LocalSTTManifestInvalidError, match="invalid local STT installed manifest"):
        validate_local_stt_install(install_dir, manifest=manifest)


def test_inspect_local_stt_install_state_returns_missing_for_absent_install(tmp_path: Path) -> None:
    manifest = _test_manifest()

    state = inspect_local_stt_install_state(
        tmp_path / manifest.install_dirname,
        manifest=manifest,
    )

    assert state == LocalSTTInstallState(status="missing", installed_manifest=None)


def test_inspect_local_stt_install_state_returns_invalid_for_broken_manifest(
    tmp_path: Path,
) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    (install_dir / "model.int8.onnx").write_bytes(b"model-bytes")
    (install_dir / "tokens.txt").write_bytes(b"token-bytes")
    default_local_stt_installed_manifest_path(install_dir).write_text("{invalid", encoding="utf-8")

    state = inspect_local_stt_install_state(install_dir, manifest=manifest)

    assert state.status == "invalid"
    assert state.installed_manifest is None
    assert state.error_message == "invalid local STT installed manifest"


def test_inspect_local_stt_install_state_returns_ready_for_valid_install(tmp_path: Path) -> None:
    manifest = _test_manifest()
    install_dir = tmp_path / manifest.install_dirname
    install_dir.mkdir()
    _write_valid_install(install_dir, manifest)

    state = inspect_local_stt_install_state(install_dir, manifest=manifest)

    assert state.status == "ready"
    assert state.installed_manifest is not None
    assert state.installed_manifest.selected_source == "huggingface"


@pytest.mark.parametrize("locale", ["zh-CN", "zh-cn", "zh-HK", "zh_HK", "zh-Hant-HK"])
def test_chinese_source_prefers_registered_modelscope_or_derived_mirror(locale) -> None:
    from dataclasses import replace

    manifest = _test_manifest()
    assert default_local_stt_source_for_locale(locale, manifest=manifest) == "modelscope"
    huggingface_only = replace(manifest, sources={"huggingface": manifest.sources["huggingface"]})
    assert default_local_stt_source_for_locale(locale, manifest=huggingface_only) == "hf_mirror"


def test_local_stt_asset_file_round_trips_source_path_overrides() -> None:
    payload = {
        "relative_path": "conv_frontend.onnx",
        "sha256": "d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e",
        "size_bytes": 44148281,
        "source_path_overrides": {
            "modelscope": "model_0.6B/conv_frontend.onnx",
        },
    }

    asset = LocalSTTAssetFile.from_dict(payload)

    assert asset.to_dict() == payload
    assert asset.remote_path_for_source("huggingface") == "conv_frontend.onnx"
    assert asset.remote_path_for_source("modelscope") == "model_0.6B/conv_frontend.onnx"
