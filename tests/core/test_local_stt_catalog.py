from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    ORUKEET_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    InstalledLocalSTTManifest,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
    LocalSTTInstallState,
)
from puripuly_heart.core.local_asr.local_stt_catalog import (
    LocalCPUInstallSnapshot,
    LocalCPUModelInstall,
    LocalSTTUnsupportedLanguageError,
    inspect_required_cpu_model_installs,
    local_cpu_model_supports_language,
    resolve_cpu_auto_model,
)

ROUTES = (
    ("en", ORUKEET_MODEL_ID),
    ("zh", FUN_ASR_NANO_MODEL_ID),
    ("ko", FUN_ASR_MLT_NANO_MODEL_ID),
    ("ja", PARAKEET_JAPANESE_MODEL_ID),
)


def _manifest(model_id: str, payload: bytes) -> LocalSTTAssetManifest:
    return LocalSTTAssetManifest(
        manifest_version=1,
        installed_manifest_version=1,
        model_id=model_id,
        engine="test-engine",
        upstream_repo=f"test/{model_id}",
        install_dirname=model_id,
        sources={"test": LocalSTTAssetSource(name="test", revision=f"{model_id}-revision")},
        files=(LocalSTTAssetFile("model.bin", hashlib.sha256(payload).hexdigest(), len(payload)),),
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
        json.dumps(installed.to_dict()), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("language", "model_id"),
    ROUTES
    + tuple(
        (language, ORUKEET_MODEL_ID)
        for language in (
            "bg",
            "hr",
            "cs",
            "da",
            "nl",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "it",
            "lv",
            "lt",
            "mt",
            "pl",
            "pt",
            "ro",
            "ru",
            "sk",
            "sl",
            "es",
            "sv",
            "uk",
        )
    )
    + tuple(
        (language, FUN_ASR_MLT_NANO_MODEL_ID)
        for language in ("vi", "id", "th", "ms", "tl", "ar", "hi", "ga", "yue")
    ),
)
def test_cpu_auto_preserves_primary_routes_and_prefers_orukeet_for_shared_languages(
    language: str, model_id: str
) -> None:
    assert resolve_cpu_auto_model(language) == model_id
    assert local_cpu_model_supports_language(model_id, language)


@pytest.mark.parametrize(
    "language",
    [
        "zh",
        "en",
        "yue",
        "ja",
        "ko",
        "vi",
        "id",
        "th",
        "ms",
        "tl",
        "ar",
        "hi",
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "et",
        "fi",
        "el",
        "hu",
        "ga",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sv",
    ],
)
def test_mlt_support_is_independent_of_cpu_auto_priority(language: str) -> None:
    assert local_cpu_model_supports_language(FUN_ASR_MLT_NANO_MODEL_ID, language)


@pytest.mark.parametrize(
    ("language", "model_id"),
    (
        (" EN_us ", ORUKEET_MODEL_ID),
        ("zh-Hant-HK", FUN_ASR_NANO_MODEL_ID),
        ("ko_KR", FUN_ASR_MLT_NANO_MODEL_ID),
        ("ja-JP", PARAKEET_JAPANESE_MODEL_ID),
        ("pt_BR", ORUKEET_MODEL_ID),
        (" VI_vn ", FUN_ASR_MLT_NANO_MODEL_ID),
        ("yue-HK", FUN_ASR_MLT_NANO_MODEL_ID),
    ),
)
def test_regional_language_canonicalization_preserves_target_route(language, model_id) -> None:
    assert resolve_cpu_auto_model(language) == model_id
    assert local_cpu_model_supports_language(model_id, language)


@pytest.mark.parametrize("language", ["ca", "no", "tr", "auto", "", "unknown"])
def test_unsupported_languages_have_no_hidden_fallback(language: str) -> None:
    with pytest.raises(LocalSTTUnsupportedLanguageError):
        resolve_cpu_auto_model(language)
    assert all(not local_cpu_model_supports_language(model_id, language) for _, model_id in ROUTES)


@pytest.mark.parametrize(
    "retired", ["qwen3-asr-0.6b-int8-sherpa", "parakeet-tdt-0.6b-v3-int8-sherpa"]
)
def test_retired_models_have_no_runtime_language_contract(retired: str) -> None:
    with pytest.raises(KeyError):
        local_cpu_model_supports_language(retired, "en")


@pytest.mark.parametrize("unavailable_model", [model_id for _, model_id in ROUTES])
def test_auto_requires_all_four_independent_valid_installs(tmp_path, unavailable_model) -> None:
    payloads = {model_id: model_id.encode() for _, model_id in ROUTES}
    manifests = {model_id: _manifest(model_id, payload) for model_id, payload in payloads.items()}
    for model_id, payload in payloads.items():
        _write_install(tmp_path, manifests[model_id], payload)
    ready = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)
    assert ready.cpu_auto_available
    (tmp_path / unavailable_model / "model.bin").write_bytes(b"corrupt")
    partial = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)
    assert not partial.cpu_auto_available
    assert partial.state_for(unavailable_model).status == "invalid"
    assert all(
        partial.state_for(model_id).status == "ready"
        for model_id in payloads
        if model_id != unavailable_model
    )


def test_one_model_identity_cannot_satisfy_another(tmp_path: Path) -> None:
    manifests = {
        model_id: _manifest(model_id, model_id.encode())
        for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
    }
    for model_id, manifest in manifests.items():
        _write_install(tmp_path, manifest, model_id.encode())
    japanese_dir = tmp_path / PARAKEET_JAPANESE_MODEL_ID
    other_identity = (tmp_path / ORUKEET_MODEL_ID / "installed-manifest.json").read_bytes()
    (japanese_dir / "installed-manifest.json").write_bytes(other_identity)
    snapshot = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)
    assert not snapshot.cpu_auto_available
    assert snapshot.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert snapshot.state_for(ORUKEET_MODEL_ID).status == "ready"


def test_one_manifest_authority_cannot_satisfy_all_models(tmp_path: Path) -> None:
    orukeet = _manifest(ORUKEET_MODEL_ID, b"orukeet")
    _write_install(tmp_path, orukeet, b"orukeet")
    manifests = {model_id: orukeet for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS}
    snapshot = inspect_required_cpu_model_installs(tmp_path, manifests=manifests)
    assert not snapshot.cpu_auto_available
    assert snapshot.state_for(ORUKEET_MODEL_ID).status == "ready"
    assert all(
        snapshot.state_for(model_id).status == "invalid"
        for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
        if model_id != ORUKEET_MODEL_ID
    )


def test_duplicate_installs_cannot_satisfy_auto() -> None:
    installed = InstalledLocalSTTManifest(
        1, ORUKEET_MODEL_ID, "test-engine", ORUKEET_MODEL_ID, "test", "revision"
    )
    duplicate = LocalCPUModelInstall(ORUKEET_MODEL_ID, LocalSTTInstallState("ready", installed))
    snapshot = LocalCPUInstallSnapshot(models=(duplicate,) * 4)
    assert not snapshot.cpu_auto_available
