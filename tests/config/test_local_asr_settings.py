from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from puripuly_heart.config.settings import (
    SETTINGS_SCHEMA_VERSION,
    AppSettings,
    STTProviderName,
)
from puripuly_heart.config.settings import (
    from_dict as legacy_from_dict,
)
from puripuly_heart.config.settings import (
    to_dict as legacy_to_dict,
)
from puripuly_heart.config.settings_vnext import compat, serialization
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    with_telemetry_consent,
)

LOCAL_PROVIDER_VALUES = (
    STTProviderName.LOCAL_CPU_AUTO,
    STTProviderName.LOCAL_PARAKEET_V3,
    STTProviderName.LOCAL_PARAKEET_JAPANESE,
    STTProviderName.LOCAL_QWEN,
    STTProviderName.LOCAL_QWEN_GPU,
)


@pytest.mark.parametrize("provider", LOCAL_PROVIDER_VALUES)
def test_local_provider_identities_roundtrip_through_legacy_facade(
    provider: STTProviderName,
) -> None:
    settings = AppSettings()
    settings.provider.stt = provider
    settings.provider.peer_stt = provider
    settings.stt.gpu_device_id = "vulkan-device-2"

    persisted = legacy_to_dict(settings)
    restored = legacy_from_dict(persisted)

    assert persisted["provider"] == {
        "stt": provider.value,
        "peer_stt": provider.value,
        "llm": settings.provider.llm.value,
    }
    assert persisted["stt"]["gpu_device_id"] == "vulkan-device-2"
    assert restored.provider.stt == provider
    assert restored.provider.peer_stt == provider
    assert restored.stt.gpu_device_id == "vulkan-device-2"


@pytest.mark.parametrize(
    ("self_provider", "peer_provider", "expected_self", "expected_peer"),
    [
        ("local_qwen", "deepgram", "local_cpu_auto", "deepgram"),
        ("soniox", "local_qwen", "soniox", "local_cpu_auto"),
        ("local_qwen", "local_qwen", "local_cpu_auto", "local_cpu_auto"),
    ],
)
def test_target_schema_29_local_qwen_migration_is_backed_up_and_idempotent(
    self_provider: str,
    peer_provider: str,
    expected_self: str,
    expected_peer: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    assert VNEXT_SETTINGS_SCHEMA_VERSION == 33
    raw["settings_version"] = 29
    raw["intent"]["stt"]["provider"] = self_provider
    raw["intent"]["stt"].pop("gpu_device_id")
    raw["intent"]["peer_stt"]["provider"] = peer_provider
    raw["intent"]["stt"]["vad_speech_threshold"] = 0.37
    raw["intent"]["desktop_audio"]["vad_speech_threshold"] = 0.52
    raw["intent"]["ui"]["locale"] = "ja"
    original_bytes = json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(original_bytes)
    fixed_now = datetime(2026, 7, 17, 2, 3, 4, tzinfo=timezone.utc)

    first = compat.load_vnext_settings(path, now=fixed_now)

    assert first.ok
    assert first.migrated is True
    assert first.backup_path is not None
    assert first.backup_path.read_bytes() == original_bytes
    assert first.settings is not None
    assert first.settings.intent.stt.provider == expected_self
    assert first.settings.intent.peer_stt.provider == expected_peer
    assert first.settings.intent.stt.gpu_device_id == "auto"
    assert first.settings.intent.stt.vad_speech_threshold == 0.37
    assert first.settings.intent.desktop_audio.vad_speech_threshold == 0.52
    assert first.settings.intent.ui.locale == "ja"

    persisted_after_first = path.read_bytes()
    second = compat.load_vnext_settings(path, now=fixed_now)

    assert second.ok
    assert second.migrated is False
    assert second.backup_path is None
    assert second.settings == first.settings
    assert path.read_bytes() == persisted_after_first
    assert len(list(tmp_path.glob("*.bak"))) == 1


def test_legacy_local_qwen_migration_preserves_unrelated_settings_after_backup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    raw = legacy_to_dict(AppSettings())
    raw["settings_version"] = SETTINGS_SCHEMA_VERSION
    raw["provider"]["stt"] = STTProviderName.LOCAL_QWEN.value
    raw["provider"]["peer_stt"] = STTProviderName.LOCAL_QWEN.value
    raw["stt"].pop("gpu_device_id")
    raw["languages"]["source_language"] = "fr"
    raw["languages"]["peer_target_language"] = "ja"
    raw["ui"]["locale"] = "ko"
    original_bytes = json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(original_bytes)

    result = compat.load_vnext_settings(path)

    assert result.ok
    assert result.migrated is True
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original_bytes
    assert result.settings is not None
    assert result.settings.intent.stt.provider == STTProviderName.LOCAL_CPU_AUTO.value
    assert result.settings.intent.peer_stt.provider == STTProviderName.LOCAL_CPU_AUTO.value
    assert result.settings.intent.stt.gpu_device_id == "auto"
    assert result.settings.intent.languages.source_language == "fr"
    assert result.settings.intent.languages.peer_target_language == "ja"
    assert result.settings.intent.ui.locale == "ko"


@pytest.mark.parametrize(
    "source_version",
    [VNEXT_SETTINGS_SCHEMA_VERSION, VNEXT_SETTINGS_SCHEMA_VERSION + 100],
)
def test_current_manual_qwen_selection_and_shared_gpu_device_remain_stable(
    source_version: int,
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    default = with_telemetry_consent(
        AppSettingsVNext(),
        "allow",
        identifier_factory=lambda: "manual-qwen-test-id",
    )
    manual_qwen = replace(
        default,
        intent=replace(
            default.intent,
            stt=replace(
                default.intent.stt,
                provider=STTProviderName.LOCAL_QWEN.value,
                gpu_device_id="vulkan-device-1",
            ),
            peer_stt=replace(
                default.intent.peer_stt,
                provider=STTProviderName.LOCAL_QWEN.value,
            ),
        ),
    )

    raw = serialization.to_dict(manual_qwen)
    raw["settings_version"] = source_version
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = compat.load_vnext_settings(path)

    assert loaded.ok
    assert loaded.migrated is False
    assert loaded.backup_path is None
    assert loaded.settings is not None
    assert loaded.settings.intent.stt.provider == STTProviderName.LOCAL_QWEN.value
    assert loaded.settings.intent.peer_stt.provider == STTProviderName.LOCAL_QWEN.value
    assert loaded.settings.intent.stt.gpu_device_id == "vulkan-device-1"
