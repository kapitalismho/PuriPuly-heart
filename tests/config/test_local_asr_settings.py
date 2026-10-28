from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext import compat, serialization
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    with_telemetry_enabled,
)

LOCAL_PROVIDER_VALUES = (
    STTProviderName.LOCAL_CPU_AUTO,
    STTProviderName.LOCAL_ORUKEET,
    STTProviderName.LOCAL_FUN_ASR_NANO,
    STTProviderName.LOCAL_FUN_ASR_MLT_NANO,
    STTProviderName.LOCAL_PARAKEET_JAPANESE,
    STTProviderName.LOCAL_QWEN_GPU,
)


@pytest.mark.parametrize("provider", LOCAL_PROVIDER_VALUES)
def test_local_provider_identities_roundtrip_through_canonical_serialization(
    provider: STTProviderName,
) -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            stt=replace(
                current.intent.stt,
                provider=provider.value,
                gpu_device_id="vulkan-device-2",
            ),
            peer_stt=replace(current.intent.peer_stt, provider=provider.value),
        ),
    )

    persisted = serialization.to_dict(settings)
    restored = serialization.from_dict(persisted)

    assert persisted["intent"]["stt"]["provider"] == provider.value
    assert persisted["intent"]["peer_stt"]["provider"] == provider.value
    assert persisted["intent"]["stt"]["gpu_device_id"] == "vulkan-device-2"
    assert restored.intent.stt.provider == provider.value
    assert restored.intent.peer_stt.provider == provider.value
    assert restored.intent.stt.gpu_device_id == "vulkan-device-2"


@pytest.mark.parametrize(
    ("self_provider", "peer_provider", "expected_self", "expected_peer"),
    [
        ("local_qwen", "deepgram", "local_cpu_auto", "rolling_free"),
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


@pytest.mark.parametrize("version", [53, VNEXT_SETTINGS_SCHEMA_VERSION])
def test_retired_choices_migrate_recoverably_and_remain_canonical_on_restart(
    tmp_path: Path,
    version: int,
) -> None:
    path = tmp_path / "settings.json"
    default = with_telemetry_enabled(
        AppSettingsVNext(),
        True,
        identifier_factory=lambda: "cpu-migration-test-id",
    )
    raw = serialization.to_dict(default)
    raw["settings_version"] = version
    raw["intent"]["stt"]["provider"] = "local_parakeet_v3"
    raw["intent"]["peer_stt"]["provider"] = "local_qwen"
    raw["intent"]["stt"]["gpu_device_id"] = "vulkan-device-1"
    raw["intent"]["languages"]["source_language"] = "fr"
    raw["intent"]["languages"]["peer_source_language"] = "ko"
    raw["intent"]["stt"]["custom_terms"] = {"fr": ["PuriPuly"]}
    original = json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(original)

    loaded = compat.load_vnext_settings(path)

    assert loaded.ok
    assert loaded.migrated
    assert loaded.backup_path is not None
    assert loaded.backup_path.read_bytes() == original
    assert loaded.settings is not None
    assert loaded.settings.intent.stt.provider == "local_orukeet"
    assert loaded.settings.intent.peer_stt.provider == "local_cpu_auto"
    assert loaded.settings.intent.stt.gpu_device_id == "vulkan-device-1"
    assert loaded.settings.intent.languages.source_language == "fr"
    assert loaded.settings.intent.languages.peer_source_language == "ko"
    assert loaded.settings.intent.stt.custom_terms == {"fr": ["PuriPuly"]}
    persisted = path.read_bytes()
    assert json.loads(persisted)["intent"]["stt"]["provider"] == "local_orukeet"
    assert json.loads(persisted)["intent"]["peer_stt"]["provider"] == "local_cpu_auto"

    restarted = compat.load_vnext_settings(path)

    assert restarted.ok
    assert not restarted.migrated
    assert restarted.backup_path is None
    assert restarted.settings == loaded.settings
    assert path.read_bytes() == persisted
