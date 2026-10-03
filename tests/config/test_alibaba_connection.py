from __future__ import annotations

import json
from dataclasses import replace

import pytest

from puripuly_heart.app.wiring.wiring_llm_factory import runtime_resolution_input_from_vnext
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_stt_provider_signature_from_vnext,
    build_self_stt_provider_signature_from_vnext,
    peer_stt_runtime_intent_from_vnext,
    self_stt_runtime_intent_from_vnext,
)
from puripuly_heart.config.alibaba_connection import (
    AlibabaRegionalSettings,
    normalize_api_host,
    resolve_alibaba_connection,
    workspace_api_host_region,
)
from puripuly_heart.config.runtime_resolution import resolve_llm_config, resolve_stt_config
from puripuly_heart.config.settings_vnext import compat, serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


@pytest.mark.parametrize(
    ("region", "host", "shared"),
    [
        ("beijing", "work-123.cn-beijing.maas.aliyuncs.com", "dashscope.aliyuncs.com"),
        ("singapore", "work-456.ap-southeast-1.maas.aliyuncs.com", "dashscope-intl.aliyuncs.com"),
    ],
)
@pytest.mark.parametrize("mode", ["legacy_shared", "workspace_dedicated"])
def test_region_mode_protocols_and_execution_agree(region, host, shared, mode) -> None:
    regional = AlibabaRegionalSettings(mode, host if mode == "workspace_dedicated" else "")
    connection = resolve_alibaba_connection(region, regional)
    expected_host = host if mode == "workspace_dedicated" else shared
    assert connection.host == expected_host
    assert connection.credential_reference == f"qwen:{region}"
    assert connection.compatible_url == f"https://{expected_host}/compatible-mode/v1"
    assert connection.native_url == f"https://{expected_host}/api/v1"
    assert connection.websocket_url == f"wss://{expected_host}/api-ws/v1/inference"
    initial = AppSettingsVNext()
    qwen = replace(initial.intent.translation.qwen, region=region, **{region: regional})
    settings = replace(
        initial,
        intent=replace(
            initial.intent,
            translation=replace(
                initial.intent.translation,
                model="qwen38_flash",
                connection="official_byok",
                qwen=qwen,
            ),
            stt=replace(initial.intent.stt, provider="qwen_audio"),
            peer_stt=replace(initial.intent.peer_stt, provider="qwen_audio"),
        ),
    )
    target = resolve_llm_config(runtime_resolution_input_from_vnext(settings)).primary
    self_stt = resolve_stt_config(self_stt_runtime_intent_from_vnext(settings))
    peer_stt = resolve_stt_config(peer_stt_runtime_intent_from_vnext(settings))
    assert target.service_endpoint == connection.native_url
    assert target.credential.reference == connection.credential_reference
    for stt in (self_stt, peer_stt):
        assert stt.endpoint == connection.websocket_url
        assert stt.credential.reference == connection.credential_reference
        assert stt.model == "qwen-audio-3.1-asr-flash-streaming"
    assert connection in build_self_stt_provider_signature_from_vnext(settings)
    assert connection in build_peer_stt_provider_signature_from_vnext(settings)


@pytest.mark.parametrize(
    "value",
    [
        "evil.cn-beijing.maas.aliyuncs.com.attacker.net",
        "work-123.ap-southeast-1.maas.aliyuncs.com",
        "127.0.0.1",
        "localhost",
        "https://user:secret@work-123.cn-beijing.maas.aliyuncs.com",
        "https://work-123.cn-beijing.maas.aliyuncs.com:443",
        "https://work-123.cn-beijing.maas.aliyuncs.com/api/v1/extra",
        "https://work-123.cn-beijing.maas.aliyuncs.com/api/v1?key=private",
        "wss://work-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
    ],
)
def test_unsafe_host_is_rejected_without_echo(value: str) -> None:
    with pytest.raises(ValueError) as failure:
        resolve_alibaba_connection("beijing", AlibabaRegionalSettings("workspace_dedicated", value))
    assert value not in str(failure.value)
    assert "secret" not in str(failure.value)


def test_recognized_https_paste_normalizes_without_suffix_duplication() -> None:
    host = "work-123.cn-beijing.maas.aliyuncs.com"
    assert normalize_api_host(f"https://{host}/compatible-mode/v1", "beijing") == host
    assert normalize_api_host(f"https://{host}/api/v1", "beijing") == host


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "work-123.cn-beijing.maas.aliyuncs.com",
            ("beijing", "work-123.cn-beijing.maas.aliyuncs.com"),
        ),
        (
            "https://Work-456.AP-Southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            ("singapore", "work-456.ap-southeast-1.maas.aliyuncs.com"),
        ),
        ("dashscope.aliyuncs.com", None),
        ("dashscope-intl.aliyuncs.com", None),
        ("evil.cn-beijing.maas.aliyuncs.com.attacker.net", None),
        ("https://user:secret@work-123.cn-beijing.maas.aliyuncs.com", None),
        ("", None),
    ],
)
def test_workspace_api_host_identifies_region_from_host_alone(value, expected) -> None:
    assert workspace_api_host_region(value) == expected


def test_version_48_migrates_legacy_without_changing_region_or_models(tmp_path) -> None:
    settings = AppSettingsVNext()
    old = serialization.to_dict(settings)
    old["settings_version"] = 48
    old["intent"]["translation"]["qwen"] = {"region": "singapore", "llm_model": "qwen3.8-flash"}
    old["intent"]["ui"]["locale"] = "ko"
    original_text = json.dumps(old)
    path = tmp_path / "settings.json"
    path.write_text(original_text, encoding="utf-8")
    migrated = compat.load_vnext_settings(path)
    assert migrated.ok and migrated.migrated and migrated.backup_path is not None
    assert migrated.settings.intent.translation.qwen.region == "singapore"
    assert migrated.settings.intent.translation.qwen.beijing.endpoint_mode == "legacy_shared"
    assert migrated.settings.intent.translation.qwen.singapore.endpoint_mode == "legacy_shared"
    assert migrated.settings.intent.ui.locale == "ko"
    assert compat.load_vnext_settings(path).migrated is False
    assert migrated.backup_path.read_text(encoding="utf-8") == original_text


def test_version_49_gemma_consolidated_settings_gain_shared_connections(tmp_path) -> None:
    settings = AppSettingsVNext()
    old = serialization.to_dict(settings)
    old["settings_version"] = 49
    old["intent"]["translation"].update(
        model="gemma4_26b_31b",
        connection="openrouter",
        connection_history={"gemma4_26b_31b": "openrouter", "qwen38_flash": "official_byok"},
        qwen={"region": "singapore", "llm_model": "qwen3.8-flash"},
    )
    original_text = json.dumps(old)
    path = tmp_path / "settings.json"
    path.write_text(original_text, encoding="utf-8")

    migrated = compat.load_vnext_settings(path)

    assert migrated.ok and migrated.migrated and migrated.backup_path is not None
    translation = migrated.settings.intent.translation
    assert (translation.model, translation.connection) == ("gemma4_26b_31b", "openrouter")
    assert translation.connection_history == {
        "gemma4_26b_31b": "openrouter",
        "qwen38_flash": "official_byok",
    }
    assert translation.qwen.region == "singapore"
    assert translation.qwen.beijing == AlibabaRegionalSettings()
    assert translation.qwen.singapore == AlibabaRegionalSettings()
    assert json.loads(path.read_text(encoding="utf-8"))["settings_version"] == 51
    assert compat.load_vnext_settings(path).migrated is False
    assert migrated.backup_path.read_text(encoding="utf-8") == original_text


def test_dedicated_roundtrip_and_region_switch_preserve_other_region(tmp_path) -> None:
    initial = AppSettingsVNext()
    qwen = replace(
        initial.intent.translation.qwen,
        beijing=AlibabaRegionalSettings(
            "workspace_dedicated", "work-123.cn-beijing.maas.aliyuncs.com", 2
        ),
        singapore=AlibabaRegionalSettings(
            "workspace_dedicated", "work-456.ap-southeast-1.maas.aliyuncs.com", 3
        ),
    )
    settings = replace(
        initial,
        intent=replace(initial.intent, translation=replace(initial.intent.translation, qwen=qwen)),
    )
    path = tmp_path / "settings.json"
    assert compat.save_vnext_settings(path, settings).ok
    loaded = compat.load_vnext_settings(path)
    assert loaded.ok and not loaded.migrated
    assert loaded.settings.intent.translation.qwen == qwen
    switched = replace(
        loaded.settings,
        intent=replace(
            loaded.settings.intent,
            translation=replace(
                loaded.settings.intent.translation, qwen=replace(qwen, region="singapore")
            ),
        ),
    )
    assert compat.save_vnext_settings(path, switched).ok
    restarted = compat.load_vnext_settings(path)
    assert (
        restarted.ok
        and restarted.settings.intent.translation.qwen == switched.intent.translation.qwen
    )
    assert (
        resolve_alibaba_connection(
            "beijing", restarted.settings.intent.translation.qwen.beijing
        ).host
        == "work-123.cn-beijing.maas.aliyuncs.com"
    )
