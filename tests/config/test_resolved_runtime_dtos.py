from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import pytest

from tests.helpers.ast_sources import imported_modules

DTO_MODULE_NAME = "puripuly_heart.config.resolved"

RAW_SECRET_BEARING_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential_value",
        "headers",
        "password",
        "secret",
        "secret_value",
        "token",
        "token_value",
    }
)

ALLOWED_INTERNAL_IMPORTS = frozenset(
    {
        "puripuly_heart.config.audio_host_api",
        "puripuly_heart.config.llm_profiles",
    }
)
FORBIDDEN_INTERNAL_IMPORT_PREFIXES = (
    "puripuly_heart.app",
    "puripuly_heart.config.settings",
    "puripuly_heart.core.storage",
    "puripuly_heart.providers",
    "puripuly_heart.ui",
)
FORBIDDEN_EXTERNAL_IMPORT_ROOTS = frozenset({"flet", "httpx", "keyring", "requests"})
FORBIDDEN_FILE_IO_CALL_NAMES = frozenset({"open"})
FORBIDDEN_FILE_IO_ATTR_CALLS = frozenset(
    {"mkdir", "open", "read_bytes", "read_text", "unlink", "write_bytes", "write_text"}
)


def _resolved_module() -> ModuleType:
    return importlib.import_module(DTO_MODULE_NAME)


def _load_boundary_guard() -> ModuleType:
    guard_path = (
        Path(__file__).resolve().parents[1] / "architecture" / "test_dependency_boundaries.py"
    )
    spec = importlib.util.spec_from_file_location("_resolved_dto_boundary_guard", guard_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_no_file_io_calls(source_path: Path) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_FILE_IO_CALL_NAMES
        if isinstance(node.func, ast.Attribute):
            assert node.func.attr not in FORBIDDEN_FILE_IO_ATTR_CALLS


def _credential(resolved: ModuleType, *, required: bool = True) -> Any:
    if required:
        return resolved.ResolvedCredentialRequirement(
            source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
            required=True,
            reference="llm/openrouter/byok",
        )
    return resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_NONE,
        required=False,
        reference=None,
    )


def _llm_config(resolved: ModuleType, *, provider_options: dict[str, object] | None = None) -> Any:
    return resolved.ResolvedLLMConfig(
        primary=resolved.ResolvedLLMTarget(
            provider="openrouter",
            model="google/gemma-4-26b-a4b-it",
            credential=_credential(resolved),
            routing_mode="latency",
            provider_routing="default",
            provider_options={} if provider_options is None else provider_options,
        ),
        fallback=resolved.ResolvedLLMFallbackPlan(
            target=resolved.ResolvedLLMTarget(
                provider="openrouter",
                model="deepseek/deepseek-v4-flash-0731",
                credential=_credential(resolved, required=False),
                provider_routing="deepseek_only",
            )
        ),
        concurrency_limit=5,
    )


def _stt_config(
    resolved: ModuleType,
    *,
    channel: str,
    source_language: str = "ko-KR",
    input_device: str | None,
    output_device: str | None,
    provider_options: dict[str, object] | None = None,
) -> Any:
    return resolved.ResolvedSTTConfig(
        channel=channel,
        source_language=source_language,
        provider="local_qwen" if channel == resolved.RUNTIME_CHANNEL_SELF else "soniox",
        model="qwen3-asr-flash-realtime",
        endpoint="wss://example.invalid/stt",
        region="beijing",
        credential=_credential(resolved, required=False),
        input_host_api="Windows WASAPI",
        input_device=input_device,
        output_device=output_device,
        sample_rate_hz=16000,
        channels=1,
        ring_buffer_ms=500,
        drain_timeout_s=2.0,
        vad_speech_threshold=0.5,
        vad_hangover_ms=600,
        vad_pre_roll_ms=500,
        low_latency_enabled=True,
        low_latency_merge_gap_ms=600,
        low_latency_spec_retry_max=10,
        custom_vocabulary_enabled=True,
        custom_terms={"en": ["airi"], "ko": ["아이리"]},
        provider_options={} if provider_options is None else provider_options,
    )


def test_resolved_module_is_import_safe_and_dependency_light() -> None:
    resolved = _resolved_module()
    source_path = Path(resolved.__file__ or "")

    assert source_path.name == "resolved.py"
    imported = imported_modules(source_path)
    for imported_module in imported:
        if imported_module.startswith("puripuly_heart."):
            assert imported_module in ALLOWED_INTERNAL_IMPORTS
        assert not imported_module.startswith(FORBIDDEN_INTERNAL_IMPORT_PREFIXES)
        assert imported_module.split(".", 1)[0] not in FORBIDDEN_EXTERNAL_IMPORT_ROOTS
    _assert_no_file_io_calls(source_path)


def test_resolved_contracts_expose_dtos_and_literal_constants() -> None:
    resolved = _resolved_module()

    for class_name in (
        "ResolvedCredentialRequirement",
        "ResolvedLLMConfig",
        "ResolvedLLMFallbackPlan",
        "ResolvedLLMTarget",
        "ResolvedOverlayConfig",
        "ResolvedRuntimePolicy",
        "ResolvedSTTConfig",
    ):
        assert hasattr(resolved, class_name)

    assert resolved.RUNTIME_CHANNEL_SELF == "self"
    assert resolved.RUNTIME_CHANNEL_PEER == "peer"
    assert resolved.RUNTIME_CHANNELS == ("self", "peer")
    assert resolved.CREDENTIAL_SOURCES == ("none", "secret_store", "managed")
    assert resolved.RESOLVED_FEATURE_STATES == ("enabled", "disabled")
    assert resolved.OVERLAY_TARGETS == ("steamvr", "desktop")


def test_resolved_llm_config_carries_explicit_fallback_branch_target() -> None:
    resolved = _resolved_module()

    llm_config = resolved.ResolvedLLMConfig(
        primary=resolved.ResolvedLLMTarget(
            provider="openrouter",
            model="google/gemma-4-26b-a4b-it",
            credential=_credential(resolved),
            routing_mode="latency",
            provider_routing="default",
        ),
        fallback=resolved.ResolvedLLMFallbackPlan(
            target=resolved.ResolvedLLMTarget(
                provider="openrouter",
                model="deepseek/deepseek-v4-flash-0731",
                credential=_credential(resolved),
                provider_routing="deepseek_only",
            )
        ),
    )
    no_fallback = resolved.ResolvedLLMConfig(
        primary=resolved.ResolvedLLMTarget(
            provider="openrouter",
            model="google/gemma-4-26b-a4b-it",
            credential=_credential(resolved),
        ),
    )

    assert llm_config.fallback is not None
    assert llm_config.fallback.target.provider_routing == "deepseek_only"
    assert no_fallback.fallback is None


def test_resolved_dtos_are_frozen_and_slotted() -> None:
    resolved = _resolved_module()
    dto_classes = (
        resolved.ResolvedCredentialRequirement,
        resolved.ResolvedLLMConfig,
        resolved.ResolvedLLMFallbackPlan,
        resolved.ResolvedLLMTarget,
        resolved.ResolvedOverlayConfig,
        resolved.ResolvedRuntimePolicy,
        resolved.ResolvedSTTConfig,
    )

    for dto_class in dto_classes:
        assert is_dataclass(dto_class)
        assert dto_class.__dataclass_params__.frozen is True
        assert hasattr(dto_class, "__slots__")
        assert "__dict__" not in dto_class.__slots__

    llm_config = _llm_config(resolved)
    with pytest.raises(FrozenInstanceError):
        llm_config.primary = llm_config.primary


def test_nested_mappings_are_deep_frozen_and_detached_from_inputs() -> None:
    resolved = _resolved_module()
    original_options = {"outer": {"inner": ["alpha"]}}
    llm_config = _llm_config(resolved, provider_options=original_options)

    original_options["outer"]["inner"].append("mutated")
    original_options["outer"]["new"] = "mutated"

    assert isinstance(llm_config.primary.provider_options, MappingProxyType)
    assert isinstance(llm_config.primary.provider_options["outer"], MappingProxyType)
    assert llm_config.primary.provider_options["outer"]["inner"] == ("alpha",)
    with pytest.raises(TypeError):
        llm_config.primary.provider_options["new"] = "value"
    with pytest.raises(TypeError):
        llm_config.primary.provider_options["outer"]["new"] = "value"

    custom_terms = {"en": ["airi"]}
    stt_config = _stt_config(
        resolved,
        channel=resolved.RUNTIME_CHANNEL_SELF,
        input_device="Microphone Array",
        output_device=None,
        provider_options={},
    )
    detached_stt_config = resolved.ResolvedSTTConfig(
        **{
            **{field.name: getattr(stt_config, field.name) for field in fields(stt_config)},
            "custom_terms": custom_terms,
        }
    )
    custom_terms["en"].append("mutated")

    assert isinstance(detached_stt_config.custom_terms, MappingProxyType)
    assert detached_stt_config.custom_terms["en"] == ("airi",)


def test_self_and_peer_stt_share_one_resolved_shape_with_channel_values() -> None:
    resolved = _resolved_module()
    self_stt = _stt_config(
        resolved,
        channel=resolved.RUNTIME_CHANNEL_SELF,
        input_device="Microphone Array",
        output_device=None,
        provider_options={"capture": {"source": "mic"}},
    )
    peer_stt = _stt_config(
        resolved,
        channel=resolved.RUNTIME_CHANNEL_PEER,
        source_language="zh-CN",
        input_device=None,
        output_device="Steam Streaming Speakers",
        provider_options={"capture": {"source": "desktop"}},
    )

    assert type(self_stt) is type(peer_stt) is resolved.ResolvedSTTConfig
    assert tuple(field.name for field in fields(self_stt)) == tuple(
        field.name for field in fields(peer_stt)
    )
    assert self_stt.channel == "self"
    assert self_stt.source_language == "ko-KR"
    assert self_stt.input_device == "Microphone Array"
    assert self_stt.output_device is None
    assert peer_stt.channel == "peer"
    assert peer_stt.source_language == "zh-CN"
    assert peer_stt.input_device is None
    assert peer_stt.output_device == "Steam Streaming Speakers"


def test_overlay_and_runtime_policy_mappings_are_immutable() -> None:
    resolved = _resolved_module()
    overlay_field_names = {field.name for field in fields(resolved.ResolvedOverlayConfig)}

    assert "desktop_overlay_options" in overlay_field_names
    assert "desktop_flet" not in overlay_field_names

    overlay = resolved.ResolvedOverlayConfig(
        enabled=True,
        target=resolved.OVERLAY_TARGET_STEAMVR,
        show_translation=True,
        show_peer_original=False,
        calibration={"position": {"x": 1.0, "y": 2.0}},
        desktop_overlay_options={"visual": {"background_alpha": 0.6}},
    )
    policy = resolved.ResolvedRuntimePolicy(
        translation=resolved.RESOLVED_FEATURE_ENABLED,
        peer_translation=resolved.RESOLVED_FEATURE_DISABLED,
        integrated_context=resolved.RESOLVED_FEATURE_ENABLED,
        clipboard_auto_translate=resolved.RESOLVED_FEATURE_DISABLED,
        llm_concurrency_limit=5,
        policy_options={"diagnostics": {"visibility": "basic"}},
    )

    assert isinstance(overlay.calibration, MappingProxyType)
    assert isinstance(overlay.desktop_overlay_options["visual"], MappingProxyType)
    assert isinstance(policy.policy_options, MappingProxyType)
    assert isinstance(policy.policy_options["diagnostics"], MappingProxyType)
    with pytest.raises(TypeError):
        overlay.calibration["position"] = {"x": 0.0, "y": 0.0}
    with pytest.raises(TypeError):
        policy.policy_options["diagnostics"] = {"visibility": "detailed"}


def test_raw_secret_bearing_names_are_excluded_from_dto_fields_and_options() -> None:
    resolved = _resolved_module()
    dto_classes = (
        resolved.ResolvedCredentialRequirement,
        resolved.ResolvedLLMConfig,
        resolved.ResolvedLLMFallbackPlan,
        resolved.ResolvedLLMTarget,
        resolved.ResolvedOverlayConfig,
        resolved.ResolvedRuntimePolicy,
        resolved.ResolvedSTTConfig,
    )

    for dto_class in dto_classes:
        assert not (
            RAW_SECRET_BEARING_FIELD_NAMES & {field.name.lower() for field in fields(dto_class)}
        )

    with pytest.raises(ValueError, match="secret-bearing"):
        _llm_config(resolved, provider_options={"api_key": "placeholder"})
    with pytest.raises(ValueError, match="secret-bearing"):
        _stt_config(
            resolved,
            channel=resolved.RUNTIME_CHANNEL_SELF,
            input_device="Microphone Array",
            output_device=None,
            provider_options={"secret_value": "placeholder"},
        )


@pytest.mark.parametrize(
    "sensitive_key",
    (
        "access_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "headers",
        "secret",
        "token",
    ),
)
def test_sensitive_resolved_option_names_are_rejected(sensitive_key: str) -> None:
    resolved = _resolved_module()

    with pytest.raises(ValueError, match="secret-bearing"):
        _llm_config(resolved, provider_options={sensitive_key: "placeholder"})


def test_sensitive_resolved_option_names_are_rejected_in_nested_mappings() -> None:
    resolved = _resolved_module()

    with pytest.raises(ValueError, match="secret-bearing"):
        _stt_config(
            resolved,
            channel=resolved.RUNTIME_CHANNEL_SELF,
            input_device="Microphone Array",
            output_device=None,
            provider_options={"auth": {"client_secret": "placeholder"}},
        )
    with pytest.raises(ValueError, match="secret-bearing"):
        resolved.ResolvedOverlayConfig(
            enabled=True,
            target=resolved.OVERLAY_TARGET_DESKTOP,
            show_translation=True,
            show_peer_original=True,
            calibration={},
            desktop_overlay_options={"http": {"headers": {"Authorization": "Bearer x"}}},
        )
    with pytest.raises(ValueError, match="secret-bearing"):
        resolved.ResolvedRuntimePolicy(
            translation=resolved.RESOLVED_FEATURE_ENABLED,
            peer_translation=resolved.RESOLVED_FEATURE_ENABLED,
            integrated_context=resolved.RESOLVED_FEATURE_ENABLED,
            clipboard_auto_translate=resolved.RESOLVED_FEATURE_DISABLED,
            llm_concurrency_limit=5,
            policy_options={"broker": {"access_token": "placeholder"}},
        )


def test_resolved_dto_layer_is_covered_by_dependency_boundary_guard() -> None:
    resolved = _resolved_module()
    guard = _load_boundary_guard()

    assert guard._layer_for_module(resolved.__name__) == guard.RESOLVED_DTOS
    rule = guard._rule_for_layer(guard.RESOLVED_DTOS)
    assert resolved.__name__ in rule.prefixes
    assert rule.rule_id == "resolved-dtos-stay-pure"
