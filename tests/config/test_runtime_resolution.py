from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.helpers.ast_sources import imported_modules

MODULE_NAME = "puripuly_heart.config.runtime_resolution"

ALLOWED_INTERNAL_IMPORTS = frozenset(
    {
        "puripuly_heart.config.llm_profiles",
        "puripuly_heart.config.resolved",
    }
)
FORBIDDEN_INTERNAL_IMPORT_PREFIXES = (
    "puripuly_heart.app",
    "puripuly_heart.config.settings",
    "puripuly_heart.core.managed_openrouter_broker_client",
    "puripuly_heart.core.storage",
    "puripuly_heart.providers",
    "puripuly_heart.ui",
)
FORBIDDEN_EXTERNAL_IMPORT_ROOTS = frozenset({"flet", "httpx", "keyring", "requests"})
FORBIDDEN_FILE_IO_CALL_NAMES = frozenset({"open"})
FORBIDDEN_FILE_IO_ATTR_CALLS = frozenset(
    {"mkdir", "open", "read_bytes", "read_text", "unlink", "write_bytes", "write_text"}
)


def _runtime_resolution_module() -> ModuleType:
    return importlib.import_module(MODULE_NAME)


def _resolved_module() -> ModuleType:
    return importlib.import_module("puripuly_heart.config.resolved")


def _profiles_module() -> ModuleType:
    return importlib.import_module("puripuly_heart.config.llm_profiles")


def _load_boundary_guard() -> ModuleType:
    guard_path = (
        Path(__file__).resolve().parents[1] / "architecture" / ("test_dependency_boundaries.py")
    )
    spec = importlib.util.spec_from_file_location("_runtime_resolution_boundary_guard", guard_path)
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


def _runtime_input(
    runtime_resolution: ModuleType,
    *,
    model: str,
    connection: str,
    openrouter: Any | None = None,
    direct: Any | None = None,
    translation_fallback: Any | None = None,
    concurrency_limit: int = 5,
) -> Any:
    return runtime_resolution.RuntimeResolutionInput(
        translation=runtime_resolution.TranslationRuntimeIntent(
            model=model,
            connection=connection,
            concurrency_limit=concurrency_limit,
        ),
        translation_fallback=(
            translation_fallback or runtime_resolution.TranslationFallbackRuntimeIntent()
        ),
        openrouter=openrouter or runtime_resolution.OpenRouterRuntimeIntent(),
        direct=direct or runtime_resolution.DirectProviderRuntimeIntent(),
    )


def _credential_assertion(resolved: ModuleType, source: str, reference: str | None) -> Any:
    return resolved.ResolvedCredentialRequirement(
        source=source,
        required=source != resolved.CREDENTIAL_SOURCE_NONE,
        reference=reference,
    )


def test_runtime_resolution_module_is_import_safe_and_dependency_light() -> None:
    runtime_resolution = _runtime_resolution_module()
    source_path = Path(runtime_resolution.__file__ or "")

    assert source_path.name == "runtime_resolution.py"
    imported = imported_modules(source_path)
    for imported_module in imported:
        if imported_module.startswith("puripuly_heart."):
            assert imported_module in ALLOWED_INTERNAL_IMPORTS
        assert not imported_module.startswith(FORBIDDEN_INTERNAL_IMPORT_PREFIXES)
        assert imported_module.split(".", 1)[0] not in FORBIDDEN_EXTERNAL_IMPORT_ROOTS
    _assert_no_file_io_calls(source_path)


def test_runtime_resolution_layer_is_covered_by_dependency_boundary_guard() -> None:
    runtime_resolution = _runtime_resolution_module()
    guard = _load_boundary_guard()

    assert guard._layer_for_module(runtime_resolution.__name__) == guard.RUNTIME_RESOLUTION
    rule = guard._rule_for_layer(guard.RUNTIME_RESOLUTION)
    assert runtime_resolution.__name__ in rule.prefixes
    assert rule.rule_id == "runtime-resolution-stays-pure"


def test_canonical_runtime_intent_contracts_are_frozen_and_slotted() -> None:
    runtime_resolution = _runtime_resolution_module()

    for class_name in (
        "DirectProviderRuntimeIntent",
        "OverlayRuntimeIntent",
        "OpenRouterRuntimeIntent",
        "RuntimeResolutionInput",
        "STTRuntimeIntent",
        "TranslationRuntimeIntent",
    ):
        dto_class = getattr(runtime_resolution, class_name)
        assert is_dataclass(dto_class)
        assert dto_class.__dataclass_params__.frozen is True
        assert hasattr(dto_class, "__slots__")
        assert "__dict__" not in dto_class.__slots__

    intent = runtime_resolution.TranslationRuntimeIntent(
        model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
        connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED,
    )
    with pytest.raises(FrozenInstanceError):
        intent.model = runtime_resolution.TRANSLATION_MODEL_LOCAL_LLM


def test_stt_runtime_resolution_produces_channel_specific_resolved_dto() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_stt_config(
        runtime_resolution.STTRuntimeIntent(
            channel=resolved.RUNTIME_CHANNEL_PEER,
            provider=runtime_resolution.STT_PROVIDER_SONIOX,
            source_language="zh-CN",
            output_device="Steam Streaming Speakers",
            sample_rate_hz=16000,
            vad_speech_threshold=0.62,
            vad_hangover_ms=450,
            vad_pre_roll_ms=275,
            soniox_model="stt-rt-v4-peer",
            soniox_endpoint="wss://peer-soniox.example/realtime",
            soniox_keepalive_interval_s=12.5,
            soniox_trailing_silence_ms=700,
            soniox_language_hints=("zh",),
            soniox_language_hints_strict=True,
        )
    )

    assert isinstance(config, resolved.ResolvedSTTConfig)
    assert config.channel == resolved.RUNTIME_CHANNEL_PEER
    assert config.source_language == "zh-CN"
    assert config.provider == runtime_resolution.STT_PROVIDER_SONIOX
    assert config.model == "stt-rt-v4-peer"
    assert config.endpoint == "wss://peer-soniox.example/realtime"
    assert config.output_device == "Steam Streaming Speakers"
    assert config.sample_rate_hz == 16000
    assert config.vad_speech_threshold == 0.62
    assert config.vad_hangover_ms == 450
    assert config.vad_pre_roll_ms == 275
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="soniox:stt",
    )
    assert config.provider_options == {
        "keepalive_interval_s": 12.5,
        "trailing_silence_ms": 700,
        "language_hints": ("zh",),
        "language_hints_strict": True,
    }


def test_stt_runtime_resolution_resolves_qwen_region_endpoint_and_custom_terms() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_stt_config(
        runtime_resolution.STTRuntimeIntent(
            channel=resolved.RUNTIME_CHANNEL_SELF,
            provider=runtime_resolution.STT_PROVIDER_QWEN_ASR,
            source_language="ko-KR",
            input_host_api="Windows WASAPI",
            input_device="Microphone Array",
            qwen_region=runtime_resolution.QWEN_REGION_SINGAPORE,
            qwen_asr_model="qwen3-asr-custom",
            custom_vocabulary_enabled=True,
            custom_terms={"ko-KR": ("Puripuly", "VRChat")},
        )
    )

    assert config.channel == resolved.RUNTIME_CHANNEL_SELF
    assert config.source_language == "ko-KR"
    assert config.provider == runtime_resolution.STT_PROVIDER_QWEN_ASR
    assert config.model == "qwen3-asr-custom"
    assert config.region == runtime_resolution.QWEN_REGION_SINGAPORE
    assert config.endpoint == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    assert config.input_host_api == "Windows WASAPI"
    assert config.input_device == "Microphone Array"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="qwen:singapore",
    )
    assert config.custom_vocabulary_enabled is True
    assert config.custom_terms == {"ko-KR": ("Puripuly", "VRChat")}


def test_peer_auto_source_mode_requires_provider_capability() -> None:
    runtime_resolution = _runtime_resolution_module()

    gpu = runtime_resolution.resolve_stt_config(
        runtime_resolution.STTRuntimeIntent(
            channel="peer",
            provider=runtime_resolution.STT_PROVIDER_LOCAL_QWEN_GPU,
            source_language="ja",
            source_mode="auto",
        )
    )
    unsupported = runtime_resolution.resolve_stt_config(
        runtime_resolution.STTRuntimeIntent(
            channel="peer",
            provider=runtime_resolution.STT_PROVIDER_LOCAL_CPU_AUTO,
            source_language="ja",
            source_mode="auto",
        )
    )
    self_gpu = runtime_resolution.resolve_stt_config(
        runtime_resolution.STTRuntimeIntent(
            channel="self",
            provider=runtime_resolution.STT_PROVIDER_LOCAL_QWEN_GPU,
            source_language="ja",
            source_mode="auto",
        )
    )

    assert gpu.source_mode == "auto"
    assert unsupported.source_mode == "manual"
    assert self_gpu.source_mode == "manual"


def test_default_peer_stt_runtime_intent_uses_desktop_peer_vad_defaults() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_stt_config(
        runtime_resolution.RuntimeResolutionInput().peer_stt
    )

    assert config.channel == resolved.RUNTIME_CHANNEL_PEER
    assert config.vad_speech_threshold == 0.5
    assert config.vad_hangover_ms == 500
    assert config.vad_pre_roll_ms == 500


def test_default_self_stt_runtime_intent_uses_low_latency_vad_defaults() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_stt_config(runtime_resolution.STTRuntimeIntent())

    assert config.channel == resolved.RUNTIME_CHANNEL_SELF
    assert config.vad_speech_threshold == 0.4
    assert config.vad_hangover_ms == 500
    assert config.vad_pre_roll_ms == 500


@pytest.mark.parametrize(
    "provider",
    [
        "local_cpu_auto",
        "local_parakeet_v3",
        "local_parakeet_ja",
        "local_qwen",
        "local_qwen_gpu",
    ],
)
def test_stt_runtime_resolution_preserves_local_provider_identity(provider: str) -> None:
    runtime_resolution = _runtime_resolution_module()

    config = runtime_resolution.resolve_stt_config(
        runtime_resolution.STTRuntimeIntent(provider=provider)
    )

    assert config.provider == provider
    assert config.credential.source == "none"
    assert config.credential.required is False


def test_soniox_runtime_default_uses_realtime_v5_model() -> None:
    runtime_resolution = _runtime_resolution_module()

    config = runtime_resolution.resolve_stt_config(
        runtime_resolution.STTRuntimeIntent(provider=runtime_resolution.STT_PROVIDER_SONIOX)
    )

    assert config.model == "stt-rt-v5"
    assert runtime_resolution.SONIOX_STT_MODEL_RT_V5 == "stt-rt-v5"


def test_overlay_runtime_resolution_maps_desktop_options_without_legacy_name() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_overlay_config(
        runtime_resolution.OverlayRuntimeIntent(
            enabled=True,
            target=resolved.OVERLAY_TARGET_DESKTOP,
            show_translation=False,
            show_peer_original=True,
            calibration={"distance": 3.0, "offset_x": 1.25},
            desktop_overlay_options={
                "size_preset": "large",
                "position": {"x": 123, "y": 456},
                "locked": True,
                "visual": {"background_alpha": 0.42},
            },
        )
    )

    assert isinstance(config, resolved.ResolvedOverlayConfig)
    assert config.enabled is True
    assert config.target == resolved.OVERLAY_TARGET_DESKTOP
    assert config.show_translation is False
    assert config.show_peer_original is True
    assert config.calibration["distance"] == 3.0
    assert config.desktop_overlay_options["size_preset"] == "large"
    assert config.desktop_overlay_options["position"] == {"x": 123, "y": 456}
    assert config.desktop_overlay_options["locked"] is True
    assert "desktop_flet" not in config.desktop_overlay_options


@pytest.mark.parametrize(
    (
        "model",
        "connection",
        "openrouter_source",
        "expected_provider",
        "expected_model",
        "expected_credential_source",
        "expected_credential_reference",
        "expected_region",
        "expected_provider_routing",
    ),
    [
        (
            "gemma4",
            "managed",
            "managed",
            "openrouter",
            "google/gemma-4-26b-a4b-it",
            "managed",
            "openrouter:managed",
            None,
            "gemma4_26b_latency",
        ),
        (
            "gemma4",
            "openrouter",
            "byok",
            "openrouter",
            "google/gemma-4-26b-a4b-it",
            "secret_store",
            "openrouter:byok",
            None,
            "gemma4_26b_latency",
        ),
        (
            "gemma4",
            "openrouter",
            "none",
            "openrouter",
            "google/gemma-4-26b-a4b-it",
            "secret_store",
            "openrouter:byok",
            None,
            "gemma4_26b_latency",
        ),
        (
            "deepseek_v4_flash",
            "managed",
            "managed",
            "openrouter",
            "deepseek/deepseek-v4-flash-0731",
            "managed",
            "openrouter:managed",
            None,
            "default",
        ),
        (
            "deepseek_v4_flash",
            "managed_china",
            "managed",
            "openrouter",
            "deepseek/deepseek-v4-flash-0731",
            "managed",
            "openrouter:managed_qq",
            None,
            "deepseek_only",
        ),
        (
            "deepseek_v4_flash",
            "openrouter",
            "byok",
            "openrouter",
            "deepseek/deepseek-v4-flash-0731",
            "secret_store",
            "openrouter:byok",
            None,
            "default",
        ),
        (
            "deepseek_v4_flash",
            "official_byok",
            "byok",
            "deepseek",
            "deepseek-v4-flash",
            "secret_store",
            "deepseek:byok",
            None,
            None,
        ),
        (
            "gemini37_flash",
            "openrouter",
            "byok",
            "openrouter",
            "google/gemini-3.7-flash",
            "secret_store",
            "openrouter:byok",
            None,
            "google_gemini_latency",
        ),
        (
            "gemini31_flash_lite",
            "openrouter",
            "byok",
            "openrouter",
            "google/gemini-3.1-flash-lite",
            "secret_store",
            "openrouter:byok",
            None,
            "google_gemini_latency",
        ),
        (
            "gemini37_flash",
            "official_byok",
            "byok",
            "gemini",
            "gemini-3.7-flash",
            "secret_store",
            "gemini:byok",
            None,
            None,
        ),
        (
            "gemini31_flash_lite",
            "official_byok",
            "byok",
            "gemini",
            "gemini-3.1-flash-lite",
            "secret_store",
            "gemini:byok",
            None,
            None,
        ),
        (
            "qwen35_plus",
            "official_byok",
            "byok",
            "qwen",
            "qwen3.5-plus",
            "secret_store",
            "qwen:beijing",
            "beijing",
            None,
        ),
        (
            "gemma4_31b",
            "cerebras",
            "byok",
            "cerebras",
            "gemma-4-31b",
            "secret_store",
            "cerebras:byok",
            None,
            None,
        ),
        (
            "local_llm",
            "ollama",
            "byok",
            "local_llm",
            "llama3.1:8b",
            "none",
            None,
            None,
            None,
        ),
    ],
)
def test_translation_model_connection_matrix_resolves_llm_config(
    model: str,
    connection: str,
    openrouter_source: str,
    expected_provider: str,
    expected_model: str,
    expected_credential_source: str,
    expected_credential_reference: str | None,
    expected_region: str | None,
    expected_provider_routing: str | None,
) -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    openrouter_intent = runtime_resolution.OpenRouterRuntimeIntent(
        selected_source=openrouter_source,
    )

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=model,
            connection=connection,
            openrouter=openrouter_intent,
        )
    )

    assert isinstance(config, resolved.ResolvedLLMConfig)
    assert config.provider == expected_provider
    assert config.model == expected_model
    assert config.credential == _credential_assertion(
        resolved,
        expected_credential_source,
        expected_credential_reference,
    )
    assert config.region == expected_region
    assert config.provider_routing == expected_provider_routing
    assert config.concurrency_limit == 5
    assert config.fallback is None


@pytest.mark.parametrize(
    ("connection", "backend"),
    [
        ("cpu", "cpu"),
        ("gpu", "gpu"),
    ],
)
def test_managed_gemma_resolves_distinct_local_target_without_provider_fallback(
    connection: str,
    backend: str,
) -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_MANAGED_GEMMA,
            connection=connection,
            translation_fallback=runtime_resolution.TranslationFallbackRuntimeIntent(
                enabled=True,
                model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
                connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            ),
        )
    )

    assert config.provider == runtime_resolution.PROVIDER_MANAGED_GEMMA
    assert config.model == runtime_resolution.MANAGED_GEMMA_MODEL
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_NONE,
        required=False,
        reference=None,
    )
    assert config.provider_options == {"backend": backend}
    assert config.fallback is None
    assert len(config.attempts) == 1
    assert config.attempts[0].target == config.primary


def test_legacy_managed_gemma_provider_derives_cpu_product_intent() -> None:
    runtime_resolution = _runtime_resolution_module()

    intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm=runtime_resolution.PROVIDER_MANAGED_GEMMA,
    )

    assert intent.model == runtime_resolution.TRANSLATION_MODEL_MANAGED_GEMMA
    assert intent.connection == runtime_resolution.TRANSLATION_CONNECTION_CPU


@pytest.mark.parametrize(
    (
        "selection_alias",
        "fallback_enabled",
        "fallback_model",
        "fallback_connection",
        "expected_fallback_source",
        "expected_fallback_reference",
        "expected_fallback_provider",
        "expected_fallback_model",
        "expected_provider_routing",
    ),
    [
        (
            "none",
            False,
            "deepseek_v4_flash",
            "official_byok",
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "deepseek_v4_flash_official",
            True,
            "deepseek_v4_flash",
            "official_byok",
            "secret_store",
            "deepseek:byok",
            "deepseek",
            "deepseek-v4-flash",
            None,
        ),
        (
            "openrouter_deepseek_v4_flash",
            True,
            "deepseek_v4_flash",
            "openrouter",
            "secret_store",
            "openrouter:byok",
            "openrouter",
            "deepseek/deepseek-v4-flash-0731",
            "deepseek_v4_flash_latency",
        ),
        (
            "openrouter_gemma4_26b_a4b",
            True,
            "gemma4",
            "openrouter",
            "secret_store",
            "openrouter:byok",
            "openrouter",
            "google/gemma-4-26b-a4b-it",
            "gemma4_26b_latency",
        ),
        (
            "cerebras_gemma4_31b",
            True,
            "gemma4_31b",
            "cerebras",
            "secret_store",
            "cerebras:byok",
            "cerebras",
            "gemma-4-31b",
            None,
        ),
    ],
)
def test_canonical_translation_fallback_branch_resolves_explicit_provider_route(
    selection_alias: str,
    fallback_enabled: bool,
    fallback_model: str,
    fallback_connection: str,
    expected_fallback_source: str,
    expected_fallback_reference: str | None,
    expected_fallback_provider: str,
    expected_fallback_model: str,
    expected_provider_routing: str | None,
) -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    original_fallback = runtime_resolution.TranslationFallbackRuntimeIntent(
        enabled=fallback_enabled,
        model=fallback_model,
        connection=fallback_connection,
    )

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
            connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED,
            translation_fallback=original_fallback,
        )
    )

    assert original_fallback == runtime_resolution.TranslationFallbackRuntimeIntent(
        enabled=fallback_enabled,
        model=fallback_model,
        connection=fallback_connection,
    )
    if selection_alias == "none":
        assert config.fallback is None
        return
    assert config.fallback is not None
    assert config.fallback.target.provider == expected_fallback_provider
    assert config.fallback.target.model == expected_fallback_model
    assert config.fallback.target.credential == _credential_assertion(
        resolved,
        expected_fallback_source,
        expected_fallback_reference,
    )
    assert config.fallback.target.provider_routing == expected_provider_routing


def test_openrouter_no_fallback_selected_has_no_fallback_credential() -> None:
    runtime_resolution = _runtime_resolution_module()
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        selection_alias="gemma4_byok",
    )

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
            connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            openrouter=openrouter_intent,
        )
    )

    assert config.fallback is None


def test_openrouter_china_fallback_resolves_deepseek_only_fallback_routing() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        selection_alias="gemma4_managed",
    )

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
            connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED,
            openrouter=openrouter_intent,
            translation_fallback=runtime_resolution.TranslationFallbackRuntimeIntent(
                enabled=True,
                model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
                connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED_CHINA,
            ),
        )
    )

    assert config.fallback is not None
    assert config.fallback.target.provider == "openrouter"
    assert config.fallback.target.model == "deepseek/deepseek-v4-flash-0731"
    assert config.fallback.target.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_MANAGED,
        required=True,
        reference="openrouter:managed_qq",
    )
    assert config.fallback.target.provider_routing == "deepseek_only"
    assert config.fallback.force_managed_wrapper is True


def test_openrouter_gemma_fallback_preserves_duplicate_target_and_adds_emergency() -> None:
    runtime_resolution = _runtime_resolution_module()
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        selection_alias="gemma4_byok",
    )

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
            connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            openrouter=openrouter_intent,
            translation_fallback=runtime_resolution.TranslationFallbackRuntimeIntent(
                enabled=True,
                model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
                connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            ),
        )
    )

    assert config.fallback is not None
    assert config.fallback.target.provider_routing == "gemma4_26b_latency"
    assert len(config.attempts) == 3
    assert config.attempts[1].target == config.fallback.target
    assert config.attempts[2].target.provider_routing == "gemma4_31b_cerebras_only"


def test_openrouter_deepseek_only_primary_keeps_fallback_and_emergency_schedule() -> None:
    runtime_resolution = _runtime_resolution_module()
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        selection_alias="deepseek_v4_flash_managed",
        provider_routing="deepseek_only",
    )

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
            connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED_CHINA,
            openrouter=openrouter_intent,
            translation_fallback=runtime_resolution.TranslationFallbackRuntimeIntent(
                enabled=True,
                model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
                connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED_CHINA,
            ),
        )
    )

    assert config.provider == "openrouter"
    assert config.model == "deepseek/deepseek-v4-flash-0731"
    assert config.provider_routing == "deepseek_only"
    assert config.fallback is not None
    assert config.fallback.target.provider_routing == "deepseek_only"
    assert len(config.attempts) == 3
    assert config.attempts[2].target.provider_routing == "gemma4_31b_cerebras_only"


def test_managed_china_resolves_explicit_qq_managed_credential_reference() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
            connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED_CHINA,
            openrouter=runtime_resolution.OpenRouterRuntimeIntent(
                selected_source=runtime_resolution.OPENROUTER_SOURCE_MANAGED,
            ),
        )
    )

    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_MANAGED,
        required=True,
        reference="openrouter:managed_qq",
    )
    assert config.provider_routing == "deepseek_only"


def test_standard_managed_resolves_standard_managed_credential_reference() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
            connection=runtime_resolution.TRANSLATION_CONNECTION_MANAGED,
            openrouter=runtime_resolution.OpenRouterRuntimeIntent(
                selected_source=runtime_resolution.OPENROUTER_SOURCE_MANAGED,
                managed_credential_kind=runtime_resolution.OPENROUTER_MANAGED_CREDENTIAL_QQ,
            ),
        )
    )

    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_MANAGED,
        required=True,
        reference="openrouter:managed",
    )


def test_openrouter_deepseek_byok_deepseek_only_preserves_routing_and_suppresses_fallback() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm="openrouter",
        model="deepseek/deepseek-v4-flash-0731",
        selected_source="byok",
        fallback_selection_alias="qwen35_flash",
        routing_mode="parasail_first",
        provider_routing="deepseek_only",
        broker_base_url="https://broker.fixture.test/v1",
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm="openrouter",
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        concurrency_limit=8,
    )

    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
        )
    )

    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH
    assert translation_intent.connection == runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER
    assert config.provider == "openrouter"
    assert config.model == "deepseek/deepseek-v4-flash-0731"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="openrouter:byok",
    )
    assert config.routing_mode == "latency"
    assert config.provider_routing == "deepseek_only"
    assert config.service_endpoint == "https://broker.fixture.test/v1"
    assert config.fallback is None


def test_openrouter_runtime_intent_normalizes_legacy_deepseek_model() -> None:
    runtime_resolution = _runtime_resolution_module()

    intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm="openrouter",
        model="deepseek/deepseek-v4-flash",
        selected_source="byok",
    )

    assert intent.model == "deepseek/deepseek-v4-flash-0731"
    assert intent.selection_alias == "deepseek_v4_flash_byok"


def test_legacy_current_openrouter_aliases_normalize_to_canonical_intent_and_resolve() -> None:
    runtime_resolution = _runtime_resolution_module()
    profiles = _profiles_module()
    resolved = _resolved_module()

    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        selection_alias=profiles.LEGACY_OPENROUTER_SELECTION_ALIAS_BYOK_GEMMA_4_26B_A4B_IT,
        selected_source=profiles.OPENROUTER_CREDENTIAL_SOURCE_MANAGED,
        fallback_selection_alias=(
            profiles.LEGACY_OPENROUTER_FALLBACK_SELECTION_ALIAS_GEMINI31_FLASH_LITE
        ),
    )

    assert openrouter_intent.model == profiles.OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT
    assert openrouter_intent.selected_source == profiles.OPENROUTER_CREDENTIAL_SOURCE_BYOK
    assert openrouter_intent.selection_alias == profiles.OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK
    assert not hasattr(openrouter_intent, "fallback_selection_alias")
    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
            connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            openrouter=openrouter_intent,
            concurrency_limit=7,
        )
    )

    assert config.provider == "openrouter"
    assert config.model == profiles.OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="openrouter:byok",
    )
    assert config.fallback is None
    assert config.concurrency_limit == 7


@pytest.mark.parametrize(
    (
        "legacy_alias",
        "selected_source",
        "expected_alias",
        "expected_source",
        "expected_model",
        "expected_credential_source",
        "expected_credential_reference",
    ),
    [
        (
            "openrouter:byok:google/gemma-4-26b-a4b-it",
            "managed",
            "gemma4_byok",
            "byok",
            "google/gemma-4-26b-a4b-it",
            "secret_store",
            "openrouter:byok",
        ),
        (
            "openrouter:managed:google/gemma-4-26b-a4b-it",
            "byok",
            "gemma4_managed",
            "managed",
            "google/gemma-4-26b-a4b-it",
            "managed",
            "openrouter:managed",
        ),
        (
            "openrouter:none:google/gemma-4-26b-a4b-it",
            "managed",
            "gemma4_managed",
            "managed",
            "google/gemma-4-26b-a4b-it",
            "managed",
            "openrouter:managed",
        ),
        (
            "openrouter:byok:qwen/qwen3.5-flash-02-23",
            "managed",
            "qwen35_flash_byok",
            "byok",
            "qwen/qwen3.5-flash-02-23",
            "secret_store",
            "openrouter:byok",
        ),
        (
            "openrouter:none:qwen/qwen3.5-flash-02-23",
            "managed",
            "qwen35_flash_managed",
            "managed",
            "qwen/qwen3.5-flash-02-23",
            "managed",
            "openrouter:managed",
        ),
    ],
)
def test_legacy_openrouter_selection_aliases_normalize_before_resolved_runtime(
    legacy_alias: str,
    selected_source: str,
    expected_alias: str,
    expected_source: str,
    expected_model: str,
    expected_credential_source: str,
    expected_credential_reference: str,
) -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm="openrouter",
        selected_source=selected_source,
        selection_alias=legacy_alias,
        fallback_selection_alias="none",
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm="openrouter",
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        concurrency_limit=5,
    )
    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
        )
    )

    assert openrouter_intent.selection_alias == expected_alias
    assert openrouter_intent.selected_source == expected_source
    assert openrouter_intent.model == expected_model
    assert config.provider == "openrouter"
    assert config.model == expected_model
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=expected_credential_source,
        required=True,
        reference=expected_credential_reference,
    )
    assert not hasattr(config, "selection_alias")
    assert not hasattr(config, "fallback_selection_alias")


@pytest.mark.parametrize("legacy_key", ["credential_source", "selected_credential_source"])
def test_old_openrouter_credential_source_keys_normalize_through_settings_to_resolved_dto(
    legacy_key: str,
) -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    from puripuly_heart.config.settings import AppSettings, from_dict, to_dict  # noqa: PLC0415

    raw_settings = to_dict(AppSettings())
    raw_settings.pop("translation", None)
    raw_settings["provider"]["llm"] = "openrouter"
    raw_settings["openrouter"]["llm_model"] = "google/gemma-4-26b-a4b-it"
    raw_settings["openrouter"].pop("selected_source", None)
    raw_settings["openrouter"][legacy_key] = "managed"
    raw_settings["openrouter"]["selection_alias"] = "openrouter:none:google/gemma-4-26b-a4b-it"
    raw_settings["openrouter"]["fallback_selection_alias"] = "none"

    settings = from_dict(raw_settings)
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm=settings.provider.llm.value,
        model=settings.openrouter.llm_model.value,
        selected_source=settings.openrouter.selected_source.value,
        selection_alias=(
            settings.openrouter.selection_alias.value
            if settings.openrouter.selection_alias is not None
            else None
        ),
        fallback_selection_alias=settings.openrouter.fallback_selection_alias.value,
        routing_mode=settings.openrouter.routing_mode.value,
        provider_routing=settings.openrouter.provider_routing.value,
        broker_base_url=settings.openrouter.broker_base_url,
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm=settings.provider.llm.value,
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        gemini_model=settings.gemini.llm_model.value,
        qwen_model=settings.qwen.llm_model.value,
        concurrency_limit=settings.llm.concurrency_limit,
    )
    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
        )
    )

    assert settings.openrouter.selected_source.value == "managed"
    assert settings.openrouter.selection_alias is not None
    assert settings.openrouter.selection_alias.value == "gemma4_26b_31b_managed"
    assert openrouter_intent.selected_source == "managed"
    assert openrouter_intent.selection_alias == "gemma4_26b_31b_managed"
    assert config.provider == "openrouter"
    assert config.model == "google/gemma-4-26b-a4b-it"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_MANAGED,
        required=True,
        reference="openrouter:managed",
    )
    assert not hasattr(config, legacy_key)
    assert not hasattr(config, "selection_alias")


def test_runtime_resolution_resolved_llm_path_has_no_legacy_alias_literals() -> None:
    runtime_resolution = _runtime_resolution_module()
    source = "\n".join(
        inspect.getsource(getattr(runtime_resolution, name))
        for name in (
            "resolve_llm_config",
            "_resolve_translation_target",
            "_resolved_openrouter_target",
        )
    )

    for legacy_literal in (
        "LEGACY_OPENROUTER",
        "LEGACY_PROFILE_BY_ALIAS",
        "gemini25_flash_lite",
        "gemini31_flash_lite",
        "openrouter:none:google/gemma-4-26b-a4b-it",
        "openrouter:managed:google/gemma-4-26b-a4b-it",
        "openrouter:byok:google/gemma-4-26b-a4b-it",
        "openrouter:none:qwen/qwen3.5-flash-02-23",
        "openrouter:byok:qwen/qwen3.5-flash-02-23",
    ):
        assert legacy_literal not in source


def test_current_and_legacy_setting_value_snapshots_convert_to_canonical_input_and_resolve() -> (
    None
):
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    from tests.config.settings_migration_fixtures import (  # noqa: PLC0415
        legacy_compatibility_settings_fixture,
        maximal_v24_settings_fixture,
    )

    for raw_settings in (
        maximal_v24_settings_fixture(),
        legacy_compatibility_settings_fixture(),
    ):
        raw_openrouter = raw_settings["openrouter"]
        raw_translation = raw_settings["translation"]
        openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
            model=raw_openrouter.get("llm_model"),
            selected_source=(
                raw_openrouter.get("selected_source")
                or raw_openrouter.get("credential_source")
                or raw_openrouter.get("selected_credential_source")
            ),
            selection_alias=raw_openrouter.get("selection_alias"),
            fallback_selection_alias=raw_openrouter.get("fallback_selection_alias"),
            routing_mode=raw_openrouter.get("routing_mode"),
            provider_routing=raw_openrouter.get("provider_routing"),
            broker_base_url=raw_openrouter.get("broker_base_url"),
        )
        translation_intent = runtime_resolution.normalize_translation_runtime_intent(
            model=raw_translation.get("model"),
            connection=raw_translation.get("connection"),
            concurrency_limit=raw_settings["llm"].get("concurrency_limit"),
        )

        config = runtime_resolution.resolve_llm_config(
            runtime_resolution.RuntimeResolutionInput(
                translation=translation_intent,
                openrouter=openrouter_intent,
                direct=runtime_resolution.DirectProviderRuntimeIntent(
                    local_llm_model=raw_settings["local_llm"]["model"],
                    local_llm_base_url=raw_settings["local_llm"]["base_url"],
                    local_llm_extra_body=raw_settings["local_llm"]["extra_body"],
                    qwen_region=raw_settings["qwen"]["region"],
                ),
            )
        )

        assert isinstance(config, resolved.ResolvedLLMConfig)
        assert config.provider in {
            "cerebras",
            "deepseek",
            "gemini",
            "local_llm",
            "openrouter",
            "qwen",
        }
        assert config.concurrency_limit == raw_settings["llm"]["concurrency_limit"]


def test_derive_runtime_from_openrouter_gemini_compatibility_values() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm="openrouter",
        model="google/gemini-3.1-flash-lite",
        selected_source="byok",
        selection_alias="gemini31_flash_lite_byok",
        fallback_selection_alias="none",
        routing_mode="latency",
        provider_routing="google_gemini_latency",
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm="openrouter",
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        concurrency_limit=6,
    )

    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
        )
    )

    assert openrouter_intent.selection_alias == "gemini31_flash_lite_byok"
    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_GEMINI_31_FLASH_LITE
    assert translation_intent.connection == runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER
    assert config.provider == "openrouter"
    assert config.model == "google/gemini-3.1-flash-lite"
    assert config.provider_routing == "google_gemini_latency"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="openrouter:byok",
    )


def test_derive_runtime_from_cerebras_compatibility_values() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()

    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm="cerebras",
        cerebras_model="gemma-4-31b",
        concurrency_limit=3,
    )
    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            direct=runtime_resolution.DirectProviderRuntimeIntent(cerebras_model="gemma-4-31b"),
        )
    )

    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_GEMMA4_31B
    assert translation_intent.connection == runtime_resolution.TRANSLATION_CONNECTION_CEREBRAS
    assert config.provider == "cerebras"
    assert config.model == "gemma-4-31b"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="cerebras:byok",
    )


def test_missing_translation_openrouter_compatibility_values_derive_exact_runtime_config() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    raw_settings = {
        "provider": {"llm": "openrouter"},
        "openrouter": {
            "llm_model": "google/gemma-4-26b-a4b-it",
            "selected_source": "managed",
            "selection_alias": "deepseek_v4_flash_byok",
            "fallback_selection_alias": "qwen35_flash",
            "routing_mode": "parasail_first",
            "provider_routing": "default",
            "broker_base_url": "https://broker.fixture.test/v1",
        },
        "gemini": {"llm_model": "gemini-3.1-flash-lite"},
        "qwen": {"llm_model": "qwen3.5-plus", "region": "beijing"},
        "deepseek": {"llm_model": "deepseek-v4-flash"},
        "local_llm": {
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "llama3.1:8b",
            "extra_body": {},
        },
        "llm": {"concurrency_limit": 4},
    }
    raw_openrouter = raw_settings["openrouter"]
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        model=raw_openrouter["llm_model"],
        selected_source=raw_openrouter["selected_source"],
        selection_alias=raw_openrouter["selection_alias"],
        fallback_selection_alias=raw_openrouter["fallback_selection_alias"],
        routing_mode=raw_openrouter["routing_mode"],
        provider_routing=raw_openrouter["provider_routing"],
        broker_base_url=raw_openrouter["broker_base_url"],
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm=raw_settings["provider"]["llm"],
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        gemini_model=raw_settings["gemini"]["llm_model"],
        qwen_model=raw_settings["qwen"]["llm_model"],
        concurrency_limit=raw_settings["llm"]["concurrency_limit"],
    )

    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
            direct=runtime_resolution.DirectProviderRuntimeIntent(
                qwen_region=raw_settings["qwen"]["region"],
                local_llm_model=raw_settings["local_llm"]["model"],
                local_llm_base_url=raw_settings["local_llm"]["base_url"],
                local_llm_extra_body=raw_settings["local_llm"]["extra_body"],
            ),
        )
    )

    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH
    assert translation_intent.connection == runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER
    assert openrouter_intent.model == "deepseek/deepseek-v4-flash-0731"
    assert openrouter_intent.selected_source == "byok"
    assert config.provider == "openrouter"
    assert config.model == "deepseek/deepseek-v4-flash-0731"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="openrouter:byok",
    )
    assert config.routing_mode == "latency"
    assert config.provider_routing == "default"
    assert config.base_url is None
    assert config.service_endpoint == "https://broker.fixture.test/v1"
    assert config.fallback is None
    assert config.concurrency_limit == 4


@pytest.mark.parametrize(
    ("selected_source", "expected_credential_source", "expected_credential_reference"),
    [
        ("byok", "secret_store", "openrouter:byok"),
        ("managed", "managed", "openrouter:managed"),
    ],
)
def test_openrouter_qwen_primary_compatibility_preserves_model_and_source(
    selected_source: str,
    expected_credential_source: str,
    expected_credential_reference: str,
) -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm="openrouter",
        model="qwen/qwen3.5-flash-02-23",
        selected_source=selected_source,
        fallback_selection_alias="none",
        routing_mode="parasail_first",
        provider_routing="default",
        broker_base_url="https://broker.fixture.test/v1",
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm="openrouter",
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        concurrency_limit=8,
    )

    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
        )
    )

    assert config.provider == "openrouter"
    assert config.model == "qwen/qwen3.5-flash-02-23"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=expected_credential_source,
        required=True,
        reference=expected_credential_reference,
    )
    assert config.routing_mode == "latency"
    assert config.provider_routing == "default"
    assert config.service_endpoint == "https://broker.fixture.test/v1"
    assert config.fallback is None
    assert config.concurrency_limit == 8


def test_openrouter_qwen_primary_deepseek_only_preserves_routing_and_suppresses_fallback() -> None:
    runtime_resolution = _runtime_resolution_module()
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm="openrouter",
        model="qwen/qwen3.5-flash-02-23",
        selected_source="byok",
        fallback_selection_alias="deepseek_v4_flash",
        routing_mode="parasail_first",
        provider_routing="deepseek_only",
        broker_base_url="https://broker.fixture.test/v1",
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm="openrouter",
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        concurrency_limit=8,
    )

    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
        )
    )

    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_OPENROUTER_QWEN_35_FLASH
    assert config.provider == "openrouter"
    assert config.model == "qwen/qwen3.5-flash-02-23"
    assert config.provider_routing == "deepseek_only"
    assert config.routing_mode == "latency"
    assert config.fallback is None


def test_missing_openrouter_source_defaults_to_byok_for_openrouter_provider() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    raw_settings = {
        "provider": {"llm": "openrouter"},
        "openrouter": {
            "llm_model": "google/gemma-4-26b-a4b-it",
            "fallback_selection_alias": "none",
            "routing_mode": "latency",
            "provider_routing": "default",
            "broker_base_url": "https://broker.fixture.test/v1",
        },
        "gemini": {"llm_model": "gemini-3.1-flash-lite"},
        "qwen": {"llm_model": "qwen3.5-plus", "region": "beijing"},
        "deepseek": {"llm_model": "deepseek-v4-flash"},
        "llm": {"concurrency_limit": 3},
    }
    raw_openrouter = raw_settings["openrouter"]
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        provider_llm=raw_settings["provider"]["llm"],
        model=raw_openrouter["llm_model"],
        fallback_selection_alias=raw_openrouter["fallback_selection_alias"],
        routing_mode=raw_openrouter["routing_mode"],
        provider_routing=raw_openrouter["provider_routing"],
        broker_base_url=raw_openrouter["broker_base_url"],
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm=raw_settings["provider"]["llm"],
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        gemini_model=raw_settings["gemini"]["llm_model"],
        qwen_model=raw_settings["qwen"]["llm_model"],
        concurrency_limit=raw_settings["llm"]["concurrency_limit"],
    )

    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
        )
    )

    assert openrouter_intent.model == "google/gemma-4-26b-a4b-it"
    assert openrouter_intent.selected_source == "byok"
    assert openrouter_intent.selection_alias == "gemma4_byok"
    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_GEMMA4
    assert translation_intent.connection == runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER
    assert config.provider == "openrouter"
    assert config.model == "google/gemma-4-26b-a4b-it"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="openrouter:byok",
    )
    assert config.routing_mode == "latency"
    assert config.provider_routing == "gemma4_26b_latency"
    assert config.service_endpoint == "https://broker.fixture.test/v1"
    assert config.fallback is None
    assert config.concurrency_limit == 3


@pytest.mark.parametrize(
    "source_kwargs",
    [
        {},
        {"openrouter_selected_source": None},
    ],
)
def test_derive_translation_compatibility_defaults_missing_openrouter_source_to_byok(
    source_kwargs: dict[str, object],
) -> None:
    runtime_resolution = _runtime_resolution_module()

    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm="openrouter",
        openrouter_model="google/gemma-4-26b-a4b-it",
        openrouter_provider_routing="default",
        concurrency_limit=3,
        **source_kwargs,
    )

    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_GEMMA4
    assert translation_intent.connection == runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER
    assert translation_intent.concurrency_limit == 3


def test_missing_translation_direct_provider_compatibility_values_derive_exact_config() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    raw_settings = {
        "provider": {"llm": "deepseek"},
        "openrouter": {
            "llm_model": "google/gemma-4-26b-a4b-it",
            "selected_source": "managed",
            "selection_alias": "gemma4_managed",
            "fallback_selection_alias": "qwen35_flash",
            "routing_mode": "latency",
            "provider_routing": "default",
            "broker_base_url": "https://broker.fixture.test/v1",
        },
        "gemini": {"llm_model": "gemini-3.1-flash-lite"},
        "qwen": {"llm_model": "qwen3.5-plus", "region": "singapore"},
        "deepseek": {"llm_model": "deepseek-v4-flash"},
        "llm": {"concurrency_limit": 6},
    }
    raw_openrouter = raw_settings["openrouter"]
    openrouter_intent = runtime_resolution.normalize_openrouter_runtime_intent(
        model=raw_openrouter["llm_model"],
        selected_source=raw_openrouter["selected_source"],
        selection_alias=raw_openrouter["selection_alias"],
        fallback_selection_alias=raw_openrouter["fallback_selection_alias"],
        routing_mode=raw_openrouter["routing_mode"],
        provider_routing=raw_openrouter["provider_routing"],
        broker_base_url=raw_openrouter["broker_base_url"],
    )
    translation_intent = runtime_resolution.derive_translation_runtime_intent_from_compatibility(
        provider_llm=raw_settings["provider"]["llm"],
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        gemini_model=raw_settings["gemini"]["llm_model"],
        qwen_model=raw_settings["qwen"]["llm_model"],
        concurrency_limit=raw_settings["llm"]["concurrency_limit"],
    )

    config = runtime_resolution.resolve_llm_config(
        runtime_resolution.RuntimeResolutionInput(
            translation=translation_intent,
            openrouter=openrouter_intent,
            direct=runtime_resolution.DirectProviderRuntimeIntent(
                deepseek_v4_flash_model=raw_settings["deepseek"]["llm_model"],
                qwen_region=raw_settings["qwen"]["region"],
            ),
        )
    )

    assert translation_intent.model == runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH
    assert translation_intent.connection == runtime_resolution.TRANSLATION_CONNECTION_OFFICIAL_BYOK
    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="deepseek:byok",
    )
    assert config.base_url is None
    assert config.service_endpoint is None
    assert config.region is None
    assert config.routing_mode is None
    assert config.provider_routing is None
    assert config.fallback is None
    assert config.concurrency_limit == 6


def test_resolved_output_uses_lookup_references_not_raw_secret_values() -> None:
    runtime_resolution = _runtime_resolution_module()
    resolved = _resolved_module()
    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            runtime_resolution,
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4,
            connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            openrouter=runtime_resolution.OpenRouterRuntimeIntent(
                selected_source=runtime_resolution.OPENROUTER_SOURCE_BYOK,
            ),
            translation_fallback=runtime_resolution.TranslationFallbackRuntimeIntent(
                enabled=True,
                model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
                connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            ),
        )
    )

    assert config.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="openrouter:byok",
    )
    assert config.fallback is not None
    assert config.fallback.target.credential == resolved.ResolvedCredentialRequirement(
        source=resolved.CREDENTIAL_SOURCE_SECRET_STORE,
        required=True,
        reference="openrouter:byok",
    )
    assert config.credential.reference is not None
    assert "sk-" not in config.credential.reference
    assert "secret" not in config.credential.reference
    assert config.fallback.target.credential.reference is not None
    assert "sk-" not in config.fallback.target.credential.reference
    assert "secret" not in config.fallback.target.credential.reference


def test_runtime_fallback_rejects_custom_http() -> None:
    runtime_resolution = _runtime_resolution_module()

    with pytest.raises(ValueError, match="cannot be used as fallback"):
        runtime_resolution.TranslationFallbackRuntimeIntent(
            enabled=True,
            model=runtime_resolution.TRANSLATION_MODEL_CUSTOM_HTTP,
            connection=runtime_resolution.TRANSLATION_CONNECTION_CUSTOM_HTTP,
        )
