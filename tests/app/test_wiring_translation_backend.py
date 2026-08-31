from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import get_type_hints

import pytest

from puripuly_heart.app.wiring import wiring_translation_backend as wiring_module
from puripuly_heart.app.wiring.root import (
    create_llm_provider as create_exported_llm_provider,
)
from puripuly_heart.app.wiring.root import (
    create_llm_provider_from_resolved_config as create_exported_llm_provider_from_resolved_config,
)
from puripuly_heart.app.wiring.wiring_llm_factory import (
    create_llm_provider,
    llm_factory_extras_from_vnext,
    runtime_resolution_input_from_vnext,
)
from puripuly_heart.app.wiring.wiring_provider_runtime import compose_provider_runtime
from puripuly_heart.app.wiring.wiring_provider_runtime_policy import build_llm_provider_signature
from puripuly_heart.app.wiring.wiring_runtime_pipeline import compose_runtime_pipeline
from puripuly_heart.app.wiring.wiring_translation_backend import create_translation_backend
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.core.orchestrator.ports import TranslationRuntimeLoggingPort
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.translation_backend import LlmTranslationBackend
from puripuly_heart.providers.extensions.http_extension_backend import (
    HttpExtensionTranslationBackend,
)


def _registry(tmp_path: Path) -> HttpExtensionRegistry:
    (tmp_path / "libretranslate.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "libretranslate",
                "name": "LibreTranslate",
                "url": "http://127.0.0.1:1/translate",
                "request": {
                    "body": {
                        "type": "json",
                        "value": {"q": "{{text}}"},
                    }
                },
                "response": {"type": "text"},
            }
        ),
        encoding="utf-8",
    )
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    return registry


def _custom_settings() -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model="custom_http",
                connection="custom_http",
                http_extension_id="libretranslate",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_custom_http_factory_creates_only_the_extension_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    settings = _custom_settings()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(settings.intent.translation, concurrency_limit=2),
        ),
    )
    monkeypatch.setattr(
        wiring_module,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail("Custom HTTP must not create an LLM provider"),
    )

    backend = create_translation_backend(
        translation_model=settings.intent.translation.model,
        secrets=InMemorySecretStore(),
        http_extensions=registry,
        http_extension_id=settings.intent.translation.http_extension_id,
        concurrency_limit=settings.intent.translation.concurrency_limit,
        managed_release_service=pytest.fail,
    )

    assert isinstance(backend, HttpExtensionTranslationBackend)
    assert backend.concurrency_limit == 2
    await backend.close()


def test_custom_http_factory_rejects_missing_selected_extension(tmp_path: Path) -> None:
    settings = _custom_settings()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(settings.intent.translation, http_extension_id="missing"),
        ),
    )

    with pytest.raises(ValueError, match="selected HTTP extension is unavailable"):
        create_translation_backend(
            translation_model=settings.intent.translation.model,
            secrets=InMemorySecretStore(),
            http_extensions=_registry(tmp_path),
            http_extension_id=settings.intent.translation.http_extension_id,
        )


def test_llm_factory_path_remains_delegated(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettingsVNext()
    expected = object()
    calls: list[object] = []
    runtime_input = runtime_resolution_input_from_vnext(settings)

    def create_llm(runtime_input_value: object, **_kwargs: object) -> object:
        calls.append(runtime_input_value)
        return expected

    monkeypatch.setattr(wiring_module, "create_llm_provider", create_llm)

    result = create_translation_backend(
        translation_model=settings.intent.translation.model,
        secrets=InMemorySecretStore(),
        http_extensions=HttpExtensionRegistry(Path("unused")),
        runtime_input=runtime_input,
        extras=llm_factory_extras_from_vnext(settings),
    )

    assert isinstance(result, LlmTranslationBackend)
    assert result.provider is expected
    assert calls == [runtime_input]


def test_provider_observation_capability_is_typed_through_production_composition() -> None:
    provider_observation = ProviderObservationPort | None

    assert get_type_hints(create_llm_provider)["runtime_logging"] == provider_observation
    assert get_type_hints(create_exported_llm_provider)["runtime_logging"] == provider_observation
    assert (
        get_type_hints(create_exported_llm_provider_from_resolved_config)["runtime_logging"]
        == provider_observation
    )
    assert get_type_hints(create_translation_backend)["runtime_logging"] == provider_observation
    assert get_type_hints(compose_provider_runtime)["runtime_logging"] is ProviderObservationPort
    assert (
        get_type_hints(compose_runtime_pipeline)["runtime_logging"] is TranslationRuntimeLoggingPort
    )


def test_custom_http_definition_fingerprint_rebuilds_runtime_signature(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    settings = _custom_settings()

    before = build_llm_provider_signature(settings, http_extensions=registry)
    extension_path = tmp_path / "libretranslate.json"
    changed = json.loads(extension_path.read_text(encoding="utf-8"))
    changed["description"] = "Changed local definition"
    extension_path.write_text(json.dumps(changed), encoding="utf-8")
    registry.reload()
    after = build_llm_provider_signature(settings, http_extensions=registry)

    assert before != after
    assert "local-secret" not in repr(after)
