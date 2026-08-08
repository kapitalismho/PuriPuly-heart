from __future__ import annotations

import json
from pathlib import Path

import pytest

from puripuly_heart.app.wiring import wiring_translation_backend as wiring_module
from puripuly_heart.app.wiring.wiring_provider_runtime_policy import build_llm_provider_signature
from puripuly_heart.app.wiring.wiring_translation_backend import create_translation_backend
from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    TranslationSettings,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.translation_extensions import (
    HttpExtensionTranslationBackend,
    TranslationExtensionRegistry,
)


def _registry(tmp_path: Path) -> TranslationExtensionRegistry:
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
    registry = TranslationExtensionRegistry(tmp_path)
    registry.reload()
    return registry


def _custom_settings() -> AppSettings:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.CUSTOM_HTTP,
        connection=TranslationConnection.CUSTOM_HTTP,
        extension_id="libretranslate",
    )
    return settings


@pytest.mark.asyncio
async def test_custom_http_factory_creates_only_the_extension_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    settings = _custom_settings()
    settings.llm.concurrency_limit = 2
    monkeypatch.setattr(
        wiring_module,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail("Custom HTTP must not create an LLM provider"),
    )

    backend = create_translation_backend(
        settings,
        secrets=InMemorySecretStore(),
        translation_extensions=registry,
        managed_release_service=pytest.fail,
    )

    assert isinstance(backend, HttpExtensionTranslationBackend)
    assert backend.concurrency_limit == 2
    await backend.close()


def test_custom_http_factory_rejects_missing_selected_extension(tmp_path: Path) -> None:
    settings = _custom_settings()
    settings.translation.extension_id = "missing"

    with pytest.raises(ValueError, match="selected translation extension is unavailable"):
        create_translation_backend(
            settings,
            secrets=InMemorySecretStore(),
            translation_extensions=_registry(tmp_path),
        )


def test_llm_factory_path_remains_delegated(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings()
    expected = object()
    calls: list[object] = []

    def create_llm(settings_value: object, **_kwargs: object) -> object:
        calls.append(settings_value)
        return expected

    monkeypatch.setattr(wiring_module, "create_llm_provider", create_llm)

    result = create_translation_backend(
        settings,
        secrets=InMemorySecretStore(),
        translation_extensions=TranslationExtensionRegistry(Path("unused")),
    )

    assert result is expected
    assert calls == [settings]


def test_custom_http_definition_fingerprint_rebuilds_runtime_signature(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    settings = _custom_settings()

    before = build_llm_provider_signature(settings, translation_extensions=registry)
    extension_path = tmp_path / "libretranslate.json"
    changed = json.loads(extension_path.read_text(encoding="utf-8"))
    changed["description"] = "Changed local definition"
    extension_path.write_text(json.dumps(changed), encoding="utf-8")
    registry.reload()
    after = build_llm_provider_signature(settings, translation_extensions=registry)

    assert before != after
    assert "local-secret" not in repr(after)
