from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.services.http_extension_registry import (
    HttpExtensionRegistryService,
)
from puripuly_heart.app.services.settings_secrets import SettingsSecretsOwner
from puripuly_heart.config.provider_values import QwenLLMModel
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    TranslationFallbackIntent,
)
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.ui.views import settings as settings_view


class SecretStore:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


def _write_extension(directory: Path, *, http_extension_id: str = "demo") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{http_extension_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": http_extension_id,
                "name": "Demo translator",
                "description": "Local demo translator",
                "url": "http://127.0.0.1:1/translate",
                "request": {
                    "body": {
                        "type": "json",
                        "value": {
                            "q": "{{text}}",
                            "api_key": "{{secret:api_key}}",
                        },
                    }
                },
                "response": {"type": "text"},
                "secrets": [{"id": "api_key", "label": "API Key"}],
            }
        ),
        encoding="utf-8",
    )


def _view(
    monkeypatch: pytest.MonkeyPatch,
    registry: HttpExtensionRegistry,
    store: SecretStore,
    directory_opener: object | None = None,
) -> settings_view.SettingsView:
    monkeypatch.setattr(settings_view.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "_refresh_microphones", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "update", lambda self: None)
    view = settings_view.SettingsView(
        http_extension_registry=HttpExtensionRegistryService(
            registry,
            directory_opener,
        )
    )
    view._settings_secrets = SettingsSecretsOwner(secret_store_factory=lambda: store)
    return view


def _custom_settings() -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model=TranslationModel.QWEN_38_FLASH.value,
                connection="official_byok",
                connection_history={TranslationModel.QWEN_38_FLASH.value: "official_byok"},
                qwen=replace(
                    settings.intent.translation.qwen,
                    region="singapore",
                    llm_model=QwenLLMModel.QWEN_38_FLASH.value,
                ),
                fallback=TranslationFallbackIntent(selection_alias="openrouter_gemma4_26b_a4b"),
            ),
        ),
    )


def _custom_http_settings(*, http_extension_id: str = "demo") -> AppSettingsVNext:
    settings = _custom_settings()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model="custom_http",
                connection="custom_http",
                http_extension_id=http_extension_id,
            ),
        ),
    )


def test_custom_http_card_replaces_llm_detail_surface_and_preserves_switch_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_extension(tmp_path)
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    store = SecretStore({"http_extension.demo.api_key": "saved-secret"})
    view = _view(monkeypatch, registry, store)
    settings = _custom_settings()
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    fallback = settings.intent.translation.fallback
    view._on_llm_selected(TranslationModel.CUSTOM_HTTP.value)
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.intent.translation.model == TranslationModel.CUSTOM_HTTP.value
    assert pending.intent.translation.connection == TranslationConnection.CUSTOM_HTTP.value
    assert pending.intent.translation.previous_llm_model == TranslationModel.QWEN_38_FLASH.value
    assert pending.intent.translation.fallback.enabled == fallback.enabled
    assert pending.intent.translation.fallback.model == fallback.model
    assert pending.intent.translation.fallback.connection == fallback.connection
    assert view._http_extension_row.visible is True
    assert view._http_extension_host.visible is True
    assert view._translation_connection_row.visible is False
    assert view._openrouter_fallback_card.visible is False
    assert view._local_llm_connection_card.visible is False
    assert view._google_key.visible is False
    assert view._openrouter_key.visible is False
    assert view._deepseek_key.visible is False
    assert view._cerebras_key.visible is False

    view._on_http_extension_selected("demo")
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.intent.translation.http_extension_id == "demo"
    assert set(view._http_extension_secret_fields) == {"api_key"}
    assert view._http_extension_secret_fields["api_key"].value == "saved-secret"
    assert view._http_extension_secret_fields["api_key"].password is True
    assert not hasattr(view, "_http_extension_request_editor")

    view._on_llm_selected(TranslationModel.QWEN_38_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.intent.translation.model == TranslationModel.QWEN_38_FLASH.value
    assert pending.intent.translation.connection == TranslationConnection.OFFICIAL_BYOK.value
    assert pending.intent.translation.previous_llm_model is None
    assert pending.intent.translation.fallback.enabled == fallback.enabled
    assert pending.intent.translation.fallback.model == fallback.model
    assert pending.intent.translation.fallback.connection == fallback.connection


def test_custom_http_credentials_use_namespaced_secret_callback_and_reload_isolated_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_extension(tmp_path)
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    store = SecretStore({"http_extension.demo.api_key": "saved-secret"})
    view = _view(monkeypatch, registry, store)
    settings = _custom_http_settings()
    callbacks: list[tuple[str, str]] = []
    notices: list[str] = []

    def save_secret(key: str, value: str) -> bool:
        callbacks.append((key, value))
        if value:
            store.set(key, value)
        else:
            store.delete(key)
        return True

    view.on_provider_secret_change = save_secret
    view.show_snackbar = lambda message, _color: notices.append(message)
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    field = view._http_extension_secret_fields["api_key"]
    assert field.value == "saved-secret"
    view._on_http_extension_secret_blur("api_key")
    assert callbacks == []
    field.value = "new-secret"
    field.on_change(None)
    view._on_http_extension_secret_blur("api_key")

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    changed: list[bool] = []
    view.on_providers_changed = lambda: changed.append(True)
    view._on_http_extension_reload(None)

    field = view._http_extension_secret_fields["api_key"]
    assert field.value == "new-secret"
    view._on_http_extension_secret_blur("api_key")
    assert callbacks == [("http_extension.demo.api_key", "new-secret")]
    field.value = ""
    field.on_change(None)
    view._on_http_extension_secret_blur("api_key")

    assert callbacks == [
        ("http_extension.demo.api_key", "new-secret"),
        ("http_extension.demo.api_key", ""),
    ]
    assert "new-secret" not in repr(settings)
    assert [loaded.definition.id for loaded in view._http_extension_snapshot.extensions] == ["demo"]
    assert len(view._http_extension_snapshot.errors) == 1
    assert changed == []
    assert notices


def test_custom_http_reload_uses_active_engine_with_unsaved_llm_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_extension(tmp_path)
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    view = _view(monkeypatch, registry, SecretStore())
    settings = _custom_http_settings()
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    draft = view._ensure_provider_settings_draft()
    view._provider_draft = replace(
        draft,
        translation=replace(draft.translation, model=TranslationModel.QWEN_38_FLASH),
    )
    changed: list[bool] = []
    view.on_providers_changed = lambda: changed.append(True)

    extension_path = tmp_path / "demo.json"
    extension_data = json.loads(extension_path.read_text(encoding="utf-8"))
    extension_data["description"] = "Changed local demo translator"
    extension_path.write_text(json.dumps(extension_data), encoding="utf-8")

    view._on_http_extension_reload(None)

    assert changed == [True]
    assert view.consume_http_extension_runtime_reload() is True
    assert view._provider_snapshot.translation.model is TranslationModel.CUSTOM_HTTP
    assert view._provider_draft.translation.model is TranslationModel.QWEN_38_FLASH


def test_custom_http_card_surfaces_missing_selected_extension_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_extension(tmp_path)
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    store = SecretStore()
    view = _view(monkeypatch, registry, store)
    settings = _custom_http_settings()
    changed: list[bool] = []
    view.on_providers_changed = lambda: changed.append(True)
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    (tmp_path / "demo.json").unlink()
    view._on_http_extension_reload(None)

    assert view._http_extension_text.content.value == settings_view.t(
        "settings.http_extension.none"
    )
    assert view._http_extension_selected_id == "demo"
    assert view.build_provider_apply_settings().intent.translation.http_extension_id == "demo"
    assert len(view._http_extension_credentials.controls) == 0
    assert changed == [True]


def test_custom_http_form_shows_only_when_extension_declares_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_extension(tmp_path)
    extension_path = tmp_path / "demo.json"
    extension_data = json.loads(extension_path.read_text(encoding="utf-8"))
    extension_data["request"]["body"]["value"].pop("api_key")
    extension_data["secrets"] = []
    extension_path.write_text(json.dumps(extension_data), encoding="utf-8")
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    view = _view(monkeypatch, registry, SecretStore())
    settings = _custom_http_settings()
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    assert view._http_extension_credentials in view._api_keys_column.controls
    assert view._http_extension_credentials.visible is True
    assert view._http_extension_secret_fields == {}
    assert len(view._http_extension_credentials.controls) == 0


def test_custom_http_open_folder_uses_resolved_registry_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = HttpExtensionRegistry(tmp_path / "http_extensions")
    registry.reload()
    calls: list[Path] = []
    view = _view(
        monkeypatch,
        registry,
        SecretStore(),
        SimpleNamespace(open=lambda directory: calls.append(directory)),
    )

    view._on_http_extension_open_folder(None)

    assert (tmp_path / "http_extensions").is_dir()
    assert calls == [tmp_path / "http_extensions"]
