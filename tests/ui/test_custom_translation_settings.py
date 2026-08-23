from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.services.http_extension_registry import (
    HttpExtensionRegistryService,
)
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    QwenLLMModel,
    QwenRegion,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
)
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
    monkeypatch.setattr(settings_view, "create_secret_store", lambda *_args, **_kwargs: store)
    return settings_view.SettingsView(
        http_extension_registry=HttpExtensionRegistryService(
            registry,
            directory_opener,
        )
    )


def _custom_settings() -> AppSettings:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.region = QwenRegion.SINGAPORE
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS
    settings.translation = TranslationSettings(
        model=TranslationModel.QWEN_35_PLUS,
        connection=TranslationConnection.OFFICIAL_BYOK,
        connection_history={
            TranslationModel.QWEN_35_PLUS.value: TranslationConnection.OFFICIAL_BYOK,
        },
        fallback=TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMMA4,
            connection=TranslationConnection.OPENROUTER,
        ),
    )
    return settings


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

    fallback = settings.translation.fallback
    view._on_llm_selected(TranslationModel.CUSTOM_HTTP.value)
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.model is TranslationModel.CUSTOM_HTTP
    assert pending.translation.connection is TranslationConnection.CUSTOM_HTTP
    assert pending.translation.previous_llm_model is TranslationModel.QWEN_35_PLUS
    assert pending.provider.llm is LLMProviderName.QWEN
    assert pending.translation.fallback == fallback
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
    assert pending.translation.http_extension_id == "demo"
    assert set(view._http_extension_secret_fields) == {"api_key"}
    assert view._http_extension_secret_fields["api_key"].value == ""
    assert "saved-secret" not in repr(view._http_extension_secret_fields["api_key"])
    assert not hasattr(view, "_http_extension_request_editor")

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.model is TranslationModel.QWEN_35_PLUS
    assert pending.translation.connection is TranslationConnection.OFFICIAL_BYOK
    assert pending.translation.previous_llm_model is None
    assert pending.translation.fallback == fallback
    assert pending.provider.llm is LLMProviderName.QWEN


def test_custom_http_credentials_use_namespaced_secret_callback_and_reload_isolated_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_extension(tmp_path)
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    store = SecretStore({"http_extension.demo.api_key": "saved-secret"})
    view = _view(monkeypatch, registry, store)
    settings = _custom_settings()
    settings.translation.model = TranslationModel.CUSTOM_HTTP
    settings.translation.connection = TranslationConnection.CUSTOM_HTTP
    settings.translation.http_extension_id = "demo"
    callbacks: list[tuple[str, str]] = []
    notices: list[str] = []
    view.on_provider_secret_change = lambda key, value: (callbacks.append((key, value)) or True)
    view.show_snackbar = lambda message, _color: notices.append(message)
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    field = view._http_extension_secret_fields["api_key"]
    view._on_http_extension_secret_blur("api_key")
    assert callbacks == []
    field.value = "new-secret"
    view._on_http_extension_secret_blur("api_key")
    field.value = ""
    field.on_change(None)
    view._on_http_extension_secret_blur("api_key")

    assert callbacks == [
        ("http_extension.demo.api_key", "new-secret"),
        ("http_extension.demo.api_key", ""),
    ]
    assert "new-secret" not in repr(settings)

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    changed: list[bool] = []
    view.on_providers_changed = lambda: changed.append(True)
    view._on_http_extension_reload(None)

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
    settings = _custom_settings()
    settings.translation.model = TranslationModel.CUSTOM_HTTP
    settings.translation.connection = TranslationConnection.CUSTOM_HTTP
    settings.translation.http_extension_id = "demo"
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    draft = view._ensure_provider_settings_draft()
    draft.translation.model = TranslationModel.QWEN_35_PLUS
    changed: list[bool] = []
    view.on_providers_changed = lambda: changed.append(True)

    extension_path = tmp_path / "demo.json"
    extension_data = json.loads(extension_path.read_text(encoding="utf-8"))
    extension_data["description"] = "Changed local demo translator"
    extension_path.write_text(json.dumps(extension_data), encoding="utf-8")

    view._on_http_extension_reload(None)

    assert changed == [True]
    assert view.consume_http_extension_runtime_reload() is True
    assert view._settings.translation.model is TranslationModel.CUSTOM_HTTP
    assert view._provider_settings_draft.translation.model is TranslationModel.QWEN_35_PLUS


def test_custom_http_card_surfaces_missing_selected_extension_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_extension(tmp_path)
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    store = SecretStore()
    view = _view(monkeypatch, registry, store)
    settings = _custom_settings()
    settings.translation.model = TranslationModel.CUSTOM_HTTP
    settings.translation.connection = TranslationConnection.CUSTOM_HTTP
    settings.translation.http_extension_id = "demo"
    changed: list[bool] = []
    view.on_providers_changed = lambda: changed.append(True)
    view.load_from_settings(settings, config_path=tmp_path / "settings.json")

    (tmp_path / "demo.json").unlink()
    view._on_http_extension_reload(None)

    assert view._http_extension_text.content.value == settings_view.t(
        "settings.http_extension.none"
    )
    assert view._http_extension_selected_id == "demo"
    assert view.build_provider_apply_settings().translation.http_extension_id == "demo"
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
    settings = _custom_settings()
    settings.translation.model = TranslationModel.CUSTOM_HTTP
    settings.translation.connection = TranslationConnection.CUSTOM_HTTP
    settings.translation.http_extension_id = "demo"
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
