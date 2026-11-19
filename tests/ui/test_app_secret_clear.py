from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui.app import TranslatorApp
from tests.helpers.ui_application import compose_test_ui_application_boundary


class DummyDashboard:
    def __init__(self) -> None:
        self.translation_calls: list[tuple[bool, bool]] = []
        self.stt_calls: list[tuple[bool, bool]] = []
        self.peer_calls: list[tuple[bool, bool]] = []

    def set_translation_needs_key(self, value: bool, *, update_ui: bool = True) -> None:
        self.translation_calls.append((value, update_ui))

    def set_stt_needs_key(self, value: bool, *, update_ui: bool = True) -> None:
        self.stt_calls.append((value, update_ui))

    def set_peer_needs_key(self, value: bool, *, update_ui: bool = True) -> None:
        self.peer_calls.append((value, update_ui))


def _make_app_with_verified_state(save_settings=None) -> TranslatorApp:
    app = TranslatorApp.__new__(TranslatorApp)
    save = save_settings or (lambda: None)
    controller = SimpleNamespace(
        settings=SimpleNamespace(
            api_key_verified=SimpleNamespace(
                deepgram=True,
                soniox=True,
                gemini=True,
                openrouter=True,
                deepseek=True,
                alibaba_beijing=True,
                alibaba_singapore=True,
            )
        ),
        config_path=Path("settings.json"),
        _save_settings=save,
        persist_settings=save,
    )

    def clear_provider_verification(provider: str) -> None:
        setattr(controller.settings.api_key_verified, provider, False)
        save()

    controller.clear_provider_verification = clear_provider_verification
    app.controller = controller
    app._ui_application = compose_test_ui_application_boundary(controller)
    app.view_dashboard = DummyDashboard()
    return app


@pytest.mark.parametrize(
    ("secret_name", "provider", "refresh_stt"),
    [
        ("alibaba_api_key_beijing", "alibaba_beijing", True),
        ("alibaba_api_key_singapore", "alibaba_singapore", True),
        ("openrouter_api_key", "openrouter", False),
        ("deepseek_api_key", "deepseek", False),
    ],
)
def test_secret_clear_resets_provider_verification_and_dashboard(
    secret_name: str, provider: str, refresh_stt: bool
) -> None:
    saves: list[bool] = []
    app = _make_app_with_verified_state(save_settings=lambda: saves.append(True))

    app._on_secret_cleared(secret_name)

    assert getattr(app.controller.settings.api_key_verified, provider) is False
    assert app.view_dashboard.translation_calls == [(True, False)]
    assert app.view_dashboard.stt_calls == ([(True, False)] if refresh_stt else [])
    assert len(saves) == 1


def test_on_secret_cleared_ignores_unknown_key() -> None:
    saves: list[bool] = []
    app = _make_app_with_verified_state(save_settings=lambda: saves.append(True))

    app._on_secret_cleared("unknown_key")

    assert app.controller.settings.api_key_verified.alibaba_beijing is True
    assert app.controller.settings.api_key_verified.alibaba_singapore is True
    assert app.controller.settings.api_key_verified.openrouter is True
    assert app.view_dashboard.translation_calls == []
    assert app.view_dashboard.stt_calls == []
    assert saves == []


@pytest.mark.asyncio
async def test_on_provider_secret_change_routes_atomic_invalidation_before_verification() -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    calls: list[tuple[str, str]] = []

    async def persist(key: str, value: str) -> bool:
        calls.append((key, value))
        return True

    app.controller = SimpleNamespace(persist_provider_secret_change=persist)
    app._ui_application = compose_test_ui_application_boundary(app.controller)
    app.view_dashboard = DummyDashboard()

    result = await app._on_provider_secret_change(
        "openrouter_api_key",
        "new-secret",
    )

    assert result is True
    assert calls == [("openrouter_api_key", "new-secret")]
    assert app.view_dashboard.translation_calls == [(True, False)]
    assert app.view_dashboard.stt_calls == []


@pytest.mark.asyncio
async def test_on_provider_secret_change_skips_dashboard_update_when_transaction_fails() -> None:
    app = TranslatorApp.__new__(TranslatorApp)

    async def persist(_key: str, _value: str) -> bool:
        return False

    app.controller = SimpleNamespace(persist_provider_secret_change=persist)
    app._ui_application = compose_test_ui_application_boundary(app.controller)
    app.view_dashboard = DummyDashboard()

    result = await app._on_provider_secret_change(
        "deepgram_api_key",
        "new-secret",
    )

    assert result is False
    assert app.view_dashboard.translation_calls == []
    assert app.view_dashboard.stt_calls == []


@pytest.mark.asyncio
async def test_on_provider_secret_change_flags_stt_and_translation_for_alibaba_key() -> None:
    app = TranslatorApp.__new__(TranslatorApp)

    async def persist(_key: str, _value: str) -> bool:
        return True

    app.controller = SimpleNamespace(persist_provider_secret_change=persist)
    app._ui_application = compose_test_ui_application_boundary(app.controller)
    app.view_dashboard = DummyDashboard()

    result = await app._on_provider_secret_change(
        "alibaba_api_key_beijing",
        "new-secret",
    )

    assert result is True
    assert app.view_dashboard.stt_calls == [(True, False)]
    assert app.view_dashboard.translation_calls == [(True, False)]
