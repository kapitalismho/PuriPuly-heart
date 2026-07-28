from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.config.settings import (
    AppSettings,
    QwenLLMModel,
    STTProviderName,
)
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


class DummyDashboard:
    def __init__(self) -> None:
        self.translation_needs_key: bool | None = None
        self.translation_enabled: bool | None = None
        self.stt_needs_key: bool | None = None
        self.stt_enabled: bool | None = None
        self.local_stt_notice_status: str | None = None
        self.local_stt_notice_percent: int | None = None
        self.local_stt_notice_model_id: str | None = None

    def set_translation_needs_key(self, value: bool) -> None:
        self.translation_needs_key = value

    def set_translation_enabled(self, value: bool) -> None:
        self.translation_enabled = value

    def set_stt_needs_key(self, value: bool) -> None:
        self.stt_needs_key = value

    def set_stt_enabled(self, value: bool) -> None:
        self.stt_enabled = value

    def set_local_stt_notice(self, status: str | None, percent: int | None = None) -> None:
        self.local_stt_notice_status = status
        self.local_stt_notice_percent = percent

    def set_local_stt_notice_model(self, model_id: str | None) -> None:
        self.local_stt_notice_model_id = model_id


class DummyOutputRuntime:
    def __init__(self) -> None:
        self.started_bridges: list[object] = []
        self.bridge_tasks: list[asyncio.Task[object]] = []

    def start_ui_event_bridge(self, bridge: object) -> asyncio.Task[object]:
        self.started_bridges.append(bridge)
        task = asyncio.create_task(bridge.run())  # type: ignore[attr-defined]
        self.bridge_tasks.append(task)
        return task


class DummyHub:
    def __init__(self, *, llm: object | None = object(), stt: object | None = object()) -> None:
        self.llm = llm
        self.stt = stt
        self.translation_enabled = True
        self.ui_events: asyncio.Queue[object] = asyncio.Queue()
        self.output_runtime = DummyOutputRuntime()
        self.start_calls: list[bool] = []
        self.replace_llm_calls: list[object | None] = []

    async def start(self, *, auto_flush_osc: bool) -> None:
        self.start_calls.append(auto_flush_osc)

    def has_stt_provider(self, channel: str) -> bool:
        return self.stt is not None if channel == "self" else False

    async def replace_llm_provider(self, llm: object | None) -> None:
        old_llm = self.llm
        self.replace_llm_calls.append(llm)
        self.llm = llm
        if old_llm is not None and old_llm is not llm and hasattr(old_llm, "close"):
            await old_llm.close()


def test_local_stt_download_prompt_helpers_removed() -> None:
    assert not hasattr(GuiController, "_show_local_stt_download_prompt")
    assert not hasattr(GuiController, "_on_local_stt_download_action")


def test_manual_local_asr_mismatches_persist_qwen_for_self_and_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_PARAKEET_V3
    settings.provider.peer_stt = STTProviderName.LOCAL_PARAKEET_JAPANESE
    settings.languages.source_language = "ko"
    settings.languages.peer_source_language = "en"
    saved: list[AppSettings] = []
    messages: list[str] = []
    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(
            SimpleNamespace(
                show_snackbar=lambda message, _color: messages.append(message),
            )
        ),
        config_path=Path("settings.json"),
    )
    controller.settings = settings

    monkeypatch.setattr(GuiController, "_sync_ui_from_settings", lambda self: None)
    monkeypatch.setattr(
        GuiController,
        "_save_settings",
        lambda self: saved.append(self.settings) or True,
    )

    assert controller._persist_current_manual_local_asr_fallback() is True
    assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN
    assert controller.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN
    assert len(saved) == 1
    assert messages == [i18n_module.t("local_stt.language_fallback_qwen")]


def test_action_snackbar_helper_removed_from_app_source() -> None:
    app_source = (Path(controller_module.__file__).parent / "app.py").read_text(encoding="utf-8")

    assert "def show_action_snackbar(" not in app_source


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN"])
def test_obsolete_local_stt_prompt_keys_are_removed(locale: str) -> None:
    bundle = i18n_module._load_bundle(locale)

    assert "local_stt.download_prompt_missing" not in bundle
    assert "local_stt.download_prompt_invalid" not in bundle
    assert "local_stt.download_prompt_failed" not in bundle
    assert "local_stt.download_action" not in bundle


@pytest.mark.asyncio
async def test_verify_api_key_returns_model_unavailable_when_fallback_model_works(
    monkeypatch,
) -> None:
    settings = AppSettings()
    settings.stt.low_latency_mode = True
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_FLASH
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings

    seen_models: list[str] = []

    async def fake_async_verify(api_key: str, *, base_url: str, model: str) -> bool:
        _ = api_key, base_url
        seen_models.append(model)
        return model == QwenLLMModel.QWEN_35_PLUS.value

    monkeypatch.setattr(AsyncQwenLLMProvider, "verify_api_key", staticmethod(fake_async_verify))

    success, msg = await controller.verify_api_key("alibaba_beijing", "secret")

    assert success is False
    assert msg == "qwen_model_unavailable:qwen3.5-flash"
    assert seen_models == ["qwen3.5-flash", "qwen3.5-plus"]


@pytest.mark.asyncio
async def test_verify_api_key_uses_deepseek_verifier(monkeypatch) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(SimpleNamespace(view_dashboard=DummyDashboard())),
        config_path=Path("settings.json"),
    )

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(DeepSeekLLMProvider, "verify_api_key", staticmethod(fake_verify))

    ok, message = await controller.verify_api_key("deepseek", "secret")

    assert ok is True
    assert message == "Verification successful"
    assert seen == ["secret"]
