from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    QwenLLMModel,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


class DummySecrets:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def get(self, key: str) -> str | None:
        return self._values.get(key)


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
async def test_verify_qwen_llm_api_key_uses_async_verifier_in_low_latency(monkeypatch) -> None:
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

    seen: dict[str, str] = {}

    async def fake_async_verify(api_key: str, *, base_url: str, model: str) -> bool:
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        seen["model"] = model
        return True

    async def fail_sync_verify(*_args, **_kwargs) -> bool:
        raise AssertionError("sync verifier must not be called in low latency mode")

    monkeypatch.setattr(AsyncQwenLLMProvider, "verify_api_key", staticmethod(fake_async_verify))
    monkeypatch.setattr(QwenLLMProvider, "verify_api_key", staticmethod(fail_sync_verify))

    ok = await controller._verify_qwen_llm_api_key(
        "secret", base_url="https://dashscope.aliyuncs.com/api/v1"
    )

    assert ok is True
    assert seen == {
        "api_key": "secret",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.5-flash",
    }


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_qwen_specific_verifiers(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.provider.stt = STTProviderName.QWEN_ASR
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"alibaba_api_key_beijing": "secret"}),
    )

    llm_seen: list[tuple[str, str]] = []

    async def fake_verify_qwen_llm(
        api_key: str,
        *,
        base_url: str,
        model: str,
        low_latency: bool,
    ) -> bool:
        assert low_latency is True
        llm_seen.append((api_key, base_url))
        return True

    async def fail_qwen_asr_verify(*_args, **_kwargs) -> bool:
        raise AssertionError("qwen ASR verifier should not be called when Alibaba result is shared")

    async def fail_legacy_verify(*_args, **_kwargs) -> bool:
        raise AssertionError("legacy llm verifier path must not be called")

    controller.provider_verifier = SimpleNamespace(
        verify_qwen_llm_api_key=fake_verify_qwen_llm,
    )
    monkeypatch.setattr(
        QwenASRRealtimeSTTBackend, "verify_api_key", staticmethod(fail_qwen_asr_verify)
    )
    monkeypatch.setattr(QwenLLMProvider, "verify_api_key", staticmethod(fail_legacy_verify))

    await controller._verify_and_update_status()

    assert llm_seen == [("secret", "https://dashscope.aliyuncs.com/api/v1")]
    assert app.view_dashboard.translation_needs_key is False
    assert app.view_dashboard.stt_needs_key is False


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
async def test_verify_and_update_status_splits_llm_model_access_from_stt_key_validity(
    monkeypatch,
) -> None:
    settings = AppSettings()
    settings.stt.low_latency_mode = True
    settings.provider.llm = LLMProviderName.QWEN
    settings.provider.stt = STTProviderName.QWEN_ASR
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_FLASH
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"alibaba_api_key": "secret"}),
    )

    seen_models: list[str] = []

    async def fake_async_verify(api_key: str, *, base_url: str, model: str) -> bool:
        _ = api_key, base_url
        seen_models.append(model)
        return model == QwenLLMModel.QWEN_35_PLUS.value

    monkeypatch.setattr(AsyncQwenLLMProvider, "verify_api_key", staticmethod(fake_async_verify))

    await controller._verify_and_update_status()

    assert app.view_dashboard.translation_needs_key is True
    assert app.view_dashboard.stt_needs_key is False
    assert seen_models == ["qwen3.5-flash", "qwen3.5-plus"]


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_selected_qwen_model_for_both_llm_and_stt_when_valid(
    monkeypatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.provider.stt = STTProviderName.QWEN_ASR
    settings.qwen.region = QwenRegion.SINGAPORE
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"alibaba_api_key_singapore": "secret"}),
    )

    seen_models: list[str] = []

    async def fake_verify_qwen_llm(
        api_key: str,
        *,
        base_url: str,
        model: str,
        low_latency: bool,
    ) -> bool:
        assert api_key == "secret"
        assert base_url == "https://dashscope-intl.aliyuncs.com/api/v1"
        assert low_latency is True
        seen_models.append(model)
        return True

    controller.provider_verifier = SimpleNamespace(
        verify_qwen_llm_api_key=fake_verify_qwen_llm,
    )

    await controller._verify_and_update_status()

    assert app.view_dashboard.translation_needs_key is False
    assert app.view_dashboard.stt_needs_key is False


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_openrouter_verifier(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"openrouter_api_key": "secret"}),
    )

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(OpenRouterLLMProvider, "verify_api_key", staticmethod(fake_verify))

    await controller._verify_and_update_status()

    assert seen == ["secret"]
    assert app.view_dashboard.translation_needs_key is False


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


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_deepseek_verifier(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.DEEPSEEK
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"deepseek_api_key": "secret"}),
    )

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(DeepSeekLLMProvider, "verify_api_key", staticmethod(fake_verify))

    await controller._verify_and_update_status()

    assert seen == ["secret"]
    assert app.view_dashboard.translation_needs_key is False


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_deepseek_env_key(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.DEEPSEEK
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-secret")

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(DeepSeekLLMProvider, "verify_api_key", staticmethod(fake_verify))

    await controller._verify_and_update_status()

    assert seen == ["env-secret"]
    assert app.view_dashboard.translation_needs_key is False


@pytest.mark.asyncio
async def test_local_llm_status_update_skips_connection_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.LOCAL_LLM
    app = SimpleNamespace(view_dashboard=DummyDashboard())
    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)

    monkeypatch.setenv("LOCAL_LLM_API_KEY", "env-secret")

    await controller._verify_and_update_status()

    assert app.view_dashboard.translation_needs_key is False
    assert app.view_dashboard.translation_enabled is True


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_selected_managed_openrouter_key(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"openrouter_managed_api_key": "managed-secret"}),
    )

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(OpenRouterLLMProvider, "verify_api_key", staticmethod(fake_verify))

    await controller._verify_and_update_status()

    assert seen == ["managed-secret"]
    assert app.view_dashboard.translation_needs_key is False


@pytest.mark.asyncio
async def test_verify_and_update_status_keeps_managed_openrouter_toggle_available_without_local_key(
    monkeypatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub(llm=object())

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )

    async def fail_verify(_api_key: str) -> bool:
        raise AssertionError("verify_api_key should not be called without a local managed key")

    monkeypatch.setattr(OpenRouterLLMProvider, "verify_api_key", staticmethod(fail_verify))

    await controller._verify_and_update_status()

    assert app.view_dashboard.translation_needs_key is False


@pytest.mark.asyncio
async def test_verify_and_update_status_marks_openrouter_none_selected_source_as_needs_key(
    monkeypatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.NONE
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub(llm=None)

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"openrouter_api_key": "secret"}),
    )

    async def fail_verify(_api_key: str) -> bool:
        raise AssertionError("verify_api_key should not be called")

    monkeypatch.setattr(OpenRouterLLMProvider, "verify_api_key", staticmethod(fail_verify))

    await controller._verify_and_update_status()

    assert app.view_dashboard.translation_needs_key is True
    assert app.view_dashboard.translation_enabled is False


@pytest.mark.asyncio
async def test_verify_and_update_status_treats_local_qwen_stt_as_keyless(
    monkeypatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.provider.llm = LLMProviderName.GEMINI
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(app),
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyHub()

    def fail_secret_store(*_args, **_kwargs):
        raise RuntimeError("secret store should not be needed for local STT")

    monkeypatch.setattr(controller_module, "create_secret_store", fail_secret_store)

    await controller._verify_and_update_status()

    assert app.view_dashboard.stt_needs_key is False
