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
from puripuly_heart.core.local_stt_assets import LocalSTTInstallState
from puripuly_heart.providers.llm.cerebras import CerebrasLLMProvider
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.controller import GuiController


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


class DummyHub:
    def __init__(self, *, llm: object | None = object(), stt: object | None = object()) -> None:
        self.llm = llm
        self.stt = stt
        self.translation_enabled = True
        self.ui_events: asyncio.Queue[object] = asyncio.Queue()
        self.start_calls: list[bool] = []

    async def start(self, *, auto_flush_osc: bool) -> None:
        self.start_calls.append(auto_flush_osc)


async def _start_controller_with_inspected_stt_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: STTProviderName,
    install_state: LocalSTTInstallState,
    hub_stt: object | None = object(),
) -> tuple[GuiController, DummyDashboard, list[str], list[str]]:
    settings = AppSettings()
    settings.provider.stt = provider
    dash = DummyDashboard()
    hub = DummyHub(stt=hub_stt)
    inspect_calls: list[str] = []
    install_calls: list[str] = []

    class FakeBridge:
        def __init__(self, *, app, event_queue, runtime_logging=None) -> None:
            _ = (app, event_queue, runtime_logging)

        async def run(self) -> None:
            await asyncio.sleep(0)

        def report_overlay_state(
            self,
            overlay_state: str,
            *,
            failure_reason: str | None = None,
        ) -> None:
            _ = (overlay_state, failure_reason)

    async def fake_init_pipeline(self) -> None:
        self.hub = hub

    async def fake_install(**kwargs):
        _ = kwargs
        install_calls.append("install")
        return object()

    def fake_inspect(*_args, **_kwargs):
        inspect_calls.append("inspect")
        return install_state

    monkeypatch.setattr(GuiController, "_load_or_init_settings", lambda self, path: settings)
    monkeypatch.setattr(GuiController, "_sync_ui_from_settings", lambda self: None)
    monkeypatch.setattr(GuiController, "_init_pipeline", fake_init_pipeline)
    monkeypatch.setattr(controller_module, "set_locale", lambda _locale: None)
    monkeypatch.setattr(controller_module, "UIEventBridge", FakeBridge)
    monkeypatch.setattr(controller_module, "inspect_local_stt_install_state", fake_inspect)
    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)

    controller = GuiController(
        page=SimpleNamespace(),
        app=SimpleNamespace(view_dashboard=dash),
        config_path=Path("settings.json"),
    )

    await controller.start()
    await asyncio.sleep(0)
    return controller, dash, inspect_calls, install_calls


def test_local_stt_download_prompt_helpers_removed() -> None:
    assert not hasattr(GuiController, "_show_local_stt_download_prompt")
    assert not hasattr(GuiController, "_on_local_stt_download_action")


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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"alibaba_api_key_beijing": "secret"}),
    )

    llm_seen: list[tuple[str, str]] = []

    async def fake_verify_qwen_llm(self, api_key: str, *, base_url: str, model: str) -> bool:
        llm_seen.append((api_key, base_url))
        return True

    async def fail_qwen_asr_verify(*_args, **_kwargs) -> bool:
        raise AssertionError("qwen ASR verifier should not be called when Alibaba result is shared")

    async def fail_legacy_verify(*_args, **_kwargs) -> bool:
        raise AssertionError("legacy llm verifier path must not be called")

    monkeypatch.setattr(GuiController, "_verify_qwen_llm_api_key", fake_verify_qwen_llm)
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"alibaba_api_key_singapore": "secret"}),
    )

    seen_models: list[str] = []

    async def fake_verify_qwen_llm(self, api_key: str, *, base_url: str, model: str) -> bool:
        assert api_key == "secret"
        assert base_url == "https://dashscope-intl.aliyuncs.com/api/v1"
        seen_models.append(model)
        return True

    monkeypatch.setattr(GuiController, "_verify_qwen_llm_api_key", fake_verify_qwen_llm)

    await controller._verify_and_update_status()

    assert app.view_dashboard.translation_needs_key is False
    assert app.view_dashboard.stt_needs_key is False


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_openrouter_verifier(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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
        app=SimpleNamespace(view_dashboard=DummyDashboard()),
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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
async def test_verify_api_key_uses_cerebras_verifier(monkeypatch) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=SimpleNamespace(view_dashboard=DummyDashboard()),
        config_path=Path("settings.json"),
    )

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(CerebrasLLMProvider, "verify_api_key", staticmethod(fake_verify))

    ok, message = await controller.verify_api_key("cerebras", "secret")

    assert ok is True
    assert message == "Verification successful"
    assert seen == ["secret"]


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_cerebras_verifier(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.CEREBRAS
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"cerebras_api_key": "secret"}),
    )

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(CerebrasLLMProvider, "verify_api_key", staticmethod(fake_verify))

    await controller._verify_and_update_status()

    assert seen == ["secret"]
    assert app.view_dashboard.translation_needs_key is False


@pytest.mark.asyncio
async def test_verify_and_update_status_uses_cerebras_env_key(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.CEREBRAS
    app = SimpleNamespace(view_dashboard=DummyDashboard())

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyHub()

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )
    monkeypatch.setenv("CEREBRAS_API_KEY", "env-secret")

    seen: list[str] = []

    async def fake_verify(api_key: str) -> bool:
        seen.append(api_key)
        return True

    monkeypatch.setattr(CerebrasLLMProvider, "verify_api_key", staticmethod(fake_verify))

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
    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
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

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyHub()

    def fail_secret_store(*_args, **_kwargs):
        raise RuntimeError("secret store should not be needed for local STT")

    monkeypatch.setattr(controller_module, "create_secret_store", fail_secret_store)

    await controller._verify_and_update_status()

    assert app.view_dashboard.stt_needs_key is False


@pytest.mark.asyncio
async def test_set_stt_enabled_starts_local_qwen_runtime_install_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    install_calls: list[str] = []
    release = SimpleNamespace(done=False)
    app = SimpleNamespace(
        view_dashboard=DummyDashboard(),
        _show_snackbar=lambda *_args, **_kwargs: None,
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    async def fake_install(**_kwargs):
        install_calls.append("install")
        while not release.done:
            await asyncio.sleep(0)
        return object()

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", lambda self: asyncio.sleep(0))

    controller = GuiController(
        page=SimpleNamespace(),
        app=app,
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._local_stt_install_state = LocalSTTInstallState(status="missing")

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)

    assert controller._stt_desired is False
    assert app.view_dashboard.stt_enabled is False
    assert app.view_dashboard.local_stt_notice_status == "downloading"
    assert app.view_dashboard.local_stt_notice_percent == 0
    assert install_calls == ["install"]

    release.done = True
    await controller._local_stt_download_task


@pytest.mark.asyncio
async def test_set_stt_enabled_starts_local_qwen_runtime_install_when_model_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    install_calls: list[str] = []
    release = SimpleNamespace(done=False)
    app = SimpleNamespace(
        view_dashboard=DummyDashboard(),
        _show_snackbar=lambda *_args, **_kwargs: None,
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    async def fake_install(**_kwargs):
        install_calls.append("install")
        while not release.done:
            await asyncio.sleep(0)
        return object()

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", lambda self: asyncio.sleep(0))

    controller = GuiController(
        page=SimpleNamespace(),
        app=app,
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._local_stt_install_state = LocalSTTInstallState(status="invalid")

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)

    assert controller._stt_desired is False
    assert app.view_dashboard.stt_enabled is False
    assert app.view_dashboard.local_stt_notice_status == "downloading"
    assert app.view_dashboard.local_stt_notice_percent == 0
    assert install_calls == ["install"]

    release.done = True
    await controller._local_stt_download_task


@pytest.mark.asyncio
async def test_set_stt_enabled_retries_runtime_install_after_download_failed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    install_calls: list[str] = []
    release = SimpleNamespace(done=False)
    app = SimpleNamespace(
        view_dashboard=DummyDashboard(),
        _show_snackbar=lambda *_args, **_kwargs: None,
    )

    async def fake_install(**_kwargs):
        install_calls.append("install")
        while not release.done:
            await asyncio.sleep(0)
        return object()

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", lambda self: asyncio.sleep(0))

    controller = GuiController(
        page=SimpleNamespace(),
        app=app,
        config_path=Path("settings.json"),
    )
    controller.settings = settings
    controller._local_stt_install_state = LocalSTTInstallState(status="invalid")
    controller._local_stt_runtime_status = "download_failed"

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)

    assert controller._stt_desired is False
    assert app.view_dashboard.stt_enabled is False
    assert app.view_dashboard.local_stt_notice_status == "downloading"
    assert app.view_dashboard.local_stt_notice_percent == 0
    assert install_calls == ["install"]

    release.done = True
    await controller._local_stt_download_task


@pytest.mark.asyncio
async def test_local_qwen_repeated_enable_during_runtime_install_is_single_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    dashboard = DummyDashboard()
    status_messages: list[str] = []
    install_calls: list[str] = []
    release = SimpleNamespace(done=False)

    app = SimpleNamespace(
        view_dashboard=dashboard,
        _show_snackbar=lambda message, *_args, **_kwargs: status_messages.append(message),
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    async def fake_install(**_kwargs):
        install_calls.append("install")
        while not release.done:
            await asyncio.sleep(0)
        return object()

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", lambda self: asyncio.sleep(0))

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._local_stt_install_state = LocalSTTInstallState(status="missing")

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)
    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)

    assert install_calls == ["install"]
    assert dashboard.local_stt_notice_status == "downloading"
    assert dashboard.local_stt_notice_percent == 0
    assert controller_module.t("local_stt.download_in_progress") in status_messages

    release.done = True
    await controller._local_stt_download_task


@pytest.mark.asyncio
async def test_stop_cancels_active_local_stt_download_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=SimpleNamespace(view_dashboard=DummyDashboard()),
        config_path=Path("settings.json"),
    )

    async def fake_set_stt_enabled(self, enabled: bool) -> None:
        _ = self, enabled

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = self, enabled

    async def fake_shutdown_overlay_runtime(self, *, preserve_failure_reason: bool) -> None:
        _ = self, preserve_failure_reason

    monkeypatch.setattr(GuiController, "set_stt_enabled", fake_set_stt_enabled)
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )
    monkeypatch.setattr(
        GuiController,
        "_shutdown_overlay_runtime",
        fake_shutdown_overlay_runtime,
    )

    controller._local_stt_download_task = asyncio.create_task(asyncio.sleep(3600))

    await controller.stop()

    assert controller._local_stt_download_task is None


@pytest.mark.asyncio
async def test_local_qwen_successful_runtime_install_retries_enable_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    dashboard = DummyDashboard()
    rebuild_calls: list[str] = []
    status_messages: list[str] = []
    switch_calls: list[bool] = []

    app = SimpleNamespace(
        view_dashboard=dashboard,
        _show_snackbar=lambda message, *_args, **_kwargs: status_messages.append(message),
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    async def fake_install(**_kwargs):
        return object()

    async def fake_rebuild(self):
        rebuild_calls.append("rebuild")

    async def fake_switch(self):
        switch_calls.append(self._stt_desired)

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", fake_rebuild)
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", fake_switch)

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._local_stt_install_state = LocalSTTInstallState(status="missing")

    await controller.set_stt_enabled(True)
    await controller._local_stt_download_task

    assert rebuild_calls == ["rebuild"]
    assert switch_calls == [True]
    assert dashboard.local_stt_notice_status is None
    assert controller_module.t("local_stt.download_success") not in status_messages


@pytest.mark.asyncio
async def test_local_qwen_runtime_install_does_not_auto_enable_after_provider_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    dashboard = DummyDashboard()
    switch_calls: list[bool] = []
    release = SimpleNamespace(done=False)

    app = SimpleNamespace(
        view_dashboard=dashboard,
        _show_snackbar=lambda *_args, **_kwargs: None,
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    async def fake_install(**_kwargs):
        while not release.done:
            await asyncio.sleep(0)
        return object()

    async def fake_switch(self):
        switch_calls.append(self._stt_desired)

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", fake_switch)

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._local_stt_install_state = LocalSTTInstallState(status="missing")

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)

    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    release.done = True
    await controller._local_stt_download_task

    assert switch_calls == []


@pytest.mark.asyncio
async def test_local_qwen_explicit_disable_during_runtime_install_clears_pending_auto_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    dashboard = DummyDashboard()
    rebuild_calls: list[str] = []
    switch_calls: list[bool] = []
    release = SimpleNamespace(done=False)

    app = SimpleNamespace(
        view_dashboard=dashboard,
        _show_snackbar=lambda *_args, **_kwargs: None,
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    async def fake_install(**_kwargs):
        while not release.done:
            await asyncio.sleep(0)
        return object()

    async def fake_rebuild(self):
        rebuild_calls.append("rebuild")

    async def fake_switch(self):
        switch_calls.append(self._stt_desired)

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", fake_rebuild)
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", fake_switch)

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._local_stt_install_state = LocalSTTInstallState(status="missing")

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)

    assert controller._local_stt_pending_enable_after_install is True

    await controller.set_stt_enabled(False)

    assert controller._local_stt_pending_enable_after_install is False

    release.done = True
    await controller._local_stt_download_task

    assert rebuild_calls == []
    assert switch_calls == [False]
    assert dashboard.stt_enabled is False


@pytest.mark.asyncio
async def test_local_qwen_reenable_during_runtime_install_rearms_pending_auto_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    dashboard = DummyDashboard()
    status_messages: list[str] = []
    rebuild_calls: list[str] = []
    switch_calls: list[bool] = []
    install_calls: list[str] = []
    release = SimpleNamespace(done=False)

    app = SimpleNamespace(
        view_dashboard=dashboard,
        _show_snackbar=lambda message, *_args, **_kwargs: status_messages.append(message),
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    async def fake_install(**_kwargs):
        install_calls.append("install")
        while not release.done:
            await asyncio.sleep(0)
        return object()

    async def fake_rebuild(self):
        rebuild_calls.append("rebuild")

    async def fake_switch(self):
        switch_calls.append(self._stt_desired)

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", fake_rebuild)
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", fake_switch)

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._local_stt_install_state = LocalSTTInstallState(status="missing")

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)
    await controller.set_stt_enabled(False)
    await controller.set_stt_enabled(True)

    assert install_calls == ["install"]
    assert controller._local_stt_pending_enable_after_install is True
    assert controller_module.t("local_stt.download_in_progress") in status_messages

    release.done = True
    await controller._local_stt_download_task

    assert rebuild_calls == ["rebuild"]
    assert switch_calls == [False, True]
    assert dashboard.stt_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("install_state", "expected_notice"),
    [
        (LocalSTTInstallState(status="missing"), "missing"),
        (
            LocalSTTInstallState(status="invalid", error_message="broken manifest"),
            "invalid",
        ),
        (LocalSTTInstallState(status="ready"), None),
    ],
    ids=["missing", "invalid", "ready"],
)
async def test_start_with_local_qwen_inspects_runtime_read_only(
    monkeypatch: pytest.MonkeyPatch,
    install_state: LocalSTTInstallState,
    expected_notice: str | None,
) -> None:
    (
        controller,
        dash,
        inspect_calls,
        install_calls,
    ) = await _start_controller_with_inspected_stt_state(
        monkeypatch,
        provider=STTProviderName.LOCAL_QWEN,
        install_state=install_state,
        hub_stt=object(),
    )

    assert inspect_calls == ["inspect"]
    assert install_calls == []
    assert controller._local_stt_download_task is None
    assert dash.stt_enabled is False
    assert dash.local_stt_notice_status == expected_notice
    assert dash.local_stt_notice_percent is None


@pytest.mark.asyncio
async def test_set_stt_enabled_local_qwen_download_path_does_not_prepare_managed_translation_or_mutate_selected_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    install_calls: list[str] = []
    release = SimpleNamespace(done=False)
    app = SimpleNamespace(
        view_dashboard=DummyDashboard(),
        _show_snackbar=lambda *_args, **_kwargs: None,
    )

    class DummyWarmupHub:
        def __init__(self) -> None:
            self.stt = object()
            self.peer_stt = None
            self.promo_calls = 0

        def mark_promo_eligible(self) -> None:
            self.promo_calls += 1

    class DummyManagedReleaseService:
        def __init__(self) -> None:
            self.prepare_calls = 0

        async def prepare_for_translation(self):
            self.prepare_calls += 1
            raise AssertionError("STT runtime path must not prepare managed translation")

    async def fake_install(**_kwargs):
        install_calls.append("install")
        while not release.done:
            await asyncio.sleep(0)
        return object()

    monkeypatch.setattr(controller_module, "ensure_local_stt_installed", fake_install)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", lambda self: asyncio.sleep(0))

    controller = GuiController(page=SimpleNamespace(), app=app, config_path=Path("settings.json"))
    controller.settings = settings
    controller.hub = DummyWarmupHub()
    controller._managed_openrouter_release_service = DummyManagedReleaseService()
    controller._local_stt_install_state = LocalSTTInstallState(status="missing")

    await controller.set_stt_enabled(True)
    await asyncio.sleep(0)

    assert install_calls == ["install"]
    assert controller._managed_openrouter_release_service.prepare_calls == 0
    assert controller.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED

    release.done = True
    await controller._local_stt_download_task

    assert controller._managed_openrouter_release_service.prepare_calls == 0
    assert controller.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED


@pytest.mark.asyncio
async def test_start_inspects_local_stt_without_auto_download_for_non_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        controller,
        dash,
        inspect_calls,
        install_calls,
    ) = await _start_controller_with_inspected_stt_state(
        monkeypatch,
        provider=STTProviderName.DEEPGRAM,
        install_state=LocalSTTInstallState(status="missing"),
    )

    assert inspect_calls == ["inspect"]
    assert install_calls == []
    assert controller._local_stt_download_task is None
    assert dash.local_stt_notice_status is None
    assert dash.local_stt_notice_percent is None
