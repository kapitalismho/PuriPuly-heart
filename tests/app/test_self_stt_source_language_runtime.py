from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

import pytest
from puripuly_heart.app.services.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)
from puripuly_heart.app.wiring_stt_factory import (
    build_self_capture_session_config,
    build_self_capture_vad_signature,
    build_self_stt_provider_request,
    build_self_stt_provider_signature,
    build_self_stt_runtime_signature,
)

from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
    SelfCaptureProviderMutation,
    SelfCaptureProviderMutationStatus,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionState,
)

_CLOUD_PROVIDERS = (
    STTProviderName.DEEPGRAM,
    STTProviderName.QWEN_ASR,
    STTProviderName.SONIOX,
)


class _Admission:
    async def admit(self, config: SelfCaptureSessionConfig) -> SelfCaptureAdmission:
        _ = config
        return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)


class _RecordingProvider:
    def __init__(self) -> None:
        self.ready = False
        self.replace_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)
        self.handoff_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.APPLIED)
        self.replace_calls: list[tuple[object, bool]] = []
        self.handoff_calls: list[tuple[object, bool]] = []
        self.reconfigure_calls: list[object] = []
        self.release_calls: list[tuple[str, float | None]] = []
        self.start_calls = 0
        self.warmup_calls = 0
        self.terminal_failure_handler = None

    def is_ready(self, config: SelfCaptureSessionConfig) -> bool:
        _ = config
        return self.ready

    async def replace(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: Callable[[Exception], Awaitable[None]],
    ) -> SelfCaptureProviderMutation:
        self.replace_calls.append((request, start))
        self.terminal_failure_handler = on_terminal_failure
        if self.replace_result.status is SelfCaptureProviderMutationStatus.APPLIED:
            self.ready = True
        return self.replace_result

    async def handoff(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: Callable[[Exception], Awaitable[None]],
    ) -> SelfCaptureProviderMutation:
        self.handoff_calls.append((request, start))
        self.terminal_failure_handler = on_terminal_failure
        return self.handoff_result

    async def cancel_handoff(self) -> bool:
        return True

    async def start_ingress(self) -> None:
        self.start_calls += 1

    async def warmup(self) -> None:
        self.warmup_calls += 1

    async def reconfigure(self, session_options: object) -> None:
        self.reconfigure_calls.append(session_options)

    async def release(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None:
        self.release_calls.append((mode, release_backend_after))
        self.ready = False


class _Source:
    async def close(self) -> None:
        return None


class _Loop:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def run(self, **_kwargs: object) -> None:
        await self.release.wait()


def _settings(provider: STTProviderName, language: str) -> AppSettings:
    settings = AppSettings()
    settings.provider.stt = provider
    settings.languages.source_language = language
    return settings


def _build_owner(
    provider: _RecordingProvider,
    settings_holder: dict[str, AppSettings],
) -> SelfCaptureSessionOwner:
    loop = _Loop()
    return SelfCaptureSessionOwner(
        admission=_Admission(),
        provider=provider,
        provider_request_factory=lambda _config, warmup: build_self_stt_provider_request(
            settings_holder["settings"],
            warmup=warmup,
        ),
        source_factory=lambda _config: _Source(),
        vad_factory=lambda _config: object(),
        run_audio_loop=loop.run,
        vad_sink=object(),
    )


def _application_owner(
    *,
    session: SelfCaptureSessionOwner,
    settings_holder: dict[str, AppSettings],
) -> SelfCaptureApplicationOwner:
    return SelfCaptureApplicationOwner(
        settings_provider=lambda: SelfCaptureApplicationSettings(
            config=build_self_capture_session_config(settings_holder["settings"]),
            provider_id=settings_holder["settings"].provider.stt.value,
            qwen_region=None,
        ),
        runtime_available=lambda: True,
        capture_owner=lambda: session,
        capture_owner_if_created=lambda: session,
        persist_manual_fallback=lambda: True,
        reset_local_pending=lambda: None,
        clear_gpu_pending=lambda: None,
        overlay_state_provider=lambda: "off",
        mark_promo_eligible=lambda: None,
        dashboard_enabled_sink=lambda _enabled: None,
        dashboard_needs_key_sink=lambda _needs_key: None,
        dashboard_needs_key=lambda _available: False,
        state_sink=lambda _snapshot: None,
        sync_effective_flags=lambda: None,
        sync_local_notice=lambda: None,
        log_basic=lambda _message: None,
        log_detailed=lambda _message, _level: None,
    )


@pytest.mark.parametrize("provider", _CLOUD_PROVIDERS)
def test_cloud_source_language_is_owned_by_self_provider_identity(
    provider: STTProviderName,
) -> None:
    korean = _settings(provider, "ko")
    japanese = _settings(provider, "ja")

    assert build_self_stt_runtime_signature(korean) != build_self_stt_runtime_signature(japanese)
    assert build_self_stt_provider_signature(korean) != build_self_stt_provider_signature(japanese)
    assert build_self_capture_vad_signature(korean) == build_self_capture_vad_signature(japanese)
    assert build_self_capture_session_config(korean).session_options is None
    assert build_self_stt_provider_request(japanese).config.source_language == "ja"


@pytest.mark.parametrize(
    "provider",
    (STTProviderName.LOCAL_QWEN, STTProviderName.LOCAL_QWEN_GPU),
)
def test_local_source_language_is_owned_by_session_options(
    provider: STTProviderName,
) -> None:
    korean = _settings(provider, "ko")
    japanese = _settings(provider, "ja")
    korean_config = build_self_capture_session_config(korean)
    japanese_config = build_self_capture_session_config(japanese)

    assert build_self_stt_provider_signature(korean) == build_self_stt_provider_signature(japanese)
    assert korean_config.session_options is not None
    assert japanese_config.session_options is not None
    assert korean_config.session_options != japanese_config.session_options
    assert japanese_config.session_options.source_language == "ja"


def test_local_cpu_auto_model_identity_changes_with_source_language() -> None:
    english = _settings(STTProviderName.LOCAL_CPU_AUTO, "en")
    japanese = _settings(STTProviderName.LOCAL_CPU_AUTO, "ja")
    korean = _settings(STTProviderName.LOCAL_QWEN, "ko")
    japanese_qwen = _settings(STTProviderName.LOCAL_QWEN, "ja")

    assert build_self_stt_provider_signature(english) != build_self_stt_provider_signature(japanese)
    assert build_self_stt_provider_signature(korean) == build_self_stt_provider_signature(
        japanese_qwen
    )


def test_custom_vocabulary_still_changes_self_runtime_signature() -> None:
    settings = _settings(STTProviderName.DEEPGRAM, "ko")
    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": ["Puripuly"]}
    before = build_self_stt_runtime_signature(settings)
    provider_before = build_self_stt_provider_signature(settings)
    settings.stt.custom_terms = {"ko": ["Puripuly", "VRChat"]}

    assert build_self_stt_runtime_signature(settings) != before
    assert build_self_stt_provider_signature(settings) == provider_before


@pytest.mark.asyncio
async def test_running_cloud_source_language_change_handoffs_new_backend() -> None:
    korean = _settings(STTProviderName.DEEPGRAM, "ko")
    japanese = _settings(STTProviderName.DEEPGRAM, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.apply_intent(build_self_capture_session_config(korean), enabled=True)
    settings_holder["settings"] = japanese
    snapshot = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.state is SelfCaptureSessionState.RUNNING
    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.reconfigure_calls == []
    assert len(provider.handoff_calls) == 1
    request, started = provider.handoff_calls[0]
    assert started is True
    assert request.config.source_language == "ja"
    await owner.close()


@pytest.mark.asyncio
async def test_running_local_source_language_change_reconfigures_session_options() -> None:
    korean = _settings(STTProviderName.LOCAL_QWEN, "ko")
    japanese = _settings(STTProviderName.LOCAL_QWEN, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.apply_intent(build_self_capture_session_config(korean), enabled=True)
    settings_holder["settings"] = japanese
    snapshot = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.state is SelfCaptureSessionState.RUNNING
    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.handoff_calls == []
    assert [options.source_language for options in provider.reconfigure_calls] == ["ja"]
    await owner.close()


@pytest.mark.asyncio
async def test_running_local_cpu_auto_language_change_handoffs_when_model_changes() -> None:
    english = _settings(STTProviderName.LOCAL_CPU_AUTO, "en")
    japanese = _settings(STTProviderName.LOCAL_CPU_AUTO, "ja")
    settings_holder = {"settings": english}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.apply_intent(build_self_capture_session_config(english), enabled=True)
    settings_holder["settings"] = japanese
    snapshot = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.state is SelfCaptureSessionState.RUNNING
    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.reconfigure_calls == []
    assert len(provider.handoff_calls) == 1
    request, started = provider.handoff_calls[0]
    assert started is True
    assert request.config.source_language == "ja"
    await owner.close()


@pytest.mark.asyncio
async def test_language_change_while_stt_off_prepares_new_language() -> None:
    korean = _settings(STTProviderName.DEEPGRAM, "ko")
    japanese = _settings(STTProviderName.DEEPGRAM, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    owner = _build_owner(provider, settings_holder)

    await owner.prepare_provider(build_self_capture_session_config(korean))
    settings_holder["settings"] = japanese
    snapshot = await owner.prepare_provider(build_self_capture_session_config(japanese))
    started = await owner.apply_intent(build_self_capture_session_config(japanese), enabled=True)

    assert snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert started.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert [
        cast(Any, request).config.source_language for request, _start in provider.replace_calls
    ][-1] == "ja"
    await owner.close()


@pytest.mark.asyncio
async def test_replace_provider_hot_cloud_language_change_uses_application_path() -> None:
    korean = _settings(STTProviderName.DEEPGRAM, "ko")
    japanese = _settings(STTProviderName.DEEPGRAM, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    session = _build_owner(provider, settings_holder)
    application = _application_owner(session=session, settings_holder=settings_holder)

    await session.apply_intent(build_self_capture_session_config(korean), enabled=True)
    settings_holder["settings"] = japanese
    await application.replace_provider(smooth_local=True)

    assert session.snapshot.runtime_signature == build_self_stt_runtime_signature(japanese)
    assert provider.handoff_calls
    assert provider.handoff_calls[-1][0].config.source_language == "ja"
    await session.close()


@pytest.mark.asyncio
async def test_failed_cloud_language_handoff_does_not_look_applied() -> None:
    korean = _settings(STTProviderName.DEEPGRAM, "ko")
    japanese = _settings(STTProviderName.DEEPGRAM, "ja")
    settings_holder = {"settings": korean}
    provider = _RecordingProvider()
    session = _build_owner(provider, settings_holder)
    application = _application_owner(session=session, settings_holder=settings_holder)

    await session.apply_intent(build_self_capture_session_config(korean), enabled=True)
    provider.handoff_result = SelfCaptureProviderMutation(SelfCaptureProviderMutationStatus.FAILED)
    settings_holder["settings"] = japanese

    with pytest.raises(RuntimeError, match="did not apply the requested configuration"):
        await application.replace_provider(smooth_local=True)

    assert session.snapshot.runtime_signature == build_self_stt_runtime_signature(korean)
    assert session.snapshot.provider_status is SelfCaptureProviderStatus.READY
    await session.close()
