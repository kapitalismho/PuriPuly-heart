from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app import wiring_runtime_pipeline as runtime_module
from puripuly_heart.app.wiring.wiring_runtime_pipeline import compose_runtime_pipeline
from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    TranslationSettings,
)
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.runtime.prebuilt_local_asr_provider_runtime import (
    PrebuiltLocalASRProviderRuntimeFactory,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore


class RecordingRelease:
    service = None

    def __init__(self) -> None:
        self.rebuild_calls = 0

    async def rebuild(self, *, secrets: object) -> None:
        self.rebuild_calls += 1


class RecordingBackend:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class RecordingSender:
    def close(self) -> None:
        return


class RecordingChatbox:
    def enqueue(self, message: object) -> None:
        _ = message

    def send_typing(self, is_typing: bool) -> None:
        _ = is_typing

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = reason, active

    def clear_typing_reasons(self) -> None:
        return

    def process_due(self) -> None:
        return

    def send_immediate(self, text: str) -> bool:
        _ = text
        return True


class RecordingCapture:
    async def prepare_provider(self, config: object) -> object:
        _ = config
        return SimpleNamespace(provider_status=SimpleNamespace(value="ready"))

    async def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_custom_http_pipeline_skips_managed_rebuild_and_owns_backend_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.CUSTOM_HTTP,
        connection=TranslationConnection.CUSTOM_HTTP,
        http_extension_id="demo",
    )
    release = RecordingRelease()
    backend = RecordingBackend()
    registry = HttpExtensionRegistry(tmp_path)
    registry.reload()
    created: list[object] = []

    monkeypatch.setattr(
        runtime_module, "create_secret_store", lambda *_a, **_k: InMemorySecretStore()
    )
    monkeypatch.setattr(
        runtime_module,
        "create_translation_backend",
        lambda *_a, **_k: (created.append(backend), backend)[1],
    )
    monkeypatch.setattr(runtime_module, "VrchatOscUdpSender", lambda *_a, **_k: RecordingSender())
    monkeypatch.setattr(runtime_module, "ChatboxPaginator", lambda *_a, **_k: RecordingChatbox())

    pipeline = await compose_runtime_pipeline(
        settings=settings,
        config_path=Path("settings.json"),
        clock=SystemClock(),
        runtime_logging=None,
        managed_release=release,
        managed_delegate_ready=lambda: None,
        local_asr_factory=lambda _secrets: PrebuiltLocalASRProviderRuntimeFactory(
            self_provider=None,
            peer_provider=None,
        ),
        self_capture_factory=lambda *_args: RecordingCapture(),
        peer_capture_factory=lambda *_args: RecordingCapture(),
        vrc_mic_state=None,
        vrc_mic_audio_gate=None,
        receiver_active=False,
        stt_failure_sink=lambda _message: None,
        http_extensions=registry,
    )

    assert created == [backend]
    assert release.rebuild_calls == 0
    assert pipeline.llm_runtime.provider is backend

    await pipeline.resource_owner.close()

    assert backend.closed is True
