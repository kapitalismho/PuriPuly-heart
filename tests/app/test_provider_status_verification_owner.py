from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

import pytest

from puripuly_heart.app.services.provider_status_verification import (
    ConfiguredProviderStatusVerificationRequest,
    ProviderStatusVerificationOwner,
)


@dataclass
class RecordingVerifier:
    outcomes: dict[tuple[str, str | None], bool] = field(default_factory=dict)
    calls: list[tuple[str, str, str | None, str | None, bool]] = field(default_factory=list)
    entered: asyncio.Event | None = None

    async def verify_api_key(
        self,
        provider: str,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        low_latency: bool = False,
    ) -> bool:
        self.calls.append((provider, api_key, model, base_url, low_latency))
        if self.entered is not None:
            self.entered.set()
            await asyncio.Event().wait()
        return self.outcomes.get((provider, model), False)

    async def verify_qwen_llm_api_key(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str | None,
        low_latency: bool,
    ) -> bool:
        self.calls.append(("qwen", api_key, model, base_url, low_latency))
        return self.outcomes.get(("qwen", model), False)


def _request(
    **changes: object,
) -> ConfiguredProviderStatusVerificationRequest:
    request = ConfiguredProviderStatusVerificationRequest(
        llm_runtime_present=True,
        stt_runtime_present=True,
        llm_provider="gemini",
        stt_provider="deepgram",
        llm_requires_secret=True,
        stt_requires_secret=True,
        runtime_translation_enabled=True,
        managed_openrouter_can_attempt=False,
        openrouter_managed_selected=False,
        gemini_model="gemini-model",
        qwen_selected_model="qwen-selected",
        qwen_fallback_models=("qwen-selected", "qwen-fallback"),
        qwen_base_url="https://dashscope.aliyuncs.com/api/v1",
        fast_translation_enabled=True,
        google_api_key="google-secret",
        openrouter_api_key="openrouter-secret",
        deepseek_api_key="deepseek-secret",
        qwen_api_key="qwen-secret",
        deepgram_api_key="deepgram-secret",
        soniox_api_key="soniox-secret",
    )
    return replace(request, **changes)


def test_status_request_repr_excludes_provider_credentials() -> None:
    rendered = repr(_request())

    assert "google-secret" not in rendered
    assert "openrouter-secret" not in rendered
    assert "deepseek-secret" not in rendered
    assert "qwen-secret" not in rendered
    assert "deepgram-secret" not in rendered
    assert "soniox-secret" not in rendered


@pytest.mark.asyncio
async def test_owner_names_verification_and_cancels_it_on_close() -> None:
    entered = asyncio.Event()
    verifier = RecordingVerifier(entered=entered)
    owner = ProviderStatusVerificationOwner(verifier=verifier)

    assert (
        owner.schedule(
            request_factory=_request,
            result_handler=lambda _result: None,
        )
        is True
    )
    await entered.wait()

    assert owner.active_task_names == ("verification-1",)

    await owner.close()

    assert owner.active_task_names == ()


@pytest.mark.asyncio
async def test_owner_contains_result_handler_failure_and_reports_diagnostics() -> None:
    error = RuntimeError("boom")
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []
    verifier = RecordingVerifier(
        outcomes={
            ("google", "gemini-model"): True,
            ("deepgram", None): True,
        }
    )

    def fail(_result: object) -> None:
        raise error

    owner = ProviderStatusVerificationOwner(
        verifier=verifier,
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        ),
    )

    assert owner.schedule(request_factory=_request, result_handler=fail) is True
    for _ in range(20):
        if diagnostics and not owner.active_task_names:
            break
        await asyncio.sleep(0)

    assert diagnostics == [
        (
            "provider_status_verification_failed",
            {"error_type": "RuntimeError"},
            error,
        )
    ]
    assert owner.active_task_names == ()

    await owner.close()


@pytest.mark.asyncio
async def test_owner_rejects_verification_after_ingress_stops() -> None:
    invoked = False
    verifier = RecordingVerifier()
    owner = ProviderStatusVerificationOwner(verifier=verifier)

    def handle(_result: object) -> None:
        nonlocal invoked
        invoked = True

    owner.stop_ingress()

    assert owner.schedule(request_factory=_request, result_handler=handle) is False
    await asyncio.sleep(0)
    assert invoked is False

    await owner.close()


@pytest.mark.asyncio
async def test_owner_verifies_configured_llm_and_stt_and_returns_presentation_result() -> None:
    verifier = RecordingVerifier(
        outcomes={
            ("google", "gemini-model"): True,
            ("deepgram", None): True,
        }
    )
    owner = ProviderStatusVerificationOwner(verifier=verifier)

    result = await owner.verify(_request())

    assert result.llm_valid is True
    assert result.stt_valid is True
    assert result.translation_needs_key is False
    assert result.stt_needs_key is False
    assert result.translation_enabled_update is None
    assert result.stt_enabled_update is None
    assert verifier.calls == [
        ("google", "google-secret", "gemini-model", None, False),
        ("deepgram", "deepgram-secret", None, None, False),
    ]


@pytest.mark.asyncio
async def test_owner_reuses_qwen_selected_probe_for_llm_and_stt() -> None:
    verifier = RecordingVerifier(outcomes={("qwen", "qwen-selected"): True})
    owner = ProviderStatusVerificationOwner(verifier=verifier)

    result = await owner.verify(
        _request(
            llm_provider="qwen",
            stt_provider="qwen_asr",
        )
    )

    assert result.llm_valid is True
    assert result.stt_valid is True
    assert verifier.calls == [
        (
            "qwen",
            "qwen-secret",
            "qwen-selected",
            "https://dashscope.aliyuncs.com/api/v1",
            True,
        )
    ]


@pytest.mark.asyncio
async def test_owner_uses_qwen_fallback_only_for_stt_key_validity() -> None:
    verifier = RecordingVerifier(outcomes={("qwen", "qwen-fallback"): True})
    owner = ProviderStatusVerificationOwner(verifier=verifier)

    result = await owner.verify(
        _request(
            llm_provider="qwen",
            stt_provider="qwen_asr",
        )
    )

    assert result.llm_valid is False
    assert result.stt_valid is True
    assert result.translation_needs_key is True
    assert result.translation_enabled_update is False
    assert verifier.calls == [
        (
            "qwen",
            "qwen-secret",
            "qwen-selected",
            "https://dashscope.aliyuncs.com/api/v1",
            True,
        ),
        (
            "qwen",
            "qwen-secret",
            "qwen-fallback",
            "https://dashscope.aliyuncs.com/api/v1",
            True,
        ),
    ]


@pytest.mark.asyncio
async def test_owner_accepts_managed_openrouter_without_local_key_when_runtime_can_attempt() -> (
    None
):
    verifier = RecordingVerifier()
    owner = ProviderStatusVerificationOwner(verifier=verifier)

    result = await owner.verify(
        _request(
            llm_provider="openrouter",
            openrouter_managed_selected=True,
            openrouter_api_key="",
            managed_openrouter_can_attempt=True,
            stt_requires_secret=False,
        )
    )

    assert result.llm_valid is True
    assert result.translation_needs_key is False
    assert verifier.calls == []


@pytest.mark.asyncio
async def test_owner_projects_invalid_and_local_llm_runtime_status() -> None:
    verifier = RecordingVerifier()
    owner = ProviderStatusVerificationOwner(verifier=verifier)

    invalid = await owner.verify(_request())
    local = await owner.verify(
        _request(
            llm_provider="local_llm",
            stt_requires_secret=False,
            runtime_translation_enabled=True,
        )
    )

    assert invalid.translation_needs_key is True
    assert invalid.stt_needs_key is True
    assert invalid.translation_enabled_update is False
    assert invalid.stt_enabled_update is False
    assert local.translation_needs_key is False
    assert local.translation_enabled_update is True
