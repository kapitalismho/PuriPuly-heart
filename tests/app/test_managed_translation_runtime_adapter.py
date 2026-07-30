from __future__ import annotations

from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.managed_auth import ManagedAuthState
from puripuly_heart.app.wiring_managed_auth_factory import (
    ManagedTranslationRuntimeAdapter,
)
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    QwenRegion,
    TranslationConnection,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseDiagnostics,
    ManagedOpenRouterReleaseResult,
)


class AuthAdapter:
    def __init__(self, state: ManagedAuthState, claim_guard=None) -> None:
        self._state = state
        self._claim_guard = claim_guard

    def state(self) -> ManagedAuthState:
        return self._state

    def claim_guard(self, _settings: AppSettings):
        if self._claim_guard is None:
            raise AssertionError("claim guard was not expected")
        return self._claim_guard


def _auth_state(
    *,
    managed_selected: bool = True,
    managed_china: bool = False,
    local_key_available: bool = False,
    release_service_available: bool = True,
) -> ManagedAuthState:
    return ManagedAuthState(
        settings_available=True,
        managed_selected=managed_selected,
        managed_china=managed_china,
        local_key_available=local_key_available,
        release_service_available=release_service_available,
        runtime_available=True,
        ingress_frozen=False,
    )


def _adapter(
    settings: AppSettings,
    auth: AuthAdapter,
    *,
    service=None,
    runtime_snapshot=(True, False, None),
    founder_dialog=lambda: False,
    persist_settings=lambda: True,
) -> ManagedTranslationRuntimeAdapter:
    return ManagedTranslationRuntimeAdapter(
        auth=auth,
        settings_provider=lambda: settings,
        release_service_provider=lambda: service,
        runtime_snapshot_provider=lambda: runtime_snapshot,
        ingress_provider=lambda: False,
        founder_dialog=founder_dialog,
        persist_settings=persist_settings,
    )


def test_state_projects_runtime_provider_region_and_managed_auth_snapshot() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.region = QwenRegion.BEIJING
    adapter = _adapter(
        settings,
        AuthAdapter(
            _auth_state(
                managed_china=True,
                local_key_available=True,
            )
        ),
        runtime_snapshot=(True, True, object()),
    )

    state = adapter.state()

    assert state.runtime_available is True
    assert state.translation_enabled is True
    assert state.llm_available is True
    assert state.provider_name == "qwen"
    assert state.qwen_region == "beijing"
    assert state.managed_selected is True
    assert state.managed_china is True
    assert state.managed_local_key_available is True


@pytest.mark.asyncio
async def test_prepare_ready_records_discord_claim_and_persists_identity() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.MANAGED
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    events: list[str] = []

    class ClaimGuard:
        managed_state = SimpleNamespace(persist=lambda: events.append("persist"))

        async def preflight(self, source: str):
            events.append(f"preflight:{source}")
            return None

        def record_success(self, source: str) -> None:
            events.append(f"record:{source}")

    async def prepare_for_translation() -> ManagedOpenRouterReleaseResult:
        events.append("prepare")
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            local_key_available=True,
        )

    adapter = _adapter(
        settings,
        AuthAdapter(_auth_state(), ClaimGuard()),
        service=SimpleNamespace(prepare_for_translation=prepare_for_translation),
    )

    result = await adapter.prepare()

    assert result.ready is True
    assert events == [
        "preflight:discord",
        "prepare",
        "record:discord",
        "persist",
    ]


@pytest.mark.asyncio
async def test_china_prepare_maps_required_result_to_qq_dialog_and_safe_diagnostics() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    async def prepare_for_translation() -> ManagedOpenRouterReleaseResult:
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="qq_managed_auth.required",
            message_kwargs={"retry_after_ms": 50},
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="prepare",
                code="required",
                message="raw failure",
            ),
        )

    adapter = _adapter(
        settings,
        AuthAdapter(_auth_state(managed_china=True)),
        service=SimpleNamespace(prepare_for_translation=prepare_for_translation),
    )

    result = await adapter.prepare()

    assert result.ready is False
    assert result.message_key == "qq_managed_auth.required"
    assert result.message_kwargs == {"retry_after_ms": 50}
    assert result.diagnostics_text == ("operation=prepare code=required message=<redacted>")
    assert result.show_qq_dialog is True


def test_founder_letter_marks_active_identity_and_persists_after_dialog() -> None:
    settings = AppSettings()
    settings.managed_identity.active_managed_credential_ref = "managed-ref"
    persisted: list[str] = []
    adapter = _adapter(
        settings,
        AuthAdapter(_auth_state()),
        founder_dialog=lambda: True,
        persist_settings=lambda: persisted.append("persist"),
    )

    adapter.show_founder_letter()

    assert settings.managed_identity.founder_letter_seen_credential_ref == "managed-ref"
    assert persisted == ["persist"]
