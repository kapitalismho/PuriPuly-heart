from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.managed_auth import ManagedAuthState
from puripuly_heart.app.wiring_managed_auth_factory import (
    ManagedTranslationRuntimeAdapter,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseDiagnostics,
    ManagedOpenRouterReleaseResult,
)

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class AuthAdapter:
    def __init__(self, state: ManagedAuthState, claim_guard=None) -> None:
        self._state = state
        self._claim_guard = claim_guard

    def state(self) -> ManagedAuthState:
        return self._state

    def claim_guard(self):
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


def _owner(settings: AppSettingsVNext, path: Path) -> SettingsOwner:
    return SettingsOwner(
        path=path,
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=settings,
        authoritative=True,
    )


def _with_translation(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(settings.intent.translation, **changes),
        ),
    )


def _qwen_settings() -> AppSettingsVNext:
    return _with_translation(
        AppSettingsVNext(),
        model="qwen38_flash",
        connection="official_byok",
        qwen=replace(AppSettingsVNext().intent.translation.qwen, region="beijing"),
    )


def _custom_http_settings() -> AppSettingsVNext:
    return _with_translation(
        AppSettingsVNext(),
        model="custom_http",
        connection="custom_http",
        http_extension_id="demo",
    )


def _managed_settings(*, china: bool = False) -> AppSettingsVNext:
    return _with_translation(
        AppSettingsVNext(),
        model="deepseek_v4_flash" if china else AppSettingsVNext().intent.translation.model,
        connection="managed_china" if china else "managed",
        openrouter_selected_source="managed",
    )


def _adapter(
    settings: AppSettingsVNext,
    auth: AuthAdapter,
    *,
    config_path: Path,
    service=None,
    runtime_snapshot=(True, False, None),
    founder_dialog=lambda: False,
    persist_settings=lambda: True,
) -> ManagedTranslationRuntimeAdapter:
    return ManagedTranslationRuntimeAdapter(
        auth=auth,
        settings=_owner(settings, config_path),
        release_service_provider=lambda: service,
        runtime_snapshot_provider=lambda: runtime_snapshot,
        ingress_provider=lambda: False,
        founder_dialog=founder_dialog,
        persist_settings=persist_settings,
    )


def test_state_projects_runtime_provider_region_and_managed_auth_snapshot(tmp_path: Path) -> None:
    settings = _qwen_settings()
    adapter = _adapter(
        settings,
        AuthAdapter(
            _auth_state(
                managed_china=True,
                local_key_available=True,
            )
        ),
        config_path=tmp_path / "settings.json",
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


def test_state_labels_custom_http_without_reusing_inactive_llm_metadata(tmp_path: Path) -> None:
    settings = _custom_http_settings()
    adapter = _adapter(
        settings,
        AuthAdapter(_auth_state()),
        config_path=tmp_path / "settings.json",
        runtime_snapshot=(True, True, object()),
    )

    state = adapter.state()

    assert state.llm_available is True
    assert state.provider_name == "custom_http"
    assert state.qwen_region is None
    assert state.managed_selected is False


@pytest.mark.asyncio
async def test_custom_http_prepare_skips_managed_release_service(tmp_path: Path) -> None:
    settings = _custom_http_settings()

    async def unexpected_prepare() -> object:
        raise AssertionError("managed release must not prepare for Custom HTTP")

    adapter = _adapter(
        settings,
        AuthAdapter(_auth_state()),
        config_path=tmp_path / "settings.json",
        service=SimpleNamespace(prepare_for_translation=unexpected_prepare),
    )

    result = await adapter.prepare()

    assert result.ready is True


@pytest.mark.asyncio
async def test_prepare_ready_records_discord_claim_and_persists_identity(tmp_path: Path) -> None:
    settings = _managed_settings()
    events: list[str] = []

    class ClaimGuard:
        managed_state = SimpleNamespace(persist=lambda: events.append("persist"))

        async def preflight(self, source: str):
            events.append(f"preflight:{source}")
            return None

        def record_success(self, source: str) -> None:
            events.append(f"record:{source}")

    async def prepare_for_translation(
        *, allow_authorization: bool = True
    ) -> ManagedOpenRouterReleaseResult:
        _ = allow_authorization
        events.append("prepare")
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.retry",
            local_key_available=True,
        )

    adapter = _adapter(
        settings,
        AuthAdapter(_auth_state(), ClaimGuard()),
        config_path=tmp_path / "settings.json",
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
async def test_china_prepare_maps_required_result_to_qq_dialog_and_safe_diagnostics(
    tmp_path: Path,
) -> None:
    settings = _managed_settings(china=True)

    async def prepare_for_translation(
        *, allow_authorization: bool = True
    ) -> ManagedOpenRouterReleaseResult:
        _ = allow_authorization
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
        config_path=tmp_path / "settings.json",
        service=SimpleNamespace(prepare_for_translation=prepare_for_translation),
    )

    result = await adapter.prepare()

    assert result.ready is False
    assert result.message_key == "qq_managed_auth.required"
    assert result.message_kwargs == {"retry_after_ms": 50}
    assert result.diagnostics_text == ("operation=prepare code=required message=<redacted>")
    assert result.show_qq_dialog is True


def test_founder_letter_marks_active_identity_and_persists_after_dialog(tmp_path: Path) -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        state=replace(
            settings.state,
            managed_connection=replace(
                settings.state.managed_connection,
                active_managed_credential_ref="managed-ref",
            ),
        ),
    )
    persisted: list[str] = []
    config_path = tmp_path / "settings.json"

    def show_founder_dialog() -> bool:
        assert not config_path.exists()
        return True

    def observe_persisted_settings() -> None:
        fresh = (
            SettingsVNextCanonicalPersistenceAdapter().load_active(config_path).canonical_settings
        )
        assert fresh.state.managed_connection.active_managed_credential_ref == "managed-ref"
        assert fresh.state.managed_connection.founder_letter_seen_credential_ref == "managed-ref"
        persisted.append("persist")

    adapter = _adapter(
        settings,
        AuthAdapter(_auth_state()),
        config_path=config_path,
        founder_dialog=show_founder_dialog,
        persist_settings=observe_persisted_settings,
    )

    adapter.show_founder_letter()

    canonical = adapter.settings.canonical
    assert canonical is not None
    assert canonical.state.managed_connection.founder_letter_seen_credential_ref == "managed-ref"
    assert persisted == ["persist"]
    fresh_owner = SettingsOwner(
        path=config_path,
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
    )
    assert (
        fresh_owner.start().settings.state.managed_connection == canonical.state.managed_connection
    )
