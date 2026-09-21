from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from puripuly_heart.app.services.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
    SelfCaptureRuntimeApplyError,
)
from puripuly_heart.app.wiring_capture_runtime import CaptureOwnerFactory
from puripuly_heart.app.wiring_stt_factory import build_self_capture_session_config

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.self_capture import (
    SelfCaptureProviderStatus,
    SelfCaptureSessionState,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("desired_active", "settings_moved_on", "expected_required"),
    [
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
async def test_replace_provider_reports_only_required_non_application(
    desired_active: bool,
    settings_moved_on: bool,
    expected_required: bool,
) -> None:
    previous = AppSettingsVNext()
    next_settings = replace(
        previous,
        intent=replace(
            previous.intent,
            languages=replace(previous.intent.languages, source_language="ja"),
        ),
    )
    previous_config = build_self_capture_session_config(previous)
    requested_config = build_self_capture_session_config(next_settings)
    newer_settings = replace(
        next_settings,
        intent=replace(
            next_settings.intent,
            languages=replace(next_settings.intent.languages, source_language="en"),
        ),
    )
    newer_config = (
        build_self_capture_session_config(newer_settings) if settings_moved_on else requested_config
    )
    reads: list[object] = []
    logs: list[str] = []
    state = SelfCaptureSessionState.RUNNING if desired_active else SelfCaptureSessionState.STOPPED

    def snapshot() -> SimpleNamespace:
        return SimpleNamespace(
            provider_status=SelfCaptureProviderStatus.READY,
            failure_reason=None,
            runtime_signature=previous_config.runtime_signature,
            desired_active=desired_active,
            effective_active=desired_active,
            state=state,
        )

    class CaptureOwner:
        snapshot = SimpleNamespace(desired_active=desired_active)
        loop_task = object() if desired_active else None

        async def prepare_provider(self, config: object) -> object:
            return snapshot()

        async def apply_intent(self, config: object, **kwargs: object) -> object:
            return snapshot()

    def settings_provider() -> SelfCaptureApplicationSettings:
        reads.append(None)
        config = requested_config if len(reads) == 1 else newer_config
        return SelfCaptureApplicationSettings(
            config=config,
            provider_id=config.provider_id,
            qwen_region=None,
        )

    owner = SelfCaptureApplicationOwner(
        settings_provider=settings_provider,
        runtime_available=lambda: True,
        capture_owner=lambda: cast(Any, CaptureOwner()),
        capture_owner_if_created=lambda: cast(Any, CaptureOwner()),
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
        log_basic=logs.append,
        log_diagnostic=lambda _message, _level: None,
    )

    if expected_required:
        with pytest.raises(
            SelfCaptureRuntimeApplyError,
            match="did not apply the requested configuration",
        ):
            await owner.replace_provider(smooth_local=True)
    else:
        await owner.replace_provider(smooth_local=True)

    assert logs == [
        "[STT] Provider replacement not applied: failure_reason=none "
        f"provider_status=ready state={state.value} signature_mismatch=True "
        f"required={expected_required}"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("desired_active", "expected_operation"),
    [(False, "prepare"), (True, "apply")],
)
async def test_replace_provider_propagates_updated_config_by_capture_activity(
    desired_active: bool,
    expected_operation: str,
) -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(
                settings.intent.stt,
                provider=STTProviderName.DEEPGRAM.value,
                custom_vocabulary_enabled=True,
                custom_terms={"ko": ["Puripuly", "VRChat"]},
            ),
            languages=replace(settings.intent.languages, source_language="ko"),
        ),
    )
    capture_config = build_self_capture_session_config(settings)
    calls: list[tuple[str, object, dict[str, object]]] = []

    class CaptureOwner:
        snapshot = SimpleNamespace(desired_active=desired_active)
        loop_task = object() if desired_active else None

        async def prepare_provider(self, config: object) -> object:
            calls.append(("prepare", config, {}))
            return SimpleNamespace(
                provider_status=SelfCaptureProviderStatus.READY,
                failure_reason=None,
                runtime_signature=capture_config.runtime_signature,
                desired_active=desired_active,
                effective_active=desired_active,
            )

        async def apply_intent(self, config: object, **kwargs: object) -> object:
            calls.append(("apply", config, kwargs))
            return SimpleNamespace(
                provider_status=SelfCaptureProviderStatus.READY,
                failure_reason=None,
                runtime_signature=capture_config.runtime_signature,
                desired_active=desired_active,
                effective_active=desired_active,
            )

    capture_owner = CaptureOwner()
    owner = SelfCaptureApplicationOwner(
        settings_provider=lambda: SelfCaptureApplicationSettings(
            config=capture_config,
            provider_id=capture_config.provider_id,
            qwen_region=None,
        ),
        runtime_available=lambda: True,
        capture_owner=lambda: cast(Any, capture_owner),
        capture_owner_if_created=lambda: cast(Any, capture_owner),
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
        log_diagnostic=lambda _message, _level: None,
    )

    await owner.replace_provider(smooth_local=True)

    [(operation, propagated_config, kwargs)] = calls
    assert operation == expected_operation
    assert propagated_config is capture_config
    assert kwargs == (
        {
            "enabled": True,
            "restart": False,
            "explicit_toggle_off": False,
        }
        if desired_active
        else {}
    )

    factory = CaptureOwnerFactory(
        canonical_provider=lambda: settings,
        self_admission=cast(Any, None),
        ensure_peer_local_ready=cast(Any, None),
        clock=cast(Any, None),
        log_basic=lambda _message: None,
        log_diagnostic=lambda _message: None,
        source_wrapper=lambda source: source,
        self_state_sink=lambda _snapshot: None,
        self_diagnostic_sink=lambda _diagnostic: None,
        peer_state_sink=lambda _snapshot: None,
        peer_diagnostic_sink=lambda _diagnostic: None,
        local_asr_diagnostic_sink=lambda _diagnostic: None,
    )
    request = factory.self_provider_request(capture_config, False)
    assert request.config.source_language == "ko"
    assert request.config.custom_vocabulary_enabled is True
    assert request.config.custom_terms == {"ko": ("Puripuly", "VRChat")}
