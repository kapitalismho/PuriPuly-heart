from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.settings_runtime_effects import SettingsRuntimeTransition
from puripuly_heart.app.services.settings.settings_runtime_effects import (
    SettingsRuntimeEffectsAdapter,
)
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_runtime_signature_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class _AsyncNoop:
    async def apply_controls(self, _controls) -> None:
        return None

    async def sync(self, **_kwargs) -> None:
        return None

    async def inspect_cpu(self) -> None:
        return None

    async def inspect_gpu(self, **_kwargs) -> None:
        return None


class _Overlay:
    snapshot = SimpleNamespace(fallback_active=False)

    def current_presenter(self):
        return None

    def runtime_is_active(self) -> bool:
        return False

    def set_enabled(self, _enabled):
        return None

    def publish_presentation(self) -> None:
        return None


class _VrcMic:
    last_enabled = False

    async def configure(self, **_kwargs) -> None:
        return None


def test_peer_runtime_convergence_requires_capture_and_committed_live_provider() -> None:
    settings = AppSettingsVNext()
    adapter = object.__new__(SettingsRuntimeEffectsAdapter)
    capture_results = iter((None, True, True))
    adapter._peer = SimpleNamespace(
        owner=SimpleNamespace(
            capture_runtime_convergence=lambda _config, _state: next(capture_results)
        ),
        state_for=lambda _channel: object(),
    )
    channel = SimpleNamespace(
        provider_id=build_peer_capture_session_config_from_vnext(settings).provider_id,
        has_resources=True,
        provider_live=True,
        pending_handoff=True,
        phase="ready",
    )
    adapter._pipeline = SimpleNamespace(
        local_asr_runtime=SimpleNamespace(
            snapshot=SimpleNamespace(channel_for=lambda _channel: channel)
        )
    )
    adapter._canonical_settings = lambda value: value

    assert adapter._peer_runtime_converged(settings) is None
    assert adapter._peer_runtime_converged(settings) is False
    channel.pending_handoff = False
    assert adapter._peer_runtime_converged(settings) is True


@pytest.mark.asyncio
async def test_failed_self_runtime_apply_does_not_write_target_signature_cache_first() -> None:
    previous = AppSettingsVNext()
    target = replace(
        previous,
        intent=replace(
            previous.intent,
            stt=replace(previous.intent.stt, provider="soniox"),
        ),
    )
    events: list[str] = []

    async def refresh_idle_peer() -> None:
        events.append("peer_refresh")

    adapter = object.__new__(SettingsRuntimeEffectsAdapter)
    adapter._desktop_overlay = _AsyncNoop()
    adapter._clipboard = SimpleNamespace(strict_runtime_errors=False, sync=_AsyncNoop().sync)
    adapter._provisioning = _AsyncNoop()
    adapter._clear_local_pending = lambda: None
    adapter._pipeline = SimpleNamespace(
        translation_runtime_configuration=None,
        peer_translation_channel=object(),
    )
    adapter._peer = SimpleNamespace(
        owner=SimpleNamespace(
            effective_enabled=lambda: False,
            activation_requested=lambda **_kwargs: False,
            refresh_runtime=refresh_idle_peer,
        )
    )
    adapter._settings = SimpleNamespace(
        overlay_enabled=lambda: False,
        peer_translation_enabled=lambda: False,
    )
    adapter._gpu = SimpleNamespace(
        state_provider=lambda: SimpleNamespace(selected_provider_requires_model=False)
    )
    adapter._overlay = _Overlay()
    adapter._vrc_mic_sync = _VrcMic()
    adapter._runtime_logging = SimpleNamespace(emit_detailed=lambda *_args, **_kwargs: None)
    adapter._presentation = SimpleNamespace(current_locale=lambda: "en")
    adapter._projection = SimpleNamespace()
    adapter._rebuild_managed_gemma = lambda: None
    adapter._canonical_settings = lambda value: value
    adapter._sync_effective_translation_flags = lambda _settings: None
    adapter._peer_runtime_converged = lambda _settings: None
    adapter._sync_signatures = lambda _settings: events.append("signature_cache_write")

    async def fail_self_replace(_smooth: bool) -> None:
        events.append("self_replace")
        raise RuntimeError("boundary handoff failed")

    adapter._replace_self_stt = fail_self_replace
    transition = SettingsRuntimeTransition(
        settings=target,
        previous_settings=previous,
        previous_locale="en",
        previous_overlay_enabled=False,
        previous_self_signature=("previous",),
        previous_peer_signature=None,
        previous_peer_translation_enabled=False,
        previous_peer_activation_requested=False,
        source_language_changed=False,
        target_language_changed=False,
        effective_peer_source_changed=False,
        effective_peer_target_changed=False,
        peer_source_language_changed=False,
        peer_target_language_changed=False,
        peer_source_mode_changed=False,
        desktop_runtime_controls=(),
    )

    with pytest.raises(RuntimeError, match="boundary handoff failed"):
        await adapter.apply_after_persist(
            transition,
            strict_runtime_errors=False,
            reload_settings_view=False,
        )

    assert events == ["self_replace"]


@pytest.mark.asyncio
async def test_peer_refresh_recomputes_activation_after_eula_transition() -> None:
    previous = AppSettingsVNext()
    target = replace(
        previous,
        state=replace(
            previous.state,
            peer_translation=replace(
                previous.state.peer_translation,
                eula_accepted=True,
            ),
        ),
    )
    events: list[str] = []
    activation_calls: list[tuple[bool, bool]] = []
    adapter = object.__new__(SettingsRuntimeEffectsAdapter)
    adapter._desktop_overlay = _AsyncNoop()
    adapter._clipboard = SimpleNamespace(strict_runtime_errors=False, sync=_AsyncNoop().sync)
    adapter._provisioning = _AsyncNoop()
    adapter._clear_local_pending = lambda: None
    adapter._pipeline = SimpleNamespace(
        translation_runtime_configuration=None,
        peer_translation_channel=object(),
    )

    def activation_requested(*, intent_enabled: bool, eula_accepted: bool) -> bool:
        activation_calls.append((intent_enabled, eula_accepted))
        return intent_enabled and eula_accepted

    async def refresh_peer() -> None:
        events.append("peer_refresh")

    adapter._peer = SimpleNamespace(
        owner=SimpleNamespace(
            effective_enabled=lambda _state=None: True,
            activation_requested=activation_requested,
            refresh_runtime=refresh_peer,
        ),
        state_for=lambda _channel: object(),
    )
    adapter._settings = SimpleNamespace(
        authoritative=False,
        overlay_enabled=lambda: False,
        peer_translation_enabled=lambda: True,
    )
    adapter._gpu = SimpleNamespace(
        state_provider=lambda: SimpleNamespace(selected_provider_requires_model=False)
    )
    adapter._overlay = _Overlay()
    adapter._vrc_mic_sync = _VrcMic()
    adapter._runtime_logging = SimpleNamespace(
        emit_detailed=lambda *_args, **_kwargs: None,
        emit_basic=lambda *_args, **_kwargs: None,
    )
    adapter._presentation = SimpleNamespace(current_locale=lambda: "en")
    adapter._projection = SimpleNamespace()
    adapter._rebuild_managed_gemma = lambda: None
    adapter._canonical_settings = lambda value: value
    adapter._sync_effective_translation_flags = lambda _settings: events.append("peer_flags")
    adapter._peer_runtime_converged = lambda _settings: None
    adapter._sync_signatures = lambda _settings: events.append("signature_cache_write")

    async def no_self_replace(_smooth: bool) -> None:
        events.append("self_replace")

    adapter._replace_self_stt = no_self_replace
    transition = SettingsRuntimeTransition(
        settings=target,
        previous_settings=previous,
        previous_locale="en",
        previous_overlay_enabled=False,
        previous_self_signature=None,
        previous_peer_signature=build_peer_stt_runtime_signature_from_vnext(target),
        previous_peer_translation_enabled=True,
        previous_peer_activation_requested=False,
        source_language_changed=False,
        target_language_changed=False,
        effective_peer_source_changed=False,
        effective_peer_target_changed=False,
        peer_source_language_changed=False,
        peer_target_language_changed=False,
        peer_source_mode_changed=False,
        desktop_runtime_controls=(),
    )

    await adapter.apply_after_persist(
        transition,
        strict_runtime_errors=False,
        reload_settings_view=False,
    )

    assert activation_calls == [(True, True)]
    assert events == ["peer_refresh", "peer_flags", "signature_cache_write"]


@pytest.mark.asyncio
async def test_stale_active_peer_with_matching_cache_retries_without_caching_non_convergence() -> (
    None
):
    target = AppSettingsVNext()
    events: list[str] = []
    adapter = object.__new__(SettingsRuntimeEffectsAdapter)
    adapter._desktop_overlay = _AsyncNoop()
    adapter._clipboard = SimpleNamespace(strict_runtime_errors=False, sync=_AsyncNoop().sync)
    adapter._provisioning = _AsyncNoop()
    adapter._clear_local_pending = lambda: None
    adapter._pipeline = SimpleNamespace(
        translation_runtime_configuration=None,
        peer_translation_channel=object(),
    )

    async def refresh_peer() -> None:
        events.append("peer_refresh")

    adapter._peer = SimpleNamespace(
        owner=SimpleNamespace(
            effective_enabled=lambda _state=None: False,
            activation_requested=lambda **_kwargs: True,
            refresh_runtime=refresh_peer,
        ),
        state_for=lambda _channel: object(),
    )
    adapter._settings = SimpleNamespace(
        overlay_enabled=lambda: False,
        peer_translation_enabled=lambda: True,
    )
    adapter._gpu = SimpleNamespace(
        state_provider=lambda: SimpleNamespace(selected_provider_requires_model=False)
    )
    adapter._overlay = _Overlay()
    adapter._vrc_mic_sync = _VrcMic()
    adapter._runtime_logging = SimpleNamespace(
        emit_detailed=lambda *_args, **_kwargs: None,
        emit_basic=lambda *_args, **_kwargs: None,
    )
    adapter._presentation = SimpleNamespace(current_locale=lambda: "en")
    adapter._projection = SimpleNamespace()
    adapter._rebuild_managed_gemma = lambda: None
    adapter._canonical_settings = lambda value: value
    adapter._sync_effective_translation_flags = lambda _settings: events.append("peer_flags")
    adapter._sync_signatures = lambda _settings: events.append("signature_cache_write")
    adapter._peer_runtime_converged = lambda _settings: False

    async def no_self_replace(_smooth: bool) -> None:
        events.append("self_replace")

    adapter._replace_self_stt = no_self_replace
    transition = SettingsRuntimeTransition(
        settings=target,
        previous_settings=target,
        previous_locale="en",
        previous_overlay_enabled=False,
        previous_self_signature=None,
        previous_peer_signature=build_peer_stt_runtime_signature_from_vnext(target),
        previous_peer_translation_enabled=True,
        previous_peer_activation_requested=True,
        source_language_changed=False,
        target_language_changed=False,
        effective_peer_source_changed=False,
        effective_peer_target_changed=False,
        peer_source_language_changed=False,
        peer_target_language_changed=False,
        peer_source_mode_changed=False,
        desktop_runtime_controls=(),
    )

    with pytest.raises(RuntimeError, match="Peer STT runtime did not converge"):
        await adapter.apply_after_persist(
            transition,
            strict_runtime_errors=False,
            reload_settings_view=False,
        )

    assert events == ["peer_refresh", "peer_flags"]
