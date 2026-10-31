import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.peer_application_state import PeerApplicationSettings
from puripuly_heart.app.adapters.peer_capture.peer_capture_admission import (
    PeerCaptureAdmissionAdapter,
)
from puripuly_heart.app.adapters.ui_runtime import UiPeerCaptureRuntimeAdapter
from puripuly_heart.app.ports.settings_view import LoopbackCaptureSettingsIntent
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.app.services.settings.settings_transaction_result import (
    SettingsTransactionResultOwner,
)
from puripuly_heart.app.wiring.wiring_peer_application import compose_peer_application
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.local_asr import retired_cpu_assets
from puripuly_heart.core.peer_capture import (
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureProviderStatus,
)
from tests.core.runtime.test_peer_capture_session import make_owner
from tests.helpers.ui_application import compose_test_ui_application_boundary


def capture_application(tmp_path, monkeypatch, *, demand=True, ready=False):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setattr(
        retired_cpu_assets, "default_local_stt_model_root", lambda: tmp_path / "models"
    )
    settings = compose_settings_owner(tmp_path / "settings.json")
    initial = AppSettingsVNext()
    initial = replace(
        initial,
        state=replace(
            initial.state,
            peer_translation=replace(initial.state.peer_translation, eula_accepted=True),
        ),
    )
    settings.persistence.persist(settings.path, initial)
    settings.start()
    settings.set_peer_translation_enabled(demand)
    readiness = SimpleNamespace(available=ready, calls=0)

    async def ensure_ready(*_args):
        readiness.calls += 1
        return readiness.available

    runtime, _, _, provider, sources, _ = make_owner(
        admission=PeerCaptureAdmissionAdapter(
            runtime_available=lambda: True,
            ensure_local_ready=ensure_ready,
        ),
    )
    local_runtime = SimpleNamespace(
        snapshot=SimpleNamespace(
            channel_for=lambda _channel: SimpleNamespace(provider_id=provider.provider_id)
        )
    )
    overlay = SimpleNamespace(
        snapshot=SimpleNamespace(state="connected"),
        record_lifecycle_trace=lambda *_args, **_kwargs: None,
    )
    results = SettingsTransactionResultOwner()
    presented = []
    peer = compose_peer_application(
        settings_provider=lambda: PeerApplicationSettings(
            intent_enabled=settings.peer_translation_enabled(),
            eula_accepted=settings.require_canonical().state.peer_translation.eula_accepted,
            overlay_intent_enabled=False,
            provider_id=settings.require_canonical().intent.peer_stt.provider,
        ),
        settings_owner=settings,
        canonical_settings=settings.require_canonical,
        peer_intent_sink=settings.set_peer_translation_enabled,
        overlay_intent_sink=settings.set_overlay_enabled,
        runtime_provider=lambda: local_runtime,
        translation_runtime_configuration_provider=lambda: None,
        overlay_provider=lambda: overlay,
        ingress_frozen=lambda: False,
        ensure_local_ready=ensure_ready,
        clear_cpu_pending=lambda: None,
        clear_gpu_pending=lambda: None,
        clear_switched_pending=lambda: None,
        sync_local_notice=lambda: None,
        presentation_changed=lambda: None,
        disclosure_sink=lambda: None,
        superseded_sink=lambda: None,
        localize=lambda key: key,
        settings_presentation_sink=presented.append,
        log_basic=lambda *_args, **_kwargs: None,
        log_diagnostic=lambda *_args, **_kwargs: None,
        stop_audio_test=lambda: asyncio.sleep(0),
        transaction_result_sink=results.set,
    )
    peer.owner.bind_runtime(runtime)
    peer.target.devices = SimpleNamespace(names=lambda: ("Synthetic A", "Synthetic B"))
    peer.target.processes = SimpleNamespace(candidates=lambda: ())
    app = compose_test_ui_application_boundary(SimpleNamespace(settings=initial))
    app._peer_capture = UiPeerCaptureRuntimeAdapter(peer=peer, overlay=overlay)
    app._settings_results = results
    control = app.control()
    control.settings = settings
    control.results = results
    control.peer = lambda: peer.owner
    settings.observe_commits(control._publish_committed)
    return SimpleNamespace(
        app=app,
        control=control,
        settings=settings,
        peer=peer,
        runtime=runtime,
        provider=provider,
        sources=sources,
        readiness=readiness,
        presented=presented,
    )


async def test_pending_capture_target_save_remains_unapplied_until_explicit_application(
    tmp_path, monkeypatch
):
    harness = capture_application(tmp_path, monkeypatch)
    try:
        result = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic A")
        )
        assert result.saved is True
        assert result.applied is False
        assert result.transaction.status == "settings_commit_success_runtime_degraded"
        saved = load_vnext_settings(harness.settings.path).settings
        assert saved.intent.desktop_audio.capture_target.device_name == "Synthetic A"
        assert harness.runtime.snapshot.desired_active is True
        assert harness.runtime.snapshot.effective_active is False
        assert harness.runtime.snapshot.provider_status is PeerCaptureProviderStatus.PENDING
        assert harness.peer.owner.snapshot().effective_enabled is False
        assert harness.sources == []
        assert harness.provider.requests == []
        assert harness.presented

        current = harness.settings.require_committed()
        latest = replace(
            current, intent=replace(current.intent, ui=replace(current.intent.ui, locale="ja"))
        )
        harness.settings.commit_delta(current, latest)
        harness.readiness.available = True
        result = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic A")
        )
        assert result.saved is True
        assert result.applied is True
        assert result.transaction.status == "settings_commit_success_runtime_applied"
        assert harness.peer.owner.snapshot().effective_enabled is True
        assert harness.runtime.snapshot.effective_active is True
        assert harness.runtime.snapshot.capture_target.device_name == "Synthetic A"
        assert harness.settings.peer_translation_enabled() is True
        assert harness.settings.require_committed().intent.ui.locale == "ja"
        assert load_vnext_settings(harness.settings.path).settings.intent.ui.locale == "ja"
    finally:
        await harness.peer.owner.close()
    assert all(source.close_calls == 1 for source in harness.sources)


@pytest.mark.parametrize("demand", [False, True])
async def test_capture_target_reports_applied_for_active_rebinding_or_no_listen_demand(
    tmp_path, monkeypatch, demand
):
    harness = capture_application(tmp_path, monkeypatch, demand=demand, ready=True)
    try:
        await harness.peer.owner.refresh_runtime()
        old_sources = tuple(harness.sources)
        result = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic B")
        )
        assert result.saved is True
        assert result.applied is True
        assert result.transaction.status == "settings_commit_success_runtime_applied"
        assert harness.runtime.snapshot.desired_active is demand
        assert harness.runtime.snapshot.effective_active is demand
        assert harness.peer.owner.snapshot().effective_enabled is demand
        assert harness.settings.peer_translation_enabled() is demand
        saved = load_vnext_settings(harness.settings.path).settings
        assert saved.intent.desktop_audio.capture_target.device_name == "Synthetic B"
        if demand:
            assert old_sources and all(source.close_calls == 1 for source in old_sources)
            assert harness.runtime.snapshot.capture_target.device_name == "Synthetic B"
            assert len(harness.provider.requests) == 2
        else:
            assert harness.sources == []
            assert harness.provider.requests == []
            assert harness.readiness.calls == 0
    finally:
        await harness.peer.owner.close()
    assert all(source.close_calls == 1 for source in harness.sources)


@pytest.mark.parametrize(
    "status", [PeerCaptureProviderMutationStatus.PENDING, PeerCaptureProviderMutationStatus.FAILED]
)
async def test_capture_target_provider_pending_or_failure_preserves_saved_unapplied_intent(
    tmp_path, monkeypatch, status
):
    harness = capture_application(tmp_path, monkeypatch, ready=True)
    harness.provider.replace_result = PeerCaptureProviderMutation(status)
    try:
        result = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic A")
        )
        assert result.saved is True
        assert result.applied is False
        assert result.transaction.status == "settings_commit_success_runtime_degraded"
        assert harness.peer.owner.snapshot().effective_enabled is False
        assert harness.settings.peer_translation_enabled() is True
        saved = load_vnext_settings(harness.settings.path).settings
        assert saved.intent.desktop_audio.capture_target.device_name == "Synthetic A"
    finally:
        await harness.peer.owner.close()


async def test_capture_target_save_failure_does_not_apply_runtime(tmp_path, monkeypatch):
    harness = capture_application(tmp_path, monkeypatch, ready=True)
    original = harness.settings.path.read_bytes()

    def fail_persist(*_args, **_kwargs):
        raise OSError("controlled storage failure")

    monkeypatch.setattr(harness.settings.persistence, "persist", fail_persist)
    try:
        result = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic A")
        )
        assert result.saved is False
        assert result.applied is False
        assert result.transaction.status == "settings_commit_failed"
        assert harness.settings.path.read_bytes() == original
        assert harness.sources == []
        assert harness.provider.requests == []
        assert harness.settings.peer_translation_enabled() is True
    finally:
        await harness.peer.owner.close()
