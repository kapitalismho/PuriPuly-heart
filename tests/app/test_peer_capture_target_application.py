import asyncio
import errno
import logging
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
from puripuly_heart.core import runtime_logging
from puripuly_heart.core.local_asr import retired_cpu_assets
from puripuly_heart.core.peer_capture import (
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureProviderStatus,
    PeerCaptureSessionState,
)
from puripuly_heart.core.runtime import peer_channel
from tests.core.runtime.test_peer_capture_session import FakeSource, make_owner, wait_until
from tests.helpers.ui_application import compose_test_ui_application_boundary


def capture_application(
    tmp_path,
    monkeypatch,
    *,
    demand=True,
    ready=False,
    provider_id="local_cpu_auto",
    **owner_options,
):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setattr(
        retired_cpu_assets, "default_local_stt_model_root", lambda: tmp_path / "models"
    )
    settings = compose_settings_owner(tmp_path / "settings.json")
    initial = AppSettingsVNext()
    initial = replace(
        initial,
        intent=replace(
            initial.intent, peer_stt=replace(initial.intent.peer_stt, provider=provider_id)
        ),
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
        **owner_options,
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
    control.pipeline = SimpleNamespace(current=SimpleNamespace(peer_capture=runtime))
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
        results=results,
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


@pytest.fixture
def peer_diagnostic_history(tmp_path, monkeypatch):
    logger = logging.getLogger(f"test.peer-recovery.{tmp_path.name}")
    logger.propagate = False
    sinks = runtime_logging.configure_main_logging(root_logger=logger, log_dir=tmp_path / "logs")
    service = runtime_logging.SessionRuntimeLoggingService(root_logger=logger, sinks=sinks)
    service.configure_diagnostics(instance_id="peer-recovery")
    monkeypatch.setattr(peer_channel, "logger", logger)
    try:
        yield service
    finally:
        service.close()
        sinks.close(force=True)
        logger.handlers.clear()


async def assert_applied_capture_target(harness, result, target):
    assert result.saved is True
    assert result.applied is True
    assert result.transaction.status == "settings_commit_success_runtime_applied"
    config = harness.peer.owner.config_factory()
    snapshot = harness.runtime.snapshot
    assert snapshot.state is PeerCaptureSessionState.RUNNING
    assert snapshot.provider_status is PeerCaptureProviderStatus.READY
    assert snapshot.desired_active and snapshot.effective_active
    assert snapshot.provider_id == config.provider_id
    assert snapshot.capture_target == config.capture_target
    assert snapshot.capture_target.device_name == target
    assert snapshot.runtime_signature == config.runtime_signature
    assert snapshot.failure_reason is None
    assert snapshot.failure_diagnostics is None
    assert harness.peer.owner.capture_runtime_convergence(config) is True
    assert harness.peer.owner.snapshot().effective_enabled is True
    channel = (await harness.control.query("capture.channels", {}))["channels"]["peer"]
    assert channel["effective_active"] is True
    assert channel["failure_reason"] is None
    assert channel["capture"]["failure_diagnostics"] is None
    saved = load_vnext_settings(harness.settings.path).settings
    assert saved.intent.desktop_audio.capture_target.device_name == target
    assert saved.intent.peer_stt.provider == config.provider_id
    assert saved.state.peer_translation.eula_accepted is True
    assert harness.settings.peer_translation_enabled() is True


@pytest.mark.parametrize("provider_id", ["local_cpu_auto", "soniox"])
@pytest.mark.parametrize("failure_kind", ["terminal", "source_open", "before_adoption"])
async def test_capture_recovery_reconciles_receipt_without_reprojecting_failure_history(
    tmp_path, monkeypatch, peer_diagnostic_history, provider_id, failure_kind
):
    diagnostics = []
    sources = []
    fail_open = failure_kind == "source_open"
    forbidden = "private_source_provider_exception_body"

    def create_source(_config, _target):
        if fail_open:
            raise PermissionError(errno.EACCES, forbidden)
        source = FakeSource()
        sources.append(source)
        return source

    harness = capture_application(
        tmp_path,
        monkeypatch,
        ready=True,
        provider_id=provider_id,
        sources=sources,
        source_factory=create_source,
        diagnostics=diagnostics,
    )
    harness.app._runtime_logging = peer_diagnostic_history
    if failure_kind == "before_adoption":
        harness.provider.replace_terminal_error = PermissionError(errno.EACCES, forbidden)
    try:
        initial = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic A")
        )
        if failure_kind == "terminal":
            await assert_applied_capture_target(harness, initial, "Synthetic A")
            await harness.provider.terminal_handlers[-1](PermissionError(errno.EACCES, forbidden))
        else:
            assert initial.saved and not initial.applied
        failed = harness.runtime.snapshot
        assert failed.state is PeerCaptureSessionState.FAULTED
        cause_field = "cause_class_1" if failure_kind == "source_open" else "exception_class"
        assert failed.failure_diagnostics.fields[cause_field] == "PermissionError"
        assert failed.failure_diagnostics.fields["errno"] == errno.EACCES
        failure_id = failed.failure_diagnostics.diagnostic_id
        failure_history = peer_diagnostic_history.diagnostic_history({"diagnostic_id": failure_id})
        original_record = failure_history["records"][0]
        assert original_record["generation"] == failed.failure_diagnostics.generation
        assert original_record["fields"]["errno"] == errno.EACCES
        assert original_record["fields"][cause_field] == "PermissionError"
        assert forbidden not in repr(failure_history)
        fail_open = False
        harness.provider.replace_terminal_error = None

        recovered = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic B")
        )
        await assert_applied_capture_target(harness, recovered, "Synthetic B")
        assert peer_diagnostic_history.diagnostic_history({"diagnostic_id": failure_id})[
            "records"
        ] == [original_record]
        history = peer_diagnostic_history.diagnostic_history({})["records"]
        generation = harness.runtime.snapshot.generation
        counts = (len(sources), len(harness.provider.requests), harness.provider.start_calls)
        unchanged = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic B")
        )
        await assert_applied_capture_target(harness, unchanged, "Synthetic B")
        assert harness.runtime.snapshot.generation == generation
        assert counts == (
            len(sources),
            len(harness.provider.requests),
            harness.provider.start_calls,
        )
        assert peer_diagnostic_history.diagnostic_history({})["records"] == history
        failures = [item for item in diagnostics if item.event.value == "failure"]
        assert len(failures) == 1
        assert failures[0].failure_diagnostics.diagnostic_id == failure_id

        await harness.provider.terminal_handlers[-1](
            ConnectionResetError(errno.ECONNRESET, forbidden)
        )
        current = harness.runtime.snapshot
        assert current.state is PeerCaptureSessionState.FAULTED
        assert current.failure_reason.value == "provider_failed"
        assert current.failure_diagnostics.diagnostic_id != failure_id
        assert (
            harness.peer.owner.capture_runtime_convergence(harness.peer.owner.config_factory())
            is False
        )
        channel = (await harness.control.query("capture.channels", {}))["channels"]["peer"]
        assert channel["failure_reason"] == "provider_failed"
        assert (
            channel["capture"]["failure_diagnostics"]["diagnostic_id"]
            == current.failure_diagnostics.diagnostic_id
        )
        current_history = peer_diagnostic_history.diagnostic_history(
            {"diagnostic_id": current.failure_diagnostics.diagnostic_id}
        )["records"]
        assert len(current_history) == 1
        assert current_history[0]["generation"] == generation
        assert current_history[0]["fields"]["exception_class"] == "ConnectionResetError"
        assert peer_diagnostic_history.diagnostic_history({"diagnostic_id": failure_id})[
            "records"
        ] == [original_record]
        assert forbidden not in repr(peer_diagnostic_history.diagnostic_history({}))
    finally:
        await harness.peer.owner.close()
    assert sources and all(source.close_calls == 1 for source in sources)
    assert harness.runtime.loop_task is None
    assert harness.runtime.snapshot.cleanup_debt == 0


@pytest.mark.parametrize(
    "status",
    [
        PeerCaptureProviderMutationStatus.PENDING,
        PeerCaptureProviderMutationStatus.FAILED,
        PeerCaptureProviderMutationStatus.SUPERSEDED,
    ],
)
async def test_unsuccessful_recovery_remains_degraded_until_ready(tmp_path, monkeypatch, status):
    harness = capture_application(tmp_path, monkeypatch, ready=True)
    try:
        await harness.app.apply_settings_change(LoopbackCaptureSettingsIntent("device:Synthetic A"))
        await harness.provider.terminal_handlers[-1](RuntimeError("prior failure"))
        prior_failure = harness.runtime.snapshot.failure_diagnostics
        harness.provider.replace_result = PeerCaptureProviderMutation(status, reason="controlled")
        result = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic B")
        )
        assert result.saved and not result.applied
        assert result.transaction.status == "settings_commit_success_runtime_degraded"
        assert harness.runtime.snapshot.provider_status is not PeerCaptureProviderStatus.READY
        assert (
            harness.peer.owner.capture_runtime_convergence(harness.peer.owner.config_factory())
            is False
        )
        if status is PeerCaptureProviderMutationStatus.PENDING:
            assert harness.runtime.snapshot.state is PeerCaptureSessionState.PROVIDER_PENDING
            assert harness.runtime.snapshot.failure_diagnostics == prior_failure
        else:
            assert harness.runtime.snapshot.state is PeerCaptureSessionState.FAULTED
            assert harness.runtime.snapshot.failure_diagnostics != prior_failure
        saved = load_vnext_settings(harness.settings.path).settings
        assert saved.intent.desktop_audio.capture_target.device_name == "Synthetic B"
        assert saved.state.peer_translation.eula_accepted is True
        assert harness.settings.peer_translation_enabled() is True
        harness.provider.replace_result = PeerCaptureProviderMutation(
            PeerCaptureProviderMutationStatus.APPLIED
        )
        recovered = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic B")
        )
        await assert_applied_capture_target(harness, recovered, "Synthetic B")
    finally:
        await harness.peer.owner.close()
    assert all(source.close_calls == 1 for source in harness.sources)


@pytest.mark.parametrize("interrupt", ["cancelled", "superseded"])
async def test_interrupted_recovery_does_not_clear_failure_before_readiness(
    tmp_path, monkeypatch, interrupt
):
    harness = capture_application(tmp_path, monkeypatch, ready=True)
    recovery = None
    try:
        await harness.app.apply_settings_change(LoopbackCaptureSettingsIntent("device:Synthetic A"))
        await harness.provider.terminal_handlers[-1](RuntimeError("prior failure"))
        prior_failure = harness.runtime.snapshot.failure_diagnostics
        harness.provider.start_gate = asyncio.Event()
        recovery = asyncio.create_task(
            harness.app.apply_settings_change(LoopbackCaptureSettingsIntent("device:Synthetic B"))
        )
        await wait_until(lambda: harness.provider.start_calls == 2)
        assert harness.runtime.snapshot.state is PeerCaptureSessionState.STARTING
        assert harness.runtime.snapshot.failure_diagnostics == prior_failure
        assert (
            harness.peer.owner.capture_runtime_convergence(harness.peer.owner.config_factory())
            is False
        )
        if interrupt == "cancelled":
            recovery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recovery
        else:
            await harness.runtime.apply_intent(harness.peer.owner.config_factory(), enabled=False)
            result = await recovery
            assert result.saved and not result.applied
        assert harness.results.current.status == "settings_commit_success_runtime_degraded"
        assert harness.runtime.snapshot.failure_diagnostics == prior_failure
        assert (
            harness.peer.owner.capture_runtime_convergence(harness.peer.owner.config_factory())
            is False
        )
        harness.provider.start_gate.set()
        harness.provider.start_gate = None
        recovered = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic B")
        )
        await assert_applied_capture_target(harness, recovered, "Synthetic B")
    finally:
        if recovery is not None and not recovery.done():
            recovery.cancel()
            await asyncio.gather(recovery, return_exceptions=True)
        await harness.peer.owner.close()
    assert all(source.close_calls == 1 for source in harness.sources)


async def test_retired_callback_during_gated_target_rebind_cannot_fault_replacement(
    tmp_path, monkeypatch
):
    diagnostics = []
    harness = capture_application(tmp_path, monkeypatch, ready=True, diagnostics=diagnostics)
    replacement = late_failure = None
    try:
        initial = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic A")
        )
        await assert_applied_capture_target(harness, initial, "Synthetic A")
        retired_terminal = harness.provider.terminal_handlers[-1]
        harness.provider.replace_gate = asyncio.Event()
        replacement = asyncio.create_task(
            harness.app.apply_settings_change(LoopbackCaptureSettingsIntent("device:Synthetic B"))
        )
        await wait_until(lambda: len(harness.provider.requests) == 2)
        late_failure = asyncio.create_task(retired_terminal(RuntimeError("retired failure")))
        await asyncio.sleep(0)
        assert not replacement.done() and not late_failure.done()
        assert (
            harness.peer.owner.capture_runtime_convergence(harness.peer.owner.config_factory())
            is False
        )
        harness.provider.replace_gate.set()
        result, _ = await asyncio.gather(replacement, late_failure)
        await assert_applied_capture_target(harness, result, "Synthetic B")
        assert not any(item.event.value == "failure" for item in diagnostics)
        harness.provider.replace_gate = None
        rebound = await harness.app.apply_settings_change(
            LoopbackCaptureSettingsIntent("device:Synthetic A")
        )
        await assert_applied_capture_target(harness, rebound, "Synthetic A")
    finally:
        if harness.provider.replace_gate is not None:
            harness.provider.replace_gate.set()
        await asyncio.gather(
            *(task for task in (replacement, late_failure) if task is not None),
            return_exceptions=True,
        )
        await harness.peer.owner.close()
    assert all(source.close_calls == 1 for source in harness.sources)
