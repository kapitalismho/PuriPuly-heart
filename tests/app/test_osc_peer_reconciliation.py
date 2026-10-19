from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.peer_application_state import PeerApplicationSettings
from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.osc.state_publisher import state_from_settings
from puripuly_heart.app.services.provider.provider_runtime_apply import (
    ProviderRuntimeConvergenceError,
)
from puripuly_heart.app.wiring.wiring_audio_test import AudioTestCaptureControl
from puripuly_heart.app.wiring.wiring_peer_application import compose_peer_application
from puripuly_heart.config.settings_vnext.facade import save_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.peer_capture import (
    PeerCaptureSessionState,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from tests.app.test_osc_reconciliation import (
    PREFIX,
    RUNTIME_CONTROLS,
    TRANS,
    MemoryReceiver,
    MemorySender,
    RuntimeHarness,
)
from tests.core.runtime.test_peer_capture_session import (
    FailingCloseSource,
    FakeAdmission,
    make_config,
    make_owner,
)
from tests.helpers.ui_application import compose_test_ui_application_boundary

LISTEN = "PuriPuly_Listen"


class PeerHarness:
    def __init__(self, tmp_path: Path, *, source=None, admission=None):
        seed = AppSettingsVNext()
        seed = replace(
            seed,
            intent=replace(seed.intent, peer_stt=replace(seed.intent.peer_stt, provider="soniox")),
            state=replace(
                seed.state,
                peer_translation=replace(seed.state.peer_translation, eula_accepted=True),
            ),
        )
        path = tmp_path / "peer-settings.json"
        assert save_vnext_settings(path, seed).ok
        self.settings_owner = compose_settings_owner(path)
        self.settings_owner.start()
        self.settings_owner.set_overlay_enabled(True)
        self.settings = self.settings_owner.require_canonical()
        self.sender = MemorySender()
        self.receiver = MemoryReceiver()
        self.commands = []
        self.errors = []
        self.presentations = []
        self.settlements = []
        self.runtime_observations = []
        self.integration = None
        self.chatbox_source = False
        self.core = None
        self.peer = None
        overlay = SimpleNamespace(
            snapshot=SimpleNamespace(state="connected"),
            record_lifecycle_trace=lambda *_args, **_kwargs: None,
        )
        provider_runtime = SimpleNamespace(
            snapshot=SimpleNamespace(
                channel_for=lambda _channel: SimpleNamespace(provider_id="soniox")
            )
        )
        self.peer = compose_peer_application(
            settings_provider=lambda: PeerApplicationSettings(
                intent_enabled=self.settings_owner.peer_translation_enabled(),
                eula_accepted=True,
                overlay_intent_enabled=self.settings_owner.overlay_enabled(),
                provider_id="soniox",
            ),
            settings_owner=self.settings_owner,
            canonical_settings=self.settings_owner.require_canonical,
            peer_intent_sink=self.settings_owner.set_peer_translation_enabled,
            overlay_intent_sink=self.settings_owner.set_overlay_enabled,
            runtime_provider=lambda: provider_runtime,
            translation_runtime_configuration_provider=lambda: None,
            overlay_provider=lambda: overlay,
            ingress_frozen=lambda: False,
            persist_manual_fallback=lambda: True,
            ensure_local_ready=self.ready,
            clear_cpu_pending=lambda: None,
            clear_gpu_pending=lambda: None,
            clear_switched_pending=lambda: None,
            sync_local_notice=lambda: None,
            presentation_changed=lambda: self.presentations.append(self.canonical().peer_capture),
            mutation_settled=self.settled,
            disclosure_sink=lambda: None,
            superseded_sink=lambda: None,
            localize=lambda key: key,
            settings_presentation_sink=lambda _state: None,
            log_basic=lambda *_args, **_kwargs: None,
            log_diagnostic=lambda *_args, **_kwargs: None,
            stop_audio_test=self.noop,
        )
        harness = self

        class StateSink:
            def append(self, snapshot):
                harness.peer.owner.on_runtime_state_changed(snapshot)
                harness.publish()
                harness.runtime_observations.append(
                    (
                        snapshot.state,
                        harness.peer.owner.pending_mutations(),
                        list(harness.sender.messages),
                    )
                )

        (
            self.core,
            self.admission,
            self.resolver,
            self.provider,
            self.sources,
            _sink,
        ) = make_owner(
            admission=admission,
            source_factory=(lambda _config, _target: source) if source is not None else None,
            state_changes=StateSink(),
        )
        self.peer.owner.bind_runtime(self.core)
        self.application = compose_test_ui_application_boundary(
            self,
            osc_state_publisher=self.publish,
            runtime_control_authority=lambda parameter, task: (
                self.peer.owner.mutation_authority(task) if parameter == LISTEN else None
            ),
            runtime_control_activity=lambda: tuple(
                (LISTEN, value) for value in self.peer.owner.pending_mutations()
            ),
        )
        self.integration = OscControlIntegrationOwner(
            receiver_owner=self.receiver,
            settings_provider=lambda: self.settings,
            apply_settings=self.noop,
            application_provider=lambda: self.application,
            sender_provider=lambda: self.sender,
            state_provider=self.canonical,
            language_state_provider=lambda: ("ko", "en", "en", "ko"),
            translation_model_normalizer=materialize_canonical_translation_settings,
            error_sink=self.errors.append,
        )
        self.audio_capture = AudioTestCaptureControl(
            self_capture_provider=lambda: None,
            local_pending_provider=lambda: False,
            set_self_enabled=self.noop,
            peer_owner_provider=lambda: self.peer.owner,
            peer_capture_provider=lambda: self.core,
            set_peer_enabled=self.peer.owner.set_enabled,
        )

    async def noop(self, *_args, **_kwargs):
        return None

    async def ready(self, _generation):
        return True

    def release_audio_test_capture_restore(self, _channel):
        pass

    def canonical(self):
        state = state_from_settings(
            self.settings,
            peer_capture=self.peer.owner.snapshot().effective_enabled if self.peer else False,
            captions=self.settings_owner.overlay_enabled(),
        )
        return replace(state, chatbox_source=self.chatbox_source)

    def publish(self):
        if self.integration is not None:
            self.integration.publish_delta()

    def settled(self):
        self.settlements.append(self.peer.owner.pending_mutations())
        self.publish()

    async def set_peer_translation_enabled(self, enabled):
        self.commands.append(enabled)
        return await self.peer.owner.set_enabled(enabled)

    async def start(self):
        await self.integration.configure_connection(
            mode="manual", send_port=19000, receive_port=19001
        )
        for parameter, value in self.sender.messages:
            assert not self.receiver.handlers["control_packet_handler"](
                PREFIX + parameter, (value,)
            )
        self.sender.messages.clear()

    async def dispatch(self, value):
        return await self.integration.router.dispatch_packet(PREFIX + LISTEN, value)

    async def refresh(self, configured):
        if configured:
            await self.peer.owner.refresh_runtime_with_config(
                self.peer.owner.config_factory(), state=self.peer.owner.state_provider()
            )
        else:
            await self.peer.owner.refresh_runtime()

    async def close(self):
        if self.admission.gate is not None:
            self.admission.gate.set()
        await self.integration.close()
        await self.core.close()


async def wait_for_admission(h):
    for _ in range(100):
        if h.admission.calls:
            return
        await asyncio.sleep(0)
    raise AssertionError("real peer capture admission did not enter")


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", (False, True))
async def test_real_peer_background_refresh_admits_off_before_preparation_finishes(
    tmp_path, configured
):
    h = PeerHarness(tmp_path)
    await h.start()
    h.settings_owner.set_peer_translation_enabled(True)
    h.admission.gate = asyncio.Event()
    refreshing = asyncio.create_task(h.refresh(configured))
    await wait_for_admission(h)
    try:
        assert not refreshing.done()
        assert h.canonical().peer_capture is False
        assert (LISTEN, True) in h.integration._pending_controls()
        assert h.sender.messages == []
        stopped = await asyncio.wait_for(h.dispatch(False), 2)
        result = (await asyncio.gather(refreshing, return_exceptions=True))[0]
        assert isinstance(result, ProviderRuntimeConvergenceError)
        assert stopped.applied
        assert h.commands == [False]
        assert h.settings_owner.peer_translation_enabled() is False
        assert h.canonical().peer_capture is False
        assert h.core.snapshot.state is PeerCaptureSessionState.STOPPED
        assert h.sources == []
        assert h.sender.messages == []
        assert h.integration._pending_controls() == ()
        assert (await h.dispatch(False)).error == "echo_suppressed"
        assert h.commands == [False]
    finally:
        h.admission.gate.set()
        await asyncio.gather(refreshing, return_exceptions=True)
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", (False, True))
async def test_real_peer_background_refresh_publishes_settled_state_without_unrelated_delta(
    tmp_path, configured
):
    h = PeerHarness(tmp_path)
    await h.start()
    h.settings_owner.set_peer_translation_enabled(True)
    h.admission.gate = asyncio.Event()
    refreshing = asyncio.create_task(h.refresh(configured))
    await wait_for_admission(h)
    try:
        assert h.sender.messages == []
        h.chatbox_source = True
        h.publish()
        assert h.sender.messages == [("PuriPuly_ChatboxSource", True)]
        h.admission.gate.set()
        await refreshing
        assert h.canonical().peer_capture is True
        assert h.core.snapshot.state is PeerCaptureSessionState.RUNNING
        assert h.sender.messages == [("PuriPuly_ChatboxSource", True), (LISTEN, True)]
        assert h.integration._pending_controls() == ()
        assert (await h.dispatch(True)).error == "echo_suppressed"
        assert h.commands == []
    finally:
        h.admission.gate.set()
        await asyncio.gather(refreshing, return_exceptions=True)
        await h.close()


@pytest.mark.asyncio
async def test_real_peer_direct_owner_settlement_is_independent_of_overlay_presentation(tmp_path):
    h = PeerHarness(tmp_path)
    await h.start()
    try:
        await h.peer.owner.set_enabled(True)
        assert h.canonical().peer_capture is True
        assert h.core.snapshot.state is PeerCaptureSessionState.RUNNING
        assert h.sender.messages == [(LISTEN, True)]
        assert h.settlements == [()]
        assert h.presentations
        assert h.integration._pending_controls() == ()
        assert all(
            not any(name == LISTEN for name, _value in sent)
            for _state, pending, sent in h.runtime_observations
            if pending
        )
        assert (await h.dispatch(True)).error == "echo_suppressed"
        assert h.commands == []
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_real_audio_test_capture_pause_and_restore_publish_without_ui_finally(tmp_path):
    h = PeerHarness(tmp_path)
    await h.start()
    try:
        await h.peer.owner.set_enabled(True)
        h.sender.messages.clear()
        assert h.audio_capture.state("peer").enabled
        await h.audio_capture.set_enabled("peer", False)
        assert h.audio_capture.state("peer").source_open is False
        assert h.canonical().peer_capture is False
        assert h.sender.messages == [(LISTEN, False)]
        assert (await h.dispatch(False)).error == "echo_suppressed"
        await h.audio_capture.set_enabled("peer", True)
        assert h.audio_capture.state("peer").source_open
        assert h.canonical().peer_capture is True
        assert h.sender.messages == [(LISTEN, False), (LISTEN, True)]
        assert (await h.dispatch(True)).error == "echo_suppressed"
        assert h.commands == []
        assert h.integration._pending_controls() == ()
        assert h.sources[0].close_calls == 1
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_real_peer_failed_on_correction_echo_is_fixed_point_and_retry_recovers(tmp_path):
    h = PeerHarness(tmp_path)
    await h.start()
    h.provider.start_error = RuntimeError("controlled provider ingress failure")
    try:
        failed = await h.dispatch(True)
        assert failed.error == "ProviderRuntimeConvergenceError"
        assert h.canonical().peer_capture is False
        assert h.sender.messages == [(LISTEN, False)]
        before = list(h.commands)
        for _ in range(4):
            assert (await h.dispatch(False)).error == "echo_suppressed"
        assert h.commands == before
        assert h.sender.messages == [(LISTEN, False)]
        assert (await h.dispatch(True)).error == "ProviderRuntimeConvergenceError"
        assert h.commands == [True, True]
        assert h.canonical().peer_capture is False
        assert h.core.snapshot.failure_reason is not None
        assert h.sender.messages == [(LISTEN, False), (LISTEN, False)]
        h.provider.start_error = None
        prepared = await h.core.prepare_provider(h.peer.owner.config_factory())
        assert prepared.failure_reason is None
        assert (await h.dispatch(True)).applied
        assert h.canonical().peer_capture is True
        assert h.commands == [True, True, True]
        assert h.sender.messages == [(LISTEN, False), (LISTEN, False)]
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_real_peer_failed_stop_keeps_cleanup_retry_across_full_local_send(tmp_path):
    source = FailingCloseSource()
    h = PeerHarness(tmp_path, source=source)
    await h.start()
    try:
        assert (await h.dispatch(True)).applied
        failed = await h.dispatch(False)
        assert failed.error == "RuntimeError"
        assert h.canonical().peer_capture is False
        assert source.close_calls == 1
        assert h.sender.messages == []
        h.integration._publish_full()
        assert len(h.sender.messages) == 15
        assert (await h.dispatch(False)).applied
        assert source.close_calls == 2
        assert h.commands == [True, False, False]
        assert len(h.sender.messages) == 15
        assert h.integration._pending_controls() == ()
        assert (await h.dispatch(False)).error == "echo_suppressed"
    finally:
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", (False, True))
async def test_real_translation_failed_teardown_retry_survives_local_full_send(malformed):
    h = RuntimeHarness()
    await h.start()
    try:
        assert (await h.dispatch(True)).applied
        h.failure = "off"
        assert (await h.dispatch(False)).error == "RuntimeError"
        assert h.state.translation is False
        assert h.teardowns == 1
        if malformed:
            assert not h.receiver.handlers["control_packet_handler"](PREFIX + TRANS, ("bad",))
        else:
            h.integration._publish_full()
        assert len(h.sender.messages) == 15
        assert h.integration._publisher.last_published[TRANS] is False
        h.failure = None
        assert (await h.dispatch(False)).applied
        assert h.teardowns == 2
        assert len(h.sender.messages) == 15
        assert (await h.dispatch(False)).error == "echo_suppressed"
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("parameter,field,_method", RUNTIME_CONTROLS)
async def test_failed_runtime_request_is_not_recovered_by_successful_local_send(
    parameter, field, _method
):
    h = RuntimeHarness()
    await h.start()
    try:
        h.state = replace(h.state, **{field: True})
        h.reject = True
        assert (await h.dispatch(True, parameter)).error == "application_rejected"
        h.integration._publish_full()
        assert len(h.sender.messages) == 15
        before = len(h.commands)
        h.reject = None
        assert (await h.dispatch(True, parameter)).applied
        assert len(h.commands) == before + 1
        assert len(h.sender.messages) == 15
        assert (await h.dispatch(True, parameter)).error == "echo_suppressed"
    finally:
        await h.close()


class CleanupAdmission(FakeAdmission):
    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()
        self.cancelling = asyncio.Event()
        self.cleanup = asyncio.Event()

    async def admit(self, config):
        try:
            return await super().admit(config)
        except asyncio.CancelledError:
            self.cancelling.set()
            await self.cleanup.wait()
            raise


@pytest.mark.asyncio
async def test_real_peer_off_supersedes_background_authority_but_waits_for_cleanup(tmp_path):
    admission = CleanupAdmission()
    h = PeerHarness(tmp_path, admission=admission)
    await h.start()
    h.settings_owner.set_peer_translation_enabled(True)
    refreshing = asyncio.create_task(h.refresh(False))
    await wait_for_admission(h)
    stopped = asyncio.create_task(h.dispatch(False))
    await admission.cancelling.wait()
    try:
        assert not refreshing.done()
        assert not stopped.done()
        assert h.core.mutation_authority(refreshing) is False
        assert h.commands == [False]
        assert (LISTEN, True) not in h.integration._pending_controls()
        assert (LISTEN, False) in h.integration._pending_controls()
        h.chatbox_source = True
        h.publish()
        assert h.sender.messages == [("PuriPuly_ChatboxSource", True)]
        admission.cleanup.set()
        assert (await stopped).applied
        result = (await asyncio.gather(refreshing, return_exceptions=True))[0]
        assert isinstance(result, ProviderRuntimeConvergenceError)
        assert h.canonical().peer_capture is False
        assert h.sender.messages == [("PuriPuly_ChatboxSource", True)]
        assert h.integration._pending_controls() == ()
        assert h.core.mutation_authority(refreshing) is None
    finally:
        admission.cleanup.set()
        await asyncio.gather(refreshing, stopped, return_exceptions=True)
        await h.close()


@pytest.mark.asyncio
async def test_real_peer_background_cancellation_keeps_barrier_through_cleanup(tmp_path):
    admission = CleanupAdmission()
    h = PeerHarness(tmp_path, admission=admission)
    await h.start()
    h.settings_owner.set_peer_translation_enabled(True)
    refreshing = asyncio.create_task(h.refresh(False))
    await wait_for_admission(h)
    refreshing.cancel()
    await admission.cancelling.wait()
    try:
        assert h.core.mutation_authority(refreshing) is True
        assert (LISTEN, True) in h.integration._pending_controls()
        assert h.sender.messages == []
        admission.cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await refreshing
        assert h.canonical().peer_capture is False
        assert h.settings_owner.peer_translation_enabled() is True
        assert h.sender.messages == []
        assert h.integration._pending_controls() == ()
        assert h.core.mutation_authority(refreshing) is None
        assert h.commands == []
    finally:
        admission.cleanup.set()
        await asyncio.gather(refreshing, return_exceptions=True)
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", (False, True))
async def test_real_peer_process_retry_preparation_observes_off_and_settlement(tmp_path, reverse):
    h = PeerHarness(tmp_path)
    await h.start()
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="generic_executable",
        executable_identity=r"c:\controlled\synthetic.exe",
    )
    config = make_config(target=target)
    h.peer.owner.config_factory = lambda: config
    h.settings_owner.set_peer_translation_enabled(True)
    h.resolver.results = [
        PeerCaptureTargetResolution(PeerCaptureTargetStatus.UNAVAILABLE, reason="no_process")
    ]
    faulted = await h.core.apply_intent(config, enabled=True)
    assert faulted.retry_available
    entered = asyncio.Event()
    release = asyncio.Event()

    async def prepare(_generation):
        entered.set()
        await release.wait()
        return True

    h.peer.owner.ensure_local_ready = prepare
    retrying = asyncio.create_task(h.peer.owner.retry_process_capture())
    await entered.wait()
    try:
        assert h.canonical().peer_capture is False
        assert (LISTEN, True) in h.integration._pending_controls()
        assert h.sender.messages == []
        if reverse:
            assert (await h.dispatch(False)).applied
            assert not retrying.done()
            assert h.peer.owner.mutation_authority(retrying) is False
            assert h.integration._pending_controls() == ()
            release.set()
            assert await retrying is False
            assert h.canonical().peer_capture is False
            assert h.settings_owner.peer_translation_enabled() is False
            assert h.sources == []
            assert h.sender.messages == []
            assert h.commands == [False]
        else:
            release.set()
            assert await retrying is True
            assert h.canonical().peer_capture is True
            assert h.sender.messages == [(LISTEN, True)]
            assert h.commands == []
            assert len(h.sources) == 1
        assert h.integration._pending_controls() == ()
    finally:
        release.set()
        await asyncio.gather(retrying, return_exceptions=True)
        await h.close()


@pytest.mark.asyncio
async def test_root_composition_wires_peer_settlement_to_osc_independently_of_presentation(
    tmp_path, monkeypatch
):
    import io
    import logging

    from puripuly_heart.composition import application_runtime
    from puripuly_heart.composition.headless_application import compose_headless_application
    from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "filters", list(root.filters))
    for handler in root.handlers:
        monkeypatch.setattr(handler, "filters", list(handler.filters))
    calls = []
    composed = []
    real_factory = application_runtime.compose_peer_application

    def peer_factory(**kwargs):
        assert kwargs["mutation_settled"] is not None
        result = real_factory(**kwargs)
        composed.append(result.owner)
        return result

    monkeypatch.setattr(application_runtime, "compose_peer_application", peer_factory)
    monkeypatch.setattr(
        application_runtime,
        "compose_vrc_mic_sync",
        lambda **_kwargs: SimpleNamespace(
            publish_delta=lambda: calls.append("osc"),
            output_snapshot=lambda: {"local_availability": "controlled"},
        ),
    )
    log_path = tmp_path / "composition.log"
    sinks = RuntimeLoggingSinks(
        logging.StreamHandler(io.StringIO()),
        logging.FileHandler(log_path, encoding="utf-8"),
        log_path,
    )
    app = compose_headless_application(
        tmp_path / "composition-settings.json", runtime_logging_sinks=sinks
    )
    try:
        assert app.output_status()["osc"]["local_availability"] == "controlled"
        assert calls == []
        await composed[0].set_enabled(False)
        assert calls == ["osc"]
    finally:
        installed = app._runtime_logging.installed_service
        if installed is not None:
            installed.close()
        sinks.close()
        sinks.stream_handler.close()
