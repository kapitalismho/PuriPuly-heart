from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState
from puripuly_heart.app.services.translation_enable import (
    ManagedTranslationPreparation,
    TranslationEnableOwner,
    TranslationEnableState,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from tests.helpers.ui_application import compose_test_ui_application_boundary

PREFIX = "/avatar/parameters/"
TRANS = "PuriPuly_Trans"


class MemoryReceiver:
    last_enabled = None
    receiver = None
    runtime = None
    effective_port = 19001

    def set_packet_handlers(self, **handlers):
        self.handlers = handlers

    async def configure_control(self, **_kwargs):
        self.receiver = object()

    async def ensure_receiver(self):
        return self.receiver

    async def stop_receiver(self):
        self.receiver = None

    def stop_ingress(self):
        pass

    async def close(self):
        self.receiver = None


class MemorySender:
    def __init__(self):
        self.messages = []
        self.fail_parameter = None

    def set_destination(self, *_args):
        pass

    def send_message(self, address, value):
        if address == PREFIX + str(self.fail_parameter):
            raise OSError("synthetic local send failure")
        self.messages.append((address.removeprefix(PREFIX), value))


class RuntimeHarness:
    def __init__(self):
        self.settings = AppSettingsVNext()
        self.state = OscCanonicalState()
        self.hub = SimpleNamespace(translation_enabled=False, llm=object())
        self.sender = MemorySender()
        self.receiver = MemoryReceiver()
        self.errors = []
        self.commands = []
        self.warmup_entered = asyncio.Event()
        self.warmup_gate = asyncio.Event()
        self.warmup_gate.set()
        self.cleanup_entered = asyncio.Event()
        self.cleanup_gate = asyncio.Event()
        self.cleanup_gate.set()
        self.slow_cancel_gate = None
        self.failure = None
        self.reject = None
        self.collateral = False
        self.teardowns = 0
        self.translation = TranslationEnableOwner(
            state_provider=lambda: TranslationEnableState(
                runtime_available=True,
                translation_enabled=self.state.translation,
                llm_available=True,
                settings_available=True,
                provider_name="synthetic",
                qwen_region=None,
                managed_selected=False,
                managed_china=False,
                managed_local_key_available=False,
                managed_release_service_available=False,
                ingress_frozen=False,
            ),
            managed_prepare=self.prepare,
            founder_route=self.founder,
            pending_sink=lambda _value: None,
            runtime_ensurer=self.ensure_runtime,
            usage_refresh_sink=lambda: None,
            usage_refresh_now=self.noop,
            runtime_sink=self.set_translation_state,
            dashboard_sink=lambda _value: None,
            clear_context=lambda: None,
            warmup=self.warmup,
            teardown=self.teardown,
            message_sink=lambda *_args: None,
            qq_dialog_sink=lambda: None,
            result_sink=lambda _value: None,
            log_basic=lambda _message: None,
            log_diagnostic=lambda _message: None,
            log_error=self.errors.append,
            founder_letter_sink=lambda: None,
            mutation_settled=lambda: self.integration.publish_delta(),
        )
        self.application = compose_test_ui_application_boundary(
            self,
            osc_state_publisher=lambda: self.integration.publish_delta(),
            runtime_control_authority=lambda parameter, task: (
                self.translation.mutation_authority(task) if parameter == TRANS else None
            ),
            runtime_control_activity=lambda: tuple(
                (TRANS, value) for value in self.translation.pending_mutations()
            ),
        )
        self.integration = OscControlIntegrationOwner(
            receiver_owner=self.receiver,
            settings_provider=lambda: self.settings,
            apply_settings=self.apply_settings,
            application_provider=lambda: self.application,
            sender_provider=lambda: self.sender,
            state_provider=lambda: self.state,
            language_state_provider=lambda: ("ko", "en", "en", "ko"),
            translation_model_normalizer=materialize_canonical_translation_settings,
            error_sink=self.errors.append,
        )

    async def start(self):
        await self.integration.configure_connection(
            mode="manual", send_port=19000, receive_port=19001
        )
        for parameter, value in self.sender.messages:
            assert not self.receiver.handlers["control_packet_handler"](
                PREFIX + parameter, (value,)
            )
        self.sender.messages.clear()

    async def prepare(self, _authorization):
        return ManagedTranslationPreparation(ready=True)

    async def founder(self):
        return False

    async def ensure_runtime(self, _mode):
        return True

    async def noop(self):
        pass

    async def apply_settings(self, settings):
        self.settings = settings
        self.integration.publish_delta()

    def set_translation_state(self, enabled):
        self.state = replace(self.state, translation=enabled)
        self.hub.translation_enabled = enabled
        self.integration.publish_delta()

    async def warmup(self):
        self.warmup_entered.set()
        try:
            await self.warmup_gate.wait()
            if self.failure == "on":
                raise RuntimeError("synthetic warmup failure")
        except asyncio.CancelledError:
            self.cleanup_entered.set()
            await self.cleanup_gate.wait()
            if self.slow_cancel_gate is not None:
                await self.slow_cancel_gate.wait()
            raise

    async def teardown(self):
        self.teardowns += 1
        if self.failure == "off":
            raise RuntimeError("synthetic teardown failure")

    async def set_translation_enabled(self, enabled, *, allow_authorization):
        self.commands.append((TRANS, enabled))
        if self.reject is enabled:
            return False
        return await self.translation.set_enabled(enabled, allow_authorization=allow_authorization)

    async def set_stt_enabled(self, enabled):
        return self.set_runtime_bool("PuriPuly_Talk", "self_capture", enabled)

    async def set_peer_translation_enabled(self, enabled):
        return self.set_runtime_bool("PuriPuly_Listen", "peer_capture", enabled)

    async def set_overlay_enabled(self, enabled):
        return self.set_runtime_bool("PuriPuly_Captions", "captions", enabled)

    def set_runtime_bool(self, parameter, field, enabled):
        self.commands.append((parameter, enabled))
        if self.reject is enabled:
            return False
        values = {field: enabled}
        if self.collateral and enabled:
            if field == "self_capture":
                values["self_asr"] = "local_orukeet"
            if field == "peer_capture":
                values.update(peer_asr="local_orukeet", captions=True)
        self.state = replace(self.state, **values)
        self.integration.publish_delta()
        return True

    def release_audio_test_capture_restore(self, _channel):
        pass

    async def dispatch(self, value, parameter=TRANS):
        return await self.integration.router.dispatch_packet(PREFIX + parameter, value)

    async def close(self):
        self.warmup_gate.set()
        self.cleanup_gate.set()
        if self.slow_cancel_gate is not None:
            self.slow_cancel_gate.set()
        await self.integration.close()


async def turns(count=30):
    for _ in range(count):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_successful_off_is_success_without_equal_reflections():
    h = RuntimeHarness()
    await h.start()
    try:
        on = await h.dispatch(True)
        off = await h.dispatch(False)
        assert on.applied and off.applied
        assert h.state.translation is False
        assert h.commands == [(TRANS, True), (TRANS, False)]
        assert h.sender.messages == []
        assert h.errors == []
        h.integration.publish_delta()
        assert h.sender.messages == []
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_off_reaches_owner_before_slow_on_exits():
    h = RuntimeHarness()
    await h.start()
    h.warmup_gate.clear()
    on = asyncio.create_task(h.dispatch(True))
    await h.warmup_entered.wait()
    off = asyncio.create_task(h.dispatch(False))
    try:
        await turns()
        assert off.done()
        assert (await off).applied
        assert h.teardowns == 1
        assert h.state.translation is False
        assert h.sender.messages == []
    finally:
        h.warmup_gate.set()
        await asyncio.gather(on, off, return_exceptions=True)
        await h.close()


RUNTIME_CONTROLS = (
    ("PuriPuly_Talk", "self_capture", "set_stt_enabled"),
    ("PuriPuly_Listen", "peer_capture", "set_peer_translation_enabled"),
    (TRANS, "translation", "set_translation_enabled"),
    ("PuriPuly_Captions", "captions", "set_overlay_enabled"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("parameter,field,method", RUNTIME_CONTROLS)
async def test_runtime_bool_reflection_and_later_ui_and_background_changes(
    parameter, field, method
):
    h = RuntimeHarness()
    await h.start()
    try:
        assert (await h.dispatch(True, parameter)).applied
        assert getattr(h.state, field) is True
        assert h.sender.messages == []
        h.integration.publish_delta()
        assert h.sender.messages == []
        assert h.integration._publisher.last_published[parameter] is False
        assert h.integration._ingress_observations[parameter].value is True

        await getattr(h.application, method)(False)
        assert h.sender.messages == [(parameter, False)]
        assert getattr(h.state, field) is False
        assert (await h.dispatch(False, parameter)).error == "echo_suppressed"

        h.state = replace(h.state, **{field: True})
        h.integration.publish_delta()
        assert h.sender.messages == [(parameter, False), (parameter, True)]
        assert h.integration._publisher.last_published[parameter] is True
        assert h.integration._ingress_observations[parameter].value is True
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("parameter,field,_method", RUNTIME_CONTROLS)
@pytest.mark.parametrize("rejected", (True, False))
async def test_rejected_runtime_bool_corrects_only_divergence_and_explicit_retry(
    parameter, field, _method, rejected
):
    h = RuntimeHarness()
    await h.start()
    try:
        if not rejected:
            assert (await h.dispatch(True, parameter)).applied
        h.reject = rejected
        result = await h.dispatch(rejected, parameter)
        assert result.error == "application_rejected"
        assert getattr(h.state, field) is not rejected
        assert h.sender.messages == [(parameter, not rejected)]
        count = len(h.commands)
        for _ in range(3):
            assert (await h.dispatch(not rejected, parameter)).error == "echo_suppressed"
        assert len(h.commands) == count
        assert h.sender.messages == [(parameter, not rejected)]
        h.reject = None
        assert (await h.dispatch(rejected, parameter)).applied
        assert getattr(h.state, field) is rejected
        assert len(h.commands) == count + 1
        assert h.sender.messages == [(parameter, not rejected)]
    finally:
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parameter,collateral",
    (
        ("PuriPuly_Talk", {"PuriPuly_SelfASR"}),
        ("PuriPuly_Listen", {"PuriPuly_PeerASR", "PuriPuly_Captions"}),
    ),
)
async def test_successful_runtime_input_preserves_collateral_publication(parameter, collateral):
    h = RuntimeHarness()
    h.collateral = True
    await h.start()
    try:
        assert (await h.dispatch(True, parameter)).applied
        assert {name for name, _value in h.sender.messages} == collateral
        assert parameter not in {name for name, _value in h.sender.messages}
        h.integration.publish_delta()
        assert len(h.sender.messages) == len(collateral)
    finally:
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("on", "off"))
async def test_translation_exception_reconciles_live_state_without_blanket_snapshot(failure):
    h = RuntimeHarness()
    await h.start()
    try:
        if failure == "off":
            assert (await h.dispatch(True)).applied
        h.failure = failure
        result = await h.dispatch(failure == "on")
        assert result.error == "RuntimeError"
        assert h.state.translation is False
        assert h.sender.messages == ([(TRANS, False)] if failure == "on" else [])
        h.failure = None
        assert (await h.dispatch(True)).applied
        assert h.state.translation is True
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_cancellation_waits_for_cleanup_and_keeps_unrelated_publication_live():
    h = RuntimeHarness()
    await h.start()
    h.warmup_gate.clear()
    h.cleanup_gate.clear()
    on = asyncio.create_task(h.dispatch(True))
    await h.warmup_entered.wait()
    on.cancel()
    await h.cleanup_entered.wait()
    try:
        h.state = replace(h.state, captions=True)
        h.integration.publish_delta()
        assert h.state.translation is True
        assert h.sender.messages == [("PuriPuly_Captions", True)]
        assert (TRANS, True) in h.integration._pending_controls()
        h.cleanup_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await on
        assert h.state.translation is False
        assert h.sender.messages == [("PuriPuly_Captions", True), (TRANS, False)]
        assert (await h.dispatch(False)).error == "echo_suppressed"
        assert h.commands == [(TRANS, True)]
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_superseded_slow_exit_does_not_hold_authoritative_ui_stop_publication():
    h = RuntimeHarness()
    await h.start()
    h.warmup_gate.clear()
    h.slow_cancel_gate = asyncio.Event()
    on = asyncio.create_task(h.dispatch(True))
    await h.warmup_entered.wait()
    try:
        await h.application.set_translation_enabled(False)
        await h.cleanup_entered.wait()
        assert not on.done()
        assert h.state.translation is False
        assert h.sender.messages == [(TRANS, False)]
        assert h.integration._pending_controls() == ()
        assert (await h.dispatch(False)).error == "echo_suppressed"
        h.slow_cancel_gate.set()
        await asyncio.gather(on, return_exceptions=True)
        assert h.sender.messages == [(TRANS, False)]
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_forced_disable_revokes_warmup_authority_before_task_exits():
    h = RuntimeHarness()
    await h.start()
    h.warmup_gate.clear()
    on = asyncio.create_task(h.dispatch(True))
    await h.warmup_entered.wait()
    try:
        h.translation.disable_for_managed_exhaustion(reopen_founder_letter=False)
        assert not on.done()
        assert h.state.translation is False
        assert h.sender.messages == [(TRANS, False)]
        assert h.integration._pending_controls() == ()
        assert (await h.dispatch(False)).error == "echo_suppressed"
        h.warmup_gate.set()
        result = await on
        assert result.error == "application_rejected"
        assert h.state.translation is False
        assert h.sender.messages == [(TRANS, False)]
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_failed_stop_keeps_older_mutation_authority_through_cancellation_cleanup():
    h = RuntimeHarness()
    await h.start()
    h.warmup_gate.clear()
    h.cleanup_gate.clear()
    on = asyncio.create_task(h.dispatch(True))
    await h.warmup_entered.wait()
    h.reject = False
    try:
        stopped = await h.dispatch(False)
        assert stopped.error == "application_rejected"
        await h.cleanup_entered.wait()
        assert h.state.translation is True
        assert h.sender.messages == []
        assert (TRANS, True) in h.integration._pending_controls()
        h.cleanup_gate.set()
        await asyncio.gather(on, return_exceptions=True)
        assert h.state.translation is False
        assert h.sender.messages == []
        assert h.integration._pending_controls() == ()
        h.reject = None
        before_retry = len(h.commands)
        assert (await h.dispatch(False)).applied
        assert len(h.commands) == before_retry + 1
        assert h.state.translation is False
        assert h.sender.messages == []
        assert (await h.dispatch(False)).error == "echo_suppressed"
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_partial_local_send_preserves_truthful_cache_expectations_and_obligations():
    h = RuntimeHarness()
    await h.start()
    try:
        assert (await h.dispatch(True)).applied
        h.state = replace(h.state, self_capture=True, captions=True)
        h.sender.fail_parameter = "PuriPuly_Captions"
        h.integration.publish_delta()
        assert h.sender.messages == [("PuriPuly_Talk", True)]
        publisher = h.integration._publisher
        assert publisher.last_published["PuriPuly_Talk"] is True
        assert publisher.last_published["PuriPuly_Captions"] is False
        assert publisher.last_published[TRANS] is False
        assert h.integration._peer_comparison["PuriPuly_Captions"] is False
        assert h.integration._peer_comparison[TRANS] is True
        assert h.integration._publication_pending == {"PuriPuly_Captions"}
        assert h.integration.output_snapshot()["state_publication"]["disposition"] == "partial"
        h.integration.publish_delta()
        assert h.integration._publication_pending == {"PuriPuly_Captions"}
        h.sender.fail_parameter = None
        h.integration.publish_delta()
        assert h.sender.messages == [("PuriPuly_Talk", True), ("PuriPuly_Captions", True)]
        assert h.integration.output_snapshot()["state_publication"] is None
        assert publisher.last_published[TRANS] is False
        assert h.integration._publication_pending is None
    finally:
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ((False, True, False), (True, False, True), (False, False, True)))
async def test_delayed_reordered_duplicate_bool_packets_settle_without_claiming_user_order(order):
    h = RuntimeHarness()
    await h.start()
    try:
        await h.application.set_translation_enabled(True)
        await h.application.set_translation_enabled(False)
        assert h.sender.messages == [(TRANS, True), (TRANS, False)]
        results = [await h.dispatch(value) for value in order]
        assert h.state.translation is order[-1]
        assert sum(result.applied for result in results) >= 1
        assert h.sender.messages == [(TRANS, True), (TRANS, False)]
        commands = list(h.commands)
        for _ in range(4):
            h.integration.publish_delta()
            assert (await h.dispatch(order[-1])).error == "echo_suppressed"
        assert h.commands == commands
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_lifecycle_snapshot_and_fence_do_not_accept_initialization_as_observation():
    h = RuntimeHarness()
    await h.start()
    try:
        assert (await h.dispatch(True)).applied
        assert h.integration._ingress_observations[TRANS].value is True
        await h.integration.configure_connection(mode="manual", send_port=19002, receive_port=19001)
        assert h.integration._ingress_observations == {}
        assert (TRANS, True) in h.sender.messages
        handler = h.receiver.handlers["control_packet_handler"]
        assert handler(PREFIX + TRANS, (False,)) is False
        assert handler(PREFIX + TRANS, (True,)) is False
        assert h.integration._ingress_observations == {}
        assert h.state.translation is True
        assert h.commands == [(TRANS, True)]
        assert handler(PREFIX + TRANS, (False,)) is True
        await turns()
        assert h.state.translation is False
        assert h.commands == [(TRANS, True), (TRANS, False)]
    finally:
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("on_surface", ("osc", "ui"))
@pytest.mark.parametrize("off_surface", ("osc", "ui"))
async def test_off_during_preparation_survives_equal_canonical_and_old_send(
    on_surface, off_surface
):
    class Selection:
        ready_for_enable = False
        generation = 1

        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_enable(self):
            self.entered.set()
            await self.release.wait()
            self.ready_for_enable = True
            return True

    h = RuntimeHarness()
    selection = Selection()
    h.application._index_selection = selection
    await h.start()
    on = asyncio.create_task(
        h.dispatch(True) if on_surface == "osc" else h.application.set_translation_enabled(True)
    )
    await selection.entered.wait()
    try:
        assert h.state.translation is False
        assert h.integration._publisher.last_published[TRANS] is False
        if off_surface == "osc":
            assert (await h.dispatch(False)).applied
        else:
            assert await h.application.set_translation_enabled(False)
        assert h.commands == [(TRANS, False)]
        assert h.state.translation is False
        selection.release.set()
        await on
        assert h.state.translation is False
        assert h.commands == [(TRANS, False)]
        assert h.sender.messages == (
            [(TRANS, False)] if on_surface == "osc" and off_surface == "ui" else []
        )
        assert h.integration._pending_controls() == ()
    finally:
        selection.release.set()
        await asyncio.gather(on, return_exceptions=True)
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", (False, True))
async def test_background_owner_operation_is_gated_until_authoritative_settlement(fails):
    h = RuntimeHarness()
    await h.start()
    h.warmup_gate.clear()
    h.failure = "on" if fails else None
    background = asyncio.create_task(h.translation.set_enabled(True))
    await h.warmup_entered.wait()
    try:
        assert h.state.translation is True
        assert h.sender.messages == []
        assert (TRANS, True) in h.integration._pending_controls()
        h.state = replace(h.state, captions=True)
        h.integration.publish_delta()
        assert h.sender.messages == [("PuriPuly_Captions", True)]
        h.warmup_gate.set()
        if fails:
            with pytest.raises(RuntimeError):
                await background
            assert h.state.translation is False
            assert h.sender.messages == [("PuriPuly_Captions", True)]
        else:
            assert await background is True
            assert h.state.translation is True
            assert h.sender.messages == [("PuriPuly_Captions", True), (TRANS, True)]
            assert (await h.dispatch(True)).error == "echo_suppressed"
        assert h.commands == []
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()


@pytest.mark.asyncio
async def test_returned_values_after_overlap_reach_a_correct_publication_fixed_point():
    h = RuntimeHarness()
    await h.start()
    h.warmup_gate.clear()
    on = asyncio.create_task(h.dispatch(True))
    await h.warmup_entered.wait()
    off = asyncio.create_task(h.dispatch(False))
    try:
        await turns()
        assert off.done() and (await off).applied
        h.warmup_gate.set()
        await asyncio.gather(on, return_exceptions=True)
        pending = list(h.sender.messages)
        cursor = len(h.sender.messages)
        accepted = 0
        errors = []
        for _ in range(30):
            if not pending:
                break
            parameter, value = pending.pop(0)
            result = await h.dispatch(value, parameter)
            accepted += int(result.applied)
            if result.error and result.error != "echo_suppressed":
                errors.append(result.error)
            pending.extend(h.sender.messages[cursor:])
            cursor = len(h.sender.messages)
        assert pending == []
        assert accepted == 0
        assert errors == []
        assert h.commands == [(TRANS, True), (TRANS, False)]
        assert h.state.translation is False
        assert h.sender.messages == []
        assert h.integration._pending_controls() == ()
        assert h.integration.router.pending_controls == ()
    finally:
        h.warmup_gate.set()
        await asyncio.gather(on, off, return_exceptions=True)
        await h.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("parameter,field,_method", RUNTIME_CONTROLS)
@pytest.mark.parametrize("requested", (False, True))
async def test_failed_same_state_runtime_request_remains_an_explicit_retry(
    parameter, field, _method, requested
):
    h = RuntimeHarness()
    await h.start()
    try:
        if not requested:
            assert (await h.dispatch(True, parameter)).applied
        h.state = replace(h.state, **{field: requested})
        h.reject = requested
        before = len(h.commands)
        assert (await h.dispatch(requested, parameter)).error == "application_rejected"
        assert getattr(h.state, field) is requested
        assert h.sender.messages == []
        h.reject = None
        assert (await h.dispatch(requested, parameter)).applied
        assert len(h.commands) == before + 2
        assert getattr(h.state, field) is requested
        assert h.sender.messages == []
        assert (await h.dispatch(requested, parameter)).error == "echo_suppressed"
        assert h.integration._pending_controls() == ()
    finally:
        await h.close()
