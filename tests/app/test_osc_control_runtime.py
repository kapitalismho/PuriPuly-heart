from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from puripuly_heart.app.ports.osc_control import ASR_ID_BY_PROVIDER, OSC_PARAMETER_DEFINITIONS
from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
)
from puripuly_heart.app.ports.ui_models import (
    OscControlPresentationName,
    OscControlPresentationState,
)
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresence
from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.osc import control_runtime as control_runtime_module
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.osc.state_publisher import (
    OscCanonicalState,
    state_from_settings,
)
from puripuly_heart.app.services.settings.settings_application import osc_control_presentation_state
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class FakeSender:
    def __init__(self) -> None:
        self.destinations: list[tuple[str, int]] = []
        self.messages: list[tuple[str, object]] = []

    def set_destination(self, host: str, port: int) -> None:
        self.destinations.append((host, port))

    def send_message(self, address: str, *values: object) -> None:
        self.messages.append((address, values[0] if len(values) == 1 else values))

    def send_chatbox(self, _text: str) -> None:
        return None

    def send_typing(self, _is_typing: bool) -> None:
        return None


class FakeReceiverOwner:
    def __init__(self) -> None:
        self.receiver: object | None = None
        self.effective_port = 49152
        self.last_enabled: bool | None = None
        self.control_calls: list[tuple[bool, str, int, bool]] = []
        self.packet_handlers: dict[str, object] = {}
        self.closed = False

    def set_packet_handlers(self, **handlers: object) -> None:
        self.packet_handlers = handlers

    async def configure_control(
        self,
        *,
        active: bool,
        host: str,
        port: int,
        force_restart: bool = False,
    ) -> None:
        self.control_calls.append((active, host, port, force_restart))
        self.receiver = object() if active else None
        if port:
            self.effective_port = port

    async def ensure_receiver(self) -> object | None:
        if self.receiver is None:
            self.receiver = object()
        return self.receiver

    async def stop_receiver(self) -> None:
        self.receiver = None

    async def configure(self, *, enabled: bool) -> None:
        self.last_enabled = enabled

    def stop_ingress(self) -> None:
        return None

    async def close(self) -> None:
        self.receiver = None
        self.closed = True


class PacketInjectingReceiverOwner(FakeReceiverOwner):
    def __init__(self, *packets: tuple[str, tuple[object, ...]]) -> None:
        super().__init__()
        self.packets = packets
        self.results: list[object] = []

    async def configure_control(
        self,
        *,
        active: bool,
        host: str,
        port: int,
        force_restart: bool = False,
    ) -> None:
        await super().configure_control(
            active=active,
            host=host,
            port=port,
            force_restart=force_restart,
        )
        if active:
            handler = self.packet_handlers["control_packet_handler"]
            assert callable(handler)
            for address, values in self.packets:
                self.results.append(handler(address, values))


class DashboardApplication:
    def __init__(self) -> None:
        self.stt_values: list[bool] = []

    async def set_stt_enabled(self, _enabled: bool) -> None:
        self.stt_values.append(_enabled)
        return None


class BlockingDashboardApplication:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.completed = False

    async def set_stt_enabled(self, _enabled: bool) -> None:
        self.started.set()
        await self.gate.wait()
        self.completed = True


@dataclass
class FakeService:
    info: OscQueryServiceInfo | None
    started: int = 0
    stopped: int = 0
    avatar_queries: int = 0
    advertisements: list[OscQueryAdvertisement] | None = None

    def __post_init__(self) -> None:
        self.advertisements = []

    async def start(self, _callback=None) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def discover_vrchat(self) -> OscQueryServiceInfo | None:
        return self.info

    async def advertise_receiver(self, advertisement: OscQueryAdvertisement) -> None:
        assert self.advertisements is not None
        self.advertisements.append(advertisement)

    async def unadvertise_receiver(self) -> None:
        assert self.advertisements is not None
        self.advertisements.clear()

    async def query_avatar(self, _service: OscQueryServiceInfo) -> dict[str, object]:
        self.avatar_queries += 1
        return {
            "CONTENTS": {
                "avatar": {
                    "CONTENTS": {
                        "parameters": {
                            "CONTENTS": {
                                "PuriPuly_Talk": {"TYPE": "T"},
                                "PuriPuly_SelfASR": {"TYPE": "i"},
                                "PuriPuly_Trans": {"TYPE": "i"},
                            }
                        }
                    }
                }
            }
        }


class DelayedAvatarService(FakeService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.block_avatar_queries = False
        self.query_started = asyncio.Event()
        self.query_gate = asyncio.Event()

    async def query_avatar(self, service: OscQueryServiceInfo) -> dict[str, object]:
        if self.block_avatar_queries:
            self.query_started.set()
            await self.query_gate.wait()
        return await super().query_avatar(service)


class DelayedDiscoveryService(FakeService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.block_discovery = False
        self.discovery_started = asyncio.Event()
        self.discovery_gate = asyncio.Event()

    async def discover_vrchat(self) -> OscQueryServiceInfo | None:
        if self.block_discovery:
            self.discovery_started.set()
            await self.discovery_gate.wait()
        return await super().discover_vrchat()


class RetryDiscoveryService(DelayedDiscoveryService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.fail_discovery = True

    async def discover_vrchat(self) -> OscQueryServiceInfo | None:
        if self.fail_discovery:
            raise OSError("controlled discovery failure")
        return await super().discover_vrchat()


class DelayedAdvertiseService(FakeService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.advertise_started = asyncio.Event()
        self.advertise_gate = asyncio.Event()

    async def advertise_receiver(self, advertisement: OscQueryAdvertisement) -> None:
        self.advertise_started.set()
        await self.advertise_gate.wait()
        await super().advertise_receiver(advertisement)


async def _wait_automatic_query(integration: OscControlIntegrationOwner) -> None:
    await integration.wait_automatic_query_start()


def _integration(
    settings: AppSettingsVNext,
    receiver_owner: FakeReceiverOwner,
    sender: FakeSender,
    service: FakeService,
    *,
    application: object | None = None,
    osc_state: OscCanonicalState | None = None,
    state_provider: Callable[[], OscCanonicalState] | None = None,
    resync_timeout_seconds: float = 1.5,
    connection_changed: Callable[[], None] | None = None,
) -> OscControlIntegrationOwner:
    return OscControlIntegrationOwner(
        receiver_owner=receiver_owner,
        settings_provider=lambda: settings,
        apply_settings=lambda _settings: None,
        application_provider=lambda: application,
        sender_provider=lambda: sender,
        state_provider=state_provider or (lambda: osc_state or OscCanonicalState()),
        language_state_provider=lambda: ("ko", "en", "en", "ko"),
        translation_model_normalizer=materialize_canonical_translation_settings,
        query_service=service,
        resync_timeout_seconds=resync_timeout_seconds,
        connection_changed=connection_changed,
    )


@pytest.mark.asyncio
async def test_connection_observer_failure_does_not_fail_effective_transition() -> None:
    def connection_changed() -> None:
        raise RuntimeError("presentation unavailable")

    integration = _integration(
        AppSettingsVNext(),
        FakeReceiverOwner(),
        FakeSender(),
        FakeService(info=None),
        connection_changed=connection_changed,
    )
    try:
        await integration.configure_connection(mode="manual", send_port=9123, receive_port=9124)
        assert integration.output_snapshot()["effective_mode"] == "manual"
        assert integration.output_snapshot()["local_availability"] == "available"
        await integration.configure_connection(mode="off", send_port=9123, receive_port=9124)
        assert integration.output_snapshot()["effective_mode"] == "off"
    finally:
        await integration.close()


@pytest.mark.asyncio
async def test_output_snapshot_separates_local_bind_and_discovery_from_remote_delivery() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(info=None)
    integration = _integration(settings, receiver_owner, sender, service)
    assert integration.output_snapshot()["local_availability"] == "off"

    await integration.configure_connection(mode="manual", send_port=9030, receive_port=9031)
    active = integration.output_snapshot()
    assert active["configured_send_port"] == 9030
    assert active["effective_receive_port"] == 9031
    assert active["receiver_available"] is True
    assert active["local_availability"] == "available"
    assert active["remote_delivery"] == "unacknowledged"

    receiver_owner.receiver = None
    missing = integration.output_snapshot()
    assert missing["effective_receive_port"] is None
    assert missing["local_availability"] == "receiver_unavailable"
    await integration.close()

    class FailedDiscoveryService(FakeService):
        async def discover_vrchat(self) -> OscQueryServiceInfo | None:
            raise OSError("discovery unavailable")

    fallback = _integration(
        settings, FakeReceiverOwner(), FakeSender(), FailedDiscoveryService(info=None)
    )
    await fallback.configure_connection(mode="automatic", send_port=9030, receive_port=9031)
    await fallback.wait_automatic_query_start()
    degraded = fallback.output_snapshot()
    assert degraded["query_service_started"] is False
    assert degraded["query_failure"] == "oscquery_unavailable"
    assert degraded["local_availability"] == "available"
    assert degraded["effective_send_port"] == 9000
    assert degraded["remote_delivery"] == "unacknowledged"
    await fallback.close()


@pytest.mark.asyncio
async def test_failed_automatic_discovery_retry_preserves_settings_and_is_not_remote_success() -> (
    None
):
    class NoChatboxSender(FakeSender):
        def send_chatbox(self, _text: str) -> None:
            pytest.fail("Discovery retry must not send chatbox text")

    settings = AppSettingsVNext()
    before = deepcopy(settings)
    receiver = FakeReceiverOwner()
    sender = NoChatboxSender()
    service = RetryDiscoveryService(
        OscQueryServiceInfo(
            service_id="controlled-VRChat",
            host="127.0.0.1",
            osc_send_port=19000,
            is_vrchat=True,
        )
    )
    integration = _integration(settings, receiver, sender, service)
    try:
        await integration.configure(enabled=False)
        initial_receiver = receiver.receiver
        await integration.wait_automatic_query_start()
        assert receiver.receiver is initial_receiver
        failed = integration.output_snapshot()
        assert failed["query_failure"] == "oscquery_unavailable"
        assert failed["local_availability"] == "available"
        assert failed["discovered_remote_present"] is False
        assert sender.destinations[-1] == ("127.0.0.1", 9000)
        service.fail_discovery = False
        service.block_discovery = True
        bound_receiver = receiver.receiver
        control_calls = list(receiver.control_calls)
        messages = list(sender.messages)
        pending = asyncio.create_task(integration.retry_automatic_discovery())
        await asyncio.wait_for(service.discovery_started.wait(), timeout=1)
        assert not pending.done()
        duplicate = await integration.retry_automatic_discovery()
        assert duplicate["status"] == "rejected"
        assert duplicate["reason"] == "discovery_in_progress"
        assert service.started == 2
        service.discovery_gate.set()
        first = await pending
        assert first["status"] == "applied"
        assert first["retry_started"] is True
        assert first["effect"] == "discovery_target_applied"
        assert first["output"]["query_service_started"] is True
        assert first["output"]["remote_delivery"] == "unacknowledged"
        assert receiver.receiver is bound_receiver
        assert receiver.control_calls == control_calls
        assert sender.messages == messages
        recovered = integration.output_snapshot()
        assert recovered["query_service_started"] is True
        assert recovered["query_failure"] is None
        assert recovered["discovered_remote_present"] is True
        assert recovered["remote_delivery"] == "unacknowledged"
        assert sender.destinations[-1] == ("127.0.0.1", 19000)
        assert settings == before
        assert recovered["configured_send_port"] == before.intent.osc.send_port
        assert recovered["configured_receive_port"] == before.intent.osc.receive_port
        calls = len(receiver.control_calls), service.started, service.stopped
        assert (await integration.retry_automatic_discovery())["status"] == "applied"
        assert (len(receiver.control_calls), service.started, service.stopped) == calls
    finally:
        service.discovery_gate.set()
        await integration.close()


@pytest.mark.asyncio
async def test_discovery_retry_failure_remains_visible_and_retryable() -> None:
    settings = AppSettingsVNext()
    service = RetryDiscoveryService(None)
    integration = _integration(settings, FakeReceiverOwner(), FakeSender(), service)
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        for _ in range(2):
            result = await integration.retry_automatic_discovery()
            assert result["retry_started"] is True
            assert result["status"] == "degraded"
            assert result["effect"] == "discovery_unavailable"
            assert result["reason"] == "oscquery_unavailable"
            assert result["diagnostics"]["code"] == "oscquery_unavailable"
            assert result["diagnostics"]["fields"]["exception_class"] == "OSError"
            await integration.wait_automatic_query_start()
            observed = integration.output_snapshot()
            assert observed["query_failure"] == "oscquery_unavailable"
            assert observed["query_service_started"] is False
            assert observed["local_availability"] == "available"
            assert observed["remote_delivery"] == "unacknowledged"
        assert service.started == 3
    finally:
        await integration.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("manual", "off"))
async def test_discovery_retry_does_not_reset_ineligible_connections(mode) -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(settings.intent, osc=replace(settings.intent.osc, connection_mode=mode)),
    )
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(None)
    integration = _integration(settings, receiver, sender, service)
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        before = integration.output_snapshot()
        calls = (
            list(receiver.control_calls),
            list(sender.destinations),
            service.started,
            service.stopped,
        )
        result = await integration.retry_automatic_discovery()
        assert result["status"] == "rejected"
        assert result["retry_started"] is False
        assert result["reason"] == "automatic_mode_required"
        assert integration.output_snapshot() == before
        assert (
            receiver.control_calls,
            sender.destinations,
            service.started,
            service.stopped,
        ) == calls
    finally:
        await integration.close()


@pytest.mark.asyncio
async def test_discovery_retry_rechecks_authoritative_connection_after_waiting() -> None:
    settings = AppSettingsVNext()
    receiver = FakeReceiverOwner()
    service = RetryDiscoveryService(None)
    integration = _integration(settings, receiver, FakeSender(), service)
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        calls = len(receiver.control_calls), service.started
        async with integration._connection_lock:
            retry = asyncio.create_task(integration.retry_automatic_discovery())
            await asyncio.sleep(0)
            changed = replace(
                settings,
                intent=replace(
                    settings.intent,
                    osc=replace(settings.intent.osc, connection_mode="manual"),
                ),
            )
            integration._settings_provider = lambda: changed
        result = await retry
        assert result["status"] == "rejected"
        assert result["reason"] == "connection_changed"
        assert (len(receiver.control_calls), service.started) == calls
        integration._settings_provider = lambda: None
        assert (await integration.retry_automatic_discovery())["reason"] == "settings_unavailable"
        integration.stop_ingress()
        assert (await integration.retry_automatic_discovery())["reason"] == "lifecycle_stopped"
    finally:
        await integration.close()
    assert (await integration.retry_automatic_discovery())["reason"] == "lifecycle_stopped"


@pytest.mark.asyncio
async def test_stopped_configuration_does_not_wait_on_shutdown_connection_lock() -> None:
    receiver = FakeReceiverOwner()
    integration = _integration(AppSettingsVNext(), receiver, FakeSender(), FakeService(None))
    try:
        await integration.configure_connection(mode="manual", send_port=19000, receive_port=19001)
        calls = list(receiver.control_calls)
        integration.stop_ingress()
        async with integration._connection_lock:
            await asyncio.wait_for(
                integration.configure_connection(mode="off", send_port=19000, receive_port=19001),
                timeout=1,
            )
        assert receiver.control_calls == calls
    finally:
        await integration.close()


@pytest.mark.asyncio
async def test_shutdown_fences_retry_during_receiver_ensure() -> None:
    class HeldReceiver(FakeReceiverOwner):
        hold = False

        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def ensure_receiver(self) -> object | None:
            if self.hold:
                self.entered.set()
                await self.release.wait()
            return await super().ensure_receiver()

    receiver = HeldReceiver()
    service = RetryDiscoveryService(None)
    integration = _integration(AppSettingsVNext(), receiver, FakeSender(), service)
    await integration.configure(enabled=False)
    await integration.wait_automatic_query_start()
    service.fail_discovery = False
    receiver.hold = True
    retry = asyncio.create_task(integration.retry_automatic_discovery())
    await receiver.entered.wait()
    closing = asyncio.create_task(integration.close())
    await asyncio.sleep(0)
    receiver.release.set()
    result = await retry
    await closing
    assert result["status"] == "rejected"
    assert result["reason"] == "lifecycle_stopped"
    assert service.started == 2
    assert receiver.receiver is None
    assert receiver.closed is True
    assert integration.output_snapshot()["local_availability"] == "closed"


@pytest.mark.asyncio
async def test_shutdown_cancels_inflight_retry_without_late_destination_or_publication() -> None:
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    service = RetryDiscoveryService(None)
    integration = _integration(AppSettingsVNext(), receiver, sender, service)
    await integration.configure(enabled=False)
    await integration.wait_automatic_query_start()
    service.fail_discovery = False
    service.block_discovery = True
    retry = asyncio.create_task(integration.retry_automatic_discovery())
    await service.discovery_started.wait()
    destinations = list(sender.destinations)
    messages = list(sender.messages)
    await integration.close()
    service.discovery_gate.set()
    result = await retry
    assert result["status"] == "rejected"
    assert result["reason"] == "lifecycle_stopped"
    assert sender.destinations == destinations
    assert sender.messages == messages
    assert receiver.receiver is None
    assert integration.query_runtime.started is False
    assert integration.output_snapshot()["remote_delivery"] == "unacknowledged"


@pytest.mark.asyncio
async def test_in_process_complete_control_matrix_projects_final_canonical_state() -> None:
    current = [AppSettingsVNext()]
    runtime = {
        "self_capture": False,
        "peer_capture": False,
        "translation": False,
        "captions": False,
    }
    runtime_calls: list[tuple[str, bool]] = []
    translation_authorization_calls: list[bool] = []
    settings_apply_calls: list[AppSettingsVNext] = []
    projected: list[OscControlPresentationState] = []

    class CanonicalDashboardApplication:
        async def set_stt_enabled(self, enabled: bool) -> None:
            runtime["self_capture"] = enabled
            runtime_calls.append(("self_capture", enabled))

        async def set_peer_translation_enabled(self, enabled: bool) -> None:
            runtime["peer_capture"] = enabled
            runtime_calls.append(("peer_capture", enabled))

        async def set_translation_enabled(
            self, enabled: bool, *, allow_authorization: bool = True
        ) -> None:
            translation_authorization_calls.append(allow_authorization)
            runtime["translation"] = enabled
            runtime_calls.append(("translation", enabled))

        async def set_overlay_enabled(self, enabled: bool) -> None:
            runtime["captions"] = enabled
            runtime_calls.append(("captions", enabled))

    async def apply_settings(value: object) -> None:
        assert isinstance(value, AppSettingsVNext)
        current[0] = value
        settings_apply_calls.append(value)

    def canonical_state() -> OscCanonicalState:
        return state_from_settings(
            current[0],
            self_capture=runtime["self_capture"],
            peer_capture=runtime["peer_capture"],
            translation=runtime["translation"],
            captions=runtime["captions"],
        )

    def presentation_state(
        control: OscControlPresentationName,
    ) -> OscControlPresentationState:
        return osc_control_presentation_state(
            current[0],
            canonical_state=canonical_state(),
            changed_control=control,
        )

    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    application = CanonicalDashboardApplication()
    integration = OscControlIntegrationOwner(
        receiver_owner=receiver_owner,
        settings_provider=lambda: current[0],
        apply_settings=apply_settings,
        application_provider=lambda: application,
        sender_provider=lambda: sender,
        state_provider=canonical_state,
        language_state_provider=lambda: (
            current[0].intent.languages.source_language,
            current[0].intent.languages.target_language,
            current[0].intent.languages.peer_source_language,
            current[0].intent.languages.peer_target_language,
        ),
        translation_model_normalizer=materialize_canonical_translation_settings,
        ui_state_provider=presentation_state,
        ui_state_sink=projected.append,
        query_service=FakeService(None),
    )
    packets: list[tuple[OscControlPresentationName, bool | int]] = [
        ("PuriPuly_Talk", True),
        ("PuriPuly_Listen", True),
        ("PuriPuly_Trans", True),
        ("PuriPuly_Captions", True),
        ("PuriPuly_PeerAuto", True),
        ("PuriPuly_MuteSync", True),
        ("PuriPuly_ChatboxSource", True),
        ("PuriPuly_SelfSrcLang", 16),
        ("PuriPuly_SelfDstLang", 11),
        ("PuriPuly_SelfDstLang2", 16),
        ("PuriPuly_PeerSrcLang", 5),
        ("PuriPuly_PeerDstLang", 7),
        ("PuriPuly_SelfASR", 7),
        ("PuriPuly_PeerASR", 4),
        ("PuriPuly_Translator", 5),
    ]

    for name, value in packets:
        result = await integration.router.dispatch_packet(
            f"/avatar/parameters/{name}",
            value,
        )

        assert result.applied is True
        assert projected[-1] == presentation_state(name)

    assert [state.changed_control for state in projected] == [name for name, _value in packets]
    assert len(runtime_calls) == 4
    assert translation_authorization_calls == [False]
    assert len(settings_apply_calls) == 11
    assert sender.messages == []
    await integration.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", [0, 1, 2])
async def test_gemma_osc_inputs_restore_connection_and_publish_canonical_after_restart(
    model_id: int,
    tmp_path: Path,
) -> None:
    from puripuly_heart.config.settings_vnext.compat import load_vnext_settings, save_vnext_settings
    from puripuly_heart.config.settings_vnext.serialization import to_dict

    initial = AppSettingsVNext()
    current = [
        replace(
            initial,
            intent=replace(
                initial.intent,
                translation=replace(
                    initial.intent.translation,
                    model="gpt_6_luna",
                    connection="official_byok",
                    connection_history={
                        "gemma4_26b_31b": "openrouter",
                        "gpt_6_luna": "official_byok",
                    },
                ),
            ),
        )
    ]
    saved: list[dict[str, object]] = []
    config_path = tmp_path / "osc-settings.json"

    async def apply_settings(value: object) -> None:
        assert isinstance(value, AppSettingsVNext)
        assert save_vnext_settings(config_path, value).ok
        saved.append(to_dict(value))
        result = load_vnext_settings(config_path)
        assert result.ok and result.settings is not None and not result.migrated
        current[0] = result.settings

    def start_owner(sender: FakeSender) -> OscControlIntegrationOwner:
        return OscControlIntegrationOwner(
            receiver_owner=FakeReceiverOwner(),
            settings_provider=lambda: current[0],
            apply_settings=apply_settings,
            application_provider=lambda: None,
            sender_provider=lambda: sender,
            state_provider=lambda: state_from_settings(current[0]),
            language_state_provider=lambda: ("ko", "en", "en", "ko"),
            translation_model_normalizer=materialize_canonical_translation_settings,
            query_service=FakeService(None),
        )

    address = "/avatar/parameters/PuriPuly_Translator"
    sender = FakeSender()
    owner = start_owner(sender)
    await owner.configure_connection(mode="manual", send_port=9000, receive_port=9001)
    result = await owner.router.dispatch_packet(address, model_id)
    assert result.applied is True
    assert current[0].intent.translation.model == "gemma4_26b_31b"
    assert current[0].intent.translation.connection == "openrouter"
    assert current[0].intent.translation.connection_history["gpt_6_luna"] == "official_byok"
    assert current[0].intent.translation.connection_history["gemma4_26b_31b"] == "openrouter"
    assert sender.messages[15:] == ([(address, 0)] if model_id in (1, 2) else [])
    assert saved and saved[-1]["intent"]["translation"]["model"] == "gemma4_26b_31b"
    await owner.close()

    loaded = load_vnext_settings(config_path)
    assert loaded.ok and loaded.settings is not None and not loaded.migrated
    current[0] = loaded.settings
    restarted_sender = FakeSender()
    restarted = start_owner(restarted_sender)
    await restarted.configure_connection(mode="manual", send_port=9000, receive_port=9001)
    assert (address, 0) in restarted_sender.messages
    published_before_alias = len(restarted_sender.messages)
    await restarted.router.dispatch_packet(address, model_id)
    assert current[0].intent.translation.connection == "openrouter"
    assert (address, 0) in restarted_sender.messages
    if model_id in (1, 2):
        assert (address, 0) in restarted_sender.messages[published_before_alias:]
    await restarted.router.dispatch_packet(address, 14)
    assert current[0].intent.translation.connection == "official_byok"
    await restarted.router.dispatch_packet(address, model_id)
    assert current[0].intent.translation.connection == "openrouter"
    assert current[0].intent.translation.connection_history["gpt_6_luna"] == "official_byok"
    await restarted.close()


@pytest.mark.asyncio
async def test_normalized_asr_rejection_projects_the_committed_canonical_state() -> None:
    current = [
        replace(
            AppSettingsVNext(),
            intent=replace(
                AppSettingsVNext().intent,
                stt=replace(AppSettingsVNext().intent.stt, provider=STTProviderName.DEEPGRAM.value),
            ),
        )
    ]
    projected: list[OscControlPresentationState] = []

    async def apply_settings(value: object) -> None:
        assert isinstance(value, AppSettingsVNext)
        current[0] = replace(
            value,
            intent=replace(
                value.intent,
                stt=replace(value.intent.stt, provider=STTProviderName.LOCAL_CPU_AUTO.value),
            ),
        )

    def canonical_state() -> OscCanonicalState:
        return state_from_settings(current[0])

    integration = OscControlIntegrationOwner(
        receiver_owner=FakeReceiverOwner(),
        settings_provider=lambda: current[0],
        apply_settings=apply_settings,
        application_provider=lambda: None,
        sender_provider=lambda: FakeSender(),
        state_provider=canonical_state,
        language_state_provider=lambda: ("ko", "en", "en", "ko"),
        translation_model_normalizer=materialize_canonical_translation_settings,
        ui_state_provider=lambda control: osc_control_presentation_state(
            current[0],
            canonical_state=canonical_state(),
            changed_control=control,
        ),
        ui_state_sink=projected.append,
        query_service=FakeService(None),
    )

    result = await integration.router.dispatch_packet(
        "/avatar/parameters/PuriPuly_SelfASR",
        ASR_ID_BY_PROVIDER[STTProviderName.LOCAL_QWEN_GPU.value],
    )

    assert result.applied is False
    assert result.error == "application_rejected"
    assert current[0].intent.stt.provider == STTProviderName.LOCAL_CPU_AUTO.value
    assert len(projected) == 1
    assert projected[0].changed_control == "PuriPuly_SelfASR"
    assert projected[0].self_asr_setting == STTProviderName.LOCAL_CPU_AUTO.value
    await integration.close()


@pytest.mark.asyncio
async def test_integration_shares_dynamic_receiver_and_transitions_modes() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(
        OscQueryServiceInfo(
            service_id="VRChat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )
    )
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure(enabled=False)
    await _wait_automatic_query(integration)

    assert integration.connection_mode == "automatic"
    assert receiver_owner.control_calls[-1] == (True, "127.0.0.1", 0, True)
    assert service.advertisements is not None
    assert service.advertisements[0].port == 49152
    assert sender.destinations[-1] == ("127.0.0.1", 9010)
    automatic_messages = len(sender.messages)
    assert automatic_messages >= 15
    diagnostics = integration.avatar_parameter_diagnostics
    assert "PuriPuly_Talk" in diagnostics["present"]
    assert "PuriPuly_SelfASR" in diagnostics["present"]
    assert "PuriPuly_Trans" in diagnostics["type_mismatches"]

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )

    assert receiver_owner.control_calls[-1] == (True, "127.0.0.1", 9021, True)
    assert service.stopped == 1
    assert sender.destinations[-1] == ("127.0.0.1", 9020)
    assert len(sender.messages) == automatic_messages + 15

    message_count = len(sender.messages)
    await integration.configure_connection(
        mode="off",
        send_port=9020,
        receive_port=9021,
    )
    assert receiver_owner.receiver is None
    assert len(sender.messages) == message_count
    await integration.close()
    assert receiver_owner.closed is True


@pytest.mark.asyncio
async def test_automatic_configure_returns_before_oscquery_advertise() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = DelayedAdvertiseService(None)
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9020,
        receive_port=9021,
    )

    assert receiver_owner.control_calls[-1] == (True, "127.0.0.1", 0, True)
    assert integration.query_runtime.started is False
    assert service.advertisements == []

    await service.advertise_started.wait()
    assert service.advertisements == []

    service.advertise_gate.set()
    await _wait_automatic_query(integration)

    assert integration.query_runtime.started is True
    assert service.advertisements is not None
    assert service.advertisements[0].port == 49152
    await integration.close()


@pytest.mark.asyncio
async def test_automatic_refresh_uses_vrchat_default_instead_of_saved_manual_port() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(None)
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9123,
        receive_port=9124,
    )
    await _wait_automatic_query(integration)
    assert sender.destinations[-1] == ("127.0.0.1", 9000)
    assert integration.effective_send_port == 9000

    await integration.query_runtime.refresh()

    assert sender.destinations[-1] == ("127.0.0.1", 9000)

    await integration.configure_connection(
        mode="manual",
        send_port=9123,
        receive_port=9124,
    )
    assert sender.destinations[-1] == ("127.0.0.1", 9123)
    assert integration.effective_send_port == 9123
    await integration.close()


@pytest.mark.asyncio
async def test_automatic_discovery_recovers_after_vrchat_disappears_and_reappears() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(
        OscQueryServiceInfo(
            service_id="VRChat",
            host="127.0.0.1",
            osc_send_port=9030,
            is_vrchat=True,
        )
    )
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9130,
        receive_port=9131,
    )
    await _wait_automatic_query(integration)
    assert sender.destinations[-1] == ("127.0.0.1", 9030)

    service.info = None
    await integration.query_runtime.refresh()
    assert sender.destinations[-1] == ("127.0.0.1", 9000)

    service.info = OscQueryServiceInfo(
        service_id="VRChat-restarted",
        host="127.0.0.1",
        osc_send_port=9040,
        is_vrchat=True,
    )
    await integration.query_runtime.refresh()
    assert sender.destinations[-1] == ("127.0.0.1", 9040)
    await integration.close()


@pytest.mark.asyncio
async def test_avatar_change_requeries_and_republishes_full_state() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(
        OscQueryServiceInfo(
            service_id="VRChat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )
    )
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9110,
        receive_port=9111,
    )
    await _wait_automatic_query(integration)
    initial_queries = service.avatar_queries
    sender.messages.clear()

    await integration.query_runtime.on_avatar_change()

    assert service.avatar_queries > initial_queries
    assert len(sender.messages) == 15
    await integration.close()


@pytest.mark.asyncio
async def test_two_integrations_keep_distinct_receivers_and_cleanup_independently() -> None:
    first_settings = AppSettingsVNext()
    second_settings = AppSettingsVNext()
    first_receiver = FakeReceiverOwner()
    second_receiver = FakeReceiverOwner()
    first_sender = FakeSender()
    second_sender = FakeSender()
    first = _integration(first_settings, first_receiver, first_sender, FakeService(None))
    second = _integration(second_settings, second_receiver, second_sender, FakeService(None))

    await first.configure_connection(mode="manual", send_port=9050, receive_port=9051)
    await second.configure_connection(mode="manual", send_port=9060, receive_port=9061)

    assert first_receiver.receiver is not None
    assert second_receiver.receiver is not None
    assert first_sender.destinations[-1] == ("127.0.0.1", 9050)
    assert second_sender.destinations[-1] == ("127.0.0.1", 9060)

    await first.close()
    assert first_receiver.closed is True
    assert second_receiver.receiver is not None
    await second.close()
    assert second_receiver.closed is True


@pytest.mark.asyncio
async def test_connection_start_fences_parameters_until_each_canonical_value_arrives() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(
        settings,
        receiver_owner,
        sender,
        FakeService(None),
        resync_timeout_seconds=60,
    )

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    control_handler = receiver_owner.packet_handlers["control_packet_handler"]

    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (11,)) is False
    assert "PuriPuly_SelfDstLang" in integration._resync_unsettled
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (7,)) is False
    assert "PuriPuly_SelfDstLang" not in integration._resync_unsettled
    assert "PuriPuly_SelfSrcLang" in integration._resync_unsettled
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (11,)) is True

    assert (
        "/avatar/parameters/PuriPuly_SelfDstLang2",
        255,
    ) in sender.messages
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang2", (255,)) is False
    assert "PuriPuly_SelfDstLang2" not in integration._resync_unsettled
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang2", (16,)) is True

    await integration.close()


@pytest.mark.asyncio
async def test_manual_connection_packet_before_snapshot_cannot_settle_parameter() -> None:
    settings = AppSettingsVNext()
    receiver_owner = PacketInjectingReceiverOwner(
        ("/avatar/parameters/PuriPuly_SelfDstLang", (7,)),
    )
    sender = FakeSender()
    integration = _integration(settings, receiver_owner, sender, FakeService(None))

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )

    assert receiver_owner.results == [False]
    assert integration._resync_ready_generation == integration._resync_generation
    assert "PuriPuly_SelfDstLang" in integration._resync_unsettled

    control_handler = receiver_owner.packet_handlers["control_packet_handler"]
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (7,)) is False
    assert "PuriPuly_SelfDstLang" not in integration._resync_unsettled

    await integration.close()


@pytest.mark.asyncio
async def test_invalid_prepublication_packet_cannot_make_generation_ready() -> None:
    settings = AppSettingsVNext()
    receiver_owner = PacketInjectingReceiverOwner(
        ("/avatar/parameters/PuriPuly_SelfASR", (99,)),
        ("/avatar/parameters/PuriPuly_SelfDstLang", (7,)),
    )
    sender = FakeSender()
    integration = _integration(settings, receiver_owner, sender, FakeService(None))

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )

    assert receiver_owner.results == [False, False]
    assert integration._resync_ready_generation == integration._resync_generation
    assert "PuriPuly_SelfDstLang" in integration._resync_unsettled

    control_handler = receiver_owner.packet_handlers["control_packet_handler"]
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (7,)) is False
    assert "PuriPuly_SelfDstLang" not in integration._resync_unsettled

    await integration.close()


@pytest.mark.asyncio
async def test_automatic_connection_packet_before_delayed_snapshot_cannot_settle() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = DelayedAvatarService(
        OscQueryServiceInfo(
            service_id="VRChat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )
    )
    service.block_avatar_queries = True
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9020,
        receive_port=9021,
    )
    await service.query_started.wait()
    control_handler = receiver_owner.packet_handlers["control_packet_handler"]

    assert integration._resync_ready_generation is None
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (7,)) is False
    assert "PuriPuly_SelfDstLang" in integration._resync_unsettled

    service.query_gate.set()
    await _wait_automatic_query(integration)

    assert integration._resync_ready_generation == integration._resync_generation
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (7,)) is False
    assert "PuriPuly_SelfDstLang" not in integration._resync_unsettled

    await integration.close()


@pytest.mark.asyncio
async def test_avatar_packet_before_delayed_snapshot_cannot_settle_parameter() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = DelayedAvatarService(
        OscQueryServiceInfo(
            service_id="VRChat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )
    )
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9020,
        receive_port=9021,
    )
    await _wait_automatic_query(integration)
    service.block_avatar_queries = True
    service.query_started.clear()
    integration._handle_avatar_change(())
    await service.query_started.wait()
    generation = integration._resync_generation
    control_handler = receiver_owner.packet_handlers["control_packet_handler"]

    assert integration._resync_ready_generation is None
    assert control_handler("/avatar/parameters/PuriPuly_Talk", (False,)) is False
    assert "PuriPuly_Talk" in integration._resync_unsettled

    service.query_gate.set()
    for _ in range(10):
        await asyncio.sleep(0)
        if integration._resync_ready_generation == generation:
            break

    assert integration._resync_ready_generation == generation
    assert control_handler("/avatar/parameters/PuriPuly_Talk", (False,)) is False
    assert "PuriPuly_Talk" not in integration._resync_unsettled

    await integration.close()


@pytest.mark.asyncio
async def test_resync_settlement_uses_live_canonical_state() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    state = [OscCanonicalState(self_target_language="en")]
    integration = _integration(
        settings,
        receiver_owner,
        sender,
        FakeService(None),
        state_provider=lambda: state[0],
        resync_timeout_seconds=60,
    )

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    control_handler = receiver_owner.packet_handlers["control_packet_handler"]

    state[0] = OscCanonicalState(self_target_language="fr")
    integration.publish_delta()

    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (7,)) is False
    assert "PuriPuly_SelfDstLang" in integration._resync_unsettled
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (11,)) is False
    assert "PuriPuly_SelfDstLang" not in integration._resync_unsettled
    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (7,)) is True

    await integration.close()


@pytest.mark.asyncio
async def test_resync_deadline_fails_open_for_all_remaining_parameters() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(
        settings,
        receiver_owner,
        sender,
        FakeService(None),
        resync_timeout_seconds=0.0,
    )

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    control_handler = receiver_owner.packet_handlers["control_packet_handler"]

    assert control_handler("/avatar/parameters/PuriPuly_SelfDstLang", (11,)) is True
    assert integration._resync_unsettled == set()

    await integration.close()


@pytest.mark.asyncio
async def test_avatar_snapshot_completion_does_not_rearm_event_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(control_runtime_module.time, "monotonic", lambda: now[0])
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(settings, receiver_owner, sender, FakeService(None))

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    now[0] = 200.0
    integration._handle_avatar_change(())
    generation = integration._resync_generation
    deadline = integration._resync_deadline
    now[0] = 300.0
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert integration._resync_generation == generation
    assert integration._resync_deadline == deadline == 201.5

    await integration.close()


@pytest.mark.asyncio
async def test_rapid_avatar_changes_ignore_stale_snapshot_completion() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(settings, receiver_owner, sender, FakeService(None))

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    initial_generation = integration._resync_generation
    sender.messages.clear()

    integration._handle_avatar_change(())
    integration._handle_avatar_change(())
    for _ in range(10):
        await asyncio.sleep(0)
        if len(sender.messages) == len(OSC_PARAMETER_DEFINITIONS):
            break

    assert integration._resync_generation == initial_generation + 2
    assert integration._resync_unsettled == set(OSC_PARAMETER_DEFINITIONS)
    assert len(sender.messages) == len(OSC_PARAMETER_DEFINITIONS)

    await integration.close()


@pytest.mark.asyncio
async def test_discovery_epoch_is_anchored_before_avatar_query_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(control_runtime_module.time, "monotonic", lambda: now[0])
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = DelayedAvatarService(
        OscQueryServiceInfo(
            service_id="VRChat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )
    )
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9020,
        receive_port=9021,
    )
    await _wait_automatic_query(integration)
    initial_generation = integration._resync_generation
    service.info = OscQueryServiceInfo(
        service_id="VRChat-restarted",
        host="127.0.0.1",
        osc_send_port=9030,
        is_vrchat=True,
    )
    service.block_avatar_queries = True
    now[0] = 200.0
    refresh_task = asyncio.create_task(integration.query_runtime.refresh())
    await service.query_started.wait()

    assert integration._resync_generation == initial_generation + 1
    assert integration._resync_deadline == 201.5

    now[0] = 300.0
    service.query_gate.set()
    await refresh_task

    assert integration._resync_generation == initial_generation + 1
    assert integration._resync_deadline == 201.5

    await integration.close()


@pytest.mark.asyncio
async def test_stale_discovery_cannot_supersede_newer_avatar_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(control_runtime_module.time, "monotonic", lambda: now[0])
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    service = DelayedDiscoveryService(
        OscQueryServiceInfo(
            service_id="VRChat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )
    )
    integration = _integration(settings, receiver_owner, sender, service)

    await integration.configure_connection(
        mode="automatic",
        send_port=9020,
        receive_port=9021,
    )
    await _wait_automatic_query(integration)
    service.info = OscQueryServiceInfo(
        service_id="VRChat-restarted",
        host="127.0.0.1",
        osc_send_port=9030,
        is_vrchat=True,
    )
    service.block_discovery = True
    now[0] = 200.0
    discovery_task = asyncio.create_task(integration.query_runtime.refresh())
    await service.discovery_started.wait()

    integration._handle_avatar_change(())
    avatar_generation = integration._resync_generation
    avatar_deadline = integration._resync_deadline
    now[0] = 210.0
    service.discovery_gate.set()
    await discovery_task
    for _ in range(10):
        await asyncio.sleep(0)
        if integration._resync_ready_generation == avatar_generation:
            break

    assert integration._resync_generation == avatar_generation
    assert integration._resync_deadline == avatar_deadline == 201.5
    assert integration._resync_ready_generation == avatar_generation

    await integration.close()


@pytest.mark.asyncio
async def test_runtime_boolean_command_reaches_existing_path_after_settlement() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    application = DashboardApplication()
    integration = _integration(
        settings,
        receiver_owner,
        sender,
        FakeService(None),
        application=application,
        osc_state=OscCanonicalState(self_capture=False),
        resync_timeout_seconds=60,
    )

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    control_handler = receiver_owner.packet_handlers["control_packet_handler"]

    assert control_handler("/avatar/parameters/PuriPuly_Talk", (True,)) is False
    assert application.stt_values == []
    assert control_handler("/avatar/parameters/PuriPuly_Talk", (False,)) is False
    assert control_handler("/avatar/parameters/PuriPuly_Talk", (True,)) is True
    for _ in range(10):
        await asyncio.sleep(0)
        if application.stt_values:
            break

    assert application.stt_values == [True]

    await integration.close()


@pytest.mark.asyncio
async def test_off_mode_settings_apply_does_not_publish_or_start_control_transport() -> None:
    settings = replace(
        AppSettingsVNext(),
        intent=replace(
            AppSettingsVNext().intent,
            osc=replace(AppSettingsVNext().intent.osc, connection_mode="off"),
        ),
    )
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(settings, receiver_owner, sender, FakeService(None))

    await integration.configure(enabled=False)

    assert sender.messages == []
    assert receiver_owner.control_calls[-1] == (False, "127.0.0.1", 9001, True)
    control_handler = receiver_owner.packet_handlers["control_packet_handler"]
    assert callable(control_handler)
    assert control_handler("/avatar/parameters/PuriPuly_Talk", (True,)) is False
    await integration.close()


@pytest.mark.asyncio
async def test_invalid_control_republishes_full_canonical_state() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(
        settings,
        receiver_owner,
        sender,
        FakeService(None),
        resync_timeout_seconds=0.0,
    )

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    sender.messages.clear()

    control_handler = receiver_owner.packet_handlers["control_packet_handler"]
    assert control_handler("/avatar/parameters/PuriPuly_SelfASR", (99,)) is False

    assert len(sender.messages) == 15
    assert {address for address, _value in sender.messages} == {
        f"/avatar/parameters/{name}"
        for name in (
            "PuriPuly_Talk",
            "PuriPuly_Listen",
            "PuriPuly_Trans",
            "PuriPuly_Captions",
            "PuriPuly_PeerAuto",
            "PuriPuly_MuteSync",
            "PuriPuly_ChatboxSource",
            "PuriPuly_SelfSrcLang",
            "PuriPuly_SelfDstLang",
            "PuriPuly_SelfDstLang2",
            "PuriPuly_PeerSrcLang",
            "PuriPuly_PeerDstLang",
            "PuriPuly_SelfASR",
            "PuriPuly_PeerASR",
            "PuriPuly_Translator",
        )
    }
    await integration.close()


@pytest.mark.asyncio
async def test_rejected_dashboard_command_corrects_only_actual_canonical_divergence() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(
        settings,
        receiver_owner,
        sender,
        FakeService(None),
        application=DashboardApplication(),
        osc_state=OscCanonicalState(self_capture=False),
    )

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    sender.messages.clear()

    result = await integration.router.dispatch_packet(
        "/avatar/parameters/PuriPuly_Talk",
        True,
    )

    assert result.applied is False
    assert result.error == "application_rejected"
    assert sender.messages == [("/avatar/parameters/PuriPuly_Talk", False)]
    await integration.close()


@pytest.mark.asyncio
async def test_off_transition_drains_an_admitted_dashboard_command() -> None:
    settings = AppSettingsVNext()
    receiver_owner = FakeReceiverOwner()
    sender = FakeSender()
    application = BlockingDashboardApplication()
    integration = _integration(
        settings,
        receiver_owner,
        sender,
        FakeService(None),
        application=application,
        state_provider=lambda: OscCanonicalState(self_capture=application.completed),
    )

    await integration.configure_connection(
        mode="manual",
        send_port=9020,
        receive_port=9021,
    )
    dispatch_task = asyncio.create_task(
        integration.router.dispatch_packet(
            "/avatar/parameters/PuriPuly_Talk",
            True,
        )
    )
    await application.started.wait()

    off_task = asyncio.create_task(
        integration.configure_connection(
            mode="off",
            send_port=9020,
            receive_port=9021,
        )
    )
    await asyncio.sleep(0)
    assert off_task.done() is False
    application.gate.set()

    result, _ = await asyncio.wait_for(
        asyncio.gather(dispatch_task, off_task),
        timeout=1,
    )
    assert result.applied is False
    assert result.error == "router_disabled"
    assert application.completed is True
    assert integration.connection_mode == "off"
    assert len(sender.messages) == 15
    assert ("/avatar/parameters/PuriPuly_Talk", True) not in sender.messages
    await integration.configure_connection(mode="manual", send_port=9020, receive_port=9021)
    assert ("/avatar/parameters/PuriPuly_Talk", True) in sender.messages[15:]
    await integration.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("loopback", [False, True])
async def test_failed_cached_full_resync_requires_observed_obligation_completion(
    caplog: pytest.LogCaptureFixture,
    loopback: bool,
) -> None:
    import logging
    import socket

    from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE
    from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender

    endpoint = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if loopback else None
    if endpoint is not None:
        endpoint.bind(("127.0.0.1", 0))
        endpoint.settimeout(1.0)
    send_port = endpoint.getsockname()[1] if endpoint is not None else 9050
    udp_sender = VrchatOscUdpSender(host="127.0.0.1", port=send_port) if loopback else None

    class ResyncSender(FakeSender):
        failed_parameter: str | None = None

        def send_message(self, address: str, *values: object) -> None:
            if address.endswith("/PuriPuly_Translator") and self.failed_parameter is not None:
                self.failed_parameter = None
                raise OSError(13, "private OSC resync credential body")
            if udp_sender is not None:
                udp_sender.send_message(address, *values)
            super().send_message(address, *values)

    state = OscCanonicalState()
    sender = ResyncSender()
    integration = _integration(
        AppSettingsVNext(),
        FakeReceiverOwner(),
        sender,
        FakeService(None),
        state_provider=lambda: state,
    )
    caplog.set_level(logging.INFO, logger=control_runtime_module.__name__)
    try:
        await integration.configure_connection(
            mode="manual", send_port=send_port, receive_port=9051
        )
        sender.failed_parameter = "PuriPuly_Translator"
        integration._publish_full()
        failure = integration.output_snapshot()["state_publication"]
        assert failure is not None
        sent_after_failure = len(sender.messages)

        integration.publish_delta()
        assert len(sender.messages) == sent_after_failure
        assert integration.output_snapshot()["state_publication"] == failure

        state = replace(state, self_capture=True)
        integration.publish_delta()
        assert len(sender.messages) == sent_after_failure + 1
        assert integration.output_snapshot()["state_publication"] == failure
        assert not any(
            getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)[0].code == "osc_state_send_recovered"
            for record in caplog.records
            if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
        )

        integration._publish_full()
        assert integration.output_snapshot()["state_publication"] is None
        recovery = next(
            getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)[0]
            for record in caplog.records
            if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
            and getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)[0].code == "osc_state_send_recovered"
        )
        assert recovery.cause_id == failure["diagnostic_id"]
        assert recovery.fields["local_send_completed"] == 15
        assert recovery.fields["remote_delivery"] == "unacknowledged"
        if endpoint is not None:
            assert len(sender.messages) == 45
            for _ in sender.messages:
                packet, _source = endpoint.recvfrom(65535)
                assert packet.startswith(b"/avatar/parameters/PuriPuly")
    finally:
        await integration.close()
        if udp_sender is not None:
            udp_sender.close()
        if endpoint is not None:
            endpoint.close()


@pytest.mark.asyncio
async def test_running_discovery_retry_recovers_lost_target_and_reapplies_without_publication() -> (
    None
):
    settings = AppSettingsVNext()
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(
        OscQueryServiceInfo(
            service_id="VRChat", host="127.0.0.1", osc_send_port=19000, is_vrchat=True
        )
    )
    integration = _integration(settings, receiver, sender, service)
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        bound_receiver = receiver.receiver
        service.info = None
        await integration.query_runtime.refresh(publish_snapshot=False)
        assert integration.output_snapshot()["destination_source"] == "fallback"
        service.info = OscQueryServiceInfo(
            service_id="VRChat", host="127.0.0.1", osc_send_port=19002, is_vrchat=True
        )
        messages = list(sender.messages)
        control_calls = list(receiver.control_calls)
        for _ in range(2):
            receipt = await integration.retry_automatic_discovery()
            assert receipt["status"] == "applied"
            assert receipt["effect"] == "discovery_target_applied"
            assert receipt["output"]["destination_source"] == "discovered"
            assert receipt["output"]["remote_delivery"] == "unacknowledged"
            assert sender.destinations[-1] == ("127.0.0.1", 19002)
            assert receiver.receiver is bound_receiver
            assert receiver.control_calls == control_calls
            assert sender.messages == messages
        assert service.started == 1
        assert service.stopped == 0
        assert len(service.advertisements) == 1
    finally:
        await integration.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_retry_timeout_settles_discovery_resources_without_replacing_receiver(
    started,
) -> None:
    settings = AppSettingsVNext()
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    service = RetryDiscoveryService(None)
    service.fail_discovery = not started
    integration = _integration(settings, receiver, sender, service)
    integration._discovery_timeout_seconds = 0.02
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        bound_receiver = receiver.receiver
        messages = list(sender.messages)
        calls = list(receiver.control_calls)
        service.fail_discovery = False
        service.block_discovery = True
        result = await integration.retry_automatic_discovery()
        assert result["status"] == "degraded"
        assert result["effect"] == "discovery_unavailable"
        assert result["reason"] == "discovery_timeout"
        assert integration.query_runtime.started is started
        assert receiver.receiver is bound_receiver
        assert receiver.control_calls == calls
        assert sender.messages == messages
        assert integration._discovery_retry_task is None
        service.discovery_gate.set()
        service.info = OscQueryServiceInfo(
            service_id="VRChat", host="127.0.0.1", osc_send_port=19002, is_vrchat=True
        )
        assert (await integration.retry_automatic_discovery())["status"] == "applied"
        assert receiver.receiver is bound_receiver
    finally:
        service.discovery_gate.set()
        await integration.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["manual", "off"])
async def test_background_refresh_and_retry_drop_results_after_saved_connection_changes(
    mode,
) -> None:
    settings = AppSettingsVNext()
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    service = DelayedDiscoveryService(None)
    integration = _integration(settings, receiver, sender, service)
    tasks = []
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        before = list(sender.destinations)
        service.block_discovery = True
        service.info = OscQueryServiceInfo(
            service_id="retired-VRChat", host="127.0.0.1", osc_send_port=19003, is_vrchat=True
        )
        background = asyncio.create_task(integration.query_runtime.refresh())
        tasks.append(background)
        await service.discovery_started.wait()
        retry = asyncio.create_task(integration.retry_automatic_discovery())
        tasks.append(retry)
        await asyncio.sleep(0)
        duplicate = await integration.retry_automatic_discovery()
        assert duplicate["reason"] == "discovery_in_progress"
        changed = replace(
            settings,
            intent=replace(
                settings.intent,
                osc=replace(settings.intent.osc, connection_mode=mode, send_port=19004),
            ),
        )
        integration._settings_provider = lambda: changed
        configuring = asyncio.create_task(
            integration.configure_connection(
                mode=mode, send_port=19004, receive_port=changed.intent.osc.receive_port
            )
        )
        tasks.append(configuring)
        service.discovery_gate.set()
        await asyncio.wait_for(background, timeout=1)
        receipt = await asyncio.wait_for(retry, timeout=1)
        await asyncio.wait_for(configuring, timeout=1)
        assert receipt["status"] == "rejected"
        assert receipt["reason"] == "connection_changed"
        assert ("127.0.0.1", 19003) not in sender.destinations[len(before) :]
        assert integration.output_snapshot()["effective_mode"] == mode
        assert integration.query_runtime.started is False
        assert service.started == 1
    finally:
        service.discovery_gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await integration.close()


@pytest.mark.asyncio
async def test_recovery_listener_observation_is_retained_and_invalidated_by_destination_and_settings() -> (
    None
):
    settings = AppSettingsVNext()
    service = FakeService(
        OscQueryServiceInfo(
            service_id="VRChat", host="127.0.0.1", osc_send_port=19000, is_vrchat=True
        )
    )
    integration = _integration(settings, FakeReceiverOwner(), FakeSender(), service)
    observed = []

    async def observe(host, port):
        observed.append((host, port))
        return VrchatOscPresence(True, True, "matched")

    integration._local_listener_probe = observe
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        assert integration.output_snapshot()["local_listener"]["observed_at"] is None
        receipt = await integration.retry_automatic_discovery()
        listener = receipt["output"]["local_listener"]
        assert listener["status"] == "matched"
        assert listener["vrchat_running"] is True
        assert isinstance(listener["observed_at"], float)
        assert listener["observed_at"] > 0
        for _ in range(20):
            assert integration.output_snapshot()["local_listener"] == listener
        assert observed == [("127.0.0.1", 19000)]
        service.info = replace(service.info, osc_send_port=19001)
        await integration.query_runtime.refresh(publish_snapshot=False)
        assert integration.output_snapshot()["local_listener"]["status"] == "unknown"
        assert integration.output_snapshot()["local_listener"]["observed_at"] is None
        await integration.retry_automatic_discovery()
        changed = replace(
            settings,
            intent=replace(
                settings.intent, osc=replace(settings.intent.osc, connection_mode="off")
            ),
        )
        integration._settings_provider = lambda: changed
        assert integration.output_snapshot()["local_listener"]["status"] == "unknown"
        assert len(observed) == 2
    finally:
        await integration.close()


@pytest.mark.asyncio
async def test_completed_retry_without_target_and_unavailable_observation_is_degraded_not_delivery_failure() -> (
    None
):
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    service = FakeService(None)
    integration = _integration(AppSettingsVNext(), receiver, sender, service)

    async def unavailable(_host, _port):
        raise PermissionError("controlled local observation failure")

    integration._local_listener_probe = unavailable
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        bound_receiver = receiver.receiver
        messages = list(sender.messages)
        result = await integration.retry_automatic_discovery()
        assert result["status"] == "degraded"
        assert result["effect"] == "discovery_unavailable"
        assert result["reason"] == "target_not_found"
        output = result["output"]
        assert output["destination_source"] == "fallback"
        assert output["query_service_started"] is True
        assert output["query_failure"] is None
        assert output["local_availability"] == "available"
        assert output["local_listener"]["status"] == "unknown"
        assert output["local_listener"]["vrchat_running"] is None
        assert isinstance(output["local_listener"]["observed_at"], float)
        assert output["remote_delivery"] == "unacknowledged"
        assert receiver.receiver is bound_receiver
        assert sender.messages == messages
    finally:
        await integration.close()


@pytest.mark.asyncio
async def test_stopped_discovery_rejects_retry_even_while_initial_start_is_unsettled() -> None:
    service = DelayedDiscoveryService(None)
    service.block_discovery = True
    integration = _integration(AppSettingsVNext(), FakeReceiverOwner(), FakeSender(), service)
    try:
        await integration.configure(enabled=False)
        await service.discovery_started.wait()
        integration.stop_ingress()
        result = await integration.retry_automatic_discovery()
        assert result["status"] == "rejected"
        assert result["reason"] == "lifecycle_stopped"
        assert result["retry_started"] is False
    finally:
        await integration.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("during_probe", "expected_status", "expected_reason"),
    [
        ("target_lost", "degraded", "target_not_found"),
        ("target_changed", "degraded", "target_not_found"),
        ("unchanged", "applied", None),
        ("config_changed", "rejected", "connection_changed"),
        ("shutdown", "rejected", "lifecycle_stopped"),
    ],
)
async def test_retry_receipt_correlates_target_after_held_listener_observation(
    during_probe, expected_status, expected_reason
) -> None:
    settings = AppSettingsVNext()
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    target = OscQueryServiceInfo(
        service_id="VRChat", host="127.0.0.1", osc_send_port=19000, is_vrchat=True
    )
    service = FakeService(target)
    integration = _integration(settings, receiver, sender, service)
    integration.query_runtime.discovery_poll_interval_seconds = 3600
    entered = asyncio.Event()
    release = asyncio.Event()
    retry = None

    async def observe(_host, _port):
        entered.set()
        await release.wait()
        return VrchatOscPresence(None, None)

    integration._local_listener_probe = observe
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        bound_receiver = receiver.receiver
        resources = list(receiver.control_calls), service.started, service.stopped
        messages = list(sender.messages)
        retry = asyncio.create_task(integration.retry_automatic_discovery())
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not retry.done()
        if during_probe in {"target_lost", "target_changed", "unchanged"}:
            service.info = (
                None
                if during_probe == "target_lost"
                else (
                    replace(target, osc_send_port=19001)
                    if during_probe == "target_changed"
                    else target
                )
            )
            await integration.query_runtime.refresh(publish_snapshot=False)
        elif during_probe == "config_changed":
            changed = replace(
                settings,
                intent=replace(
                    settings.intent,
                    osc=replace(settings.intent.osc, connection_mode="manual"),
                ),
            )
            integration._settings_provider = lambda: changed
        else:
            integration.stop_ingress()
        avatar_queries = service.avatar_queries
        release.set()
        result = await asyncio.wait_for(retry, timeout=1)
        assert result["status"] == expected_status
        assert result.get("reason") == expected_reason
        assert result["retry_started"] is True
        if expected_status == "applied":
            assert result["effect"] == "discovery_target_applied"
            assert result["output"]["local_listener"]["status"] == "unknown"
            assert isinstance(result["output"]["local_listener"]["observed_at"], float)
        elif expected_status == "degraded":
            assert result["effect"] == "discovery_unavailable"
        else:
            assert "effect" not in result
        if during_probe == "target_lost":
            assert result["output"]["discovered_remote_present"] is False
            assert result["output"]["destination_source"] == "fallback"
            assert sender.destinations[-1] == ("127.0.0.1", 9000)
        assert result["output"]["remote_delivery"] == "unacknowledged"
        assert receiver.receiver is bound_receiver
        assert (receiver.control_calls, service.started, service.stopped) == resources
        assert sender.messages == messages
        assert service.avatar_queries == avatar_queries
    finally:
        release.set()
        if retry is not None:
            await asyncio.gather(retry, return_exceptions=True)
        await integration.close()


@pytest.mark.asyncio
async def test_retry_terminal_target_confirmation_wait_is_bounded() -> None:
    target = OscQueryServiceInfo(
        service_id="VRChat", host="127.0.0.1", osc_send_port=19000, is_vrchat=True
    )
    service = DelayedAvatarService(target)
    receiver = FakeReceiverOwner()
    sender = FakeSender()
    integration = _integration(AppSettingsVNext(), receiver, sender, service)
    integration.query_runtime.discovery_poll_interval_seconds = 3600
    integration._discovery_timeout_seconds = 0.05
    entered = asyncio.Event()
    release = asyncio.Event()
    retry = None
    background = None

    async def observe(_host, _port):
        entered.set()
        await release.wait()
        return VrchatOscPresence(None, None)

    integration._local_listener_probe = observe
    try:
        await integration.configure(enabled=False)
        await integration.wait_automatic_query_start()
        bound_receiver = receiver.receiver
        messages = list(sender.messages)
        retry = asyncio.create_task(integration.retry_automatic_discovery())
        await asyncio.wait_for(entered.wait(), timeout=1)
        service.info = replace(target, osc_send_port=19001)
        service.block_avatar_queries = True
        background = asyncio.create_task(integration.query_runtime.refresh(publish_snapshot=False))
        await asyncio.wait_for(service.query_started.wait(), timeout=1)
        release.set()
        result = await asyncio.wait_for(retry, timeout=1)
        assert result["status"] == "degraded"
        assert result["effect"] == "discovery_unavailable"
        assert result["reason"] == "discovery_timeout"
        assert not background.done()
        assert receiver.receiver is bound_receiver
        assert integration.query_runtime.started is True
        assert sender.messages == messages
    finally:
        release.set()
        service.query_gate.set()
        await asyncio.gather(
            *(task for task in (retry, background) if task is not None),
            return_exceptions=True,
        )
        await integration.close()
