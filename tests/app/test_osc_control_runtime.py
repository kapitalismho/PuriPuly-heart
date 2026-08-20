from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from puripuly_heart.app.ports.osc_control import OSC_PARAMETER_DEFINITIONS
from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
)
from puripuly_heart.app.services.osc import control_runtime as control_runtime_module
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState
from puripuly_heart.config.settings import AppSettings, materialize_translation_settings


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
    settings: AppSettings,
    receiver_owner: FakeReceiverOwner,
    sender: FakeSender,
    service: FakeService,
    *,
    application: object | None = None,
    osc_state: OscCanonicalState | None = None,
    state_provider: Callable[[], OscCanonicalState] | None = None,
    resync_timeout_seconds: float = 1.5,
) -> OscControlIntegrationOwner:
    return OscControlIntegrationOwner(
        receiver_owner=receiver_owner,
        settings_provider=lambda: settings,
        apply_settings=lambda _settings: None,
        application_provider=lambda: application,
        sender_provider=lambda: sender,
        state_provider=state_provider or (lambda: osc_state or OscCanonicalState()),
        language_state_provider=lambda: ("ko", "en", "en", "ko"),
        translation_model_normalizer=materialize_translation_settings,
        query_service=service,
        resync_timeout_seconds=resync_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_integration_shares_dynamic_receiver_and_transitions_modes() -> None:
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    first_settings = AppSettings()
    second_settings = AppSettings()
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
    settings = AppSettings()
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

    await integration.close()


@pytest.mark.asyncio
async def test_manual_connection_packet_before_snapshot_cannot_settle_parameter() -> None:
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
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
    settings = AppSettings()
    settings.osc.connection_mode = "off"
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
    settings = AppSettings()
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
            "PuriPuly_PeerSrcLang",
            "PuriPuly_PeerDstLang",
            "PuriPuly_SelfASR",
            "PuriPuly_PeerASR",
            "PuriPuly_Translator",
            "PuriPuly_Fallback",
        )
    }
    await integration.close()


@pytest.mark.asyncio
async def test_rejected_dashboard_command_republishes_actual_full_canonical_state() -> None:
    settings = AppSettings()
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
    assert len(sender.messages) == 15
    await integration.close()


@pytest.mark.asyncio
async def test_off_transition_drains_an_admitted_dashboard_command() -> None:
    settings = AppSettings()
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
    assert result.applied is True
    assert result.error is None
    assert application.completed is True
    assert integration.connection_mode == "off"
    assert len(sender.messages) == 16
    assert sender.messages[-1] == ("/avatar/parameters/PuriPuly_Talk", True)
    await integration.close()
