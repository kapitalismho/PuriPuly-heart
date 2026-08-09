from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
)
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


class DashboardApplication:
    async def set_stt_enabled(self, _enabled: bool) -> None:
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


def _integration(
    settings: AppSettings,
    receiver_owner: FakeReceiverOwner,
    sender: FakeSender,
    service: FakeService,
    *,
    application: object | None = None,
    osc_state: OscCanonicalState | None = None,
    state_provider: Callable[[], OscCanonicalState] | None = None,
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

    assert integration.connection_mode == "automatic"
    assert receiver_owner.control_calls[-1] == (True, "127.0.0.1", 0, True)
    assert service.advertisements is not None
    assert service.advertisements[0].port == 49152
    assert sender.destinations[-1] == ("127.0.0.1", 9010)
    assert len(sender.messages) == 15
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
    assert len(sender.messages) == 30

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
async def test_automatic_refresh_keeps_configured_manual_fallback_port() -> None:
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
    assert sender.destinations[-1] == ("127.0.0.1", 9123)

    await integration.query_runtime.refresh()

    assert sender.destinations[-1] == ("127.0.0.1", 9123)
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
    assert sender.destinations[-1] == ("127.0.0.1", 9030)

    service.info = None
    await integration.query_runtime.refresh()
    assert sender.destinations[-1] == ("127.0.0.1", 9130)

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
    integration = _integration(settings, receiver_owner, sender, FakeService(None))

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
