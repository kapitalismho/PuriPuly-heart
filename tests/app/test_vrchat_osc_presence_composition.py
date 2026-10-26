from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.vrchat_osc_presence import PsutilVrchatOscPresenceAdapter
from puripuly_heart.composition.application_runtime import compose_application_runtime
from puripuly_heart.composition.headless_application import HeadlessApplicationPresentation
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core.osc.oscquery import CallbackOscQueryService
from puripuly_heart.core.osc.oscquery_contract import OscQueryServiceInfo
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.osc.vrchat_osc_presence import probe_vrchat_osc_presence


class NoticePresentation(HeadlessApplicationPresentation):
    def __init__(self) -> None:
        super().__init__()
        self.notice_changed = asyncio.Event()

    def set_dashboard_vrchat_osc_notice(self, active: bool) -> None:
        super().set_dashboard_vrchat_osc_notice(active)
        self.notice_changed.set()


@pytest.mark.asyncio
async def test_presence_notice_tracks_effective_connection_and_discovery_recovery(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(VrchatOscUdpSender, "send_message", lambda *_args: None)
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(settings.intent.osc, connection_mode="off", port=9123, send_port=9123),
            telemetry=replace(settings.intent.telemetry, enabled=False),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    listening_ports: set[int] = set()

    def probe(*, port: int):
        return probe_vrchat_osc_presence(
            port=port,
            process_iter=lambda *_args: [SimpleNamespace(info={"pid": 22, "name": "VRChat.exe"})],
            net_connections=lambda **_kwargs: [
                SimpleNamespace(laddr=SimpleNamespace(port=value), pid=22)
                for value in listening_ports
            ],
        )

    presentation = NoticePresentation()
    app = compose_application_runtime(
        presentation=presentation,
        config_path=path,
        vrchat_osc_presence=PsutilVrchatOscPresenceAdapter(probe=probe),
    )
    presentation.application = app
    control = app.control()
    control.bind_instance("osc-presence-regression")
    presence = app._runtime_shutdown.vrchat_presence()
    integration = app._runtime_shutdown.vrc_mic_sync()
    assert presence is not None
    assert integration is not None
    discovery_started = asyncio.Event()
    discovery_ready = asyncio.Event()
    service_info = OscQueryServiceInfo(
        service_id="synthetic-vrchat", host="127.0.0.1", osc_send_port=9020, is_vrchat=True
    )

    async def discover():
        discovery_started.set()
        await discovery_ready.wait()
        return service_info

    async def advertise(_advertisement):
        return None

    async def unadvertise():
        return None

    async def query(_service):
        return {}

    integration.query_runtime.service = CallbackOscQueryService(
        discover, advertise, unadvertise, query
    )

    async def assert_notice(expected: bool) -> None:
        presence.publish(not expected)
        presentation.notice_changed.clear()
        presence.schedule()
        await asyncio.wait_for(presentation.notice_changed.wait(), timeout=5)
        assert presentation.projections["vrchat_osc"] is expected

    async def set_mode(mode: str, receive_port: int) -> None:
        submitted = await control.submit(
            "settings.apply",
            {"changes": {"osc.connection": {"mode": mode, "receive_port": receive_port}}},
            request_id=f"osc-presence-{mode}",
        )
        result = await control.wait(submitted["operation_id"], timeout=10)
        assert result["status"] == "applied"

    try:
        await assert_notice(False)
        await app.start()
        await set_mode("automatic", 9001)
        await asyncio.wait_for(discovery_started.wait(), timeout=5)
        await assert_notice(False)

        listening_ports.add(9020)
        discovery_ready.set()
        await integration.wait_automatic_query_start()
        await assert_notice(False)

        service_info = replace(service_info, osc_send_port=9022)
        listening_ports.clear()
        listening_ports.add(9022)
        await integration.query_runtime.refresh()
        await assert_notice(False)

        service_info = None
        listening_ports.clear()
        listening_ports.add(9000)
        await integration.query_runtime.refresh()
        await assert_notice(False)

        listening_ports.clear()
        await assert_notice(True)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            manual_receive_port = reservation.getsockname()[1]
        await set_mode("manual", manual_receive_port)
        assert integration.receiver is not None
        listening_ports.add(9123)
        await assert_notice(False)

        listening_ports.clear()
        await assert_notice(True)
        await set_mode("off", integration.effective_receive_port)
        await assert_notice(False)
    finally:
        discovery_ready.set()
        await app.stop()
