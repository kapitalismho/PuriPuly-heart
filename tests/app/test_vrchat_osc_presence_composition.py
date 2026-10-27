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
        self.notices: list[bool] = []

    def set_dashboard_vrchat_osc_notice(self, active: bool) -> None:
        super().set_dashboard_vrchat_osc_notice(active)
        self.notices.append(active)
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
    probed_ports: asyncio.Queue[int] = asyncio.Queue()

    def probe(*, port: int):
        return probe_vrchat_osc_presence(
            port=port,
            process_iter=lambda *_args: [SimpleNamespace(info={"pid": 22, "name": "VRChat.exe"})],
            net_connections=lambda **_kwargs: [
                SimpleNamespace(laddr=SimpleNamespace(port=value), pid=22)
                for value in listening_ports
            ],
        )

    adapter = PsutilVrchatOscPresenceAdapter(probe=probe)

    class RecordingPresence:
        async def should_prompt_enable_osc(self, *, port: int) -> bool | None:
            result = await adapter.should_prompt_enable_osc(port=port)
            probed_ports.put_nowait(port)
            return result

    presentation = NoticePresentation()
    app = compose_application_runtime(
        presentation=presentation,
        config_path=path,
        vrchat_osc_presence=RecordingPresence(),
    )
    presentation.application = app
    control = app.control()
    control.bind_instance("osc-presence-regression")
    presence = app._runtime_shutdown.vrchat_presence()
    integration = app._runtime_shutdown.vrc_mic_sync()
    assert presence is not None
    assert integration is not None
    presence.interval_seconds = 0.01
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
        async with asyncio.timeout(5):
            while presentation.projections.get("vrchat_osc", False) is not expected:
                presentation.notice_changed.clear()
                await presentation.notice_changed.wait()

    async def assert_probed(port: int) -> None:
        async with asyncio.timeout(5):
            while await probed_ports.get() != port:
                pass

    async def set_mode(mode: str, receive_port: int) -> None:
        submitted = await control.submit(
            "settings.apply",
            {"changes": {"osc.connection": {"mode": mode, "receive_port": receive_port}}},
            request_id=f"osc-presence-{mode}",
        )
        result = await control.wait(submitted["operation_id"], timeout=10)
        assert result["status"] == "applied"

    try:
        assert presentation.projections.get("vrchat_osc", False) is False
        await app.start()
        await app.prepare_runtime_after_launch()
        await set_mode("automatic", 9001)
        await asyncio.wait_for(discovery_started.wait(), timeout=5)
        assert presentation.projections.get("vrchat_osc", False) is False
        await assert_notice(False)

        listening_ports.add(9020)
        discovery_ready.set()
        await integration.wait_automatic_query_start()
        await assert_probed(9020)
        await assert_notice(False)

        service_info = replace(service_info, osc_send_port=9022)
        listening_ports.clear()
        listening_ports.add(9022)
        await integration.query_runtime.refresh()
        await assert_probed(9022)
        await assert_notice(False)

        service_info = None
        listening_ports.clear()
        listening_ports.add(9000)
        await integration.query_runtime.refresh()
        await assert_probed(9000)
        await assert_notice(False)

        listening_ports.clear()
        await assert_notice(True)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            manual_receive_port = reservation.getsockname()[1]
        await set_mode("manual", manual_receive_port)
        assert integration.receiver is not None
        await assert_probed(9123)
        listening_ports.add(9123)
        await assert_notice(False)

        listening_ports.clear()
        await assert_notice(True)
        await set_mode("off", integration.effective_receive_port)
        assert presentation.projections["vrchat_osc"] is False
        await assert_notice(False)
    finally:
        discovery_ready.set()
        await app.stop()


class ControlledPresence:
    def __init__(self) -> None:
        self.probes: asyncio.Queue[tuple[int, asyncio.Future[bool]]] = asyncio.Queue()
        self.result: bool | None = None

    async def should_prompt_enable_osc(self, *, port: int) -> bool:
        if self.result is not None:
            return self.result
        result = asyncio.get_running_loop().create_future()
        self.probes.put_nowait((port, result))
        return await result

    async def next_probe(self) -> asyncio.Future[bool]:
        port, result = await asyncio.wait_for(self.probes.get(), timeout=5)
        assert port == 9123
        return result


@pytest.fixture
async def controlled_presence_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    monkeypatch.setattr(VrchatOscUdpSender, "send_message", lambda *_args: None)
    settings = AppSettingsVNext()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        receive_port = reservation.getsockname()[1]
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(
                settings.intent.osc,
                connection_mode="manual",
                port=9123,
                send_port=9123,
                receive_port=receive_port,
            ),
            telemetry=replace(settings.intent.telemetry, enabled=False),
        ),
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
    presentation = NoticePresentation()
    boundary = ControlledPresence()
    app = compose_application_runtime(
        presentation=presentation,
        config_path=path,
        vrchat_osc_presence=boundary,
    )
    presentation.application = app
    control = app.control()
    control.bind_instance("osc-presence-transition-regression")
    presence = app._runtime_shutdown.vrchat_presence()
    integration = app._runtime_shutdown.vrc_mic_sync()
    assert presence is not None
    assert integration is not None
    discovery_started = asyncio.Event()
    discovery_ready = asyncio.Event()

    async def discover():
        discovery_started.set()
        await discovery_ready.wait()
        return None

    async def advertise(_advertisement):
        return None

    async def unadvertise():
        return None

    async def query(_service):
        return {}

    integration.query_runtime.service = CallbackOscQueryService(
        discover, advertise, unadvertise, query
    )

    async def set_mode(mode: str) -> None:
        submitted = await control.submit(
            "settings.apply",
            {"changes": {"osc.connection": {"mode": mode, "receive_port": receive_port}}},
            request_id=f"osc-presence-{mode}",
        )
        result = await control.wait(submitted["operation_id"], timeout=10)
        assert result["status"] == "applied"

    try:
        await app.start()
        await app.prepare_runtime_after_launch()
        yield SimpleNamespace(
            app=app,
            boundary=boundary,
            presence=presence,
            integration=integration,
            presentation=presentation,
            set_mode=set_mode,
            discovery_started=discovery_started,
        )
    finally:
        discovery_ready.set()
        await app.stop()
        assert presence.task is None
        assert integration.receiver is None
        assert integration.query_runtime.started is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "automatic"])
async def test_completed_connection_transition_clears_notice_without_rescheduling(
    controlled_presence_runtime, mode
) -> None:
    runtime = controlled_presence_runtime
    probe = await runtime.boundary.next_probe()
    runtime.presentation.notice_changed.clear()
    probe.set_result(True)
    await asyncio.wait_for(runtime.presentation.notice_changed.wait(), timeout=5)
    assert runtime.presentation.projections["vrchat_osc"] is True

    await runtime.set_mode(mode)
    connection = runtime.integration.output_snapshot()
    assert connection["effective_mode"] == mode
    assert connection["local_availability"] == ("off" if mode == "off" else "starting")
    assert runtime.presentation.projections["vrchat_osc"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reenable", [False, True])
async def test_delayed_probe_cannot_publish_across_off_and_same_port_reenable(
    controlled_presence_runtime, reenable
) -> None:
    runtime = controlled_presence_runtime
    stale_probe = await runtime.boundary.next_probe()
    await runtime.set_mode("off")
    assert runtime.integration.output_snapshot()["effective_mode"] == "off"
    if reenable:
        await runtime.set_mode("manual")
        assert runtime.integration.output_snapshot()["effective_send_port"] == 9123
    stale_probe.set_result(True)
    await asyncio.sleep(0)
    assert True not in runtime.presentation.notices

    if not reenable:
        await runtime.set_mode("manual")
    current_probe = await runtime.boundary.next_probe()
    runtime.presentation.notice_changed.clear()
    runtime.presence.interval_seconds = 0.01
    runtime.boundary.result = True
    current_probe.set_result(True)
    await asyncio.wait_for(runtime.presentation.notice_changed.wait(), timeout=5)
    assert runtime.presentation.projections["vrchat_osc"] is True

    runtime.presentation.notice_changed.clear()
    runtime.boundary.result = False
    await asyncio.wait_for(runtime.presentation.notice_changed.wait(), timeout=5)
    assert runtime.presentation.projections["vrchat_osc"] is False
