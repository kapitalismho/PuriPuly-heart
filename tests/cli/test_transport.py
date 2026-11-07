from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from uuid import UUID

import pytest

from puripuly_heart.app.services.application_control import UnknownOperationError
from puripuly_heart.app.services.application_control_events import ControlEvents
from puripuly_heart.cli.transport import (
    MAX_REQUEST_BYTES,
    ControlServer,
    ControlTransportError,
    follow,
    request,
)
from puripuly_heart.domain.events import STTSessionState, UIEvent, UIEventType
from puripuly_heart.domain.models import OSCMessage, Transcript, Translation


@dataclass(frozen=True)
class _Record:
    port: int
    instance_id: str = "correct-instance"
    token: str = "local-secret-token"


class _Control:
    def __init__(self):
        self.calls = 0

    def capabilities(self):
        self.calls += 1
        return {"queries": ["app.status"]}

    async def query(self, name, arguments):
        self.calls += 1
        return {"status": "ready", "name": name}

    async def submit(self, name, arguments, *, request_id, expected_revision=None):
        self.calls += 1
        return {
            "instance_id": "correct-instance",
            "operation_id": request_id,
            "status": "accepted",
            "terminal": False,
        }


async def _raw(port: int, message: bytes):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(message)
        await writer.drain()
        return json.loads(await reader.readline())
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_security_checks_precede_application_dispatch():
    control = _Control()
    server = ControlServer(control, instance_id="correct-instance", token="local-secret-token")
    port = await server.start()
    try:
        base = {
            "protocol": 1,
            "instance_id": "correct-instance",
            "token": "local-secret-token",
            "method": "capabilities",
            "arguments": {},
        }
        for change, code in [
            ({"protocol": 2}, "unsupported_protocol"),
            ({"instance_id": "previous-instance"}, "stale_instance"),
            ({"token": "incorrect-token"}, "unauthorized"),
            ({"method": "__dict__"}, "unknown_method"),
        ]:
            response = await _raw(port, json.dumps(base | change).encode() + b"\n")
            assert response["error"]["code"] == code
        response = await _raw(port, b"{" + b"a" * (MAX_REQUEST_BYTES + 1) + b"\n")
        assert response["error"]["code"] == "request_too_large"
        for arguments in (
            {"topics": ["transcript"], "include_transcripts": "true"},
            {"topics": ["operation"], "after": "0"},
            {"topics": ["operation"], "channel": "other"},
        ):
            response = await _raw(
                port,
                json.dumps(base | {"method": "subscribe", "arguments": arguments}).encode() + b"\n",
            )
            assert response["error"]["code"] == "malformed_request"
        assert control.calls == 0
        assert (await request(_Record(port), "capabilities"))["queries"] == ["app.status"]
        assert control.calls == 1
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_stale_endpoint_and_ingress_freeze_do_not_replay_mutation():
    control = _Control()
    server = ControlServer(control, instance_id="correct-instance", token="local-secret-token")
    port = await server.start()
    try:
        with pytest.raises(ControlTransportError, match="rediscover") as error:
            await request(
                _Record(port, instance_id="previous-instance"),
                "submit",
                {"name": "capture.set", "values": {"enabled": True}, "request_id": "once"},
            )
        assert error.value.code == "stale_instance"
        assert control.calls == 0
        server.freeze()
        with pytest.raises(ControlTransportError):
            await request(_Record(port), "query", {"name": "app.status"})
        assert control.calls == 0
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_client_timeout_does_not_cancel_accepted_server_execution():
    class SlowControl(_Control):
        def __init__(self):
            super().__init__()
            self.completed = asyncio.Event()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(self, name, arguments, *, request_id, expected_revision=None):
            self.calls += 1
            self.entered.set()
            await self.release.wait()
            self.completed.set()
            return {
                "instance_id": "correct-instance",
                "operation_id": request_id,
                "status": "accepted",
                "terminal": False,
            }

    control = SlowControl()
    server = ControlServer(control, instance_id="correct-instance", token="local-secret-token")
    port = await server.start()
    client = asyncio.create_task(
        request(
            _Record(port),
            "submit",
            {
                "name": "capture.set",
                "values": {"channel": "self", "enabled": False},
                "request_id": "do-not-replay",
            },
            timeout=0.1,
        )
    )
    try:
        await asyncio.wait_for(control.entered.wait(), 1)
        with pytest.raises(ControlTransportError) as error:
            await client
        assert error.value.code == "unknown_execution"
        assert not control.completed.is_set()
        control.release.set()
        await asyncio.wait_for(control.completed.wait(), 1)
        assert control.calls == 1
    finally:
        control.release.set()
        await asyncio.gather(client, return_exceptions=True)
        await server.close()


@pytest.mark.asyncio
async def test_shutdown_waits_until_stop_receipt_is_consumed():
    class StopControl(_Control):
        def __init__(self):
            super().__init__()
            self.stop_requested = asyncio.Event()

        async def submit(self, name, arguments, *, request_id, expected_revision=None):
            self.stop_requested.set()
            return {
                "instance_id": "correct-instance",
                "operation_id": request_id,
                "status": "accepted",
                "terminal": False,
            }

    control = StopControl()
    server = ControlServer(control, instance_id="correct-instance", token="local-secret-token")
    port = await server.start()

    async def shutdown():
        await control.stop_requested.wait()
        await server.wait_for_idle()
        await server.close()

    task = asyncio.create_task(shutdown())
    try:
        result = await request(
            _Record(port), "submit", {"name": "app.stop", "values": {}, "request_id": "stop-once"}
        )
        assert result["status"] == "accepted"
        await asyncio.wait_for(task, 1)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_event_follow_reports_host_loss_instead_of_success():
    class EventControl(_Control):
        def subscribe(self, *, topics, channel, include_transcripts, include_translations, after):
            async def stream():
                yield {"topic": "operation", "sequence": 1, "status": "applied"}
                await asyncio.Event().wait()

            return stream()

    server = ControlServer(
        EventControl(), instance_id="correct-instance", token="local-secret-token"
    )
    port = await server.start()
    stream = follow(_Record(port), {"topics": ["operation"]})
    try:
        event = await anext(stream)
        assert event["status"] == "applied"
        closing = asyncio.create_task(server.close())
        with pytest.raises(ControlTransportError) as error:
            await asyncio.wait_for(anext(stream), 2)
        assert error.value.code == "connection_lost"
        await asyncio.wait_for(closing, 2)
    finally:
        await stream.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_ui_event_subscriptions_project_only_explicit_content_over_loopback():
    events = ControlEvents()
    utterance_id = UUID("00000000-0000-0000-0000-000000000193")
    events.publish_ui(
        UIEvent(
            type=UIEventType.SESSION_STATE_CHANGED,
            payload=STTSessionState.STREAMING,
            channel="peer",
            source="offline",
        )
    )
    events.publish_ui(
        UIEvent(
            type=UIEventType.SESSION_STATE_CHANGED,
            payload="OFFLINE_UNKNOWN_STATE_PRIVATE_MARKER",
            channel="peer",
            source="offline",
        )
    )
    events.publish_ui(
        UIEvent(
            type=UIEventType.OSC_SENT,
            utterance_id=utterance_id,
            payload=OSCMessage(
                utterance_id=utterance_id,
                text="OFFLINE_OSC_PRIVATE_MARKER",
                created_at=0.0,
            ),
            channel="peer",
            source="offline",
        )
    )
    events.publish_ui(
        UIEvent(
            type=UIEventType.TRANSCRIPT_FINAL,
            utterance_id=utterance_id,
            payload=Transcript(
                utterance_id=utterance_id,
                text="OFFLINE_TRANSCRIPT_PRIVATE_MARKER",
                is_final=True,
                channel="peer",
            ),
            source="offline",
        )
    )
    events.publish_ui(
        UIEvent(
            type=UIEventType.TRANSLATION_DONE,
            utterance_id=utterance_id,
            payload=Translation(
                utterance_id=utterance_id,
                text="OFFLINE_TRANSLATION_PRIVATE_MARKER",
                channel="peer",
            ),
            source="offline",
        )
    )
    server = ControlServer(events, instance_id="correct-instance", token="local-secret-token")
    port = await server.start()
    topics = ["session_state_changed", "osc_sent", "transcript", "translation"]

    async def receive(**flags):
        stream = follow(_Record(port), {"topics": topics, "after": 0, **flags})
        try:
            return [await asyncio.wait_for(anext(stream), 2) for _ in range(events.sequence)]
        finally:
            await stream.aclose()

    try:
        masked = await receive()
        assert masked[0]["state"] == STTSessionState.STREAMING.value
        assert "state" not in masked[1]
        assert masked[2]["sequence"] == 3
        assert masked[2]["topic"] == "osc_sent"
        assert masked[2]["channel"] == "peer"
        assert masked[2]["utterance_id"] == str(utterance_id)
        assert masked[2]["source"] == "unknown"
        assert masked[2]["instance_id"] is None
        assert "OFFLINE_" not in json.dumps(masked)

        transcripts = await receive(include_transcripts=True, include_translations=False)
        assert transcripts[3]["text"] == "OFFLINE_TRANSCRIPT_PRIVATE_MARKER"
        assert "text" not in transcripts[4]
        assert "OFFLINE_OSC_PRIVATE_MARKER" not in json.dumps(transcripts)

        translations = await receive(include_transcripts=False, include_translations=True)
        assert "text" not in translations[3]
        assert translations[4]["text"] == "OFFLINE_TRANSLATION_PRIVATE_MARKER"
        assert "OFFLINE_OSC_PRIVATE_MARKER" not in json.dumps(translations)

        both = await receive(include_transcripts=True, include_translations=True)
        assert both[3]["text"] == "OFFLINE_TRANSCRIPT_PRIVATE_MARKER"
        assert both[4]["text"] == "OFFLINE_TRANSLATION_PRIVATE_MARKER"
        assert "OFFLINE_OSC_PRIVATE_MARKER" not in json.dumps(both)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_expired_operation_identity_has_specific_error_without_dispatch_details():
    class ExpiredControl(_Control):
        async def operation(self, operation_id):
            raise UnknownOperationError("private application detail")

    server = ControlServer(
        ExpiredControl(), instance_id="correct-instance", token="local-secret-token"
    )
    port = await server.start()
    try:
        with pytest.raises(ControlTransportError) as error:
            await request(_Record(port), "operation", {"operation_id": "prior-host-operation"})
        assert error.value.code == "unknown_or_expired_operation"
        assert "private application detail" not in str(error.value)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_client_rejects_oversized_request_before_dispatch():
    control = _Control()
    server = ControlServer(control, instance_id="correct-instance", token="local-secret-token")
    port = await server.start()
    try:
        with pytest.raises(ControlTransportError) as error:
            await request(
                _Record(port),
                "submit",
                {
                    "name": "text.submit",
                    "values": {"text": "x" * MAX_REQUEST_BYTES},
                    "request_id": "oversized",
                },
            )
        assert error.value.code == "request_too_large"
        assert control.calls == 0
    finally:
        await server.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows authenticated instance discovery")
@pytest.mark.asyncio
async def test_cli_subprocess_discovers_authenticates_and_keeps_valid_queries_after_rejection(
    tmp_path, monkeypatch
):
    from puripuly_heart.core.control_instance import acquire, discover

    profile = tmp_path / "isolated-profile"
    for name in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME"):
        directory = profile / name.lower()
        directory.mkdir(parents=True)
        monkeypatch.setenv(name, str(directory))
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PURIPULY_HEART_", "PYTHON"))
    }
    settings = tmp_path / "synthetic-settings.json"
    instance_id = "00000000-0000-0000-0000-000000000295"
    token = "synthetic-control-token-" + "a" * 32
    control = _Control()
    server = ControlServer(control, instance_id=instance_id, token=token)
    port = await server.start()

    async def invoke(*arguments):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-B",
            "-m",
            "puripuly_heart.main",
            "cli",
            "--config",
            str(settings),
            *arguments,
            cwd=tmp_path,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert token not in (stdout + stderr).decode("utf-8")
        return process.returncode, stdout.decode("utf-8"), stderr.decode("utf-8")

    try:
        with acquire(settings) as lease:
            lease.publish(port=port, token=token, instance_id=instance_id)
            code, stdout, stderr = await invoke("capabilities")
            assert code == 0, stderr
            assert stderr == ""
            assert json.loads(stdout) == {"output_version": 1, "queries": ["app.status"]}

            private = "SYNTHETIC_PRIVATE_CREDENTIAL"
            code, stdout, stderr = await invoke(
                "command", "auth.login", "--arguments", json.dumps({"credential": private})
            )
            assert code == 3
            assert stdout == ""
            assert json.loads(stderr)["error"]["code"] == "invalid_input"
            assert private not in stderr
            assert control.calls == 1

            code, stdout, stderr = await invoke("app", "status")
            assert code == 0, stderr
            assert stderr == ""
            assert json.loads(stdout) == {
                "output_version": 1,
                "instance_id": instance_id,
                "port": port,
                "status": "ready",
                "name": "app.status",
            }
            assert control.calls == 2
        assert discover(settings) is None
        code, stdout, stderr = await invoke("capabilities")
        assert code == 3
        assert stdout == ""
        assert json.loads(stderr)["error"]["code"] == "instance_not_found"
        assert control.calls == 2
        assert not settings.exists()
        assert not list(profile.rglob("secrets*"))
        assert not list(profile.rglob("*.log"))
    finally:
        await server.close()
