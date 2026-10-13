from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.cli import main as cli
from puripuly_heart.cli.host import GuiControlBoundary


def test_generic_auth_login_cannot_put_credential_in_process_arguments(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_instance", lambda _path: pytest.fail("must reject before discovery"))
    credential = "private-credential-unique"

    result = cli.main(
        ["command", "auth.login", "--arguments", json.dumps({"credential": credential})]
    )

    output = capsys.readouterr()
    assert result == 3
    assert json.loads(output.err)["error"]["code"] == "invalid_input"
    assert credential not in output.out + output.err


@pytest.mark.parametrize(("channel", "state"), [("self", "on"), ("self", "off"), ("peer", "off")])
def test_peer_terms_cannot_be_accepted_by_an_unrelated_capture_request(
    tmp_path, capsys, channel, state
):
    result = cli.main(
        [
            "--config",
            str(tmp_path / "settings.json"),
            "capture",
            "set",
            channel,
            state,
            "--accept-peer-terms",
        ]
    )

    output = capsys.readouterr()
    assert result == 3
    assert json.loads(output.err)["error"]["code"] == "invalid_input"
    assert output.out == ""


@pytest.mark.parametrize("value", ["not-a-number", "nan"])
def test_calibration_rejects_invalid_numeric_input_before_discovery(tmp_path, capsys, value):
    result = cli.main(
        [
            "--config",
            str(tmp_path / "settings.json"),
            "overlay",
            "calibrate",
            "change",
            "--field",
            "offset_x",
            "--value",
            value,
        ]
    )

    output = capsys.readouterr()
    assert result == 3
    assert json.loads(output.err)["error"]["code"] == "invalid_input"
    assert output.out == ""


def test_auth_login_stdin_keeps_identity_and_credential_out_of_output(monkeypatch, capsys):
    credential = "private-credential-unique"
    identity = "private-identity-unique"
    submitted = []
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"qq_identity": identity, "credential": credential})),
    )

    async def fake_request(_record, method, arguments=None, *, timeout):
        assert method == "submit"
        submitted.append(arguments)
        return {
            "status": "applied",
            "terminal": True,
            "instance_id": "session",
            "operation_id": "operation",
        }

    monkeypatch.setattr(cli, "request", fake_request)
    assert cli.main(["auth", "login", "qq", "--stdin"]) == 0
    assert submitted[0]["values"] == {
        "provider": "qq",
        "qq_identity": identity,
        "credential": credential,
        "open_browser": False,
    }
    output = capsys.readouterr()
    assert credential not in output.out + output.err
    assert identity not in output.out + output.err
    assert json.loads(output.out)["status"] == "applied"


def test_oauth_without_browser_reports_matching_challenge_and_unfinished_operation(
    monkeypatch, capsys
):
    submitted = []
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))

    async def fake_request(_record, method, arguments=None, *, timeout):
        if method == "submit":
            submitted.append(arguments)
            return {"status": "accepted", "terminal": False, "operation_id": "auth-operation"}
        if method == "wait":
            return {"status": "running", "terminal": False, "operation_id": "auth-operation"}
        assert method == "query"
        return {
            "challenge": {
                "operation_id": "auth-operation",
                "provider": "discord",
                "authorization_url": "https://authorization.example/callback",
            }
        }

    monkeypatch.setattr(cli, "request", fake_request)
    assert cli.main(["auth", "login", "discord"]) == 5
    assert submitted[0]["values"] == {"provider": "discord", "open_browser": False}
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "action_required"
    assert result["operation_status"] == "running"
    assert result["operation_id"] == "auth-operation"
    assert result["terminal"] is False
    assert result["authorization_url"].startswith("https://")


@pytest.mark.parametrize(
    ("status", "terminal", "waiting", "expected"),
    [
        ("accepted", False, False, 0),
        ("running", False, True, 7),
        ("action_required", False, True, 5),
        ("action_required", True, True, 5),
        ("applied", True, True, 0),
        ("degraded", True, True, 4),
        ("persistence_failed", True, True, 4),
        ("cancelled", True, True, 6),
    ],
)
def test_operation_terminal_discriminant_controls_cli_exit(status, terminal, waiting, expected):
    assert cli._status_code({"status": status, "terminal": terminal}, waiting=waiting) == expected


def test_unsupported_cancel_does_not_report_success_or_complete_operation(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))

    async def fake_request(_record, method, arguments=None, *, timeout):
        assert method == "cancel"
        return {"status": "running", "terminal": False, "cancellation": "unsupported"}

    monkeypatch.setattr(cli, "request", fake_request)
    assert cli.main(["operation", "cancel", "active-operation"]) == 4
    result = json.loads(capsys.readouterr().out)
    assert result["terminal"] is False
    assert result["cancellation"] == "unsupported"


@pytest.mark.asyncio
async def test_stop_wait_does_not_treat_endpoint_disappearance_as_host_exit(monkeypatch, capsys):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    record = SimpleNamespace(instance_id="session")
    config = Path("isolated-settings.json")
    args = SimpleNamespace(config=config, timeout=0.08, no_wait=False)
    monkeypatch.setattr("puripuly_heart.core.control_instance.discover", lambda _path: None)

    async def fake_request(_record, method, arguments=None, *, timeout):
        if method == "query":
            return {"host_pid": process.pid}
        assert method == "submit"
        return {"status": "accepted", "terminal": False, "operation_id": "stop-id"}

    monkeypatch.setattr(cli, "request", fake_request)
    try:
        assert await cli._stop(record, args) == 7
        assert json.loads(capsys.readouterr().out)["status"] == "unknown_execution"
    finally:
        if process.poll() is None:
            process.terminate()
            await asyncio.to_thread(process.wait, 5)


@pytest.mark.asyncio
async def test_remote_gui_stop_closes_presentation_after_ordered_runtime_shutdown():
    events = []

    class Host:
        async def wait_for_stop(self):
            events.append("stop-requested")

        async def close(self):
            events.append("runtime-closed")

    class Boundary:
        async def close_presentation(self):
            assert events == ["stop-requested", "runtime-closed"]
            events.append("presentation-closed")

    gui = object.__new__(GuiControlBoundary)
    gui._host = Host()
    gui._boundary = Boundary()
    gui._close_task = None
    await gui._stop_on_request()
    assert events == ["stop-requested", "runtime-closed", "presentation-closed"]


def test_calibration_numeric_error_does_not_export_supplied_input(tmp_path, capsys, monkeypatch):
    sentinel = "PRIVATE_CALIBRATION_SENTINEL"
    monkeypatch.setattr(
        cli, "_instance", lambda _: pytest.fail("invalid numeric input must fail before discovery")
    )
    result = cli.main(
        [
            "--config",
            str(tmp_path / "settings.json"),
            "overlay",
            "calibrate",
            "change",
            "--field",
            "offset_x",
            "--value",
            sentinel,
        ]
    )
    output = capsys.readouterr()
    assert result == 3
    assert json.loads(output.err)["error"]["code"] == "invalid_input"
    assert sentinel not in output.out + output.err
