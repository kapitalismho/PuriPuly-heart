from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from puripuly_heart.cli import main as cli


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


@pytest.mark.parametrize(
    ("status", "terminal", "expected"),
    [
        ("running", False, 7),
        ("action_required", False, 5),
        ("action_required", True, 5),
        ("applied", True, 0),
        ("degraded", True, 4),
        ("rejected", True, 4),
        ("persistence_failed", True, 4),
        ("failed", True, 4),
        ("cancelled", True, 6),
        ("interrupted", True, 6),
    ],
)
def test_operation_wait_preserves_receipt_and_reports_exit(
    monkeypatch, capsys, status, terminal, expected
):
    record = SimpleNamespace(instance_id="session")
    receipt = {
        "status": status,
        "terminal": terminal,
        "instance_id": record.instance_id,
        "operation_id": "operation",
    }
    monkeypatch.setattr(cli, "_instance", lambda _path: record)

    async def fake_request(actual_record, method, arguments=None, *, timeout):
        assert actual_record is record
        assert method == "wait"
        assert arguments["operation_id"] == "operation"
        return receipt

    monkeypatch.setattr(cli, "request", fake_request)

    assert cli.main(["operation", "wait", "operation"]) == expected

    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {"output_version": 1, **receipt}


@pytest.mark.parametrize("no_wait", [False, True], ids=["wait", "no-wait"])
def test_submission_distinguishes_acceptance_from_terminal_completion(monkeypatch, capsys, no_wait):
    record = SimpleNamespace(instance_id="session")
    accepted = {
        "status": "accepted",
        "terminal": False,
        "instance_id": record.instance_id,
        "operation_id": "operation",
    }
    running = {**accepted, "status": "running"}
    calls = []
    monkeypatch.setattr(cli, "_instance", lambda _path: record)

    async def fake_request(actual_record, method, arguments=None, *, timeout):
        assert actual_record is record
        calls.append(method)
        if method == "submit":
            assert arguments["name"] == "capture.set"
            assert arguments["values"] == {"channel": "self", "enabled": False}
            return accepted
        assert method == "wait"
        assert arguments["operation_id"] == accepted["operation_id"]
        return running

    monkeypatch.setattr(cli, "request", fake_request)
    argv = ["capture", "set", "self", "off"]
    if no_wait:
        argv.append("--no-wait")

    assert cli.main(argv) == (0 if no_wait else 7)

    output = capsys.readouterr()
    result = json.loads(output.out)
    assert output.err == ""
    assert result["output_version"] == 1
    assert result["instance_id"] == record.instance_id
    assert result["operation_id"] == accepted["operation_id"]
    if no_wait:
        assert calls == ["submit"]
        assert result == {"output_version": 1, **accepted}
    else:
        assert calls == ["submit", "wait"]
        assert result["status"] == "unknown_execution"
        assert result["receipt"] == running


@pytest.mark.parametrize("phase", ["submit", "wait"])
@pytest.mark.parametrize(
    "terminal_field",
    [{}, {"terminal": None}, {"terminal": 0}, {"terminal": 1}, {"terminal": "false"}],
    ids=["missing", "null", "zero", "one", "string"],
)
def test_submission_rejects_missing_or_non_boolean_terminal(
    monkeypatch, capsys, phase, terminal_field
):
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))
    calls = []

    async def fake_request(_record, method, arguments=None, *, timeout):
        calls.append(method)
        receipt = {"status": "accepted", "terminal": False, "operation_id": "operation"}
        if method == phase:
            return {"status": "applied", "operation_id": "operation", **terminal_field}
        assert method == "submit"
        return receipt

    monkeypatch.setattr(cli, "request", fake_request)

    assert cli.main(["capture", "set", "self", "off"]) == 3

    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"]["code"] == "malformed_response"
    assert calls == (["submit"] if phase == "submit" else ["submit", "wait"])
