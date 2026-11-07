from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from puripuly_heart.cli import main as cli


@pytest.mark.parametrize("native", [False, True], ids=["source", "native"])
def test_background_start_uses_current_host_and_matching_readiness(
    tmp_path, monkeypatch, capsys, native
):
    import sys
    from pathlib import Path

    from puripuly_heart.core import control_instance

    root = tmp_path / "설치 with spaces"
    root.mkdir()
    if native:
        monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(root / "app"))
        monkeypatch.setenv("PURIPULY_HEART_NATIVE_RUNTIME_ROOT", str(root))
        monkeypatch.setenv("PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE", str(root / "python.exe"))
        monkeypatch.setenv("PURIPULY_HEART_NATIVE_HOST_EXECUTABLE", str(root / "PuriPulyHeart.exe"))
    else:
        monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", raising=False)
    config = tmp_path / "설정 with spaces.json"
    calls = []
    record = SimpleNamespace(instance_id="matching-instance")
    discovered = iter((None, record))
    monkeypatch.setattr(control_instance, "discover", lambda _path: next(discovered))

    def launch(command, **options):
        calls.append((command, options))
        ready = Path(command[command.index("--ready-file") + 1])
        ready.write_text(
            json.dumps({"state": "ready", "instance_id": record.instance_id}), encoding="utf-8"
        )
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    assert cli.main(["--config", str(config), "app", "start", "--background"]) == 0
    command, options = calls[0]
    assert command[:3] == [
        str(root / "python.exe") if native else str(Path(sys.executable).resolve()),
        "-m",
        "product_bootstrap" if native else "puripuly_heart.main",
    ]
    assert command[3:6] == ["run-headless", "--config", str(config)]
    assert not Path(command[-1]).exists()
    if native:
        assert options["env"]["PURIPULY_HEART_NATIVE_RUNTIME_ROOT"] == str(root)
        assert options["env"]["PYTHONNOUSERSITE"] == "1"
    else:
        assert "env" not in options
    assert json.loads(capsys.readouterr().out)["instance_id"] == record.instance_id


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


@pytest.mark.parametrize("status", ["accepted", "running", "unknown"])
def test_operation_wait_rejects_invalid_terminal_status(monkeypatch, capsys, status):
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))

    async def fake_request(_record, method, arguments=None, *, timeout):
        return {"status": status, "terminal": True, "operation_id": "operation"}

    monkeypatch.setattr(cli, "request", fake_request)

    assert cli.main(["operation", "wait", "operation"]) == 3
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "output_version": 1,
        "status": status,
        "terminal": True,
        "operation_id": "operation",
    }
    assert json.loads(output.err)["error"]["code"] == "malformed_response"


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


@pytest.mark.parametrize("provider", ["qq", "discord", "openrouter", "chatgpt"])
def test_auth_rejects_removed_referral_input_without_dispatch(monkeypatch, capsys, provider):
    import io

    fields = {"referral_id": "private-retired-code"}
    if provider == "qq":
        fields.update(qq_identity="synthetic-qq", credential="a" * 64)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(json.dumps(fields)))
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))

    async def unexpected(*_args, **_kwargs):
        pytest.fail("Removed auth input must never reach application dispatch")

    monkeypatch.setattr(cli, "request", unexpected)
    assert cli.main(["auth", "login", provider, "--stdin"]) == 3
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"]["code"] == "invalid_input"
    assert "private-retired-code" not in output.err


@pytest.mark.parametrize("provider", ["qq", "discord"])
def test_auth_rejects_removed_referral_cli_flag(capsys, provider):
    with pytest.raises(SystemExit) as result:
        cli.build_parser().parse_args(["auth", "login", provider, "--referral-id", "retired"])
    assert result.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["qq", "discord"])
def test_auth_protected_input_keeps_supported_login_fields(monkeypatch, provider):
    import io

    fields = {"qq_identity": "synthetic-qq", "credential": "a" * 64} if provider == "qq" else {}
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(json.dumps(fields)))
    assert cli._auth_input(SimpleNamespace(stdin=True, provider=provider)) == fields


@pytest.mark.parametrize("no_wait", (False, True))
def test_audio_diagnose_dispatches_empty_owned_command_and_preserves_terminal_evidence(
    monkeypatch, capsys, no_wait
):
    record = SimpleNamespace(instance_id="session")
    monkeypatch.setattr(cli, "_instance", lambda _path: record)
    accepted = {
        "instance_id": "session",
        "operation_id": "diagnosis-operation",
        "status": "accepted",
        "terminal": False,
    }
    terminal = {
        **accepted,
        "status": "applied",
        "terminal": True,
        "recognition": "not_exercised",
        "settings_changed": False,
        "diagnosis": {
            "scope": "local_capture_vad",
            "state": "completed",
            "cleanup_complete": True,
            "restoration": {"self": "restored", "peer": "not_needed"},
            "candidates": [{"id": "input-7", "opened": True, "frames_received": True}],
        },
    }
    calls = []

    async def request(actual_record, method, arguments=None, *, timeout):
        assert actual_record is record
        calls.append(method)
        if method == "submit":
            assert arguments["name"] == "audio.diagnose"
            assert arguments["values"] == {}
            return accepted
        assert method == "wait"
        assert arguments["operation_id"] == "diagnosis-operation"
        return terminal

    monkeypatch.setattr(cli, "request", request)
    argv = ["audio", "diagnose"] + (["--no-wait"] if no_wait else [])
    assert cli.main(argv) == 0
    assert calls == (["submit"] if no_wait else ["submit", "wait"])
    assert json.loads(capsys.readouterr().out) == {
        "output_version": 1,
        **(accepted if no_wait else terminal),
    }


@pytest.mark.parametrize("no_wait", (False, True))
def test_osc_discovery_retry_dispatches_empty_command_without_claiming_remote_delivery(
    monkeypatch, capsys, no_wait
):
    record = SimpleNamespace(instance_id="session")
    monkeypatch.setattr(cli, "_instance", lambda _path: record)
    accepted = {
        "instance_id": "session",
        "operation_id": "osc-retry-operation",
        "status": "accepted",
        "terminal": False,
    }
    terminal = {
        **accepted,
        "status": "applied",
        "terminal": True,
        "retry_started": True,
        "effect": "discovery_target_applied",
        "output": {
            "destination_source": "discovered",
            "discovered_remote_present": True,
            "remote_delivery": "unacknowledged",
        },
    }
    calls = []

    async def request(actual_record, method, arguments=None, *, timeout):
        assert actual_record is record
        calls.append(method)
        if method == "submit":
            assert arguments["name"] == "osc.discovery.retry"
            assert arguments["values"] == {}
            return accepted
        assert method == "wait"
        assert arguments["operation_id"] == "osc-retry-operation"
        return terminal

    monkeypatch.setattr(cli, "request", request)
    argv = ["osc", "discovery", "retry"] + (["--no-wait"] if no_wait else [])
    assert cli.main(argv) == 0
    assert calls == (["submit"] if no_wait else ["submit", "wait"])
    assert json.loads(capsys.readouterr().out) == {
        "output_version": 1,
        **(accepted if no_wait else terminal),
    }


def test_osc_status_remains_query_only(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))

    async def request(_record, method, arguments=None, *, timeout):
        assert method == "query"
        assert arguments == {"name": "osc.status", "values": {}}
        return {"output": {"remote_delivery": "unacknowledged"}}

    monkeypatch.setattr(cli, "request", request)
    assert cli.main(["osc", "status"]) == 0
    assert json.loads(capsys.readouterr().out)["output"]["remote_delivery"] == "unacknowledged"


@pytest.mark.parametrize(
    "extra",
    (["--host", "127.0.0.1"], ["--send-port", "19000"], ["--text", "synthetic"]),
)
def test_osc_discovery_retry_rejects_caller_owned_destinations_and_text(extra):
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(["osc", "discovery", "retry", *extra])
    assert raised.value.code == 2


def test_audio_diagnose_rejects_unowned_route_and_duration_arguments():
    for extra in (["--device", "arbitrary"], ["--duration", "999"]):
        with pytest.raises(SystemExit) as raised:
            cli.build_parser().parse_args(["audio", "diagnose", *extra])
        assert raised.value.code == 2


def test_diagnosis_operation_cancel_preserves_cleanup_debt_receipt(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_instance", lambda _path: SimpleNamespace(instance_id="session"))
    receipt = {
        "operation_id": "diagnosis",
        "status": "cancelled",
        "terminal": True,
        "diagnosis": {
            "state": "cancelled",
            "cleanup_complete": False,
            "failure_code": "resource_release_incomplete",
            "restoration": {"self": "cleanup_pending", "peer": "not_needed"},
        },
    }

    async def request(_record, method, arguments=None, *, timeout):
        assert method == "cancel"
        assert arguments == {"operation_id": "diagnosis"}
        return receipt

    monkeypatch.setattr(cli, "request", request)
    assert cli.main(["operation", "cancel", "diagnosis"]) == 6
    assert json.loads(capsys.readouterr().out) == {"output_version": 1, **receipt}
