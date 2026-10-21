from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from puripuly_heart.cli import main as cli


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
    tmp_path, capsys, monkeypatch, channel, state
):
    monkeypatch.setattr(cli, "_instance", lambda _path: pytest.fail("must reject before discovery"))
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


@pytest.mark.parametrize("value", ["not-a-number", "nan", "PRIVATE_CALIBRATION_SENTINEL"])
def test_calibration_rejects_invalid_numeric_input_without_discovery_or_disclosure(
    tmp_path, capsys, monkeypatch, value
):
    monkeypatch.setattr(cli, "_instance", lambda _path: pytest.fail("must reject before discovery"))
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
    assert value not in output.out + output.err


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
