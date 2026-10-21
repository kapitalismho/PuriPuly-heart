from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from puripuly_heart.cli import main as cli


@pytest.fixture
def host_process():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE
    )
    try:
        yield process
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


@pytest.fixture
def stop_host(monkeypatch, tmp_path, host_process):
    record = SimpleNamespace(instance_id="original-instance")
    receipt = {
        "status": "accepted",
        "terminal": False,
        "instance_id": record.instance_id,
        "operation_id": "stop-operation",
    }
    calls = []
    monkeypatch.setattr(cli, "_instance", lambda _path: record)

    async def request(_record, method, arguments=None, *, timeout):
        assert _record is record
        calls.append((method, arguments))
        if method == "query":
            assert arguments == {"name": "app.status", "values": {}}
            return {"host_pid": host_process.pid}
        assert method == "submit"
        assert arguments["name"] == "app.stop"
        assert arguments["values"] == {}
        return receipt

    monkeypatch.setattr(cli, "request", request)
    return SimpleNamespace(
        argv=["--config", str(tmp_path / "settings.json"), "app", "stop"],
        record=record,
        receipt=receipt,
        calls=calls,
        process=host_process,
    )


def test_stop_wait_does_not_treat_endpoint_disappearance_as_host_exit(
    monkeypatch, capsys, stop_host
):
    monkeypatch.setattr("puripuly_heart.core.control_instance.discover", lambda _path: None)

    assert cli.main([*stop_host.argv, "--timeout", "0.08"]) == 7

    output = capsys.readouterr()
    result = json.loads(output.out)
    assert output.err == ""
    assert result["output_version"] == 1
    assert result["status"] == "unknown_execution"
    assert result["instance_id"] == stop_host.record.instance_id
    assert result["operation_id"] == stop_host.receipt["operation_id"]
    assert "cleanup_outcome" not in result
    assert stop_host.process.poll() is None
    assert [method for method, _ in stop_host.calls] == ["query", "submit"]


def test_stop_wait_observes_original_identity_exit_without_claiming_cleanup(
    monkeypatch, capsys, stop_host
):
    discoveries = []

    def discover(_path):
        discoveries.append(stop_host.process.poll())
        if len(discoveries) == 1:
            assert stop_host.process.poll() is None
            stop_host.process.stdin.close()
            stop_host.process.wait(timeout=5)
            return stop_host.record
        return None

    monkeypatch.setattr("puripuly_heart.core.control_instance.discover", discover)

    assert cli.main([*stop_host.argv, "--timeout", "2"]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "output_version": 1,
        "status": "stopped",
        "terminal": True,
        "instance_id": stop_host.record.instance_id,
        "operation_id": stop_host.receipt["operation_id"],
        "cleanup_outcome": "unobserved",
    }
    assert len(discoveries) >= 2
    assert stop_host.process.poll() == 0
    assert [method for method, _ in stop_host.calls] == ["query", "submit"]


def test_stop_no_wait_returns_acceptance_while_original_process_is_alive(
    monkeypatch, capsys, stop_host
):
    monkeypatch.setattr(
        "puripuly_heart.core.control_instance.discover",
        lambda _path: pytest.fail("no-wait must not observe shutdown completion"),
    )

    assert cli.main([*stop_host.argv, "--no-wait"]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {"output_version": 1, **stop_host.receipt}
    assert "cleanup_outcome" not in json.loads(output.out)
    assert stop_host.process.poll() is None
    assert [method for method, _ in stop_host.calls] == ["query", "submit"]
