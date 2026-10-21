from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from websockets.asyncio.server import serve

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import SecretsIntent, with_telemetry_enabled
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["startup", "handoff"])
@pytest.mark.parametrize("stage", ["provider_build", "provider_warmup"])
async def test_peer_cli_native_failure_retains_new_operation_and_receipt_cause(
    tmp_path, admission, stage
):
    worktree = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    for key in tuple(environment):
        upper = key.upper()
        if (
            "API_KEY" in upper
            or "DASHSCOPE" in upper
            or "ALIBABA" in upper
            or upper.startswith("PURIPULY_HEART_NATIVE_")
            or upper.endswith("_PROXY")
        ):
            environment.pop(key, None)
    passphrase = uuid4().hex
    environment.update(
        {
            "PYTHONPATH": str(worktree / "src"),
            "LOCALAPPDATA": str(tmp_path / "localappdata"),
            "PURIPULY_HEART_SECRETS_PASSPHRASE": passphrase,
            "NO_PROXY": "127.0.0.1,localhost,::1",
        }
    )
    (tmp_path / "localappdata").mkdir()
    config, ready, gate = (
        tmp_path / name for name in ("settings.json", "ready.json", "failure-gate")
    )
    host = None

    async def cli(*arguments, allow_failed=False):
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "puripuly_heart.main",
                "cli",
                "--config",
                str(config),
                "--timeout",
                "20",
                *arguments,
            ],
            cwd=worktree,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or allow_failed, (arguments, result.returncode, result.stdout)
        return json.loads(result.stdout)

    async def source_socket(socket):
        await socket.recv()
        async for _payload in socket:
            await socket.send(json.dumps({"tokens": [], "finished": False}))

    try:
        async with serve(source_socket, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            settings = new_settings_for_first_run("en-US")
            settings = replace(
                settings,
                intent=replace(
                    settings.intent,
                    stt=replace(
                        settings.intent.stt,
                        provider="soniox",
                        soniox=replace(
                            settings.intent.stt.soniox, endpoint=f"ws://127.0.0.1:{port}"
                        ),
                    ),
                    peer_stt=replace(settings.intent.peer_stt, provider="soniox"),
                    osc=replace(settings.intent.osc, connection_mode="off"),
                    secrets=SecretsIntent("encrypted_file", "isolated-secrets.json"),
                ),
                state=replace(
                    settings.state,
                    peer_translation=replace(
                        settings.state.peer_translation,
                        eula_accepted=True,
                    ),
                ),
            )
            SettingsVNextCanonicalPersistenceAdapter().persist(
                config, with_telemetry_enabled(settings, False)
            )
            secrets = EncryptedFileSecretStore(
                tmp_path / "isolated-secrets.json", passphrase=passphrase
            )
            secrets.set("soniox_api_key", "PRIVATE_FIXTURE_KEY")
            secrets.set("deepgram_api_key", "PRIVATE_FIXTURE_KEY")
            with (tmp_path / "private-host-stderr.txt").open("w", encoding="utf-8") as stderr:
                host = subprocess.Popen(
                    [
                        sys.executable,
                        str(worktree / "tests/fixtures/peer_diagnostics_host.py"),
                        "--config",
                        str(config),
                        "--ready-file",
                        str(ready),
                        "--failure-stage",
                        stage,
                        "--failure-gate",
                        str(gate),
                    ],
                    cwd=worktree,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                )
                for _ in range(200):
                    if ready.exists() or host.poll() is not None:
                        break
                    await asyncio.sleep(0.1)
                assert ready.exists(), (host.poll(), "private host stderr retained")
                initial_receipt = None
                before = await cli("query", "capture.channels")
                if admission == "handoff":
                    initial_receipt = await cli(
                        "command",
                        "capture.set",
                        "--arguments",
                        json.dumps({"channel": "peer", "enabled": True}),
                    )
                    before = await cli("query", "capture.channels")
                    assert before["channels"]["peer"]["effective_active"]
                gate.touch()
                if admission == "startup":
                    receipt = await cli(
                        "command",
                        "capture.set",
                        "--arguments",
                        json.dumps({"channel": "peer", "enabled": True}),
                        allow_failed=True,
                    )
                else:
                    receipt = await cli(
                        "command",
                        "provider.apply",
                        "--arguments",
                        json.dumps({"channel": "peer", "provider": "deepgram"}),
                        allow_failed=True,
                    )
                filter_file = tmp_path / "operation-filter.json"
                filter_file.write_text(json.dumps({"operation_id": receipt["operation_id"]}))
                history = await cli("query", "diagnostics.history", "--file", str(filter_file))
                capture = await cli("query", "capture.channels")
                health = await cli("query", "diagnostics.health")
                await cli("app", "stop")
                await asyncio.to_thread(host.wait, timeout=20)
                evidence = {
                    "admission": admission,
                    "failure_stage": stage,
                    "initial_receipt": initial_receipt,
                    "receipt": receipt,
                    "filtered_history": history,
                    "before": before,
                    "capture": capture,
                    "health": health,
                    "host_exit": host.returncode,
                    "fixture_limits": {
                        "terms": "Preaccepted Peer terms seeded only in isolated fixture settings; no real-user terms acceptance command or mutation.",
                        "overlay": "Peer admission prerequisite state is fixture-connected; no actual overlay/OpenVR/HMD readiness claim.",
                        "audio": "Generated paced source and deterministic VAD only; no ambient desktop/microphone capture.",
                        "warmup": "Armed warmup fixture explicitly requests native warmup; no hardware/model readiness claim.",
                    },
                }
                (tmp_path / "peer-admission-evidence.json").write_text(
                    json.dumps(evidence, indent=2)
                )
                native = [record for record in history["records"] if record["stage"] == stage]
                assert len(native) == 1, history["records"]
                failure = native[0]
                assert failure["operation_id"] == receipt["operation_id"]
                assert failure["request_id"] == receipt["request_id"]
                assert failure["channel"] == "peer"
                assert failure["category"] == (
                    "network" if stage == "provider_build" else "timeout"
                )
                returned = (
                    receipt.get("cause_diagnostics")
                    or receipt.get("transaction", {}).get("diagnostics")
                    or receipt.get("diagnostics")
                )
                assert returned["diagnostic_id"] == failure["diagnostic_id"], receipt
                if "cause_diagnostics" in receipt:
                    assert receipt["diagnostics"]["cause_id"] == failure["diagnostic_id"]
                current = capture["channels"]["peer"]["capture"]
                assert current["failure_diagnostics"]["diagnostic_id"] == failure["diagnostic_id"]
                if admission == "handoff":
                    assert (
                        current["generation"] == before["channels"]["peer"]["capture"]["generation"]
                    )
                    assert current["effective_active"]
                    assert failure["operation_id"] != initial_receipt["operation_id"]
                else:
                    assert not current["effective_active"]
                self_capture = capture["channels"]["self"]["capture"]
                assert not self_capture["effective_active"]
                assert not self_capture["has_source"]
                assert "PRIVATE_" not in json.dumps(evidence)
                assert health["history"]["rejected_records"] == 0
                assert host.returncode == 0
    finally:
        if host is not None and host.poll() is None:
            try:
                await cli("app", "stop")
                await asyncio.to_thread(host.wait, timeout=20)
            except AssertionError, subprocess.TimeoutExpired:
                host.terminate()
                await asyncio.to_thread(host.wait, timeout=10)
