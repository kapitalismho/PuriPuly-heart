from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
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
async def test_source_composed_cli_retains_local_http_and_audio_websocket_causes(tmp_path) -> None:
    private_body = "SPEECH_DIAGNOSTICS_PROVIDER_BODY_" + uuid4().hex
    private_key = "SPEECH_DIAGNOSTICS_KEY_" + uuid4().hex
    passphrase = uuid4().hex
    requests = {"http": 0, "websocket": 0, "audio": 0}

    class HttpFailure(BaseHTTPRequestHandler):
        def do_POST(self):
            requests["http"] += 1
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps({"error": {"message": private_body}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    async def websocket_failure(socket):
        requests["websocket"] += 1
        await socket.recv()
        async for payload in socket:
            if isinstance(payload, bytes) and payload:
                requests["audio"] += 1
                await socket.send(json.dumps({"error_code": 503, "error_message": private_body}))
                await socket.close()
                return

    http = ThreadingHTTPServer(("127.0.0.1", 0), HttpFailure)
    http_thread = Thread(target=http.serve_forever, daemon=True)
    http_thread.start()
    host = None
    stderr_file = tmp_path / "private-host-stderr.txt"
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
    worktree = Path(__file__).resolve().parents[2]
    environment.update(
        {
            "PYTHONPATH": str(worktree / "src"),
            "LOCALAPPDATA": str(tmp_path / "localappdata"),
            "PURIPULY_HEART_SECRETS_PASSPHRASE": passphrase,
            "NO_PROXY": "127.0.0.1,localhost,::1",
        }
    )
    (tmp_path / "localappdata").mkdir()
    config = tmp_path / "settings.json"
    ready = tmp_path / "ready.json"

    async def cli(*arguments, input_text=None):
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
                "30",
                *arguments,
            ],
            input=input_text,
            cwd=worktree,
            env=environment,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, (arguments, result.returncode, result.stdout)
        return json.loads(result.stdout)

    try:
        async with serve(websocket_failure, "127.0.0.1", 0) as websocket:
            port = websocket.sockets[0].getsockname()[1]
            settings = new_settings_for_first_run("en-US")
            settings = replace(
                settings,
                intent=replace(
                    settings.intent,
                    translation=replace(
                        settings.intent.translation, model="local_llm", connection="ollama"
                    ),
                    local_llm=replace(
                        settings.intent.local_llm,
                        base_url=f"http://127.0.0.1:{http.server_port}/v1",
                    ),
                    stt=replace(
                        settings.intent.stt,
                        provider="soniox",
                        soniox=replace(
                            settings.intent.stt.soniox, endpoint=f"ws://127.0.0.1:{port}"
                        ),
                    ),
                    osc=replace(settings.intent.osc, connection_mode="off"),
                    secrets=SecretsIntent("encrypted_file", "isolated-secrets.json"),
                ),
            )
            SettingsVNextCanonicalPersistenceAdapter().persist(
                config, with_telemetry_enabled(settings, False)
            )
            secrets = EncryptedFileSecretStore(
                tmp_path / "isolated-secrets.json", passphrase=passphrase
            )
            secrets.set("soniox_api_key", private_key)
            secrets.set("local_llm_api_key", private_key)
            with stderr_file.open("w", encoding="utf-8") as stderr:
                host = subprocess.Popen(
                    [
                        sys.executable,
                        str(worktree / "tests/fixtures/speech_diagnostics_host.py"),
                        "--config",
                        str(config),
                        "--ready-file",
                        str(ready),
                    ],
                    cwd=worktree,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                )
                for _ in range(300):
                    if ready.exists() or host.poll() is not None:
                        break
                    await asyncio.sleep(0.1)
                assert ready.exists(), ("host startup unavailable", host.poll())
                assert json.loads(ready.read_text())["state"] == "ready"
                capabilities = await cli("capabilities")
                initial_capture = await cli("query", "capture.channels")
                translation_admission = await cli("translation", "set", "on")
                admission = await cli(
                    "text", "submit", "--stdin", input_text="Synthetic manual translation fixture"
                )
                capture_admission = await cli("capture", "set", "self", "on")
                for _ in range(200):
                    if requests["audio"] >= 3:
                        break
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.5)
                (tmp_path / "admission-evidence.json").write_text(
                    json.dumps(
                        {
                            "translation": translation_admission,
                            "manual": admission,
                            "capture": capture_admission,
                            "fixture_counts": requests,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                history = await cli("query", "diagnostics.history")
                filter_file = tmp_path / "operation-filter.json"
                filter_file.write_text(json.dumps({"operation_id": admission["operation_id"]}))
                filtered = await cli("query", "diagnostics.history", "--file", str(filter_file))
                health = await cli("query", "diagnostics.health")
                final_capture = await cli("query", "capture.channels")
                (tmp_path / "native-evidence.json").write_text(
                    json.dumps(
                        {"history": history, "health": health, "capture": final_capture}, indent=2
                    ),
                    encoding="utf-8",
                )
                await cli("app", "stop")
                await asyncio.to_thread(host.wait, timeout=20)
                assert host.returncode == 0
                assert requests["http"] >= 1
                assert requests["websocket"] >= 3
                assert requests["audio"] >= 3
                encoded = json.dumps({"history": history, "health": health})
                assert private_body not in encoded
                assert private_key not in encoded
                assert "local_llm" in encoded
                assert "soniox" in encoded
                assert "503" in encoded
                records = history["records"]
                assert len({item["diagnostic_id"] for item in records}) == len(records)
                manual = [item for item in records if item["fields"].get("provider") == "local_llm"]
                assert len(manual) == 1
                assert manual[0]["operation_id"] == admission["operation_id"]
                assert manual[0]["request_id"] == admission["request_id"]
                assert manual[0]["turn_id"] in admission["child_turn_ids"]
                assert manual[0]["status_code"] == 503
                assert manual[0]["code"] == "provider.service_unavailable"
                assert [item["diagnostic_id"] for item in filtered["records"]] == [
                    manual[0]["diagnostic_id"]
                ]
                speech = [item for item in records if item["code"] == "stt.soniox_request_failed"]
                assert [item["fields"]["attempts"] for item in speech] == [1, 2, 3]
                assert all(item["category"] == "service_unavailable" for item in speech)
                assert all(item["channel"] == "self" for item in speech)
                admitted_generation = capture_admission["channel"]["capture"]["generation"]
                assert all(item["generation"] == admitted_generation for item in speech)
                assert len({item["fields"]["provider_epoch_id"] for item in speech}) == 3
                assert len({item["fields"]["provider_turn_id"] for item in speech}) == 3
                assert all(item["cause_id"] == speech[0]["diagnostic_id"] for item in speech[1:])
                exhausted = next(
                    item for item in records if item["code"] == "stt.provider_recovery_exhausted"
                )
                assert exhausted["cause_id"] == speech[0]["diagnostic_id"]
                assert exhausted["generation"] == admitted_generation
                assert exhausted["turn_id"] == speech[-1]["turn_id"]
                assert (
                    exhausted["fields"]["provider_epoch_id"]
                    == speech[-1]["fields"]["provider_epoch_id"]
                )
                assert final_capture["channels"]["self"]["capture"]["state"] == "faulted"
                assert final_capture["channels"]["self"]["effective_active"] is False
                assert final_capture["channels"]["peer"] == initial_capture["channels"]["peer"]
                assert health["history"]["rejected_records"] == 0
                assert health["history"]["rejected_fields"] == 0
                log_files = list(tmp_path.rglob("*.log"))
                assert log_files
                file_text = "\\n".join(path.read_text(encoding="utf-8") for path in log_files)
                assert private_body not in file_text
                assert private_key not in file_text
                assert manual[0]["diagnostic_id"] in file_text
                assert speech[0]["diagnostic_id"] in file_text
                assert exhausted["diagnostic_id"] in file_text
                evidence = {
                    "capabilities": capabilities,
                    "initial_capture": initial_capture,
                    "admission": admission,
                    "capture_admission": capture_admission,
                    "history": history,
                    "health": health,
                    "final_capture": final_capture,
                    "filtered_history": filtered,
                    "fixture_counts": requests,
                    "host_exit": host.returncode,
                }
                (tmp_path / "speech-composed-evidence.json").write_text(
                    json.dumps(evidence, indent=2), encoding="utf-8"
                )
    finally:
        if host is not None and host.poll() is None:
            try:
                await cli("app", "stop")
                await asyncio.to_thread(host.wait, timeout=20)
            except AssertionError, subprocess.TimeoutExpired:
                host.terminate()
                await asyncio.to_thread(host.wait, timeout=10)
        http.shutdown()
        http.server_close()
        http_thread.join(timeout=5)
