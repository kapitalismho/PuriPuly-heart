"""Isolated output-failure evidence using real CLI, loopback UDP/HTTP and host processes.

Injected failures are software fixtures, not OpenVR/HMD/GPU execution evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import psutil

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import SecretsIntent, with_telemetry_enabled

_HOST_PROGRAM = r"""
import sys
import json, psutil
from pathlib import Path
from puripuly_heart.composition.headless_application import _HeadlessDashboard
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.overlay.process_runners import DefaultOverlayProcessRunner
from puripuly_heart.main import main
original_dashboard = _HeadlessDashboard.publish_translation
original_sender = VrchatOscUdpSender.send_message
original_output_init = OutputRuntime.__post_init__
original_prepare = DefaultOverlayProcessRunner.prepare
original_spawn = DefaultOverlayProcessRunner.spawn
ui_calls = 0
state_calls = 0
def dashboard(self, *args, **kwargs):
    global ui_calls
    ui_calls += 1
    if ui_calls == 1:
        raise OSError(5, 'SYNTHETIC_FORBIDDEN_OUTPUT_CREDENTIAL_BODY_PERSONAL_PATH')
    return original_dashboard(self, *args, **kwargs)
def sender(self, address, *args, **kwargs):
    global state_calls
    if str(address).startswith('/avatar/parameters/PuriPuly'):
        state_calls += 1
        if state_calls == 3:
            raise OSError(5, 'SYNTHETIC_FORBIDDEN_UDP_CREDENTIAL_BODY')
    return original_sender(self, address, *args, **kwargs)
def observer(_decision):
    raise ValueError('SYNTHETIC_FORBIDDEN_OBSERVER_CREDENTIAL_BODY')
def output_init(self):
    original_output_init(self)
    self.routing_observer = observer
def prepare(self, manifest):
    self.executable_path = Path(sys.argv[3])
    return original_prepare(self, manifest)
async def spawn(self, executable_path, manifest_path):
    child = await original_spawn(self, executable_path, manifest_path)
    identity = {'pid':child.pid, 'create_time':psutil.Process(child.pid).create_time()}
    with Path(sys.argv[4]).open('a', encoding='utf-8') as evidence:
        evidence.write(json.dumps(identity) + '\n')
    return child
_HeadlessDashboard.publish_translation = dashboard
VrchatOscUdpSender.send_message = sender
OutputRuntime.__post_init__ = output_init
DefaultOverlayProcessRunner.prepare = prepare
DefaultOverlayProcessRunner.spawn = spawn
raise SystemExit(main(['run-headless', '--config', sys.argv[1], '--ready-file', sys.argv[2]]))
"""
_NATIVE_PROGRAM = r"""
import asyncio, json, sys, websockets
m = json.load(open(sys.argv[2], encoding='utf-8'))
async def run():
    ws = await websockets.connect(m['bridge_url'], ping_interval=None, proxy=None)
    await ws.send(json.dumps({'type':'auth', 'session_token':m['session_token']}))
    await ws.recv()
    print('SYNTHETIC_FORBIDDEN_CHILD_STDERR_CREDENTIAL_BODY_PATH', file=sys.stderr, flush=True)
    fault = {
        'type':'startup_error', 'overlay_instance_id':m['overlay_instance_id'],
        'runtime_generation':1, 'failure_reason':'render_failed',
        'primary_failure_reason':'render_failed', 'cleanup_failure_reason':'openvr_failed',
        'exception_class':'OSError', 'errno':5, 'diagnostic_dropped':7,
        'reliable_delivery_failures':1, 'evidence_state':'incomplete',
        'detail':'SYNTHETIC_FORBIDDEN_CHILD_CREDENTIAL_BODY_PERSONAL_PATH',
    }
    await ws.send(json.dumps(fault))
    print(json.dumps(fault), flush=True)
    print(json.dumps({**fault, 'runtime_generation':2,
                      'primary_failure_reason':'gpu_query_failed',
                      'failure_reason':'gpu_query_failed'}), flush=True)
    await asyncio.sleep(0.1)
    await ws.close()
asyncio.run(run())
raise SystemExit(7)
"""


def run(report: Path) -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[2]
    evidence: dict[str, object] = {
        "schema": "puripuly-heart/output-diagnostics-232/v1",
        "python_version": sys.version.split()[0],
        "commands": [],
        "cases": {},
        "isolation": {
            "temporary_settings_secrets_discovery": True,
            "telemetry": False,
            "osc": "explicit_loopback_fixture_only",
            "ambient_capture": False,
            "terms_acceptance": False,
            "external_provider_accounts": False,
        },
        "limits": {
            "gpu_openvr_hmd": "not_exercised_by_this_software_fixture",
            "gui": "separate_owned_window_evidence",
        },
    }
    commands = evidence["commands"]
    cases = evidence["cases"]
    request_count = 0
    udp_count = 0
    stop_udp = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            nonlocal request_count
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request_count += 1
            payload = json.dumps(
                {"choices": [{"message": {"content": "synthetic fixture translation"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    http_fixture = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http_thread = threading.Thread(target=http_fixture.serve_forever, daemon=True)
    http_thread.start()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(0.1)

    def consume_udp() -> None:
        nonlocal udp_count
        while not stop_udp.is_set():
            try:
                udp.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            udp_count += 1

    udp_thread = threading.Thread(target=consume_udp, daemon=True)
    udp_thread.start()
    host: subprocess.Popen | None = None
    follower: subprocess.Popen | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="puripuly-output-232-final-") as temporary:
            root = Path(temporary)
            localappdata = root / "localappdata"
            localappdata.mkdir()
            config, ready = root / "settings.json", root / "ready.json"
            native_fixture = root / "native_fault_fixture.py"
            native_fixture.write_text(_NATIVE_PROGRAM, encoding="utf-8")
            native_processes = root / "native-processes.jsonl"
            environment = dict(os.environ)
            for name in tuple(environment):
                upper = name.upper()
                if any(
                    part in upper
                    for part in ("API_KEY", "AUTH_TOKEN", "ACCESS_TOKEN", "CREDENTIAL")
                ) or upper in {
                    "PURIPULY_HEART_NATIVE_RESOURCE_ROOT",
                    "PURIPULY_HEART_NATIVE_RUNTIME_ROOT",
                    "PURIPULY_HEART_NATIVE_HOST_EXECUTABLE",
                    "PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE",
                }:
                    environment.pop(name)
            environment.update(
                PYTHONPATH=str(source_root),
                LOCALAPPDATA=str(localappdata),
                PURIPULY_HEART_SECRETS_PASSPHRASE=secrets.token_urlsafe(48),
            )
            receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            receive.bind(("127.0.0.1", 0))
            receive_port = receive.getsockname()[1]
            receive.close()
            settings = new_settings_for_first_run("en-US")
            settings = replace(
                settings,
                intent=replace(
                    settings.intent,
                    secrets=SecretsIntent("encrypted_file", "isolated-secrets.json"),
                    osc=replace(
                        settings.intent.osc,
                        connection_mode="off",
                        host="127.0.0.1",
                        send_port=udp.getsockname()[1],
                        port=udp.getsockname()[1],
                        receive_port=receive_port,
                    ),
                    translation=replace(
                        settings.intent.translation, model="local_llm", connection="ollama"
                    ),
                    local_llm=replace(
                        settings.intent.local_llm,
                        backend="ollama",
                        base_url=f"http://127.0.0.1:{http_fixture.server_port}/v1",
                        model="synthetic-fixture",
                        extra_body={},
                    ),
                ),
            )
            SettingsVNextCanonicalPersistenceAdapter().persist(
                config, with_telemetry_enabled(settings, False)
            )

            def command(*arguments: str, text: str | None = None) -> dict[str, object]:
                argv = [
                    sys.executable,
                    "-m",
                    "puripuly_heart.main",
                    "cli",
                    "--config",
                    str(config),
                    "--timeout",
                    "30",
                    *arguments,
                ]
                commands.append(
                    [
                        "<python>",
                        "-m",
                        "puripuly_heart.main",
                        "cli",
                        "--config",
                        "<isolated-config>",
                        "--timeout",
                        "30",
                        *[value.replace(str(root), "<isolated-root>") for value in arguments],
                    ]
                )
                result = subprocess.run(
                    argv,
                    env=environment,
                    input=text,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=40,
                )
                return {
                    "exit_code": result.returncode,
                    "json": [
                        json.loads(line) for line in result.stdout.splitlines() if line.strip()
                    ],
                    "stderr_bytes": len(result.stderr.encode()),
                }

            commands.append(
                [
                    "<python>",
                    "-c",
                    "<owned-output-software-fixture>",
                    "<isolated-config>",
                    "<isolated-ready-file>",
                    "<isolated-native-fixture>",
                    "<isolated-native-process-identities>",
                ]
            )
            host = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _HOST_PROGRAM,
                    str(config),
                    str(ready),
                    str(native_fixture),
                    str(native_processes),
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 45
            while not ready.exists():
                if host.poll() is not None:
                    raise RuntimeError(f"owned_host_exited_{host.returncode}")
                if time.monotonic() >= deadline:
                    raise RuntimeError("owned_host_readiness_timeout")
                time.sleep(0.1)
            instance = json.loads(ready.read_text(encoding="utf-8"))["instance_id"]
            cases["capabilities"] = command("capabilities")
            cases["capture_before"] = command("query", "capture.channels")
            cases["translation_enable"] = command("translation", "set", "on")
            assert cases["translation_enable"]["exit_code"] == 0
            cases["first_admission"] = command(
                "text", "submit", "--stdin", text="synthetic fixture first source"
            )
            first = cases["first_admission"]["json"][-1]
            operation = first["operation_id"]
            time.sleep(0.3)
            cases["first_failure_retrospective"] = command(
                "query",
                "diagnostics.history",
                "--arguments",
                json.dumps({"operation_id": operation, "instance_id": instance, "limit": 20}),
            )
            cases["admission_operation_after_failure"] = command("operation", "status", operation)
            edit = root / "osc.json"
            edit.write_text(
                json.dumps(
                    {
                        "changes": {
                            "osc.connection": {
                                "mode": "manual",
                                "send_port": udp.getsockname()[1],
                                "receive_port": receive_port,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            cases["osc_enable"] = command("settings", "apply", "--file", str(edit))
            cases["second_admission"] = command(
                "text", "submit", "--stdin", text="synthetic fixture second source"
            )
            time.sleep(0.3)
            cases["osc_current"] = command("query", "osc.status")
            cases["native_enable"] = command("overlay", "set", "on")
            cases["native_owner_after_failure"] = command("query", "overlay.status")
            cases["native_failure_retrospective"] = command(
                "query",
                "diagnostics.history",
                "--arguments",
                json.dumps(
                    {
                        "operation_id": cases["native_enable"]["json"][-1]["operation_id"],
                        "instance_id": instance,
                        "limit": 20,
                    }
                ),
            )
            cases["capture_after"] = command("query", "capture.channels")
            cases["history"] = command(
                "query",
                "diagnostics.history",
                "--arguments",
                json.dumps({"instance_id": instance, "limit": 100}),
            )
            cases["diagnostic_health"] = command("query", "diagnostics.health")
            follow_argv = [
                sys.executable,
                "-m",
                "puripuly_heart.main",
                "cli",
                "--config",
                str(config),
                "logs",
                "follow",
                "--level",
                "info",
                "--after",
                "0",
                "--instance-id",
                instance,
            ]
            commands.append(
                [
                    "<python>",
                    "-m",
                    "puripuly_heart.main",
                    "cli",
                    "--config",
                    "<isolated-config>",
                    "logs",
                    "follow",
                    "--level",
                    "info",
                    "--after",
                    "0",
                    "--instance-id",
                    instance,
                ]
            )
            follower = subprocess.Popen(
                follow_argv,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            expected_follow = {
                item["diagnostic_id"] for item in cases["history"]["json"][-1]["records"]
            }
            followed: list[dict[str, object]] = []
            follow_complete = threading.Event()
            follow_stdout = follower.stdout
            assert follow_stdout is not None

            def consume_follow() -> None:
                seen: set[str] = set()
                for line in follow_stdout:
                    item = json.loads(line.decode("utf-8"))
                    if item.get("diagnostic"):
                        followed.append(item)
                        seen.add(item["diagnostic"]["diagnostic_id"])
                        if expected_follow <= seen:
                            follow_complete.set()

            follow_thread = threading.Thread(target=consume_follow, daemon=True)
            follow_thread.start()
            follow_complete.wait(timeout=10)
            follower.terminate()
            follower.wait(timeout=10)
            follow_thread.join(timeout=2)
            assert not follow_thread.is_alive()
            cases["follow_selected_diagnostics"] = followed
            assert follower.stderr is not None
            cases["follow_stderr_bytes"] = len(follower.stderr.read())
            follow_stdout.close()
            follower.stderr.close()
            follower = None
            cases["normal_stop"] = command("app", "stop")
            host.wait(timeout=30)
            cases["host_exit_code"] = host.returncode
            host = None
            cases["owned_native_children"] = []
            for line in native_processes.read_text(encoding="utf-8").splitlines():
                identity = json.loads(line)
                try:
                    remaining = (
                        psutil.Process(identity["pid"]).create_time() == identity["create_time"]
                    )
                except psutil.NoSuchProcess:
                    remaining = False
                cases["owned_native_children"].append({**identity, "exited": not remaining})
            assert cases["owned_native_children"] and all(
                item["exited"] for item in cases["owned_native_children"]
            )
            records = cases["history"]["json"][-1]["records"]
            retrospective = cases["first_failure_retrospective"]["json"][-1]["records"]
            primary = next(
                item for item in retrospective if item["code"] == "destination_publish_failed"
            )
            partial = next(item for item in records if item["code"] == "osc_state_send_failed")
            observer = next(item for item in records if item["code"] == "output_observer_failed")
            native_records = cases["native_failure_retrospective"]["json"][-1]["records"]
            native_primary = next(
                item for item in native_records if item["code"] == "render_failed"
            )
            native_cleanup = next(
                item for item in native_records if item["code"] == "openvr_failed"
            )
            native_loss = next(
                item for item in native_records if item["code"] == "native_diagnostic_loss"
            )
            assert (
                native_primary["operation_id"] == cases["native_enable"]["json"][-1]["operation_id"]
            )
            assert native_primary["request_id"] == cases["native_enable"]["json"][-1]["request_id"]
            assert native_cleanup["cause_id"] == native_primary["diagnostic_id"]
            assert native_loss["cause_id"] == native_primary["diagnostic_id"]
            assert native_primary["fields"]["diagnostic_dropped"] == 7
            assert native_primary["fields"]["reliable_delivery_failures"] == 1
            assert native_primary["fields"]["evidence_state"] == "incomplete"
            assert not any(item["code"] == "gpu_query_failed" for item in records)
            assert first["status"] == "applied"
            assert (
                primary["operation_id"] == operation
                and primary["request_id"] == first["request_id"]
            )
            assert primary["instance_id"] == instance
            assert primary["turn_id"] == first["turn_id"]
            assert primary["fields"]["publication_id"] == first["turn_id"]
            assert (
                partial["fields"]["local_send_completed"] == 2
                and partial["fields"]["local_send_requested"] == 15
            )
            assert partial["operation_id"] == cases["osc_enable"]["json"][-1]["operation_id"]
            assert len([item for item in records if item["code"] == "output_observer_failed"]) == 1
            assert request_count == 2 and udp_count > 0
            assert cases["host_exit_code"] == 0
            log_path = localappdata / "puripuly-heart" / "puripuly_heart.log"
            selected_ids = {
                primary["diagnostic_id"],
                partial["diagnostic_id"],
                observer["diagnostic_id"],
            }
            selected_ids.update(
                {
                    native_primary["diagnostic_id"],
                    native_cleanup["diagnostic_id"],
                    native_loss["diagnostic_id"],
                }
            )
            selected_file = [
                json.loads(line.partition("[Failure] ")[2])
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if "[Failure] {" in line
            ]
            cases["selected_main_file"] = [
                item for item in selected_file if item["diagnostic_id"] in selected_ids
            ]
            assert selected_ids <= {item["diagnostic_id"] for item in cases["selected_main_file"]}
            assert selected_ids <= {
                item["diagnostic"]["diagnostic_id"] for item in cases["follow_selected_diagnostics"]
            }
            evidence["transport_observations"] = {
                "local_http_requests": request_count,
                "local_udp_datagrams_received": udp_count,
                "remote_osc_delivery": "unacknowledged",
            }
            assert "SYNTHETIC_FORBIDDEN" not in json.dumps(evidence)
    except Exception:
        evidence["result"] = "failed"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        if follower is not None:
            follower.terminate()
            follower.wait(timeout=10)
        if host is not None and host.poll() is None:
            host.terminate()
            host.wait(timeout=10)
        stop_udp.set()
        udp.close()
        udp_thread.join(timeout=2)
        http_fixture.shutdown()
        http_fixture.server_close()
        http_thread.join(timeout=2)
    evidence["source_sha256"] = {
        str(path.relative_to(source_root))
        .replace("\\", "/"): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in (
            Path(__file__),
            source_root / "puripuly_heart/core/runtime/output.py",
            source_root / "puripuly_heart/core/orchestrator/translation_output_projection.py",
            source_root / "puripuly_heart/ui/event_dispatch.py",
            source_root / "puripuly_heart/app/services/osc/control_runtime.py",
            source_root / "puripuly_heart/core/overlay/process.py",
            source_root / "puripuly_heart/core/overlay/process_adapter.py",
            source_root / "puripuly_heart/core/diagnostics.py",
        )
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    evidence["result"] = "passed"
    report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.report)
    print(json.dumps({"status": "passed", "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
