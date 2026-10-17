from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from evidence_support import HERE, ROOT, EvidenceState, digest, install_network_guard, save


def child(arguments):
    install_network_guard()
    import puripuly_heart
    from puripuly_heart.main import main

    if arguments[0] == "run-headless":
        config = Path(arguments[arguments.index("--config") + 1])
        save(
            config.parent / "host-identity.json",
            {
                "interpreter": sys.executable,
                "module": puripuly_heart.__file__,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("puripuly-heart", "flet", "sherpa-onnx", "onnxruntime")
                },
            },
        )
    return main(arguments)


def execute(args):
    from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, with_telemetry_enabled

    state = EvidenceState(args.report)
    install_network_guard()
    report = {
        "schema": "self-stream-233/isolated-actual-app-cli/v1",
        "status": "running",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runner_sha256": digest(Path(__file__)),
        "support_sha256": digest(HERE / "evidence_support.py"),
        "interpreter": sys.executable,
        "operations": [],
        "limits": [
            "Actual supported CLI and application composition; no GUI presentation claim",
            "Translation is deliberately disabled and no external extension is installed",
            "Manual text is synthetic; this operation is not speech/ASR/progressive-turn evidence",
            "No capture, model installation, credential verification, cloud call, or physical VRChat session",
        ],
    }
    report["source_files"] = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in sorted((ROOT / "src" / "puripuly_heart").rglob("*.py"))
    }
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.2)
    port = receiver.getsockname()[1]
    if port in (9000, 9001):
        receiver.close()
        raise RuntimeError("Owned receiver must not use a default VRChat OSC port")
    settings = with_telemetry_enabled(AppSettingsVNext(), False)
    settings = dataclasses.replace(
        settings,
        intent=dataclasses.replace(
            settings.intent,
            stt=dataclasses.replace(
                settings.intent.stt, provider="local_parakeet_v3", custom_vocabulary_enabled=False
            ),
            peer_stt=dataclasses.replace(settings.intent.peer_stt, provider="local_parakeet_v3"),
            translation=dataclasses.replace(
                settings.intent.translation,
                model="custom_http",
                http_extension_id="self-stream-233-unavailable-by-design",
            ),
            languages=dataclasses.replace(
                settings.intent.languages, source_language="en", target_language="ja"
            ),
            secrets=dataclasses.replace(
                settings.intent.secrets,
                backend="encrypted_file",
                encrypted_file_path="unused-isolated-secrets.json",
            ),
            osc=dataclasses.replace(
                settings.intent.osc,
                connection_mode="off",
                host="127.0.0.1",
                port=port,
                send_port=port,
                activation_notice_enabled=False,
            ),
        ),
    )
    config = state.directory / "settings.json"
    save(config, settings)
    report["profile"] = {
        "sha256": digest(config),
        "settings": settings,
        "isolation": "LOCALAPPDATA/APPDATA/USERPROFILE/HOME/TEMP/TMP are owned state; telemetry disabled, encrypted-file secrets, empty extensions, OSC connection off, sender only owned loopback port",
        "receiver_port": port,
    }
    host = None
    host_log = None

    def command(arguments, *, input_text=None, timeout=60):
        started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--child",
                json.dumps(["cli", "--config", str(config), "--timeout", "30", *arguments]),
            ],
            cwd=ROOT,
            env=state.environment,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = None
        row = {
            "arguments": arguments,
            "exit_code": result.returncode,
            "elapsed_s": time.monotonic() - started,
            "payload": payload,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        report["operations"].append(row)
        return row

    try:
        ready = state.directory / "ready.json"
        host_log = (state.directory / "host-output.txt").open("w", encoding="utf-8")
        host = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--child",
                json.dumps(["run-headless", "--config", str(config), "--ready-file", str(ready)]),
            ],
            cwd=ROOT,
            env=state.environment,
            stdout=host_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 120
        while not ready.exists():
            if host.poll() is not None:
                raise RuntimeError("Isolated host exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("Isolated host readiness timed out")
            time.sleep(0.1)
        report["readiness"] = json.loads(ready.read_text(encoding="utf-8"))
        if report["readiness"].get("state") != "ready":
            raise RuntimeError("Isolated host reported startup failure")
        report["host_identity"] = json.loads(
            (state.directory / "host-identity.json").read_text(encoding="utf-8")
        )
        for arguments in (
            ["capabilities"],
            ["app", "status"],
            ["settings", "current"],
            ["models", "status"],
            ["osc", "status"],
            ["translation", "set", "off"],
            ["capture", "set", "self", "off"],
            ["app", "status"],
        ):
            command(arguments)
        report["source_only_text_admission"] = command(
            ["text", "submit", "--stdin"],
            input_text="SELF STREAM 233 isolated synthetic CLI probe.",
        )
        from pythonosc.osc_packet import OscPacket

        packets = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                packet, source = receiver.recvfrom(65535)
            except TimeoutError:
                continue
            packets.append(
                {
                    "source": source,
                    "messages": [
                        {"address": item.message.address, "params": item.message.params}
                        for item in OscPacket(packet).messages
                    ],
                }
            )
        report["loopback_packets"] = packets
        report["post_operation_status"] = command(["app", "status"])
        report["status"] = (
            "passed" if all(row["exit_code"] == 0 for row in report["operations"]) else "failed"
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if host is not None and host.poll() is None:
            try:
                report["stop"] = command(["app", "stop"], timeout=90)
                host.wait(timeout=30)
            except BaseException as exc:
                report["status"] = "failed"
                report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
        if host is not None:
            report["host_exit_code"] = host.poll()
        if host_log is not None:
            host_log.close()
            report["host_output"] = (state.directory / "host-output.txt").read_text(
                encoding="utf-8"
            )
        receiver.close()
        report["cleanup"] = {
            "owned_udp_receiver": "closed",
            "isolated_state": "retained for exact local bytes",
            "host_process": (
                "exited" if host is not None and host.poll() is not None else "unobserved"
            ),
            "external_user_logs": "not read or modified",
        }
        report["source_files_at_end"] = {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in sorted((ROOT / "src" / "puripuly_heart").rglob("*.py"))
        }
        report["candidate_stable_during_run"] = (
            report["source_files"] == report["source_files_at_end"]
        )
        if not report["candidate_stable_during_run"]:
            report["status"] = "failed"
        state.export(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": args.report,
                "operations": len(report["operations"]),
                "host_exit_code": report.get("host_exit_code"),
                "error": state.sanitize(report.get("error")),
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report")
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        return child(json.loads(args.child))
    if not args.report:
        parser.error("--report is required")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
