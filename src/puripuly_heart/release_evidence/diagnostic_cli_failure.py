"""Windows settings-denial evidence through real, separately attached CLI processes.

Run with the worktree src on PYTHONPATH and --report pointing outside the checkout.
Only owned temporary settings/secrets/data and owned host processes are touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import SecretsIntent, with_telemetry_enabled
from puripuly_heart.release_evidence.cli_control_failures import SettingsReplacementLock


def run(report: Path) -> dict:
    if os.name != "nt":
        raise RuntimeError("Windows host required")
    source_root = Path(__file__).resolve().parents[2]
    evidence = {
        "schema": "puripuly-heart/diagnostic-cli-failure/v1",
        "platform": "Windows",
        "python_version": sys.version.split()[0],
        "cases": {},
        "commands": [],
        "limits": {
            "gpu_openvr_hmd": "not_run",
            "gui": "separate_owner_evidence",
            "provider_accounts": "not_used",
        },
    }
    cases = evidence["cases"]
    with tempfile.TemporaryDirectory(prefix="puripuly-diagnostics-229-") as temp:
        root = Path(temp)
        config = root / "settings.json"
        ready = root / "ready.json"
        environment = dict(os.environ)
        for name in tuple(environment):
            upper = name.upper()
            if upper.endswith("_API_KEY") or upper in {
                "DASHSCOPE_API_KEY",
                "PURIPULY_HEART_NATIVE_RESOURCE_ROOT",
                "PURIPULY_HEART_NATIVE_RUNTIME_ROOT",
                "PURIPULY_HEART_NATIVE_HOST_EXECUTABLE",
                "PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE",
            }:
                environment.pop(name)
        environment.update(
            PYTHONPATH=str(source_root),
            LOCALAPPDATA=str(root / "localappdata"),
            PURIPULY_HEART_SECRETS_PASSPHRASE=secrets.token_urlsafe(48),
        )
        settings = new_settings_for_first_run("en-US")
        settings = replace(
            settings,
            intent=replace(
                settings.intent,
                secrets=SecretsIntent("encrypted_file", "isolated-secrets.json"),
                osc=replace(settings.intent.osc, connection_mode="off"),
            ),
        )
        SettingsVNextCanonicalPersistenceAdapter().persist(
            config, with_telemetry_enabled(settings, False)
        )
        evidence["isolation"] = {
            "temporary_config": True,
            "temporary_localappdata": True,
            "encrypted_secrets": "empty_isolated_file_backend",
            "credential_fallbacks_removed": True,
            "native_overrides_removed": True,
            "telemetry": False,
            "osc": "off",
            "capture_requested": False,
            "terms_accepted": False,
        }

        def command(*arguments: str, timeout: float = 35) -> dict:
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
            evidence["commands"].append(
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
                input=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            lines = result.stdout.decode("utf-8", errors="replace").splitlines()
            parsed = [json.loads(line) for line in lines if line.strip()]
            return {
                "exit_code": result.returncode,
                "json": parsed,
                "stderr_bytes": len(result.stderr),
            }

        def start_host() -> subprocess.Popen:
            ready.unlink(missing_ok=True)
            evidence["commands"].append(
                [
                    "<python>",
                    "-m",
                    "puripuly_heart.main",
                    "run-headless",
                    "--config",
                    "<isolated-config>",
                    "--ready-file",
                    "<isolated-ready-file>",
                ]
            )
            host = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "puripuly_heart.main",
                    "run-headless",
                    "--config",
                    str(config),
                    "--ready-file",
                    str(ready),
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
                    host.terminate()
                    host.wait(timeout=10)
                    raise RuntimeError("owned_host_readiness_timeout")
                time.sleep(0.1)
            return host

        def stop_host(host: subprocess.Popen, label: str) -> None:
            if host.poll() is None:
                cases[label] = command("app", "stop", timeout=40)
                try:
                    host.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    host.terminate()
                    host.wait(timeout=10)
                    cases[label]["forced_cleanup"] = True
            cases.setdefault(label, {})["host_exit_code"] = host.returncode

        host = None
        try:
            host = start_host()
            instance = json.loads(ready.read_text(encoding="utf-8"))["instance_id"]
            cases["capabilities"] = command("capabilities")
            cases["initial_channels"] = command("query", "capture.channels")
            cases["initial_settings"] = command("query", "settings.current")
            original = config.read_bytes()
            edit_file = root / "edit.json"
            edit_file.write_text(json.dumps({"changes": {"locale": "ja"}}), encoding="utf-8")
            lock = SettingsReplacementLock(config)
            try:
                cases["denied_settings_change"] = command(
                    "settings", "apply", "--file", str(edit_file)
                )
            finally:
                lock.close()
            receipt = cases["denied_settings_change"]["json"][-1]
            operation = receipt["operation_id"]
            cases["settings_file_unchanged"] = original == config.read_bytes()
            cases["effective_settings"] = command("query", "settings.current")
            cases["effective_channels"] = command("query", "capture.channels")
            cases["retained_failure"] = command(
                "query",
                "diagnostics.history",
                "--arguments",
                json.dumps({"operation_id": operation, "instance_id": instance, "limit": 10}),
            )
            cases["operation_after_failure"] = command("operation", "status", operation)
            evidence["commands"].append(
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
                [
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
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                follower.communicate(timeout=2)
                raise RuntimeError("diagnostic_follow_ended_unexpectedly")
            except subprocess.TimeoutExpired:
                follower.terminate()
                stdout, stderr = follower.communicate(timeout=10)
                cases["follow_replay"] = {
                    "consumer_stopped_after_replay": True,
                    "json": [
                        json.loads(line)
                        for line in stdout.decode("utf-8").splitlines()
                        if line.strip()
                    ],
                    "stderr_bytes": len(stderr),
                }
            time.sleep(1.2)
            cases["diagnostic_health"] = command("query", "diagnostics.health")
            health = cases["diagnostic_health"]["json"][-1]
            if (
                health["events"]["active_subscribers"] != 0
                or health["events"]["disconnected_consumers"] < 1
            ):
                raise RuntimeError("abandoned_follow_subscription_not_released")
            if health["file"]["delivery_failures"] or health["file"]["dropped_records"]:
                raise RuntimeError("healthy_file_projection_reported_loss")
            log_path = root / "localappdata" / "puripuly-heart" / "puripuly_heart.log"
            safe_lines = [
                line.partition("[Failure] ")[2]
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if "[Failure] {" in line
            ]
            cases["selected_main_file_evidence"] = [
                json.loads(line)
                for line in safe_lines
                if json.loads(line).get("operation_id") == operation
            ]
            primary = receipt.get("transaction", {}).get("diagnostics")
            retained = cases["retained_failure"]["json"][-1]["records"]
            replay = cases["follow_replay"]["json"]
            if (
                cases["denied_settings_change"]["exit_code"] != 4
                or receipt["status"] != "persistence_failed"
                or not primary
                or primary["code"] != "settings_save_failed"
            ):
                raise RuntimeError("coded_settings_denial_receipt_missing")
            if not any(
                record["diagnostic_id"] == primary["diagnostic_id"]
                and record["operation_id"] == operation
                and record["instance_id"] == instance
                for record in retained
            ):
                raise RuntimeError("retrospective_failure_identity_missing")
            if not any(
                event.get("diagnostic", {}).get("diagnostic_id") == primary["diagnostic_id"]
                for event in replay
            ):
                raise RuntimeError("follow_replay_failure_identity_missing")
            if not any(
                record["diagnostic_id"] == primary["diagnostic_id"]
                for record in cases["selected_main_file_evidence"]
            ):
                raise RuntimeError("selected_main_file_identity_missing")
            stop_host(host, "normal_stop")
            host = None
            host = start_host()
            cases["post_restart_old_instance"] = command(
                "query",
                "diagnostics.history",
                "--arguments",
                json.dumps({"operation_id": operation, "instance_id": instance}),
            )
            if cases["post_restart_old_instance"]["json"][-1]["lookup"] != "foreign_instance":
                raise RuntimeError("restart_scope_not_explicit")
            stop_host(host, "restart_stop")
            host = None
            failed_root = root / "failed-file-localappdata"
            failed_log = failed_root / "puripuly-heart" / "puripuly_heart.log"
            failed_log.mkdir(parents=True)
            environment["LOCALAPPDATA"] = str(failed_root)
            host = start_host()
            time.sleep(1.2)
            cases["failed_file_health"] = command("query", "diagnostics.health")
            cases["failed_file_history"] = command(
                "query", "diagnostics.history", "--arguments", json.dumps({"limit": 100})
            )
            failed_health = cases["failed_file_health"]["json"][-1]
            if (
                failed_health["file"]["availability"] != "failed"
                or failed_health["file"]["delivery_failures"] < 1
            ):
                raise RuntimeError("failed_file_destination_not_observable")
            causes = cases["failed_file_history"]["json"][-1]["records"]
            if not any(
                record["code"] == "log_file_delivery_failed"
                and record["fields"].get("errno") is not None
                for record in causes
            ):
                raise RuntimeError("failed_file_safe_native_cause_missing")
            cases["failed_file_destination_evidence"] = {
                "target_is_directory": failed_log.is_dir(),
                "file_records_written": failed_health["file"]["written_records"],
                "control_query_available": True,
            }
            stop_host(host, "failed_file_stop")
            host = None
        finally:
            if host is not None:
                stop_host(host, "finally_stop")
    evidence["source_sha256"] = {
        str(path.relative_to(source_root))
        .replace("\\", "/"): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in (
            Path(__file__),
            source_root / "puripuly_heart/core/diagnostics.py",
            source_root / "puripuly_heart/core/runtime_logging.py",
            source_root / "puripuly_heart/app/services/application_control.py",
        )
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.report)
    print(json.dumps({"result": "passed", "report_written": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
