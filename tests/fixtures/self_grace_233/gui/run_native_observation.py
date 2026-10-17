from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
from pywinauto import Application

TITLE = "SELF-STREAM-2 isolated presentation proof"
STAGES = (
    "source",
    "preview-one",
    "extended-source",
    "preview-two",
    "source-closed",
    "identical-final",
    "earlier-pending",
    "successor-preview",
    "delayed-old-final",
    "cancelled",
    "secondary-first",
    "dual-preview",
    "dual-final",
    "blocked-ui",
    "released-ui",
)


def wait_until(condition, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError("Owned native GUI condition was not observed before its deadline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desktop-host-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8174)
    parser.add_argument("--owner-repairs", action="store_true")
    args = parser.parse_args()
    expected_title = TITLE
    stages = STAGES
    runner = ["tests/fixtures/self_grace_233/gui/run_presentation.py"]
    if args.owner_repairs:
        expected_title = "SELF-STREAM-2 owner presentation repair proof"
        stages = (
            "progressive-source",
            "progressive-preview-one",
            "progressive-extended-source",
            "progressive-preview-two",
            "progressive-final",
            "older-speech-held",
            "manual-complete",
            "late-speech-final",
            "successor-speech",
            "dual-provisional",
            "dual-secondary-first",
            "dual-final-after-snapshots",
            "dual-final-before-snapshots",
            "partial-primary-preview",
            "partial-target-failure",
            "older-manual-held",
            "newer-speech-complete",
            "late-manual-final",
            "future-manual-complete",
            "queued-legacy-source-final",
            "pending-manual-source",
            "late-manual-while-source-pending",
            "pending-manual-admitted",
            "segmented-old-first-held",
            "segmented-new-managed",
            "segmented-old-released",
            "segmented-future-independent",
            "segmented-future-managed",
            "pressure-old-speech",
            "manual-source-timeout",
            "manual-result-after-source-timeout",
            "pressure-successor-managed",
        )
        runner = ["-m", "tests.fixtures.self_grace_233.gui.run_owner_presentation"]
    host_exe = (args.desktop_host_dir / "flet.exe").resolve(strict=True)
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    isolated_home = Path(tempfile.mkdtemp(prefix="self-stream-233-native-"))
    app_data = isolated_home / "AppData"
    local_app_data = app_data / "Local"
    roaming_app_data = app_data / "Roaming"
    local_app_data.mkdir(parents=True)
    roaming_app_data.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "USERPROFILE": str(isolated_home),
            "HOME": str(isolated_home),
            "LOCALAPPDATA": str(local_app_data),
            "APPDATA": str(roaming_app_data),
            "FLET_VIEW_PATH": str(host_exe.parent),
            "FLET_FORCE_WEB_SERVER": "false",
        }
    )
    record = {
        "boundary": (
            "New owned native Flet host, isolated actual local owner/logger/UI composition; no application bootstrap or canonical profile"
            if args.owner_repairs
            else "New owned native Flet host, isolated synthetic UI runner; no application composition or canonical profile"
        ),
        "host_executable": "<user-home>/.flet/client/flet-desktop-full-1.0.0/flet/flet.exe",
        "host_sha256": hashlib.sha256(host_exe.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_isolated": True,
        "stages": [],
        "status": "started",
        "scenario": "owner_repairs" if args.owner_repairs else "snapshot_ingress",
    }
    window = None
    process = None
    host_process = None
    output_file = (report_dir / "runner.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                *runner,
                "--native",
                "--report-dir",
                str(report_dir),
                "--port",
                str(args.port),
            ],
            env=environment,
            stdout=output_file,
            stderr=subprocess.STDOUT,
        )
        record["python_pid"] = process.pid

        def find_owned_window():
            if process.poll() is not None:
                raise RuntimeError(
                    f"Synthetic runner exited before native host readiness: {process.returncode}"
                )
            owned = {process.pid: psutil.Process(process.pid)}
            owned.update(
                (child.pid, child) for child in psutil.Process(process.pid).children(recursive=True)
            )
            matches = []
            observed = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def visit(handle, _parameter):
                pid = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
                if pid.value not in owned or not ctypes.windll.user32.IsWindowVisible(handle):
                    return True
                length = ctypes.windll.user32.GetWindowTextLengthW(handle)
                title = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(handle, title, length + 1)
                observed.append({"pid": pid.value, "title": title.value})
                if title.value == expected_title:
                    matches.append((handle, owned[pid.value]))
                return True

            ctypes.windll.user32.EnumWindows(visit, 0)
            record["owned_visible_windows"] = observed
            return matches[0] if matches else None

        record["phase"] = "finding_owned_window"
        handle, host_process = wait_until(find_owned_window, timeout=40)
        record["native_host_pid"] = host_process.pid
        record["native_host_parent_pid"] = host_process.ppid()
        app = Application(backend="win32").connect(handle=handle, timeout=10)
        window = app.window(handle=handle)
        window.wait("visible enabled", timeout=15)
        record["window_handle"] = window.handle
        record["window_title"] = window.window_text()
        record["window_process_matches_owned_host"] = window.process_id() == host_process.pid
        rectangle = window.rectangle()
        record["window_rectangle"] = [
            rectangle.left,
            rectangle.top,
            rectangle.right,
            rectangle.bottom,
        ]
        if not record["window_process_matches_owned_host"]:
            raise RuntimeError("Window process does not match the newly owned native host")

        record["phase"] = "verifying_interactable_owned_bounds"
        wait_until(
            lambda: window.rectangle().width() >= 1580 and window.rectangle().height() >= 1000,
            timeout=15,
        )
        rectangle = window.rectangle()
        record["window_rectangle"] = [
            rectangle.left,
            rectangle.top,
            rectangle.right,
            rectangle.bottom,
        ]
        record["native_bounds_verified"] = (
            rectangle.left >= 0
            and rectangle.top >= 0
            and rectangle.right <= ctypes.windll.user32.GetSystemMetrics(0)
            and rectangle.bottom <= ctypes.windll.user32.GetSystemMetrics(1)
        )
        if not record["native_bounds_verified"]:
            raise RuntimeError("Owned controlled probe bounds are not fully on-screen")

        def invoke_owned_button(title):
            button = uia_window.child_window(title=title, control_type="Button")
            button.wait("visible enabled", timeout=15)
            wrapper = button.wrapper_object()
            if wrapper.process_id() != host_process.pid:
                raise RuntimeError("Native control is not owned; refusing invocation")
            wrapper.invoke()
            record["interaction_boundary"] = (
                "UI Automation Invoke on visible controls within verified owned HWND/PID; no global pointer or keyboard input"
            )

        def capture(name, surface_name):
            surface_path = report_dir / surface_name
            previous_capture = surface_path.stat().st_mtime_ns if surface_path.exists() else None
            invoke_owned_button("Capture surface")
            wait_until(
                lambda: surface_path.exists()
                and surface_path.stat().st_mtime_ns != previous_capture
            )
            shutil.copyfile(surface_path, report_dir / name)
            record["capture_boundary"] = (
                "Native production Screenshot control only; no desktop or other windows captured"
            )
            record["capture_interaction"] = (
                "Owned visible native Capture surface control invoked; new render image observed"
            )

        def controls():
            return [
                {"name": child.element_info.name, "type": child.element_info.control_type}
                for child in uia_window.descendants()
            ]

        record["phase"] = "connecting_owned_window_accessibility"
        uia_app = Application(backend="uia").connect(handle=handle, timeout=15)
        uia_window = uia_app.window(handle=handle)
        record["initial_controls"] = controls()
        record["phase"] = "invoking_owned_native_controls"
        conversation = uia_window.child_window(title="Conversation View", control_type="Button")
        conversation.wait("visible enabled", timeout=15)
        record["conversation_control_visible"] = True
        capture("00-native-ready.png", "00-ready.png")
        invoke_owned_button("Conversation View")
        for number, stage in enumerate(stages, start=1):
            record["phase"] = f"observing_native_stage:{stage}"
            invoke_owned_button("Next snapshot")

            def consumed_stage():
                try:
                    observations = json.loads(
                        (report_dir / "observations.json").read_text(encoding="utf-8")
                    )
                except FileNotFoundError, json.JSONDecodeError:
                    return None
                values = observations["stages"]
                if len(values) > number:
                    raise RuntimeError("Native control invocation skipped an observation")
                return (
                    observations if len(values) == number and values[-1]["stage"] == stage else None
                )

            observations = wait_until(consumed_stage)
            observation = observations["stages"][-1]
            capture(f"{number:02d}-native-{stage}.png", f"{number:02d}-{stage}.png")
            stage_observation = {
                "stage": stage,
                "consumer_stage_received": True,
                "native_render_received": True,
                "dashboard": observation["dashboard"],
                "dashboard_status": observation["dashboard_status"],
                "retained_rows": len(observation["rows"]),
                "accepted_record_keys": observation["accepted_record_keys"],
            }
            if args.owner_repairs:
                stage_observation.update(
                    dashboard_status_visible=observation["dashboard_status_visible"],
                    dashboard_visible_status=observation["dashboard_visible_status"],
                )
            record["stages"].append(stage_observation)
        record["phase"] = "showing_native_system_logs"
        invoke_owned_button("System Logs")
        uia_window.child_window(title="Conversation View", control_type="Button").wait(
            "visible enabled", timeout=15
        )
        capture(
            f"{len(stages) + 1:02d}-native-system-logs.png", f"{len(stages):02d}-{stages[-1]}.png"
        )
        record["phase"] = "restoring_native_conversation"
        invoke_owned_button("Conversation View")
        uia_window.child_window(title="System Logs", control_type="Button").wait(
            "visible enabled", timeout=15
        )
        capture(
            f"{len(stages) + 2:02d}-native-conversation-restored.png",
            f"{len(stages):02d}-{stages[-1]}.png",
        )
        record["status"] = "passed"
    except Exception as exc:
        message = str(exc).replace(str(isolated_home), "<isolated-home>")
        message = message.replace(str(Path.home()), "<user-home>")
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": message}
    finally:
        if window is not None:
            try:
                window.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                for child in psutil.Process(process.pid).children(recursive=True):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait(timeout=10)
            record["runner_returncode"] = process.returncode
        record["native_host_exited"] = host_process is None or not host_process.is_running()
        output_file.close()
        if args.owner_repairs and (report_dir / "instance_exit.json").exists():
            unchanged = json.loads((report_dir / "instance_exit.json").read_text(encoding="utf-8"))[
                "entry_exit_unchanged"
            ]
            record["entry_exit_source_unchanged"] = unchanged
            if not unchanged:
                record["status"] = "failed"
                record["error"] = {
                    "type": "SourceChanged",
                    "message": "Observed source fingerprint changed during native proof",
                }
        elif args.owner_repairs:
            record["entry_exit_source_unchanged"] = None
            if record["status"] == "passed":
                record["status"] = "failed"
                record["error"] = {
                    "type": "MissingSourceExit",
                    "message": "Native proof did not record exit fingerprints",
                }
        try:
            shutil.rmtree(isolated_home)
            record["isolated_profile_removed"] = True
        except OSError as exc:
            record["isolated_profile_removed"] = False
            record["profile_cleanup_error"] = type(exc).__name__
        (report_dir / "native_host.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": record["status"],
                "stages": len(record["stages"]),
                "error": record.get("error"),
                "native_host_exited": record["native_host_exited"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
