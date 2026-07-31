from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from puripuly_heart.core.overlay import process as process_module
from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest


def _overlay_manifest(**overrides: object) -> OverlayLaunchManifest:
    values: dict[str, object] = {
        "contract_version": OVERLAY_CONTRACT_VERSION,
        "app_version": "test",
        "overlay_instance_id": "overlay-test",
        "bridge_url": "ws://127.0.0.1:8765",
        "session_token": "session-token",
        "parent_pid": 1234,
        "startup_deadline_ms": 3000,
        "log_dir": "logs",
        "log_level": "INFO",
        "locale": "en",
    }
    values.update(overrides)
    return OverlayLaunchManifest(**values)  # type: ignore[arg-type]


def test_desktop_runner_source_command_uses_current_python_module(tmp_path: Path) -> None:
    manifest_path = tmp_path / "overlay-manifest.json"
    python_path = tmp_path / "python.exe"
    runner = process_module.DesktopFletOverlayRunner(
        frozen=False,
        python_executable=python_path,
    )

    assert runner.prepare(_overlay_manifest()) == python_path
    assert runner.build_command(manifest_path) == (
        str(python_path),
        "-m",
        "puripuly_heart.ui.desktop_overlay",
        "--config",
        str(manifest_path),
    )


def test_desktop_runner_frozen_command_uses_app_executable_subcommand(tmp_path: Path) -> None:
    manifest_path = tmp_path / "overlay-manifest.json"
    app_executable = tmp_path / "PuriPulyHeart.exe"
    runner = process_module.DesktopFletOverlayRunner(
        frozen=True,
        app_executable=app_executable,
    )

    assert runner.prepare(_overlay_manifest()) == app_executable
    assert runner.build_command(manifest_path) == (
        str(app_executable),
        "run-desktop-overlay",
        "--config",
        str(manifest_path),
    )


def test_desktop_runner_command_does_not_resolve_native_overlay_or_openvr_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_native_resolution(*_args, **_kwargs):
        raise AssertionError("desktop runner must not resolve PuriPulyHeartOverlay.exe")

    def fail_openvr_check(*_args, **_kwargs):
        raise AssertionError("desktop runner must not check OpenVR DLLs")

    def fail_stale_build_check(*_args, **_kwargs):
        raise AssertionError("desktop runner must not check stale Rust overlay builds")

    monkeypatch.setattr(
        process_module.DefaultOverlayProcessRunner,
        "resolve_default_executable",
        fail_native_resolution,
    )
    monkeypatch.setattr(
        process_module.DefaultOverlayProcessRunner,
        "ensure_bundled_openvr_runtime_dll",
        fail_openvr_check,
    )
    monkeypatch.setattr(
        process_module.DefaultOverlayProcessRunner,
        "_newer_local_dev_overlay_source",
        fail_stale_build_check,
    )

    runner = process_module.DesktopFletOverlayRunner(
        frozen=False,
        python_executable=tmp_path / "python.exe",
    )

    assert runner.prepare(_overlay_manifest()) == tmp_path / "python.exe"
    assert "PuriPulyHeartOverlay.exe" not in " ".join(
        runner.build_command(tmp_path / "overlay-manifest.json")
    )


def test_overlay_manifest_serialization_omits_target_and_desktop_runtime_fields() -> None:
    payload = _overlay_manifest().to_dict()

    assert "overlay_target" not in payload
    assert "target" not in payload
    assert "desktop_flet" not in payload
    assert "supersample_scale" not in payload
    assert "desktop_supersample_scale" not in payload


def _run_isolated_cli_import_probe(
    args: list[str],
    *,
    patch_preview_runner: bool = False,
) -> tuple[int, dict[str, bool], str, str]:
    forbidden_modules = [
        "puripuly_heart.app.wiring",
        "puripuly_heart.core.openrouter.managed_openrouter_broker_client",
        "puripuly_heart.core.storage.secrets",
        "puripuly_heart.core.stt.backend",
        "puripuly_heart.core.stt.controller",
        "puripuly_heart.domain.models",
        "puripuly_heart.providers.llm.deepseek",
        "puripuly_heart.providers.llm.openrouter",
        "puripuly_heart.providers.llm.gemini",
        "puripuly_heart.providers.llm.local_openai",
        "puripuly_heart.providers.llm.qwen",
        "puripuly_heart.providers.llm.qwen_async",
        "puripuly_heart.providers.stt.deepgram",
        "puripuly_heart.providers.stt.local_qwen_sherpa",
        "puripuly_heart.providers.stt.qwen_asr",
        "puripuly_heart.providers.stt.soniox",
        "puripuly_heart.ui.app",
    ]
    script = "\n".join(
        [
            "import json",
            "import sys",
            "from puripuly_heart import main as main_module",
            *(
                [
                    "from puripuly_heart.ui import desktop_overlay as desktop_overlay_module",
                    "desktop_overlay_module._default_preview_app_runner = lambda target: None",
                ]
                if patch_preview_runner
                else []
            ),
            f"result = main_module.main({args!r})",
            f"forbidden = {forbidden_modules!r}",
            "print('IMPORT_PROBE=' + json.dumps({name: name in sys.modules for name in forbidden}, sort_keys=True))",
            "raise SystemExit(result)",
        ]
    )
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    marker = "IMPORT_PROBE="
    probe_line = next(
        (line[len(marker) :] for line in completed.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    assert probe_line is not None, (
        "isolated CLI import probe did not emit module state\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    return completed.returncode, json.loads(probe_line), completed.stdout, completed.stderr


def test_import_run_desktop_overlay_dispatch_is_provider_secret_and_stt_free(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "overlay-manifest.json"
    manifest_path.write_text(
        json.dumps(
            _overlay_manifest(
                bridge_url="ws://127.0.0.1:9",
                parent_pid=os.getpid(),
                startup_deadline_ms=50,
            ).to_dict()
        ),
        encoding="utf-8",
    )

    returncode, imported, stdout, stderr = _run_isolated_cli_import_probe(
        ["run-desktop-overlay", "--config", str(manifest_path)]
    )

    assert returncode == 1, f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert "startup_error" in stdout + stderr
    assert "session-token" not in stdout + stderr
    assert imported == dict.fromkeys(imported, False)


def test_import_preview_dispatch_is_provider_secret_and_stt_free() -> None:
    returncode, imported, stdout, stderr = _run_isolated_cli_import_probe(
        ["run-desktop-overlay-preview"],
        patch_preview_runner=True,
    )

    assert returncode == 0, f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert imported == dict.fromkeys(imported, False)
