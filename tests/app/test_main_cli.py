from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import sys
from logging.handlers import QueueHandler
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import puripuly_heart.main as main_module
from puripuly_heart import __version__
from puripuly_heart.core.runtime_logging import (
    SessionRuntimeLoggingService,
    configure_main_logging,
)

FAKE_APP_VIEW = SimpleNamespace(FLET_APP_HIDDEN=object())


def test_main_version_prints(capsys) -> None:
    result = main_module.main(["--version"])
    assert result == 0
    assert capsys.readouterr().out.strip() == __version__


def test_main_version_prints_without_soxr_runtime_startup_check(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        main_module,
        "ensure_soxr_runtime_available_for_startup",
        lambda: pytest.fail("--version should not run the soxr startup check"),
        raising=False,
    )

    result = main_module.main(["--version"])

    assert result == 0
    assert capsys.readouterr().out.strip() == __version__


def test_main_gui_startup_check_imports_gui_runtime_without_starting_flet(monkeypatch) -> None:
    import flet

    monkeypatch.setattr(
        flet,
        "run",
        lambda **_kwargs: pytest.fail("gui-startup-check should not start Flet"),
    )

    assert main_module.main(["gui-startup-check"]) == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["osc-send", "hello"],
        ["run-stdin"],
        ["run-mic"],
    ],
)
def test_main_rejects_removed_cli_commands(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main_module.main(argv)

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_main_run_gui_invokes_flet_run(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, view):
        calls["main"] = main
        calls["assets_dir"] = assets_dir
        calls["view"] = view

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)

    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(page, *, config_path, debug_ui_preview=False, **_kwargs):
        _ = page
        calls["config_path"] = config_path
        calls["debug_ui_preview"] = debug_ui_preview

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)

    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    config_path = tmp_path / "settings.json"
    result = main_module.main(["--config", str(config_path), "run-gui"])

    assert result == 0
    assert calls["assets_dir"] == str(tmp_path)
    assert calls["view"] is FAKE_APP_VIEW.FLET_APP_HIDDEN
    assert callable(calls["main"])
    asyncio.run(calls["main"](object()))
    assert calls["config_path"] == config_path
    assert calls["debug_ui_preview"] is False


def test_run_gui_logs_actionable_flet_runtime_startup_failure(
    monkeypatch, tmp_path, caplog
) -> None:
    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        _ = main, assets_dir
        raise FileNotFoundError("bundled Flet archive missing")

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)
    fake_ui_app = ModuleType("puripuly_heart.ui.app")
    fake_ui_app.main_gui = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)
    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    with caplog.at_level(logging.ERROR, logger=main_module.__name__):
        with pytest.raises(FileNotFoundError, match="bundled Flet archive missing"):
            main_module._run_gui(
                tmp_path / "settings.json",
                debug_ui_preview=False,
                runtime_logging_sinks=object(),
            )

    assert any(
        "Flet GUI runtime failed: exception_type=FileNotFoundError "
        "exception_message=bundled Flet archive missing" in record.getMessage()
        for record in caplog.records
    )


def test_run_gui_logs_actionable_ui_startup_failure(monkeypatch, tmp_path, caplog) -> None:
    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        _ = assets_dir
        asyncio.run(main(object()))

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)
    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(*_args, **_kwargs):
        raise RuntimeError("application boundary construction failed")

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)
    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    with caplog.at_level(logging.ERROR, logger=main_module.__name__):
        with pytest.raises(RuntimeError, match="application boundary construction failed"):
            main_module._run_gui(
                tmp_path / "settings.json",
                debug_ui_preview=False,
                runtime_logging_sinks=object(),
            )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "GUI startup failed: exception_type=RuntimeError "
        "exception_message=application boundary construction failed" in message
        for message in messages
    )
    assert any(
        "Flet GUI runtime failed: exception_type=RuntimeError "
        "exception_message=application boundary construction failed" in message
        for message in messages
    )


def test_run_gui_forwards_main_logging_sinks_when_supported(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    logging_sinks = object()
    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        _ = assets_dir
        asyncio.run(main(object()))

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)

    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(
        page,
        *,
        config_path,
        application_factory=None,
        debug_ui_preview=False,
        runtime_logging_sinks=None,
        vrchat_osc_presence=None,
    ):
        calls.update(
            page=page,
            config_path=config_path,
            application_factory=application_factory,
            debug_ui_preview=debug_ui_preview,
            runtime_logging_sinks=runtime_logging_sinks,
            vrchat_osc_presence=vrchat_osc_presence,
        )

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)

    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    config_path = tmp_path / "settings.json"
    result = main_module._run_gui(
        config_path,
        debug_ui_preview=False,
        runtime_logging_sinks=logging_sinks,
    )

    assert result == 0
    assert calls["config_path"] == config_path
    from puripuly_heart.composition.ui_application import compose_ui_application

    assert calls["application_factory"] is compose_ui_application
    assert calls["runtime_logging_sinks"] is logging_sinks
    assert calls["vrchat_osc_presence"] is not None


def test_main_default_invokes_gui(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        calls["main"] = main
        calls["assets_dir"] = assets_dir

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)

    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(page, *, config_path, debug_ui_preview=False, **_kwargs):
        _ = page
        calls["config_path"] = config_path
        calls["debug_ui_preview"] = debug_ui_preview

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)

    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    config_path = tmp_path / "settings.json"
    result = main_module.main(["--config", str(config_path)])

    assert result == 0
    assert calls["assets_dir"] == str(tmp_path)
    assert callable(calls["main"])
    asyncio.run(calls["main"](object()))
    assert calls["config_path"] == config_path
    assert calls["debug_ui_preview"] is False


def test_main_run_gui_passes_debug_ui_preview_flag(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        calls["main"] = main
        calls["assets_dir"] = assets_dir

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)

    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(page, *, config_path, debug_ui_preview=False, **_kwargs):
        _ = page
        calls["config_path"] = config_path
        calls["debug_ui_preview"] = debug_ui_preview

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)

    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    config_path = tmp_path / "settings.json"
    result = main_module.main(["--config", str(config_path), "run-gui", "--debug-ui-preview"])

    assert result == 0
    assert calls["assets_dir"] == str(tmp_path)
    asyncio.run(calls["main"](object()))
    assert calls["config_path"] == config_path
    assert calls["debug_ui_preview"] is True


def test_run_gui_debug_preview_constructs_vrchat_presence(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        _ = assets_dir
        asyncio.run(main(object()))

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)
    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(
        page,
        *,
        config_path,
        application_factory=None,
        debug_ui_preview=False,
        runtime_logging_sinks=None,
        vrchat_osc_presence=None,
    ):
        calls.update(
            page=page,
            config_path=config_path,
            debug_ui_preview=debug_ui_preview,
            runtime_logging_sinks=runtime_logging_sinks,
            vrchat_osc_presence=vrchat_osc_presence,
        )

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)
    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    result = main_module._run_gui(
        tmp_path / "settings.json",
        debug_ui_preview=True,
        runtime_logging_sinks=object(),
    )

    assert result == 0
    assert calls["debug_ui_preview"] is True
    assert calls["vrchat_osc_presence"] is not None


def test_main_run_gui_force_closes_logging_when_gui_runtime_logging_leaks(
    monkeypatch, tmp_path
) -> None:
    root_logger = logging.getLogger(f"test.main.gui.logging.force_close.{tmp_path.name}")
    root_logger.handlers.clear()
    root_logger.propagate = False
    leaked_services: list[SessionRuntimeLoggingService] = []
    monkeypatch.setattr("puripuly_heart.core.runtime_logging.user_config_dir", lambda: tmp_path)

    monkeypatch.setattr(
        main_module,
        "configure_main_logging",
        lambda *, log_dir=None: configure_main_logging(
            root_logger=root_logger,
            log_dir=log_dir,
        ),
    )

    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        _ = assets_dir
        asyncio.run(main(object()))

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)

    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(page, *, config_path, debug_ui_preview=False, **_kwargs):
        _ = page, config_path, debug_ui_preview
        leaked_services.append(SessionRuntimeLoggingService(root_logger=root_logger))

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)

    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    try:
        result = main_module.main(["--config", str(tmp_path / "settings.json"), "run-gui"])

        assert result == 0
        assert leaked_services
        assert [
            handler for handler in root_logger.handlers if isinstance(handler, QueueHandler)
        ] == []
    finally:
        for service in leaked_services:
            service.close()


def test_main_default_gui_passes_debug_ui_preview_flag(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    fake_flet = ModuleType("flet")

    def fake_run(*, main, assets_dir, **_kwargs):
        calls["main"] = main
        calls["assets_dir"] = assets_dir

    fake_flet.run = fake_run
    fake_flet.AppView = FAKE_APP_VIEW
    monkeypatch.setitem(sys.modules, "flet", fake_flet)

    fake_ui_app = ModuleType("puripuly_heart.ui.app")

    async def main_gui(page, *, config_path, debug_ui_preview=False, **_kwargs):
        _ = page
        calls["config_path"] = config_path
        calls["debug_ui_preview"] = debug_ui_preview

    fake_ui_app.main_gui = main_gui
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.app", fake_ui_app)

    fake_fonts = ModuleType("puripuly_heart.ui.fonts")
    fake_fonts.assets_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "puripuly_heart.ui.fonts", fake_fonts)

    config_path = tmp_path / "settings.json"
    result = main_module.main(["--config", str(config_path), "--debug-ui-preview"])

    assert result == 0
    assert calls["assets_dir"] == str(tmp_path)
    asyncio.run(calls["main"](object()))
    assert calls["config_path"] == config_path
    assert calls["debug_ui_preview"] is True


def test_real_main_gui_accepts_debug_ui_preview_keyword_only() -> None:
    from puripuly_heart.ui.app import main_gui

    parameters = inspect.signature(main_gui).parameters

    assert "debug_ui_preview" in parameters
    debug_ui_preview = parameters["debug_ui_preview"]
    assert debug_ui_preview.kind is inspect.Parameter.KEYWORD_ONLY
    assert debug_ui_preview.default is False


def test_main_local_qwen_runtime_check_dispatches_runner(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_run_local_qwen_runtime_check() -> int:
        calls["called"] = True
        return 0

    monkeypatch.setattr(
        main_module,
        "run_local_qwen_runtime_check",
        fake_run_local_qwen_runtime_check,
        raising=False,
    )

    config_path = tmp_path / "settings.json"
    try:
        result = main_module.main(["--config", str(config_path), "local-qwen-runtime-check"])
    except SystemExit as exc:  # pragma: no cover - red phase guard
        pytest.fail(f"unexpected SystemExit: {exc}")

    assert result == 0
    assert calls["called"] is True


def test_main_soxr_runtime_check_dispatches_runner(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        main_module,
        "ensure_soxr_runtime_available_for_startup",
        lambda: None,
        raising=False,
    )

    def fake_run_soxr_runtime_check() -> int:
        calls["called"] = True
        return 0

    monkeypatch.setattr(
        main_module,
        "run_soxr_runtime_check",
        fake_run_soxr_runtime_check,
        raising=False,
    )

    config_path = tmp_path / "settings.json"
    try:
        result = main_module.main(["--config", str(config_path), "soxr-runtime-check"])
    except SystemExit as exc:  # pragma: no cover - red phase guard
        pytest.fail(f"unexpected SystemExit: {exc}")

    assert result == 0
    assert calls["called"] is True


def test_main_local_asr_production_composition_evidence_dispatches_paths(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}

    def run_evidence(**kwargs) -> int:
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(
        main_module,
        "run_local_asr_production_composition_evidence",
        run_evidence,
    )
    audio = tmp_path / "speech.wav"
    report = tmp_path / "result.json"

    result = main_module.main(
        [
            "local-asr-production-composition-evidence",
            "--audio",
            str(audio),
            "--report",
            str(report),
            "--candidate",
            "candidate-sha",
            "--expected-gpu-name",
            "RX 7900 XTX",
        ]
    )

    assert result == 0
    assert calls == {
        "audio_path": audio,
        "report_path": report,
        "candidate": "candidate-sha",
        "expected_gpu_name": "RX 7900 XTX",
    }


def test_main_local_cpu_real_model_check_dispatches_isolated_paths(monkeypatch, tmp_path) -> None:
    calls: dict[str, Path] = {}

    def fake_run_local_cpu_real_model_check(
        *,
        model_root: Path,
        audio_root: Path,
        report_path: Path,
    ) -> int:
        calls["model_root"] = model_root
        calls["audio_root"] = audio_root
        calls["report_path"] = report_path
        return 0

    monkeypatch.setattr(
        main_module,
        "run_local_cpu_real_model_check",
        fake_run_local_cpu_real_model_check,
    )
    config_path = tmp_path / "settings.json"
    model_root = tmp_path / "models"
    audio_root = tmp_path / "audio"
    report_path = tmp_path / "report.json"

    result = main_module.main(
        [
            "--config",
            str(config_path),
            "local-cpu-real-model-check",
            "--model-root",
            str(model_root),
            "--audio-root",
            str(audio_root),
            "--report",
            str(report_path),
        ]
    )

    assert result == 0
    assert calls == {
        "model_root": model_root,
        "audio_root": audio_root,
        "report_path": report_path,
    }


def test_run_soxr_runtime_check_rejects_non_windows(monkeypatch, capsys) -> None:
    try:
        runtime_check_module = importlib.import_module("puripuly_heart.app.soxr_runtime_check")
    except ModuleNotFoundError:  # pragma: no cover - red phase guard
        pytest.fail("soxr_runtime_check module is missing")

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(
        runtime_check_module,
        "ensure_soxr_runtime_available_for_startup",
        lambda: pytest.fail("should not validate soxr runtime on non-Windows"),
        raising=False,
    )

    result = runtime_check_module.run_soxr_runtime_check()

    assert result == 2
    assert (
        capsys.readouterr().out.strip() == "Error: soxr-runtime-check is only supported on Windows"
    )


def test_run_soxr_runtime_check_reports_runtime_validation_failure(monkeypatch, capsys) -> None:
    runtime_check_module = importlib.import_module("puripuly_heart.app.soxr_runtime_check")

    class FakeSoxrRuntimeAvailabilityError(RuntimeError):
        pass

    def raise_runtime_error() -> None:
        raise FakeSoxrRuntimeAvailabilityError("missing packaged soxr sibling dll")

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        runtime_check_module,
        "SoxrRuntimeAvailabilityError",
        FakeSoxrRuntimeAvailabilityError,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_check_module,
        "ensure_soxr_runtime_available_for_startup",
        raise_runtime_error,
        raising=False,
    )

    result = runtime_check_module.run_soxr_runtime_check()

    assert result == 2
    assert capsys.readouterr().out.strip() == (
        "Error: failed to verify packaged soxr runtime: missing packaged soxr sibling dll"
    )


def test_run_soxr_runtime_check_reports_soxr_import_or_smoke_failure(
    monkeypatch, capsys, tmp_path
) -> None:
    runtime_check_module = importlib.import_module("puripuly_heart.app.soxr_runtime_check")

    runtime_paths = type(
        "RuntimePaths",
        (),
        {
            "extension_path": tmp_path / "soxr_ext.cp312-win_amd64.pyd",
            "runtime_dir": tmp_path,
            "sibling_dll_path": tmp_path / "soxr.dll",
        },
    )()

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        runtime_check_module,
        "ensure_soxr_runtime_available_for_startup",
        lambda: runtime_paths,
        raising=False,
    )
    real_import_module = runtime_check_module.importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name == "soxr":
            raise ImportError("native extension load failed")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(
        runtime_check_module.importlib,
        "import_module",
        fake_import_module,
    )

    result = runtime_check_module.run_soxr_runtime_check()

    assert result == 2
    assert capsys.readouterr().out.strip() == (
        "Error: failed to import or smoke-test soxr: native extension load failed"
    )


def test_run_soxr_runtime_check_imports_soxr_runs_smoke_and_reports_paths(
    monkeypatch, capsys, tmp_path
) -> None:
    runtime_check_module = importlib.import_module("puripuly_heart.app.soxr_runtime_check")
    runtime_dir = tmp_path / "soxr"
    runtime_dir.mkdir()
    extension_path = runtime_dir / "soxr_ext.cp312-win_amd64.pyd"
    extension_path.write_bytes(b"")
    sibling_dll_path = runtime_dir / "soxr.dll"
    sibling_dll_path.write_bytes(b"")

    runtime_paths = type(
        "RuntimePaths",
        (),
        {
            "extension_path": extension_path,
            "runtime_dir": runtime_dir,
            "sibling_dll_path": sibling_dll_path,
        },
    )()
    calls: dict[str, object] = {}

    class FakeResampleStream:
        def __init__(self, in_rate, out_rate, num_channels, dtype="float32"):
            calls["init"] = (in_rate, out_rate, num_channels, dtype)

        def resample_chunk(self, samples, last=False):
            calls["len"] = len(samples)
            calls["last"] = last
            return [0.0, 0.0, 0.0]

    fake_soxr = ModuleType("soxr")
    fake_soxr.ResampleStream = FakeResampleStream

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        runtime_check_module,
        "ensure_soxr_runtime_available_for_startup",
        lambda: runtime_paths,
        raising=False,
    )

    real_import_module = runtime_check_module.importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name == "soxr":
            return fake_soxr
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(runtime_check_module.importlib, "import_module", fake_import_module)

    result = runtime_check_module.run_soxr_runtime_check()

    assert result == 0
    assert calls["init"] == (48000, 16000, 1, "float32")
    assert calls["len"] == 480
    assert calls["last"] is True
    assert capsys.readouterr().out.strip().splitlines() == [
        f"soxr_extension_path={extension_path}",
        f"soxr_runtime_dir={runtime_dir}",
        f"soxr_sibling_dll={sibling_dll_path}",
    ]


def test_run_soxr_runtime_check_writes_json_report_when_env_var_is_set(
    monkeypatch, tmp_path
) -> None:
    runtime_check_module = importlib.import_module("puripuly_heart.app.soxr_runtime_check")
    report_path = tmp_path / "soxr-runtime-report.json"
    runtime_paths = type(
        "RuntimePaths",
        (),
        {
            "extension_path": Path("C:/temp/soxr/soxr_ext.cp312-win_amd64.pyd"),
            "runtime_dir": Path("C:/temp/soxr"),
            "sibling_dll_path": Path("C:/temp/soxr/soxr.dll"),
        },
    )()

    class FakeResampleStream:
        def __init__(self, in_rate, out_rate, channels, dtype="float32"):
            self.args = (in_rate, out_rate, channels, dtype)

        def resample_chunk(self, samples, last=False):
            return [0.0, 0.0, 0.0]

    fake_soxr_module = type("FakeSoxr", (), {"ResampleStream": FakeResampleStream})
    fake_soxr_ext_module = type("FakeSoxrExt", (), {"__file__": str(runtime_paths.extension_path)})

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)

    def fake_import_module(name: str):
        if name == "soxr":
            return fake_soxr_module
        if name == "soxr.soxr_ext":
            return fake_soxr_ext_module
        raise AssertionError(name)

    monkeypatch.setenv("PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH", str(report_path))
    monkeypatch.setattr(
        runtime_check_module,
        "ensure_soxr_runtime_available_for_startup",
        lambda: runtime_paths,
    )
    monkeypatch.setattr(
        runtime_check_module,
        "_resolve_loaded_soxr_dll_path",
        lambda: runtime_paths.sibling_dll_path,
    )
    monkeypatch.setattr(runtime_check_module.importlib, "import_module", fake_import_module)

    assert runtime_check_module.run_soxr_runtime_check() == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["expected_extension_path"] == str(runtime_paths.extension_path)
    assert payload["expected_sibling_dll_path"] == str(runtime_paths.sibling_dll_path)
    assert payload["imported_extension_path"] == str(runtime_paths.extension_path)
    assert payload["loaded_sibling_dll_path"] == str(runtime_paths.sibling_dll_path)


def test_run_local_qwen_runtime_check_imports_sherpa_onnx_and_offline_recognizer_before_reporting_success(
    monkeypatch, capsys, tmp_path
) -> None:
    try:
        runtime_check_module = importlib.import_module(
            "puripuly_heart.app.local_qwen_runtime_check"
        )
    except ModuleNotFoundError:  # pragma: no cover - red phase guard
        pytest.fail("local_qwen_runtime_check module is missing")

    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        runtime_check_module.local_qwen_runtime,
        "ensure_local_qwen_windows_runtime",
        lambda: tmp_path,
    )

    imported_modules: list[str] = []
    real_import_module = runtime_check_module.importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name == "sherpa_onnx":
            imported_modules.append(name)
            return ModuleType("sherpa_onnx")
        if name == "sherpa_onnx.offline_recognizer":
            imported_modules.append(name)
            return ModuleType("sherpa_onnx.offline_recognizer")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(runtime_check_module.importlib, "import_module", fake_import_module)

    result = runtime_check_module.run_local_qwen_runtime_check()

    assert result == 0
    assert imported_modules == ["sherpa_onnx", "sherpa_onnx.offline_recognizer"]
    assert capsys.readouterr().out.strip() == f"local_qwen_runtime_dir={tmp_path}"


def test_run_local_qwen_runtime_check_rejects_non_windows(monkeypatch, capsys) -> None:
    runtime_check_module = importlib.import_module("puripuly_heart.app.local_qwen_runtime_check")

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "linux", raising=False)

    result = runtime_check_module.run_local_qwen_runtime_check()

    assert result == 2
    assert capsys.readouterr().out.strip() == (
        "Error: local-qwen-runtime-check is only supported on Windows"
    )


def test_run_local_qwen_runtime_check_reports_bootstrap_failure(monkeypatch, capsys) -> None:
    runtime_check_module = importlib.import_module("puripuly_heart.app.local_qwen_runtime_check")
    runtime_error = importlib.import_module("puripuly_heart.core.local_asr.local_qwen_runtime")

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)

    def raise_bootstrap_error() -> None:
        raise runtime_error.LocalQwenRuntimeBootstrapError("missing runtime dlls")

    monkeypatch.setattr(
        runtime_check_module.local_qwen_runtime,
        "ensure_local_qwen_windows_runtime",
        raise_bootstrap_error,
    )

    result = runtime_check_module.run_local_qwen_runtime_check()

    assert result == 2
    assert capsys.readouterr().out.strip() == (
        "Error: failed to verify Local Qwen Windows runtime DLL directory: missing runtime dlls"
    )


def test_run_local_qwen_runtime_check_reports_bootstrap_failure_after_runtime_module_reload(
    monkeypatch, capsys, tmp_path
) -> None:
    runtime_check_module = importlib.reload(
        importlib.import_module("puripuly_heart.app.local_qwen_runtime_check")
    )
    runtime_module = importlib.import_module("puripuly_heart.core.local_asr.local_qwen_runtime")

    runtime_module = importlib.reload(runtime_module)

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")

    missing_runtime_dir = tmp_path / "missing-runtime"
    monkeypatch.setattr(
        runtime_module, "resolve_local_qwen_runtime_dir", lambda: missing_runtime_dir
    )

    try:
        result = runtime_check_module.run_local_qwen_runtime_check()
    finally:
        importlib.reload(runtime_check_module)

    assert result == 2
    assert capsys.readouterr().out.strip() == (
        "Error: failed to verify Local Qwen Windows runtime DLL directory: "
        f"local qwen runtime directory does not exist: {missing_runtime_dir}"
    )


def test_run_local_qwen_runtime_check_reports_sherpa_onnx_import_failure(
    monkeypatch, capsys, tmp_path
) -> None:
    runtime_check_module = importlib.import_module("puripuly_heart.app.local_qwen_runtime_check")

    monkeypatch.setattr(runtime_check_module, "sys", ModuleType("sys"), raising=False)
    monkeypatch.setattr(runtime_check_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        runtime_check_module.local_qwen_runtime,
        "ensure_local_qwen_windows_runtime",
        lambda: tmp_path,
    )
    real_import_module = runtime_check_module.importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name == "sherpa_onnx":
            return ModuleType("sherpa_onnx")
        if name == "sherpa_onnx.offline_recognizer":
            raise ImportError("native extension load failed")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(runtime_check_module.importlib, "import_module", fake_import_module)

    result = runtime_check_module.run_local_qwen_runtime_check()

    assert result == 2
    assert capsys.readouterr().out.strip() == (
        "Error: failed to import sherpa_onnx: native extension load failed"
    )


def test_load_settings_or_default_loads_when_exists(monkeypatch, tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    sentinel = object()

    class FakeLoadResult:
        settings = sentinel
        status = "ok"
        error = None
        warnings = []

    monkeypatch.setattr(
        "puripuly_heart.config.settings_vnext.facade.load_vnext_settings",
        lambda _path: FakeLoadResult(),
    )

    assert main_module._load_settings_or_default(settings_path) is sentinel


def test_load_settings_or_default_redacts_invalid_setting_value(tmp_path) -> None:
    from puripuly_heart.config.settings_vnext import serialization
    from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext

    settings_path = tmp_path / "settings.json"
    sentinel = "private-overlay-anchor-sentinel"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["overlay"]["calibration"]["anchor"] = sentinel
    settings_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match="migration_failed:ValueError") as exc_info:
        main_module._load_settings_or_default(settings_path)

    assert sentinel not in str(exc_info.value)


def test_settings_config_path_marks_default_as_implicit(monkeypatch, tmp_path) -> None:
    default_path = tmp_path / "vnext" / "settings.json"
    monkeypatch.setattr(main_module, "default_settings_path", lambda: default_path)
    args = type("Args", (), {})()

    path, explicit = main_module._settings_config_path(args)

    assert path == default_path
    assert explicit is False


def test_settings_config_path_marks_custom_config_as_explicit(tmp_path) -> None:
    custom_path = tmp_path / "custom.json"
    args = type("Args", (), {"config": custom_path})()

    path, explicit = main_module._settings_config_path(args)

    assert path == custom_path
    assert explicit is True


def test_main_explicit_config_routes_logging_to_selected_file_parent(
    monkeypatch,
    tmp_path,
) -> None:
    log_dirs: list[Path | None] = []

    class FakeLoggingSinks:
        def close(self, *, force: bool = False) -> None:
            assert force is True

    monkeypatch.setattr(
        main_module,
        "configure_main_logging",
        lambda *, log_dir=None: log_dirs.append(log_dir) or FakeLoggingSinks(),
    )
    monkeypatch.setattr(main_module, "_run_gui", lambda *_args, **_kwargs: 0)
    config_path = tmp_path / "selected" / "settings.json"

    assert main_module.main(["--config", str(config_path)]) == 0
    assert log_dirs == [config_path.parent]


def test_production_cli_does_not_advertise_or_accept_process_capture_smoke(capsys) -> None:
    help_text = main_module.build_parser().format_help()

    assert "process-capture-runtime-check" not in help_text
    with pytest.raises(SystemExit):
        main_module.main(["process-capture-runtime-check"])
    assert "process-capture-runtime-check" not in capsys.readouterr().out
