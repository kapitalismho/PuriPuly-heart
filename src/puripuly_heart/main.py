from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
    from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks

logger = logging.getLogger(__name__)


def configure_main_logging(*, log_dir: Path | None = None):
    from puripuly_heart.core.runtime_logging import configure_main_logging as configure

    return configure(log_dir=log_dir)


def default_settings_path() -> Path:
    from puripuly_heart.config.paths import default_settings_path as resolve

    return resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="puripuly-heart")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to settings JSON (default: user config dir)",
    )
    parser.add_argument(
        "--debug-ui-preview",
        action="store_true",
        default=False,
        help="Show developer-only GUI preview controls for hidden UI states",
    )

    sub = parser.add_subparsers(dest="command")

    desktop_overlay = sub.add_parser(
        "run-desktop-overlay",
        help="Run the desktop Flet overlay renderer",
    )
    desktop_overlay.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to overlay launch manifest JSON",
    )
    sub.add_parser(
        "run-desktop-overlay-preview",
        help="Run the desktop Flet overlay preview",
    )
    desktop_repro = sub.add_parser(
        "run-desktop-overlay-repro",
        help="Run Desktop overlay diagnostic repro",
    )
    desktop_repro.add_argument("--cycles", type=int, default=100)
    desktop_repro.add_argument("--dwell-ms", type=int, default=150)
    desktop_repro.add_argument("--output-dir", type=Path, required=True)
    verify_desktop_repro = sub.add_parser(
        "verify-desktop-overlay-repro",
        help="Verify Desktop overlay repro artifacts",
    )
    verify_desktop_repro.add_argument("--output-dir", type=Path, required=True)

    sub.add_parser(
        "local-qwen-runtime-check",
        help="Verify the Local Qwen Windows runtime DLL directory",
    )
    sub.add_parser(
        "soxr-runtime-check",
        help="Verify the packaged soxr runtime contract and smoke resample",
    )
    sub.add_parser(
        "hf-xet-runtime-check",
        help="Verify the packaged Hugging Face/Xet runtime",
    )
    sub.add_parser("gui-startup-check", help=argparse.SUPPRESS)
    local_cpu_real_model_check = sub.add_parser(
        "local-cpu-real-model-check",
        help="Run strict real-model checks for all direct local CPU ASR backends",
    )
    local_cpu_real_model_check.add_argument("--model-root", type=Path, required=True)
    local_cpu_real_model_check.add_argument("--audio-root", type=Path, required=True)
    local_cpu_real_model_check.add_argument("--report", type=Path, required=True)
    hf_xet_worker = sub.add_parser("hf-xet-download-worker", help=argparse.SUPPRESS)
    hf_xet_worker.add_argument("--request-file", type=Path, required=True)
    hf_xet_worker.add_argument("--event-file", type=Path, required=True)
    local_asr_production_evidence = sub.add_parser(
        "local-asr-production-composition-evidence",
        help=argparse.SUPPRESS,
    )
    local_asr_production_evidence.add_argument("--audio", type=Path, required=True)
    local_asr_production_evidence.add_argument("--report", type=Path, required=True)
    local_asr_production_evidence.add_argument("--candidate", required=True)
    local_asr_production_evidence.add_argument("--expected-gpu-name", required=True)
    run_gui = sub.add_parser("run-gui", help="Run the Graphical User Interface (Flet)")
    run_gui.add_argument(
        "--debug-ui-preview",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show developer-only GUI preview controls for hidden UI states",
    )

    return parser


def run_local_qwen_runtime_check() -> int:
    from puripuly_heart.app.local_qwen_runtime_check import run_local_qwen_runtime_check as run

    return run()


def run_soxr_runtime_check() -> int:
    from puripuly_heart.app.soxr_runtime_check import run_soxr_runtime_check as run

    return run()


def run_local_asr_production_composition_evidence(
    *,
    audio_path: Path,
    report_path: Path,
    candidate: str,
    expected_gpu_name: str,
) -> int:
    from puripuly_heart.release_evidence.local_asr_production_composition import (
        run_local_asr_production_composition,
    )

    return run_local_asr_production_composition(
        audio_path=audio_path,
        report_path=report_path,
        candidate=candidate,
        expected_gpu_name=expected_gpu_name,
    )


def run_hf_xet_runtime_check() -> int:
    import hf_xet
    import huggingface_hub

    if huggingface_hub.__version__ != "1.26.0":
        raise RuntimeError("unexpected packaged huggingface_hub version")
    if not Path(hf_xet.__file__).with_name("hf_xet.pyd").is_file():
        raise RuntimeError("packaged hf_xet native extension is missing")
    return 0


def run_local_cpu_real_model_check(
    *,
    model_root: Path,
    audio_root: Path,
    report_path: Path,
) -> int:
    from puripuly_heart.release_evidence.local_cpu_real_decode import main as run

    return run(
        [
            "--model-root",
            str(model_root),
            "--audio-root",
            str(audio_root),
            "--report",
            str(report_path),
        ]
    )


def _load_gui_runtime() -> tuple[Any, Any, Any, Any, Any]:
    import flet as ft

    from puripuly_heart.app.adapters.vrchat_osc_presence import PsutilVrchatOscPresenceAdapter
    from puripuly_heart.composition.ui_application import compose_ui_application
    from puripuly_heart.ui.app import main_gui
    from puripuly_heart.ui.fonts import assets_dir

    return ft, PsutilVrchatOscPresenceAdapter, compose_ui_application, main_gui, assets_dir


def run_gui_startup_check() -> int:
    _, vrchat_osc_presence_adapter, _, _, assets_dir = _load_gui_runtime()
    vrchat_osc_presence_adapter()
    assets_dir()
    return 0


def _run_gui(
    config_path: Path,
    *,
    debug_ui_preview: bool,
    runtime_logging_sinks: RuntimeLoggingSinks,
) -> int:
    ft, vrchat_osc_presence_adapter, compose_ui_application, main_gui, assets_dir = (
        _load_gui_runtime()
    )

    vrchat_osc_presence = vrchat_osc_presence_adapter()

    async def _target(page: ft.Page):
        try:
            return await main_gui(
                page,
                config_path=config_path,
                application_factory=compose_ui_application,
                debug_ui_preview=debug_ui_preview,
                runtime_logging_sinks=runtime_logging_sinks,
                vrchat_osc_presence=vrchat_osc_presence,
            )
        except Exception as exc:
            logger.exception(
                "GUI startup failed: exception_type=%s exception_message=%s",
                type(exc).__name__,
                str(exc),
            )
            raise

    try:
        ft.run(
            main=_target,
            assets_dir=str(assets_dir()),
            view=ft.AppView.FLET_APP_HIDDEN,
        )
    except Exception as exc:
        logger.exception(
            "Flet GUI runtime failed: exception_type=%s exception_message=%s",
            type(exc).__name__,
            str(exc),
        )
        raise
    return 0


def _run_desktop_overlay(config_path: Path) -> int:
    from puripuly_heart.ui.desktop_overlay import main as desktop_overlay_main

    return desktop_overlay_main(["--config", str(config_path)])


def _run_desktop_overlay_preview() -> int:
    from puripuly_heart.ui.desktop_overlay import main as desktop_overlay_main

    return desktop_overlay_main(["--preview"])


def _run_desktop_overlay_repro(*, cycles: int, dwell_ms: int, output_dir: Path) -> int:
    from puripuly_heart.ui.desktop_overlay_repro import run_desktop_overlay_repro

    return run_desktop_overlay_repro(cycles=cycles, dwell_ms=dwell_ms, output_dir=output_dir)


def _verify_desktop_overlay_repro(*, output_dir: Path) -> int:
    from puripuly_heart.core.desktop_overlay_repro_artifacts import verify_desktop_overlay_repro

    return verify_desktop_overlay_repro(output_dir=output_dir)


def _load_settings_or_default(
    path: Path,
) -> AppSettingsVNext:
    from dataclasses import replace

    from puripuly_heart.config.settings import detect_system_locale, resolve_first_run_ui_locale
    from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
    from puripuly_heart.config.settings_vnext.schema import (
        DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS,
        AppSettingsVNext,
        TranslationFallbackIntent,
    )

    if path.exists():
        result = load_vnext_settings(path)
        if result.settings is None:
            raise RuntimeError(
                result.error.message if result.error is not None else str(result.status)
            )
        return result.settings

    settings = AppSettingsVNext()
    system_locale = detect_system_locale()
    locale_value = resolve_first_run_ui_locale(system_locale)
    translation = replace(
        settings.intent.translation,
        fallback=TranslationFallbackIntent(
            selection_alias=DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
        ),
    )
    if locale_value == "zh-CN":
        translation = replace(
            translation,
            model="deepseek_v4_flash",
            connection="managed_china",
            openrouter_model="deepseek/deepseek-v4-flash-0731",
            openrouter_selection_alias="deepseek_v4_flash_managed",
            openrouter_provider_routing="deepseek_only",
        )
    settings = replace(settings, intent=replace(settings.intent, translation=translation))
    if locale_value:
        settings = replace(
            settings,
            intent=replace(settings.intent, ui=replace(settings.intent.ui, locale=locale_value)),
        )

    if not settings.intent.prompts.system_prompt:
        from puripuly_heart.config.prompts import load_prompt_for_provider
        from puripuly_heart.config.settings import LLMProviderName

        default_prompt = load_prompt_for_provider(LLMProviderName.GEMINI.value)
        settings = replace(
            settings,
            intent=replace(
                settings.intent,
                prompts=replace(settings.intent.prompts, system_prompt=default_prompt),
            ),
        )
    return settings


def _settings_config_path(args: argparse.Namespace) -> tuple[Path, bool]:
    if hasattr(args, "config"):
        return args.config, True
    return default_settings_path(), False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "hf-xet-download-worker":
        from puripuly_heart.core.local_stt_huggingface_xet_adapter import (
            run_huggingface_xet_worker,
        )

        return run_huggingface_xet_worker(
            request_path=args.request_file,
            event_path=args.event_file,
        )
    if args.command == "run-desktop-overlay-repro":
        return _run_desktop_overlay_repro(
            cycles=args.cycles,
            dwell_ms=args.dwell_ms,
            output_dir=args.output_dir,
        )
    if args.command == "verify-desktop-overlay-repro":
        return _verify_desktop_overlay_repro(output_dir=args.output_dir)

    settings_config_path, explicit_settings_config = _settings_config_path(args)
    debug_ui_preview = bool(getattr(args, "debug_ui_preview", False))
    logging_sinks = configure_main_logging(
        log_dir=(
            settings_config_path.parent
            if explicit_settings_config and args.command in {None, "run-gui"}
            else None
        )
    )
    try:
        if args.command != "run-desktop-overlay":
            args.config = settings_config_path

        if args.version:
            from puripuly_heart import __version__

            print(__version__)
            return 0

        if args.command == "run-desktop-overlay":
            return _run_desktop_overlay(args.config)

        if args.command == "run-desktop-overlay-preview":
            return _run_desktop_overlay_preview()

        if args.command == "run-gui":
            return _run_gui(
                args.config,
                debug_ui_preview=debug_ui_preview,
                runtime_logging_sinks=logging_sinks,
            )

        if args.command == "local-qwen-runtime-check":
            return run_local_qwen_runtime_check()

        if args.command == "soxr-runtime-check":
            return run_soxr_runtime_check()

        if args.command == "hf-xet-runtime-check":
            return run_hf_xet_runtime_check()

        if args.command == "gui-startup-check":
            return run_gui_startup_check()

        if args.command == "local-cpu-real-model-check":
            return run_local_cpu_real_model_check(
                model_root=args.model_root,
                audio_root=args.audio_root,
                report_path=args.report,
            )

        if args.command == "local-asr-production-composition-evidence":
            return run_local_asr_production_composition_evidence(
                audio_path=args.audio,
                report_path=args.report,
                candidate=args.candidate,
                expected_gpu_name=args.expected_gpu_name,
            )

        # Default: run GUI when no command specified (e.g., double-clicking EXE)
        if args.command is None:
            return _run_gui(
                args.config,
                debug_ui_preview=debug_ui_preview,
                runtime_logging_sinks=logging_sinks,
            )

        parser.print_help()
        return 2
    finally:
        logging_sinks.close(force=True)


if __name__ == "__main__":
    raise SystemExit(main())
