from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from ajebal_daera_translator.app.headless_mic import HeadlessMicRunner
from ajebal_daera_translator.app.headless_stdin import HeadlessStdinRunner
from ajebal_daera_translator.app.wiring import create_llm_provider, create_secret_store
from ajebal_daera_translator.config.paths import default_settings_path, default_vad_model_path
from ajebal_daera_translator.config.settings import AppSettings, load_settings
from ajebal_daera_translator.core.osc.udp_sender import VrchatOscUdpSender

# Configure logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="puripuly-heart")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    parser.add_argument(
        "--config",
        type=Path,
        default=default_settings_path(),
        help="Path to settings JSON (default: user config dir)",
    )

    sub = parser.add_subparsers(dest="command")

    osc_send = sub.add_parser("osc-send", help="Send a single VRChat chatbox OSC message")
    osc_send.add_argument("text", help="Text to send")

    stdin = sub.add_parser("run-stdin", help="Read lines from stdin and send to OSC")
    stdin.add_argument(
        "--use-llm",
        action="store_true",
        help="Translate each line using configured LLM provider (requires provider setup)",
    )

    mic = sub.add_parser("run-mic", help="Capture microphone audio (VAD→STT→LLM→OSC)")
    mic.add_argument(
        "--vad-model",
        type=Path,
        default=default_vad_model_path(),
        help="Path to Silero VAD ONNX model file (default: user config dir)",
    )
    mic.add_argument(
        "--use-llm",
        action="store_true",
        help="Translate STT final results using configured LLM provider",
    )

    sub.add_parser("run-gui", help="Run the Graphical User Interface (Flet)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return 0

    if args.command == "run-gui":
        import flet as ft

        from ajebal_daera_translator.ui.app import main_gui

        config_path = args.config

        async def _target(page: ft.Page):
            return await main_gui(page, config_path=config_path)

        ft.run(main=_target)
        return 0

    settings = _load_settings_or_default(args.config)

    if args.command == "osc-send":
        sender = VrchatOscUdpSender(
            host=settings.osc.host,
            port=settings.osc.port,
            chatbox_address=settings.osc.chatbox_address,
            chatbox_send=settings.osc.chatbox_send,
            chatbox_clear=settings.osc.chatbox_clear,
        )
        try:
            sender.send_chatbox(args.text)
        finally:
            sender.close()
        return 0

    if args.command == "run-stdin":
        llm = None
        if args.use_llm:
            try:
                secrets = create_secret_store(settings.secrets, config_path=args.config)
                llm = create_llm_provider(settings, secrets=secrets)
            except Exception as exc:
                print(f"Error: failed to initialize LLM provider: {exc}", flush=True)
                return 2

        runner = HeadlessStdinRunner(settings=settings, llm=llm)
        return asyncio.run(runner.run())

    if args.command == "run-mic":
        runner = HeadlessMicRunner(
            settings=settings,
            config_path=args.config,
            vad_model_path=args.vad_model,
            use_llm=args.use_llm,
        )
        return asyncio.run(runner.run())

    # Default: run GUI when no command specified (e.g., double-clicking EXE)
    if args.command is None:
        import flet as ft

        from ajebal_daera_translator.ui.app import main_gui

        config_path = args.config

        async def _target(page: ft.Page):
            return await main_gui(page, config_path=config_path)

        ft.run(main=_target)
        return 0

    parser.print_help()
    return 2


def _load_settings_or_default(path: Path) -> AppSettings:
    if path.exists():
        return load_settings(path)
    return AppSettings()


if __name__ == "__main__":
    raise SystemExit(main())
