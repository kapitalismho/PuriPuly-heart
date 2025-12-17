from __future__ import annotations

import asyncio
import argparse
from pathlib import Path

from ajebal_daera_translator.app.headless_stdin import HeadlessStdinRunner
from ajebal_daera_translator.config.settings import AppSettings, load_settings
from ajebal_daera_translator.core.osc.udp_sender import VrchatOscUdpSender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ajebal-daera-translator")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("settings.json"),
        help="Path to settings JSON (default: ./settings.json)",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
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
        if args.use_llm:
            print("Error: LLM wiring for headless runner is implemented in later phases.", flush=True)
            return 2
        runner = HeadlessStdinRunner(settings=settings, llm=None)
        return asyncio.run(runner.run())

    parser.print_help()
    return 2


def _load_settings_or_default(path: Path) -> AppSettings:
    if path.exists():
        return load_settings(path)
    return AppSettings()


if __name__ == "__main__":
    raise SystemExit(main())
