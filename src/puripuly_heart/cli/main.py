from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from puripuly_heart.cli.transport import ControlTransportError, follow, request

TERMINAL = frozenset(
    {
        "applied",
        "degraded",
        "rejected",
        "persistence_failed",
        "failed",
        "action_required",
        "cancelled",
        "interrupted",
    }
)
EXIT_CODES = {
    "applied": 0,
    "degraded": 4,
    "rejected": 4,
    "persistence_failed": 4,
    "failed": 4,
    "action_required": 5,
    "cancelled": 6,
    "interrupted": 6,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puripuly", description="Local application control (JSON output protocol v1)"
    )
    parser.add_argument(
        "--config", type=Path, help="Settings file identity (default: user settings)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Client wait timeout in seconds; timeout never retries mutations",
    )
    sub = parser.add_subparsers(dest="domain", required=True)
    app = sub.add_parser("app", help="List, start, discover, stop, and restart hosts")
    app_sub = app.add_subparsers(dest="action", required=True)
    for name in ("start", "restart"):
        p = app_sub.add_parser(name)
        mode = p.add_mutually_exclusive_group()
        mode.add_argument(
            "--background", action="store_true", help="Run independently of the launching terminal"
        )
        mode.add_argument("--foreground", action="store_true", help="Run until stopped or Ctrl+C")
    for name in ("list", "status", "discover"):
        app_sub.add_parser(name)
    app_sub.add_parser("stop").add_argument(
        "--no-wait",
        action="store_true",
        help="Return accepted shutdown receipt without observing host termination",
    )
    sub.add_parser(
        "capabilities", help="Discover available queries, commands, arguments, and choices"
    )
    query = sub.add_parser("query", help="Run a named, side-effect-free application query")
    query.add_argument("name", help="Name from capabilities, e.g. settings.current")
    _json_arguments(query)
    command = sub.add_parser("command", help="Submit a named application command")
    command.add_argument("name", help="Name from capabilities, e.g. provider.apply")
    _json_arguments(command)
    command.add_argument("--expected-revision", type=int)
    command.add_argument(
        "--no-wait", action="store_true", help="Return accepted operation ID immediately"
    )
    command.add_argument(
        "--request-id", help="Stable caller id for deduplication after uncertain delivery"
    )
    operation = sub.add_parser("operation", help="Inspect, wait for, or cancel an operation")
    op_sub = operation.add_subparsers(dest="action", required=True)
    for name in ("status", "wait", "cancel"):
        op_sub.add_parser(name).add_argument("operation_id")
    events = sub.add_parser("events", help="Follow independent bounded application events")
    event_sub = events.add_subparsers(dest="action", required=True)
    follow_parser = event_sub.add_parser("follow")
    follow_parser.add_argument(
        "--topic", action="append", default=[], help="Subscribed topic, repeatable"
    )
    follow_parser.add_argument("--channel", choices=("self", "peer"))
    follow_parser.add_argument(
        "--include-transcripts", action="store_true", help="Explicitly request recognition content"
    )
    follow_parser.add_argument(
        "--include-translations", action="store_true", help="Explicitly request translated content"
    )
    follow_parser.add_argument(
        "--after", type=int, help="Resume after a received nonnegative sequence number"
    )
    follow_parser.add_argument("--instance-id", help="Originating host identity for replay cursor")
    logs = sub.add_parser("logs", help="Follow bounded, content-redacted application logs")
    logs_sub = logs.add_subparsers(dest="action", required=True)
    log_follow = logs_sub.add_parser("follow")
    log_follow.add_argument(
        "--level", choices=("debug", "info", "warning", "error"), default="warning"
    )
    log_follow.add_argument("--after", type=int, help="Replay after an event sequence")
    log_follow.add_argument("--instance-id", help="Originating host identity for replay cursor")
    secret = sub.add_parser("secrets", help="Set a secret without exposing it in process arguments")
    secret_sub = secret.add_subparsers(dest="action", required=True)
    secret_set = secret_sub.add_parser("set")
    secret_set.add_argument("name", help="Secret identifier from capabilities")
    secret_set.add_argument(
        "--stdin",
        action="store_true",
        help="Read secret from standard input instead of a hidden prompt",
    )
    secret_set.add_argument("--no-wait", action="store_true")
    secret_sub.add_parser("delete").add_argument("name")
    secret_sub.add_parser("presence")
    secret_verify = secret_sub.add_parser("verify")
    secret_verify.add_argument("name")
    secret_verify.add_argument(
        "--stdin",
        action="store_true",
        help="Read credential to verify from standard input instead of a hidden prompt",
    )
    capture = sub.add_parser("capture", help="Control self or peer capture")
    cap_sub = capture.add_subparsers(dest="action", required=True)
    cap_set = cap_sub.add_parser("set")
    cap_set.add_argument("channel", choices=("self", "peer"))
    cap_set.add_argument("state", choices=("on", "off"))
    cap_set.add_argument("--no-wait", action="store_true")
    cap_set.add_argument(
        "--accept-peer-terms",
        action="store_true",
        help="Explicitly accept the terms shown by capture terms and enable peer capture",
    )
    cap_sub.add_parser("terms", help="Read peer-translation terms and current consent")
    translation_set = (
        sub.add_parser("translation", help="Control translation")
        .add_subparsers(dest="action", required=True)
        .add_parser("set")
    )
    translation_set.add_argument("state", choices=("on", "off"))
    translation_set.add_argument("--no-wait", action="store_true")
    asr = sub.add_parser("asr", help="Inspect and change selected ASR providers")
    asr_sub = asr.add_subparsers(dest="action", required=True)
    asr_sub.add_parser("status")
    asr_set = asr_sub.add_parser("set")
    asr_set.add_argument("--channel", choices=("self", "peer", "both"), required=True)
    asr_set.add_argument("--provider", required=True)
    asr_set.add_argument("--no-wait", action="store_true")
    cap_sub.add_parser("status")
    audio = sub.add_parser(
        "audio", help="Discover audio devices, set the loopback target, and test audio"
    )
    audio_sub = audio.add_subparsers(dest="action", required=True)
    audio_sub.add_parser("devices").add_subparsers(dest="verb", required=True).add_parser("list")
    audio_sub.add_parser("processes").add_subparsers(dest="verb", required=True).add_parser("list")
    audio_target = audio_sub.add_parser("target").add_subparsers(dest="verb", required=True)
    audio_target.add_parser("status")
    audio_target_set = audio_target.add_parser("set")
    audio_target_set.add_argument("value")
    audio_target_set.add_argument("--no-wait", action="store_true")
    audio_sub.add_parser("retry").add_argument("--no-wait", action="store_true")
    audio_test = audio_sub.add_parser("test")
    audio_test.add_argument("state", choices=("on", "off"))
    audio_test.add_argument("--no-wait", action="store_true")
    settings = sub.add_parser("settings", help="Inspect choices and apply typed settings changes")
    settings_sub = settings.add_subparsers(dest="action", required=True)
    for name in ("current", "choices"):
        settings_sub.add_parser(name)
    apply = settings_sub.add_parser("apply")
    _json_arguments(apply)
    apply.add_argument("--expected-revision", type=int)
    apply.add_argument("--no-wait", action="store_true")
    text = sub.add_parser("text", help="Submit manual text through the normal pipeline")
    text_sub = text.add_subparsers(dest="action", required=True)
    submit = text_sub.add_parser("submit")
    submit.add_argument("--stdin", action="store_true", help="Read content from standard input")
    submit.add_argument("--file", type=Path, help="Read UTF-8 content from a file")
    submit.add_argument("--no-wait", action="store_true")
    models = sub.add_parser(
        "models",
        help="Inspect, download, import, delete, install, retry or cancel local models",
    )
    models_sub = models.add_subparsers(dest="action", required=True)
    models_sub.add_parser("status")
    for action in ("install", "prepare", "cancel", "retry"):
        model_action = models_sub.add_parser(action)
        model_action.add_argument(
            "--backend",
            choices=("gpu",) if action == "retry" else ("cpu", "gpu"),
            default="cpu" if action == "prepare" else "gpu",
        )
        model_action.add_argument("--no-wait", action="store_true")
        if action == "install":
            model_action.add_argument("--model-id", action="append", default=[])
        if action == "cancel":
            model_action.add_argument("--model-id")
        if action == "prepare":
            model_action.add_argument("--verify-checksums", action="store_true")
    for action in ("download", "import", "delete"):
        asset_action = models_sub.add_parser(action)
        asset_action.add_argument("--model-id", required=True)
        asset_action.add_argument("--no-wait", action="store_true")
        if action == "download":
            asset_action.add_argument(
                "--source",
                choices=("auto", "huggingface", "modelscope", "hf-mirror"),
                default="auto",
            )
        if action == "import":
            asset_action.add_argument("--dir", type=Path, required=True)
    gemma = sub.add_parser("gemma", help="Prepare or cancel managed Gemma assets")
    gemma_sub = gemma.add_subparsers(dest="action", required=True)
    for action in ("prepare", "cancel"):
        gemma_sub.add_parser(action).add_argument("--no-wait", action="store_true")
    gpu = sub.add_parser("gpu", help="Inspect and discover GPU devices")
    gpu_sub = gpu.add_subparsers(dest="action", required=True)
    gpu_sub.add_parser("status")
    gpu_sub.add_parser("discover").add_argument("--no-wait", action="store_true")
    overlay = sub.add_parser("overlay", help="Inspect and control overlay output")
    overlay_sub = overlay.add_subparsers(dest="action", required=True)
    overlay_sub.add_parser("status")
    for action in ("set", "lock"):
        overlay_set = overlay_sub.add_parser(action)
        overlay_set.add_argument("state", choices=("on", "off"))
        overlay_set.add_argument("--no-wait", action="store_true")
    overlay_size = overlay_sub.add_parser("size")
    overlay_size.add_argument("preset")
    overlay_size.add_argument("--no-wait", action="store_true")
    overlay_sub.add_parser("reset-position").add_argument("--no-wait", action="store_true")
    calibrate = overlay_sub.add_parser("calibrate")
    calibrate.add_argument("step", choices=("begin", "change", "apply", "cancel"))
    calibrate.add_argument(
        "--field",
        choices=("anchor", "offset_x", "offset_y", "distance", "text_scale", "background_alpha"),
    )
    calibrate.add_argument("--value", help="Anchor name or finite numeric calibration value")
    calibrate.add_argument("--no-wait", action="store_true")
    osc = sub.add_parser("osc", help="Inspect effective OSC connection")
    osc.add_subparsers(dest="action", required=True).add_parser("status")
    auth = sub.add_parser("auth", help="Inspect, authorize and remove account login")
    auth_sub = auth.add_subparsers(dest="action", required=True)
    auth_sub.add_parser("status")
    auth_login = auth_sub.add_parser("login")
    auth_login.add_argument("provider", choices=("qq", "discord", "openrouter", "chatgpt"))
    auth_login.add_argument(
        "--stdin", action="store_true", help="Read protected JSON fields from stdin"
    )
    auth_login.add_argument(
        "--open-browser", action="store_true", help="Explicitly open the authorization URL"
    )
    auth_login.add_argument("--no-wait", action="store_true")
    auth_logout = auth_sub.add_parser("logout")
    auth_logout.add_argument("provider", choices=("qq", "discord", "openrouter", "chatgpt"))
    auth_logout.add_argument("--no-wait", action="store_true")
    return parser


def _json_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--arguments", help="Typed JSON object (never use for secrets or conversation content)"
    )
    group.add_argument("--file", type=Path, help="UTF-8 file containing typed JSON object")


def _parse_arguments(args: argparse.Namespace) -> dict[str, Any]:
    inline = getattr(args, "arguments", None)
    text = args.file.read_text(encoding="utf-8") if getattr(args, "file", None) else inline
    if text is None:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Arguments must be a JSON object")
    if inline is not None and _contains_protected_content(parsed):
        raise ValueError(
            "Secrets and conversation content must not appear in process arguments; use hidden input, stdin, or a protected file"
        )
    return parsed


def _contains_protected_content(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            any(
                term in key.casefold()
                for term in (
                    "secret",
                    "password",
                    "api_key",
                    "credential",
                    "token",
                    "transcript",
                    "translation_text",
                    "text",
                    "prompt",
                    "vocabulary",
                    "referral",
                    "identity",
                )
            )
            or _contains_protected_content(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_protected_content(item) for item in value)
    return False


def _auth_input(args: argparse.Namespace) -> dict[str, Any]:
    if args.stdin:
        fields = json.loads(sys.stdin.read())
        if not isinstance(fields, dict) or fields.keys() - {
            "qq_identity",
            "credential",
            "referral_id",
        }:
            raise ValueError(
                "Authorization stdin must be a JSON object of supported protected fields"
            )
    else:
        fields = {}
        if args.provider == "qq":
            if not sys.stdin.isatty():
                raise ValueError("QQ login requires --stdin or a hidden interactive prompt")
            fields = {
                "qq_identity": getpass.getpass("QQ identity: "),
                "credential": getpass.getpass("QQ credential: "),
            }
    if args.provider == "qq":
        if not all(
            isinstance(fields.get(key), str) and fields[key]
            for key in ("qq_identity", "credential")
        ):
            raise ValueError("QQ login requires nonempty qq_identity and credential")
    elif fields.keys() - {"referral_id"}:
        raise ValueError("Only QQ login accepts an identity or credential")
    if "referral_id" in fields and (
        not isinstance(fields["referral_id"], str) or not fields["referral_id"]
    ):
        raise ValueError("Referral ID must be a nonempty string")
    return fields


def _instance(path: Path) -> Any:
    from puripuly_heart.core.control_instance import discover

    record = discover(path)
    if record is None:
        raise ControlTransportError(
            "instance_not_found", "No application host for these settings; start one explicitly"
        )
    return record


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps({"output_version": 1, **value}, ensure_ascii=False, allow_nan=False))


async def _submit(record: Any, name: str, values: dict[str, Any], args: argparse.Namespace) -> int:
    request_id = getattr(args, "request_id", None) or secrets.token_hex(16)
    try:
        receipt = await request(
            record,
            "submit",
            {
                "name": name,
                "values": values,
                "request_id": request_id,
                "expected_revision": getattr(args, "expected_revision", None),
            },
            timeout=args.timeout,
        )
    except ControlTransportError as exc:
        if exc.code in ("unknown_execution", "instance_unavailable", "connection_lost"):
            _emit(
                {
                    "status": "unknown_execution",
                    "instance_id": record.instance_id,
                    "request_id": request_id,
                    "message": "Submission may have been accepted; do not replay without checking host state",
                }
            )
            return 7
        raise
    if type(receipt.get("terminal")) is not bool:
        raise ControlTransportError(
            "malformed_response", "Operation receipt omitted terminal state"
        )
    if getattr(args, "no_wait", False) or receipt["terminal"]:
        _emit(receipt)
        return _status_code(receipt, waiting=not getattr(args, "no_wait", False))
    operation_id = receipt.get("operation_id")
    if not isinstance(operation_id, str):
        raise ControlTransportError(
            "malformed_response", "Accepted operation omitted its operation ID"
        )
    try:
        if name == "auth.login" and values["provider"] in ("discord", "openrouter", "chatgpt"):
            deadline = time.monotonic() + args.timeout
            while True:
                remaining = deadline - time.monotonic()
                result = await request(
                    record,
                    "wait",
                    {"operation_id": operation_id, "timeout": max(0, min(0.25, remaining))},
                    timeout=max(0.1, min(0.25, remaining) + 2),
                )
                if result.get("terminal") is True or remaining <= 0:
                    break
                auth = await request(
                    record,
                    "query",
                    {"name": "auth.status", "values": {}},
                    timeout=max(0.1, deadline - time.monotonic()),
                )
                challenge = auth.get("challenge")
                if isinstance(challenge, dict) and challenge.get("operation_id") == operation_id:
                    _emit(
                        {
                            "status": "action_required",
                            "operation_status": result["status"],
                            "terminal": False,
                            "instance_id": record.instance_id,
                            "operation_id": operation_id,
                            "authorization_url": challenge["authorization_url"],
                            "message": "Authorize externally, then inspect or wait for operation completion",
                        }
                    )
                    return EXIT_CODES["action_required"]
        else:
            result = await request(
                record,
                "wait",
                {"operation_id": operation_id, "timeout": args.timeout},
                timeout=args.timeout + 2,
            )
    except ControlTransportError as exc:
        if exc.code in ("unknown_execution", "instance_unavailable", "connection_lost"):
            _emit(
                {
                    "status": "unknown_execution",
                    "instance_id": record.instance_id,
                    "operation_id": operation_id,
                    "message": "Operation may still be running; inspect its ID before deciding what to do",
                }
            )
            return 7
        raise
    if type(result.get("terminal")) is not bool:
        raise ControlTransportError("malformed_response", "Operation result omitted terminal state")
    if not result["terminal"]:
        _emit(
            {
                "status": "unknown_execution",
                "instance_id": record.instance_id,
                "operation_id": operation_id,
                "receipt": result,
                "message": "Operation remains in progress; inspect its ID",
            }
        )
        return 7
    _emit(result)
    return _status_code(result, waiting=True)


def _status_code(result: dict[str, Any], *, waiting: bool) -> int:
    status = result.get("status")
    terminal = result.get("terminal")
    if type(terminal) is not bool:
        raise ControlTransportError("malformed_response", "Operation result omitted terminal state")
    if not terminal:
        if status == "action_required":
            return EXIT_CODES[status]
        return 0 if not waiting and status in ("accepted", "running") else 7
    if status in TERMINAL:
        return EXIT_CODES[status]
    raise ControlTransportError("malformed_response", "Terminal operation has invalid status")


async def _start_background(path: Path, timeout: float) -> int:
    from puripuly_heart.core.control_instance import discover
    from puripuly_heart.runtime_layout import current_runtime_layout

    if discover(path) is not None:
        raise ControlTransportError(
            "instance_already_running", "Application host already owns these settings"
        )
    layout = current_runtime_layout()
    fd, filename = tempfile.mkstemp(prefix="puripuly-ready-", suffix=".json")
    os.close(fd)
    ready = Path(filename)
    ready.unlink()
    service_args = ["run-headless", "--config", str(path), "--ready-file", str(ready)]
    if layout.host_kind == "pyinstaller":
        argv = [str(layout.host_executable), *service_args]
    else:
        module = "product_bootstrap" if layout.host_kind == "native" else "puripuly_heart.main"
        argv = [str(layout.python_executable), "-m", module, *service_args]
    options: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if layout.host_kind == "native":
        options["env"] = layout.python_child_environment()
    if sys.platform == "win32":
        options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    try:
        process = subprocess.Popen(argv, **options)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ready.exists():
                status = json.loads(ready.read_text(encoding="utf-8"))
                if status.get("state") != "ready":
                    raise ControlTransportError(
                        "startup_failed",
                        f"Application failed to start ({status.get('reason', 'unknown')})",
                    )
                record = discover(path)
                if record is None or record.instance_id != status.get("instance_id"):
                    raise ControlTransportError(
                        "startup_failed", "Host readiness did not match secure discovery"
                    )
                _emit({"status": "ready", "instance_id": record.instance_id, "pid": process.pid})
                return 0
            if process.poll() is not None:
                raise ControlTransportError(
                    "startup_failed",
                    f"Application exited during startup (exit {process.returncode})",
                )
            await asyncio.sleep(0.05)
        raise ControlTransportError(
            "startup_unknown",
            "Startup did not complete before timeout; check app status before retrying",
        )
    finally:
        ready.unlink(missing_ok=True)


async def _stop(record: Any, args: argparse.Namespace) -> int:
    import psutil

    status = await request(
        record, "query", {"name": "app.status", "values": {}}, timeout=args.timeout
    )
    host_pid = status.get("host_pid")
    if type(host_pid) is not int or host_pid <= 0:
        raise ControlTransportError(
            "malformed_response", "Host status omitted its process identity"
        )
    try:
        host_created = psutil.Process(host_pid).create_time()
    except psutil.NoSuchProcess:
        raise ControlTransportError(
            "instance_unavailable", "Control host exited before shutdown could be requested"
        ) from None
    request_id = getattr(args, "request_id", None) or secrets.token_hex(16)
    try:
        result = await request(
            record,
            "submit",
            {"name": "app.stop", "values": {}, "request_id": request_id},
            timeout=args.timeout,
        )
    except ControlTransportError as exc:
        if exc.code in ("unknown_execution", "instance_unavailable", "connection_lost"):
            _emit(
                {
                    "status": "unknown_execution",
                    "instance_id": record.instance_id,
                    "request_id": request_id,
                    "message": "Shutdown request may have been accepted; inspect host status before retrying",
                }
            )
            return 7
        raise
    if type(result.get("terminal")) is not bool:
        raise ControlTransportError("malformed_response", "Shutdown receipt omitted terminal state")
    if result.get("status") not in ("accepted", "running", "applied"):
        _emit(result)
        return _status_code(result, waiting=True)
    if getattr(args, "no_wait", False):
        _emit(result)
        return _status_code(result, waiting=False)
    from puripuly_heart.core.control_instance import discover

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        next_record = discover(args.config)
        try:
            process = psutil.Process(host_pid)
            running = process.create_time() == host_created and process.is_running()
        except psutil.NoSuchProcess:
            running = False
        if not running and (next_record is None or next_record.instance_id != record.instance_id):
            _emit(
                {
                    "status": "stopped",
                    "terminal": True,
                    "instance_id": record.instance_id,
                    "operation_id": result["operation_id"],
                    "cleanup_outcome": "unobserved",
                }
            )
            return 0
        await asyncio.sleep(0.05)
    _emit(
        {
            "status": "unknown_execution",
            "instance_id": record.instance_id,
            "operation_id": result["operation_id"],
            "message": "Shutdown requested, but host process termination was not observed",
        }
    )
    return 7


async def _run(args: argparse.Namespace) -> int:
    if args.domain == "app":
        if args.action == "list":
            from puripuly_heart.core.control_instance import discover_instances

            _emit({"instances": discover_instances()})
            return 0
        if args.action == "start":
            if args.background:
                return await _start_background(args.config, args.timeout)
            from puripuly_heart.cli.host import run_headless

            await run_headless(args.config)
            return 0
        if args.action == "restart":
            stopped = await _stop(_instance(args.config), args)
            if stopped != 0:
                return stopped
            if args.background:
                return await _start_background(args.config, args.timeout)
            from puripuly_heart.cli.host import run_headless

            await run_headless(args.config)
            return 0
        if args.action == "stop":
            return await _stop(_instance(args.config), args)
        record = _instance(args.config)
        result = await request(
            record, "query", {"name": "app.status", "values": {}}, timeout=args.timeout
        )
        _emit({"instance_id": record.instance_id, "port": record.port, **result})
        return 0
    if args.domain == "command" and args.name in (
        "secrets.set",
        "secrets.verify",
        "auth.login",
        "text.submit",
    ):
        raise ValueError("Protected input requires the dedicated secrets, auth, or text command")
    if (
        args.domain == "capture"
        and args.action == "set"
        and args.accept_peer_terms
        and (args.channel != "peer" or args.state != "on")
    ):
        raise ValueError("--accept-peer-terms requires capture set peer on")
    if args.domain == "overlay" and args.action == "calibrate":
        if args.step == "change":
            if args.field is None or args.value is None:
                raise ValueError("Overlay calibration change requires --field and --value")
            if args.field != "anchor":
                args.value = float(args.value)
                if not math.isfinite(args.value):
                    raise ValueError("Overlay calibration requires a finite numeric value")
        elif args.field is not None or args.value is not None:
            raise ValueError("Calibration fields apply only to the change step")
    record = _instance(args.config)
    if args.domain == "capabilities":
        _emit(await request(record, "capabilities", timeout=args.timeout))
        return 0
    if args.domain == "query":
        _emit(
            await request(
                record,
                "query",
                {"name": args.name, "values": _parse_arguments(args)},
                timeout=args.timeout,
            )
        )
        return 0
    query_name = {
        ("capture", "status"): "capture.channels",
        ("asr", "status"): "providers.status",
        ("settings", "current"): "settings.current",
        ("settings", "choices"): "settings.choices",
        ("secrets", "presence"): "secrets.presence",
        ("auth", "status"): "auth.status",
        ("models", "status"): "models.status",
        ("gpu", "status"): "gpu.status",
        ("overlay", "status"): "overlay.status",
        ("osc", "status"): "osc.status",
        ("capture", "terms"): "consent.peer_translation",
        ("audio", "devices"): "audio.devices",
        ("audio", "processes"): "audio.processes",
        ("audio", "target"): "audio.target",
    }.get((args.domain, getattr(args, "action", None)))
    if query_name and not (
        args.domain == "audio" and args.action == "target" and args.verb == "set"
    ):
        _emit(
            await request(record, "query", {"name": query_name, "values": {}}, timeout=args.timeout)
        )
        return 0
    if args.domain in (
        "command",
        "settings",
        "capture",
        "translation",
        "asr",
        "secrets",
        "text",
        "audio",
        "models",
        "gemma",
        "gpu",
        "overlay",
        "auth",
    ):
        if args.domain == "command":
            name, values = args.name, _parse_arguments(args)
            if name == "app.stop":
                if values:
                    raise ValueError("app.stop takes no arguments")
                if args.expected_revision is not None:
                    raise ValueError("app.stop does not use a settings revision")
                return await _stop(record, args)
        elif args.domain == "settings":
            name, values = "settings.apply", _parse_arguments(args)
        elif args.domain == "capture":
            name, values = "capture.set", {"channel": args.channel, "enabled": args.state == "on"}
            if args.accept_peer_terms:
                values["accept_terms"] = True
        elif args.domain == "translation":
            name, values = "translation.set", {"enabled": args.state == "on"}
        elif args.domain == "asr":
            name, values = "provider.apply", {"channel": args.channel, "provider": args.provider}
        elif args.domain == "secrets":
            name, values = f"secrets.{args.action}", {"name": args.name}
            if args.action in ("set", "verify"):
                if not args.stdin and not sys.stdin.isatty():
                    raise ValueError(
                        "Use --stdin for noninteractive credential input; refusing an unhidden prompt"
                    )
                values["value"] = (
                    sys.stdin.readline().rstrip("\r\n")
                    if args.stdin
                    else getpass.getpass("Credential: ")
                )
        elif args.domain == "text":
            name = "text.submit"
            if bool(args.stdin) == bool(args.file):
                raise ValueError("Specify exactly one of --stdin or --file for manual text")
            values = {
                "text": sys.stdin.read() if args.stdin else args.file.read_text(encoding="utf-8")
            }
        elif args.domain == "audio":
            if args.action == "target":
                name, values = "audio.target.set", {"value": args.value}
            elif args.action == "test":
                name, values = "audio.test", {"enabled": args.state == "on"}
            else:
                name, values = "audio.retry", {}
        elif args.domain == "models" and args.action in ("download", "import", "delete"):
            name, values = f"models.{args.action}", {"model_id": args.model_id}
            if args.action == "download":
                values["source"] = args.source
            if args.action == "import":
                values["directory"] = str(args.dir.resolve())
        elif args.domain == "models":
            name, values = f"models.{args.action}", {"backend": args.backend}
            if args.action == "install" and args.model_id:
                values["model_ids"] = args.model_id
            if args.action == "cancel" and args.model_id:
                values["model_id"] = args.model_id
            if args.action == "prepare":
                values["verify_checksums"] = args.verify_checksums
        elif args.domain == "gemma":
            name, values = f"gemma.{args.action}", {}
        elif args.domain == "gpu":
            name, values = "gpu.discover", {}
        elif args.domain == "overlay":
            name = {
                "reset-position": "overlay.position.reset",
                "calibrate": "overlay.calibrate",
            }.get(args.action, f"overlay.{args.action}")
            if args.action in ("set", "lock"):
                values = {"enabled" if args.action == "set" else "locked": args.state == "on"}
            elif args.action == "size":
                values = {"preset": args.preset}
            elif args.action == "calibrate":
                values = {"action": args.step}
                if args.step == "change":
                    values.update(field=args.field, value=args.value)
            else:
                values = {}
        else:
            name, values = f"auth.{args.action}", {"provider": args.provider}
            if args.action == "login":
                values.update(_auth_input(args))
                values["open_browser"] = args.open_browser
        return await _submit(record, name, values, args)
    if args.domain == "operation":
        method = {"status": "operation", "wait": "wait", "cancel": "cancel"}[args.action]
        values = {"operation_id": args.operation_id}
        if method == "wait":
            values["timeout"] = args.timeout
        result = await request(record, method, values, timeout=args.timeout + 2)
        _emit(result)
        if method == "cancel" and result.get("cancellation") == "unsupported":
            return 4
        return _status_code(result, waiting=method == "wait")
    if args.domain in ("events", "logs"):
        values = (
            {
                "topics": args.topic
                or ["operation", "settings", "session_state_changed", "error", "gap"],
                "channel": args.channel,
                "include_transcripts": args.include_transcripts,
                "include_translations": args.include_translations,
                "after": args.after,
                "cursor_instance_id": args.instance_id,
            }
            if args.domain == "events"
            else {"topics": ["logs"], "after": args.after, "cursor_instance_id": args.instance_id}
        )
        levels = {"debug": 10, "info": 20, "warning": 30, "error": 40}
        async for event in follow(record, values):
            if (
                args.domain == "logs"
                and event.get("topic") == "logs"
                and levels.get(str(event.get("level", "")).lower(), 0) < levels[args.level]
            ):
                continue
            _emit(event)
        return 0
    raise ValueError("Unknown control action")


def _normalize_options(argv: list[str]) -> list[str]:
    global_options: list[str] = []
    remainder: list[str] = []
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in ("--json", "--wait"):
            index += 1
            continue
        if option in ("--config", "--timeout"):
            if index + 1 >= len(argv):
                remainder.append(option)
                index += 1
                continue
            global_options.extend((option, argv[index + 1]))
            index += 2
            continue
        remainder.append(option)
        index += 1
    return global_options + remainder


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_options(sys.argv[1:] if argv is None else argv))
    if args.config is None:
        from puripuly_heart.config.paths import default_settings_path

        args.config = default_settings_path()
    if not 0 < args.timeout <= 3600:
        parser.error("--timeout must be between 0 and 3600 seconds")
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except (ControlTransportError, ValueError, OSError, json.JSONDecodeError) as exc:
        code = exc.code if isinstance(exc, ControlTransportError) else "invalid_input"
        message = (
            str(exc)
            if isinstance(exc, ControlTransportError)
            else (
                "Local input or control resource could not be accessed"
                if isinstance(exc, OSError)
                else "Invalid local input or arguments"
            )
        )
        print(
            json.dumps(
                {"output_version": 1, "error": {"code": code, "message": message}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 7 if code in ("unknown_execution", "startup_unknown") else 3


if __name__ == "__main__":
    raise SystemExit(main())
