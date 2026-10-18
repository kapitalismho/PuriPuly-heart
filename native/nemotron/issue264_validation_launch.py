from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    for prefix, label in ((str(ROOT), "<CHECKOUT>"), (str(HOME), "<USER_HOME>")):
        for variant in (prefix, prefix.replace("\\", "/")):
            text = text.replace(variant.replace("\\", "\\\\"), label).replace(variant, label)
    path.write_text(text + "\n", encoding="utf-8")


def source_hashes():
    result = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        with path.open("rb") as stream:
            result[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


def main():
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument(
        "--probe", choices=("runtime", "caption", "caption_replay"), default="runtime"
    )
    selected, remaining_arguments = gate.parse_known_args()
    if not re.fullmatch("[0-9a-f]{40}", selected.candidate):
        raise ValueError("Director stable full candidate SHA is required")
    output = selected.output.resolve()
    if output.parent != ROOT / "build" or not output.name.startswith("issue264-validation-"):
        raise ValueError("Only assigned isolated build outputs are permitted")
    if output.exists():
        raise FileExistsError("Preserve previous attempts; choose a fresh output")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    if head != selected.candidate:
        raise RuntimeError("Checkout does not match Director-pinned candidate")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", selected.candidate, "--", "src"], cwd=ROOT
    )
    if unchanged.returncode != 0:
        raise RuntimeError(
            "Product sources differ from stable candidate; do not run moving product"
        )
    output.mkdir(parents=True)
    temp = output / "owned-temp"
    temp.mkdir()
    for name in tuple(os.environ):
        if re.search(
            r"(?:API_KEY|TOKEN|SECRET|CREDENTIAL|AUTHORIZATION)", name, re.I
        ) or name.startswith(("GOOGLE_", "GEMINI_")):
            os.environ.pop(name, None)
    for name in ("TEMP", "TMP", "TMPDIR"):
        os.environ[name] = str(temp)
    flet_client = HOME / ".flet/client/flet-desktop-full-1.0.0-758f21506fbb/flet/flet.exe"
    if not flet_client.is_file():
        raise FileNotFoundError("Prepared installed Flet client is unavailable")
    os.environ["FLET_VIEW_PATH"] = str(flet_client.parent)
    os.environ["FLET_SERVER_IP"] = "127.0.0.1"
    for name, suffix in (("APPDATA", "roaming"), ("LOCALAPPDATA", "local")):
        isolated = output / suffix
        isolated.mkdir()
        os.environ[name] = str(isolated)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "GGML_VK_DISABLE_GRAPH_OPTIMIZE": "1",
            "NEMO_SPEECH_SCHEDULE_CAPTURE": "1",
            "ISSUE264_VALIDATION_READY_SHA": selected.candidate,
        }
    )
    import psutil

    own = psutil.Process()
    before = source_hashes()
    observations = {
        "candidate": selected.candidate,
        "checkout_head": head,
        "started_monotonic": time.monotonic(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pid": os.getpid(),
        "product_sources_before": before,
        "connections": [],
        "subprocesses": [],
        "policy_rejections": [],
        "isolation": "Secrets scrubbed before product import; task-local temporary files; external socket/DNS rejected; direct scoped composition only; no full application startup",
        "flet_client": str(flet_client),
        "flet_client_sha256": hashlib.sha256(flet_client.read_bytes()).hexdigest(),
        "filesystem_isolation_limit": "Python env/temp/appdata are isolated and installed FLET_VIEW_PATH bypasses shared cache unpack/touch/GC. This is not an OS sandbox: native child known-folder APIs and injected third-party layers are not redirected or certified absent.",
    }

    def audit(event, args):
        if event in ("socket.connect", "socket.bind"):
            address = args[1]
            if isinstance(address, tuple):
                host = address[0]
                caller = sys._getframe(1)
                flet_port_reservation = (
                    event == "socket.bind"
                    and address == ("", 0)
                    and caller.f_code.co_name == "get_free_tcp_port"
                    and Path(caller.f_code.co_filename).resolve()
                    == ROOT / ".venv/Lib/site-packages/flet/utils/network.py"
                )
                if host not in ("127.0.0.1", "::1", "localhost") and not flet_port_reservation:
                    observations["policy_rejections"].append({"event": event, "host": host})
                    raise PermissionError("Validation allows loopback sockets only")
                observations["connections"].append(
                    {"event": event, "host": host, "port": address[1], "at": time.monotonic()}
                )
        elif event == "socket.getaddrinfo":
            if args[0] not in ("127.0.0.1", "::1", "localhost", None):
                observations["policy_rejections"].append({"event": event, "host": args[0]})
                raise PermissionError("Validation prohibits external name resolution")
        elif event == "subprocess.Popen":
            observations["subprocesses"].append(
                {"executable": args[0], "cwd": args[2], "at": time.monotonic()}
            )
        elif event == "open" and isinstance(args[0], (str, bytes)):
            filename = Path(os.fsdecode(args[0])).name.lower()
            if filename in (".env", ".env.local", "secrets.json", "credentials.json"):
                observations["policy_rejections"].append(
                    {"event": "credential_file_open", "filename": filename}
                )
                raise PermissionError("Validation prohibits credential-file access")

    sys.addaudithook(audit)
    exit_code = 1
    try:
        sys.path.insert(0, str(ROOT / "native/nemotron"))
        module_name = f"issue264_validation_{selected.probe}"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / f"native/nemotron/{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        sys.argv = [
            str(spec.origin),
            "--candidate",
            selected.candidate,
            "--output",
            str(output),
            *remaining_arguments,
        ]
        args = module.arguments()
        exit_code = asyncio.run(module.run(args))
        observations["runtime_exit_code"] = exit_code
    except BaseException as error:
        observations["error"] = f"{type(error).__name__}: {error}"
        observations["traceback"] = traceback.format_exc()
    finally:
        remaining = own.children(recursive=True)
        observations["owned_children_after_runtime_cleanup"] = [
            {"pid": child.pid, "name": child.name()} for child in remaining if child.is_running()
        ]
        for child in remaining:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(remaining, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=5)
        observations["owned_children_unreaped"] = [child.pid for child in alive]
        observations["product_sources_after"] = source_hashes()
        observations["product_sources_unchanged"] = observations["product_sources_after"] == before
        observations["actual_product_module_paths"] = {
            name: getattr(module, "__file__", None)
            for name, module in sorted(sys.modules.items())
            if name.startswith("puripuly_heart")
        }
        observations["finished_monotonic"] = time.monotonic()
        try:
            shutil.rmtree(temp)
            observations["owned_temporary_files_removed"] = True
        except OSError as error:
            observations["owned_temporary_files_removed"] = False
            observations["temporary_cleanup_error"] = str(error)
        save(output / "isolation.json", observations)
        if (
            alive
            or observations["policy_rejections"]
            or not observations["product_sources_unchanged"]
        ):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
