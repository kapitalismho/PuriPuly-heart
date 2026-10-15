from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

PINNED_COLLECTOR_COMMIT = "c3da5bb5f97ede9b2cfb31ce1d0f1f590e7e2ec6"
PINNED_COLLECTOR_ORIGINAL_SHA256 = (
    "b0a91d610bbcd5905e68470c538d15106f40991c2539026e44f759a7a08aa775"
)
PINNED_COLLECTOR_SHA256 = "d8d9782c7743be4ea5d121078317f38a725275f33d7f9c72e4e6fe5c553d9534"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def isolated_environment(root: Path, source: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        upper = name.upper()
        if any(
            part in upper
            for part in (
                "API_KEY",
                "AUTH_TOKEN",
                "ACCESS_TOKEN",
                "CREDENTIAL",
                "PASSWORD",
                "SECRET",
                "LLAMA",
                "GGML",
                "PURIPULY",
                "OPENROUTER",
                "OPENAI",
                "ANTHROPIC",
                "SONIOX",
                "DEEPGRAM",
                "HF_TOKEN",
                "HUGGING_FACE",
            )
        ) or upper.startswith(("PYTHON", "FLET_")):
            environment.pop(name)
    roots = {
        "LOCALAPPDATA": root / "Local",
        "APPDATA": root / "Roaming",
        "USERPROFILE": root / "Home",
        "HOME": root / "Home",
        "TEMP": root / "Temp",
        "TMP": root / "Temp",
        "XDG_CONFIG_HOME": root / "Config",
        "XDG_CACHE_HOME": root / "Cache",
        "HF_HOME": root / "HuggingFace",
        "TORCH_HOME": root / "Torch",
    }
    for name, path in roots.items():
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path.resolve())
    environment.update(
        PYTHONPATH=str((source / "src").resolve()),
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        PURIPULY_HEART_SECRETS_PASSPHRASE=secrets.token_urlsafe(48),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    return environment


def execute(args: argparse.Namespace, root: Path, action: str, *extra: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    environment = isolated_environment(root, args.source)
    command = [
        str(args.python),
        str(Path(__file__).with_name("session.py").resolve()),
        action,
        "--root",
        str(root.resolve()),
        "--source",
        str(args.source.resolve()),
        "--assets",
        str(args.assets.resolve()),
        *extra,
    ]
    with (
        (root / "host.stdout.txt").open("wb") as stdout,
        (root / "host.stderr.txt").open("wb") as stderr,
    ):
        try:
            result = subprocess.run(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout,
                cwd=args.source,
            )
        except subprocess.TimeoutExpired:
            cleanup_command = [
                str(args.python),
                "-m",
                "puripuly_heart.main",
                "cli",
                "--config",
                str((root / "settings.json").resolve()),
                "--timeout",
                "30",
                "app",
                "stop",
            ]
            with (
                (root / "timeout-stop.stdout.txt").open("wb") as stop_stdout,
                (root / "timeout-stop.stderr.txt").open("wb") as stop_stderr,
            ):
                try:
                    stopped = subprocess.run(
                        cleanup_command,
                        env=environment,
                        stdout=stop_stdout,
                        stderr=stop_stderr,
                        timeout=45,
                        cwd=args.source,
                    )
                    stop_status = stopped.returncode
                except subprocess.TimeoutExpired:
                    stop_status = "timeout"
            (root / "supervisor.json").write_text(
                json.dumps(
                    {
                        "status": "interrupted",
                        "reason": "owned_session_timeout",
                        "timeout_s": args.timeout,
                        "supported_owned_instance_stop_exit": stop_status,
                        "cleanup_observation": "inspect_owned_process_and_port_evidence",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 124
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "supported", "run", "summarize"))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--e4b-source", type=Path)
    parser.add_argument("--model26-source", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument(
        "--case",
        choices=("abba", "26b", "cpu", "fallback", "safety", "26b-failure", "smoke", "smoke26b"),
        default="abba",
    )
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--timeout", type=float, default=5400)
    args = parser.parse_args()
    if digest(Path(__file__).with_name("windows_metrics.py")) != PINNED_COLLECTOR_SHA256:
        raise RuntimeError("pinned Windows collector has changed")
    root = args.root or Path(tempfile.mkdtemp(prefix="puripuly-managed-251-"))
    if args.action == "prepare":
        if not all((args.e4b_source, args.model26_source, args.runtime_root)):
            parser.error("prepare requires --e4b-source, --model26-source and --runtime-root")
        code = execute(
            args,
            root,
            "prepare",
            "--e4b-source",
            str(args.e4b_source.resolve()),
            "--model26-source",
            str(args.model26_source.resolve()),
            "--runtime-root",
            str(args.runtime_root.resolve()),
        )
        print(
            json.dumps(
                {
                    "exit_code": code,
                    "local_artifact_root": str(root.resolve()),
                    "asset_manifest": str((args.assets / "assets.json").resolve()),
                }
            )
        )
        return code
    if args.action == "summarize":
        return execute(args, root, "summarize")
    if args.action == "preflight":
        return execute(args, root, "lifecycle", "--case", "preflight")
    if not args.execute_approved:
        parser.error(
            "model/host execution requires Director integrated-ready and --execute-approved"
        )
    if args.action == "supported":
        return execute(args, root, "supported")
    arms = {
        "abba": (
            ("A1", "managed_gemma", "gpu", "mmap"),
            ("B1", "managed_gemma", "gpu", "none"),
            ("B2", "managed_gemma", "gpu", "none"),
            ("A2", "managed_gemma", "gpu", "mmap"),
        ),
        "26b": (("26b", "managed_gemma_26b", "gpu", "product"),),
        "cpu": (("cpu", "managed_gemma", "cpu", "product"),),
        "fallback": (("fallback", "managed_gemma", "gpu", "product"),),
        "safety": (("safety", "managed_gemma", "gpu", "product"),),
        "26b-failure": (("26b-failure", "managed_gemma_26b", "gpu", "product"),),
        "smoke": (("smoke", "managed_gemma", "gpu", "product"),),
        "smoke26b": (("smoke26b", "managed_gemma_26b", "gpu", "product"),),
    }[args.case]
    outcomes = []
    for label, model, backend, mode in arms:
        start = time.time()
        code = execute(
            args,
            root / label,
            "lifecycle",
            "--model",
            model,
            "--backend",
            backend,
            "--mode",
            mode,
            "--case",
            args.case,
        )
        outcomes.append(
            {
                "arm": label,
                "model": model,
                "backend": backend,
                "mode": mode,
                "exit_code": code,
                "start_unix_s": start,
                "end_unix_s": time.time(),
            }
        )
        (root / "outcomes.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        print(json.dumps(outcomes[-1]), flush=True)
        if code:
            return code
    print(json.dumps({"local_artifact_root": str(root.resolve()), "outcomes": outcomes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
