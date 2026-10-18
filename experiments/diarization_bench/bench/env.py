"""Isolated host environment for benchmark runs. Import before any puripuly_heart module."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parents[1]
REAL_LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
REAL_MODELS = REAL_LOCALAPPDATA / "puripuly-heart" / "models"
LINKED_MODELS = (
    "qwen3-asr-1.7b-q6-k-transcribe-vulkan",
    "parakeet-tdt-0.6b-v3-int8-sherpa",
    "parakeet-tdt-ctc-0.6b-ja-int8-sherpa",
    "qwen3-asr-0.6b-int8-sherpa",
    "smart-turn-v3.2-cpu.onnx",
    "gemma-4-e4b-it-qat-ud-q4-k-xl",
)
PROVIDER_ENV_VARS = (
    "DEEPGRAM_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_API_KEY2",
    "GEMINI_API_KEY",
    "SONIOX_API_KEY",
    "ElevenLabs_API_KEY",
    "ELEVENLABS_API_KEY",
    "ALIBABA_API_KEY",
    "ALIBABA_API_KEY_BEIJING",
    "ALIBABA_API_KEY_SINGAPORE",
    "ALIBABA_API_KEY_TOKYO",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
    "CEREBRAS_API_KEY",
)


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def provider_credentials() -> dict[str, str]:
    candidates = [REPO_ROOT / ".env.local"]
    candidates += [parent / ".env.local" for parent in REPO_ROOT.parents]
    merged: dict[str, str] = {}
    for path in reversed(candidates):
        merged.update(read_env_file(path))
    return {k: v for k, v in merged.items() if k in PROVIDER_ENV_VARS and v}


def _link_directory(link: Path, target: Path) -> None:
    if link.exists():
        return
    if target.is_dir():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        os.link(target, link)


def prepare(root: Path) -> dict[str, object]:
    root = root.resolve()
    local = root / "localappdata"
    models = local / "puripuly-heart" / "models"
    models.mkdir(parents=True, exist_ok=True)
    linked = []
    for name in LINKED_MODELS:
        target = REAL_MODELS / name
        if target.exists():
            _link_directory(models / name, target)
            linked.append(name)
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    passphrase_file = root / "passphrase.txt"
    if not passphrase_file.exists():
        passphrase_file.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    for name in PROVIDER_ENV_VARS:
        os.environ.pop(name, None)
    os.environ["LOCALAPPDATA"] = str(local)
    os.environ["PURIPULY_HEART_SECRETS_PASSPHRASE"] = passphrase_file.read_text(encoding="utf-8")
    os.environ["DIARIZATION_BENCH_CONTROL_DIR"] = str(control)
    for path in (str(BENCH_ROOT), str(REPO_ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return {"root": str(root), "control": str(control), "linked_models": linked}


def write_isolated_settings(path: Path) -> None:
    if path.exists():
        return
    from dataclasses import replace

    from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
    from puripuly_heart.config.settings_vnext.serialization import to_dict

    settings = AppSettingsVNext()
    intent = settings.intent
    settings = replace(
        settings,
        intent=replace(
            intent,
            secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
            osc=replace(intent.osc, connection_mode="off"),
            telemetry=replace(intent.telemetry, enabled=False),
            overlay=replace(intent.overlay, target="desktop"),
        ),
        state=replace(
            settings.state,
            peer_translation=replace(settings.state.peer_translation, eula_accepted=True),
        ),
    )
    path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
