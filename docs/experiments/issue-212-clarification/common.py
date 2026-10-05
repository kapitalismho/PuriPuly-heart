from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
DIRECTORY = Path(__file__).resolve().parent
_SECRETS: list[str] = []


def settings_data() -> dict:
    from puripuly_heart.config.paths import default_settings_path

    path = default_settings_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return data.get("intent", data)


def secret(name: str, *environment_names: str, legacy: tuple[str, ...] = ()) -> str | None:
    from puripuly_heart.config.paths import default_settings_path
    from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore, KeyringSecretStore

    config = settings_data().get("secrets", {})
    if config.get("backend", "keyring") == "encrypted_file":
        path = Path(config.get("encrypted_file_path", "secrets.json"))
        if not path.is_absolute():
            path = default_settings_path().parent / path
        passphrase = os.environ.get("PURIPULY_HEART_SECRETS_PASSPHRASE")
        store = EncryptedFileSecretStore(path, passphrase=passphrase) if path.is_file() and passphrase else None
    else:
        store = KeyringSecretStore()
    value = None
    if store is not None:
        for key in (name, *legacy):
            value = store.get(key)
            if value:
                break
    if not value:
        value = next((os.environ[key] for key in environment_names if os.environ.get(key)), None)
    if value and value not in _SECRETS:
        _SECRETS.append(value)
    return value


def safe_error(error: BaseException) -> str:
    message = str(error)
    for value in _SECRETS:
        message = message.replace(value, "[redacted]")
    return f"{type(error).__name__}: {message[:1200]}"


def synthesize(name: str, text: str, *, voice: str = "Microsoft Zira Desktop", rate: int = 0) -> Path:
    directory = DIRECTORY / "audio"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.wav"
    metadata_path = path.with_suffix(".json")
    expected = {"text": text, "voice": voice, "rate": rate, "sample_rate": 16000}
    if path.is_file() and metadata_path.is_file():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing synthetic audio parameters differ")
        return path
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { "
        f"$s.SelectVoice({quote(voice)}); $s.Rate = {int(rate)}; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile({quote(path)}, $f); $s.Speak({quote(text)}); "
        "} finally { $s.Dispose() }"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], check=True, capture_output=True, timeout=30)
    with wave.open(str(path), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, 16000):
            raise ValueError("Unexpected synthetic audio format")
    metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, 16000):
            raise ValueError("Probe requires mono PCM16 16 kHz")
        return reader.readframes(reader.getnframes())
