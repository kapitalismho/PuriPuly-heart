"""Live CLI-control provider handoffs with offline speech through VB-Audio Cable.

Run with --report in a private directory outside the checkout. This tool uses the
actual CLI, captured devices, cloud providers, and the existing credential store;
it never supplies secrets or creates peer consent.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from puripuly_heart.config.paths import default_settings_path
from puripuly_heart.runtime_layout import current_runtime_layout

SCHEMA = "puripuly-heart/cli-control-live-speech/v1"
PHRASE = "The purple lantern is glowing beside the quiet river."
WORDS = {"purple", "lantern"}
PROVIDERS = ("soniox", "qwen_audio")


class EvidenceFailure(Exception):
    pass


def _wav(path: Path) -> dict:
    literal_path = str(path).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { $v = $s.GetInstalledVoices() | Where-Object { "
        "$_.VoiceInfo.Culture.Name -eq 'en-US' } | Select-Object -First 1; "
        "if ($null -eq $v) { throw 'No installed en-US offline SAPI voice' }; "
        "$s.SelectVoice($v.VoiceInfo.Name); "
        f"$s.SetOutputToWaveFile('{literal_path}'); "
        f"$s.Speak('{PHRASE}'); "
        "} finally { $s.Dispose() }"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise EvidenceFailure("offline_sapi_fixture_unavailable")
    with wave.open(str(path), "rb") as recording:
        channels, width = recording.getnchannels(), recording.getsampwidth()
        rate, count = recording.getframerate(), recording.getnframes()
        frames = recording.readframes(count)
    if channels not in (1, 2) or width != 2 or rate <= 0 or count / rate < 1:
        raise EvidenceFailure("offline_sapi_fixture_invalid_pcm")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "phrase": PHRASE,
        "sample_rate_hz": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "duration_seconds": count / rate,
        "samples": np.frombuffer(frames, dtype="<i2").reshape((-1, channels)).astype(np.float32)
        / 32768,
    }


def _cable_devices() -> tuple[int, str, str]:
    apis = sd.query_hostapis()
    all_devices = sd.query_devices()
    outputs = [
        (index, device)
        for index, device in enumerate(all_devices)
        if apis[device["hostapi"]]["name"] == "Windows DirectSound"
        and device["max_output_channels"] > 0
        and "CABLE Input(VB-Audio Virtual Cable)" in device["name"]
    ]
    inputs = [
        device
        for device in all_devices
        if apis[device["hostapi"]]["name"] == "Windows WASAPI"
        and device["max_input_channels"] > 0
        and "CABLE Output(VB-Audio Virtual Cable)" in device["name"]
    ]
    if len(outputs) != 1 or len(inputs) != 1:
        raise EvidenceFailure("vb_audio_cable_directsound_render_wasapi_capture_unavailable")
    import pyaudiowpatch as pyaudio

    manager = pyaudio.PyAudio()
    try:
        loopbacks = [
            info
            for info in manager.get_loopback_device_info_generator()
            if "CABLE Input(VB-Audio Virtual Cable)" in info["name"]
        ]
    finally:
        manager.terminate()
    if len(loopbacks) != 1:
        raise EvidenceFailure("vb_audio_cable_loopback_unavailable")
    return outputs[0][0], inputs[0]["name"], outputs[0][1]["name"]


async def _cli(config: Path, *arguments: str, timeout: float = 90) -> dict:
    command = [
        sys.executable,
        "-m",
        "puripuly_heart.main",
        "cli",
        "--config",
        str(config),
        "--timeout",
        str(int(timeout)),
        *arguments,
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout + 10)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise EvidenceFailure("cli_execution_timed_out_unknown_execution") from exc
    try:
        result = json.loads(stdout)
    except ValueError as exc:
        raise EvidenceFailure(f"cli_invalid_json_exit_{process.returncode}") from exc
    if process.returncode != 0 or not isinstance(result, dict):
        raise EvidenceFailure(
            f"cli_{'.'.join(arguments[:2])}_exit_{process.returncode}_status_{result.get('status', 'unknown')}"
        )
    return result


async def _command(
    config: Path, name: str, values: dict, workdir: Path, *, timeout: float = 90
) -> dict:
    args_file = workdir / "control-arguments.json"
    args_file.write_text(json.dumps(values), encoding="utf-8")
    try:
        result = await _cli(config, "command", name, "--file", str(args_file), timeout=timeout)
    finally:
        args_file.unlink(missing_ok=True)
    if result.get("status") != "applied":
        raise EvidenceFailure(f"command_{name}_status_{result.get('status', 'unknown')}")
    return result


async def _settings(config: Path, changes: dict, workdir: Path) -> None:
    await _command(config, "settings.apply", {"changes": changes}, workdir)


async def _events(config: Path, channel: str, matches: list[dict]) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "puripuly_heart.main",
        "cli",
        "--config",
        str(config),
        "events",
        "follow",
        "--topic",
        "transcript",
        "--channel",
        channel,
        "--include-transcripts",
        stdin=subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("topic") != "transcript" or item.get("channel") != channel:
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            tokens = set(re.findall(r"[a-z]+", text.casefold()))
            if WORDS <= tokens:
                matches.append(
                    {
                        "sequence": item.get("sequence"),
                        "utterance_id": item.get("utterance_id"),
                        "source": item.get("source"),
                        "matched_fixed_phrase_tokens": True,
                    }
                )
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()


async def _recognize(config: Path, channels: tuple[str, ...], device: int, wav: dict) -> dict:
    matches = {channel: [] for channel in channels}
    tasks = [
        asyncio.create_task(_events(config, channel, matches[channel])) for channel in channels
    ]
    try:
        await asyncio.sleep(1.5)
        if any(task.done() for task in tasks):
            raise EvidenceFailure("transcript_subscriber_disconnected_before_playback")
        for _ in range(3):
            await asyncio.to_thread(
                sd.play,
                wav["playback_samples"],
                wav["playback_rate_hz"],
                device=device,
                blocking=True,
            )
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and not all(matches.values()):
                await asyncio.sleep(0.25)
            if all(matches.values()):
                break
            await asyncio.sleep(1)
        if not all(matches.values()):
            raise EvidenceFailure(
                "fixed_speech_recognition_missing_"
                + "_".join(channel for channel in channels if not matches[channel])
            )
        return {channel: matches[channel][0] for channel in channels}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _verify_channels(snapshot: dict, target: str, active: tuple[str, ...]) -> dict:
    channels = snapshot.get("channels")
    if not isinstance(channels, dict):
        raise EvidenceFailure("provider_snapshot_missing_channels")
    result = {}
    for channel in ("self", "peer"):
        state = channels.get(channel)
        if not isinstance(state, dict):
            raise EvidenceFailure("provider_snapshot_missing_" + channel)
        result[channel] = {
            key: state.get(key)
            for key in (
                "selected_provider",
                "runtime_provider",
                "capture_attached_provider",
                "desired_active",
                "effective_active",
                "pending_handoff",
                "failure_reason",
            )
        }
        if channel in active and (
            result[channel]["selected_provider"] != target
            or result[channel]["runtime_provider"] != target
            or result[channel]["capture_attached_provider"] != target
            or result[channel]["desired_active"] is not True
            or result[channel]["effective_active"] is not True
            or result[channel]["pending_handoff"] is not False
        ):
            raise EvidenceFailure(f"provider_snapshot_not_attached_{channel}_{target}")
    return result


async def run(report_path: Path) -> int:
    canonical = default_settings_path()
    if not canonical.is_file():
        raise EvidenceFailure("canonical_profile_missing")
    profile = json.loads(canonical.read_text(encoding="utf-8"))
    if profile.get("state", {}).get("peer_translation", {}).get("eula_accepted") is not True:
        raise EvidenceFailure("peer_consent_human_action_required")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = report_path.with_suffix(".wav")
    audio = _wav(wav_path)
    device, microphone, render = _cable_devices()
    import soxr

    playback_rate = int(sd.query_devices(device)["default_samplerate"])
    if playback_rate != audio["sample_rate_hz"]:
        audio["playback_samples"] = soxr.resample(
            audio["samples"], audio["sample_rate_hz"], playback_rate
        )
    else:
        audio["playback_samples"] = audio["samples"]
    audio["playback_rate_hz"] = playback_rate
    report = {
        "schema": SCHEMA,
        "status": "failed",
        "started_unix_seconds": time.time(),
        "application": {
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "host_kind": current_runtime_layout().host_kind,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "fixture": {
            key: value for key, value in audio.items() if key not in ("samples", "playback_samples")
        },
        "route": {
            "render": render,
            "self_microphone": microphone,
            "peer_loopback_render": render,
            "capture_host_api": "Windows WASAPI",
            "render_host_api": "Windows DirectSound",
        },
        "scenarios": [],
    }
    with tempfile.TemporaryDirectory(
        prefix="puripuly-cli-control-", dir=report_path.parent
    ) as directory:
        workdir = Path(directory)
        config = workdir / "settings.json"
        shutil.copyfile(canonical, config)
        instance_id = None
        try:
            started = await _cli(config, "app", "start", "--background", timeout=90)
            instance_id = started.get("instance_id")
            if not isinstance(instance_id, str):
                raise EvidenceFailure("headless_instance_identity_missing")
            report["instance_id"] = instance_id
            if (await _cli(config, "query", "auth.status")).get("consent_accepted") is not True:
                raise EvidenceFailure("peer_consent_human_action_required")
            choices = await _cli(config, "settings", "choices")
            if not set(PROVIDERS) <= set(choices.get("stt_providers", [])):
                raise EvidenceFailure("cloud_provider_choices_unavailable")
            devices = await _cli(config, "audio", "devices", "list")
            microphones = devices.get("microphones", [])
            outputs = devices.get("loopback_outputs", [])
            if not any(
                item.get("name") == microphone and item.get("host_api") == "Windows WASAPI"
                for item in microphones
            ):
                raise EvidenceFailure("app_cable_microphone_unavailable")
            if not any(render in item for item in outputs):
                raise EvidenceFailure("app_cable_loopback_unavailable")
            osc = profile["intent"]["osc"]
            await _settings(
                config,
                {
                    "osc.connection": {
                        "mode": "off",
                        "send_port": osc["send_port"],
                        "receive_port": osc["receive_port"],
                    }
                },
                workdir,
            )
            await _settings(
                config,
                {
                    "audio.input_host_api": "Windows WASAPI",
                    "audio.input_device": microphone,
                    "audio.output_device": render,
                    "languages": {
                        "source": "en",
                        "peer_source": "en",
                        "peer_source_mode": "manual",
                    },
                    "clipboard.enabled": False,
                },
                workdir,
            )
            targets = await _cli(config, "audio", "target", "status")
            selected = [
                item["value"]
                for item in targets.get("options", [])
                if isinstance(item, dict) and item.get("value") == f"device:{render} [Loopback]"
            ]
            if len(selected) != 1:
                raise EvidenceFailure("app_cable_peer_capture_target_unavailable")
            await _command(config, "audio.target.set", {"value": selected[0]}, workdir)
            target = await _cli(config, "audio", "target", "status")
            if target.get("selected") != selected[0]:
                raise EvidenceFailure("app_cable_peer_capture_target_not_selected")
            report["route"]["peer_capture_target"] = target["selected"]
            await _command(config, "translation.set", {"enabled": False}, workdir)
            await _command(config, "overlay.set", {"enabled": False}, workdir)
            for mode, active in (
                ("both", ("self", "peer")),
                ("self_only", ("self",)),
                ("peer_only", ("peer",)),
            ):
                for channel in ("self", "peer"):
                    await _command(
                        config,
                        "capture.set",
                        {"channel": channel, "enabled": channel in active},
                        workdir,
                    )
                for before, after in (("soniox", "qwen_audio"), ("qwen_audio", "soniox")):
                    await _command(
                        config,
                        "provider.apply",
                        {"channel": "both", "provider": before},
                        workdir,
                        timeout=120,
                    )
                    before_state = _verify_channels(
                        await _cli(config, "asr", "status"), before, active
                    )
                    receipt = await _command(
                        config,
                        "provider.apply",
                        {"channel": "both", "provider": after},
                        workdir,
                        timeout=120,
                    )
                    state = _verify_channels(await _cli(config, "asr", "status"), after, active)
                    scenario = {
                        "activity": mode,
                        "from": before,
                        "to": after,
                        "status": receipt["status"],
                        "operation_id": receipt.get("operation_id"),
                        "before_channels": before_state,
                        "after_channels": state,
                        "recognition": None,
                    }
                    report["scenarios"].append(scenario)
                    scenario["recognition"] = await _recognize(config, active, device, audio)
            report["status"] = "passed"
        except Exception as exc:
            report["failure"] = (
                str(exc)
                if isinstance(exc, (EvidenceFailure, sd.PortAudioError))
                else type(exc).__name__
            )
        finally:
            if instance_id is not None:
                try:
                    current = await _cli(config, "app", "status", timeout=10)
                    if current.get("instance_id") == instance_id:
                        await _cli(config, "app", "stop", timeout=30)
                        report["own_host_stopped"] = True
                except EvidenceFailure, OSError, TimeoutError:
                    report["own_host_stopped"] = False
    report["completed_unix_seconds"] = time.time()
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Report JSON outside the checkout; WAV saved alongside",
    )
    args = parser.parse_args()
    if args.report.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        parser.error("report must be outside the checkout")
    return asyncio.run(run(args.report.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
