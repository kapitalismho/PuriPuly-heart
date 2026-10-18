from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sanitize(value):
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    for prefix, label in ((str(ROOT), "<CHECKOUT>"), (str(HOME), "<USER_HOME>")):
        for variant in (prefix, prefix.replace("\\", "/")):
            text = text.replace(variant.replace("\\", "\\\\"), label).replace(variant, label)
    return text + "\n"


def inventory(path):
    if path.is_dir():
        return {
            "path": path,
            "status": "present",
            "files": {
                child.relative_to(path).as_posix(): {
                    "bytes": child.stat().st_size,
                    "sha256": digest(child),
                }
                for child in sorted(path.rglob("*"))
                if child.is_file()
            },
        }
    if path.is_file():
        return {
            "path": path,
            "status": "present",
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    return {"path": path, "status": "missing"}


def run(output):
    started = time.monotonic()
    output = output.resolve()
    if output.parent != ROOT / "build" or not output.name.startswith("issue264-validation-"):
        raise ValueError("Only assigned isolated build resources are permitted")
    if output.exists():
        raise FileExistsError("Preserve previous attempts; select a fresh output directory")
    output.mkdir(parents=True)
    paths = {
        "nemotron_artifact": HOME / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.gguf",
        "nemotron_artifact_manifest": HOME
        / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.manifest.json",
        "nemotron_runtime": ROOT / "build/nemotron",
        "gpu_qwen_worker": HOME
        / "Documents/dev/puripuly_heart/build/gpu_worker/PuriPulyHeartGpuWorker.exe",
        "gpu_qwen_model": ROOT.parent
        / "issue/.agents/goals/goal-issue-20-local-asr-backends-001/evidence/gpu-production/model-root/qwen3-asr-1.7b-q6-k-transcribe-vulkan/Qwen3-ASR-1.7B-Q6_K.gguf",
        "gemma_model": HOME / "AppData/Local/puripuly-heart/models/gemma-4-e4b-it-qat-ud-q4-k-xl",
        "gemma_runtime": HOME / "Documents/dev/puripuly_heart/build/llama.cpp/llama.cpp-b11537",
        "public_corpus_manifest": HOME
        / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1/manifest.json",
        "public_A": HOME
        / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1/clips/common-voice-17.0/en/common_voice_en_11289258.mp3",
        "public_B": HOME
        / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1/clips/common-voice-17.0/en/common_voice_en_12274.mp3",
        "public_long": HOME
        / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1/clips/fleurs/en_us/1009709090964908274.wav",
        "synthetic_No": ROOT.parent
        / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification/audio/local_no_trim.wav",
        "synthetic_correction": ROOT.parent
        / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification/audio/local_correction_trim.wav",
        "synthetic_provenance": ROOT.parent
        / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification/local-results-real.json",
        "deepgram_native_archive": ROOT / "native/nemotron/issue218_cloud_deepgram_raw.zip",
        "flet_visibility_guard": ROOT / "native/nemotron/issue218_policy_visibility.py",
        "flet_client": HOME / ".flet/client/flet-desktop-full-1.0.0-758f21506fbb/flet/flet.exe",
        "flet_ownership_guard": ROOT / "native/nemotron/issue216_presentation.py",
    }
    assets = {name: inventory(path) for name, path in paths.items()}
    approved = {
        "public_A": "19865499680804b33e8ac1f3a7b96fa8762e31be6a143c9e711bba72984272ad",
        "public_B": "cccded3f9d5a5bb4c28110b8273e217cc44d5d9e9de95eb70fd19208dce9b4f5",
        "public_long": "e5664ae10a6273fbe0009068e512f4c416cccbeff19d3d2d11e5fb8d72d03fb7",
        "synthetic_No": "e474e0808b047ccbc2a678a22531adc9599cd07a2a5fccea0e7be8047b777c49",
        "synthetic_correction": "34a41d87d8752cc290945044af90309c36d7076c0a1f236394a095a78d7eadac",
    }
    for name, expected in approved.items():
        assets[name]["approved_input_hash_match"] = assets[name].get("sha256") == expected
    dependencies = {}
    for name in (
        "numpy",
        "onnxruntime",
        "httpx",
        "flet",
        "flet-desktop",
        "psutil",
        "deepgram-sdk",
        "sherpa-onnx",
        "sherpa-onnx-core",
        "truststore",
        "soundfile",
    ):
        try:
            distribution = importlib.metadata.distribution(name)
            dependencies[name] = {
                "version": distribution.version,
                "location": str(distribution.locate_file("")),
            }
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = {"status": "missing"}
    tools = {name: shutil.which(name) for name in ("ffmpeg", "vulkaninfo")}
    vulkan = {"status": "not_available"}
    if tools["vulkaninfo"]:
        result = subprocess.run(
            [tools["vulkaninfo"], "--summary"], capture_output=True, text=True, timeout=30
        )
        vulkan = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    archive_record = {"status": "missing"}
    if paths["deepgram_native_archive"].is_file():
        with zipfile.ZipFile(paths["deepgram_native_archive"]) as archive:
            raw = archive.read("valid/trace.json")
        tape = json.loads(raw)
        archive_record = {
            "member": "valid/trace.json",
            "member_sha256": hashlib.sha256(raw).hexdigest(),
            "historical_candidate": tape["candidate"],
            "available_cases": [
                {"case": c["case"], "mode": c["mode"], "repetition": c["repetition"]}
                for c in tape["cases"]
            ],
            "status": "readable",
            "native_service_calls": 0,
        }
    import psutil

    background = []
    for process in psutil.process_iter(["pid", "name"]):
        name = (process.info["name"] or "").lower()
        if any(
            word in name
            for word in (
                "vrchat",
                "vrserver",
                "vrmonitor",
                "llama",
                "gpuworker",
                "flet",
                "nemotron",
            )
        ):
            background.append(process.info)
    report = {
        "schema": "issue264-validation-prerequisites-1",
        "baseline": "cc75792e9704e8609f38a3aa0794c03a1c7e3416",
        "candidate": "pending Director stable pin; no product imported or inference executed",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
            "pid": os.getpid(),
        },
        "driver_sha256": digest(Path(__file__)),
        "assets": assets,
        "dependencies": dependencies,
        "tools": tools,
        "vulkan_enumeration_only": vulkan,
        "native_tape": archive_record,
        "potentially_relevant_background_processes_not_owned_or_stopped": background,
        "observed_product_modules_loaded": sorted(
            name for name in sys.modules if name.startswith("puripuly_heart")
        ),
        "scope": "Read-only artifact/dependency/corpus inspection and Vulkan device enumeration; no model, GUI, provider or product startup",
        "startup_design_inspection": {
            "composition": "No application composition, persisted settings, keyring, Broker, OSCQuery, VRChat scene discovery or telemetry owner constructed",
            "translation": "Explicit managed Gemma install/runtime; InMemorySecretStore; task-local HttpExtensionRegistry; prefix disk cache absent; loopback server only",
            "workers": "Qwen factory creates owned authenticated loopback socket and temporary launch manifest; child TEMP/TMP will be task-local; Nemotron child only; Flet diagnostic ingress/backdrop explicitly owned",
            "remaining_runtime_observations": "Confirm process/executable/import/device/effective geometry, socket destinations, owned windows uncloaked and valid layouts/readbacks after stable pin. Presence and Vulkan enumeration do not prove inference readiness.",
        },
        "new_resources": [str(output), str(output / "prerequisites.json")],
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "prerequisites.json").write_text(sanitize(report), encoding="utf-8")
    print(
        sanitize(
            {
                "status": "inventory_complete",
                "output": output,
                "missing_assets": [
                    name for name, value in assets.items() if value["status"] == "missing"
                ],
                "input_hash_mismatches": [
                    name for name in approved if not assets[name]["approved_input_hash_match"]
                ],
                "product_modules_loaded": report["observed_product_modules_loaded"],
                "dependencies": dependencies,
                "background": background,
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
