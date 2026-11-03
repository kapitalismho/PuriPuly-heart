from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src")

from resources import ResourceSampler, identity, sha256


def read_pcm(path, expected_hash=None):
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"Audio hash mismatch: {path.name}")
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Expected mono16k PCM16: {path.name}")
        return stream.readframes(stream.getnframes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_config(config):
    if config["adapter"] != "app":
        return config
    from puripuly_heart.core.local_asr.local_stt_assets import load_local_stt_asset_manifest
    from puripuly_heart.core.local_asr.local_stt_catalog import LOCAL_CPU_MODEL_CONTRACTS

    manifest = load_local_stt_asset_manifest(config["model_id"])
    return {
        **config,
        "languages": sorted(LOCAL_CPU_MODEL_CONTRACTS[manifest.model_id].supported_language_codes),
        "repository": manifest.upstream_repo,
        "revision": manifest.sources["huggingface"].revision,
        "files": [
            {
                "root": "app",
                "path": f"{manifest.install_dirname}/{asset.relative_path}",
                "bytes": asset.size_bytes,
                "sha256": asset.sha256,
            }
            for asset in manifest.files
        ],
    }


def verify_models(config, args):
    files = []
    for artifact in config["files"]:
        path = (args.app_model_root if artifact["root"] == "app" else args.model_root) / artifact[
            "path"
        ]
        measured = {"file": artifact["path"], "bytes": path.stat().st_size, "sha256": sha256(path)}
        if measured["bytes"] != artifact["bytes"] or measured["sha256"] != artifact["sha256"]:
            raise ValueError(f"Model artifact integrity mismatch: {path.name}")
        files.append(measured)
    return files


async def measured_decode(adapter, pcm, row, sampler, timeout):
    before = sampler.sample()
    start = time.perf_counter()
    result = {**row, "text": "", "status": "ok", "error": None}
    try:
        result["text"] = await asyncio.wait_for(adapter.decode(pcm), timeout=timeout)
    except TimeoutError as error:
        result.update(status="timeout", error=type(error).__name__)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result["elapsed_ms"] = (time.perf_counter() - start) * 1000
    result["cpu_seconds"] = sampler.sample() - before
    result["native"] = adapter.observed.last_native
    return result


async def execute(args, configs):
    from adapters import ModelAdapter

    config = model_config(configs["models"][args.model])
    if args.language not in config["languages"]:
        raise ValueError("Model does not declare requested language")
    artifacts = verify_models(config, args)
    inputs_path = args.inputs / f"{args.language}.jsonl"
    rows = [
        json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if args.limit:
        rows = rows[: args.limit]
    elif args.mode == "smoke":
        rows = [row for row in rows if row["variant"] == "clean"][:3]
    audio = {row["id"]: read_pcm(args.data_root / row["path"], row["sha256"]) for row in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    cell = f"{args.language}-{args.model}"
    tag = f"{cell}-{args.mode}"
    metadata = {
        "cell": cell,
        "mode": args.mode,
        "utc_started": datetime.datetime.now(datetime.UTC).isoformat(),
        "host": identity(ROOT, args.worker),
        "model": config,
        "verified_artifacts": artifacts,
        "inputs_sha256": sha256(inputs_path),
        "model_config_sha256": sha256(args.config),
        "settings": {
            "device": "cpu",
            "inference_threads": 2,
            "batch": 1,
            "resident": True,
            "serial": True,
            "language_hint": None,
            "prompt_context_custom_vocabulary_external_correction": False,
            "warmups": 2,
            "first_decode_separate": True,
        },
        "boundary": "Preloaded mono16k PCM16 bytes -> app decode_pcm16le -> returned text. GGUF includes float conversion, requantized temporary WAV/write/read and authenticated IPC; sherpa is resident in-process float conversion/owned thread/CTC greedy. Native GGUF decode_seconds excludes WAV/IPC and is recorded separately. File loading/hash checks excluded; native ITN follows unchanged app defaults when model advertises support.",
        "rows": len(rows),
    }
    source_paths = [
        "benchmarks/asr/run.py",
        "benchmarks/asr/adapters.py",
        "benchmarks/asr/replay.py",
        "benchmarks/asr/resources.py",
        "benchmarks/asr/score.py",
        "native/asr_worker/Cargo.toml",
        "src/puripuly_heart/providers/stt/local_cpp.py",
        "src/puripuly_heart/providers/stt/local_session.py",
        "src/puripuly_heart/providers/stt/local_decode.py",
        "src/puripuly_heart/providers/stt/local_parakeet_sherpa.py",
        "src/puripuly_heart/core/vad/gating.py",
    ]
    metadata["source_sha256"] = {path: sha256(ROOT / path) for path in source_paths}
    metadata["thread_environment"] = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    native_site = Path(sys.prefix) / "Lib/site-packages"
    metadata["python_native_build"] = [
        {
            "file": path.relative_to(native_site).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for pattern in (
            "sherpa_onnx/lib/*.pyd",
            "sherpa_onnx/lib/*.dll",
            "onnxruntime/capi/*.pyd",
            "onnxruntime/capi/*.dll",
        )
        for path in sorted(native_site.glob(pattern))
    ]
    adapter = ModelAdapter(config, args.language, args.app_model_root, args.model_root, args.worker)
    metadata["settings"]["language_hint"] = getattr(adapter.backend, "language_hint", None)
    sampler = ResourceSampler()
    try:
        start = time.perf_counter()
        try:
            await adapter.initialize()
            metadata["initialize_ms"] = (time.perf_counter() - start) * 1000
            metadata["activation"] = adapter.observed.activation
        except Exception as error:
            metadata.update(
                initialize_ms=(time.perf_counter() - start) * 1000,
                initialization_error=f"{type(error).__name__}: {error}",
            )
            save(args.output / f"{tag}-meta.json", metadata)
            raise
        warmup_rows = []
        clean = [r for r in rows if r["variant"] == "clean"]
        for index in range(3):
            row = clean[index % len(clean)]
            result = await measured_decode(
                adapter,
                audio[row["id"]],
                {**row, "model": args.model, "phase": "first" if index == 0 else "warmup"},
                sampler,
                args.timeout,
            )
            warmup_rows.append(result)
        metadata["first_and_warmups"] = warmup_rows
        if args.mode in ("primary", "smoke"):
            output = args.output / f"{tag}.jsonl"
            with output.open("w", encoding="utf-8") as stream:
                for index, row in enumerate(rows, 1):
                    result = await measured_decode(
                        adapter,
                        audio[row["id"]],
                        {
                            **{
                                key: row.get(key)
                                for key in (
                                    "id",
                                    "language",
                                    "reference",
                                    "duration_s",
                                    "variant",
                                    "noise_pair_id",
                                    "path",
                                    "sha256",
                                    "speech_end_s",
                                )
                            },
                            "model": args.model,
                            "phase": args.mode,
                        },
                        sampler,
                        args.timeout,
                    )
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(
                        f"{tag} {index}/{len(rows)} {result['status']} {result['elapsed_ms']:.1f}ms",
                        flush=True,
                    )
        else:
            from replay import replay

            manifest = json.loads(
                (args.inputs / f"replay-{args.language}.json").read_text(encoding="utf-8")
            )
            for variant in ("clean", "noise15"):
                pcm = read_pcm(
                    args.data_root / manifest["paths"][variant], manifest["sha256"][variant]
                )
                summary = await replay(
                    adapter,
                    manifest,
                    pcm,
                    ROOT,
                    variant,
                    args.output / f"{cell}-{variant}-replay.jsonl",
                )
                save(args.output / f"{cell}-{variant}-replay-summary.json", summary)
                print(f"{cell} replay {variant}: {summary['counts']}", flush=True)
    finally:
        metadata["resources_before_close"] = sampler.snapshot()
        await adapter.close()
        metadata["resources"] = sampler.close()
        metadata["utc_finished"] = datetime.datetime.now(datetime.UTC).isoformat()
        metadata["cleanup"] = (
            "Owned backend closed; worker shutdown requested and awaited; temporary WAV/worker launch directories cleaned by app API. No settings/secrets/discovery/translation/TTS/network inference initialized."
        )
        save(args.output / f"{tag}-meta.json", metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BENCHMARK / "models.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--language")
    parser.add_argument(
        "--mode", choices=("download", "smoke", "primary", "replay"), default="primary"
    )
    parser.add_argument("--data-root", type=Path, default=BENCHMARK / "data")
    parser.add_argument("--model-root", type=Path, default=BENCHMARK / "models")
    parser.add_argument("--app-model-root", type=Path, default=BENCHMARK / "models")
    parser.add_argument(
        "--worker", type=Path, default=ROOT / "build/asr_worker/PuriPulyHeartASRWorker.exe"
    )
    parser.add_argument("--inputs", type=Path, default=BENCHMARK / "inputs")
    parser.add_argument("--output", type=Path, default=BENCHMARK / "results")
    parser.add_argument("--source", choices=("huggingface", "hf_mirror"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    configs = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mode == "download":
        from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
            ensure_local_stt_installed,
        )

        config = model_config(configs["models"][args.model])
        if config["adapter"] != "app":
            parser.error("Place experimental GGUF files under --model-root before running")
        installed = asyncio.run(
            ensure_local_stt_installed(
                model_id=config["model_id"],
                model_root=args.app_model_root,
                preferred_source=args.source,
            )
        )
        print(json.dumps(installed.to_dict()))
        return
    if not args.language or not args.worker:
        parser.error("Inference requires --language and explicit --worker path")
    asyncio.run(execute(args, configs))


if __name__ == "__main__":
    main()
