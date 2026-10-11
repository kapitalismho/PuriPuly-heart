from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from prepare import prepare, sha256

from puripuly_heart.app.adapters.nemotron.artifacts import (
    CHECKPOINT_SHA256,
    MODEL_REVISION,
    SOURCE_REVISION,
    inspect_artifact,
)

CHECKPOINT_SIZE = 198676480
CHECKPOINT_URL = f"https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/{MODEL_REVISION}/Nemotron-3-Diarization.nemo"


def acquire(cache: Path, consent: bool) -> Path:
    path = cache / "Nemotron-3-Diarization.nemo"
    if path.exists():
        if sha256(path) != CHECKPOINT_SHA256:
            raise ValueError("cached official checkpoint hash mismatch")
        return path
    if not consent:
        raise PermissionError("explicit model download consent required")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".downloading")
    try:
        total = 0
        with (
            urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            while block := response.read(1048576):
                total += len(block)
                if total > CHECKPOINT_SIZE:
                    raise ValueError("official checkpoint exceeds pinned size")
                output.write(block)
        if total != CHECKPOINT_SIZE or sha256(temporary) != CHECKPOINT_SHA256:
            raise ValueError("official checkpoint checksum/size mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main(args):
    repo = Path(__file__).resolve().parents[2]
    source, _ = prepare(args.cache, repo)
    checkpoint = args.checkpoint or acquire(args.cache, args.download_consent)
    if checkpoint.stat().st_size != CHECKPOINT_SIZE or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("requires exact pinned official checkpoint")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(source), str(source / "llama.cpp/gguf-py")))
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    versions = subprocess.run(
        [
            str(args.converter_python),
            "-c",
            "import importlib.metadata,json;print(json.dumps({p:importlib.metadata.version(p) for p in ['numpy','PyYAML','huggingface-hub','torch']}))",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    expected = {
        "numpy": "1.26.4",
        "PyYAML": "6.0.2",
        "huggingface-hub": "0.34.4",
        "torch": "2.8.0+cpu",
    }
    installed = json.loads(versions.stdout)
    if installed != expected:
        raise ValueError(f"converter environment differs from pins: {installed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".official-converting.gguf")
    command = [
        str(args.converter_python),
        str(source / "convert_model.py"),
        str(checkpoint.resolve()),
        "--architecture",
        "diarization",
        "--outtype",
        "fp16",
        "--outfile",
        str(temporary.resolve()),
    ]
    try:
        subprocess.run(command, env=environment, check=True)
        manifest = inspect_artifact(temporary)
        manifest.update(
            {
                "schema_version": 1,
                "source": {
                    "repository": "nvidia/Nemotron-3-Diarization",
                    "revision": MODEL_REVISION,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "checkpoint_size_bytes": CHECKPOINT_SIZE,
                    "url": CHECKPOINT_URL,
                    "license": "OpenMDW-1.1",
                },
                "conversion": {
                    "recipe": "official-nemo-pinned-convert_model-diarization-fp16",
                    "runtime_revision": SOURCE_REVISION,
                    "converter_sha256": sha256(source / "conversion/diarization.py"),
                    "requirements_lock_sha256": sha256(
                        repo / "native/nemotron/conversion-requirements.lock"
                    ),
                    "packages": installed,
                    "gguf_source_revision": "bd4f514db14d87fded667787a7a963bfbaa98e89",
                    "original_checkpoint_lineage": "verified exact pinned checkpoint",
                },
            }
        )
        temporary.replace(args.output)
        args.output.with_name("model.manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--download-consent", action="store_true")
    parser.add_argument("--converter-python", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
