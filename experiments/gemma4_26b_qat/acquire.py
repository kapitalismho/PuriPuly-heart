from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

TARGET = {
    "repo": "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
    "revision": "7b92b5b28818151e8669af2e45e88d6086f490dd",
    "file": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    "size": 14249047104,
    "sha256": "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891",
}
BUILDS = {"b11537": "8e2d31e0eb658d0382d3106a61369dff81f0c080", "b10423": "a94d563ed801d1da1b8c2432946de07d0231bb3d"}


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()}


def acquire(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        result = subprocess.run(["curl.exe", "--fail", "--location", "--retry", "2", "--output", str(partial), url])
        if result.returncode:
            raise RuntimeError(f"Download failed with exit {result.returncode}: {destination.name}")
        partial.replace(destination)
    result = identity(destination)
    result.update(url=url, acquisition_seconds=time.time() - started)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for build, commit in BUILDS.items():
        filename = f"llama-{build}-bin-win-vulkan-x64.zip"
        record = acquire(f"https://github.com/ggml-org/llama.cpp/releases/download/{build}/{filename}", root / "downloads" / filename)
        record.update(build=build, commit=commit, origin="official prebuilt Windows x64 Vulkan release")
        with zipfile.ZipFile(root / "downloads" / filename) as archive:
            archive.extractall(root / "runtimes" / build)
        record["binaries"] = [identity(p) for p in sorted((root / "runtimes" / build).rglob("*")) if p.suffix.lower() in {".dll", ".exe"}]
        records.append(record)
        (root / "acquisition.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    url = f"https://huggingface.co/{TARGET['repo']}/resolve/{TARGET['revision']}/{TARGET['file']}?download=true"
    record = acquire(url, root / "models" / TARGET["file"])
    record.update(repo=TARGET["repo"], revision=TARGET["revision"])
    record["verified"] = record["size"] == TARGET["size"] and record["sha256"] == TARGET["sha256"]
    records.append(record)
    (root / "acquisition.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if not record["verified"]:
        raise RuntimeError("Target artifact identity mismatch")


if __name__ == "__main__":
    main()
