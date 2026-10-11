from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from puripuly_heart.app.adapters.nemotron.artifacts import (
    PRECONVERTED_FILENAME,
    PRECONVERTED_REPOSITORY,
    PRECONVERTED_REVISION,
    PRECONVERTED_SHA256,
    PRECONVERTED_SIZE,
    adopt_preconverted_artifact,
    sha256_file,
)


def acquire(destination: Path, consent: bool) -> dict:
    if not consent:
        raise PermissionError("explicit preconverted model download consent required")
    if destination.exists():
        return adopt_preconverted_artifact(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".downloading")
    url = f"https://huggingface.co/{PRECONVERTED_REPOSITORY}/resolve/{PRECONVERTED_REVISION}/{PRECONVERTED_FILENAME}"
    total = 0
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as out:
            while block := response.read(1048576):
                total += len(block)
                if total > PRECONVERTED_SIZE:
                    raise ValueError("preconverted artifact exceeds pinned size")
                out.write(block)
        if total != PRECONVERTED_SIZE or sha256_file(temporary) != PRECONVERTED_SHA256:
            raise ValueError("preconverted artifact differs from source pin")
        temporary.replace(destination)
        return adopt_preconverted_artifact(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--download-consent", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire(args.destination, args.download_consent), indent=2))
