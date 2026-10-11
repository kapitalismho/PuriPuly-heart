from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from probe import audio

from puripuly_heart.app.adapters.nemotron import NemotronBackend


async def run(args):
    os.environ["GGML_VK_PIPELINE_STATS"] = "_"
    values = audio(args.audio)
    result = {
        "scope": "actual lazily compiled/requested Vulkan pipeline statistics for up to three seconds of approved public fixture; not a universal per-node accumulator guarantee",
        "public_fixture": {
            "filename": args.audio.name,
            "sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        },
        "profiles": {},
    }
    for profile in ("1.04", "0.64"):
        backend = NemotronBackend(
            artifact_path=args.model, runtime_dir=args.runtime, device=args.device, profile=profile
        )
        ready = await backend.start()
        frames = 0
        try:
            await backend.reset(1)
            for offset in range(0, min(values.size, 48000), 16000):
                block = np.array(
                    values[offset : min(values.size, offset + 16000, 48000)], dtype=np.float32
                )
                block.flags.writeable = False
                update = await backend.feed(block, offset)
                frames += len(update.frames)
        finally:
            await backend.stop()
        stderr = backend._stderr.decode("utf-8", errors="replace")
        prefix = "ggml_vulkan: pipeline stats for "
        pipelines = [
            line[len(prefix) :].removesuffix(":")
            for line in stderr.splitlines()
            if line.startswith(prefix)
        ]
        result["profiles"][profile] = {
            "frames": frames,
            "ready_diagnostics": dict(ready.diagnostics),
            "observed_pipeline_names": pipelines,
            "statistics_stderr_tail": stderr,
            "bounded_stderr_bytes": len(backend._stderr),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                profile: {"frames": value["frames"], "pipelines": value["observed_pipeline_names"]}
                for profile, value in result["profiles"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
