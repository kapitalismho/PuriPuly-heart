from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from issue264_validation_prerequisites import HOME, ROOT, digest, sanitize


def load_definition(path, name, namespace):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    isolated = ast.Module(body=[node], type_ignores=[])
    exec(compile(isolated, str(path), "exec"), namespace)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest(path),
        "function": name,
        "first_line": node.lineno,
        "last_line": node.end_lineno,
    }


def run(output):
    output = output.resolve()
    if output.parent != ROOT / "build" or not output.name.startswith("issue264-validation-"):
        raise ValueError("Only assigned isolated build outputs are permitted")
    if output.exists():
        raise FileExistsError("Preserve previous preparation attempts")
    output.mkdir(parents=True)
    namespace = {"np": np, "Path": Path, "subprocess": subprocess, "json": json, "digest": digest}
    definitions = [
        load_definition(ROOT / "native/nemotron/probe.py", "audio", namespace),
        load_definition(ROOT / "native/nemotron/issue215_smoke.py", "input_cases", namespace),
    ]
    args = SimpleNamespace(
        corpus=HOME / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1",
        correction_root=ROOT.parent
        / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification",
    )
    provenance, assembled = namespace["input_cases"](args)
    selected = (
        "clean_A_B",
        "distinct_repeated_A_B_A",
        "short_A_B_A",
        "linguistic_correction",
        "six_second_rollover",
    )
    inputs = {
        name: {
            "samples": assembled[name][0].size,
            "seconds": assembled[name][0].size / 16000,
            "pcm_f32_sha256": hashlib.sha256(assembled[name][0].tobytes()).hexdigest(),
            "scoring_reference_only": assembled[name][1],
        }
        for name in selected
    }
    with zipfile.ZipFile(ROOT / "native/nemotron/issue218_cloud_deepgram_raw.zip") as archive:
        raw = archive.read("valid/trace.json")
    tape = json.loads(raw)
    canonical = {
        case["case"]: case
        for case in tape["cases"]
        if case["mode"] == "baseline" and case["repetition"] == 1
    }
    replay = {}
    for name in ("distinct_repeated_A_B_A", "short_A_B_A"):
        case = canonical[name]
        finals = [event for event in case["trace"] if event["event"] == "raw_wire_result"]
        replay[name] = {
            "assembled_input_matches_original_tape": inputs[name]["pcm_f32_sha256"]
            == case["input_pcm_f32_sha256"],
            "historical_successful_input_samples": case["actual_audio_samples_written"],
            "historical_native_results": len(finals),
            "historical_final_texts": [
                event["value"].get("channel", {}).get("alternatives", [{}])[0].get("transcript")
                for event in finals
            ],
            "historical_final_timings_retained": True,
        }
    import sys

    report = {
        "schema": "issue264-validation-inputs-1",
        "scope": "Offline ffmpeg preparation of existing public/synthetic PCM; exact existing function AST definitions executed without importing product or inference modules",
        "definitions": definitions,
        "provenance": provenance,
        "cases": inputs,
        "canonical_replay": replay,
        "tape_member_sha256": hashlib.sha256(raw).hexdigest(),
        "product_modules_loaded": sorted(
            name for name in sys.modules if name.startswith("puripuly_heart")
        ),
        "status": (
            "prepared"
            if all(value["assembled_input_matches_original_tape"] for value in replay.values())
            else "replay_input_mismatch"
        ),
    }
    (output / "inputs.json").write_text(sanitize(report), encoding="utf-8")
    print(
        sanitize(
            {
                "status": report["status"],
                "cases": inputs,
                "canonical_replay": replay,
                "product_modules_loaded": report["product_modules_loaded"],
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "prepared" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
