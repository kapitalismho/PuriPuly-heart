from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path


def units(text, language):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = "".join(
        c if unicodedata.category(c)[0] in "LNM" else (" " if language == "en" else "")
        for c in text
    )
    return text.split() if language == "en" else list(text)


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}

    def quantile(q):
        p = (len(values) - 1) * q
        lo = math.floor(p)
        hi = math.ceil(p)
        return values[lo] + (values[hi] - values[lo]) * (p - lo)

    return {"n": len(values), "p50": quantile(0.5), "p95": quantile(0.95), "max": values[-1]}


def bucket(duration):
    return "0.5-2s" if duration <= 2 else ">2-5s" if duration <= 5 else ">5-15s"


def aggregate(rows, language):
    speech = [r for r in rows if r["variant"] != "control"]
    reference_units = sum(len(units(r["reference"], language)) for r in speech)
    edits = sum(
        distance(
            units(r["reference"], language),
            units(r.get("text", "") if r["status"] == "ok" else "", language),
        )
        for r in speech
    )
    successful = [r for r in rows if r["status"] == "ok"]
    return {
        "rows": len(rows),
        "speech_rows": len(speech),
        "status": dict(Counter(r["status"] for r in rows)),
        "failed_speech_ids": [r["id"] for r in speech if r["status"] != "ok"],
        "empty_speech": sum(not r.get("text") and r["status"] == "ok" for r in speech),
        "control_false_outputs": sum(
            bool(r.get("text", "").strip()) for r in rows if r["variant"] == "control"
        ),
        "control_output_text_counts": dict(
            Counter(r.get("text", "") for r in rows if r["variant"] == "control")
        ),
        "speech_special_token_output_ids": [r["id"] for r in speech if "<|" in r.get("text", "")],
        "edits": edits,
        "reference_units": reference_units,
        "pooled_error_rate": edits / reference_units if reference_units else None,
        "failure_scoring": "Failed/timeout speech retained and scored as empty hypothesis (all reference units deleted); failures also enumerated separately.",
        "pcm_to_text_ms": percentiles([r.get("elapsed_ms") for r in successful]),
        "rtf": percentiles([r["elapsed_ms"] / (1000 * r["duration_s"]) for r in successful]),
        "native_decode_ms": percentiles(
            [r["native"]["decode_seconds"] * 1000 for r in successful if r.get("native")]
        ),
        "outside_native_decode_ms": percentiles(
            [
                r["elapsed_ms"] - r["native"]["decode_seconds"] * 1000
                for r in successful
                if r.get("native")
            ]
        ),
        "cpu_seconds": sum(r.get("cpu_seconds") or 0 for r in rows),
    }


def summarize(results: Path):
    cells = {}
    for path in sorted(results.glob("*-primary.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            continue
        language = rows[0]["language"]
        parents = {r["noise_pair_id"] for r in rows if r["variant"] in ("noise15", "noise5")}
        groups = {
            variant: aggregate([r for r in rows if r["variant"] == variant], language)
            for variant in ("clean", "noise15", "noise5", "control")
        }
        groups["paired-clean"] = aggregate([r for r in rows if r["id"] in parents], language)
        durations = {
            variant: {
                b: aggregate(
                    [r for r in rows if r["variant"] == variant and bucket(r["duration_s"]) == b],
                    language,
                )
                for b in ("0.5-2s", ">2-5s", ">5-15s")
            }
            for variant in ("clean", "noise15", "noise5")
        }
        cells[path.stem.removesuffix("-primary")] = {
            "overall": aggregate(rows, language),
            "groups": groups,
            "duration_groups": durations,
        }
        meta_path = path.with_name(path.stem + "-meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cells[path.stem.removesuffix("-primary")]["run"] = {
                "metadata_file": meta_path.name,
                "checkout": meta["host"]["checkout"],
                "initialize_ms": meta.get("initialize_ms"),
                "first_and_warmups": [
                    {
                        "phase": row["phase"],
                        "id": row["id"],
                        "status": row["status"],
                        "elapsed_ms": row["elapsed_ms"],
                    }
                    for row in meta.get("first_and_warmups", [])
                ],
                "model_bytes": sum(row["bytes"] for row in meta["verified_artifacts"]),
                "ram_available_bytes_before_load": meta["host"]["ram_available_bytes"],
                "resources": meta.get("resources_before_close"),
            }
    replays = {
        p.stem.removesuffix("-replay-summary"): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(results.glob("*-replay-summary.json"))
    }
    return {
        "normalization": "NFKC + casefold; retain Unicode L/N/M. EN punctuation becomes separators, word Levenshtein; JA punctuation/whitespace removed, character Levenshtein. No kana/kanji or orthographic equivalence rewrite.",
        "percentiles": "Linear interpolation at (n-1)*q; successful measured warm calls only for speed, all speech for accuracy.",
        "cells": cells,
        "replays": replays,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("docs/experiments/issue-353/results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summarize(args.results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output.as_posix())


if __name__ == "__main__":
    main()
