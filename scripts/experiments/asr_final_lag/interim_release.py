# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.27",
#   "numpy>=2.0",
#   "rapidfuzz>=3.9",
# ]
# ///
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze
import decide

DATA = ("ko_conv", "ja_conv", "en_conv", "ko_kr", "ja_jp", "en_us")
VARIANTS = ("natural", "end_smart", "end_app")
WINDOWS = (0.0, 0.6, 1.0)
STEP_S = 0.05
TAIL_CHARS = 40
MATCH_MAX_COST = 0.35


def _events(run: dict) -> list[tuple[float, str | None, str | None]]:
    return [(e["t"], e.get("fa"), e.get("tail")) for e in run["events"]]


def _find_end(text: str, final: str) -> int | None:
    if not text:
        return 0
    tail = text[-TAIL_CHARS:]
    cost, end = analyze._semi_global_end(tail, final)
    return end if cost <= MATCH_MAX_COST * len(tail) else None


def simulate(run: dict, window: float, remainder: str) -> tuple[list[tuple[float, str]], dict]:
    events = _events(run)
    end_t = (events[-1][0] if events else 0.0) + 1.0
    releases: list[tuple[float, str]] = []
    stats = {"turns": 0, "unmatched": 0}
    interim, released, since = "", 0, {}
    index = 0
    t = 0.0
    while t <= end_t:
        while index < len(events) and events[index][0] <= t:
            event_t, final, tail = events[index]
            index += 1
            if final:
                stats["turns"] += 1
                norm_final = analyze.normalize(final)
                norm_interim = analyze.normalize(interim)
                done = norm_interim[:released]
                if remainder == "final":
                    q = _find_end(done, norm_final)
                    if q is None:
                        stats["unmatched"] += 1
                        piece = norm_interim[released:]
                        q2 = _find_end(norm_interim, norm_final)
                        piece += norm_final[q2:] if q2 is not None else ""
                    else:
                        piece = norm_final[q:]
                else:
                    q2 = _find_end(norm_interim, norm_final)
                    piece = norm_interim[released:] + (norm_final[q2:] if q2 is not None else "")
                    if q2 is None:
                        stats["unmatched"] += 1
                if piece:
                    releases.append((event_t, piece))
                interim, released, since = "", 0, {}
            if tail is not None:
                interim = tail
        if remainder != "finals":
            norm, terminals = analyze._terminal_indices(interim)
            current = {}
            for p in terminals:
                key = (p, norm[:p])
                current[key] = since.get(key, t)
            since = current
            ready = [p for (p, _), s in since.items() if p > released and t - s >= window]
            if ready:
                p = max(ready)
                releases.append((round(t, 3), norm[released:p]))
                released = p
        t = round(t + STEP_S, 6)
    return releases, stats


def finals_only(run: dict) -> list[tuple[float, str]]:
    return [(t, analyze.normalize(fa)) for t, fa, _ in _events(run) if fa]


def evaluate(stream: decide.Stream, reference_norm: str, releases) -> dict:
    text = "".join(piece for _, piece in releases)
    times, positions, cumulative = [], [], ""
    for t, piece in releases:
        cumulative += piece
        position = stream.locate(cumulative)
        if positions and position <= positions[-1]:
            continue
        times.append(t)
        positions.append(position)
    horizon = stream.duration_s - decide.END_EXCLUSION_S
    latencies, missing = [], 0
    for j in range(1, len(stream.final_norm) + 1):
        spoken = float(stream.char_time[j - 1])
        if spoken > horizon:
            continue
        k = bisect.bisect_left(positions, j - decide.LABEL_SLACK)
        if k == len(positions):
            missing += 1
            continue
        latencies.append(times[k] - spoken)
    ops = {"insert": 0, "delete": 0, "replace": 0}
    for op in Levenshtein.editops(reference_norm, text):
        ops[op.tag] += 1
    return {
        "releases": len(releases),
        "cer": round(Levenshtein.distance(reference_norm, text) / max(1, len(reference_norm)), 4),
        "deleted": ops["delete"],
        "inserted": ops["insert"],
        "latencies": latencies,
        "unreleased_chars": missing,
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.array(values)
    return {
        "n": len(values),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
    }


def policies() -> list[tuple[str, float | None, str]]:
    items = [("finals only", None, "finals")]
    for window in WINDOWS:
        items.append((f"interim W={window} + final remainder", window, "final"))
    for window in WINDOWS:
        items.append((f"interim W={window} + interim remainder", window, "interim"))
    return items


def render(rows: list[dict], meta: dict) -> str:
    lines = [
        "# Gemini interim release replay",
        "",
        "Replays recorded Gemini runs. `finals only` releases each authoritative final as it "
        "arrives (the current application). The interim policies release interim text up to "
        "its last sentence-final punctuation once that prefix has been unchanged for W seconds. "
        "When a final arrives, they release only what is left:",
        "",
        "- `final remainder`: the final after the already released text (located by text "
        "alignment). When the released text cannot be found in the final, the rest of the last "
        "interim is used instead (`unmatched`).",
        "- `interim remainder`: the rest of the last interim, plus any final text after the end "
        "of that interim.",
        "",
        "`latency` is seconds from each character's Soniox end timestamp to its release "
        "(`p50 / p90`). CER, `deleted` and `inserted` compare all released text with the "
        "reference text. Conversation references are uploaded captions.",
        "",
        "| data | variant | policy | releases | latency | CER | deleted | inserted | turns "
        "| unmatched |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lat = r["latency"]
        latency = f"{lat['p50']:.2f} / {lat['p90']:.2f}" if lat.get("n") else "-"
        lines.append(
            f"| {r['data']} | {r['variant']} | {r['policy']} | {r['releases']} | {latency} "
            f"| {r['cer']:.3f} | {r['deleted']} | {r['inserted']} | {r.get('turns', '-')} "
            f"| {r.get('unmatched', '-')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Gemini interim release policies.")
    parser.add_argument("--results", nargs="*", type=Path, help="ASR results folders")
    parser.add_argument("--data", nargs="*", default=list(DATA))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    folders = args.results or sorted((HERE / "results").glob("2*Z"))
    rows, sources = [], {}
    for data in args.data:
        for variant in VARIANTS:
            folder = next(
                (
                    f
                    for f in sorted(folders, reverse=True)
                    if (f / "runs" / f"gemini_{variant}_{data}.json.gz").exists()
                    and (f / "runs" / f"soniox_natural_{data}.json.gz").exists()
                ),
                None,
            )
            if folder is None:
                continue
            run = analyze.load_run(folder / "runs" / f"gemini_{variant}_{data}.json.gz")
            if run.get("error"):
                continue
            stream = decide.build_stream(folder, "gemini", data, variant)
            if stream is None:
                continue
            reference = json.loads((folder / "reference" / f"{data}.json").read_text("utf-8"))
            reference_norm = analyze.normalize("".join(s["text"] for s in reference["sentences"]))
            sources[f"{data}/{variant}"] = folder.name
            for name, window, remainder in policies():
                if remainder == "finals":
                    releases, stats = finals_only(run), {}
                else:
                    releases, stats = simulate(run, window, remainder)
                result = evaluate(stream, reference_norm, releases)
                rows.append(
                    {
                        "data": data,
                        "variant": variant,
                        "policy": name,
                        **{k: v for k, v in result.items() if k != "latencies"},
                        "latency": _stats(result["latencies"]),
                        **stats,
                    }
                )
            print(f"[interim] {data:8} {variant}", flush=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or HERE / "results" / f"interim-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_utc": stamp,
        "sources": sources,
        "windows": WINDOWS,
        "step_s": STEP_S,
        "match_max_cost": MATCH_MAX_COST,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    (out / "summary.md").write_text(render(rows, meta), encoding="utf-8")
    print(f"[interim] wrote {out}")


if __name__ == "__main__":
    main()
