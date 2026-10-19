"""Score benchmark runs and write per-case results plus a comparison report."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bench import reference, units
from bench.score import PARAMETERS, score_case

BENCH_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = BENCH_ROOT / "corpus" / "references"
DEFAULT_GAP_S = 0.5
MAX_FRAME_LAG_S = 1.0
ARM_ORDER = ["baseline", "p104", "p064"]


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _quantile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def score_runs(root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted((root / "runs").glob("*/*/r*")):
        route, arm, rep = run_dir.parts[-3], run_dir.parts[-2], run_dir.parts[-1]
        observations = run_dir / "observations.jsonl"
        events = units.load_events(observations) if observations.is_file() else []
        for case_file in sorted((run_dir / "cases").glob("*.json")):
            status = json.loads(case_file.read_text(encoding="utf-8"))
            case_id = case_file.stem
            row = {
                "route": route,
                "arm": arm,
                "repetition": rep,
                "case_id": case_id,
                "status": status.get("status"),
            }
            if status.get("status") != "completed":
                row["reason"] = status.get("reason")
                rows.append(row)
                continue
            hangover = status.get("effective_vad_hangover_ms")
            gap_s = hangover / 1000 if hangover else DEFAULT_GAP_S
            ref = reference.load(REFERENCES / f"{case_id}.json", gap_s=gap_s)
            try:
                run = units.case_run(
                    events,
                    status.get("observation_tag") or case_id,
                    max_frame_lag_s=MAX_FRAME_LAG_S,
                )
            except ValueError as exc:
                row.update({"status": "invalid", "reason": str(exc)})
                rows.append(row)
                continue
            result = score_case(ref, run)
            result["drain_timed_out"] = status.get("drain_timed_out")
            result["attempt"] = status.get("attempt", 1)
            result["incidents"] = status.get("incidents", [])
            result["frame_lag_s"] = run.source_record.get("frame_lag_s")
            result["stalls_over_100ms"] = run.source_record.get("stalls_over_100ms")
            if run.invalid:
                row["status"] = "invalid"
                row["reason"] = ",".join(run.invalid)
            row["score"] = result
            out = root / "scores" / route / arm / rep / f"{case_id}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            rows.append(row)
    return rows


def _scored(rows):
    return [r for r in rows if r.get("status") == "completed" and "score" in r]


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in _scored(rows):
        groups[(row["route"], row["arm"])].append(row)
    baseline = {
        (r["route"], r["case_id"], r["repetition"]): r["score"]
        for r in _scored(rows)
        if r["arm"] == "baseline"
    }
    summary = {}
    for (route, arm), group in sorted(groups.items()):
        attribution = Counter()
        transitions = defaultdict(list)
        presentation = Counter()
        violations = Counter()
        boundaries = Counter()
        content = Counter()
        delays, completion, short = [], [], []
        returns = Counter()
        mixed = colored_mixed = scored_units = bursts = 0
        speech_s = 0.0
        leads, completion_delta = [], []
        cuts = Counter()
        for row in group:
            s = row["score"]
            attribution.update(s["attribution"])
            presentation.update(s["presentation"])
            violations.update(s["presentation_violations"])
            boundaries.update(b["kind"] for b in s["boundaries"])
            content.update(s["content"])
            delays.extend(s["late_delays_s"])
            completion.extend(c["latency_s"] for c in s["completion"])
            short.extend(s["short_segments"])
            returns.update(r["outcome"] for r in s["returns"])
            mixed += s["mixed_units"]
            colored_mixed += s["colored_mixed_units"]
            scored_units += s["scored_units"]
            bursts += s["bursts"]
            kinds = {t["id"]: t["kind"] for t in s["transitions"]}
            for cut in s.get("speaker_cuts", []):
                cuts["total"] += 1
                cuts[f"at_{kinds[cut['transition']]}" if cut["transition"] else "elsewhere"] += 1
            speech_s += s["speech_s"]
            base = baseline.get((route, row["case_id"], row["repetition"]))
            base_tr = {t["id"]: t for t in base["transitions"]} if base else {}
            base_completion = (
                {c["segment"]: c["latency_s"] for c in base["completion"]} if base else {}
            )
            for c in s["completion"]:
                if c["segment"] in base_completion:
                    completion_delta.append(c["latency_s"] - base_completion[c["segment"]])
            for t in s["transitions"]:
                transitions[t["kind"]].append(t)
                b = base_tr.get(t["id"])
                if (
                    t["kind"] == "continuous"
                    and b is not None
                    and t.get("predecessor_release_s") is not None
                    and b.get("predecessor_release_s") is not None
                ):
                    leads.append(b["predecessor_release_s"] - t["predecessor_release_s"])
        recognized = sum(v for k, v in attribution.items() if k != "not_recognized")
        cont = transitions.get("continuous", [])
        scored_cont = [t for t in cont if t["outcome"] != "content_missing"]
        errors = [t["boundary_error_s"] for t in cont if t.get("boundary_error_s") is not None]
        readable_units = sum(presentation.values())
        summary[f"{route}/{arm}"] = {
            "route": route,
            "arm": arm,
            "cases": len(group),
            "speech_minutes": round(speech_s / 60, 2),
            "attribution": {
                "recognized_tokens": recognized,
                "colored_correct": (
                    attribution["colored_correct"] / recognized if recognized else None
                ),
                "colored_wrong": attribution["colored_wrong"] / recognized if recognized else None,
                "overflow": (
                    (attribution["overflow_correct"] + attribution["overflow_wrong"]) / recognized
                    if recognized
                    else None
                ),
                "gray": (
                    sum(v for k, v in attribution.items() if k.startswith("gray_")) / recognized
                    if recognized
                    else None
                ),
                "gray_by_state": {
                    k[5:]: v for k, v in attribution.items() if k.startswith("gray_")
                },
                "not_recognized_tokens": attribution["not_recognized"],
            },
            "early_boundary": {
                "continuous_transfers": len(cont),
                "content_missing": len(cont) - len(scored_cont),
                "separated": sum(1 for t in scored_cont if t["separated"]),
                "misplaced_boundary": sum(
                    1 for t in scored_cont if t["outcome"] == "misplaced_boundary"
                ),
                "speaker_cut": sum(1 for t in cont if t.get("speaker_cut")),
                "evidence_supported": sum(
                    1 for t in cont if (t.get("evidence") or {}).get("supported")
                ),
                "evidence_lead_p50_s": _median(
                    [(t.get("evidence") or {}).get("lead_s") for t in cont]
                ),
                "missed_with_evidence": sum(
                    1
                    for t in scored_cont
                    if not t["separated"] and (t.get("evidence") or {}).get("before_region_end")
                ),
                "missed_without_evidence": sum(
                    1
                    for t in scored_cont
                    if not t["separated"] and not (t.get("evidence") or {}).get("supported")
                ),
                "cut_detection_p50_s": _median([t.get("cut_detection_s") for t in cont]),
                "cut_detection_p90_s": _quantile([t.get("cut_detection_s") for t in cont], 0.9),
                "merged": sum(1 for t in scored_cont if t["outcome"] == "merged"),
                "early_release": sum(1 for t in scored_cont if t["early_release"]),
                "predecessor_latency_p50_s": _median(
                    t.get("predecessor_latency_s") for t in scored_cont
                ),
                "predecessor_latency_p90_s": _quantile(
                    [t.get("predecessor_latency_s") for t in scored_cont], 0.9
                ),
                "release_lead_vs_baseline_p50_s": _median(leads) if arm != "baseline" else None,
                "release_lead_vs_baseline_p10_s": (
                    _quantile(leads, 0.1) if arm != "baseline" else None
                ),
                "boundary_error_within_200ms": sum(1 for e in errors if abs(e) <= 0.2),
                "boundary_error_measured": len(errors),
                "boundary_error_p50_abs_s": _median([abs(e) for e in errors]),
                "edge": {
                    k: sum(t.get("edge", {}).get(k, 0) for t in scored_cont)
                    for k in ("deleted", "misplaced", "duplicated")
                },
                "adjacent_labels": dict(
                    Counter(f"{t['a_label']}/{t['b_label']}" for t in scored_cont)
                ),
            },
            "speaker_cuts": dict(cuts),
            "other_transitions": {
                kind: dict(Counter(t["outcome"] for t in items))
                for kind, items in sorted(transitions.items())
                if kind != "continuous"
            },
            "boundary_harm": {
                "unit_boundaries": dict(boundaries),
                "same_speaker_splits_per_speech_min": (
                    boundaries["same_speaker_split"] / (speech_s / 60) if speech_s else None
                ),
                "bursts": bursts,
                "mixed_unit_rate": mixed / scored_units if scored_units else None,
                "colored_mixed_units": colored_mixed,
            },
            "short_segments": {
                "count": len(short),
                "retained_mean": statistics.mean(x["retained"] for x in short) if short else None,
                "separate": sum(1 for x in short if x["separate"]),
                "identity": dict(Counter(x["identity"] for x in short)),
            },
            "returns": dict(returns),
            "presentation": {
                "readable_units": readable_units,
                **dict(presentation),
                "violations": dict(violations),
                "late_delay_p50_s": _median(delays),
            },
            "latency": {
                "turn_completion_p50_s": _median(completion),
                "turn_completion_p90_s": _quantile(completion, 0.9),
                "turn_completion_delta_vs_baseline_p50_s": (
                    _median(completion_delta) if arm != "baseline" else None
                ),
            },
            "content": {
                "error_rate": (
                    content["errors"] / content["reference"] if content["reference"] else None
                ),
                **dict(content),
            },
        }
    return summary


def _fmt(value, kind="pct"):
    if value is None:
        return "–"
    if kind == "pct":
        return f"{100 * value:.1f}%"
    if kind == "s":
        return f"{value:.2f}"
    return str(value)


def markdown(summary: dict, rows: list[dict]) -> str:
    lines = ["# Speaker benchmark results", ""]
    statuses = Counter((r["route"], r["arm"], r["status"]) for r in rows)
    lines += ["## Run status", "", "| Route | Arm | Status | Cases |", "| --- | --- | --- | ---: |"]
    for (route, arm, status), count in sorted(statuses.items()):
        lines.append(f"| {route} | {arm} | {status} | {count} |")
    lines += [
        "",
        "## Early boundaries (continuous transfers)",
        "",
        "| Route / arm | Transfers | Separated | Early release | Misplaced boundary | Merged | Speaker cut at transfer | Cut detection p50 / p90 (s) | Pred. latency p50 / p90 (s) | Lead vs baseline p50 (s) | Boundary ±200 ms | Edge del/misplaced/dup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for name, s in summary.items():
        e = s["early_boundary"]
        lines.append(
            f"| {name} | {e['continuous_transfers']} | {e['separated']} | {e['early_release']} | {e['misplaced_boundary']} | {e['merged']} | "
            f"{e['speaker_cut']} | {_fmt(e['cut_detection_p50_s'], 's')} / {_fmt(e['cut_detection_p90_s'], 's')} | "
            f"{_fmt(e['predecessor_latency_p50_s'], 's')} / {_fmt(e['predecessor_latency_p90_s'], 's')} | "
            f"{_fmt(e['release_lead_vs_baseline_p50_s'], 's')} | {e['boundary_error_within_200ms']}/{e['boundary_error_measured']} | "
            f"{e['edge']['deleted']}/{e['edge']['misplaced']}/{e['edge']['duplicated']} |"
        )
    lines += [
        "",
        "## Model evidence versus policy action (continuous transfers)",
        "",
        "Evidence is supported when two consecutive frames within 1 s after the successor onset show a clear slot other than the predecessor's (top ≥ 0.8, runner-up ≤ 0.2). This diagnostic is harness-defined, not the product rule.",
        "",
        "| Route / arm | Transfers | Evidence supported | Evidence lead p50 (s) | Not separated, evidence before region end | Not separated, no evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in summary.items():
        e = s["early_boundary"]
        lines.append(
            f"| {name} | {e['continuous_transfers']} | {e['evidence_supported']} | {_fmt(e['evidence_lead_p50_s'], 's')} | "
            f"{e['missed_with_evidence']} | {e['missed_without_evidence']} |"
        )
    lines += [
        "",
        "## Speaker-triggered cuts (batch routes)",
        "",
        "| Route / arm | Cuts | At continuous | At overlap | At contained | At paused | Elsewhere (no transfer within 0.5 s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in summary.items():
        c = s["speaker_cuts"]
        lines.append(
            f"| {name} | {c.get('total', 0)} | {c.get('at_continuous', 0)} | {c.get('at_overlap', 0)} | "
            f"{c.get('at_contained', 0)} | {c.get('at_paused', 0)} | {c.get('elsewhere', 0)} |"
        )
    lines += [
        "",
        "## Attribution (word-weighted, final presented assignment)",
        "",
        "| Route / arm | Recognized tokens | Colored correct | Colored wrong | Overflow | Gray | Mixed units | Colored mixed | Returns same/diff/gray |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, s in summary.items():
        a, h, r = s["attribution"], s["boundary_harm"], s["returns"]
        lines.append(
            f"| {name} | {a['recognized_tokens']} | {_fmt(a['colored_correct'])} | {_fmt(a['colored_wrong'])} | {_fmt(a['overflow'])} | "
            f"{_fmt(a['gray'])} | {_fmt(h['mixed_unit_rate'])} | {h['colored_mixed_units']} | "
            f"{r.get('same', 0)}/{r.get('different', 0)}/{r.get('gray', 0)} |"
        )
    lines += [
        "",
        "## Boundary harm, short turns, presentation and latency",
        "",
        "| Route / arm | Same-speaker splits / speech min | Bursts | Short turns kept/separate/correct | Identified before readable | Late ≤500 ms / ≤1 s | Stayed gray | Violations | Turn completion p50 / p90 (s) | Δ vs baseline p50 (s) | Token error rate |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for name, s in summary.items():
        h, sh, p, lat, c = (
            s["boundary_harm"],
            s["short_segments"],
            s["presentation"],
            s["latency"],
            s["content"],
        )
        kept = "–" if sh["retained_mean"] is None else f"{100 * sh['retained_mean']:.0f}%"
        lines.append(
            f"| {name} | {_fmt(h['same_speaker_splits_per_speech_min'], 's')} | {h['bursts']} | "
            f"{kept}/{sh['separate']}/{sh['identity'].get('correct', 0)} of {sh['count']} | {p.get('identified_before_readable', 0)} | "
            f"{p.get('late_within_500ms', 0)} / {p.get('late_within_1s', 0)} | {p.get('stayed_gray', 0)} | "
            f"{json.dumps(p['violations']) if p['violations'] else '0'} | "
            f"{_fmt(lat['turn_completion_p50_s'], 's')} / {_fmt(lat['turn_completion_p90_s'], 's')} | "
            f"{_fmt(lat['turn_completion_delta_vs_baseline_p50_s'], 's')} | {_fmt(c['error_rate'])} |"
        )
    lines += [
        "",
        "## Classification parameters",
        "",
        f"G = effective Peer VAD hangover per run (default {DEFAULT_GAP_S} s). Other parameters: `{json.dumps(PARAMETERS)}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = score_runs(root)
    summary = summarize(rows)
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (root / "report.md").write_text(markdown(summary, rows), encoding="utf-8")
    statuses = [
        {k: r.get(k) for k in ("route", "arm", "repetition", "case_id", "status", "reason")}
        for r in rows
    ]
    (root / "statuses.json").write_text(json.dumps(statuses, indent=1) + "\n", encoding="utf-8")
    print(markdown(summary, rows))


if __name__ == "__main__":
    main()
