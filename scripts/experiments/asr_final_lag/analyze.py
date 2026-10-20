from __future__ import annotations

import gzip
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

TAIL_CHARS = 40
TOLERANCE = 0
END_EXCLUSION_S = 2.0
GAP_FINALIZE_OFFSET_S = 0.10
DEFAULT_DECIDER_RTT_S = 0.25
PUNCT_WINDOW_S = 8.0
PUNCT_SLACK = 3
TERMINAL = ".?!。？！…"
CLOSERS = "\"'”’」』)）"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def _index_normalized(raw: str) -> tuple[str, list[int]]:
    chars, positions = [], []
    for i, ch in enumerate(raw):
        for c in unicodedata.normalize("NFKC", ch).lower():
            if unicodedata.category(c)[0] in "LN":
                chars.append(c)
                positions.append(i)
    return "".join(chars), positions


def _terminal_after(raw: str, positions: list[int], index: int) -> bool:
    if not 0 < index <= len(positions):
        return False
    j = positions[index - 1] + 1
    while j < len(raw) and (raw[j].isspace() or raw[j] in CLOSERS):
        j += 1
    if j + 1 < len(raw) and raw[j] == "." and raw[j + 1].isdigit():
        return False
    return j < len(raw) and raw[j] in TERMINAL


def _terminal_indices(raw: str) -> tuple[str, list[int]]:
    norm, positions = _index_normalized(raw)
    return norm, [i for i in range(1, len(norm) + 1) if _terminal_after(raw, positions, i)]


def _semi_global_end(hyp: str, ref: str) -> tuple[float, int]:
    n = len(ref)
    ref_codes = np.fromiter((ord(c) for c in ref), dtype=np.int64, count=n)
    idx = np.arange(n + 1, dtype=np.float64)
    prev = np.zeros(n + 1)
    for ch in hyp:
        tmp = np.empty(n + 1)
        tmp[0] = prev[0] + 1
        tmp[1:] = np.minimum(prev[:-1] + (ref_codes != ord(ch)), prev[1:] + 1)
        prev = np.minimum.accumulate(tmp - idx) + idx
    end = int(np.argmin(prev))
    return float(prev[end]), end


class _Coverage:
    def __init__(self, ref: str) -> None:
        self.ref = ref
        self.value = 0
        self.length = 0

    def update(self, hyp: str) -> int:
        if not hyp:
            self.value, self.length = 0, 0
            return 0
        tail = hyp[-TAIL_CHARS:]
        delta = len(hyp) - self.length
        lo = max(0, self.value - 60 - max(0, -delta) * 2)
        hi = min(len(self.ref), self.value + 2 * max(0, delta) + 80)
        if self.length == 0 or delta > TAIL_CHARS * 4:
            lo, hi = 0, len(self.ref)
        cost, end = _semi_global_end(tail, self.ref[lo:hi])
        if cost <= 0.5 * len(tail):
            self.value = lo + end
        self.length = len(hyp)
        return self.value


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
        "max": round(float(array.max()), 3),
    }


def speech_end(sentence: dict) -> float:
    return sentence.get("speech_end_s", sentence["end_s"])


def _pauses(sentences: list[dict]) -> list[float]:
    return [
        nxt.get("speech_start_s", nxt["start_s"]) - speech_end(cur)
        for cur, nxt in zip(sentences, sentences[1:])
    ]


def _hyp_index(opcodes: list, ref_index: int, hyp_len: int) -> int:
    for tag, i1, i2, j1, j2 in opcodes:
        if i1 <= ref_index < i2:
            return j1 if tag == "delete" else j1 + min(ref_index - i1, j2 - j1)
    return hyp_len


def _map(src: str, dst: str, indices: list[int]) -> list[int]:
    opcodes = [op for op in Levenshtein.opcodes(src, dst).as_list() if op[0] != "insert"]
    return [_hyp_index(opcodes, index, len(dst)) for index in indices]


def load_run(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_run(reference: dict, run: dict) -> dict:
    sentences = reference["sentences"]
    ref = ""
    ends = []
    for sentence in sentences:
        ref += normalize(sentence["text"])
        ends.append(len(ref))
    final_cov, partial_cov = _Coverage(ref), _Coverage(ref)
    final, tail = "", ""
    series = []
    hyps = []
    raws = []
    for event in run["events"]:
        final += event.get("fa", "")
        tail = event.get("tail", tail)
        raws.append(final + tail)
        hyps.append(normalize(raws[-1]))
        series.append(
            (
                event["t"],
                final_cov.update(normalize(final)),
                partial_cov.update(hyps[-1]),
            )
        )
    hypothesis = normalize(final)
    settled_ends = _map(ref, hypothesis, ends)

    rows = []
    for k, sentence in enumerate(sentences[:-1]):
        target = ends[k] - TOLERANCE
        first_index = next((i for i, (_, _, p) in enumerate(series) if p >= target), None)
        first_partial = None if first_index is None else series[first_index][0]
        first_final = next((t for t, f, _ in series if f >= target), None)
        stable, release_diff = None, None
        for i, (t, _, p) in enumerate(series):
            if p >= target and all(q >= target for _, _, q in series[i:]):
                stable = t
                lo = settled_ends[k - 1] if k else 0
                hi = settled_ends[k]
                window = hypothesis[: hi + TAIL_CHARS]
                a, b = _map(window, hyps[i], [lo, hi])
                release_diff = round(
                    Levenshtein.distance(hyps[i][a:b], hypothesis[lo:hi]) / max(1, hi - lo), 4
                )
                break
        punct = None
        for i in range(first_index if first_index is not None else len(series), len(series)):
            if series[i][0] > first_partial + PUNCT_WINDOW_S:
                break
            norm, positions = _index_normalized(raws[i])
            cost, index = _semi_global_end(ref[max(0, ends[k] - TAIL_CHARS) : ends[k]], norm)
            if cost <= 0.5 * min(TAIL_CHARS, ends[k]) and any(
                _terminal_after(raws[i], positions, index + d) for d in range(PUNCT_SLACK + 1)
            ):
                punct = series[i][0]
                break
        end_s = speech_end(sentence)
        rows.append(
            {
                "k": k,
                "end_s": end_s,
                "partial_first": None if first_partial is None else round(first_partial - end_s, 3),
                "partial_stable": None if stable is None else round(stable - end_s, 3),
                "final": None if first_final is None else round(first_final - end_s, 3),
                "release_diff": release_diff,
                "punct": None if punct is None else round(punct - end_s, 3),
            }
        )

    horizon = reference["duration_s"] - END_EXCLUSION_S
    final_words = [w for w in run["final_words"] if w["end_s"] <= horizon]
    by_text: dict[str, list[dict]] = {}
    for word in run["partial_words"]:
        by_text.setdefault(word["w"], []).append(word)
    partial_lags = []
    for word in final_words:
        first = min(
            (
                p["t"]
                for p in by_text.get(word["w"], ())
                if abs(p["end_s"] - word["end_s"]) < 0.3 and p["t"] <= word["t"]
            ),
            default=None,
        )
        if first is not None:
            partial_lags.append(first - word["end_s"])

    finalize_times = [c["t"] for c in run["controls"] if c["name"] == "finalize"]
    ack_times = [c["t"] for c in run["controls"] if c["name"] == "fin_ack"]
    ack_lags = []
    for sent in finalize_times:
        later = [t for t in ack_times if t >= sent]
        if later:
            ack_lags.append(later[0] - sent)

    ref_terminals = []
    for k, sentence in enumerate(sentences):
        offset = ends[k - 1] if k else 0
        ref_terminals += [offset + i for i in _terminal_indices(sentence["text"])[1]]
    norm_final, final_terminals = _terminal_indices(final)
    boundaries = _map(ref, norm_final, sorted(set(ref_terminals + ends)))
    extra_ends = sum(1 for i in final_terminals if all(abs(i - b) > 2 for b in boundaries))

    counts = {"insert": 0, "delete": 0, "replace": 0}
    for op in Levenshtein.editops(ref, hypothesis):
        counts[op.tag] += 1
    return {
        "provider": run["provider"],
        "variant": run["variant"],
        "lang": run["lang"],
        "error": run["error"],
        "connect_s": run.get("connect_s"),
        "unit": reference.get("unit", "sentence"),
        "sentence_pause": _stats(_pauses(sentences)),
        "cer": round(Levenshtein.distance(ref, hypothesis) / max(1, len(ref)), 4),
        "edit_counts": counts,
        "final_events": sum(1 for event in run["events"] if event.get("fa")),
        "partial_events": sum(1 for event in run["events"] if event.get("tail")),
        "sentence_partial_first": _stats(
            [r["partial_first"] for r in rows if r["partial_first"] is not None]
        ),
        "sentence_partial_stable": _stats(
            [r["partial_stable"] for r in rows if r["partial_stable"] is not None]
        ),
        "sentence_final": _stats([r["final"] for r in rows if r["final"] is not None]),
        "sentence_final_missing": sum(1 for r in rows if r["final"] is None),
        "release_diff": _stats([r["release_diff"] for r in rows if r["release_diff"] is not None]),
        "release_changed": sum(1 for r in rows if r["release_diff"]),
        "sentence_punct": _stats([r["punct"] for r in rows if r["punct"] is not None]),
        "punct_missing": sum(
            1 for r in rows if r["partial_first"] is not None and r["punct"] is None
        ),
        "extra_ends": extra_ends,
        "word_final_lag": _stats([w["t"] - w["end_s"] for w in final_words]),
        "word_partial_first_lag": _stats(partial_lags),
        "finalize_ack_lag": _stats(ack_lags),
        "finalize_count": len(finalize_times),
        "tasks": _task_summary((run.get("provider_config") or {}).get("tasks") or []),
        "sentences": rows,
    }


def _task_summary(tasks: list[dict]) -> dict | None:
    if not tasks:
        return None
    usage: dict[str, float] = {}
    for task in tasks:
        for key, value in (task.get("usage") or {}).items():
            if isinstance(value, int | float):
                usage[key] = usage.get(key, 0) + value
    return {
        "count": len(tasks),
        "failed": sum(1 for t in tasks if t.get("error") or t.get("t_finished") is None),
        "audio_s": round(sum(t["end_s"] - t["start_s"] for t in tasks), 2),
        "finish_lag": _stats(
            [
                t["t_finished"] - t["t_finish"]
                for t in tasks
                if t.get("t_finished") is not None and t.get("t_finish") is not None
            ]
        ),
        "usage": usage,
    }


def _fmt(stat: dict) -> str:
    if not stat["n"]:
        return "-"
    return f"{stat['p50']:.2f} / {stat['p90']:.2f}"


def _release(analysis: dict) -> str:
    rows = [r["release_diff"] for r in analysis["sentences"] if r["release_diff"] is not None]
    if not rows:
        return "-"
    return f"{sum(rows) / len(rows):.3f} ({analysis['release_changed']}/{len(rows)})"


def _partial_path(natural: dict, gap: dict | None, decider_rtt_s: float) -> dict:
    ack = gap["finalize_ack_lag"]["p50"] if gap else None
    natural_final, release, finalized = [], [], []
    for row in natural["sentences"]:
        if row["final"] is None or row["partial_stable"] is None:
            continue
        natural_final.append(row["final"])
        release.append(max(row["partial_stable"], GAP_FINALIZE_OFFSET_S) + decider_rtt_s)
        if ack is not None:
            finalized.append(release[-1] + ack)
    gap_final = (
        [row["final"] for row in gap["sentences"] if row["final"] is not None] if gap else []
    )
    return {
        "natural_final": _stats(natural_final),
        "finalize_at_gap_final": _stats(gap_final),
        "partial_release": _stats(release),
        "partial_finalize": _stats(finalized),
        "punct_rule": natural["sentence_punct"],
        "release_saving": _stats([a - b for a, b in zip(natural_final, release)]),
    }


def summarize(results_dir: Path, decider_rtt_s: float = DEFAULT_DECIDER_RTT_S) -> list[dict]:
    references = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results_dir / "reference").glob("*.json"))
    }
    analyses = []
    for path in sorted((results_dir / "runs").glob("*.json*")):
        run = load_run(path)
        analyses.append(analyze_run(references[run["lang"]], run))
    (results_dir / "summary.json").write_text(
        json.dumps(analyses, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (results_dir / "summary.md").write_text(
        render_markdown(analyses, decider_rtt_s), encoding="utf-8"
    )
    return analyses


def render_markdown(analyses: list[dict], decider_rtt_s: float) -> str:
    lines = [
        "# ASR partial/final timing",
        "",
        "Times are seconds after the reference sentence's speech end (audio time = wall time "
        "since the first chunk was sent). The speech end is the last Silero VAD speech frame "
        "of the sentence; provider word end timestamps are typically 0.1-0.3 s earlier. Cells "
        "show `p50 / p90`. Sentence rows exclude the last sentence of each recording; word "
        f"rows exclude the last {END_EXCLUSION_S:.0f} s.",
        "",
        "- `partial first`: the partial text first covers the sentence end.",
        "- `partial stable`: the partial text covers the sentence end and never retracts.",
        "- `final`: final text covers the sentence end (no forced finalize unless the variant "
        "says so).",
        "- `word final` / `word partial`: provider word/token end timestamps to final or first "
        "partial arrival.",
        "- `release diff`: character error rate of the partial text of a sentence at its "
        "`partial stable` time against the final text of the same sentence; mean and the "
        "number of sentences whose partial differs.",
        "- `punct (missing)`: the partial text first shows terminal punctuation at the sentence "
        f"end, within {PUNCT_SLACK} characters after the aligned reference end (searched for "
        f"{PUNCT_WINDOW_S:.0f} s after `partial first`).",
        "- `extra ends`: terminal punctuation in the final text that is not at a reference "
        "sentence end or at terminal punctuation inside a reference sentence.",
        "- `fin ack`: finalize request to the provider's finalize acknowledgement.",
        "",
    ]
    for lang in sorted({a["lang"] for a in analyses}):
        first = next(a for a in analyses if a["lang"] == lang)
        note = (
            "Reference units are caption lines, not sentences: sentence columns measure "
            "caption line ends, and the CER reference is the uploaded caption text."
            if first.get("unit") == "caption"
            else "Silence between sentences (speech end to next speech start): "
            f"{_fmt(first['sentence_pause'])} s."
        )
        lines += [
            f"## {lang}",
            "",
            note,
            "",
            "| provider | variant | error | CER | finals | partial first | partial stable "
            "| release diff | punct (missing) | extra ends | final (missing) | word final "
            "| word partial | fin ack | connect s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for a in sorted(
            (a for a in analyses if a["lang"] == lang), key=lambda a: (a["provider"], a["variant"])
        ):
            error = (a["error"] or "").replace("|", "/")[:60]
            lines.append(
                f"| {a['provider']} | {a['variant']} | {error} | {a['cer']:.3f} "
                f"| {a['final_events']} | {_fmt(a['sentence_partial_first'])} "
                f"| {_fmt(a['sentence_partial_stable'])} | {_release(a)} "
                f"| {_fmt(a['sentence_punct'])} ({a['punct_missing']}) | {a['extra_ends']} "
                f"| {_fmt(a['sentence_final'])} "
                f"({a['sentence_final_missing']}) | {_fmt(a['word_final_lag'])} "
                f"| {_fmt(a['word_partial_first_lag'])} | {_fmt(a['finalize_ack_lag'])} "
                f"| {a['connect_s'] if a['connect_s'] is not None else '-'} |"
            )
        lines.append("")

    segmented = [a for a in analyses if a["variant"].startswith(("cut_", "end_"))]
    if segmented:
        lines += [
            "## Cut and end variants",
            "",
            "- `deleted` / `inserted`: character edit operations against the reference text "
            "(lost or duplicated words show up here).",
            "- `cuts`: finalize requests (`finish-task` for Qwen Audio, `audio_stream_end` for "
            "Gemini). `tasks`, `audio sent` and `task finish` (finish-task to task-finished, "
            "p50 / p90) are Qwen Audio only.",
            "",
            "| data | provider | variant | CER | deleted | inserted | cuts | tasks | audio sent s "
            "| task finish | usage |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        rows = [a for a in analyses if a["provider"] in {s["provider"] for s in segmented}]
        for a in sorted(rows, key=lambda a: (a["lang"], a["provider"], a["variant"])):
            tasks = a.get("tasks") or {}
            usage = ", ".join(f"{k} {v:g}" for k, v in (tasks.get("usage") or {}).items())
            lines.append(
                f"| {a['lang']} | {a['provider']} | {a['variant']} | {a['cer']:.3f} "
                f"| {a['edit_counts']['delete']} | {a['edit_counts']['insert']} "
                f"| {a.get('finalize_count', '-')} | {tasks.get('count', '-')} "
                f"| {tasks.get('audio_s', '-')} "
                f"| {_fmt(tasks['finish_lag']) if tasks else '-'} | {usage or '-'} |"
            )
        lines.append("")

    lines += [
        "## Final-only vs partial decision",
        "",
        f"- `partial release` = max(partial stable, {GAP_FINALIZE_OFFSET_S:.2f} s) + decider RTT "
        f"({decider_rtt_s:.2f} s): translation starts from the partial text of the sentence.",
        "- `partial + finalize` = `partial release` + median finalize ack of the `finalize_gap` "
        "run: translation waits for a forced final.",
        f"- `finalize at gap` = measured final lag when finalize is sent "
        f"{GAP_FINALIZE_OFFSET_S:.2f} s after each sentence's speech end.",
        "- `punct rule` = `punct` of the natural run: translation starts when the partial text "
        "punctuates the sentence end, without a decider.",
        "- `saving` = natural final - partial release.",
        "",
        "| provider | lang | natural final | finalize at gap | punct rule | partial release "
        "| partial + finalize | saving |",
        "|---|---|---|---|---|---|---|---|",
    ]
    index = {(a["provider"], a["variant"], a["lang"]): a for a in analyses}
    for provider, variant, lang in sorted(index):
        analysis = index[(provider, variant, lang)]
        if (
            variant != "natural"
            or not analysis.get("partial_events")
            or analysis.get("unit") == "caption"
        ):
            continue
        path = _partial_path(
            index[(provider, variant, lang)],
            index.get((provider, "finalize_gap", lang)),
            decider_rtt_s,
        )
        lines.append(
            f"| {provider} | {lang} | {_fmt(path['natural_final'])} "
            f"| {_fmt(path['finalize_at_gap_final'])} | {_fmt(path['punct_rule'])} "
            f"| {_fmt(path['partial_release'])} "
            f"| {_fmt(path['partial_finalize'])} | {_fmt(path['release_saving'])} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    rtt = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DECIDER_RTT_S
    summarize(target, rtt)
    print(target / "summary.md")
