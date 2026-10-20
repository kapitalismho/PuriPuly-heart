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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


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
    for event in run["events"]:
        final += event.get("fa", "")
        tail = event.get("tail", tail)
        series.append(
            (
                event["t"],
                final_cov.update(normalize(final)),
                partial_cov.update(normalize(final + tail)),
            )
        )

    rows = []
    for k, sentence in enumerate(sentences[:-1]):
        target = ends[k] - TOLERANCE
        first_partial = next((t for t, _, p in series if p >= target), None)
        first_final = next((t for t, f, _ in series if f >= target), None)
        stable = None
        for i, (t, _, p) in enumerate(series):
            if p >= target and all(q >= target for _, _, q in series[i:]):
                stable = t
                break
        end_s = sentence["end_s"]
        rows.append(
            {
                "k": k,
                "end_s": end_s,
                "partial_first": None if first_partial is None else round(first_partial - end_s, 3),
                "partial_stable": None if stable is None else round(stable - end_s, 3),
                "final": None if first_final is None else round(first_final - end_s, 3),
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

    hypothesis = normalize(final)
    counts = {"insert": 0, "delete": 0, "replace": 0}
    for op in Levenshtein.editops(ref, hypothesis):
        counts[op.tag] += 1
    return {
        "provider": run["provider"],
        "variant": run["variant"],
        "lang": run["lang"],
        "error": run["error"],
        "connect_s": run.get("connect_s"),
        "cer": round(Levenshtein.distance(ref, hypothesis) / max(1, len(ref)), 4),
        "edit_counts": counts,
        "final_events": sum(1 for event in run["events"] if event.get("fa")),
        "sentence_partial_first": _stats(
            [r["partial_first"] for r in rows if r["partial_first"] is not None]
        ),
        "sentence_partial_stable": _stats(
            [r["partial_stable"] for r in rows if r["partial_stable"] is not None]
        ),
        "sentence_final": _stats([r["final"] for r in rows if r["final"] is not None]),
        "sentence_final_missing": sum(1 for r in rows if r["final"] is None),
        "word_final_lag": _stats([w["t"] - w["end_s"] for w in final_words]),
        "word_partial_first_lag": _stats(partial_lags),
        "finalize_ack_lag": _stats(ack_lags),
        "sentences": rows,
    }


def _fmt(stat: dict) -> str:
    if not stat["n"]:
        return "-"
    return f"{stat['p50']:.2f} / {stat['p90']:.2f}"


def _partial_path(natural: dict, gap: dict, decider_rtt_s: float) -> dict:
    ack = gap["finalize_ack_lag"]["p50"]
    natural_final, estimated, gap_final = [], [], []
    for n_row, g_row in zip(natural["sentences"], gap["sentences"]):
        if n_row["final"] is None or n_row["partial_stable"] is None or ack is None:
            continue
        natural_final.append(n_row["final"])
        estimated.append(max(n_row["partial_stable"], GAP_FINALIZE_OFFSET_S) + decider_rtt_s + ack)
        if g_row["final"] is not None:
            gap_final.append(g_row["final"])
    return {
        "natural_final": _stats(natural_final),
        "finalize_at_gap_final": _stats(gap_final),
        "estimated_partial_path": _stats(estimated),
        "estimated_saving": _stats([a - b for a, b in zip(natural_final, estimated)]),
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
        "since the first chunk was sent). Cells show `p50 / p90`. Sentence rows exclude the "
        "last sentence of each recording; word rows exclude the last "
        f"{END_EXCLUSION_S:.0f} s.",
        "",
        "- `partial first`: the partial text first covers the sentence end.",
        "- `partial stable`: the partial text covers the sentence end and never retracts.",
        "- `final`: final text covers the sentence end (no forced finalize unless the variant "
        "says so).",
        "- `word final` / `word partial`: provider word/token end timestamps to final or first "
        "partial arrival.",
        "- `fin ack`: finalize request to the provider's finalize acknowledgement.",
        "",
    ]
    for lang in sorted({a["lang"] for a in analyses}):
        lines += [
            f"## {lang}",
            "",
            "| provider | variant | error | CER | finals | partial first | partial stable "
            "| final (missing) | word final | word partial | fin ack | connect s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for a in sorted(
            (a for a in analyses if a["lang"] == lang), key=lambda a: (a["provider"], a["variant"])
        ):
            error = (a["error"] or "").replace("|", "/")[:60]
            lines.append(
                f"| {a['provider']} | {a['variant']} | {error} | {a['cer']:.3f} "
                f"| {a['final_events']} | {_fmt(a['sentence_partial_first'])} "
                f"| {_fmt(a['sentence_partial_stable'])} | {_fmt(a['sentence_final'])} "
                f"({a['sentence_final_missing']}) | {_fmt(a['word_final_lag'])} "
                f"| {_fmt(a['word_partial_first_lag'])} | {_fmt(a['finalize_ack_lag'])} "
                f"| {a['connect_s'] if a['connect_s'] is not None else '-'} |"
            )
        lines.append("")

    lines += [
        "## Final-only vs partial decision + forced finalize",
        "",
        "`estimated partial path` = max(partial stable, "
        f"{GAP_FINALIZE_OFFSET_S:.2f} s) + decider RTT ({decider_rtt_s:.2f} s) + median "
        "finalize ack of the `finalize_gap` run. `finalize at gap` is the measured final "
        f"lag when finalize is sent {GAP_FINALIZE_OFFSET_S:.2f} s after each sentence end.",
        "",
        "| provider | lang | natural final | finalize at gap | estimated partial path "
        "| estimated saving |",
        "|---|---|---|---|---|---|",
    ]
    index = {(a["provider"], a["variant"], a["lang"]): a for a in analyses}
    for provider, variant, lang in sorted(index):
        if variant != "natural" or (provider, "finalize_gap", lang) not in index:
            continue
        path = _partial_path(
            index[(provider, variant, lang)], index[(provider, "finalize_gap", lang)], decider_rtt_s
        )
        lines.append(
            f"| {provider} | {lang} | {_fmt(path['natural_final'])} "
            f"| {_fmt(path['finalize_at_gap_final'])} | {_fmt(path['estimated_partial_path'])} "
            f"| {_fmt(path['estimated_saving'])} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    rtt = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DECIDER_RTT_S
    summarize(target, rtt)
    print(target / "summary.md")
