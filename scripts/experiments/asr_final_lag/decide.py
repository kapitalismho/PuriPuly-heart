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
import asyncio
import bisect
import datetime as dt
import gzip
import json
import math
import sys
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import analyze
import envkeys

URL = "https://openrouter.ai/api/alpha/decisions"
KEY_CANDIDATES = {"openrouter": ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPENROUTER_KEY")}
DEFAULT_MODELS = (
    "typesafe/jev-1.14",
    "inception/mercury-decide",
    "perplexity/pplx-decider-v1.2",
    "perplexity/pplx-decider-flash-v1.2",
)
PROVIDERS = ("soniox", "deepgram", "gemini", "qwen")
DATA = ("ko_kr", "ja_jp", "en_us", "ko_conv", "ja_conv", "en_conv")
LANGUAGE = {"ko": "Korean", "ja": "Japanese", "en": "English"}
WINDOW_CHARS = {"Korean": 80, "Japanese": 80, "English": 240}
RIGHT_CHARS = {"Korean": 2, "Japanese": 3, "English": 3}
CANDIDATE_MIN = 0.5
PROPOSE_THRESHOLDS = (0.5, 0.7, 0.9)
VERIFY_THRESHOLDS = (0.5, 0.7, 0.9)
SILENCE_S = 0.8
DEBOUNCE_S = (0.3, 0.6)
LABEL_SLACK = 1
END_EXCLUSION_S = 2.0
MAX_RETRIES = 5
COMMAS = ",、，"
MAX_MARKERS = 12
LOCATE_SLACK = 150
FAKE_RTT_S = 0.3

PROPOSE_INSTRUCTIONS = (
    "The transcript is live speech recognition that stops at the latest recognized word. "
    "Numbered markers such as [1] sit between the newest words. Which marker is the last point "
    "where the speaker has finished a sentence or a self-contained clause, so that an "
    "interpreter could translate everything before it right now? The speaker may still be "
    "talking after the last marker."
)
VERIFY = {
    "type": "noul",
    "instructions": (
        "`before` is live speech recognition and `after` is what the speaker said next. Does "
        "`before` end exactly where a sentence or a self-contained clause ends, so that `before` "
        "can be translated without the words in `after`?"
    ),
    "criteria": {
        "true": "`before` ends a complete sentence or clause; `after` starts a new one.",
        "false": "`after` continues the sentence or clause that `before` ends with.",
    },
}


def strip_text(text: str) -> str:
    kept = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        category = unicodedata.category(ch)
        if category[0] in "LN" or ch == "'":
            kept.append(ch)
        elif ch.isspace() or category[0] in "PSZ":
            kept.append(" ")
    return " ".join("".join(kept).split())


def split_after(text: str, count: int) -> tuple[str, str]:
    seen = 0
    for i, ch in enumerate(text):
        if count == 0:
            return "", text
        seen += len(analyze.normalize(ch))
        if seen >= count:
            return text[: i + 1], text[i + 1 :]
    return text, ""


def propose_question(markers: int) -> dict:
    criteria = {
        "none": "No marker ends a sentence or self-contained clause; the speaker is still inside one."
    }
    for k in range(1, markers + 1):
        criteria[str(k)] = f"Everything before [{k}] ends a sentence or self-contained clause."
    return {"type": "choice", "instructions": PROPOSE_INSTRUCTIONS, "criteria": criteria}


def unit_ends(text: str, language: str, since: int) -> list[tuple[int, int]]:
    ends, count = [], 0
    for i, ch in enumerate(text):
        size = len(analyze.normalize(ch))
        count += size
        if not size:
            continue
        boundary = language == "Japanese" or i + 1 == len(text) or text[i + 1] == " "
        if boundary and count > since:
            ends.append((count, i + 1))
    return ends[-MAX_MARKERS:]


def marked_window(text: str, ends: list[tuple[int, int]], language: str) -> str:
    pieces, last = [], 0
    for k, (_, cut) in enumerate(ends, 1):
        pieces += [text[last:cut], f" [{k}]"]
        last = cut
    pieces.append(text[last:])
    return window("".join(pieces), language, extra=5 * len(ends))


def window(text: str, language: str, extra: int = 0) -> str:
    limit = WINDOW_CHARS[language] + extra
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    if language == "English" and " " in cut:
        cut = cut.split(" ", 1)[1]
    return "… " + cut


@dataclass
class Stream:
    data: str
    provider: str
    language: str
    duration_s: float
    times: list[float]
    texts: list[str]
    raws: list[str]
    final_raw: str
    final_norm: str
    char_time: np.ndarray
    terminals: list[int]
    commas: list[int]
    final_growth: list[tuple[float, int]]

    def at(self, t: float) -> int:
        return bisect.bisect_right(self.times, t) - 1

    def next_change(self, t: float) -> float:
        i = bisect.bisect_right(self.times, t)
        return self.times[i] if i < len(self.times) else math.inf

    def to_final(self, text: str) -> int:
        return self.locate(analyze.normalize(text))

    def locate(self, norm: str) -> int:
        if not norm:
            return 0
        tail = norm[-analyze.TAIL_CHARS :]
        lo = max(0, len(norm) - len(tail) - LOCATE_SLACK)
        hi = min(len(self.final_norm), len(norm) + LOCATE_SLACK)
        _, end = analyze._semi_global_end(tail, self.final_norm[lo:hi])
        return lo + end


@dataclass
class Candidate:
    ready_t: float
    text: str
    count: int
    p: float
    call: int
    text_t: float = 0.0
    verify: dict | None = None
    dropped: bool = False


@dataclass
class Simulation:
    calls: list[dict] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)


def _load_run(results_dir: Path, provider: str, data: str, variant: str = "natural") -> dict | None:
    path = results_dir / "runs" / f"{provider}_{variant}_{data}.json.gz"
    return analyze.load_run(path) if path.exists() else None


def _latest_results(data: str) -> Path | None:
    for path in sorted((HERE / "results").glob("2*Z"), reverse=True):
        if (path / "runs" / f"soniox_natural_{data}.json.gz").exists():
            return path
    return None


def build_stream(
    results_dir: Path, provider: str, data: str, variant: str = "natural"
) -> Stream | None:
    run = _load_run(results_dir, provider, data, variant)
    timing = _load_run(results_dir, "soniox", data)
    if run is None or timing is None or run.get("error"):
        return None
    reference = json.loads((results_dir / "reference" / f"{data}.json").read_text("utf-8"))
    language = LANGUAGE[data[:2]]
    times, texts, raws = [], [], []
    final, tail = "", ""
    final_growth = []
    for event in run["events"]:
        final += event.get("fa", "")
        tail = event.get("tail", tail)
        if event.get("fa"):
            final_growth.append((event["t"], len(analyze.normalize(final))))
        text = strip_text(final + tail)
        if texts and text == texts[-1]:
            continue
        times.append(event["t"])
        texts.append(text)
        raws.append(final + tail)
    timing_norm, timing_ends = "", []
    for word in timing["final_words"]:
        piece = analyze.normalize(word["w"])
        timing_norm += piece
        timing_ends += [word["end_s"]] * len(piece)
    final_norm = analyze.normalize(final)
    if not final_norm or not timing_norm:
        return None
    to_timing = analyze._map(final_norm, timing_norm, list(range(len(final_norm))))
    char_time = np.array([timing_ends[min(j, len(timing_ends) - 1)] for j in to_timing])
    timing_raw = "".join(e.get("fa", "") for e in timing["events"])
    soniox_norm, soniox_terminals = analyze._terminal_indices(timing_raw)
    soniox_commas = [
        i
        for i in range(1, len(soniox_norm) + 1)
        if _comma_after(timing_raw, analyze._index_normalized(timing_raw)[1], i)
    ]
    to_final = analyze._map(soniox_norm, final_norm, sorted(set(soniox_terminals + soniox_commas)))
    mapped = dict(zip(sorted(set(soniox_terminals + soniox_commas)), to_final))
    return Stream(
        data=data,
        provider=provider,
        language=language,
        duration_s=reference["duration_s"],
        times=times,
        texts=texts,
        raws=raws,
        final_raw=final,
        final_norm=final_norm,
        char_time=char_time,
        terminals=sorted({mapped[i] for i in soniox_terminals}),
        commas=sorted({mapped[i] for i in soniox_commas} - {mapped[i] for i in soniox_terminals}),
        final_growth=final_growth,
    )


def _comma_after(raw: str, positions: list[int], index: int) -> bool:
    if not 0 < index <= len(positions):
        return False
    j = positions[index - 1] + 1
    while j < len(raw) and raw[j].isspace():
        j += 1
    return j < len(raw) and raw[j] in COMMAS


class FatalDeciderError(RuntimeError):
    pass


class Decider:
    def __init__(self, model: str, key: str | None, concurrency: int, fake: bool) -> None:
        self.model = model
        self.key = key
        self.fake = fake
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=30)

    async def ask(self, question: dict, state: dict, oracle: dict) -> dict:
        if self.fake:
            return {"answer": oracle, "rtt": FAKE_RTT_S, "tokens": 0, "cost": 0.0, "retries": 0}
        body = {"model": self.model, "state": state, "questions": {"q": question}}
        headers = {"Authorization": f"Bearer {self.key}", "X-Title": "PuriPuly decider probe"}
        retries, error, rtt = 0, None, 0.0
        while retries <= MAX_RETRIES:
            async with self.semaphore:
                started = time.perf_counter()
                try:
                    response = await self.client.post(URL, json=body, headers=headers)
                    rtt = time.perf_counter() - started
                    if response.status_code == 200:
                        payload = response.json()
                        usage = payload.get("usage") or {}
                        return {
                            "answer": payload.get("answers", {}).get("q", {}),
                            "rtt": rtt,
                            "tokens": usage.get("input_tokens"),
                            "cost": usage.get("cost"),
                            "retries": retries,
                            "served": payload.get("model"),
                        }
                    error = f"HTTP {response.status_code}: {response.text[:200]}"
                    if response.status_code in (401, 402, 403, 404):
                        raise FatalDeciderError(f"{self.model}: {error}")
                    if response.status_code not in (408, 429, 500, 502, 503, 524, 529):
                        break
                except httpx.HTTPError as exc:
                    rtt = time.perf_counter() - started
                    error = f"{type(exc).__name__}: {exc}"[:200]
            retries += 1
            await asyncio.sleep(min(8.0, 0.5 * 2**retries))
        return {"answer": {}, "rtt": rtt, "retries": retries, "error": error}

    async def close(self) -> None:
        await self.client.aclose()


def _at_label(stream: Stream, text: str) -> bool:
    position = stream.to_final(text)
    return any(abs(position - b) <= LABEL_SLACK for b in stream.terminals + stream.commas)


def _oracle_choice(stream: Stream, text: str, ends: list[tuple[int, int]]) -> dict:
    hits = [k for k, (_, cut) in enumerate(ends, 1) if _at_label(stream, text[:cut])]
    choice = str(hits[-1]) if hits else "none"
    return {"type": "choice", "choice": choice, "probabilities": {choice: 0.95}}


def _oracle_noul(stream: Stream, text: str) -> dict:
    return {"type": "noul", "noul": 0.95 if _at_label(stream, text) else 0.05}


def _choice(answer: dict) -> tuple[str | None, float | None]:
    choice = answer.get("choice")
    probability = (answer.get("probabilities") or {}).get(choice, answer.get("confidence"))
    return choice, None if probability is None else float(probability)


def _noul(answer: dict) -> float | None:
    value = answer.get("noul")
    if value is None:
        value = (answer.get("probabilities") or {}).get("true")
    return None if value is None else float(value)


def _record(result: dict, **fields) -> dict:
    keep = ("rtt", "tokens", "cost", "retries", "served", "error")
    return fields | {k: result[k] for k in keep if result.get(k) is not None}


async def simulate(
    stream: Stream, decider: Decider, progress: "_ReplayProgress | None" = None
) -> Simulation:
    sim = Simulation()
    lanes = {"propose": stream.times[0], "verify": math.inf}
    verify_idle = True
    end_t = stream.times[-1] + 2.0
    pending: deque[Candidate] = deque()
    last_proposed, last_norm, last_count = None, "", 0
    while min(lanes.values()) <= end_t:
        lane = min(lanes, key=lanes.get)
        now = lanes[lane]
        text = stream.texts[stream.at(now)]
        if lane == "propose":
            if not text or text == last_proposed:
                lanes[lane] = stream.next_change(now)
                continue
            norm = analyze.normalize(text)
            since = last_count if norm.startswith(last_norm) else 0
            last_proposed, last_norm, last_count = text, norm, len(norm)
            ends = unit_ends(text, stream.language, since)
            if not ends:
                lanes[lane] = stream.next_change(now)
                continue
            state = {
                "language": stream.language,
                "transcript": marked_window(text, ends, stream.language),
            }
            result = await decider.ask(
                propose_question(len(ends)), state, _oracle_choice(stream, text, ends)
            )
            choice, p = _choice(result["answer"])
            sim.calls.append(
                _record(
                    result,
                    lane=lane,
                    t=round(now, 4),
                    count=len(norm),
                    markers=len(ends),
                    choice=choice,
                    p=p,
                    state=state,
                )
            )
            if progress is not None:
                progress.mark(decider.model, stream, lane)
            lanes[lane] = now + result["rtt"]
            if choice not in (None, "none") and choice.isdigit() and p is not None:
                index = int(choice) - 1
                if 0 <= index < len(ends) and p >= CANDIDATE_MIN:
                    count, cut = ends[index]
                    candidate = Candidate(
                        now + result["rtt"],
                        text[:cut],
                        count,
                        p,
                        len(sim.calls) - 1,
                        text_t=stream.times[stream.at(now)],
                    )
                    sim.candidates.append(candidate)
                    pending.append(candidate)
                    if verify_idle:
                        lanes["verify"] = min(lanes["verify"], candidate.ready_t)
            continue
        while pending and pending[0].ready_t <= now:
            before, _ = split_after(text, pending[0].count)
            if analyze.normalize(before) == analyze.normalize(pending[0].text):
                break
            pending.popleft().dropped = True
        head = pending[0] if pending and pending[0].ready_t <= now else None
        if head is not None:
            before, after = split_after(text, head.count)
            if len(analyze.normalize(after)) < RIGHT_CHARS[stream.language]:
                head = None
        if head is None:
            wake = stream.next_change(now)
            if pending and pending[0].ready_t > now:
                wake = min(wake, pending[0].ready_t)
            lanes[lane] = wake
            verify_idle = True
            continue
        pending.popleft()
        state = {
            "language": stream.language,
            "before": window(before.strip(), stream.language),
            "after": after.strip(),
        }
        result = await decider.ask(VERIFY, state, _oracle_noul(stream, before))
        p = _noul(result["answer"])
        sim.calls.append(
            _record(result, lane=lane, t=round(now, 4), count=head.count, p=p, state=state)
        )
        if progress is not None:
            progress.mark(decider.model, stream, lane)
        head.verify = {"t": now + result["rtt"], "p": p}
        lanes[lane] = now + result["rtt"]
        verify_idle = False
    return sim


def _silence_time(stream: Stream, candidate: Candidate) -> float | None:
    quiet_until = candidate.text_t + SILENCE_S
    if stream.next_change(candidate.text_t) <= quiet_until:
        return None
    return max(candidate.ready_t, quiet_until)


def _releases_from_candidates(stream, sim, theta, theta_v, silence) -> list[tuple[float, int]]:
    releases = []
    for c in sim.candidates:
        if c.p < theta:
            continue
        if theta_v is None:
            releases.append((c.ready_t, stream.to_final(c.text)))
            continue
        times = []
        if c.verify and c.verify["p"] is not None and c.verify["p"] >= theta_v:
            times.append(c.verify["t"])
        quiet = _silence_time(stream, c) if silence else None
        if quiet is not None:
            times.append(quiet)
        if times:
            releases.append((min(times), stream.to_final(c.text)))
    return releases


def _punct_releases(stream: Stream) -> list[tuple[float, int]]:
    releases, seen = [], set()
    for t, raw in zip(stream.times, stream.raws):
        norm, terminals = analyze._terminal_indices(raw)
        fresh = [i for i in terminals if i not in seen]
        if not fresh:
            continue
        seen.update(terminals)
        releases += [(t, stream.locate(norm[:i])) for i in fresh]
    return releases


def _debounce_releases(stream: Stream, gap: float) -> list[tuple[float, int]]:
    releases = []
    for i, t in enumerate(stream.times):
        nxt = stream.times[i + 1] if i + 1 < len(stream.times) else math.inf
        if nxt - t >= gap and stream.texts[i]:
            releases.append((t + gap, stream.to_final(stream.texts[i])))
    return releases


def _natural_releases(stream: Stream) -> list[tuple[float, int]]:
    return [(t, count) for t, count in stream.final_growth]


def evaluate(stream: Stream, releases: list[tuple[float, int]]) -> dict:
    horizon = stream.duration_s - END_EXCLUSION_S
    ordered, last = [], 0
    for t, position in sorted(releases):
        if position > last:
            ordered.append((t, position))
            last = position
    kinds = {"sentence": 0, "clause": 0, "mid": 0}
    for _, position in ordered:
        if any(abs(position - b) <= LABEL_SLACK for b in stream.terminals):
            kinds["sentence"] += 1
        elif any(abs(position - b) <= LABEL_SLACK for b in stream.commas):
            kinds["clause"] += 1
        else:
            kinds["mid"] += 1
    labels = [
        b
        for b in stream.terminals
        if 0 < b <= len(stream.char_time) and stream.char_time[b - 1] <= horizon
    ]
    latencies, matched = [], 0
    for b in labels:
        first = next(
            (t for t, position in ordered if position >= b - LABEL_SLACK),
            None,
        )
        hit = next((t for t, position in ordered if abs(position - b) <= LABEL_SLACK), None)
        if hit is not None:
            matched += 1
        if first is not None:
            latencies.append(first - float(stream.char_time[b - 1]))
    return {
        "releases": len(ordered),
        "kinds": kinds,
        "labels": len(labels),
        "label_hits": matched,
        "label_latency": latencies,
    }


def _policies(stream: Stream, sim: Simulation) -> dict[str, list[tuple[float, int]]]:
    policies = {}
    for theta in PROPOSE_THRESHOLDS:
        policies[f"immediate p>={theta}"] = _releases_from_candidates(
            stream, sim, theta, None, False
        )
    for theta_v in VERIFY_THRESHOLDS:
        policies[f"verify v>={theta_v}"] = _releases_from_candidates(
            stream, sim, CANDIDATE_MIN, theta_v, False
        )
        policies[f"verify v>={theta_v} +silence"] = _releases_from_candidates(
            stream, sim, CANDIDATE_MIN, theta_v, True
        )
    return policies


def baselines(stream: Stream) -> dict[str, list[tuple[float, int]]]:
    result = {"natural final": _natural_releases(stream), "punct rule": _punct_releases(stream)}
    for gap in DEBOUNCE_S:
        result[f"debounce {gap:.1f}s"] = _debounce_releases(stream, gap)
    return result


def _pool(entries: list[dict]) -> dict:
    kinds = {k: sum(e["kinds"][k] for e in entries) for k in ("sentence", "clause", "mid")}
    releases = sum(e["releases"] for e in entries)
    labels = sum(e["labels"] for e in entries)
    latency = [x for e in entries for x in e["label_latency"]]
    stats = analyze._stats(latency)
    return {
        "releases": releases,
        "sentence_share": kinds["sentence"] / releases if releases else None,
        "clause_share": kinds["clause"] / releases if releases else None,
        "mid_share": kinds["mid"] / releases if releases else None,
        "recall": sum(e["label_hits"] for e in entries) / labels if labels else None,
        "latency": stats,
    }


def _call_stats(sims: list[tuple[Stream, Simulation]]) -> dict:
    calls = [c for _, sim in sims for c in sim.calls]
    audio_h = sum(stream.duration_s for stream, _ in sims) / 3600
    rtts = [c["rtt"] for c in calls if "error" not in c]
    cost = sum(c.get("cost") or 0.0 for c in calls)
    tokens = [c["tokens"] for c in calls if c.get("tokens")]
    return {
        "calls": len(calls),
        "errors": sum(1 for c in calls if "error" in c),
        "retries": sum(c.get("retries", 0) for c in calls),
        "calls_per_min": len(calls) / (audio_h * 60) if audio_h else None,
        "rtt": analyze._stats(rtts),
        "tokens_per_call": float(np.mean(tokens)) if tokens else None,
        "cost": cost,
        "cost_per_audio_hour": cost / audio_h if audio_h else None,
    }


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _row(name: str, pooled: dict, extra: str = "") -> str:
    lat = pooled["latency"]
    latency = "-" if not lat["n"] else f"{lat['p50']:.2f} / {lat['p90']:.2f}"
    return (
        f"| {name} | {pooled['releases']} | {_pct(pooled['sentence_share'])} "
        f"| {_pct(pooled['clause_share'])} | {_pct(pooled['mid_share'])} "
        f"| {_pct(pooled['recall'])} | {latency} |{extra}"
    )


def render(summary: dict, meta: dict) -> str:
    lines = [
        "# Decision model release probe",
        "",
        "Replays recorded partial-transcript streams through decision models on OpenRouter and "
        "measures where and when each policy would release text for translation. Text sent to "
        "the models is lowercased and has punctuation removed.",
        "",
        "- Labels are sentence-final punctuation in the Soniox final transcript of the same audio "
        "(hindsight), mapped onto each provider's transcript. Commas count as clause labels.",
        f"- `sentence` / `clause` / `mid`: share of releases within {LABEL_SLACK} characters of a "
        "sentence label, of a comma label, or neither.",
        "- `recall`: share of sentence labels that some release hits.",
        "- `latency`: seconds from the audio end of a labeled sentence's last character "
        "(Soniox token timestamp) to the first release that covers it. Cells show `p50 / p90`.",
        "- Proposal: a choice question over numbered markers placed after each new word "
        "(each new character in Japanese, at most "
        f"{MAX_MARKERS}); the chosen marker becomes a candidate when its probability is at least "
        f"{CANDIDATE_MIN}.",
        "- `immediate p>=x`: release a candidate as soon as the proposal answer arrives, when "
        "its probability is at least x.",
        "- `verify v>=x`: every candidate is checked once more after "
        f"{RIGHT_CHARS['English']} English / {RIGHT_CHARS['Korean']} Korean / "
        f"{RIGHT_CHARS['Japanese']} Japanese characters of right context arrive; release when "
        "that answer is at least x. `+silence` also releases a candidate when the transcript "
        f"stays unchanged for {SILENCE_S} s.",
        "- `punct rule` releases at sentence-final punctuation in the partial transcript, which "
        "is close to how the labels were made. `natural final` releases final text as it arrives.",
        "",
    ]
    for group, title in (("fleurs", "Read speech (FLEURS)"), ("conv", "Conversation (YODAS2)")):
        block = summary["groups"].get(group)
        if not block:
            continue
        lines += [
            f"## {title}",
            "",
            "| policy | releases | sentence | clause | mid | recall | latency |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, pooled in block["baselines"].items():
            lines.append(_row(name, pooled))
        lines.append("")
        lines += [
            "| model | policy | releases | sentence | clause | mid | recall | latency |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model, policies in block["models"].items():
            for name, pooled in policies.items():
                lines.append(_row(f"{model} | {name}", pooled))
        lines.append("")
    lines += [
        "## Calls",
        "",
        "| model | calls | errors | retries | calls/min of audio | RTT p50 / p90 | tokens/call "
        "| cost | cost per audio hour |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, stats in summary["calls"].items():
        rtt = stats["rtt"]
        lines.append(
            f"| {model} | {stats['calls']} | {stats['errors']} | {stats['retries']} "
            f"| {stats['calls_per_min'] or 0:.0f} "
            f"| {'-' if not rtt['n'] else f'{rtt['p50']:.3f} / {rtt['p90']:.3f}'} "
            f"| {stats['tokens_per_call'] or 0:.0f} | ${stats['cost']:.4f} "
            f"| ${stats['cost_per_audio_hour'] or 0:.3f} |"
        )
    for model, reason in summary.get("failures", {}).items():
        lines.append(f"| {model} | stopped: {reason[:120]} | | | | | | | |")
    lines += ["", "Per-language and per-provider rows are in `summary.json`.", ""]
    return "\n".join(lines)


class _ReplayProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.calls = 0
        self.done = 0
        self.started = time.monotonic()
        self.last_print = 0.0
        self.last = ""

    def line(self) -> str:
        tail = f"  {self.last}" if self.last else ""
        return (
            f"[decide] in progress  calls={self.calls}  streams={self.done}/{self.total}  "
            f"{time.monotonic() - self.started:.0f}s{tail}"
        )

    def mark(self, model: str, stream: Stream, lane: str) -> None:
        self.calls += 1
        self.last = f"{model} {stream.data} {stream.provider} {lane}"
        self.show(self.calls == 1)

    def show(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_print < 10:
            return
        self.last_print = now
        print(self.line(), flush=True)


async def _simulate_all(streams, models, key, args, out_dir) -> dict:
    deciders = {m: Decider(m, key, args.concurrency, args.fake_decider) for m in models}
    results: dict[str, list[tuple[Stream, Simulation]]] = {m: [] for m in models}
    failures: dict[str, str] = {}
    progress = _ReplayProgress(len(streams) * len(models))

    async def one(stream: Stream, model: str) -> None:
        if model in failures:
            return
        started = time.monotonic()
        try:
            sim = await simulate(stream, deciders[model], progress)
        except FatalDeciderError as exc:
            if model not in failures:
                failures[model] = str(exc).replace(key or "<none>", "<key>")
                print(f"[decide] {model}: stopped, {failures[model]}", flush=True)
            return
        results[model].append((stream, sim))
        name = f"{stream.data}_{stream.provider}_{model.replace('/', '__')}.jsonl.gz"
        with gzip.open(out_dir / "calls" / name, "wt", encoding="utf-8") as handle:
            for call in sim.calls:
                handle.write(json.dumps(call, ensure_ascii=False) + "\n")
        errors = sum(1 for c in sim.calls if "error" in c)
        progress.done += 1
        print(
            f"[decide] {model:38} {stream.data:8} {stream.provider:8} calls={len(sim.calls):4} "
            f"errors={errors:3} {time.monotonic() - started:5.0f} s",
            flush=True,
        )

    async def preflight(model: str) -> None:
        decider = deciders[model]
        checks = (
            (
                propose_question(2),
                {"language": "English", "transcript": "so i went there [1] and then [2]"},
                {"type": "choice", "choice": "1", "probabilities": {"1": 0.9}},
            ),
            (
                VERIFY,
                {"language": "English", "before": "so i went there", "after": "and then"},
                {"type": "noul", "noul": 0.1},
            ),
        )
        try:
            for question, state, oracle in checks:
                result = await decider.ask(question, state, oracle)
                if "error" in result:
                    raise FatalDeciderError(f"{model}: {result['error']}")
                print(f"[check] {model}: {json.dumps(result['answer'])[:160]}", flush=True)
        except FatalDeciderError as exc:
            failures[model] = str(exc).replace(key or "<none>", "<key>")
            print(f"[check] {model}: stopped, {failures[model]}", flush=True)

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(10)
            progress.show(force=True)

    beat: asyncio.Task | None = None
    try:
        await asyncio.gather(*(preflight(m) for m in models))
        progress.total = len(streams) * sum(1 for model in models if model not in failures)
        progress.started = time.monotonic()
        print("[decide] checks done, replaying", flush=True)
        progress.show(force=True)
        beat = asyncio.create_task(heartbeat())
        await asyncio.gather(*(one(s, m) for s in streams for m in models))
    finally:
        if beat is not None:
            beat.cancel()
            try:
                await beat
            except asyncio.CancelledError:
                pass
        for decider in deciders.values():
            await decider.close()
    return {m: sims for m, sims in results.items() if m not in failures}, failures


def summarize(streams: list[Stream], results: dict) -> dict:
    groups: dict[str, dict] = {}
    detail = []
    for group in ("fleurs", "conv"):
        members = [s for s in streams if (s.data.endswith("_conv")) == (group == "conv")]
        member_ids = {id(s) for s in members}
        if not members:
            continue
        block = {"baselines": {}, "models": {}}
        base_evals: dict[str, list[dict]] = {}
        for stream in members:
            for name, releases in baselines(stream).items():
                entry = evaluate(stream, releases)
                base_evals.setdefault(name, []).append(entry)
                detail.append(
                    {"data": stream.data, "provider": stream.provider, "policy": name}
                    | _pool([entry])
                )
        block["baselines"] = {name: _pool(entries) for name, entries in base_evals.items()}
        for model, sims in results.items():
            evals: dict[str, list[dict]] = {}
            for stream, sim in sims:
                if id(stream) not in member_ids:
                    continue
                for name, releases in _policies(stream, sim).items():
                    entry = evaluate(stream, releases)
                    evals.setdefault(name, []).append(entry)
                    detail.append(
                        {
                            "data": stream.data,
                            "provider": stream.provider,
                            "model": model,
                            "policy": name,
                        }
                        | _pool([entry])
                    )
            block["models"][model] = {name: _pool(entries) for name, entries in evals.items()}
        groups[group] = block
    calls = {model: _call_stats(sims) for model, sims in results.items()}
    return {"groups": groups, "calls": calls, "detail": detail}


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Replay ASR partials through decision models.")
    parser.add_argument(
        "--results", nargs="*", help="ASR results folders (default: newest per data)"
    )
    parser.add_argument("--data", nargs="+", default=list(DATA), choices=DATA)
    parser.add_argument("--providers", nargs="+", default=list(PROVIDERS), choices=PROVIDERS)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--concurrency", type=int, default=6, help="in-flight requests per model")
    parser.add_argument("--env-file", action="append", help="defaults to .env and .env.local")
    parser.add_argument("--key", default=None, metavar="VARIABLE")
    parser.add_argument("--fake-decider", action="store_true", help="label oracle, no API calls")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    key = None
    if not args.fake_decider:
        env_files = (
            [Path(p) for p in args.env_file]
            if args.env_file
            else [REPO_ROOT / ".env", REPO_ROOT / ".env.local"]
        )
        overrides = {"openrouter": args.key} if args.key else {}
        keys = envkeys.resolve(KEY_CANDIDATES, env_files, overrides)
        if "openrouter" not in keys:
            names = args.key or ", ".join(KEY_CANDIDATES["openrouter"])
            raise SystemExit(f"none of {names} is set")
        print(f"[keys] openrouter: using {keys['openrouter'][0]}")
        key = keys["openrouter"][1]

    sources = {}
    for data in args.data:
        if args.results:
            found = [
                Path(p) for p in args.results if (Path(p) / "reference" / f"{data}.json").exists()
            ]
            source = found[-1] if found else None
        else:
            source = _latest_results(data)
        if source is None:
            print(f"[data] {data}: no results with natural runs, skipped")
            continue
        sources[data] = source
    streams = []
    for data, source in sources.items():
        for provider in args.providers:
            stream = build_stream(source, provider, data)
            if stream is None:
                print(f"[data] {data} {provider}: missing or failed run in {source.name}, skipped")
                continue
            streams.append(stream)
    if not streams:
        raise SystemExit("no streams to replay")

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else HERE / "results" / f"decide-{stamp}"
    (out_dir / "calls").mkdir(parents=True, exist_ok=True)
    meta = {
        "started_utc": stamp,
        "models": args.models,
        "fake_decider": args.fake_decider,
        "sources": {data: source.name for data, source in sources.items()},
        "streams": [f"{s.data}/{s.provider}" for s in streams],
        "questions": {"propose": propose_question(2), "verify": VERIFY},
        "max_markers": MAX_MARKERS,
        "window_chars": WINDOW_CHARS,
        "right_chars": RIGHT_CHARS,
        "candidate_min": CANDIDATE_MIN,
        "silence_s": SILENCE_S,
        "concurrency": args.concurrency,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    print(f"[decide] {len(streams)} streams x {len(args.models)} models", flush=True)
    results, failures = asyncio.run(_simulate_all(streams, args.models, key, args, out_dir))
    summary = summarize(streams, results) | {"failures": failures}
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    if key and key in text:
        raise RuntimeError("refusing to write a summary that contains an API key")
    (out_dir / "summary.json").write_text(text, "utf-8")
    (out_dir / "summary.md").write_text(render(summary, meta), "utf-8")
    relative = out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir
    print(f"\nsummary: {out_dir / 'summary.md'}")
    print(f"to share: git add {relative.as_posix()} && git commit && git push")


if __name__ == "__main__":
    main()
