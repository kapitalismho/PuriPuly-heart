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
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

import analyze
import decide
import envkeys

from puripuly_heart.config import prompts
from puripuly_heart.providers.llm.messages import build_translation_user_message

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = ("perplexity/pplx-decider-v1.2", "inception/mercury-decide")
DEFAULT_PROVIDERS = ("soniox", "deepgram")
DEFAULT_DATA = ("ko_conv", "ja_conv", "en_conv")
DEFAULT_TRANSLATOR = "google/gemma-4-26b-a4b-it"
DEFAULT_JUDGE = "openai/gpt-6.1-sol"
TARGETS = {"ko": "English", "ja": "Korean", "en": "Korean"}
TRIGGER_S = 3.0
FINALIZE_S = 0.4
MIN_BEFORE = {"Korean": 4, "Japanese": 4, "English": 8}
AFTER_CHARS = {"Korean": 40, "Japanese": 40, "English": 120}
START_SLACK = 3
CONTEXT_LINES = 3
MIN_PAUSE_S = 0.25
VERDICTS = ("same", "minor", "changed")

SENTENCE_QUESTION = {
    "type": "noul",
    "instructions": (
        "`previous` is earlier speech, `before` is live speech recognition and `after` is what the "
        "speaker said next. Does `before` end exactly where a sentence or a self-contained clause "
        "ends, so that `before` can be translated without the words in `after`?"
    ),
    "criteria": decide.VERIFY["criteria"],
}

JUDGE_SYSTEM = (
    "You review live subtitle translation for a social VR app. A remote speaker's utterance was "
    "split into consecutive chunks. Each chunk was translated as its own subtitle as soon as it "
    "was released, with earlier source lines as context. Compare the chunked subtitles with a "
    "translation of the whole utterance made after it ended. The source is speech recognition "
    "and may contain errors; judge only the effect of splitting, not problems that the whole "
    "translation shares.\n\n"
    "Answer with a JSON object:\n"
    '{"verdict": "same" | "minor" | "changed", "premature": true | false, '
    '"reason": "<one short English sentence>"}\n\n'
    "verdict:\n"
    "- same: read in order, the chunked subtitles convey the same meaning as the whole "
    "translation, and each chunk reads acceptably as a subtitle.\n"
    "- minor: the meaning is preserved, but a chunk is awkward, a nuance is lost, or a word is "
    "repeated or dropped without changing the meaning.\n"
    "- changed: meaning is lost, added or wrong, or a chunk cannot be understood on its own.\n"
    "premature: true when an earlier subtitle, read before the next one appears, would make a "
    "reader believe something that the rest of the utterance contradicts, such as a missing "
    "negation, a question or condition shown as a plain statement, or the wrong subject or object."
)


def translation_question(target: str) -> dict:
    return {
        "type": "noul",
        "instructions": (
            "`before` is live speech recognition of what a speaker just said, `after` is what the "
            "speaker said next, and `previous` was already shown as subtitles. An interpreter "
            f"wants to show a {target} subtitle for `before` now and translate `after` as the next "
            f"subtitle. Would the {target} translation of `before` alone stay correct and natural "
            "once `after` is heard?"
        ),
        "criteria": {
            "true": f"`before` can be translated into {target} now; `after` does not change it.",
            "false": (
                f"Translating `before` into {target} now would be wrong, misleading or unnatural "
                "because it depends on `after`."
            ),
        },
    }


@dataclass
class Candidate:
    position: int
    count: int
    cut: int
    comma: bool
    gap_s: float | None
    state: dict
    label: bool
    answers: dict = field(default_factory=dict)


@dataclass
class Trigger:
    t: float
    k: int
    text: str
    start_count: int
    candidates: list[Candidate]
    rtt: dict = field(default_factory=dict)


@dataclass
class Gap:
    index: int
    t0: float
    p0: int
    t1: float
    p1: int
    triggers: list[Trigger] = field(default_factory=list)


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    model: str | None = None
    question: str | None = None
    theta: float | None = None
    forced: bool = False


@dataclass
class StreamWork:
    stream: decide.Stream
    target: str
    gaps: list[Gap]
    timeline: list[tuple[float, int]]


class FatalChatError(RuntimeError):
    pass


class Chat:
    def __init__(self, key: str | None, concurrency: int, fake: bool) -> None:
        self.key = key
        self.fake = fake
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=180)

    async def complete(self, role: str, model: str, messages: list[dict], fake_text: str) -> dict:
        if self.fake:
            return {"text": fake_text, "rtt": 0.0, "retries": 0}
        body: dict = {"model": model, "messages": messages, "usage": {"include": True}}
        if role == "translate":
            body |= {"temperature": 0, "reasoning": {"effort": "none"}, "max_tokens": 400}
        else:
            body |= {
                "reasoning": {"effort": "low"},
                "max_tokens": 4000,
                "response_format": {"type": "json_object"},
            }
        headers = {"Authorization": f"Bearer {self.key}", "X-Title": "PuriPuly split probe"}
        retries, error, rtt = 0, None, 0.0
        while retries <= decide.MAX_RETRIES:
            async with self.semaphore:
                started = time.perf_counter()
                try:
                    response = await self.client.post(CHAT_URL, json=body, headers=headers)
                    rtt = time.perf_counter() - started
                    if response.status_code == 200:
                        payload = response.json()
                        if payload.get("error"):
                            error = str(payload["error"])[:200]
                        else:
                            usage = payload.get("usage") or {}
                            message = (payload.get("choices") or [{}])[0].get("message") or {}
                            return {
                                "text": (message.get("content") or "").strip(),
                                "rtt": rtt,
                                "retries": retries,
                                "tokens_in": usage.get("prompt_tokens"),
                                "tokens_out": usage.get("completion_tokens"),
                                "cost": usage.get("cost"),
                                "served": payload.get("model"),
                            }
                    else:
                        error = f"HTTP {response.status_code}: {response.text[:200]}"
                        if response.status_code in (401, 402, 403, 404):
                            raise FatalChatError(f"{model}: {error}")
                        if response.status_code not in (408, 429, 500, 502, 503, 524, 529):
                            break
                except httpx.HTTPError as exc:
                    rtt = time.perf_counter() - started
                    error = f"{type(exc).__name__}: {exc}"[:200]
            retries += 1
            await asyncio.sleep(min(8.0, 0.5 * 2**retries))
        return {"text": None, "rtt": rtt, "retries": retries, "error": error}

    async def close(self) -> None:
        await self.client.aclose()


def all_ends(text: str, language: str) -> list[tuple[int, int]]:
    ends, count = [], 0
    for i, ch in enumerate(text):
        size = len(analyze.normalize(ch))
        count += size
        if not size:
            continue
        if language == "Japanese" or i + 1 == len(text) or text[i + 1] == " ":
            ends.append((count, i + 1))
    return ends


def punct_timeline(stream: decide.Stream) -> list[tuple[float, int]]:
    ordered, last = [], 0
    for t, position in sorted(decide._punct_releases(stream)):
        if position > last:
            ordered.append((t, position))
            last = position
    return ordered


def _trim_after(after: str, language: str) -> str:
    limit = AFTER_CHARS[language]
    if len(after) <= limit:
        return after
    cut = after[:limit]
    if language == "English" and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + " …"


def _gap_s(stream: decide.Stream, position: int) -> float | None:
    if 0 < position < len(stream.char_time):
        return float(stream.char_time[position] - stream.char_time[position - 1])
    return None


def _candidates(
    stream: decide.Stream, t: float, p0: int, target: str
) -> tuple[str, int, list[Candidate]] | None:
    index = stream.at(t)
    text, raw = stream.texts[index], stream.raws[index]
    ends = all_ends(text, stream.language)
    if not ends:
        return None
    located = []
    for count, cut in reversed(ends):
        position = stream.locate(analyze.normalize(text[:cut]))
        located.append((count, cut, position))
        if position < p0 - START_SLACK:
            break
    located.reverse()
    if p0 == 0:
        start_count, start_cut = 0, 0
    else:
        near = [e for e in located if abs(e[2] - p0) <= START_SLACK]
        if not near:
            return None
        start_count, start_cut, _ = min(near, key=lambda e: (abs(e[2] - p0), -e[0]))
    raw_norm, raw_positions = analyze._index_normalized(raw)
    commas_ok = raw_norm == analyze.normalize(text)
    total = ends[-1][0]
    language = stream.language
    previous = decide.window(text[:start_cut].strip(), language) if start_cut else ""
    candidates = []
    for count, cut, position in located:
        if count - start_count < MIN_BEFORE[language]:
            continue
        if total - count < decide.RIGHT_CHARS[language]:
            continue
        before = text[start_cut:cut].strip()
        after = _trim_after(text[cut:].strip(), language)
        state = {"language": language, "before": before, "after": after}
        if previous:
            state = {"language": language, "previous": previous, "before": before, "after": after}
        candidates.append(
            Candidate(
                position=position,
                count=count,
                cut=cut,
                comma=commas_ok and decide._comma_after(raw, raw_positions, count),
                gap_s=_gap_s(stream, position),
                state=state,
                label=decide._at_label(stream, text[:cut]),
            )
        )
    return text, start_count, candidates


def build_gaps(stream: decide.Stream, target: str, trigger_s: float) -> list[Gap]:
    cover = [stream.to_final(text) for text in stream.texts]
    timeline = punct_timeline(stream)
    end = (stream.times[-1], len(stream.final_norm))
    points = [(stream.times[0], 0)] + timeline + [end]
    gaps = []
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        grown = next(
            (stream.times[i] for i in range(len(cover)) if stream.times[i] > t0 and cover[i] > p0),
            None,
        )
        if grown is None or grown >= t1:
            continue
        gap = Gap(len(gaps), t0, p0, t1, p1)
        seen: dict[int, Candidate] = {}
        k = 1
        while grown + k * trigger_s < min(t1, stream.times[-1]):
            t = grown + k * trigger_s
            built = _candidates(stream, t, p0, target)
            if built is not None:
                text, start_count, found = built
                current = []
                for candidate in found:
                    if candidate.position <= p0 or candidate.position >= p1:
                        continue
                    current.append(seen.setdefault(candidate.position, candidate))
                gap.triggers.append(Trigger(t, k, text, start_count, current))
            k += 1
        if gap.triggers:
            gaps.append(gap)
    return gaps


def _question_payload(name: str, target: str) -> dict:
    return SENTENCE_QUESTION if name == "sentence" else translation_question(target)


def _state_for(name: str, candidate: Candidate, target: str) -> dict:
    if name == "sentence":
        return candidate.state
    state = {"source_language": candidate.state["language"], "target_language": target}
    return state | {k: v for k, v in candidate.state.items() if k != "language"}


async def ask_all(works: list[StreamWork], deciders: dict, out_dir: Path) -> dict:
    failures: dict[str, str] = {}
    logs: dict[str, list[dict]] = {}

    async def one(work: StreamWork, gap: Gap, trigger: Trigger, model: str, name: str) -> None:
        asked = []
        for candidate in trigger.candidates:
            if (model, name) in candidate.answers:
                continue
            candidate.answers[(model, name)] = None
            asked.append(candidate)
        if not asked or model in failures:
            return
        stream = work.stream
        decider = deciders[model]
        question = _question_payload(name, work.target)

        async def call(candidate: Candidate) -> dict:
            oracle = {"type": "noul", "noul": 0.9 if candidate.label else 0.1}
            return await decider.ask(question, _state_for(name, candidate, work.target), oracle)

        try:
            results = await asyncio.gather(*(call(c) for c in asked))
        except decide.FatalDeciderError as exc:
            failures.setdefault(model, str(exc))
            return
        key = f"{stream.data}_{stream.provider}_{model.replace('/', '__')}"
        rtts = []
        for candidate, result in zip(asked, results):
            p = decide._noul(result["answer"])
            candidate.answers[(model, name)] = p
            rtts.append(result["rtt"])
            logs.setdefault(key, []).append(
                decide._record(
                    result,
                    question=name,
                    gap=gap.index,
                    k=trigger.k,
                    t=round(trigger.t, 4),
                    position=candidate.position,
                    label=candidate.label,
                    p=p,
                    state=_state_for(name, candidate, work.target),
                )
            )
        trigger.rtt[(model, name)] = max(rtts)

    jobs = []
    for work in works:
        for gap in work.gaps:
            for trigger in gap.triggers:
                for model in deciders:
                    for name in ("sentence", "translation"):
                        jobs.append((work, gap, trigger, model, name))
    for start in range(0, len(jobs), 64):
        await asyncio.gather(*(one(*job) for job in jobs[start : start + 64]))
        print(
            f"[split] decisions {min(start + 64, len(jobs))}/{len(jobs)} trigger jobs", flush=True
        )
    calls_dir = out_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    for key, entries in logs.items():
        with gzip.open(calls_dir / f"decide_{key}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"failures": failures, "logs": logs}


def policies_for(models: list[str]) -> list[Policy]:
    policies = [
        Policy("punctuation only", "none"),
        Policy("time cut (latest word)", "time"),
        Policy("provider comma", "comma"),
        Policy("longest pause", "pause"),
    ]
    short = {m: m.split("/")[-1] for m in models}
    for model in models:
        for name, theta, forced in (
            ("sentence", 0.5, False),
            ("sentence", 0.5, True),
            ("translation", 0.3, False),
            ("translation", 0.5, False),
            ("translation", 0.7, False),
            ("translation", 0.5, True),
        ):
            label = f"{short[model]} Q{'1' if name == 'sentence' else '2'} p>={theta}"
            label += " +forced" if forced else ""
            policies.append(Policy(label, "decider", model, name, theta, forced))
    return policies


def choose(policy: Policy, trigger: Trigger, last: int) -> tuple[int, float] | None:
    eligible = [c for c in trigger.candidates if c.position > last]
    if not eligible or policy.kind == "none":
        return None
    if policy.kind == "time":
        return max(c.position for c in eligible), 0.0
    if policy.kind == "comma":
        commas = [c.position for c in eligible if c.comma]
        return (max(commas), 0.0) if commas else None
    if policy.kind == "pause":
        paused = [c for c in eligible if c.gap_s is not None and c.gap_s >= MIN_PAUSE_S]
        if not paused:
            return None
        best = max(paused, key=lambda c: (c.gap_s, c.position))
        return best.position, 0.0
    key = (policy.model, policy.question)
    scored = [(c, c.answers.get(key)) for c in eligible if c.answers.get(key) is not None]
    extra = trigger.rtt.get(key, 0.0)
    accepted = [c.position for c, p in scored if p >= policy.theta]
    if accepted:
        return max(accepted), extra
    if policy.forced and trigger.k >= 2 and scored:
        best = max(scored, key=lambda item: (item[1], item[0].position))
        return best[0].position, extra
    return None


def splits_for(policy: Policy, work: StreamWork) -> list[tuple[float, int, int]]:
    splits = []
    for gap in work.gaps:
        last = gap.p0
        for trigger in gap.triggers:
            picked = choose(policy, trigger, last)
            if picked is None:
                continue
            position, extra = picked
            t = trigger.t + FINALIZE_S + extra
            if t >= gap.t1:
                break
            splits.append((t, position, gap.index))
            last = position
    return splits


def char_latency(work: StreamWork, releases: list[tuple[float, int]]) -> tuple[list, list, int]:
    stream = work.stream
    ordered, last = [], 0
    for t, position in sorted(releases):
        if position > last:
            ordered.append((t, position))
            last = position
    times = [t for t, _ in ordered]
    positions = [p for _, p in ordered]
    horizon = stream.duration_s - decide.END_EXCLUSION_S
    spans = [(g.p0, g.p1) for g in work.gaps]
    all_lat, long_lat, missing = [], [], 0
    for j in range(1, len(stream.final_norm) + 1):
        spoken = float(stream.char_time[j - 1])
        if spoken > horizon:
            continue
        k = bisect.bisect_left(positions, j - decide.LABEL_SLACK)
        if k == len(positions):
            missing += 1
            continue
        latency = times[k] - spoken
        all_lat.append(latency)
        if any(p0 < j <= p1 for p0, p1 in spans):
            long_lat.append(latency)
    return all_lat, long_lat, missing


def _raw_cut(raw: str, positions: list[int], position: int) -> int:
    if position <= 0:
        return 0
    if position >= len(positions):
        return len(raw)
    j = positions[position - 1] + 1
    while j < len(raw) and not raw[j].isspace() and not raw[j].isalnum():
        j += 1
    return j


def _context_lines(work: StreamWork, p0: int) -> list[str]:
    raw = work.stream.final_raw
    _, positions = analyze._index_normalized(raw)
    bounds = [0] + [p for _, p in work.timeline if p <= p0]
    if bounds[-1] != p0:
        bounds.append(p0)
    lines = []
    for a, b in zip(bounds, bounds[1:]):
        piece = raw[_raw_cut(raw, positions, a) : _raw_cut(raw, positions, b)].strip()
        if piece:
            lines.append(piece)
    return lines[-CONTEXT_LINES:]


def _chunks(work: StreamWork, gap: Gap, cuts: tuple[int, ...]) -> list[str]:
    raw = work.stream.final_raw
    _, positions = analyze._index_normalized(raw)
    points = [gap.p0, *cuts, gap.p1]
    pieces = [
        raw[_raw_cut(raw, positions, a) : _raw_cut(raw, positions, b)].strip()
        for a, b in zip(points, points[1:])
    ]
    return pieces


class Evaluator:
    def __init__(self, chat: Chat, translator: str, judge: str | None) -> None:
        self.chat = chat
        self.translator = translator
        self.judge = judge
        self.translations: dict[str, dict] = {}
        self.judgments: dict[str, dict] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.system_prompts: dict[tuple[str, str], str] = {}

    def _system(self, source: str, target: str) -> str:
        key = (source, target)
        if key not in self.system_prompts:
            template = prompts.get_translation_prompt_template(model=self.translator)
            self.system_prompts[key] = prompts.render_translation_prompt_template(
                template, source_name=source, target_name=target, input_channel="peer"
            )
        return self.system_prompts[key]

    async def translate(self, source: str, target: str, context: list[str], text: str) -> str:
        context_text = "\n".join(f'- [peer] "{line}"' for line in context)
        key = hashlib.sha256(
            json.dumps([self.translator, source, target, context_text, text]).encode()
        ).hexdigest()
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self.translations:
                messages = [
                    {"role": "system", "content": self._system(source, target)},
                    {
                        "role": "user",
                        "content": build_translation_user_message(text=text, context=context_text),
                    },
                ]
                result = await self.chat.complete(
                    "translate", self.translator, messages, fake_text=f"<{target}: {text}>"
                )
                self.translations[key] = result | {
                    "source_language": source,
                    "target_language": target,
                    "context": context,
                    "input": text,
                }
        return self.translations[key].get("text") or ""

    async def assess(
        self, work: StreamWork, gap: Gap, cuts: tuple[int, ...], labels: list[bool]
    ) -> dict:
        stream = work.stream
        source = stream.language
        target = work.target
        context = _context_lines(work, gap.p0)
        chunks = _chunks(work, gap, cuts)
        whole = _chunks(work, gap, ())[0]
        jobs = [self.translate(source, target, context, whole)]
        running = list(context)
        for chunk in chunks:
            jobs.append(self.translate(source, target, running[-CONTEXT_LINES:], chunk))
            running.append(chunk)
        translated = await asyncio.gather(*jobs)
        whole_translation, pieces = translated[0], translated[1:]
        key = hashlib.sha256(
            json.dumps([stream.data, stream.provider, gap.index, list(cuts)]).encode()
        ).hexdigest()
        if self.judge is None:
            return {"key": key, "verdict": None}
        lines = [
            f"Source language: {source}",
            f"Subtitle language: {target}",
            "Earlier lines (source):",
            *[f'- "{line}"' for line in context],
            "Chunks in the order they were shown:",
        ]
        for n, (chunk, piece) in enumerate(zip(chunks, pieces), 1):
            lines += [f'{n}. source: "{chunk}"', f'   subtitle: "{piece}"']
        lines += [f'Whole utterance: "{whole}"', f'Whole translation: "{whole_translation}"']
        fake = json.dumps(
            {
                "verdict": "same" if all(labels) else "changed",
                "premature": not all(labels),
                "reason": "label oracle",
            }
        )
        result = await self.chat.complete(
            "judge",
            self.judge,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": "\n".join(lines)},
            ],
            fake_text=fake,
        )
        verdict, premature, reason = _parse_judgment(result.get("text"))
        record = result | {
            "data": stream.data,
            "provider": stream.provider,
            "gap": gap.index,
            "cuts": list(cuts),
            "chunks": chunks,
            "subtitles": pieces,
            "whole": whole,
            "whole_translation": whole_translation,
            "verdict": verdict,
            "premature": premature,
            "reason": reason,
        }
        self.judgments[key] = record
        return record


def _parse_judgment(text: str | None) -> tuple[str | None, bool | None, str | None]:
    if not text:
        return None, None, None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None, None, None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None, None
    verdict = data.get("verdict")
    if verdict not in VERDICTS:
        verdict = None
    premature = data.get("premature")
    return verdict, premature if isinstance(premature, bool) else None, data.get("reason")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.array(values)
    return {
        "n": len(values),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
    }


async def evaluate(works, policies, evaluator: Evaluator | None) -> dict:
    rows: dict[str, dict] = {}
    pending = []
    for policy in policies:
        for work in works:
            stream = work.stream
            splits = splits_for(policy, work)
            releases = list(work.timeline) + [(t, p) for t, p, _ in splits]
            all_lat, long_lat, missing = char_latency(work, releases)
            kinds = {"sentence": 0, "clause": 0, "mid": 0}
            for _, position, _ in splits:
                if any(abs(position - b) <= decide.LABEL_SLACK for b in stream.terminals):
                    kinds["sentence"] += 1
                elif any(abs(position - b) <= decide.LABEL_SLACK for b in stream.commas):
                    kinds["clause"] += 1
                else:
                    kinds["mid"] += 1
            row = {
                "policy": policy.name,
                "data": stream.data,
                "provider": stream.provider,
                "minutes": stream.duration_s / 60,
                "splits": len(splits),
                "kinds": kinds,
                "latency_all": all_lat,
                "latency_long": long_lat,
                "unreleased_chars": missing,
                "judged": [],
            }
            rows[f"{policy.name}|{stream.data}|{stream.provider}"] = row
            if evaluator is None:
                continue
            by_gap: dict[int, list[int]] = {}
            for _, position, index in splits:
                by_gap.setdefault(index, []).append(position)
            for gap in work.gaps:
                cuts = tuple(sorted(by_gap.get(gap.index, [])))
                if not cuts:
                    continue
                labels = [
                    any(abs(c - b) <= decide.LABEL_SLACK for b in stream.terminals + stream.commas)
                    for c in cuts
                ]
                pending.append((row, work, gap, cuts, labels))
    if evaluator is not None:
        unique: dict[tuple, asyncio.Task] = {}
        for row, work, gap, cuts, labels in pending:
            key = (work.stream.data, work.stream.provider, gap.index, cuts)
            if key not in unique:
                unique[key] = asyncio.ensure_future(evaluator.assess(work, gap, cuts, labels))
        print(f"[split] assessing {len(unique)} distinct split utterances", flush=True)
        done = 0
        for task in asyncio.as_completed(list(unique.values())):
            await task
            done += 1
            if done % 25 == 0 or done == len(unique):
                print(f"[split] assessed {done}/{len(unique)}", flush=True)
        for row, work, gap, cuts, labels in pending:
            key = (work.stream.data, work.stream.provider, gap.index, cuts)
            row["judged"].append(unique[key].result())
    return rows


def _pool(rows: list[dict]) -> dict:
    minutes = sum(r["minutes"] for r in rows)
    splits = sum(r["splits"] for r in rows)
    kinds = {k: sum(r["kinds"][k] for r in rows) for k in ("sentence", "clause", "mid")}
    judged = [j for r in rows for j in r["judged"] if j.get("verdict")]
    verdicts = {v: sum(1 for j in judged if j["verdict"] == v) for v in VERDICTS}
    premature = sum(1 for j in judged if j.get("premature"))
    return {
        "splits": splits,
        "splits_per_min": round(splits / minutes, 2) if minutes else None,
        "kinds": kinds,
        "latency_all": _stats([x for r in rows for x in r["latency_all"]]),
        "latency_long": _stats([x for r in rows for x in r["latency_long"]]),
        "judged": len(judged),
        "verdicts": verdicts,
        "premature": premature,
    }


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def _lat(stat: dict) -> str:
    return f"{stat['p50']:.2f} / {stat['p90']:.2f}" if stat.get("n") else "-"


def _row_md(name: str, pooled: dict) -> str:
    kinds = pooled["kinds"]
    total = sum(kinds.values())
    judged = pooled["judged"]
    v = pooled["verdicts"]
    return (
        f"| {name} | {pooled['splits_per_min']} | {_pct(kinds['sentence'], total)} | "
        f"{_pct(kinds['clause'], total)} | {_pct(kinds['mid'], total)} | "
        f"{_lat(pooled['latency_long'])} | {_lat(pooled['latency_all'])} | {judged} | "
        f"{_pct(v['same'], judged)} | {_pct(v['minor'], judged)} | {_pct(v['changed'], judged)} | "
        f"{_pct(pooled['premature'], judged)} |"
    )


def render(summary: dict, meta: dict) -> str:
    lines = [
        "# Long-utterance split probe",
        "",
        "Replays recorded partial transcripts. Punctuation in the partial text releases text as "
        "usual. When no punctuation release happens for "
        f"{meta['trigger_s']:.1f} s after new text starts, a split policy may release an earlier "
        "part of the unreleased text. A split costs a "
        f"{meta['finalize_s']:.1f} s finalize allowance plus, for decision models, the slowest "
        "round trip of that trigger's calls.",
        "",
        "- `time cut`: the latest word end with right context (a stand-in for the acoustic "
        "4-6 s rules). `provider comma`: the latest comma in the partial text. `longest pause`: "
        f"the word end with the largest gap of at least {MIN_PAUSE_S:.2f} s in Soniox word "
        "timing.",
        "- `Q1`: sentence or clause end question. `Q2`: translation question with the target "
        "language. Each word end with right context is asked once per question. A policy "
        "releases the latest word end at or above the threshold; `+forced` releases the "
        "highest-scored word end at the second trigger when none passes.",
        "- `sentence` / `clause` / `mid`: share of splits within 1 character of a Soniox "
        "sentence-final punctuation label, a comma label, or neither.",
        "- `latency long`: seconds from each character's Soniox end timestamp to its release, "
        "for characters in utterances that reached a trigger. `latency all`: every character. "
        "Cells show `p50 / p90`.",
        "- `judged`: split utterances translated chunk by chunk with the application prompt "
        f"and `{meta['translator']}`, then compared with the whole-utterance translation by "
        f"`{meta['judge']}`. `premature`: an earlier subtitle misleads before the next one.",
        "",
    ]
    header = (
        "| policy | splits/min | sentence | clause | mid | latency long | latency all | judged "
        "| same | minor | changed | premature |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines += ["## All data", "", header, sep]
    lines += [_row_md(name, pooled) for name, pooled in summary["pooled"].items()]
    for data, table in summary["by_data"].items():
        lines += ["", f"## {data}", "", header, sep]
        lines += [_row_md(name, pooled) for name, pooled in table.items()]
    lines += [
        "",
        "## Calls",
        "",
        "| model | role | calls | errors | cost | RTT p50 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in summary["calls"]:
        lines.append(
            f"| {entry['model']} | {entry['role']} | {entry['calls']} | {entry['errors']} | "
            f"${entry['cost']:.4f} | {entry['rtt_p50']} |"
        )
    if summary.get("failures"):
        lines += ["", "## Failures", ""]
        lines += [f"- {model}: {reason}" for model, reason in summary["failures"].items()]
    return "\n".join(lines) + "\n"


def _call_rows(decisions: dict, evaluator: Evaluator | None, models, translator, judge) -> list:
    rows = []
    for model in models:
        entries = [
            e
            for key, items in decisions["logs"].items()
            if key.endswith(model.replace("/", "__"))
            for e in items
        ]
        rows.append(_call_row(model, "decision", entries))
    if evaluator is not None:
        rows.append(_call_row(translator, "translation", list(evaluator.translations.values())))
        if judge:
            rows.append(_call_row(judge, "judge", list(evaluator.judgments.values())))
    return rows


def _call_row(model: str, role: str, entries: list[dict]) -> dict:
    rtts = [e["rtt"] for e in entries if e.get("rtt")]
    return {
        "model": model,
        "role": role,
        "calls": len(entries),
        "errors": sum(1 for e in entries if e.get("error")),
        "cost": float(sum(e.get("cost") or 0 for e in entries)),
        "rtt_p50": round(float(np.percentile(rtts, 50)), 3) if rtts else "-",
    }


def _write(path: Path, text: str, key: str | None) -> None:
    if key and key in text:
        raise RuntimeError(f"refusing to write {path.name}: it contains the API key")
    path.write_text(text, encoding="utf-8")


def _check_gz(directory: Path, key: str | None) -> None:
    if not key:
        return
    for path in directory.glob("*.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            if key in handle.read():
                path.unlink()
                raise RuntimeError(f"removed {path.name}: it contained the API key")


async def preflight(models, deciders, chat: Chat, translator: str, judge: str | None) -> dict:
    failures = {}
    sample = {"language": "English", "before": "i went home", "after": "and then"}
    for model in models:
        try:
            result = await deciders[model].ask(SENTENCE_QUESTION, sample, {"noul": 0.5})
            if decide._noul(result["answer"]) is None:
                failures[model] = f"no noul answer: {str(result)[:200]}"
        except decide.FatalDeciderError as exc:
            failures[model] = str(exc)
    checks = [("translate", translator)] + ([("judge", judge)] if judge else [])
    for role, model in checks:
        messages = [{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}]
        result = await chat.complete(role, model, messages, fake_text='{"ok": true}')
        if result.get("text") is None:
            raise FatalChatError(f"{model}: {result.get('error')}")
    return failures


async def run(args, works, key) -> tuple[dict, dict, Evaluator | None]:
    deciders = {m: decide.Decider(m, key, args.concurrency, args.fake) for m in args.models}
    chat = Chat(key, args.chat_concurrency, args.fake)
    try:
        failures = await preflight(args.models, deciders, chat, args.translator, args.judge)
        for model, reason in failures.items():
            print(f"[check] {model}: stopped, {reason.replace(key or '<none>', '<key>')}")
        active = [m for m in args.models if m not in failures]
        decisions = await ask_all(works, {m: deciders[m] for m in active}, args.out)
        decisions["failures"] |= failures
        evaluator = None if args.no_translate else Evaluator(chat, args.translator, args.judge)
        rows = await evaluate(works, policies_for(active), evaluator)
    finally:
        for decider in deciders.values():
            await decider.close()
        await chat.close()
    return rows, decisions, evaluator


def summarize(rows: dict, works, policies) -> dict:
    pooled = {}
    by_data: dict[str, dict] = {}
    for policy in policies:
        selected = [r for r in rows.values() if r["policy"] == policy.name]
        if not selected:
            continue
        pooled[policy.name] = _pool(selected)
        for data in sorted({r["data"] for r in selected}):
            by_data.setdefault(data, {})[policy.name] = _pool(
                [r for r in selected if r["data"] == data]
            )
    detail = [
        r
        | {
            "latency_all": _stats(r["latency_all"]),
            "latency_long": _stats(r["latency_long"]),
            "judged": [
                {k: j.get(k) for k in ("gap", "cuts", "verdict", "premature", "reason")}
                for j in r["judged"]
            ],
        }
        for r in rows.values()
    ]
    return {"pooled": pooled, "by_data": by_data, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="*", type=Path)
    parser.add_argument("--data", nargs="*", default=list(DEFAULT_DATA))
    parser.add_argument("--providers", nargs="*", default=list(DEFAULT_PROVIDERS))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--target", action="append", default=[], help="source=Target, e.g. ko=Japanese"
    )
    parser.add_argument("--trigger", type=float, default=TRIGGER_S)
    parser.add_argument("--translator", default=DEFAULT_TRANSLATOR)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--chat-concurrency", type=int, default=4)
    parser.add_argument("--env-file", action="append", type=Path, default=[])
    parser.add_argument("--key", help="environment variable that holds the OpenRouter key")
    parser.add_argument("--fake", action="store_true", help="label oracle and echo translation")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.no_judge or args.no_translate:
        args.judge = None
    targets = dict(TARGETS)
    for item in args.target:
        source, _, target = item.partition("=")
        targets[source] = target

    key = None
    if not args.fake:
        env_files = args.env_file or [REPO_ROOT / ".env.local", REPO_ROOT / ".env"]
        overrides = {"openrouter": args.key} if args.key else {}
        resolved = envkeys.resolve(decide.KEY_CANDIDATES, env_files, overrides)
        if "openrouter" not in resolved:
            sys.exit("No OpenRouter key found (OPENROUTER_API_KEY or --key VARIABLE).")
        key_variable, key = resolved["openrouter"]
        print(f"[key] openrouter from {key_variable}")

    sources = {}
    works = []
    for data in args.data:
        results_dir = None
        if args.results:
            results_dir = next(
                (
                    p
                    for p in args.results
                    if (p / "runs" / f"soniox_natural_{data}.json.gz").exists()
                ),
                None,
            )
        results_dir = results_dir or decide._latest_results(data)
        if results_dir is None:
            print(f"[skip] {data}: no natural runs")
            continue
        sources[data] = results_dir.name
        for provider in args.providers:
            stream = decide.build_stream(results_dir, provider, data)
            if stream is None:
                print(f"[skip] {data} {provider}: no usable run")
                continue
            target = targets[data[:2]]
            gaps = build_gaps(stream, target, args.trigger)
            works.append(StreamWork(stream, target, gaps, punct_timeline(stream)))
            candidates = len(
                {(g.index, c.position) for g in gaps for t in g.triggers for c in t.candidates}
            )
            print(
                f"[split] {data:8} {provider:8} long utterances={len(gaps):3} "
                f"triggers={sum(len(g.triggers) for g in gaps):3} candidates={candidates:4}",
                flush=True,
            )
    if not works:
        sys.exit("Nothing to replay.")

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out = args.out or HERE / "results" / f"split-{stamp}{'-fake' if args.fake else ''}"
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        rows, decisions, evaluator = asyncio.run(run(args, works, key))
    except FatalChatError as exc:
        if not any(args.out.iterdir()):
            args.out.rmdir()
        sys.exit(f"[check] stopped: {str(exc).replace(key or '<none>', '<key>')}")
    active = [m for m in args.models if m not in decisions["failures"]]
    policies = policies_for(active)
    summary = summarize(rows, works, policies)
    summary["calls"] = _call_rows(decisions, evaluator, active, args.translator, args.judge)
    summary["failures"] = {
        m: r.replace(key or "<none>", "<key>") for m, r in decisions["failures"].items()
    }
    meta = {
        "started_utc": stamp,
        "elapsed_s": round(time.monotonic() - started, 1),
        "fake": args.fake,
        "sources": sources,
        "streams": [f"{w.stream.data}/{w.stream.provider}" for w in works],
        "models": active,
        "translator": args.translator if evaluator else None,
        "judge": args.judge,
        "targets": {w.stream.data: w.target for w in works},
        "trigger_s": args.trigger,
        "finalize_s": FINALIZE_S,
        "min_before_chars": MIN_BEFORE,
        "right_chars": decide.RIGHT_CHARS,
        "after_chars": AFTER_CHARS,
        "questions": {
            "sentence": SENTENCE_QUESTION,
            "translation": {t: translation_question(t) for t in sorted(set(targets.values()))},
        },
        "judge_system": JUDGE_SYSTEM,
        "translation_prompt": prompts.TRANSLATION_PROMPT_NAME,
    }
    if evaluator is not None:
        with gzip.open(args.out / "calls" / "translations.jsonl.gz", "wt", encoding="utf-8") as h:
            for entry in evaluator.translations.values():
                h.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with gzip.open(args.out / "calls" / "judgments.jsonl.gz", "wt", encoding="utf-8") as h:
            for entry in evaluator.judgments.values():
                h.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _check_gz(args.out / "calls", key)
    _write(args.out / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1), key)
    _write(args.out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=1), key)
    _write(args.out / "summary.md", render(summary, meta), key)
    print(f"[split] wrote {args.out}")


if __name__ == "__main__":
    main()
