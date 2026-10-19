"""Offline batch-route policy simulation from recorded runs.

Replays recorded capture-owned inputs (request lifecycles, submitted ranges) and full speaker
evidence through the product `CausalBatchSession` and `SpeakerEvidenceView` on a virtual clock.
ASR is replaced by the reference words inside each submitted source range ("oracle ASR"), with
decode latency taken from the recorded run. Presentation follows a simplified model of the
one-second late-assignment window. Results are counterfactual estimates for choosing candidates,
not live measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import importlib
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from uuid import UUID

from bench import reference, units
from bench.score import score_case

BENCH_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = BENCH_ROOT / "corpus" / "references"
SAMPLE_RATE_HZ = 16_000
_SLICE = re.compile(
    r"\[BatchSpeaker\] request=(?P<request>\S+) slice=(?P<slice>\S+) .*?submitted_at=(?P<submitted>\S+)"
)


def _span(values):
    from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity, AudioCaptureSpan

    epoch, callback, rate, start, end, mono_start, mono_end, n_rate, n_start, n_end, disc = values
    return AudioCaptureSpan(
        epoch,
        callback,
        rate,
        start,
        end,
        mono_start,
        mono_end,
        None if disc is None else AudioCaptureDiscontinuity(disc[0], disc[2], disc[1]),
        n_rate,
        n_start,
        n_end,
    )


def _evidence(item):
    from puripuly_heart.core.speaker.contracts import (
        ProbabilityFrame,
        SourceMapPiece,
        SpeakerEvidence,
    )

    frame = item["frame"]
    return SpeakerEvidence(
        item["session"],
        item["epoch"],
        item["ready"],
        tuple(
            SourceMapPiece(cs, ce, None if capture is None else _span(capture), kind)
            for cs, ce, kind, capture in item["map"]
        ),
        item["processed"],
        item["immutable"],
        (
            None
            if frame is None
            else ProbabilityFrame(frame[0], frame[1], tuple(frame[2]), tuple(frame[3]))
        ),
        item["range_state"],
        item["reason"],
    )


def _identity(raw, epoch_id):
    from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
    from puripuly_heart.core.stt.backend import STTProviderTurnIdentity

    return STTProviderTurnIdentity(
        AudioSegmentIdentity(
            raw["activation_generation"],
            raw["segment_order"],
            UUID(raw["segment_id"]),
            raw["capture_epoch"],
        ),
        epoch_id,
        raw["turn"],
        ("diarization-bench-sim",),
    )


class OracleInner:
    """Fake inner ASR session: reference words inside each sealed source range."""

    def __init__(self, sim):
        self.sim = sim
        self.queue: asyncio.Queue = asyncio.Queue()
        self.requests = {}
        self.audio = defaultdict(list)

    async def begin_turn(self, request):
        self.requests[request.identity] = request

    async def send_turn_audio(
        self, identity, pcm16le, *, payload_sequence, source_ranges, context_only
    ):
        self.audio[identity].append((tuple(source_ranges), context_only))

    async def seal_turn(
        self, identity, *, sealed_content_ranges, seal_reason, observed_trailing_silence_ms
    ):
        self.sim.seal_child(identity, tuple(sealed_content_ranges), seal_reason)

    async def abort_turn(self, identity, *, reason):
        return None

    async def turn_events(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event

    async def stop(self):
        await self.queue.put(None)

    async def close(self):
        await self.queue.put(None)


class CaseSimulation:
    def __init__(self, ref, events, case_id, latency_model, readable_delay):
        self.ref = ref
        self.events = [e for e in events if e.get("case_id") == case_id]
        self.case_id = case_id
        self.latency_model = latency_model
        self.readable_delay = readable_delay
        record = next(e for e in self.events if e["event"] == "source_close")["record"]
        self.start = float(record["case_start_monotonic"])
        self.lead = int(record["lead_in_samples"])
        self.now = 0.0
        self.heap = []
        self.counter = itertools.count()
        self.children = {}
        self.order = []
        self.slices = []

    def seconds(self, sample):
        return (sample - self.lead) / SAMPLE_RATE_HZ

    def oracle_text(self, ranges):
        spans = [
            (self.seconds(r.normalized_start_sample), self.seconds(r.normalized_end_sample))
            for r in ranges
            if r.normalized_start_sample is not None
        ]
        words = [
            t.token
            for t in self.ref.tokens
            if any(lo <= (t.start + t.end) / 2 < hi for lo, hi in spans)
        ]
        joiner = "" if self.ref.language in ("ko", "ja") else " "
        return joiner.join(words)

    def seal_child(self, identity, ranges, reason):
        duration = sum(r.normalized_sample_count for r in ranges) / SAMPLE_RATE_HZ
        latency = self.latency_model(duration)
        self.children[identity] = {
            "ranges": ranges,
            "reason": reason,
            "submitted": self.now,
            "text": self.oracle_text(ranges),
        }
        self.order.append(identity)
        heapq.heappush(self.heap, (self.now + latency, next(self.counter), "native", identity))

    async def run(self, overrides=None):
        from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
        from puripuly_heart.core.stt import batch_speaker
        from puripuly_heart.core.stt.backend import STTProviderTurnRequest, STTProviderTurnTerminal

        view = SpeakerEvidenceView()
        inner = OracleInner(self)
        batch = batch_speaker.CausalBatchSession(inner, clock=lambda: self.now, evidence_view=view)
        original_submit = batch._submit

        async def submit(parent, boundary, reason, ready_at=None):
            before = len(inner.requests)
            await original_submit(parent, boundary, reason, ready_at)
            if len(inner.requests) > before:
                self.slices.append(
                    units.Slice(
                        reason,
                        self.seconds(boundary if boundary is not None else parent.submitted_cursor),
                        None if ready_at is None else ready_at - self.start,
                        self.now - self.start,
                    )
                )

        batch._submit = submit
        views = defaultdict(int)
        for e in self.events:
            if e["event"] == "evidence_items":
                views[e["view"]] += len(e["items"])
        main_view = max(views, key=views.get) if views else None
        timeline = []
        for e in self.events:
            if e["event"] == "evidence_items" and e["view"] == main_view and e["items"]:
                timeline.append((e["t"], "evidence", e))
            elif (
                e["event"] == "policy_input"
                and e.get("session") == "CausalBatchSession"
                and e["call"]
                in (
                    "begin_turn",
                    "send_turn_audio",
                    "seal_turn",
                    "abort_turn",
                )
            ):
                timeline.append((e["t"], "input", e))
        for t, kind, e in timeline:
            heapq.heappush(self.heap, (t, next(self.counter), kind, e))
        epoch_id = "sim-" + self.case_id
        requests = {}
        consumed = []

        async def consume():
            async for event in batch.turn_events():
                consumed.append((self.now, event))

        consumer = asyncio.create_task(consume())
        proof_ready = {}
        while self.heap:
            t, _, kind, payload = heapq.heappop(self.heap)
            self.now = t
            if kind == "evidence":
                items = tuple(_evidence(i) for i in payload["items"])
                view.observe(items)
                await batch.observe_speaker_evidence(items)
                proof_ready.setdefault("times", []).append(t)
            elif kind == "native":
                identity = payload
                child = self.children[identity]
                request = inner.requests[identity]
                terminal = STTProviderTurnTerminal(
                    identity,
                    "final" if child["text"] else "empty",
                    child["text"],
                    text_authority="authoritative",
                )
                if request.on_native_event is not None:
                    request.on_native_event(terminal)
                state = batch._slices.get(identity)
                child["native"] = t
                child["speaker"] = None if state is None else state.speaker
                child["proof"] = None if state is None else state.audio.speaker_proof_id
                if child["proof"] is not None:
                    view.adopt(child["proof"])
                await inner.queue.put(terminal)
            else:
                call = payload["call"]
                identity = _identity(payload["identity"], epoch_id)
                if call == "begin_turn":
                    request = STTProviderTurnRequest(
                        identity,
                        None,
                        "peer",
                        (
                            None
                            if payload["display_group_id"] is None
                            else UUID(payload["display_group_id"])
                        ),
                        payload["isolate_failure"],
                        None,
                        payload["optional_cutoff_at_monotonic_s"],
                    )
                    requests[identity.provider_turn_id] = identity
                    await batch.begin_turn(request)
                elif call == "send_turn_audio":
                    ranges = tuple(_span(r) for r in payload["ranges"])
                    await batch.send_turn_audio(
                        identity,
                        b"\0\0" * payload["samples"],
                        payload_sequence=payload["payload_sequence"],
                        source_ranges=ranges,
                        context_only=payload["context_only"],
                    )
                elif call == "seal_turn":
                    await batch.seal_turn(
                        identity,
                        sealed_content_ranges=tuple(_span(r) for r in payload["sealed"]),
                        seal_reason=payload["reason"],
                        observed_trailing_silence_ms=payload["trailing_silence_ms"],
                    )
                else:
                    await batch.abort_turn(identity, reason=payload["reason"])
            for _ in range(5):
                await asyncio.sleep(0)
        await inner.queue.put(None)
        await asyncio.wait_for(consumer, timeout=5)
        return self._units(view, sorted(proof_ready.get("times", [])))

    def _units(self, view, evidence_times):
        from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView  # noqa: F401

        out = []
        released = 0.0
        keys = {}
        for identity in self.order:
            child = self.children[identity]
            if "native" not in child:
                continue
            released = max(released, child["native"])
            if not child["text"]:
                continue
            admitted = released - self.start
            readable = admitted + self.readable_delay
            attribution = None if child["speaker"] is None else child["speaker"].attribution
            settled_at = readable
            final_key = None if attribution is None else attribution.key
            state = "unknown" if attribution is None else attribution.state
            at_readable_key = final_key
            if final_key is None and child["proof"] is not None:
                for t in evidence_times:
                    if t - self.start < readable:
                        continue
                    if t - self.start > readable + 1.0:
                        break
                    resolved = view.resolve(child["proof"]).attribution
                    if resolved.key is not None:
                        final_key, state, settled_at = resolved.key, resolved.state, t - self.start
                        break
            spans = [
                (self.seconds(r.normalized_start_sample), self.seconds(r.normalized_end_sample))
                for r in child["ranges"]
                if r.normalized_start_sample is not None
            ]

            def assignment(key, state_):
                if key is None:
                    return units.Assignment(
                        state_ if state_ != "identified" else "unavailable", None, None, False
                    )
                name = str(key)
                if name not in keys:
                    keys[name] = len(keys)
                index = keys[name]
                return units.Assignment(
                    "identified", name, index if index < 4 else None, index >= 4
                )

            readable_assignment = assignment(at_readable_key, state)
            final = assignment(final_key, state)
            out.append(
                units.Unit(
                    str(identity.provider_turn_id),
                    (len(out),),
                    child["text"],
                    self.ref.language,
                    admitted,
                    spans,
                    readable,
                    readable_assignment,
                    settled_at,
                    final,
                    1,
                )
            )
        return out


def _latency_model(events):
    submitted = {}
    native = {}
    durations = {}
    for e in events:
        if e["event"] == "log":
            m = _SLICE.search(e.get("message", ""))
            if m:
                submitted[m.group("slice")] = float(m.group("submitted"))
        elif e["event"] == "policy_input" and e.get("call") == "_snapshot_native_event":
            if e.get("event_type") == "STTProviderTurnTerminal":
                native[e["identity"]["turn"]] = float(e["observed_at"])
    samples = [native[k] - submitted[k] for k in native if k in submitted]
    del durations
    base = statistics.median(samples) if samples else 0.4
    return lambda duration: base


def _readable_delay(events):
    admitted = {e["source_id"]: e["t"] for e in events if e["event"] == "admitted"}
    delays = [
        e["readable_at"] - admitted[e["source_id"]]
        for e in events
        if e["event"] == "first_readable" and e["source_id"] in admitted
    ]
    return statistics.median(delays) if delays else 0.05


def simulate_run(run_dir: Path, case_ids=None, gap_s=0.5):
    events = units.load_events(run_dir / "observations.jsonl")
    latency = _latency_model(events)
    readable = _readable_delay(events)
    results = {}
    for case_file in sorted((run_dir / "cases").glob("*.json")):
        case_id = case_file.stem
        status = json.loads(case_file.read_text(encoding="utf-8"))
        if status.get("status") != "completed" or (case_ids and case_id not in case_ids):
            continue
        ref = reference.load(REFERENCES / f"{case_id}.json", gap_s=gap_s)
        tag = status.get("observation_tag") or case_id
        sim = CaseSimulation(ref, events, tag, latency, readable)
        simulated_units = asyncio.run(sim.run())
        live = units.case_run(events, tag, max_frame_lag_s=10.0)
        run = units.CaseRun(
            case_id,
            sim.start,
            sim.lead,
            simulated_units,
            live.source_record,
            [],
            sim.slices,
            live.evidence,
        )
        results[case_id] = {
            "score": score_case(ref, run),
            "slices": sim.slices,
            "live_slices": live.slices,
        }
    return results


def _apply_overrides(pairs):
    for pair in pairs or ():
        target, _, value = pair.partition("=")
        module_name, _, attribute = target.rpartition(".")
        module = importlib.import_module(module_name)
        setattr(module, attribute, json.loads(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="runs/<route>/<arm>/<rep> directory of a recorded batch-route run",
    )
    parser.add_argument("--cases", type=lambda s: s.split(","))
    parser.add_argument(
        "--set", action="append", help="module.attribute=<json> override applied before simulation"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    sys.path.insert(0, str(BENCH_ROOT.parents[1] / "src"))
    _apply_overrides(args.set)
    results = simulate_run(Path(args.run_dir), args.cases)
    out = {}
    for case_id, r in results.items():
        s = r["score"]
        cont = [t for t in s["transitions"] if t["kind"] == "continuous"]
        out[case_id] = {
            "continuous": len(cont),
            "separated": sum(1 for t in cont if t.get("separated")),
            "early_release": sum(1 for t in cont if t.get("early_release")),
            "speaker_cuts_sim": [
                round(x.boundary_s, 3) for x in r["slices"] if x.reason == "speaker"
            ],
            "speaker_cuts_live": [
                round(x.boundary_s, 3) for x in r["live_slices"] if x.reason == "speaker"
            ],
            "attribution": s["attribution"],
        }
    text = json.dumps(out, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
