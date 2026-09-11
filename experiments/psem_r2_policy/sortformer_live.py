from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from puripuly_heart.core.audio.psem_receiver import ProspectiveSpeakerHypothesis

FRAME_SAMPLES = 1280
CONFIRMATION_SAMPLES = 1600
SOURCE_HZ = 16000
TAU = 0.5
PACED_EXE = Path(r"C:/tmp/psem-e2o2-paced/bin/transcribe-cli.exe")
SORTFORMER_MODEL = Path(
    r"C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf"
)


def classify_masked(row: list[float] | tuple[float, ...]) -> int | str:
    assigned = [index for index, value in enumerate(row) if float(value) >= TAU]
    if len(assigned) == 1:
        return assigned[0]
    if len(assigned) > 1:
        return "OVERLAP"
    return "NONE"


@dataclass(slots=True)
class LiveEvidenceInterval:
    event_id: str
    start_sample: int
    end_sample: int
    relation: str
    available_at_monotonic_s: float
    kind: str


@dataclass(slots=True)
class NativeChunkReceipt:
    emit_start_frame: int
    start_sample: int
    end_sample: int
    available_at_monotonic_s: float
    label: int | str
    receipt_kind: str = "native_arrival"


@dataclass(slots=True)
class LiveTransitionEvent:
    event_id: str
    boundary: int
    frontier: int
    candidate_slot: int
    available_at_monotonic_s: float
    receipt_kind: str


class LiveTransitionDecoder:
    def __init__(self) -> None:
        self.last: int | None = None
        self.anchor_slot: int | None = None
        self.pending: int | None = None
        self.pend_start: int | None = None
        self.pend_n = 0
        self.prev_end: int | None = None
        self.seg_n = 0
        self.events: list[LiveTransitionEvent] = []
        self.evidence: list[LiveEvidenceInterval] = []
        self.chunk_receipts: list[NativeChunkReceipt] = []
        self.n_frames_seen = 0
        self._evidence_n = 0
        self._anchor_emitted = False

    def _evidence(
        self,
        *,
        start_sample: int,
        end_sample: int,
        relation: str,
        available_at_monotonic_s: float,
        kind: str,
    ) -> LiveEvidenceInterval:
        self._evidence_n += 1
        item = LiveEvidenceInterval(
            event_id=f"ev.{self._evidence_n}",
            start_sample=start_sample,
            end_sample=end_sample,
            relation=relation,
            available_at_monotonic_s=available_at_monotonic_s,
            kind=kind,
        )
        self.evidence.append(item)
        return item

    def ingest_chunk(
        self,
        emit_start_frame: int,
        rows: list[list[float]],
        *,
        available_at_monotonic_s: float,
        receipt_kind: str = "native_arrival",
    ) -> list[LiveTransitionEvent]:
        new: list[LiveTransitionEvent] = []
        for offset, row in enumerate(rows):
            frame = int(emit_start_frame) + offset
            self.n_frames_seen += 1
            start = frame * FRAME_SAMPLES
            end = (frame + 1) * FRAME_SAMPLES
            label = classify_masked(row)
            self.chunk_receipts.append(
                NativeChunkReceipt(
                    emit_start_frame=int(emit_start_frame),
                    start_sample=start,
                    end_sample=end,
                    available_at_monotonic_s=available_at_monotonic_s,
                    label=label,
                    receipt_kind=receipt_kind,
                )
            )
            if label in ("OVERLAP", "NONE"):
                self._evidence(
                    start_sample=start,
                    end_sample=end,
                    relation="UNKNOWN",
                    available_at_monotonic_s=available_at_monotonic_s,
                    kind="overlap" if label == "OVERLAP" else "none",
                )
                self.pending, self.pend_n, self.prev_end = None, 0, None
                continue
            assert isinstance(label, int)
            if self.last is None:
                self.last = label
                self.pending, self.pend_n, self.prev_end = None, 0, None
                if self.anchor_slot is None:
                    self.anchor_slot = label
                if not self._anchor_emitted:
                    self._evidence(
                        start_sample=start,
                        end_sample=end,
                        relation="CURRENT",
                        available_at_monotonic_s=available_at_monotonic_s,
                        kind="anchor",
                    )
                    self._anchor_emitted = True
                continue
            if label == self.last:
                self.pending, self.pend_n, self.prev_end = None, 0, None
                continue
            if self.prev_end is not None and start != self.prev_end:
                self.pending, self.pend_n = None, 0
                self.prev_end = None
            if self.pending is None or self.pending != label:
                self.pending = label
                self.pend_start = start
                self.pend_n = 0
                self.prev_end = start
            duration = end - start
            need = CONFIRMATION_SAMPLES - self.pend_n
            if duration >= need:
                self.seg_n += 1
                event = LiveTransitionEvent(
                    event_id=f"e.{self.seg_n}",
                    boundary=int(self.pend_start or start),
                    frontier=int(end),
                    candidate_slot=int(label),
                    available_at_monotonic_s=available_at_monotonic_s,
                    receipt_kind=receipt_kind,
                )
                self.events.append(event)
                new.append(event)
                self._evidence(
                    start_sample=int(self.pend_start or start),
                    end_sample=int(end),
                    relation="OTHER",
                    available_at_monotonic_s=available_at_monotonic_s,
                    kind="transition",
                )
                self.last = label
                self.pending, self.pend_n, self.prev_end = None, 0, None
                continue
            self.pend_n += duration
            self.prev_end = end
        return new


def hypothesis_from_live_event(
    event: LiveTransitionEvent,
    *,
    capture_epoch: int,
    producer_generation: object,
    reference_generation: object,
) -> ProspectiveSpeakerHypothesis:
    boundary = max(int(event.boundary), 0)
    return ProspectiveSpeakerHypothesis(
        hypothesis_id=event.event_id,
        revision=1,
        capture_epoch=capture_epoch,
        support_start_sample=max(boundary - 1, 0),
        support_end_sample=max(boundary, 1),
        estimated_transition_sample=boundary,
        observed_frontier_sample=max(int(event.frontier), boundary),
        available_at_monotonic_s=event.available_at_monotonic_s,
        producer_generation=producer_generation,
        reference_generation=reference_generation,
        producer_valid=True,
        reference_valid=True,
        local_slot=event.candidate_slot,
    )


def hypothesis_at_boundary(
    boundary: int,
    *,
    capture_epoch: int,
    available_at_monotonic_s: float,
    producer_generation: object,
    reference_generation: object,
    hypothesis_id: str | None = None,
    local_slot: int | None = 1,
) -> ProspectiveSpeakerHypothesis:
    sample = max(int(boundary), 0)
    return ProspectiveSpeakerHypothesis(
        hypothesis_id=hypothesis_id or f"h-{uuid4().hex[:8]}",
        revision=1,
        capture_epoch=capture_epoch,
        support_start_sample=max(sample - 1, 0),
        support_end_sample=max(sample, 1),
        estimated_transition_sample=sample,
        observed_frontier_sample=max(sample, 1),
        available_at_monotonic_s=available_at_monotonic_s,
        producer_generation=producer_generation,
        reference_generation=reference_generation,
        producer_valid=True,
        reference_valid=True,
        local_slot=local_slot,
    )


def evidence_payload(
    item: LiveEvidenceInterval,
    *,
    capture_epoch: int,
    producer_generation: object,
    reference_generation: object,
    reference_valid: bool = True,
) -> dict[str, Any]:
    return {
        "capture_epoch": capture_epoch,
        "start_sample": item.start_sample,
        "end_sample": item.end_sample,
        "available_at_monotonic_s": item.available_at_monotonic_s,
        "relation": item.relation,
        "producer_generation": producer_generation,
        "reference_generation": reference_generation,
        "reference_valid": reference_valid,
        "kind": item.kind,
        "event_id": item.event_id,
    }

class NativeSortformerProducer:
    def __init__(
        self,
        wav_path: str | Path,
        *,
        exe: Path = PACED_EXE,
        model: Path = SORTFORMER_MODEL,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.wav_path = Path(wav_path)
        self.exe = Path(exe)
        self.model = Path(model)
        self._clock = clock or time.monotonic
        self._proc: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._buf = b""
        self.decoder = LiveTransitionDecoder()
        self.tcp_lines: list[dict[str, Any]] = []
        self._evidence_i = 0

    @property
    def available(self) -> bool:
        return self.exe.is_file() and self.model.is_file() and self.wav_path.is_file()

    def start(self) -> None:
        if not self.available:
            raise FileNotFoundError("native Sortformer producer binary or model is missing")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(30)
        port = server.getsockname()[1]
        env = os.environ.copy()
        env["TRANSCRIBE_PSEM_PACE_16KHZ"] = "1"
        env["TRANSCRIBE_PSEM_EVENTS_TCP"] = f"127.0.0.1:{port}"
        env["TRANSCRIBE_PSEM_CAUSAL_FRONTEND"] = "1"
        self._proc = subprocess.Popen(
            [str(self.exe), "-m", str(self.model), "--backend", "vulkan", str(self.wav_path)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._sock = server
        self._conn, _peer = server.accept()
        self._conn.settimeout(0.05)

    def poll(self) -> list[LiveTransitionEvent]:
        conn = self._conn
        if conn is None:
            return []
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            return []
        except OSError:
            return []
        if not chunk:
            return []
        self._buf += chunk
        arrived: list[LiveTransitionEvent] = []
        now = self._clock()
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            if not raw.strip():
                continue
            message = json.loads(raw.decode("utf-8"))
            message["_receipt_monotonic_s"] = now
            self.tcp_lines.append(message)
            rows = message.get("probs") or message.get("rows") or []
            start_frame = int(message.get("emit_start_frame") or message.get("start_frame") or 0)
            if rows:
                arrived.extend(
                    self.decoder.ingest_chunk(
                        start_frame,
                        rows,
                        available_at_monotonic_s=now,
                        receipt_kind="native_arrival",
                    )
                )
        return arrived

    def drain_evidence(self) -> list[LiveEvidenceInterval]:
        items = self.decoder.evidence[self._evidence_i :]
        self._evidence_i = len(self.decoder.evidence)
        return items

    def chunk_payloads(self) -> list[dict[str, Any]]:
        return [
            {
                "start_sample": item.start_sample,
                "end_sample": item.end_sample,
                "available_at_monotonic_s": item.available_at_monotonic_s,
                "label": item.label,
                "receipt_kind": item.receipt_kind,
                "emit_start_frame": item.emit_start_frame,
            }
            for item in self.decoder.chunk_receipts
        ]

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def iter_live_events(producer: NativeSortformerProducer) -> Iterator[LiveTransitionEvent]:
    while True:
        events = producer.poll()
        if events:
            yield from events
        elif producer._proc is not None and producer._proc.poll() is not None:
            leftover = producer.poll()
            if leftover:
                yield from leftover
            return
        else:
            time.sleep(0.01)
