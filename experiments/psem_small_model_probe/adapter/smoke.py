from __future__ import annotations

import hashlib
import json
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.adapter.decoder import CommonPersistenceDecoder
from experiments.psem_small_model_probe.adapter.protocol import (
    BindingError,
    frame_bytes,
    load_wav_mono16k,
    validate_pcm16_chunk,
)
from experiments.psem_small_model_probe.adapter.stub import StubAdapter

FRAME_MS = 20
TAU_SMOKE = 0.5
N_STEPS = 64
TIMING_N = 2000

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def zeros_frame() -> bytes:
    return b"\x00" * frame_bytes(FRAME_MS)


def run_episode(adapter: StubAdapter, episode_id: str, span: bytes) -> tuple[dict, list[dict]]:
    adapter.reset()
    reset_ok = adapter._step_index == 0 and adapter._source_time_ms == 0
    adapter.bind(span)
    header = {"model": "stub", "model_sha": "none", "onnx_sha": "none",
              "reset_ok": reset_ok, "bind_span_hash": adapter.bind_span_hash}
    rows = []
    for _ in range(N_STEPS):
        wall = time.perf_counter_ns() // 1_000_000
        out = adapter.step(zeros_frame())
        rows.append({"episode_id": episode_id, "model": "stub", "regime": "smoke",
                     "frame_ms": FRAME_MS, "source_time_ms": out.source_time_ms,
                     "wall_time_ms": wall, "speech": out.speech, "anchor": out.anchor,
                     "aux": out.aux, "lifecycle": "BOUND", "action": "n/a"})
    return header, rows


def log_hash(header: dict, rows: list[dict]) -> str:
    canon = json.dumps({"header": header,
                        "rows": [{k: r[k] for k in r if k != "wall_time_ms"} for r in rows]},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def test_reset_isolation() -> None:
    span_a = b"\x01" * (1000 * 16 * 2)
    span_b = b"\x02" * (1000 * 16 * 2)
    adapter = StubAdapter(frame_ms=FRAME_MS, anchor_pattern=(0.0, 1.0))
    h1, r1 = run_episode(adapter, "A", span_a)
    run_episode(adapter, "B", span_b)
    h2, r2 = run_episode(adapter, "A", span_a)
    check("reset-isolation", log_hash(h1, r1) == log_hash(h2, r2),
          f"hash {log_hash(h1, r1)[:12]} vs {log_hash(h2, r2)[:12]}")


def test_source_time() -> None:
    adapter = StubAdapter(frame_ms=FRAME_MS, anchor_pattern=(1.0,))
    adapter.reset()
    adapter.bind(b"\x01" * (1000 * 16 * 2))
    times = [adapter.step(zeros_frame()).source_time_ms for _ in range(8)]
    strict = all(b - a == FRAME_MS for a, b in zip([0] + times, times))
    chunk_ok = len(zeros_frame()) // 2 == 16000 * FRAME_MS // 1000
    dec = CommonPersistenceDecoder(frame_ms=FRAME_MS)
    outs = [dec.update({"speech_gt": True, "anchor": 0.0, "lifecycle": "BOUND",
                        "source_time_ms": (i + 1) * FRAME_MS}, tau=TAU_SMOKE)
            for i in range(30)]
    cuts = [o for o in outs if o["action"] == "CUT"]
    inv = all(c["source_boundary_time"] <= c["decision_time"] for c in cuts)
    check("source-time", strict and chunk_ok and len(cuts) == 1 and inv,
          f"times={times[:3]}..{times[-1]}, cuts={len(cuts)}, invariant={inv}")


def write_wav(path: Path, framerate: int, channels: int, sampwidth: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setframerate(framerate)
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.writeframes(b"\x00" * framerate * channels * sampwidth)


def test_mono16k_contract() -> None:
    ok = True
    with tempfile.TemporaryDirectory() as d:
        cases = [("8k.wav", 8000, 1, 2), ("stereo.wav", 16000, 2, 2), ("8bit.wav", 16000, 1, 1)]
        for name, sr, ch, sw in cases:
            p = Path(d) / name
            write_wav(p, sr, ch, sw)
            try:
                load_wav_mono16k(p)
                ok = False
            except ValueError:
                pass
    for bad, kw in [(b"", {}), (b"\x00" * 3, {}), (b"\x00" * 10, {})]:
        try:
            validate_pcm16_chunk(bad, **{"frame_ms": FRAME_MS} | kw)
            ok = False
        except ValueError:
            pass
    try:
        validate_pcm16_chunk(zeros_frame(), frame_ms=FRAME_MS, sample_rate_hz=8000)
        ok = False
    except ValueError:
        pass
    adapter = StubAdapter(frame_ms=FRAME_MS)
    try:
        adapter.bind(b"\x01" * (1000 * 16 * 2))
        ok = False
    except RuntimeError:
        pass
    adapter.reset()
    try:
        adapter.bind(b"")
        ok = False
    except BindingError:
        pass
    check("16k-mono-contract", ok, "8k/stereo/8bit + bad chunks + bind-before-reset + empty bind")


def test_no_smoothing() -> None:
    pattern = (0.0, 1.0)
    adapter = StubAdapter(frame_ms=FRAME_MS, anchor_pattern=pattern)
    adapter.reset()
    adapter.bind(b"\x01" * (1000 * 16 * 2))
    outs = [adapter.step(zeros_frame()).anchor for _ in range(8)]
    expected = [pattern[i % 2] for i in range(8)]
    check("no-smoothing", outs == expected, f"anchors={outs}")


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(int(q / 100 * len(s)), len(s) - 1)]


def test_cpu_scaffold() -> None:
    adapter = StubAdapter(frame_ms=FRAME_MS)
    adapter.reset()
    t0 = time.perf_counter()
    adapter.bind(b"\x01" * (1000 * 16 * 2))
    bind_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    adapter.reset()
    reset_ms = (time.perf_counter() - t0) * 1000
    adapter.reset()
    adapter.bind(b"\x01" * (1000 * 16 * 2))
    chunk = zeros_frame()
    adapter.step(chunk)
    samples = []
    for _ in range(TIMING_N):
        t0 = time.perf_counter()
        adapter.step(chunk)
        samples.append((time.perf_counter() - t0) * 1000)
    med, p95, p99 = statistics.median(samples), pct(samples, 95), pct(samples, 99)
    audio_ms = TIMING_N * FRAME_MS
    rtf = sum(samples) / audio_ms
    check("cpu-scaffold", True,
          f"stub step median={med:.4f}ms p95={p95:.4f}ms p99={p99:.4f}ms "
          f"RTF={rtf:.4f} bind={bind_ms:.3f}ms reset={reset_ms:.3f}ms "
          f"(real-model timing deferred to Gate 3/4)")


def test_decoder_edges() -> None:
    dec = CommonPersistenceDecoder(frame_ms=FRAME_MS)
    f = lambda **kw: {"speech_gt": True, "anchor": 1.0, "lifecycle": "BOUND",
                       "source_time_ms": FRAME_MS} | kw
    gates = [
        (dec.update(f(speech_gt=False)), "KEEP"),
        (dec.update(f(speech_gt=None)), "KEEP"),
        (dec.update(f(lifecycle="UNBOUND")), "HOLD"),
        (dec.update(f(lifecycle="UNCERTAIN")), "HOLD"),
        (dec.update(f(lifecycle="POISONED")), "HOLD"),
        (dec.update(f(anchor=0.0)), "HOLD"),
        (dec.update(f(anchor=1.0)), "KEEP"),
    ]
    ok = all(o["action"] == want and o["source_boundary_time"] is None for o, want in gates)
    dec.reset()
    seq = [dec.update({"speech_gt": True, "anchor": 0.0, "lifecycle": "BOUND",
                       "source_time_ms": (i + 1) * FRAME_MS}, tau=TAU_SMOKE)
           for i in range(15)]
    sens = [o for o in seq if o["action"] == "CUT_SENS"]
    ok = ok and len(sens) > 0 and all(o["sensitivity"] for o in sens)
    ok = ok and all(o["action"] != "CUT" for o in seq)
    check("decoder-edges", ok,
          f"gates={[o['action'] for o, _ in gates]}, sens_frames={len(sens)}, tau={TAU_SMOKE}")


def main() -> int:
    test_reset_isolation()
    test_source_time()
    test_mono16k_contract()
    test_no_smoothing()
    test_decoder_edges()
    test_cpu_scaffold()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} smoke checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
