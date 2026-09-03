from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.adapter.protocol import BindingError
from experiments.psem_small_model_probe.adapter.firered import FireRedAdapter
from experiments.psem_small_model_probe.adapter.neovad import NeoVADAdapter

ADAPTER = Path(__file__).resolve().parent
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def synth_pcm16(seconds: float, freq_hz: float = 440.0) -> bytes:
    n = int(16000 * seconds)
    return b"".join(
        struct.pack("<h", int(10000 * math.sin(2 * math.pi * freq_hz * i / 16000)))
        for i in range(n)
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_vendor_hashes() -> None:
    vendor = json.loads((ADAPTER / "vendor.json").read_text())
    for key, local in (
        ("pvad_onnx", "pvad.onnx"),
        ("neovad_gru_pt", "neovad_gru.pt"),
        ("neovad_gru_yaml", "neovad_gru.yaml"),
    ):
        p = ADAPTER / "vendor" / local
        ok = p.exists() and sha(p) == vendor["artifacts"][key]["sha256"]
        check(f"vendor-hash:{local}", ok, f"{vendor['artifacts'][key]['sha256'][:12]}")


def contract_checks(name: str, make) -> None:
    try:
        make().bind(b"\x00" * 320)
        check(f"{name}:bind-before-reset", False, "no RuntimeError")
    except RuntimeError:
        check(f"{name}:bind-before-reset", True, "RuntimeError")
    except Exception as exc:
        check(f"{name}:bind-before-reset", False, repr(exc)[:100])
    a = make()
    a.reset()
    for label, blob in (("empty", b""), ("short", b"\x00" * 320)):
        try:
            a.bind(blob)
            check(f"{name}:bind-{label}", False, "no BindingError")
        except BindingError:
            check(f"{name}:bind-{label}", True, "BindingError")
        except Exception as exc:
            check(f"{name}:bind-{label}", False, repr(exc)[:100])


def main() -> int:
    check_vendor_hashes()
    span = synth_pcm16(1.0)
    audio_note = "synthetic-only: manifest rows carry no local audio paths"

    try:
        fr_probe = FireRedAdapter()
        check("firered:init-pinned", True, f"ECAPA cache {fr_probe.ecapa_dir}")
        live_fr = True
    except FileNotFoundError as exc:
        check("firered:init-pinned", "ECAPA" in str(exc), f"error-path: {exc}"[:140])
        live_fr = False
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp)
        (stub / "pvad.onnx").write_bytes(b"\x00" * 64)
        (stub / "spkrec-ecapa-voxceleb").mkdir()
        (stub / "spkrec-ecapa-voxceleb" / "embedding_model.ckpt").write_bytes(b"\x00" * 64)
        make = lambda: FireRedAdapter(weights_dir=stub, ecapa_dir=stub / "spkrec-ecapa-voxceleb")
        contract_checks("firered-stubweights", make)
        a = make()
        a.reset()
        try:
            a.bind(span)
            check("firered-stub:bind-blocked", False, "stub weights unexpectedly loadable")
        except Exception as exc:
            check("firered-stub:bind-blocked", True, f"stub not loadable: {type(exc).__name__}")
        try:
            a.step(b"\x00" * 320)
            check("firered:step-guard", False, "expected RuntimeError (unbound)")
        except RuntimeError:
            check("firered:step-guard", True, "RuntimeError before bind")
        except Exception as exc:
            check("firered:step-guard", False, repr(exc)[:100])
    if live_fr:
        import time
        fr = FireRedAdapter()
        try:
            fr.bind(b"\x00" * 320)
            check("firered-live:bind-before-reset", False, "no RuntimeError")
        except RuntimeError:
            check("firered-live:bind-before-reset", True, "RuntimeError")
        except Exception as exc:
            check("firered-live:bind-before-reset", False, repr(exc)[:100])
        fr.reset()
        t0 = time.perf_counter()
        fr.bind(span)
        bind_ms = (time.perf_counter() - t0) * 1000.0
        h = fr.episode_header()
        ok = (
            h["model"] == "fireredchat-pvad"
            and h["reset_ok"] is True
            and h["bind_span_hash"] == hashlib.sha256(span).hexdigest()
            and isinstance(h["ecapa_sha"], str)
            and len(h["ecapa_sha"]) == 64
            and h["onnx_sha"] == h["model_sha"]
        )
        check("firered-live:bind", ok, f"bind={bind_ms:.0f}ms ecapa={h['ecapa_sha'][:12]}")
        chunk = span[:3200]
        dts: list[float] = []
        anchors: list[float] = []
        speeches: list = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = fr.step(chunk)
            dts.append((time.perf_counter() - t0) * 1000.0)
            anchors.append(out.anchor)
            speeches.append(out.speech)
        dts.sort()
        med = dts[len(dts) // 2]
        p95 = dts[min(len(dts) - 1, int(0.95 * len(dts)))]
        ok = (
            all(s is None for s in speeches)
            and all(0.0 <= v <= 1.0 for v in anchors)
            and len(fr.frames) == 20 * (len(chunk) // 320)
        )
        check(
            "firered-live:step",
            ok,
            f"median={med:.2f}ms p95={p95:.2f}ms RTF={med / 100.0:.3f} anchor0={anchors[0]:.3f}"
            " (chunk=100ms x20; informational, Gate 3/4 owns RTF<=0.25)",
        )
        print("firered: live reset→bind→step PASS; " + audio_note)
    else:
        print(
            "firered: live reset→bind→step BLOCKED (no local ECAPA cache, "
            "no onnxruntime/speechbrain); " + audio_note
        )

    try:
        b = NeoVADAdapter()
        check("neovad:init-pinned", True, f"weights {b.weights_path.name}")
    except Exception as exc:
        check("neovad:init-pinned", False, repr(exc)[:140])
        failed = [r for r in results if not r[1]]
        print(f"neovad: FAIL ({len(failed)} failing checks); " + audio_note)
        return 1
    contract_checks("neovad", NeoVADAdapter)
    b.reset()
    try:
        b.bind(span)
        h = b.episode_header()
        vendor = json.loads((ADAPTER / "vendor.json").read_text())
        ok = (
            h["model"] == "neovad-gru"
            and h["model_sha"] == vendor["artifacts"]["neovad_gru_pt"]["sha256"]
            and h["onnx_sha"] == "none"
            and h["ecapa_sha"] == "none"
            and h["reset_ok"] is True
            and h["bind_span_hash"] == hashlib.sha256(span).hexdigest()
        )
        check("neovad:bind-header", ok, f"bind_span_hash={h['bind_span_hash'][:12]}")
    except Exception as exc:
        check("neovad:bind-header", False, repr(exc)[:140])
    try:
        out = b.step(b"\x00" * 320)
        aux = out.aux
        s = aux["p_nonspeech"] + aux["p_primary"] + aux["p_secondary"]
        ok = (
            abs(s - 1.0) < 1e-3
            and 0.0 <= out.anchor <= 1.0
            and out.speech is not None
            and abs(out.speech - (1.0 - aux["p_nonspeech"])) < 1e-6
        )
        check("neovad:step-live", ok, f"live anchor={out.anchor:.3f} speech={out.speech:.3f}")
        import time
        dts = []
        for _ in range(50):
            t0 = time.perf_counter()
            b.step(b"\x00" * 320)
            dts.append((time.perf_counter() - t0) * 1000.0)
        dts.sort()
        med = dts[len(dts) // 2]
        p95 = dts[min(len(dts) - 1, int(0.95 * len(dts)))]
        check(
            "neovad:step-timing",
            True,
            f"median={med:.2f}ms p95={p95:.2f}ms RTF={med / 10.0:.3f} (informational; Gate 3/4 owns RTF<=0.25)",
        )
    except RuntimeError as exc:
        check("neovad:step-live", "torch" in str(exc), f"error-path: {exc}"[:140])
    except Exception as exc:
        check("neovad:step-live", False, f"{type(exc).__name__}: {exc}"[:140])

    failed = [r for r in results if not r[1]]
    firered_ok = all(ok for n, ok, _ in results if n.startswith("firered") or n.startswith("vendor"))
    neovad_ok = all(ok for n, ok, _ in results if n.startswith("neovad"))
    print(f"FireRed: {'PASS' if firered_ok else 'FAIL'} (stub contracts + live bind/step on synthetic PCM)")
    print(f"NeoVAD: {'PASS' if neovad_ok else 'FAIL'} (bind + live step on synthetic PCM)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
