"""PSEM Phase A headroom observations — durable reproducibility driver.

Covers the exact executed path without requiring re-inference (~24 min GPU):
  --smoke        fast verification only (no GPU, no build, no inference)
  --prefixes     recreate five [0,prefix_end) WAVs from canonical corpus + verify SHAs
  --patch-check  verify timing patch clean-reapplies byte-identical on temp copies
  --harvest      rebuild OBSERVATIONS payload from frozen per-source outputs -> --out,
                 compare hash to frozen OBSERVATIONS.json (no inference)

Full rebuild+rerun (documented, NOT run by smoke):
  1. extract C:/tmp/psem-vulkan-fp16-source/d42c3bb.zip -> C:/tmp/psem-phase-a-vulkan/
  2. patch -p1 -d <src> -i <each of 3 pinned patches> (patch.exe from Git usr/bin)
  3. patch -p1 -d <src> -i observations/psem_phase_a_timing.patch
  4. cmake -G "Visual Studio 17 2022" -A x64 -DTRANSCRIBE_VULKAN=ON
     -DCMAKE_CXX_FLAGS=/utf-8 -DCMAKE_C_FLAGS=/utf-8
     -DVulkan_INCLUDE_DIR=C:/VulkanSDK/1.4.350.0/include
     -DVulkan_LIBRARY=C:/VulkanSDK/1.4.350.0/Lib/vulkan-1.lib
     -S <src> -B C:/tmp/psem-phase-a-vulkan/build
  5. cmake --build C:/tmp/psem-phase-a-vulkan/build --config Release --target transcribe-cli -j 8
  6. per source: env TRANSCRIBE_SORTFORMER_STREAM_PRESET=low_latency,
     TRANSCRIBE_SORTFORMER_EXPORT=hidden,logits, TRANSCRIBE_DUMP_DIR=<obs/sid>,
     TRANSCRIBE_VK_NO_MUL_MAT_VEC=1, TRANSCRIBE_SORTFORMER_F32_HEAD=1,
     TRANSCRIBE_PSEM_CAUSAL_FRONTEND=1;
     transcribe-cli -m <model> --backend vulkan <prefix.wav>
"""
import argparse
import contextlib
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
REPO = pathlib.Path(__file__).resolve().parents[3]

OBS = REPO / "experiments" / "psem_phase_a_headroom" / "observations"
SRC_ZIP = pathlib.Path(r"C:/tmp/psem-vulkan-fp16-source/d42c3bb.zip")
OLD_BASE = pathlib.Path(r"C:/tmp/psem-vulkan-fp16-source/transcribe.cpp-d42c3bbdfa2f63c37e5891e27de47a612d62f221")
MODEL = pathlib.Path(r"C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf")
NEW_EXE = pathlib.Path(r"C:/tmp/psem-phase-a-vulkan/build/bin/Release/transcribe-cli.exe")
PREFIX_DIR = pathlib.Path(r"C:/tmp/psem-phase-a-prefixes")
PATCH_EXE = pathlib.Path(r"C:\Program Files\Git\usr\bin\patch.exe")
CORPUS = pathlib.Path(r"C:/Users/salee/.psem-corpus/ami/audio")

PREFIX_ENDS = {
    "ES2009c": 19872000, "ES2009d": 33536000, "ES2002b": 2684160,
    "ES2009a": 9144320, "EN2009d": 774400,
}
SID_OF = {"ES2009c": "ami_ES2009c", "ES2009d": "ami_ES2009d", "ES2002b": "ami_ES2002b",
          "ES2009a": "ami_ES2009a", "EN2009d": "ami_EN2009d"}
PREFIX_SHA = {
    "ES2009c": "44ee37081d6482d37adba737f79c570ce58552b1f129421e496bd135d388cf8a",
    "ES2009d": "92ca6da91afc70043df5a4af23d5adf0f2d0498f49123146262fdaf735f92028",
    "ES2002b": "18a04ad1d3d4cbc3640df81c4409140514136d85b6f32495ec23526c5600e163",
    "ES2009a": "702af1850e9340e809b526c32de3fe55ebea50561aadefab225cfb39da680275",
    "EN2009d": "44cdf330e7d33d7e7052ebd8235b0074e98366358b39cd776c77915a413f1b71",
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fail(msg):
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def check_freeze_bounds():
    fr = json.loads((OBS / "FREEZE.json").read_text())
    fmap = {s["source_id"]: s for s in fr["sources"]}
    for m, pe in PREFIX_ENDS.items():
        sid = SID_OF[m]
        e = fmap[sid]
        assert e["prefix_end_sample"] == pe, (sid, e["prefix_end_sample"], pe)
        assert pe <= e["source_total_samples"], sid
        assert pe % 1280 == 0, sid
        # exact rule: ceil1280(max+18240)
        assert pe == ((e["max_end"] + 18240 + 1279) // 1280) * 1280, sid
    print("freeze bounds OK (5 prefixes, ceil1280(max+16640+1600), all fit)")


def check_patch_reapply(verbose=False):
    patchfile = (OBS / "psem_phase_a_timing.patch").resolve()
    assert patchfile.exists(), "timing patch missing"
    rels = ["src/transcribe-mel.h", "src/transcribe-mel.cpp", "src/arch/sortformer/model.cpp"]
    new_base = pathlib.Path(r"C:/tmp/psem-phase-a-vulkan/transcribe.cpp-d42c3bbdfa2f63c37e5891e27de47a612d62f221")
    tmp = pathlib.Path(r"C:/tmp/psem-phase-a-repro-check")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for rel in rels:
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(OLD_BASE / rel, dst)
    exe = str(PATCH_EXE) if PATCH_EXE.exists() else "patch"
    r = subprocess.run([exe, "-p1", "-d", str(tmp), "-i", str(patchfile)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"patch reapply rc={r.returncode}: {(r.stderr or '')[-1500:]}")
    for rel in rels:
        if (tmp / rel).read_bytes() != (new_base / rel).read_bytes():
            fail(f"reapply mismatch {rel}")
    if verbose:
        print("timing patch clean-reapply byte-identical (3 files)")
    else:
        print("patch reapply OK")
    shutil.rmtree(tmp, ignore_errors=True)


def check_hashes():
    assert sha_file(SRC_ZIP) == "0f695cbfc2f0a28908dc287b3547341b869c0e9b13637b8473fcf5afe85a913a", "source zip sha"
    assert sha_file(MODEL) == "62faec7b99ad23e323087597604b50728abe85089b6364970b019a845547bf99", "model sha"
    assert sha_file(NEW_EXE) == "3706db55ecf78c09188516cde829eccfcc04373a069d4f1834be0be22597540d", "exe sha"
    print("pinned hashes OK (zip/model/exe)")


def check_traces():
    total_rows = 0
    for m, pe in PREFIX_ENDS.items():
        sid = SID_OF[m]
        d = OBS / sid
        tr = json.loads((d / "diar.trace.json").read_text())
        assert tr["pcm_samples"] == pe, sid
        assert tr["used_n"] == pe // 1280, sid
        tot = 0
        maxabs = 0.0
        for c in tr["chunks"]:
            assert c["emit_start_frame"] == tot, (sid, c["chunk"])
            tot += c["emit_count"]
            maxabs = max(maxabs, float(c.get("mel_parity_maxabs", 0)))
            assert c["raw_support_end_sample"] == (c["win_hi"] - 1) * 160 + 256, sid
        assert tot == tr["used_n"], sid
        assert maxabs == 0.0, (sid, maxabs)
        assert tr["chunks"][-1]["raw_support_end_sample"] == pe + 96, (sid, "tail pad")
        hj = json.loads((d / "diar.hidden.json").read_text())
        lj = json.loads((d / "diar.logits.json").read_text())
        assert hj["shape"] == [pe // 1280, 192], sid
        assert lj["shape"] == [pe // 1280, 4], sid
        total_rows += len(tr["chunks"])
    assert total_rows == 8597, total_rows
    print(f"traces OK (8597 chunk rows, continuity + exact parity + tail N+96)")


def cmd_prefixes():
    PREFIX_DIR.mkdir(parents=True, exist_ok=True)
    for m, pe in PREFIX_ENDS.items():
        src = CORPUS / m / f"{m}.Mix-Headset.wav"
        dst = PREFIX_DIR / f"{m}_prefix_{pe}.wav"
        with contextlib.closing(wave.open(str(src), "rb")) as w:
            params = w.getparams()
            assert params.framerate == 16000 and params.nchannels == 1 and params.sampwidth == 2
            frames = w.readframes(pe)
            assert len(frames) == pe * 2
        with contextlib.closing(wave.open(str(dst), "wb")) as w:
            w.setparams(params)
            w.writeframes(frames)
        got = sha_file(dst)
        assert got == PREFIX_SHA[m], (m, got)
        print(f"{m} prefix {pe} sha OK")


def harvest_to(out_path):
    fr = json.loads((OBS / "FREEZE.json").read_text())
    fmap = {s["source_id"]: s for s in fr["sources"]}
    frozen = json.loads((OBS / "OBSERVATIONS.json").read_text())
    sources = []
    for m, pe in PREFIX_ENDS.items():
        sid = SID_OF[m]
        d = OBS / sid
        tr = json.loads((d / "diar.trace.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        fz = fmap[sid]
        feats = {}
        for kind, n, fname in [("hidden", 192, "diar.hidden.f32"), ("logits", 4, "diar.logits.f32"),
                               ("probs", 4, "diar.probs.f32")]:
            j = json.loads((d / fname.replace(".f32", ".json")).read_text())
            feats[kind] = {"path": f"experiments/psem_phase_a_headroom/observations/{sid}/{fname}",
                           "sha256": sha_file(d / fname), "shape": j["shape"], "dtype": "float32"}
        chunks = [{"index": c["chunk"], "emit_start_frame": c["emit_start_frame"],
                   "emit_count": c["emit_count"], "raw_support_end_sample": c["raw_support_end_sample"],
                   "service_us": c["service_us"], "frontend_us": c["frontend_us"],
                   "graph_a_us": c["graph_a_us"], "graph_b_us": c["graph_b_us"],
                   "host_us": c["host_us"], "mel_parity_maxabs": c["mel_parity_maxabs"],
                   "win_lo": c["win_lo"], "win_hi": c["win_hi"], "M": c["M"],
                   "T_diar": c["T_diar"], "S": c["S"], "F": c["F"], "T_concat": c["T_concat"],
                   "lc": c["lc"], "rc": c["rc"], "C": c["C"], "base": c["base"],
                   "emit_global": c["emit_global"]} for c in tr["chunks"]]
        pad = tr["chunks"][-1]["raw_support_end_sample"] - pe
        sources.append({"source_id": sid, "sample_rate": 16000,
                        "source_audio_sha256": fz["source_audio_sha256"],
                        "prefix_end_sample": pe, "source_total_samples": fz["source_total_samples"],
                        "prefix_wav_sha256": meta["wav_sha256"], "feature_files": feats,
                        "trace_path": f"experiments/psem_phase_a_headroom/observations/{sid}/diar.trace.json",
                        "trace_sha256": sha_file(d / "diar.trace.json"), "chunks": chunks,
                        "initialization_us": tr.get("initialization_us"),
                        "mel_full_diagnostic_us": tr.get("mel_full_diagnostic_us"),
                        "sched_setup_us": tr.get("sched_setup_us"), "load_us": tr.get("load_us"),
                        "valid_native_frames": tr["used_n"],
                        "tail_support_status": f"eos_zero_pad_{pad}_samples_allowed_only_EOS_known" if pad > 0 else "no_pad",
                        "wall_dur_s_actual": meta["wall_dur_s"]})
    regen = {"schema": "phase_a.observations.v1", "profile": frozen["profile"],
             "sources": sources, "validation": frozen["validation"]}
    out_path = pathlib.Path(out_path)
    out_path.write_text(json.dumps(regen, indent=1))
    got = sha_file(out_path)
    want = sha_file(OBS / "OBSERVATIONS.json")
    print(f"regen {out_path} sha {got}")
    print(f"frozen OBSERVATIONS.json sha {want}")
    if got != want:
        fail("harvest regen hash differs from frozen interface (do not edit frozen file; escalate)")
    print("harvest regen hash MATCHES frozen interface")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prefixes", action="store_true")
    ap.add_argument("--patch-check", action="store_true")
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--out", default=r"C:/tmp/OBS.regen.json")
    a = ap.parse_args()
    if not (a.smoke or a.prefixes or a.patch_check or a.harvest):
        a.smoke = True
    if a.smoke:
        check_freeze_bounds()
        check_patch_reapply()
        check_traces()
        frozen = OBS / "OBSERVATIONS.json"
        print(f"OBSERVATIONS.json sha {sha_file(frozen)} bytes {frozen.stat().st_size}")
        print("SMOKE PASS (no GPU/build/inference)")
    if a.patch_check:
        check_patch_reapply(verbose=True)
    if a.prefixes:
        cmd_prefixes()
    if a.harvest:
        check_hashes()
        harvest_to(a.out)


if __name__ == "__main__":
    main()
