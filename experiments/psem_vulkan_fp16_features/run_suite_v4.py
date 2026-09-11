import argparse
import json
import pathlib
import hashlib
import subprocess
import os
import time
import datetime
HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_EXE = pathlib.Path("C:/tmp/psem-vulkan-fp16-source/build-fp16-vulkan/bin/Release/transcribe-cli.exe")
DEFAULT_MODEL = pathlib.Path("C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf")
EXE = pathlib.Path(os.environ.get("TRANSCRIBE_CLI_EXE", str(DEFAULT_EXE)))
MODEL = pathlib.Path(os.environ.get("TRANSCRIBE_SORTFORMER_MODEL", str(DEFAULT_MODEL)))
VEC = {"TRANSCRIBE_VK_NO_MUL_MAT_VEC": "1"}
F32HEAD = {"TRANSCRIBE_SORTFORMER_F32_HEAD": "1"}
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()
def run_one(root, key, wav, backend, dump, export=None, extra=None):
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("diar.*"):
        f.unlink()
    for f in ["stdout.txt", "stderr.txt", "meta.json"]:
        try:
            (d / f).unlink()
        except FileNotFoundError:
            pass
    cmd = [str(EXE), "-m", str(MODEL), "--backend", backend, str(wav)]
    env = dict(os.environ)
    env["TRANSCRIBE_SORTFORMER_STREAM_PRESET"] = "low_latency"
    if export is not None:
        env["TRANSCRIBE_SORTFORMER_EXPORT"] = export
    else:
        env.pop("TRANSCRIBE_SORTFORMER_EXPORT", None)
    if dump:
        env["TRANSCRIBE_DUMP_DIR"] = str(d)
    else:
        env.pop("TRANSCRIBE_DUMP_DIR", None)
    env.pop("TRANSCRIBE_VK_NO_MUL_MAT_VEC", None)
    env.pop("TRANSCRIBE_SORTFORMER_F32_HEAD", None)
    if extra:
        env.update(extra)
    t0 = time.time()
    wall = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    t1 = time.time()
    (d / "stdout.txt").write_text(r.stdout)
    (d / "stderr.txt").write_text(r.stderr)
    meta = {"key": key, "cmd": cmd, "rc": r.returncode, "wall_start_utc": wall, "wall_dur_s": round(t1 - t0, 3), "dump": dump, "export": export, "preset": "low_latency", "backend_req": backend, "wav": str(wav), "extra_env": extra or {}, "wav_sha256": sha256_file(wav), "exe_sha256": sha256_file(EXE), "model_sha256": sha256_file(MODEL), "files": sorted([f.name for f in d.iterdir()])}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{key} rc={r.returncode} dur={t1-t0:.2f}s files={meta['files']}", flush=True)
    return meta
def main():
    clips = HERE / "clips"
    prof = dict(VEC); prof.update(F32HEAD)
    v4 = HERE / "results_v4"
    m = {}
    m["v7_full"] = run_one(v4, "v7_full", clips / "clip_7s.wav", "vulkan", True, "hidden,logits", prof)
    m["v7_repeat"] = run_one(v4, "v7_repeat", clips / "clip_7s.wav", "vulkan", True, "hidden,logits", prof)
    m["v7_disabled"] = run_one(v4, "v7_disabled", clips / "clip_7s.wav", "vulkan", False, None, prof)
    m["v7_hidden"] = run_one(v4, "v7_hidden", clips / "clip_7s.wav", "vulkan", True, "hidden", prof)
    m["v7_logits"] = run_one(v4, "v7_logits", clips / "clip_7s.wav", "vulkan", True, "logits", prof)
    m["v7_defaulthead"] = run_one(v4, "v7_defaulthead", clips / "clip_7s.wav", "vulkan", True, "hidden,logits", dict(VEC))
    m["v6"] = run_one(v4, "v6", clips / "clip_6s.wav", "vulkan", True, "hidden,logits", prof)
    m["v8s"] = run_one(v4, "v8s", clips / "clip_8s.wav", "vulkan", True, "hidden,logits", prof)
    m["v20s"] = run_one(v4, "v20s", clips / "clip_20s.wav", "vulkan", True, "hidden,logits", prof)
    m["v5p04s"] = run_one(v4, "v5p04s", clips / "prefix_5p04s.wav", "vulkan", True, "hidden,logits", prof)
    m["v4s"] = run_one(v4, "v4s", clips / "prefix_4s.wav", "vulkan", True, "hidden,logits", prof)
    m["v40s"] = run_one(v4, "v40s", clips / "clip_40s.wav", "vulkan", True, "hidden,logits", prof)
    (v4 / "metas.json").write_text(json.dumps(m, indent=1))
    c4 = HERE / "results_cpu_v4"
    c = {}
    c["c7"] = run_one(c4, "c7", clips / "clip_7s.wav", "cpu", True, "hidden,logits", None)
    c["c7_f32head"] = run_one(c4, "c7_f32head", clips / "clip_7s.wav", "cpu", True, "hidden,logits", dict(F32HEAD))
    c["c7_disabled"] = run_one(c4, "c7_disabled", clips / "clip_7s.wav", "cpu", False, None, None)
    c["c5p04s"] = run_one(c4, "c5p04s", clips / "prefix_5p04s.wav", "cpu", True, "hidden,logits", None)
    (c4 / "metas.json").write_text(json.dumps(c, indent=1))
    print("suite v4 done")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(EXE))
    ap.add_argument("--model", default=str(MODEL))
    a = ap.parse_args()
    EXE = pathlib.Path(a.exe)
    MODEL = pathlib.Path(a.model)
    print(f"exe={EXE} sha={sha256_file(EXE)}")
    print(f"model={MODEL} sha={sha256_file(MODEL)}")
    main()
