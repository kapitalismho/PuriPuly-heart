import json
import pathlib
import hashlib
import subprocess
import os
import time
import datetime
HERE = pathlib.Path(__file__).resolve().parent
EXE = pathlib.Path("C:/tmp/psem-vulkan-fp16-source/build-fp16-vulkan/bin/Release/transcribe-cli.exe")
MODEL = pathlib.Path("C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf")
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()
def run_one(root, key, wav, backend, dump, export=None):
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
    t0 = time.time()
    wall = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    t1 = time.time()
    (d / "stdout.txt").write_text(r.stdout)
    (d / "stderr.txt").write_text(r.stderr)
    meta = {"key": key, "cmd": cmd, "rc": r.returncode, "wall_start_utc": wall, "wall_dur_s": round(t1 - t0, 3), "dump": dump, "export": export, "preset": "low_latency", "backend_req": backend, "wav": str(wav), "wav_sha256": sha256_file(wav), "exe_sha256": sha256_file(EXE), "model_sha256": sha256_file(MODEL), "files": sorted([f.name for f in d.iterdir()])}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{key} rc={r.returncode} dur={t1-t0:.2f}s files={meta['files']}", flush=True)
    return meta
def main():
    clips = HERE / "clips"
    w6 = pathlib.Path("C:/tmp/psem-s6.wav")
    assert w6.exists()
    low = HERE / "results_low"
    metas = {}
    metas["v6_full"] = run_one(low, "v6_full", w6, "vulkan", True, "hidden,logits")
    metas["v6_repeat"] = run_one(low, "v6_repeat", w6, "vulkan", True, "hidden,logits")
    metas["v6_disabled"] = run_one(low, "v6_disabled", w6, "vulkan", False, None)
    metas["v6_hidden"] = run_one(low, "v6_hidden", w6, "vulkan", True, "hidden")
    metas["v6_logits"] = run_one(low, "v6_logits", w6, "vulkan", True, "logits")
    metas["v5p04s"] = run_one(low, "v5p04s", clips / "prefix_5p04s.wav", "vulkan", True, "hidden,logits")
    metas["v4s"] = run_one(low, "v4s", clips / "prefix_4s.wav", "vulkan", True, "hidden,logits")
    metas["v40s"] = run_one(low, "v40s", clips / "clip_40s.wav", "vulkan", True, "hidden,logits")
    (low / "metas.json").write_text(json.dumps(metas, indent=1))
    cpu = HERE / "results_cpu"
    cmetas = {}
    cmetas["c7_full"] = run_one(cpu, "c7_full", clips / "clip_7s.wav", "cpu", True, "hidden,logits")
    cmetas["c7_repeat"] = run_one(cpu, "c7_repeat", clips / "clip_7s.wav", "cpu", True, "hidden,logits")
    cmetas["c7_disabled"] = run_one(cpu, "c7_disabled", clips / "clip_7s.wav", "cpu", False, None)
    cmetas["c5p04s"] = run_one(cpu, "c5p04s", clips / "prefix_5p04s.wav", "cpu", True, "hidden,logits")
    (cpu / "metas.json").write_text(json.dumps(cmetas, indent=1))
    print("suite done")
if __name__ == "__main__":
    main()
