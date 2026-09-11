import json
import pathlib
import numpy as np
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import project_head as P
HERE = pathlib.Path(__file__).resolve().parent
V4 = HERE / "results_v4"
C4 = HERE / "results_cpu_v4"
W = P.load_weights()
W1, b1 = W["diar.fc1.weight"], W["diar.fc1.bias"]
W2, b2 = W["diar.single_spk_head.weight"], W["diar.single_spk_head.bias"]
def load(d, name):
    dd = pathlib.Path(d)
    data = (dd / f"{name}.f32").read_bytes()
    shape = tuple(json.loads((dd / f"{name}.json").read_text())["shape"])
    return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
def proj_err(h, l):
    lp = np.maximum(np.maximum(h, 0) @ W1.T + b1, 0) @ W2.T + b2
    return float(np.abs(lp - l).max())
def sig_err(l, p):
    return float(np.abs(1 / (1 + np.exp(-l.astype(np.float64))) - p.astype(np.float64)).max())
out = {"projection": {}, "sigmoid": {}, "parity": {}}
for key in ["v7_full", "v7_repeat", "v7_defaulthead", "v6", "v8s", "v20s", "v5p04s", "v4s", "v40s"]:
    h = load(V4 / key, "diar.hidden"); l = load(V4 / key, "diar.logits"); p = load(V4 / key, "diar.probs")
    e, s = proj_err(h, l), sig_err(l, p)
    out["projection"][key] = {"maxabs": e, "pass_1e-2": bool(e <= 1e-2)}
    out["sigmoid"][key] = s
    print(f"{key}: proj={e:.3e} pass={e <= 1e-2} sig={s:.2e} finite={bool(np.all(np.isfinite(h)) and np.all(np.isfinite(l)))}")
for key in ["c7", "c7_f32head", "c5p04s"]:
    h = load(C4 / key, "diar.hidden"); l = load(C4 / key, "diar.logits"); p = load(C4 / key, "diar.probs")
    e, s = proj_err(h, l), sig_err(l, p)
    out["projection"][key] = {"maxabs": e, "pass_1e-2": bool(e <= 1e-2)}
    out["sigmoid"][key] = s
    print(f"{key}: proj={e:.3e} pass={e <= 1e-2} sig={s:.2e}")
a = load(V4 / "v7_full", "diar.hidden"); b = load(V4 / "v7_repeat", "diar.hidden")
out["parity"]["repeat_hidden_exact"] = bool((a == b).all())
a = load(V4 / "v7_full", "diar.probs"); b = load(V4 / "v7_hidden", "diar.probs")
out["parity"]["split_export_probs_exact"] = bool((a == b).all())
out["parity"]["split_export_probs_maxabs"] = float(np.abs(a - b).max())
a = load(C4 / "c7", "diar.logits"); b = load(C4 / "c7_f32head", "diar.logits")
out["parity"]["cpu_f32head_identical"] = bool((a == b).all())
ah = load(C4 / "c7", "diar.hidden"); bh = load(C4 / "c7_f32head", "diar.hidden")
out["parity"]["cpu_f32head_hidden_identical"] = bool((ah == bh).all())
a = load(V4 / "v7_full", "diar.logits"); b = load(V4 / "v7_defaulthead", "diar.logits")
out["parity"]["vulkan_head_gap_maxabs"] = float(np.abs(a - b).max())
f = load(V4 / "v7_full", "diar.logits"); g = load(V4 / "v5p04s", "diar.logits")
n = g.shape[0]
d = np.abs(f[:n] - g).max(axis=1)
first = int(np.argmax(d > 0)) if bool((d > 0).any()) else n
out["parity"]["prefix_exact_first"] = first
out["parity"]["prefix_tail_maxdiff"] = float(d[first:].max()) if first < n else 0.0
t = json.load(open(V4 / "v40s" / "diar.trace.json"))
ev = t["chunks"] if isinstance(t, dict) and "chunks" in t else t
comp = [e.get("compress_after", e.get("compress_count", 0)) for e in ev] if isinstance(ev, list) else []
out["parity"]["v40s_chunks"] = len(ev) if isinstance(ev, list) else None
out["parity"]["v40s_compress_max"] = max(comp) if comp else None
dis = json.load(open(V4 / "v7_disabled" / "meta.json"))
dout = pathlib.Path(V4 / "v7_disabled" / "stdout.txt").read_text()
out["parity"]["disabled_rc0_no_dumps"] = bool(dis["rc"] == 0 and dout.startswith("audio:") and not list(pathlib.Path(V4 / "v7_disabled").glob("diar.*")))
print("parity:", json.dumps(out["parity"], indent=1))
(V4 / "verify.json").write_text(json.dumps(out, indent=1))
