import json
import pathlib
import numpy as np
HERE = pathlib.Path("experiments/psem_vulkan_fp16_features")
def load_dump(d, name):
    data = (d / f"{name}.f32").read_bytes()
    shape = tuple(json.loads((d / f"{name}.json").read_text())["shape"])
    return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
def main():
    d = HERE / "results_v4" / "v7_full"
    h = load_dump(d, "diar.hidden")
    l = load_dump(d, "diar.logits")
    t = int(np.argmax(l.sum(axis=1)))
    assert t != 0
    sel = int(np.argmax(l[t]))
    best = float(l[t, sel])
    frontier_s = round(t * 0.08, 2)
    h199 = np.concatenate([h[t], l[t], np.array([float(sel)], np.float32), np.array([best], np.float32), np.array([1.04], np.float32)])
    assert h199.shape == (199,) and bool(np.all(np.isfinite(h199)))
    out = {"frame": t, "layout": "hidden192[0:192] logits4[192:196] selected_idx[196] best_logit[197] delay_s[198, legacy constant 1.04]", "selected": sel, "best": best, "delay_s": 1.04, "actual_source_frontier_s": frontier_s, "frontier_note": "fixture delay keeps legacy 1.04 constant; actual argmax frame frontier is reported separately, no silent contract change", "label_note": "derived from Vulkan VEC+F32HEAD low_latency 7s export, argmax-total-logit non-anchor frame; actual anchor decision, not GT", "finite": True, "norm": float(np.linalg.norm(h199))}
    (HERE / "results_v4" / "h199_fixture.json").write_text(json.dumps(out, indent=1))
    print(f"h199 frame={t} sel={sel} best={best:.4f} delay=1.04 frontier={frontier_s:.2f}s norm={out['norm']:.4f}")
if __name__ == "__main__":
    main()
