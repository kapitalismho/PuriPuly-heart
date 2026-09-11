import json
import struct
import pathlib
import sys
import math
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
MODEL = pathlib.Path("C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf")
def rs(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode()
def skip_val(f, t):
    if t in (0, 1, 7):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    elif t == 8:
        rs(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                rs(f)
        elif et in (0, 1, 7):
            f.read(n)
        elif et in (2, 3):
            f.read(2 * n)
        elif et in (4, 5, 6):
            f.read(4 * n)
        elif et in (10, 11, 12):
            f.read(8 * n)
        else:
            raise ValueError(et)
    elif t in (10, 11, 12):
        f.read(8)
    else:
        raise ValueError(t)
def load_weights():
    f = open(MODEL, "rb")
    magic, ver = struct.unpack("<II", f.read(8))
    assert magic == 0x46554747
    nt, nkv = struct.unpack("<QQ", f.read(16))
    for _ in range(nkv):
        k = rs(f)
        t = struct.unpack("<I", f.read(4))[0]
        skip_val(f, t)
    infos = {}
    for _ in range(nt):
        name = rs(f)
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd)) if nd else ()
        dtype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos[name] = (dims, dtype, off)
    pos = f.tell()
    data_start = (pos + 31) & ~31
    out = {}
    for name in ["diar.fc1.weight", "diar.fc1.bias", "diar.single_spk_head.weight", "diar.single_spk_head.bias"]:
        dims, dtype, off = infos[name]
        f.seek(data_start + off)
        n = 1
        for d in dims:
            n *= d
        if dtype == 1:
            raw = np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32)
        elif dtype == 0:
            raw = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
        else:
            raise ValueError((name, dtype))
        if len(dims) == 2:
            raw = raw.reshape((dims[1], dims[0]))
        out[name] = raw
        print(f"{name} dims={dims} dtype={dtype} shape={raw.shape}")
    f.close()
    return out
def load_dump(d, name):
    data = (d / f"{name}.f32").read_bytes()
    meta = json.loads((d / f"{name}.json").read_text())
    shape = tuple(meta["shape"])
    n = 1
    for s in shape:
        n *= s
    vals = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
    return vals
def main():
    w = load_weights()
    W1, b1 = w["diar.fc1.weight"], w["diar.fc1.bias"]
    W2, b2 = w["diar.single_spk_head.weight"], w["diar.single_spk_head.bias"]
    res = {}
    for tag, d in [("v6", HERE / "results_low" / "v6_full"), ("v5p04s", HERE / "results_low" / "v5p04s"), ("c7", HERE / "results_cpu" / "c7_full")]:
        h = load_dump(d, "diar.hidden")
        l = load_dump(d, "diar.logits")
        assert h.shape[1] == 192 and l.shape[1] == 4 and h.shape[0] == l.shape[0]
        h1 = np.maximum(h, 0)
        h2 = h1 @ W1.T + b1
        h3 = np.maximum(h2, 0)
        lp = h3 @ W2.T + b2
        diff = np.abs(lp - l)
        denom = np.maximum(np.abs(l), 1e-6)
        rel = diff / denom
        res[tag] = {"n": int(h.shape[0]), "maxabs": float(diff.max()), "meanabs": float(diff.mean()), "maxrel": float(rel.max()), "pass_1e-2": bool(diff.max() <= 1e-2)}
        print(f"{tag} n={h.shape[0]} maxabs={diff.max():.3e} meanabs={diff.mean():.3e} maxrel={rel.max():.3e} pass={diff.max() <= 1e-2}")
    (HERE / "results_low" / "projection.json").write_text(json.dumps(res, indent=1))
if __name__ == "__main__":
    main()
