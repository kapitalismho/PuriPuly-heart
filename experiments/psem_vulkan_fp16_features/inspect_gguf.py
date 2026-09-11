import struct
import collections
import pathlib
import hashlib
import sys
GGUF_MAGIC = 0x46554747
TYPE_NAMES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"I8",17:"I16",18:"I32",19:"I64",20:"F64",21:"BOOL",24:"Q4_0_4_4",25:"Q4_0_4_8",26:"Q4_0_8_8",27:"Q6_0"}
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()
def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")
def main(path):
    p = pathlib.Path(path)
    size = p.stat().st_size
    print(f"file={p} size={size}")
    print(f"sha256={sha256_file(p)}")
    f = open(p, "rb")
    magic, version = struct.unpack("<II", f.read(8))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    assert magic == GGUF_MAGIC, hex(magic)
    print(f"gguf version={version} n_tensors={n_tensors} n_kv={n_kv}")
    for _ in range(n_kv):
        k = read_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        if t in (0, 1, 7):
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t == 8:
            read_str(f)
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if et == 8:
                for _ in range(n):
                    read_str(f)
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
    counts = collections.Counter()
    names_by_type = collections.defaultdict(list)
    for _ in range(n_tensors):
        name = read_str(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)) if n_dims else ()
        dtype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tname = TYPE_NAMES.get(dtype, f"UNK{dtype}")
        counts[tname] += 1
        if len(names_by_type[tname]) < 5:
            names_by_type[tname].append((name, dims))
    print("type_counts:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]} e.g. {names_by_type[k][:2]}")
    n_q8 = sum(v for k, v in counts.items() if "Q8" in k)
    n_f16 = counts.get("F16", 0)
    n_f32 = counts.get("F32", 0)
    print(f"summary F32={n_f32} F16={n_f16} Q8family={n_q8} total={sum(counts.values())}")
    print(f"mixed_fp16_legit={n_f16 > 0 and n_f32 > 0 and n_q8 == 0}")
if __name__ == "__main__":
    main(sys.argv[1])
