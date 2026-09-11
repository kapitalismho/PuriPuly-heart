import json
import pathlib
import hashlib
import os
HERE = pathlib.Path("experiments/psem_vulkan_fp16_features")
EXTRA = {"model": pathlib.Path("C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf"), "exe": pathlib.Path("C:/tmp/psem-vulkan-fp16-source/build-fp16-vulkan/bin/Release/transcribe-cli.exe"), "source_zip": pathlib.Path("C:/tmp/psem-vulkan-fp16-source/d42c3bb.zip")}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()
def main():
    out = {}
    for p in sorted(HERE.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST.json":
            out[str(p.relative_to(HERE).as_posix())] = {"sha256": sha(p), "bytes": p.stat().st_size}
    for k, p in EXTRA.items():
        out[k + ":" + p.as_posix()] = {"sha256": sha(p), "bytes": p.stat().st_size}
    (HERE / "MANIFEST.json").write_text(json.dumps(out, indent=1))
    print(f"manifest entries={len(out)}")
if __name__ == "__main__":
    main()
