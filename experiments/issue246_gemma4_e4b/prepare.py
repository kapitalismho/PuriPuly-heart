from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTERNAL = Path(os.environ.get("PP246_EXTERNAL", str(Path(os.environ["LOCALAPPDATA"]) / "puripuly-heart-issue246")))
MODEL = Path(os.environ.get("PP246_MODEL_DIR", str(Path(os.environ["LOCALAPPDATA"]) / "puripuly-heart/models/gemma-4-e4b-it-qat-ud-q4-k-xl")))
RELEASES = {
    "b10423": {
        "commit": "a94d563ed801d1da1b8c2432946de07d0231bb3d",
        "cpu": (18456396, "b5a396f113a344578c0766331704bd541fd743c4c8e92858bea18440ee0ab19a"),
        "vulkan": (34563676, "510447fb021c80a264b2181c885b5f2ce9cc5b66c65d447cd1f9ce7ba81dc222"),
    },
    "b11537": {
        "commit": "8e2d31e0eb658d0382d3106a61369dff81f0c080",
        "cpu": (19513247, "a05c2d35ebe9787489277d6e039b391fd006384a6c49d069395f3a7b47186fd2"),
        "vulkan": (33458882, "ae34d6a850c5d362cdfc0ef8f88f31fc499dab6df8853cec7887a32891aa5bbe"),
    },
    "b11630": {
        "commit": "6868a52cb337450658006ad2f9281c2eb63f1666",
        "cpu": (19563017, "82463bff7b42c41e8dc934202fb60ba28e1120c04264484b8d349b4a5958190d"),
        "vulkan": (32191318, "597684a3573bc0d0993a50cfa78433b8af7041db5ad008dc693ac6105b353e35"),
    },
}
EXPECTED = {
    "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf": (4215695776, "df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3"),
    "mtp-gemma-4-E4B-it.gguf": (59678016, "423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605"),
}
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 29: "IQ1_M", 30: "BF16"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name, data):
    path = HERE / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitize(json.dumps(data, indent=2, ensure_ascii=False)) + "\n", encoding="utf-8")


def sanitize(text):
    python = Path(sys.executable)
    for path, label in ((python, "<PYTHON>"), (EXTERNAL, "<EXTERNAL>"), (MODEL, "<MODEL>"), (ROOT, "<CHECKOUT>"), (Path.home(), "<HOME>")):
        for form in (str(path), str(path).replace("\\", "\\\\"), str(path).replace("\\", "/")):
            text = text.replace(form, label)
    redacted_python = str(python).replace(str(Path.home()), "<HOME>")
    for form in (redacted_python, redacted_python.replace("\\", "\\\\"), redacted_python.replace("\\", "/")):
        text = text.replace(form, "<PYTHON>")
    return text


def gguf(path):
    with path.open("rb") as handle:
        def unpack(fmt):
            return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))[0]
        def string():
            return handle.read(unpack("Q")).decode("utf-8")
        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element_kind, count = unpack("I"), unpack("Q")
                return [value(element_kind) for _ in range(count)]
            return unpack({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind])
        assert handle.read(4) == b"GGUF"
        version, tensor_count, metadata_count = unpack("I"), unpack("Q"), unpack("Q")
        metadata = {}
        for _ in range(metadata_count):
            key, kind = string(), unpack("I")
            metadata[key] = value(kind)
        tensors = []
        for _ in range(tensor_count):
            name = string()
            dimensions = [unpack("Q") for _ in range(unpack("I"))]
            kind, offset = unpack("I"), unpack("Q")
            tensors.append({"name": name, "dimensions": dimensions, "type_id": kind, "type": TYPE_NAMES.get(kind, str(kind)), "offset": offset})
        alignment = metadata.get("general.alignment", 32)
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment
        tensors.sort(key=lambda x: x["offset"])
        for i, tensor in enumerate(tensors):
            end = tensors[i + 1]["offset"] if i + 1 < len(tensors) else path.stat().st_size - data_offset
            tensor["aligned_bytes"] = end - tensor["offset"]
        chat_template = metadata.get("tokenizer.chat_template", "")
        result = {"version": version, "tensor_count": tensor_count, "histogram": dict(Counter(t["type"] for t in tensors)), "type_aligned_bytes": dict(Counter({kind: sum(t["aligned_bytes"] for t in tensors if t["type"] == kind) for kind in {t["type"] for t in tensors}})), "data_offset": data_offset, "tensors": tensors, "metadata": {key: val for key, val in metadata.items() if not key.startswith("tokenizer.ggml.")}, "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest()}
        return result


def main():
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    profile_path = Path(os.environ.get("PP246_THREAD_PROFILE", str(Path(os.environ["LOCALAPPDATA"]) / "puripuly-heart/llama_cpp_threads.json")))
    identity = {"checkout": "7a8151be6114a950f062e2430bd36f899d57e4de", "initial_git_status": "clean", "initial_branch": "experiment-optimize-gemma-4-e4b-on-cpu-vulkan-wi", "persisted_threads": json.loads(profile_path.read_text()), "runtimes": {}, "models": {}}
    for filename, (size, expected) in EXPECTED.items():
        path = MODEL / filename
        actual = digest(path)
        assert path.stat().st_size == size and actual == expected, filename
        identity["models"][filename] = {"size": size, "sha256": actual, "repo": "unsloth/gemma-4-E4B-it-qat-GGUF", "revision": "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265"}
        save(filename + ".json", gguf(path))
        print("verified", filename, flush=True)
    for release in ("b10423", "b11537"):
        release_info = RELEASES[release]
        for backend in ("cpu", "vulkan"):
            size, expected = release_info[backend]
            filename = f"llama-{release}-bin-win-{backend}-x64.zip"
            archive = EXTERNAL / filename
            url = f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{filename}"
            if not archive.exists():
                urllib.request.urlretrieve(url, archive)
            actual = digest(archive)
            assert archive.stat().st_size == size and actual == expected, filename
            runtime = EXTERNAL / release / backend
            runtime.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(runtime)
            binary = next(runtime.rglob("llama-server.exe"))
            for option, suffix in (("--help", "help"), ("--version", "version"), ("--list-devices", "devices")):
                result = subprocess.run([str(binary), option], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                (HERE / "results" / f"{release}-{backend}-{suffix}.txt").write_text(sanitize(result.stdout + result.stderr), encoding="utf-8")
            identity["runtimes"][f"{release}-{backend}"] = {"commit": release_info["commit"], "url": url, "archive_size": size, "archive_sha256": actual, "binary_sha256": digest(binary), "files": [{"name": str(p.relative_to(runtime)), "size": p.stat().st_size, "sha256": digest(p)} for p in sorted(runtime.rglob("*")) if p.is_file()]}
            print("verified", filename, flush=True)
    save("identity.json", identity)


if __name__ == "__main__":
    main()
