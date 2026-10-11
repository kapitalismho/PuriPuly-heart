from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def balanced_block(text: str, marker: str) -> str:
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 1
    for end in range(opening + 1, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start : end + 1]
    raise ValueError("unterminated pinned source block")


def run(args):
    repo = Path(__file__).resolve().parents[2]
    patches = json.loads((repo / "native/nemotron/patches.json").read_text(encoding="utf-8"))
    paths = {
        "vulkan": "llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        "selection": "src/runtime/ggml/backend.cpp",
    }
    code = {}
    hashes = {}
    for label, name in paths.items():
        raw = (args.source / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != patches["files"][name]["after_sha256"]:
            raise ValueError("guard source differs from recorded patch")
        code[label], hashes[name] = raw.decode("utf-8"), digest
    classification = balanced_block(code["vulkan"], "switch (physical_type)")
    selection_start = code["selection"].index("const auto type = ggml_backend_dev_type(dev);")
    selection_start = code["selection"].index("if (type !=", selection_start)
    selection_end = code["selection"].index(";", selection_start) + 1
    selection = code["selection"][selection_start:selection_end]
    harness = f"""#include <vulkan/vulkan.hpp>
#include "ggml-backend.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>
struct Context {{ enum ggml_backend_dev_type type; }};
static enum ggml_backend_dev_type classify(vk::PhysicalDeviceType physical_type) {{
    Context context;
    auto *ctx = &context;
    {classification}
    return ctx->type;
}}
static bool accepts(enum ggml_backend_dev_type type) {{
    try {{
        {selection}
        return true;
    }} catch (const std::runtime_error&) {{ return false; }}
}}
int main() {{
    assert(classify(vk::PhysicalDeviceType::eCpu) == GGML_BACKEND_DEVICE_TYPE_CPU);
    assert(classify(vk::PhysicalDeviceType::eOther) == GGML_BACKEND_DEVICE_TYPE_ACCEL);
    assert(!accepts(classify(vk::PhysicalDeviceType::eCpu)));
    assert(!accepts(classify(vk::PhysicalDeviceType::eOther)));
    assert(!accepts(classify(static_cast<vk::PhysicalDeviceType>(0x7fffffff))));
    assert(accepts(classify(vk::PhysicalDeviceType::eIntegratedGpu)));
    assert(accepts(classify(vk::PhysicalDeviceType::eDiscreteGpu)));
    assert(accepts(classify(vk::PhysicalDeviceType::eVirtualGpu)));
    std::puts("PHYSICAL_GPU_CLASSIFICATION_AND_SELECTION_PREDICATE_PASSED");
}}
"""
    with tempfile.TemporaryDirectory(prefix="nemotron-physical-guard-") as temporary:
        directory = Path(temporary)
        source = directory / "guard.cpp"
        executable = directory / "guard.exe"
        source.write_text(harness, encoding="utf-8")
        command = f'call "{args.vcvars}" >nul && cl /nologo /EHsc /std:c++17 /I"{args.vulkan_sdk / "Include"}" /I"{args.source / "llama.cpp/ggml/include"}" "{source}" /Fe:"{executable}"'
        compiled = subprocess.run(
            f'cmd.exe /d /s /c "{command}"', cwd=directory, capture_output=True
        )
        if compiled.returncode:
            raise RuntimeError(
                (compiled.stdout + compiled.stderr).decode("utf-8", errors="replace")
            )
        observed = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True
        ).stdout.strip()
    result = {
        "status": "passed",
        "scope": "compiled actual patched classification switch and strict selection predicate with real Vulkan/ggml enum headers; model-free deterministic CPU/Other/unknown and integrated/discrete/virtual GPU cases",
        "software_cpu_icd_inference": "not exercised; no software CPU ICD installed",
        "cases": 6,
        "observed": observed,
        "patched_source_sha256": hashes,
        "harness_sha256": hashlib.sha256(harness.encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vcvars", type=Path, required=True)
    parser.add_argument("--vulkan-sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
