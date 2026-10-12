"""Offline approved input/artifact/credential inspection; zero model/server calls."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
from issue216_smoke import COMMON, ROOT, add_long_transfer, digest, input_cases, inspect_artifact, save, validate_runtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cases", default="clean_A_B,distinct_repeated_A_B_A,long_A_long_B")
parser.add_argument("--output", type=Path, default=ROOT / "native/nemotron/issue216_preparation.json")
args = parser.parse_args()
if not args.output.name.startswith("issue216"):
    raise ValueError("preparation evidence must remain issue216 task-local")
home = Path.home()
source_args = SimpleNamespace(corpus=home / ".cache/puripuly-heart/sim-v02-public-corpus-600-v1", correction_root=ROOT.parent / "listen-mega-integrate-nemotron-speaker-tracking/docs/experiments/issue-212-clarification")
inputs, cases = input_cases(source_args)
add_long_transfer(cases)
selected = args.cases.split(",")
for name in selected:
    if name not in cases:
        raise ValueError(f"unknown approved input case: {name}")
model = home / "AppData/Local/Temp/puripuly-nemotron-bonenk/model.gguf"
runtime = ROOT / "build/nemotron"
report = {
    "schema": "issue216-offline-preparation-2",
    "python": sys.version,
    "model_calls": 0,
    "network_calls": 0,
    "private_uploads": False,
    "inputs": inputs,
    "selected_cases": {name: {"samples": int(cases[name][0].size), "seconds": cases[name][0].size / 16000, "pcm_float32_sha256": hashlib.sha256(cases[name][0].tobytes()).hexdigest(), "assembly": cases[name][1]} for name in selected},
    "planned_single_playback_input_seconds": sum(cases[name][0].size for name in selected) / 16000,
    "input_note": "Capture durations exclude request replay/padding; actual live budget accounts every attempted PCM write. Assembly annotations never select a model/source boundary.",
    "credentials_present": {name: bool(COMMON.secret(name, environment)) for name, environment in (("deepgram_api_key", "DEEPGRAM_API_KEY"), ("soniox_api_key", "SONIOX_API_KEY"), ("elevenlabs_scribe_api_key", "ELEVENLABS_API_KEY"), ("alibaba_api_key_beijing", "ALIBABA_API_KEY_BEIJING"))},
    "artifact": inspect_artifact(model),
    "runtime": validate_runtime(runtime),
    "runtime_manifest_sha256": digest(runtime / "runtime-manifest.json"),
    "imported_helper_sha256": {relative: digest(ROOT / relative) for relative in ("native/nemotron/issue215_smoke.py", "native/nemotron/probe.py")},
    "local_model_runtime": "Existence/hash/manifest inspection only; no inference performed by preparation",
    "selected_route": "Deepgram nova-3 actual product adapter; endpointingfalse/interimfalse and production application natural/six-second controls",
    "trimming": "OFF",
    "soniox": "Reuse issue212 comment6114014900 OFF/ON observations; post-manual lexical clock unsupported and productionautomaticOFF",
    "presentation": "No actual GUI/native-caption/HMD observation; source payload metadata is not visible caption qualification",
    "required_live_barrier": "Every fresh probe requires explicit Director scenario/candidate-ready release; preparation is not automatic authorization",
    "reproduction_examples": [".venv/Scripts/python.exe -B native/nemotron/issue216_prepare.py --cases long_A_long_B --output native/nemotron/issue216_preparation_reproduced.json", ".venv/Scripts/python.exe -B native/nemotron/issue216_smoke.py --mode new --cases long_A_long_B --output native/nemotron/issue216_transfer_reproduced.json"],
}
save(args.output, report)
print({"status": "prepared_offline", "cases": selected, "planned_single_playback_input_seconds": report["planned_single_playback_input_seconds"], "network_calls": 0, "model_calls": 0})
