# #218 Parakeet returning-A native repair

## Result

The smallest demonstrated repair is the matched **sherpa-onnx / sherpa-onnx-core 1.13.5** runtime cutover. Both exact previously failing returning-A packets now retain the sentence through the actual installed product's scoped session, frozen native options, PCM16 queue and recognition terminal. Speaker attribution metadata and submitted PCM were not changed.

This is **not full-content success**: the exact predeadline inputs still substitute **hand / hands** for **hair**, and the common hard-rollover successor remains empty. No synthetic tail, retry/fallback, minimum-duration deletion, extra context, model replacement or optional-cut restriction was added.

The complete structured evidence, commands and limits are in [`issue218_parakeet_report.json`](issue218_parakeet_report.json). The approved task-local sequence is [`issue218_parakeet_plan.md`](issue218_parakeet_plan.md), executing Outcome C of the Director's shared plan.

## Upstream and actual runtime

Primary upstream sources confirm:

- [Release v1.13.5](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.5), published **2026-08-11**, tag commit `3dc7c569f31ca2cd4a20ed6f7db780327e6714c5`.
- [PR #3857 / commit `0967a08db705d8eec9cf5cef962c8ec57c16e4a8`](https://github.com/k2-fsa/sherpa-onnx/commit/0967a08db705d8eec9cf5cef962c8ec57c16e4a8), merged **2026-08-11**, replaces cancellation-prone `E[x²] − E[x]²` variance with centered `(x − mean)²` variance in `NemoNormalizePerFeature`.
- [Upstream #3853](https://github.com/k2-fsa/sherpa-onnx/issues/3853) demonstrates a related INT8 empty/batch-contamination failure on **Parakeet Unified**, not this exact v3 artifact or recording. Do not conflate the two inputs.

The old installed and locked pair was actually **1.13.4**, not an already-fixed runtime. The new installed native module reports **1.13.5**, Git SHA `3dc7c569`, and ONNX Runtime **1.28.0**. Python remains **3.14.7 x64**, NumPy **2.5.1**. The model's encoder/decoder/joiner/tokens/installed-manifest hashes exactly match the earlier #218 report.

Configured feature dimension is **80**, but the selected encoder metadata and unchanged NeMo frontend override it to actual **128**, with `per_feature` normalization and TDT decoding. CPU provider, **two threads**, greedy search, no language hint and no hotwords are unchanged. Both exact-tag `offline-stream.cc` and NeMo recognizer frontend sources are identical between 1.13.4 and 1.13.5; their inspected call path reaches the corrected normalization function.

**Causality limit:** the exact-PCM comparison isolates changing the Sherpa runtime/packages, not a single-commit-only rebuild or directly captured old/new normalized feature tensors. Attributing this PCM's recovery specifically to the variance correction is **[INFERENCE]**, strongly supported by the verified upstream fix and relevant call path. The package-level repair is directly exercised; exact-function causality is not presented as measured.

The separate [NeMo trailing-silence report #15757](https://github.com/NVIDIA-NeMo/NeMo/issues/15757) is an open upstream observation using NeMo 2.7.3 / Torch CPU and an added 400 ms tail. It is not a proved mechanism for these exact Sherpa requests. No trim/retry or added tail was adopted from it.

## Exact input and native results

`issue215_smoke.input_cases`, reused by the original #218 driver, reconstructed the public A–B–A assembly. It matched assembled float32 SHA256:

`47484dedc6a7e69a33495b6fffb115ffdfcce59fedf55c59d41af5c79a97ac67`

All original per-payload hashes and concatenated PCM16 request hashes matched both archived CPU runs. Existing old outputs were reused as ground truth; no old negative decode was rerun.

| Exact supplied capture range at 16 kHz | PCM16 SHA256 | Reused 1.13.4 output | Actual 1.13.5 product output |
|---|---|---|---|
| Returning A, 1.04: `[75968,105472)`, no context | `563945ae319de1dbb7286714e9bf651d30ce1462ce1a6b75750021162b85a586` | `Yeah.` twice | `He washed the detritus from his hand.` |
| Returning A, 0.64: `[75808,105472)`, no context | `93978b19455fbc61574098c2d3898b988e4bf820730855d453b407063ece6d0a` | Empty twice | `He washed the detritus from his hands.` |
| Common hard-rollover successor, context `[100672,105472)` + new `[105472,120832)` | `f2aef23a6dfe8f58c63aba5ac0d567574a26479d39b923bae22172aab4eddf3b` | Empty | Empty |

Both initial-A and B requests still produce their exact reference sentences. The baseline's larger enclosing input still retains A–B–A with lexical errors; its lexical wording changes with the corrected runtime. These controls are retained in the raw reports, not silently described as unchanged full quality.

The 10 ms input-start difference remains **not proved as the sole cause** of the earlier divergent `Yeah.` / empty outputs. The repaired native outputs still differ lexically; no diarizer-profile ASR-quality conclusion is inferred.

### Bounded remaining-mechanism comparison

One offline diagnostic supplied only original returning-A capture `[75808,115225)`—no appended samples or added successor speech—and returned:

`He washed the detritus from his hair.`

Its PCM16 SHA256 is `6d7517d4a3f6f7e73139725cfcbc31ad95efd4a1f2417c2c147d30da1a962580`. This narrows the remaining limitation to native recognition of the actual fragmented/hard-safety windows rather than demonstrating that a complete natural returning utterance cannot be recognized. It is **not** a live causal cut policy: later capture was not available at the original predeadline submission. No product recut, future-context wait or re-decode fallback was implemented. The exact product split's `hand/hands` substitution and empty continuation remain failed content observations.

## Product cutover and compatibility surfaces

Changed product surfaces:

- `pyproject.toml`: exact matching wrapper/core **1.13.5** pins prevent ABI drift.
- `uv.lock`: targeted update reported **only those two packages** changed.
- `release_evidence/release_identity.py`: required packaged license path advances to `sherpa_onnx-1.13.5.dist-info/licenses/LICENSE`.
- `tests/app/test_release_dependency_guards.py`: removes incidental old-version/text assertions; adds matched native-pair, hashed production export, CPython 3.14 Windows x64 wheel readiness, and lock-derived packaged-license acceptance/empty rejection.

Provider decoding logic remains intentionally unchanged. No fake unit test attempts to simulate the upstream floating-point defect.

The supplied wrapper LICENSE remains **11,560 bytes**, SHA256 `3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5`: unchanged Apache-2.0 content. The core wheel still supplies no separate license member; the complete wrapper LICENSE is retained with both distributions' metadata.

The installed native `.pyd` SHA256 is `1d99c6d061c901a8714a59d6f3bb413c7954a47bb4b79f620ddc9e4d72f7cee7`. Windows `psutil` reported its isolated-target hardlink name even during normal `.venv` import. A separate explicit native-module inventory confirms `.venv` module origin, no isolated target on `sys.path`, and `os.path.samefile` identity of the two hardlinks. This is documented rather than silently treating a mapped path as the import path.

## Checks actually exercised

1. Isolated exact-PCM native/provider comparison: **7 unique original requests**, completed in **4.75 s**.
2. Installed product scoped queue/terminal smoke, with original request/payload metadata and frozen native options: **7 exact original requests**, completed in **4.39 s**, same texts as the isolated comparison.
3. One bounded real returning-A enclosure decode, as described above.
4. Native-module/package inventory: only wrapper/core changed in the existing environment; **google-genai remains 2.21.0**. No whole `uv sync`, SDK change, unrelated package update or model mutation occurred.
5. Explicitly authorized focused compatibility tests: initial complete three-file run **93 passed, 1 failed in 3.49 s**. Both new Sherpa guards and all local-provider/release-identity tests passed. The existing local-STT installer guard wrongly scanned the separate Nemotron speaker manifest. Its owned test now uses the canonical `LOCAL_STT_MANIFEST_RELATIVE_PATHS` inventory, preserving CPU asset checks and GPU-ASR non-bundling without changing the installer. The affected-test rerun **passed in 0.07 s**; the other 93 checks' unchanged evidence is reused, not described as a second full-suite run.

Raw observations: [`issue218_parakeet_runtime_1_13_5.json`](issue218_parakeet_runtime_1_13_5.json), [`issue218_parakeet_product_smoke.json`](issue218_parakeet_product_smoke.json), [`issue218_parakeet_runtime_inventory.json`](issue218_parakeet_runtime_inventory.json), [`issue218_parakeet_real_window_diagnostic.json`](issue218_parakeet_real_window_diagnostic.json).

The earlier decode reports' version-selector instrumentation recorded `{}` because native version attributes are strings, not functions. The supplemental runtime inventory states this explicitly. The final reproducible probe corrects only that metadata recording; no later decode is claimed for the metadata-only edit.

No build/lint/format run, new cloud call, ambient capture or VRChat/SteamVR launch was performed. Exact focused test commands executed:

```text
.venv/Scripts/python.exe -m pytest tests/app/test_release_dependency_guards.py tests/providers/test_local_cpu_backends.py tests/release_evidence/test_release_identity.py
.venv/Scripts/python.exe -m pytest tests/app/test_release_dependency_guards.py::test_installer_script_embeds_local_stt_manifest_assets_for_inno_download
```

The immutable packaged Windows payload has **not** been rebuilt or qualified. Source/runtime smoke is not installer acceptance.

## Director-maintained documentation recommendations

- Advance the THIRD_PARTY_NOTICES Sherpa wrapper/core versions from 1.13.4 to **1.13.5**, keeping the existing same-project/core-license explanation and unchanged complete Apache-2.0 license content.
- Preserve the architecture's **no synthetic Parakeet successor tail** invariant. Link this report for exact native runtime recovery while leaving lexical/hard-rollover limitations explicit. Do not rewrite historical #215/#218 measurements as fresh qualification.
- Keep both Nemotron profiles unselected until the separate assembled semantic/attribution evidence and maintainer decision. This dependency repair neither changes speaker policy nor approves feature activation.
- No optional-cut restriction is needed **based on this repaired catastrophic returning-A loss**, as directed. Residual lexical and hard-safety continuation failures are not hidden by color improvements or treated as passed complete-turn behavior.
