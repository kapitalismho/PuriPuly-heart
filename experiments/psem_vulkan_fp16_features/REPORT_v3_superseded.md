# PSEM Vulkan FP16 features — REPORT v3 (final)

Verdict: NO on the exact required proof (7 s NP2 low_latency Vulkan dump). Root cause is now PINPOINTED with a captured stack: the AMD Vulkan driver shader compiler crashes compiling a conv pipeline for the final-chunk geometry. Not fixable in pinned source. The extraction mechanism is PROVEN on every geometry the driver can execute. No generic YES is claimed.

## History of errors (integrity chain)

- v1 error (documented, superseded): ran the DEFAULT preset, not the required LOW_LATENCY operating point, and overclaimed a generic YES. `results/` retained; copy in `REPORT_v1_superseded.md`.
- v2 finding (preserved): LOW_LATENCY 7/8/20 s crash with `3221225477`, pristine-reproduced; `results_v2/` retains the failures; copy in `REPORT_v2_superseded.md`.
- v3 (this): repair probe per director order. All prior artifacts untouched. One self-inflicted incident during diagnosis (shifted multi-CUT deleted 2 functional source lines, breaking ALL backends incl. CPU) was caught by the CPU control run, restored byte-identical to pristine+patch (diff-verified), and behavior re-verified. No evidence was taken with the broken binary.

## Exact bug (stack evidence, `results_driver_bug/crash_stack_7s_stderr.txt`)

- Crash site: final streaming chunk only — 7 s chunk 14/15, 8 s chunk 16/17, 20 s chunk 41/42 (14/16/41 complete chunk lines in trace, then AV). Final-chunk geometry M=36 (7 s) / M=40 (8/20 s), T_diar=5, rc=0.
- Stage: inside `ggml_backend_sched_graph_compute` for Graph A (pre-encode conv stack), after alloc, before compute-ok (fflush-pinned locate probes, since removed).
- Fault: `0xC0000005` at `amdvlk64.dll+0x236b734` (`C:\Windows\System32\DriverStore\FileRepository\amdvlk.inf_amd64_*\amdvlk64.dll`), stack frames inside the driver's shader compiler (`boost::archive sc_xml_iarchive`, AMDSC IR), entered from exe (ggml pipeline creation). First-compile of a conv pipeline unique to M in {36,40}.
- Vulkan validation layer: silent. Host-side driver AV, no invalid API call observed.

## Repair attempts (all negative, no input special-casing, no production default change)

`TRANSCRIBE_NO_FLASH=1` still crashes. Honored, not noop: source `sortformer/model.cpp:439-441` + runtime 6 s flash-vs-noflash hidden maxabs `1.28e-02` (different kernels ran). Reviewer flash hypothesis DISPROVEN. Also still crash: `TRANSCRIBE_CONV_NO_DIRECT_DW=1`, `GGML_VK_DISABLE_COOPMAT=1`, `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`, `GGML_VK_DISABLE_ASYNC=1`, `GGML_VK_DISABLE_F16=1`. A uniform env-gated conv-to-CPU fallback in ggml requirements broke ALL geometries (CPU fallback path unusable in pinned source) and was fully reverted (ggml-vulkan.cpp, main.cpp byte-identical to zip; model.cpp byte-identical to pristine+patch; portable `patch -p1` re-apply verified byte-identical).

## What is needed (not generic)

AMD driver fix for the shader-compiler AV at amdvlk64.dll+0x236b734 on 7900 XTX, or an upstream ggml workaround routing those conv shapes around the crashing compile. After either, re-run `run_suite_v3.py` unchanged. Local-H training MUST NOT assume low_latency tails with T_diar=5 (mel tail M 33-40) on this driver.

## Proven (final exe `0729cb44...`, LOW_LATENCY, FP16 weights sha `62faec7b...`, zip `0f695cbf...`)

| run | frames | encode | rc |
|---|---|---|---|
| Vulkan 6 s full / repeat (distinct UTC, 0.65/0.61 s) | 75 | ~386 ms | 0, bit-exact all tensors |
| Vulkan 6 s export hidden-only vs both | 75 | — | 0, diar.probs arrays bit-identical (on/off parity on arrays) |
| Vulkan 5.04 s / 4 s / 10 s / 12 s / 15 s | 63/50/125/150/187 | — | 0 |
| Vulkan 40 s, 84 chunks, 2 real compress events | 500 | 9.68 s | 0 |
| CPU 7 s full / repeat / 5.04 s | 88/88/63 | — | 0, repeat bit-exact |

Sigmoid binding (predeclared 1e-5): worst `9.34e-08` over all dumps. PASS. Split hidden/logits two-pass join bit-exact. Backend device logged every run (`7900 XTX fp16:1`, `using vulkan backend: Vulkan0` / strict CPU).

## Node identity + numerics

GGUF `diar.fc1 (192,192) F16`, `single_spk_head (192,4) F16`, F32 biases. Projection relu(t)+head vs exported logits: CPU 7 s `1.12e-03` (PASS 1e-2), Vulkan 6 s `2.03e-02` (2x over, maxrel 2.3 %, uniform, ~50 % negative hidden = pre-ReLU tap). F64-vs-F32 head differential on the SAME hidden: identical errors (`2.027e-02`/`2.027e-02`, `1.122e-03`/`1.122e-03`) → gap is core F16 arithmetic in hidden, not head precision; no precision-profile change can close it. Vulkan-vs-CPU same-clip: hidden `2.5e-2`, logits `4.3e-2`, probs `7.2e-3`.

## Prefix support

5.04 s handoff prefix, Vulkan 6-vs-5.04 AND CPU 7-vs-5.04: frames [0,54) bit-exact all tensors, diverge frame 54 (full chunk9 keeps rc=7/M=112, prefix chunk9 truncated rc=3/M=80; frames 54-62 need mel to 536 vs 504). First-4 s inside exact region. Tail maxdiff logits `0.94`. No causal-training claim.

## H199 + reproduce + manifest

H199 (`results_cpu/h199_fixture.json`, norm 14.8808): CPU 7 s export, argmax-total-logit frame 17 (non-anchor), selected 0, best logit 2.9511, delay 1.36 s. Derived, not zeros, not GT. Reproduce: `apply_patch.py` (zip `0f695cbf...`, `patch -p1`, VS17 x64, `TRANSCRIBE_VULKAN=ON`), `make_clips.py`, `run_suite_v3.py` (+`project_head/check_parity/prefix_compare/head_precision/make_h199/make_manifest.py`). `MANIFEST.json` 214 entries, self-hash excluded. HEAD `83aaed9`, upstream `0/0`, preexisting `5M+4dirs` untouched, no commits.

## Caveats

NeMo never executed; no training/ROCm/perf/Q8 claims. 7 s-class low_latency tails unusable on this driver (see bug). Mixed FP16 legit (`F32 671/F16 300/Q8 0`).
