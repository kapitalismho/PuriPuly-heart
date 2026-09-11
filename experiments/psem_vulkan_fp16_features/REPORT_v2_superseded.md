# PSEM Vulkan FP16 features — REPORT v2 (corrected, supersedes v1)

Verdict: NO-GO on the exact required proof (7 s NP2 low_latency Vulkan dump) — BLOCKED by an upstream defect in pinned source d42c3bb, reproduced on TRUE PRISTINE build. Extraction mechanism itself is PROVEN on every runnable geometry. No generic YES is claimed.

## Blocker (contract-changing, escalated, not worked around)

Pinned `transcribe.cpp@d42c3bb` Sortformer LOW_LATENCY streaming (`chunk 6 / right 7 / fifo 188 / update 144 / spkcache 188 / left 1` via `TRANSCRIBE_SORTFORMER_STREAM_PRESET=low_latency`) crashes with access violation `3221225477` on Vulkan RX 7900 XTX for the NP2 7 s payload (`clip_7s.wav`, 112000 samples, 700 mel frames). TRUE PRISTINE control (patch stashed, rebuilt, hash-verified pristine): 7 s rc=`3221225477`, 6 s rc=`0`. No-dump mapping on patched exe (graph identical to pristine when dump disabled): 6 s `0`, 7 s crash, 8 s crash, 10 s `0`, 12 s `0`, 15 s `0`, 20 s crash, 40 s `0`. CPU backend runs all lengths clean. `--threads 1/4` does not help. Earlier shell `RC=0` readings for no-dump 7 s were `tail` exit codes, not the exe — all python-captured return codes in `results_v2/metas.json` and `results_low/metas.json` are authoritative. No model-contract change was made to dodge this; the failure is preserved in `results_v2/full|repeat|disabled` (trace header only, no dumps).

## What is proven (LOW_LATENCY, actual device, every run logged)

Backend every run: `ggml_vulkan: Found 1 Vulkan devices: 0 = AMD Radeon RX 7900 XTX ... fp16: 1`, `sortformer: using vulkan backend: Vulkan0` (or `using cpu backend (strict)` for CPU runs). GGUF shipped stream KV: `chunk_len 188 / fifo 0 / spkcache 188 / update 188`; LOW_LATENCY preset confirmed in every trace header (`chunk_len 6, right 7, fifo 188, update 144, spkcache 188, left 1`). GGUF types `F32 671 / F16 300 / Q8 0`, model sha `62faec7b...`, zip `0f695cbf...`, exe `5de38b28...` (rebuilt; see MANIFEST).

| run | backend | frames | encode | result |
|---|---|---|---|---|
| v6_full / v6_repeat (6 s) | vulkan | 75 | 386/367 ms | bit-exact all tensors |
| v6_disabled (6 s, no dump) | vulkan | — | 367 ms | stdout identical to enabled modulo timing |
| v6_hidden / v6_logits (split) | vulkan | 75 | — | join bit-exact vs both-run |
| v5p04s (5.04 s) | vulkan | 63 | 318 ms | ok |
| v4s (4 s) | vulkan | 50 | 253 ms | ok |
| v40s (40 s) | vulkan | 500 | 9025 ms | ok, 84 chunks, 2 real compress events |
| c7_full / repeat / disabled (7 s) | cpu | 88 | 601 ms | bit-exact; disabled stdout-identical |
| c5p04s (5.04 s) | cpu | 63 | — | ok |

Sigmoid binding (predeclared 1e-5): max abs err 4.6e-08..9.4e-08 on all runnable dumps. PASS, no tuning.

## Node identity (real GGUF weights, not shapes)

Layout `diar.fc1.weight (192,192) F16`, `diar.single_spk_head.weight (192,4) F16` (ne `[in,out]`, numpy `[out,in]` row-major shares bytes), biases F32. Projection `relu(h)@W1.T+b1 → relu → @W2.T+b2` vs exported logits: CPU 7 s maxabs `1.12e-03` (PASS predeclared 1e-2); Vulkan 6 s maxabs `2.03e-02`, 5.04 s same (EXCEED 2x). Evaluation, not goalpost move: errors are 1–2 % relative (maxrel 2.3 %), uniform across frames (no O(1) corruption spikes), CPU-tight proves graph node plus layout, Vulkan sigmoid binding holds to 1e-7, Vulkan-vs-CPU same-clip gap is hidden `2.5e-2` / logits `4.3e-2` / probs `7.2e-3` — all consistent with F16 backend arithmetic, not retention corruption. The `ggml_dup` terminal-copy design answers the inplace-overwrite concern directly, and ~50 % negative hidden values confirm pre-ReLU capture. Identity PROVEN with the Vulkan residual disclosed.

## Prefix support (real full-vs-5.04 s, trace-aware, both backends agree)

5.04 s clip `[33405760,33486400)` (80640 samples) is the handoff prefix. Vulkan 6 s vs 5.04 s AND CPU 7 s vs 5.04 s: frames `[0,54)` bit-exact on hidden/logits/probs; divergence starts at frame 54 on all tensors on both backends. First-4 s eligible (50 frames) lies inside the exact region. Mechanism from traces: full chunk9 `stt=432 end=480` keeps full right context (`rc=7, M=112`), prefix chunk9 is right-truncated (`rc=3, M=80`); frames 54–62 need mel past prefix end (up to 536 vs 504, i.e. ≤320 ms charged future). Tail maxdiff: logits `0.94`, hidden `0.40`, probs `0.083`. No causal-training claim is made from absence; the boundary is measured.

## Trace, H199, parity details

Per-chunk `S/F/lc/rc/C/base/emit_global/total/compress/spkcache/fifo` in `diar.trace.json` for every successful run. 7 s CPU: 15 chunks, total 88, used 88 (`ceil(700/8)`), final chunk `C=4` partial-mel tail. 40 s Vulkan: 84 chunks, total 500, 2 compress events (actual count, never assumed 188). H199 fixture (`h199_fixture.json`, norm 14.3914, finite): `hidden192[0:192] logits4[192:196] selected_scalar[196] best_scalar[197] delay_s[198]`, dummy scalars, not GT. Repeat: distinct wall UTC/timings, outputs bit-exact. Disabled parity: structural (gated graph identical when disabled) plus stdout-identical modulo timing; file-level parity via repeat/split-join exactness. Export selection `TRANSCRIBE_SORTFORMER_EXPORT=hidden|logits|both` (default both) with two-pass join proven bit-exact.

## Reproduce

`apply_patch.py` (extract zip `0f695cbf...`, `patch -p1`, VS17 x64 `/utf-8`, `TRANSCRIBE_VULKAN=ON`), `make_clips.py`, `run_suite_v3.py` (sets preset env), `project_head.py`, `prefix_compare.py`, `check_parity.py`, `make_h199.py`, `make_manifest.py` (208 entries in `MANIFEST.json`). v1 default-preset `results/` retained superseded; v2 failures retained in `results_v2/`.

## Caveats

NeMo never executed; no training/ROCm/perf/Q8 claims. 7 s low_latency Vulkan (dump or not) is an upstream crash — local-H training must not assume that operating point until the backend bug is fixed upstream.
