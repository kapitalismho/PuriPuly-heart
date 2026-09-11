# PSEM Phase A headroom observations — README (consumer corrections included)
# Owned surface: experiments/psem_phase_a_headroom/observations/ only.
# Frozen interface: OBSERVATIONS.json (schema phase_a.observations.v1) is IMMUTABLE
# and already delivered to the root consumer. Do not edit it, its traces,
# features, FREEZE.json, or the timing patch while root is consuming.
# If any semantic field below disagrees with OBSERVATIONS.json, that is a
# Director integration decision (pause, do not silently edit).

## What this folder holds
- FREEZE.json — exact prefix bounds frozen BEFORE any inference (source-0 prefixes).
- psem_phase_a_timing.patch — 4th patch (causal window mel + per-chunk service timing),
  applies with `patch -p1` on top of the pinned 3-patch tree. Clean reapply
  verified byte-identical.
- ami_*/ — per-source native dumps (float32 row-major) + diar.trace.json + meta/stdout/stderr.
- OBSERVATIONS.json — frozen interface (profile/sources/validation), 4.7 MB (8597 chunk rows).
- MANIFEST.json — hash list of owned files at seal time (self-hash excluded pattern).
- README.md (this file) + repro.py — reproducibility driver, added AFTER the barrier
  delivery; root rehashes at bind time.

## Freeze IDs (actual, from FREEZE.json — no guessed UTC)
- model (pinned FP16 GGUF): 62faec7b99ad23e323087597604b50728abe85089b6364970b019a845547bf99
- source zip (transcribe.cpp d42c3bb): 0f695cbfc2f0a28908dc287b3547341b869c0e9b13637b8473fcf5afe85a913a
- patches: model_cpp a55268b16aa602911d3247d420ecf05e45665a8d3a920f7dba6c12057fc3010b;
  sortformer_h fe30478b1abb5329f71cd36ad2841c6d78115329f0627672bdf1199bc77c9300;
  ggml_vulkan ca4f5847195e93164e6eac45cb3d44081e962379160c34d51caaeb761bed889d;
  timing 411226b6fc513166c689f898e0465690a917d1e25ec93cc856640633a7d68e39
- new exe (3 patches + timing, no model change): 3706db55ecf78c09188516cde829eccfcc04373a069d4f1834be0be22597540d
  (prior FP16 external build 42720fe07680eb157bbd461cb062a1f18a2396c1d5d6bbf64318ec94a21c76f7 untouched)
- full source WAV SHA256:
  ES2009c 350661f98c86a8a5973f1131e36444d93c0f44d07db5b12e174a49cb97f620a4 (31310848 fr)
  ES2009d b5ef423361ce67c804042ec479c6aeaf2cb8c19f73a779d61fabb7501e9e9b10 (33839104 fr)
  ES2002b 977fbf6cd473cfb1984b41755762eea4d7ffdad3fb15adcfdebcb8842163ec66 (36476075 fr)
  ES2009a 472adcae2cff535a251469bf5a9647ac9166de0f1bc049ce24504104cae9b1fc (22435328 fr)
  EN2009d eeb4a5ff47cadba8c75bdf32213b9264805027faa943c7f55eec0423c6ea4312 (85189974 fr)
- prefix rule: PREFIXEND=ceil1280(max(payload_end,episode_end,guard_end)+16640+1600), no cap needed.
  ami_ES2009c 19872000 (15525 native) — payload 19853040, NP1 19815744
  ami_ES2009d 33536000 (26200 native) — payload 33517760, NP2 33479616
  ami_ES2002b 2684160 (2097 native) — payload 2665520, NP3 2619136
  ami_ES2009a 9144320 (7144 native) — guard 9125280, BC1 ep 9124704
  ami_EN2009d 774400 (605 native) — T1 755520, R2 701312
- prefix WAV SHA256 ([0,prefix_end) slices, 16 kHz mono int16):
  ES2009c 44ee37081d6482d37adba737f79c570ce58552b1f129421e496bd135d388cf8a
  ES2009d 92ca6da91afc70043df5a4af23d5adf0f2d0498f49123146262fdaf735f92028
  ES2002b 18a04ad1d3d4cbc3640df81c4409140514136d85b6f32495ec23526c5600e163
  ES2009a 702af1850e9340e809b526c32de3fe55ebea50561aadefab225cfb39da680275
  EN2009d 44cdf330e7d33d7e7052ebd8235b0074e98366358b39cd776c77915a413f1b71

## Per-case timings (actual measured service; virtual-stream clock by sibling)
- ami_EN2009d: chunks 101, native 605, init_us 247871, service total 12.36 s,
  frontend total 0.170 s, wall 12.7 s
- ami_ES2002b: chunks 350, native 2097, init_us 386145, service total 55.28 s,
  frontend total 0.589 s, wall 55.7 s
- ami_ES2009a: chunks 1191, native 7144, init_us 816521, service total 195.67 s,
  frontend total 2.013 s, wall 196.6 s
- ami_ES2009c: chunks 2588, native 15525, init_us 1548286, service total 435.35 s,
  frontend total 4.338 s, wall 437.1 s
- ami_ES2009d: chunks 4367, native 26200, init_us 2639486, service total 727.38 s,
  frontend total 7.312 s, wall 730.3 s
- Totals: service 1426.0 s, frontend 14.42 s, wall 1432.4 s over ~68.8 min audio.
- Frontend scales with total mel frames (local windows M*160+352 samples each),
  not quadratically; service/audio ~0.34 steady.
- 7 s parity probe vs prior verified dump (results_v4/v7_full): hidden/logits/probs
  maxabs 0.0 bit-exact; causal-window parity maxabs 0.0 over all 8597 rows (threshold 1e-6).

## Consumer corrections (supersede the first receipt notes; OBSERVATIONS.json unchanged)
1. ALL chunks included: chunk 0 service (including one-time Vulkan pipeline compiles)
   is part of the observed cost. The earlier receipt note suggesting the caller use
   steady chunks[1:] and drop chunk 0 is WITHDRAWN. Consumer charges every emitted
   chunk in order and models virtual-realtime catchup after the slow first chunk;
   nothing is discarded.
2. EOS pad 96 is ARTIFICIAL: every prefix reports last raw_support_end = N+96
   (N multiple of 160; last mel frame reads 96 samples past N into constant zero pad).
   The full source files are LONGER than these prefixes, so the real source EOS was
   never reached and is NOT known here. The tail 96-sample pad and any terminal
   native frame(s) depending on it are UNSUPPORTED for application use: the caller
   MUST exclude or charge-as-unknown terminal frames past real availability.
   Selected case frontiers sit well before this padding, so no re-inference is needed.
3. Per-chunk service_us INCLUDES: causal window mel recompute (frontend_us) +
   Graph A alloc/compute/readback (graph_a_us) + host concat/pos-emb prep +
   Graph B alloc/compute/preds+hidden+logits readback (graph_b_us) + host
   streaming-update + slice insert (host_us) + parity-compare against the diagnostic
   full-prefix mel (instrumentation, conservative overcharge). It EXCLUDES only the
   final diar.trace.json fprintf/fflush for that chunk (trace instrumentation) and
   the one-time full-prefix mel diagnostic + model load + sched setup, which are
   reported separately as mel_full_diagnostic_us / load_us / sched_setup_us /
   initialization_us. No live-API claim: these are offline measured service costs.

## Toolchain (actual)
- VS2022 BuildTools MSVC 19.44.35207, cmake 4.3, glslc VulkanSDK 1.4.350.0,
  patch.exe from Git usr/bin, Python 3.12 (wave/hashlib/zipfile/subprocess/json).
- Frozen profile env for every run: TRANSCRIBE_SORTFORMER_STREAM_PRESET=low_latency,
  TRANSCRIBE_SORTFORMER_EXPORT=hidden,logits, TRANSCRIBE_DUMP_DIR=<per-source dir>,
  TRANSCRIBE_VK_NO_MUL_MAT_VEC=1, TRANSCRIBE_SORTFORMER_F32_HEAD=1,
  TRANSCRIBE_PSEM_CAUSAL_FRONTEND=1; CLI: transcribe-cli -m <model> --backend vulkan <prefix.wav>

## Reproduce without re-inference
- `python repro.py --smoke` — fast checks only: FREEZE bounds fit totals, timing-patch
  clean reapply byte-identical on temp copies, exe/model/zip hashes, per-source trace
  continuity + shapes + tail raw==N+96, OBSERVATIONS.json hash match. No GPU, no build.
- `python repro.py --prefixes` — recreate the five [0,prefix_end) WAVs in
  C:/tmp/psem-phase-a-prefixes/ and verify SHAs against meta.json.
- `python repro.py --patch-check` — same reapply check as smoke, verbose.
- `python repro.py --harvest --out C:/tmp/OBS.regen.json` — rebuild the
  OBSERVATIONS payload from the existing frozen per-source outputs and compare hash
  to the frozen file (no inference).
- Full rebuild + rerun path is documented in repro.py header but NOT required for
  verification (would repeat ~24 min GPU inference).
