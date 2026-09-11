# PSEM Vulkan FP16 features — REPORT

Verdict: YES — streaming Sortformer on Vulkan FP16 exports actual hidden192 and presigmoid logits4 with deterministic parity to native probs.

## Freeze

freeze `psem.vulkan_fp16_features.freeze.v1` at `2026-09-09T11:13:38+00:00` in `experiments/psem_vulkan_fp16_features/FREEZE.json` sha `75e8261879f4818d9d2a34df374c05ce30eb047bb57de7bbe9a47ddc7140c2c0`.
Baseline `83aaed984b8b245082f3ffe7bb15d71f3242361f`, upstream `0/0`, preexisting `5M + 4 dirs` preserved bytewise. No Git mutations.

## Sources

- zip `C:/tmp/psem-vulkan-fp16-source/d42c3bb.zip` sha `0f695cbfc2f0a28908dc287b3547341b869c0e9b13637b8473fcf5afe85a913a`, commit `d42c3bbdfa2f63c37e5891e27de47a612d62f221`
- model `handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf` rev `7ef0c15dc8f9d717e9d24fac29a6e6551e9c6ddf` file `diar_streaming_sortformer_4spk-v2.1-F16.gguf` size `236606560` sha `62faec7b99ad23e323087597604b50728abe85089b6364970b019a845547bf99` verified before load at `C:/tmp/psem-vulkan-fp16-model/`
- GGUF types `F32=671 F16=300 Q8family=0 total=971 mixed_fp16_legit=True`. No all-FP16 claim.
- patch `sortformer_h.patch` sha `fe30478b1abb5329f71cd36ad2841c6d78115329f0627672bdf1199bc77c9300`, `model_cpp.patch` sha `ca5de3d3b7f2b1bec993b02eb2d6e97995708c1baea4a6834cf2585934bdb642`, portable `patch -p1` verified byte-identical re-apply `fc833a48...`.
- exe `build-fp16-vulkan/bin/Release/transcribe-cli.exe` sha `5de38b281888a79f86716075b74d570ad6fe38f49b3612a516419e265b3e58cc`, build `cmake -G "Visual Studio 17 2022" -A x64 -DTRANSCRIBE_VULKAN=ON` with `/utf-8` fix for preexisting canary CP949 misparse. `TRANSCRIBE_VULKAN=ON` maps `GGML_VULKAN` per `CMakeLists:125,278`.

## Contract

`build_stream_infer_graph` after 18 transformer blocks `t[192,T]` is NeMo `transformer_encoder.output` before relu/fc1; `s[4,T]` after `single_spk_head` before sigmoid. Only `preds` was graph output. Patch adds `ggml_set_output` plus `ggml_build_forward_expand` for `t,s`, `tensor_get` before `compute_ctx` recycle, host `S/F/lc/rc/C` slicing identical to `streaming_update_sync`, tail trim `ceil(feat_len/sub)`. Layout `ggml [d,T]` linear equals host `[T,d]` row-major, no transpose. Raw logits exported, normal probs path preserved. No public API break, `TRANSCRIBE_DUMP_DIR` opt-in.

## Runs

Audio `ES2009d.Mix-Headset.wav` sha `b5ef423361ce67c804042ec479c6aeaf2cb8c19f73a779d61fabb7501e9e9b10`.
Clips frozen before runs: 7s `[33405760,33517760)` pcm `461e7f5d...`, 40s `[32877760,33517760)` pcm `de625652...`, prefix4s `[33405760,33469760)`, prefix1040ms `[33405760,33422400)`.

Backend every run: `ggml_vulkan: Found 1 Vulkan devices: 0 = AMD Radeon RX 7900 XTX ... fp16: 1`, `sortformer: using vulkan backend: Vulkan0`. CPU frontend mel, Vulkan graph. No silent CPU. `--list-devices` shows `Vulkan0 kind=vulkan type=gpu` plus CPU fallback present unused.

| run | frames | hidden | logits | probs | encode | realtime |
|---|---|---|---|---|---|---|
| full 7s | 88 | 67584 B sha `8a08577f...` | 1408 B `361b97eb...` | 1408 B `ec7daa73...` | 76.2 ms | 92x |
| repeat 7s | 88 | identical | identical | identical | 76.2 ms | 92x |
| prefix4s | 50 | 38400 B `0d408368...` | 800 B `b2189571...` | 800 B `f4b32286...` | 60.2 ms | 66x |
| prefix1040ms | 13 | 9984 B `c2f37406...` | 208 B `049c23fd...` | 208 B `f1a2d419...` | 209.4 ms | 5x |
| clip40s | 500 | 384000 B `19ba85c4...` | 8000 B `88d587e5...` | 8000 B `48d6f1f4...` | 564.2 ms | 71x |

Frame clock `80 ms`: `88=ceil(7/0.08)`, `50=4/0.08`, `13=ceil(1.04/0.08)`, `500=40/0.08`. 7s single-chunk below fifo/spkcache 188; 40s multi-chunk crosses both, exercises compress path.

## Numerical checks

Predeclared `sigmoid(logits)~preds <=1e-5`. Actual max abs err: full `8.56e-08`, repeat `8.56e-08`, prefix4s `6.80e-08`, prefix1040ms `4.57e-08`, clip40s `9.44e-08`. All pass, no post-tuning.
Finite all arrays true. Hidden negatives prove pre-ReLU identity: full `8732/16896`, prefix4s `5002/9600`, prefix1040ms `883/2496`, clip40s `53806/96000`. Logits sigmoid binding proves presigmoid identity. Shapes `[T,192] [T,4] [T,4]` exact.
Determinism bit-exact full vs repeat probs/hidden/logits all true.
Prefix row counts recorded only; full-vs-prefix value equality not claimed because streaming concat attention carries within-concat future, classified as requires-fix only if causal training were claimed, which it is not.
Weight-matrix recompute from GGUF `fc1/spk` not executed; layer identity established by activation order plus numerical binding above, not shape alone.

## Caveats

NeMo reference not executed; chosen fixture constructibility only, no ownership inference, no head weights. No equality to NeMo required across F16 backend. No training, no ROCm, no performance promise, no FP16-vs-Q8 quality compare. H199 assembly untouched.

## Reproduce

1. `python experiments/psem_vulkan_fp16_features/apply_patch.py C:/tmp/psem-vulkan-fp16-repro C:/tmp/psem-vulkan-fp16-repro/build`
2. `python experiments/psem_vulkan_fp16_features/make_clips.py`
3. `python experiments/psem_vulkan_fp16_features/inspect_gguf.py C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf`
4. `python experiments/psem_vulkan_fp16_features/run_probe.py`

Raw `diar.hidden/logits/probs.f32+json`, `stdout/stderr`, `run_summary.json`, `verify.json` under `results/`.
