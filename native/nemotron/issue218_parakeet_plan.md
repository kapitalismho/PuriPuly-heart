# #218 Parakeet exact-input repair plan

This task-local record executes Outcome C of `docs/issue218-speaker-attribution-plan.md`. The Director owns shared speaker policy and maintained documentation. Provider-local code, relevant provider behavior tests, `pyproject.toml`, `uv.lock`, and `native/nemotron/issue218_parakeet_*` records/probes are this task's surfaces.

## Frozen evidence and constraints

- Baseline candidate: `b2cfc0c42bfa07d1570c243fb17ab2e547b8da1e`.
- Actual installed and locked packages inspected: `sherpa-onnx==1.13.4`, `sherpa-onnx-core==1.13.4`; Python 3.14.7 x64; NumPy 2.5.1; Python ONNX Runtime 1.28.0. Native Sherpa ONNX Runtime may be separately bundled and will be recorded.
- Parakeet v3 INT8, CPU, two threads, configured 80-dimensional features overridden by native model metadata to 128, greedy search, `nemo_transducer`, no language hint/hotwords. Model artifacts and native options do not change.
- Reuse archived authoritative old native results, repeated twice: returning A `[75968,105472)` -> `Yeah.` at profile 1.04; `[75808,105472)` -> empty at 0.64. No source context in either request. The 10 ms difference is not established as causal.
- Existing input assembly: `issue215_smoke.input_cases`, called by `issue218_experiment.inputs`. Public audio hashes, exact crop coordinates, assembled float32 digest and each submitted PCM16 digest must match archival provenance before decode.
- No synthetic tail, large arbitrary context, minimum-duration deletion, silent fallback, model replacement, invasive tuning grid, cloud/VR applications, or Git/GitHub mutation.

## Execution

1. Inspect primary upstream release/tag/commit, not attachment assertions. Compare normalization implementation in the actual old/new tags and identify which recognizer path calls it.
2. Obtain only the necessary public runtime packages in an isolated task-local target; preserve the installed environment.
3. Reconstruct exact recorded PCM via the existing driver; validate assembled and payload hashes, including PCM16 conversion and context order. Record old outputs as reused evidence, not new old-version runs.
4. Decode the two exact failing payloads through the real product Parakeet provider with isolated 1.13.5 and unchanged model/options. Record runtime/library version/hash, model hashes, raw native text/tokens/timing and product text.
5. If the correction repairs the actual failing input, make the smallest compatible package/distribution change. Do not manufacture a consumer regression test for a native implementation bug; use real local decode as smoke.
6. If the exact input still fails, run bounded diagnostics using original captured PCM only to distinguish normalization, request enclosure and real hard-rollover/trailing-edge effects. Do not turn diagnostic windows into unapproved product padding/context policy.
7. Report any necessary optional-cut containment decision to the Director, who owns `core/stt/batch_speaker.py`. Report maintained-doc recommendations rather than editing them.

## Evidence and handoff

Durable task-local JSON/Markdown reports identify exact commands, reused versus new observations, artifact/runtime hashes, raw outputs, limitations, smallest repair and remaining mechanism. Main agent owns combined validation. Provider-focused native smoke and, by explicit follow-up authority, the release-dependency guards/local CPU provider/release-identity tests were exercised with results recorded in `issue218_parakeet_report.*`. No repository-wide build/test/format runs are part of this worker task.
