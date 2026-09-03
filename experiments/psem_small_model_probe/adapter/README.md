# Gate 1 — PSEMObservationAdapter + common decoder + smoke (issue #117)

CPU-only scaffolding. No model downloads, no weights, no new dependencies.

## Interface (`protocol.py`)

`PSEMObservationAdapter` (Protocol): `frame_ms: int`, `sample_rate_hz = 16000`,
`reset() -> None`, `bind(reference_pcm16: bytes) -> None`, `step(pcm16_chunk: bytes) -> StepOut`.

`StepOut` (frozen, slots): `speech: float | None` (raw p_speech, no smoothing),
`anchor: float` (raw p_anchor_active), `aux: dict` (model-specific extras),
`source_time_ms: int` (source-clock end time of the chunk).

PCM contract (fail-closed, never pad): `load_wav_mono16k` rejects non-16 kHz /
non-mono / non-int16 WAV with `ValueError`; `validate_pcm16_chunk` rejects empty
or non-frame-multiple chunks and non-16 kHz rates; `bind` raises `BindingError`
on empty / too-short spans and `RuntimeError` if called before `reset()`;
`step` raises `RuntimeError` before `bind()`.

## Decoder (`decoder.py`)

`CommonPersistenceDecoder(frame_ms, confirmation_ms=500, sensitivity_ms=300)`,
model-blind, thresholds passed per-call as `tau` (smoke placeholder `0.5`;
CAL12 sets real values later). `reset()` + `update(frame, *, tau)` where frame is
`{speech_gt, anchor, lifecycle, source_time_ms}` returning
`{action, source_boundary_time, decision_time, sensitivity}`.

Semantics: live speech + `anchor < tau` (anchor inactive while someone speaks =
takeover) accumulates transfer-evidence `run_ms`; anchor recovery
(`anchor >= tau`) resets the run + `KEEP` (A->A continuity never cuts).
No-speech (`None`/`False`) → reset + `KEEP`; lifecycle
`UNBOUND`/`UNCERTAIN`/`POISONED` → reset + `HOLD`; `run_ms >= 500` → `CUT` with
`source_boundary_time = run_start` (first subthreshold frame start),
`decision_time = t`, state cleared. `300 <= run_ms < 500` → diagnostic
`CUT_SENS` with `sensitivity=True` (run state retained so the primary `CUT`
still fires at 500 ms; never confused with it). Unknown lifecycle raises
`ValueError`. Invariant asserted on every `CUT`: `source_boundary <= decision`.

## Logging schema

Per-step JSONL: `{episode_id, model, regime, frame_ms, source_time_ms,
wall_time_ms, speech, anchor, aux, lifecycle, action}` — raw outputs only,
smoothing forbidden. Per-episode header: `{model, model_sha, onnx_sha,
reset_ok, bind_span_hash}` (`ecapa_sha|none` joins the header when the FireRed
adapter lands). Smoke demonstrates both in `run_episode` and hashes them
(wall-clock excluded) for the reset-isolation check.

## Smoke (`smoke.py`, 6/6 PASS 2026-09-03)

Runnable `python experiments/psem_small_model_probe/adapter/smoke.py` or
`python -m experiments.psem_small_model_probe.adapter.smoke`, CPU only:

```text
[PASS] reset-isolation — hash 68a6b048be97 vs 68a6b048be97
[PASS] source-time — times=[20, 40, 60]..160, cuts=1, invariant=True (low-anchor transfer)
[PASS] 16k-mono-contract — 8k/stereo/8bit + bad chunks + bind-before-reset + empty bind
[PASS] no-smoothing — anchors=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
[PASS] decoder-edges — gates=['KEEP','KEEP','HOLD','HOLD','HOLD','HOLD','KEEP'], sens_frames=1, tau=0.5
[PASS] cpu-scaffold — stub step median=0.0009ms p95=0.0009ms p99=0.0009ms RTF=0.0000
  bind=0.016ms reset=0.000ms (real-model timing deferred to Gate 3/4)
```

## Deferred integration points (no code yet)

- `firered.py` (Gate 3): `PSEMObservationAdapter` over vendored ECAPA embedding
  (`spkrec-ecapa-voxceleb`) + `pvad.onnx` raw prob; LiveKit smoothing explicitly
  disabled; `anchor := pvad` prob; `aux = {ecapa_dim, mel_state}`.
- `neovad.py` (Gate 4): `PSEMObservationAdapter` over streaming `gru.pt` step
  path, no external enrollment, `bind` records span hash only;
  `anchor := p(primary)`; `aux = {p_nonspeech, p_primary, p_secondary}`.
- Real `tau_primary` / `tau_sens` sweeps on CAL12; CPU budget enforcement
  (RTF ≤ 0.25, median/p95/p99 + peak RSS) against real `step()` timing.

## Vendored weights + Protocol adapters (Gate 1-extension, issue #117)

Pins in `adapter/vendor.json`; small blobs vendored under `adapter/vendor/`,
ECAPA receipt-only (`spkrec-ecapa-voxceleb.receipt.json`) with local-cache path
`$FIRERED_ECAPA_DIR` or `adapter/vendor/spkrec-ecapa-voxceleb/`. No new root
`pyproject` deps: `onnxruntime`/`torch`/`speechbrain`/`numpy` are lazy imports;
adapters import cleanly without them and raise informative errors only on
`bind()`/`step()`. No training, no thresholds, decoder untouched.

| artifact | rev | sha256 (content) | size | local |
|---|---|---|---|---|
| `pvad.onnx` | `74561b17` | `2114fd3c…0552` | 3940567 | `vendor/pvad.onnx` |
| ECAPA `embedding_model.ckpt` (receipt) | `74561b17` | blob `ed9dd7ee` | 83316686 | `$FIRERED_ECAPA_DIR` |
| `neovad_gru.pt` | `3d82cbb5` | `5b78dfc8…374a` | 3544772 | `vendor/neovad_gru.pt` |
| `neovad_gru.yaml` | `3d82cbb5` | `623ca09c…7fe7` | 228 | `vendor/neovad_gru.yaml` |

`firered.py` — `FireRedAdapter(frame_ms=10|20|…, min_bind_ms=1000)`:
ECAPA enrollment in `bind()` over caller-selected regime bytes (5 s native XOR
1 s causal); `reset()` zeroes spkemb + mel/GRU buffers; `step()` runs one
160-sample onnx frame per 10 ms (multi-frame chunks looped), LiveKit
`ExpFilter`/duration smoothing bypassed (`livekit_smoothing = False`),
`anchor :=` raw pvad prob, `speech := None`,
`aux = {ecapa_dim: 192, mel_state_hash}`.

`neovad.py` — `NeoVADAdapter(frame_ms=10, min_bind_ms=1000)`: `bind()` records
the span hash only (no embedding); `reset()` drops the GRU hidden state, which
is short-term acoustic memory, never a locked speaker identity; `step()` maps
each frame through the streaming GRU `step` path (no hysteresis gate),
`anchor := p(primary)`, `speech := 1 − p(nonspeech)`,
`aux = {p_nonspeech, p_primary, p_secondary}`.

Both fail closed on non-16 kHz/non-frame PCM, require `reset()` before `bind()`
(`RuntimeError`), raise `BindingError` on empty/short spans, keep a monotonic
`source_time_ms`, log raw per-frame rows (`adapter.frames`, no smoothing), and
expose `episode_header()` =
`{model, model_sha, onnx_sha, ecapa_sha|none, reset_ok, bind_span_hash}`.

Verify (CPU): `python experiments/psem_small_model_probe/adapter/verify_vendor.py`
(16/16 PASS 2026-09-03: vendor hashes + contracts + weight-missing error paths;
live inference blocked until `torch`/`onnxruntime`/`speechbrain` + ECAPA cache
are present; synthetic PCM only — manifest rows carry no local audio paths).
