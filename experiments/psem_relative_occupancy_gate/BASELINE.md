# Baseline and reuse decisions

## System boundary

This experiment is offline research under `experiments/`. It does not alter the application runtime, Peer channel, VAD, STT, translation, output, or composition boundaries described in `ARCHITECTURE.md`.

The future product placement is parallel to VAD on the Peer mono-audio path. This issue does not integrate that placement.

## Authoritative inputs

- Dataset: `PSEM-STRATEGY-DATA-v2`.
- Dataset freeze file: `bc7e63bb201c2a33a9b2d69b2364fed8f03839278098f0bd175d6833b330a41e`.
- Dataset freeze payload: `f9f1882d0de08a4fcd19e63f1da7ae022f940420863be5bbfc14d1d2a7b0f95e`.
- Forced-alignment repository: `nttcslab-sp/diar-forced-alignment` at `9527b7c64846fb38316a610f32e9d3466bd6d8b7`.
- Source time: zero-based half-open source samples at 16 kHz.
- V2 EVAL remains sealed until Gate 0, model pins, mapping rules, grids, enrollment rules, and the evaluator are frozen.

The V2 normalization implementation is reused only to reconstruct and hash-check canonical activity intervals and ambiguity masks. The old `handoff_confirmed` events, candidate generators, thresholds, FE/h gates, and event reducers are rejected as scientific inputs.

## Frozen model surfaces

Streaming Sortformer uses the R8 Q8 model and patched telemetry-capable `transcribe-bench` CPU path. Only the native four-slot posterior dump and runtime telemetry are reused. R8 change-event decoding, threshold selection, duplicate suppression, and event metrics are rejected.

LS-EEND uses the domain-matched `L-AMI` stateful ONNX checkpoint. The streaming frontend, recurrent model state, raw four-slot logits, exact source-time availability functions, and model receipts are reused. The old `StreamingReducer` and boundary events are rejected.

Neither adapter exposes explicit slot-alive or eviction metadata. The trace records that limitation. Slot identity is treated as continuous only inside one uninterrupted model epoch; a reset invalidates the anchor. Ambiguous continuity disables speaker-induced cuts.

## Existing posterior disposition

The R8 posterior cache covers ten historically exposed sessions. The V2 split places most of those sources in TRAIN, so the cache is not a valid substitute for a fresh V2 DEV run. Any matching source may be used only after waveform, model, backend, frame, and trace-schema hashes pass.

## Lifecycle proxy

No pinned deterministic replay of the production PuriPuly VAD over V2 is present. Gate 0–2 therefore use authoritative GT speech/non-speech only as the explicitly allowed outer lifecycle proxy. The proxy preserves an anchor through ordinary silence and treats 1200 ms of unmasked silence as final `SpeechEnd`, reusing V2's frozen local-continuity maximum only for lifecycle control. It is not an occupancy label or model target. GT never defines `other_present` for Gate 1/2 and never selects or repairs a causal anchor. Product-VAD integration remains deferred.
