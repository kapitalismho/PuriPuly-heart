# PSEM Small-Model Probe — Gate 0 Manifest + Gate 1 Adapter Design (Issue #117)

## 0. Layer separation (normative vocabulary)

| Layer | Name | Values | Trained? |
|---|---|---|---|
| Acoustic observation | `p_speech`, `p_anchor_active` | float [0,1], raw, per native frame | Yes (FireRed pVAD / NeoVAD GRU) |
| Anchor lifecycle | `lifecycle` | UNBOUND / BOUND / UNCERTAIN / POISONED | No — derived from binding regime + causal span |
| Temporal ownership | evidence accumulator | ms of live-speech transfer evidence | No — fixed 500 ms persistence decoder |
| Product action | `action` | KEEP / HOLD / CUT | No — decoder output |
| Boundary commitment | `source_boundary_time` vs `decision_time` | both ms, distinct fields | — |

`source_boundary_time` = backdated evidence-run start (source clock). `decision_time` = when confirmation completed (source clock) + `compute_lag_ms` (wall). Never conflate.

## 1. Gate 0 manifest schema: PSEM-SMALL-MODEL-PROBE-v1

One JSONL row per episode. Times in **ms** (int); durations in **s** (float) only in derived metrics.

```jsonc
{
  "schema_version": "psem.small_probe.manifest.v1",
  "episode_id": "str, unique, {session_id}:A{NNNNN}",
  "corpus": "ami | alimeeting",
  "session_id": "str, native session key",
  "topology": "A | A+B | A+A+B | A->A+B->A | A->A+B->B | overlap_return",
  "split": "CAL12 | MAIN48 | EXT24",
  "evaluation_start_ms": "int >= 0, scoring window start (source clock)",
  "evaluation_end_ms": "int > start",
  "anchor_speaker": "str, enrolled identity A",
  "native_reference_start_ms": "int, 5 s clean non-overlap A span start",
  "native_reference_end_ms": "int = start + 5000",
  "causal_reference_start_ms": "int | null, earliest 1 s A span ending <= transition",
  "causal_reference_end_ms": "int | null = start + 1000",
  "causal_bindable": "bool",
  "authoritative_transition_time_ms": "int, GT replacement boundary (source clock)",
  "ontology_subset": "bool, member of ONTOLOGY16",
  "control_subset": "bool, member of CONTROL24"
}
```

Counts: CAL12 = 6 topology groups x 2; MAIN48 = 6 strata C1-C6 x 8 (AMI 4 + AliMeeting 4 per stratum); EXT24 reserve = 6 x 4; ONTOLOGY16/CONTROL24 are boolean flags on MAIN48 rows (no duplicate rows).

### Span derivation

- Native (regime O): scan GT activity for longest 5 s non-overlap A-only span in `(-inf, transition)`; record exact `[start, start+5000)`. Must contain zero other-speaker samples, speech-activity > 95%.
- Causal (regime C): scan backward from transition for earliest 1 s A-only non-overlap span with `end <= transition`; record `[start, start+1000)`. If none exists -> both fields `null`, `causal_bindable=false`.
- `lifecycle` at episode start: `causal_bindable ? BOUND : UNBOUND`. `UNCERTAIN` reserved for adapter-reported low-enrollment confidence; `POISONED` for overlap-contaminated enrollment (diagnostic only, forces HOLD).

### Disjointness enforcement (build-time asserts, fail-closed)

1. `session_id` sets: CAL12 ∩ MAIN48 = ∅; EXT24 ∩ (CAL12 ∪ MAIN48) = ∅. Compare on `(corpus, session_id)`.
2. `episode_id` unique across file; `evaluation_*` non-overlapping within a session.
3. ONTOLOGY16/CONTROL24 ⊆ MAIN48 (`split == MAIN48` required when flag true); counts exactly 16/24.
4. `native_reference_*` within session audio; `causal_*` ⇒ `end <= authoritative_transition_time_ms`.

### Hash procedure (mirrors training-strategy-gate preflight)

```text
canonical_json(row) = json.dumps(row, sort_keys=True, separators=(',',':'), ensure_ascii=True)
file_sha   = sha256(raw bytes of manifest.jsonl)
freeze_sha = sha256(canonical_json({"rows": [...sorted by episode_id...], "counts": {...}}))
```

Record both in `dataset_freeze.json`; Gate 1 refuses to run on mismatch.

## 2. PSEMObservationAdapter interface

```python
class PSEMObservationAdapter(Protocol):
    frame_ms: int            # native grid, e.g. 10 or 20; fixed per model, logged
    sample_rate_hz: int = 16000
    def reset(self) -> None: ...
    def bind(self, reference_pcm16: bytes) -> None: ...  # mono 16 kHz int16 LE
    def step(self, pcm16_chunk: bytes) -> StepOut: ...

@dataclass(frozen=True, slots=True)
class StepOut:
    speech: float            # p_speech, raw, no smoothing
    anchor: float            # p_anchor_active; FireRed=pvad prob, NeoVAD=p(primary)
    aux: dict                # NeoVAD: {p_nonspeech, p_primary, p_secondary}; FireRed: {ecapa_dim, mel_state}
    source_time_ms: int      # source-clock end time of this chunk
```

- PCM: 16 kHz mono int16 LE; chunk length = exactly one native frame-grid multiple; reject otherwise (`ValueError`).
- `reset()`: clears ALL state (VAD history, ECAPA cache, GRU hidden, mel buffer). Called once per episode before `bind()`; `bind()` after `reset()` only, else `RuntimeError`.
- `bind()`: FireRed computes ECAPA embedding from the regime span (5 s native XOR 1 s causal — caller selects bytes); NeoVAD records span hash, no embedding. Empty/too-short span -> `BindingError`, episode scored as UNBOUND/HOLD.
- Error cases: wrong SR/channels/dtype, non-multiple chunk, `step` before `bind`, sample-count discontinuity -> raise, never silently pad.
- Logging (per step, JSONL): `{episode_id, model, regime, frame_ms, source_time_ms, wall_time_ms, speech, anchor, aux, lifecycle, action}`; plus per-episode header `{model, model_sha, onnx_sha, ecapa_sha|none, reset_ok, bind_span_hash}`. Raw outputs only — smoothing is forbidden here.
- FireRed specifics: enrollment via speechbrain `spkrec-ecapa-voxceleb` + mel/state preprocessing + `pvad.onnx` raw prob; LiveKit smoothing explicitly disabled. NeoVAD: direct streaming `step` path, no external enrollment.

## 3. Common 500 ms decoder (shared, model-blind)

Inputs per frame: GT `speech_gt` (gate first), adapter `anchor`, `lifecycle`, thresholds `(tau_primary, tau_sens)` set later on CAL12 (sweep is out of scope for Gate 1; use placeholder defaults and log them).

```text
state: run_ms = 0, run_start = null
on frame f (source_time t, dt = frame_ms):
  if not speech_gt[f]:              run_ms, run_start = 0, null; return KEEP (no accumulation)
  if lifecycle in {UNBOUND, UNCERTAIN, POISONED}: run_ms, run_start = 0, null; return HOLD
  # BOUND only from here
  if anchor[f] >= tau_primary:      run_start ??= start of first suprathreshold frame
                                    run_ms += dt
  else:                             run_ms, run_start = 0, null; return KEEP
  if run_ms >= 500:  emit CUT(source_boundary_time=run_start, decision_time=t)
  elif run_ms >= 300 and SENS_MODE: emit CUT (sensitivity check only, flagged)
  else:                             return HOLD
```

CUT backdates `source_boundary_time` to evidence-run start; `decision_time - source_boundary_time` ≈ 500 ms + compute lag. Delay reporting splits `source_boundary_error = pred_source_boundary - authoritative_transition` from `decision_delay = decision_time - pred_source_boundary` (p50/p90 each).

## 4. Gate 1 smoke checklist (each row = one concrete verification)

1. Full reset/episode isolation: run episode twice (A then B then A again); assert identical step logs for both A runs (hash equal).
2. Source-time integrity: assert `source_time_ms` strictly increases by `frame_ms`, chunk sample count matches, and CUT `source_boundary_time <= decision_time` always.
3. 16 kHz mono contract: feed 8 kHz / stereo / float32 fixtures; assert `ValueError` on each.
4. No model-specific smoothing: feed alternating 0.0/1.0 anchor impulse (stub + real adapter); assert raw `anchor` toggles with zero lag vs one-frame-delayed baseline; FireRed config asserts LiveKit smoother disabled.
5. CPU path + budget: run MAIN48 on CPU-only wheel, record median/p95/p99 `step()` wall ms, RTF = infer_time/audio_time <= 0.25, peak RSS, mean `bind()` and `reset()` cost; fail Gate 1 on RTF breach.
6. Regime-matrix + topology views: 2 models x 2 regimes (O/C) x MAIN48; headline metrics (contamination s/active-hour, false cuts, missed-replacement rate, p50/p90 split delays) sliced by `A->A+B->A` (expect KEEP) vs `A->A+B->B` (expect CUT).

## 5. Non-goals (explicit, per issue)

No third model; no threshold sweeps (CAL12 sets them later); no training/fine-tuning of any kind; no VAD retuning; no non-causal future context; no ontology/control resampling inside Gate 1; no product integration.

## 6. Top 3 design risks

1. **1 s causal enrollment too thin** — ECAPA/GRU under-determined -> mass UNBOUND/HOLD, missed-replacement rate uninformative. Mitigation: `causal_bindable` rate is itself a Gate 1 metric; do not silently fall back to native span.
2. **GT speech-gate masks acoustic VAD failure** — decoder uses GT speech, so real p_speech errors hide in `aux`. Mitigation: log both, report GT-gated (primary) and acoustic-gated (diagnostic) action streams.
3. **Frame-grid / source-clock skew between models** — FireRed vs NeoVAD native grids differ, breaking CUT comparability. Mitigation: boundary math in source samples (not frames) + per-model `frame_ms` in log header; alignment tolerance fixed at build time.
