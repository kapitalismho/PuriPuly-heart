# Speaker turn boundary experiment — Phase 0 harness

Fresh experiment harness for GitHub issue #51 (speaker-change turn boundaries:
LS-EEND vs ERes2NetV2 and streaming ASR policies).

This directory implements **Phase 0 only**: baseline recording, canonical
timeline, B0 VAD replay, coalescing semantics, ground-truth transition rules,
deterministic manifests/result schemas, and the Phase 0 usage documentation.
No detector, model, or provider work is included.

## Baseline and issue authority

- Experiment baseline: exact `dev` SHA `adf8cde2b5b166beb95c50a39e8941d2fee3601e`
  (recorded in `config.py` as `BASELINE_SHA` and re-derived from git at run
  time by `metadata.resolve_baseline_sha`).
- Authority: issue #51 body plus its normative v2 addendum comment.
- This experiment has **no execution dependency on the #49 experimental
  branch/harness**. The worktree was created fresh from `dev`; nothing here
  imports or copies #49 code.

## Layout

```text
experiments/speaker_turn_boundary/
    README.md                 this file
    config.py                 pinned constants (16 kHz domain, B0 profile, coalescing)
    timeline.py               SourcePosition, epoch registry, validation
    events.py                 SpeakerBoundaryEvent, DetectorProgress
    ground_truth.py           active-speaker-set transition classifier
    synthetic.py              deterministic 16 kHz audio generators (numpy only)
    vad_baseline.py           B0 adapter over the actual dev VAD + replay driver
    coalescing.py             VAD+detector coalescing and logical cuts
    schemas.py                manifest/result schemas, canonical JSON, sha256
    metadata.py               runtime metadata collection
    build_synthetic_cases.py  generate cases + dataset manifest (CLI)
    run_b0_replay.py          replay production VAD over manifest cases (CLI)
    data/                     manifests + generated wavs (all regenerable)
    results/                  run artifacts
    tests/                    Phase 0 automated tests
```

## Usage

Run tests (from the repository root):

```powershell
uv run pytest experiments/speaker_turn_boundary/tests -q
```

Regenerate the committed dataset (byte-identical output; the committed
`data/manifests/b0_phase0.json` pins the hashes):

```powershell
uv run python -m experiments.speaker_turn_boundary.build_synthetic_cases
```

Replay production VAD (B0) over the manifest and write a result artifact:

```powershell
uv run python -m experiments.speaker_turn_boundary.run_b0_replay
```

Coalesce an external detector trace (list of `SpeakerBoundaryEvent` dicts)
against B0:

```powershell
uv run python -m experiments.speaker_turn_boundary.run_b0_replay `
  --detector-events experiments/speaker_turn_boundary/results/oracle_events.json
```

## Pinned Phase 0 decisions

### Canonical source timeline and epochs

- Canonical time domain is the continuous 16 kHz mono source-audio timeline.
  `SourcePosition { audio_epoch: int, sample_index_16k: int }` is the canonical
  identity; `milliseconds = sample_index_16k / 16`.
- An `audio_epoch` is a continuous source session. Epochs are non-negative
  strictly increasing integers; samples restart at 0 per epoch. An epoch's
  valid sample range is `[0, epoch_length_samples)` once the epoch is closed.
- A stale epoch is any epoch below the current one. Events referencing a stale
  epoch are dropped and counted by consumers (`coalescing` reports
  `stale_detector_events`); events with out-of-range samples or unknown epochs
  raise `TimelineError` (`EpochRegistry.validate_sample`).
- All detectors, VAD, GT, and policies must express positions on this domain.
  No provider-relative or wall-clock-only values are canonical.

### SpeakerBoundaryEvent

```text
audio_epoch: int
boundary_source_sample: int
observed_source_sample_at_emit: int
emitted_monotonic_ns: int
confidence: float | null
source: str
debug: dict
```

- `boundary_source_sample`: the detector's best retrospective estimate of the
  new-speaker onset on the 16 kHz source timeline.
- `observed_source_sample_at_emit`: source-audio frontier available to the
  detector when the event became available.
- `event_lookback_ms = (observed_source_sample_at_emit - boundary_source_sample) / 16`.
- `emitted_monotonic_ns`: real scheduling time (for worker/backlog latency
  measurement only; never canonical).
- Validation: `0 <= boundary <= observed`, `emitted_monotonic_ns >= 0`,
  `confidence` in `[0, 1]` or `None`, non-empty `source`.

### DetectorProgress

```text
audio_epoch: int
observed_source_sample: int
safe_boundary_frontier_sample: int
```

`safe_boundary_frontier_sample` means: the detector guarantees that no future
event in this epoch will refer to a boundary at or before this sample
(required for safe Qwen end-of-speech flushing). Invariant:
`0 <= safe_boundary_frontier_sample <= observed_source_sample`, and both are
non-decreasing within an epoch.

B0's own progress: after processing chunk *k*, `observed_source_sample` is the
end of chunk *k* and `safe_boundary_frontier_sample` is the start of chunk *k*
(the next possible B0 boundary is at least one full chunk later, so this bound
is tight for the VAD-only policy).

### B0 — current production VAD boundaries

- B0 replays the **actual dev behavior**: `SileroVadOnnx` (bundled model,
  session options intra/inter threads = 1, `ORT_ENABLE_ALL`,
  `CPUExecutionProvider`) driven by `VadGating` via
  `create_peer_vad_gating`, i.e. the peer profile: threshold 0.5, start
  debounce/commit 3 chunks, max segment 7000 ms, hangover 500 ms, pre-roll
  500 ms, 512-sample chunks at 16 kHz.
- B0 emits one `SpeakerBoundaryEvent` per VAD turn boundary (the transition
  between two completed VAD utterances), pinned at:
  - `boundary_source_sample` = first sample of the next utterance's committed
    start chunk (`SpeechStart` chunk start);
  - `observed_source_sample_at_emit` = end of that chunk;
  - `source = "vad_b0"`, `confidence = None`;
  - `debug` carries profile constants, `prev_utterance_seq`, start chunk
    index, pre-roll samples, `prev_speech_end_sample`, `gap_samples`,
    trailing silence, and end reason.
- `prev_speech_end_sample` is derived from the dev `SpeechEnd` event as
  `(end_chunk_index + 1 - silence_run) * chunk_samples` where
  `silence_run = round(trailing_silence_ms / chunk_ms)` (the last
  above-threshold chunk; `trailing_silence_ms = 0` for `max_duration` ends).
- The first utterance of an epoch never emits a boundary (no previous turn),
  mirroring the GT rule that `{} -> {A}` is an initial start, not a change.
  An utterance end without a following start emits nothing (the boundary
  belongs to the next onset).
- B0 is a "detector" in the experiment's sense but has no retrospective
  component: its safe frontier equals the start of the last processed chunk.

### VAD + detector coalescing

- One logical cut is created per VAD boundary; a detector boundary near an
  existing VAD boundary must be coalesced rather than creating a duplicate
  cut.
- Pinned rule (deterministic): detector events are processed in
  `(audio_epoch, boundary_source_sample, arrival index)` order. Each event
  matches the nearest VAD boundary in the same epoch whose distance
  `|detector_boundary - vad_boundary| <= window_samples` (tie → earlier VAD
  boundary). One VAD boundary absorbs at most one detector event:
  - matched and free → `coalesced` (no new cut; `coalesced_count += 1`);
  - matched but already absorbed → `duplicate` (no cut; `duplicate_count += 1`);
  - unmatched → new `detector` logical cut (`detector_cut_count += 1`);
  - stale epoch (below the B0 epoch) → dropped (`stale_detector_events += 1`).
- Default `window = 500 ms` (8000 samples at 16 kHz, `VAD_COALESCE_WINDOW_MS`);
  it is a sweepable parameter to be frozen on development data in Phase 1.
- Report counts: `vad_cut_count`, `detector_events_total`,
  `stale_detector_events`, `coalesced_count`, `duplicate_count`,
  `detector_cut_count`, `total_logical_cuts = vad_cut_count + detector_cut_count`.

### Phase 1 benchmark metrics (corrected semantics)

Phase 1 adds benchmark metrics over the coalesced logical cuts. The pinned
definitions (fixed after the Phase 1 metric review; see `PHASE1_REPORT.md`
section 7a):

- **Matching**: deterministic maximum-cardinality one-to-one GT/cut matching
  per `audio_epoch` (both sides sorted by position, 1D two-pointer greedy,
  which is optimal for maximum cardinality). Used consistently for recall at
  every deadline and for matched false-cut accounting.
- **Recall**: `recall_at_ms` matches GT changes against the product
  (coalesced VAD + detector) cuts at 250/500/1000/1500/2000 ms deadlines;
  `detector_only_recall_at_ms` matches GT changes against the **raw
  pre-coalescing detector boundaries** (converted to canonical cuts),
  independent of product coalescing: a detector event absorbed by a VAD
  boundary still counts for the detector-only arm.
- **False cuts**: one-to-one matched at the pinned 500 ms product tolerance
  (`PRODUCT_FALSE_CUT_TOLERANCE_MS = VAD_COALESCE_WINDOW_MS`). A cut that
  matches no GT change within the tolerance is false; a cut exactly matching
  GT is never false. `product_false_cuts` counts over the coalesced cuts;
  `detector_only_false_cuts` counts over the raw detector events.
- **Smoke separation**: unannotated smoke wavs are never used in GT recall,
  false-cut, or speech-hour aggregates. They are kept as separate
  `smoke_epochs` diagnostics plus `smoke_case_count` and
  `smoke_detector_cut_count_total` on the aggregate.
- **Speech-hour denominator**: `false_cuts_per_speech_hour` divides by the
  annotated active-speech sample count (union of non-ambiguous regions with
  active speakers; overlapping regions are never double-counted). The exact
  sample count is stored machine-readably as `active_speech_samples` on every
  case metric, aggregate, and summary entry. Whole WAV length is never used
  as the denominator.
- **B0 and incremental**: every profile result carries the true VAD-only B0
  aggregate (`b0_aggregate`, computed with no detector events) and
  `incremental_over_b0`: `incremental_recall_at_500ms` and
  `incremental_false_cuts` = candidate product false cuts minus B0 product
  false cuts (clamped at 0).
- **Result schema**: sweep results use
  `experiments.speaker_turn_boundary.sweep.v2`; summaries use
  `experiments.speaker_turn_boundary.sweep_summary.v1` and are structured as
  `{schema_version, manifest_id, detector_family, variants: {checkpoint:
  {profile_id: {aggregate, b0, incremental_over_b0}}}}`.

### Ground-truth active-speaker-set transitions

Regions are contiguous segments with a constant active-speaker set
(`SpeakerRegion {audio_epoch, start_sample, end_sample, speakers, ambiguous}`).
The classifier implements the issue's table exactly:

```text
{A}     -> {B}       clean_handoff (positive, at B onset)
{A}     -> {A,B}     interruption_onset (positive, at B onset)
{A,B}   -> {B}       speaker_left (NOT positive)
{A}     -> {} -> {B} gap_speaker_change (positive, at B onset)
{A}     -> {} -> {A} gap_same_speaker (NOT positive)
{}      -> {A}       initial_start (NOT positive)
```

Generalizations pinned for sets outside the table:

- disjoint non-empty sets → `clean_handoff` (positive);
- overlapping sets with at least one new speaker → `interruption_onset`
  (positive);
- no new speaker (subset/equal) → `speaker_left` / `same_speaker`
  (not positive);
- gap comparisons use the last non-empty set before the silence; silence
  persists across multiple empty regions;
- `ambiguous: true` regions are tagged `ambiguous`, comparisons across them
  are excluded (next region is tagged `ambiguous_adjacent`), and no positive
  or negative label is produced for the excluded transition.

`SpeakerChangeGT` carries only positive changes
(`clean_handoff` / `interruption_onset` / `gap_speaker_change`) with
`change_sample` = onset sample of the next region.

### Dataset manifests and result schemas

- Deterministic serialization: `canonical_json` =
  `json.dumps(sort_keys=True, indent=2, ensure_ascii=False)`; all hashes are
  SHA-256 over the canonical JSON bytes (result hash computed over the
  artifact excluding the `result_sha256` field itself).
- The manifest pins: baseline SHA, schema version, canonical sample rate,
  generator identity + seed, and per case: `case_id`, wav relative path,
  duration, `wav_sha256` (over raw wav file bytes), seed, and GT regions.
  It is regenerable byte-for-byte (`build_synthetic_cases.py`, seed 7) and
  `validate_manifest` rejects wrong sample rates, missing/tampered wavs, and
  duration mismatches.
- Every run artifact records: baseline SHA, profile ID, manifest id + hash,
  seed, runtime metadata (Python, Windows, CPU, RAM, ORT version, ORT thread
  configuration, VAD profile), start/end UTC times, per-epoch event/progress
  traces, and the coalescing report with counts. A canonical hash over the
  artifact is embedded as `result_sha256` and verifiable
  (`RunResult.verify_self_hash`).
- No timestamps, UUIDs, or wall clocks enter manifest content; run artifacts
  may contain wall time by design (reproducibility identity lives in the
  hash fields).

### Runtime metadata

`metadata.collect_runtime_metadata()` records: git baseline SHA (resolved
from the repo, falling back to the pinned constant), Python version/full
string/implementation, `platform.platform()`/release/version, machine,
processor, `os.cpu_count()`, RAM total (via `psutil`, `None` if unavailable),
`onnxruntime.__version__` (or `None`), and the B0 ORT session contract
(CPUExecutionProvider, intra/inter = 1, ORT_ENABLE_ALL) plus B0/coalescing
profile constants.

## B0 equivalence evidence

- Deterministic fake-engine tests exercise the exact boundary derivation
  (first-utterance rule, boundary at next start chunk, observed-at-emit,
  previous speech-end formula, gap, max-duration ends, progress invariants,
  epoch reset) with sample-exact expectations.
- Real-model tests run the bundled Silero ONNX through the B0 adapter and
  the raw dev pipeline on the same frozen clips and assert: run-to-run
  determinism (identical event and progress traces), and equivalence with the
  dev event stream (same boundary positions, observed frontiers, and debug
  fields).
- Regeneration tests prove the committed manifest validates the
  regenerated wavs byte-for-byte.

Known limitation: the Phase 0 synthetic corpus is generated "speech-like"
audio (harmonic stacks, formant-filtered noise); the production Silero VAD
detects at most one utterance per clip, so the real-model clips exercise
determinism/equivalence but not a two-utterance boundary. The two-utterance
boundary path is covered sample-exactly by the deterministic fake-engine
tests. Phase 1 replaces this corpus with LibriSpeech-based synthesis per the
issue's D1 rules.

## What is explicitly out of scope here

- Detector/model/provider implementation, model/corpus downloads, paid or
  credentialed calls, and production wiring (Phase 1+ and follow-up
  implementation issues).
- #49 branch/harness reuse.
- Committing large blobs: everything under `data/` is regenerable and the
  committed wavs are tiny (tens of KB each).
