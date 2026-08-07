# Phase 1 report — Speaker-change turn boundaries (LS-EEND vs ERes2NetV2)

GitHub issue #51, Phase 1 only. Executed on top of the committed Phase 0
commit `94b545ad9bd937345319422a36802e911e047502` in the
`experiment-v2-speaker-change-turn-boundaries-ls` worktree.

**Status: Phase 1 detector sweep executed end to end on the available
controlled development inputs. A second review found and fixed three metric
defects (GT epoch rebase for later cases, matched false-cut accounting,
raw-event detector-only metrics) plus smoke/denominator separation; all
results below were regenerated after the corrections and supersede every
earlier curated number (section 7a). All results are non-selection evidence;
no production finalist is chosen. Phase 2 corpus work, provider policies, and
production wiring were not begun.**

## 1. Scope and decisions made

Phase 1 scope owned here:

- LS-EEND: provenance/parity gate, pinned streaming frontend, stateful ONNX
  step runtime, reducer sweep (threshold 0.30-0.70 step 0.05, persistence
  1/2/3, new-speaker-onset + dominant-replacement policies, median filter
  on/off) across all four checkpoints (L-AMI, L-CALLHOME, L-DIHARD-II,
  L-DIHARD-III).
- ERes2NetV2: official artifact resolution (E-standard and E-w24s4ep4),
  official PyTorch -> ONNX export with parity, adjacent-window matrix
  (W = 0.50/0.75/1.00/1.50/2.00 s with the mandated step sets), stable-anchor
  profiles, confirmation 1/2, mutual-similarity rule, explicit skip instead of
  silent padding.
- VAD+detector coalescing and GT matching on the available corpus.
- Parity set: the three committed Phase 0 golden wavs plus three hash-pinned
  official ModelScope example wavs (`speaker1_a/b`, `speaker2_a`).

Out of scope by instruction: provider policies, production wiring, Phase 2
held-out/corpus conclusions, threshold freezing for production.

## 2. Environment

- Baseline `dev` SHA: `adf8cde2b5b166beb95c50a39e8941d2fee3601e` (recorded in
  `config.py`); worktree HEAD `94b545ad` (Phase 0 commit) at run time.
- Main experiment env: Python 3.12.10 (uv), numpy 2.5.1, onnxruntime 1.28.0
  (CPUExecutionProvider, intra/inter threads 1, ORT_ENABLE_ALL), scipy 1.18.0,
  Windows, x86-64. Full metadata is embedded in every run artifact.
- Isolated research env (out of Git, under the user's temp cache): Python
  3.12 + PyTorch 2.13.0+cpu, pytorch-lightning, librosa 0.11.0, torchaudio
  2.11.0+cpu, soundfile, modelscope, gdown. No PyTorch or research-only
  package was added to production dependencies.

## 3. Provenance and artifact registry

Machine-readable authority: `models/registry.json` (schema
`experiments.speaker_turn_boundary.provenance.v1`); documentation in
`models/README.md`.

| Artifact | Identity | License |
| --- | --- | --- |
| L-AMI ONNX step model | HF `GradientDescent2718/LS-EEND-ONNX` @ `cc40a1e1242c148fbbc15c132e43b8ac15056e53`, `ls_eend_ami_step.onnx` SHA-256 `5a2b813f...` (registry) | MIT |
| L-CALLHOME ONNX | same repo, `ls_eend_callhome_step.onnx` SHA-256 `b79b1b1c...` | MIT |
| L-DIHARD-II ONNX | same repo, `ls_eend_dih2_step.onnx` SHA-256 `5df89a22...` | MIT |
| L-DIHARD-III ONNX | same repo, `ls_eend_dih3_step.onnx` SHA-256 `587ad263...` | MIT |
| FS-EEND source | `Audio-WestlakeU/FS-EEND` @ `adcdde1327bc731cc4e718aa009b8d78317388e5` | MIT |
| Official PyTorch ckpts | `ami.ckpt` `5b1df8f0...`, `ch.ckpt` `eab0b718...`, `dih2.ckpt` `2d6de53d...`, `dih3.ckpt` `da62f7e1...` (Google Drive links from the FS-EEND README) | MIT |
| E-standard | ModelScope `iic/speech_eres2netv2_sv_zh-cn_16k-common` @ `1cf80d41fb3435bd3d8df185b5c423333b2db42a`, `pretrained_eres2netv2.ckpt` SHA-256 `0eb40571...` (matches ModelScope-declared hash) | Apache-2.0 |
| E-w24s4ep4 | ModelScope `iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common` @ `d41a54156a6216b4c7611447be0548e4b0afb1ba`, `pretrained_eres2netv2w24s4ep4.ckpt` SHA-256 `740bb658...` (matches declared hash) | Apache-2.0 |
| ERes ONNX exports (ours) | opset 17, dynamic time axis, `fbank (1,T,80)` -> `embedding (1,192)`; hashes in `models/registry.json` under `eres_onnx_exports` | derived from Apache-2.0 sources |

E-w24s4ep4 disposition: **resolved and gated** (official artifact exists and
was run), not rejected. The 3D-Speaker release notes list
`iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common` as an official pretrained
model; the checkpoint loads cleanly into the official `ERes2NetV2`
architecture with baseWidth 24 / scale 4 / expansion 4 and embedding size 192.

### LS-EEND frontend finding (addendum A)

The official FS-EEND feature code (`datasets/feature.py::stft`) computes a
**256-point FFT** (`fft_size = 1 << (frame_size - 1).bit_length()` for
`frame_size = 200`), despite `n_fft: 1024` appearing in the yaml configs and
in the third-party ONNX sidecar metadata. Phase 1 therefore pins the official
256-point semantics. The third-party example streaming extractor uses a
1024-point FFT; measured feature deviation of that extractor vs the official
256-point reference on the parity set is small (<= 1.4e-5 max abs on features)
but its tail frame-count behavior differs (one extra zero-padded frame on some
clips, one fewer on others). This is recorded in
`results/parity_research.json` under
`ls_eend_thirdparty_streaming_deviation`.

## 4. Parity gates (all passed)

Fixed parity set: 3 golden wavs + 3 official ModelScope example wavs
(hash-pinned). Full records: `results/parity_research.json` (research env)
and `results/parity_frontend.json` (main env).

| Gate | Result |
| --- | --- |
| LS-EEND frontend: my numpy logmel23_cummn vs official `extract_fbank` | max abs feature error <= 1.4e-5 on all 6 clips |
| LS-EEND whole-file vs chunked streaming (my pipeline) | same frame counts; max abs error <= 1e-6 (float64 cumsum batch order) |
| Resampler determinism | bit-identical across runs |
| LS-EEND neural: official PyTorch vs ONNX (same official features) | max abs probability error: L-AMI 4.5e-6, L-CALLHOME 6.0e-7, L-DIHARD-II 5.1e-7, L-DIHARD-III 8.9e-7 |
| ERes frontend: numpy kaldi fbank vs torchaudio | max abs error <= 5.8e-4 (float32 FFT path) |
| ERes neural: torch vs ONNX embeddings | cosine >= 0.99999994 on all clips, both checkpoints |

The ONNX profiles therefore pass the L0/E0 gate as far as the available
evidence goes (weights parity, frontend parity, export contract). The third
-party repo does not state which official checkpoint file hash each ONNX file
was exported from; weights equivalence was established numerically instead.

## 5. LS-EEND runtime and reducer semantics (pinned)

- Frontend: 16 kHz -> 8 kHz via the pinned half-band windowed-sinc decimator
  (63 taps, Hamming, DC gain 1, group delay 31 input samples); 256-point FFT,
  periodic-hann window of 200 samples centered in the FFT frame, hop 80,
  slaney mel (23 bins), log10 with 1e-10 floor, cumulative-mean
  normalization over all frames since epoch start, context 7, subsampling 10
  -> 10 Hz model frames. Streaming and offline paths are frame-identical
  (verified).
- Frame -> source mapping (16 kHz samples): model input frame `m` center is
  8 kHz sample `800*m`, available at 16 kHz count `1600*m + 1406`; decoded
  output frame `d` (input frame `d + 9`, conv_delay 9) has
  `boundary position = 1600*d + 14431` and causal availability
  `1600*d + 15806` (lookback 1375 samples = 85.9 ms; first output at
  987.9 ms). The resampler center offset (31 input samples) is folded into
  both position and availability.
- Recurrent state (`enc_ret_kv`, `enc_ret_scale`, `enc_conv_cache`,
  `dec_ret_kv`, `dec_ret_scale`, `top_buffer`) is carried across VAD pauses
  and reset only by epoch start (`start_epoch`). Tail flush uses `ingest=0`
  zero frames; tail outputs whose frame center lies beyond the epoch's source
  audio are dropped (no phantom boundaries).
- Reducer: binary decisions at `p > threshold`; optional centered median
  filter (width 11 = the official CALLHOME value; 1 = off) applied to the
  binary track with zero-padded edges; `new-speaker-onset`: one event per
  onset run (onset = first frame of the run, confirmed at `onset + P - 1`,
  emitted at that frame's availability, median shift included); onsets at
  frame 0 of an epoch are skipped (mirrors B0's first-utterance rule and the
  GT `initial_start` semantics); `dominant-replacement`: event when the
  dominant track (argmax over active tracks) changes to a previously
  inactive track and persists P frames.
- `DetectorProgress.safe_boundary_frontier_sample`: reducer guarantee, clamped
  to the observed frontier; after processing frame `f` it equals the position
  of the earliest frame that could still produce a boundary
  (`f - P + 2 - median_shift`), and the epoch-end count after finalize.
- Events carry `boundary_source_sample` (onset frame center),
  `observed_source_sample_at_emit` (16 kHz frontier when the confirming frame
  was processed), `confidence` (mean onset-track probability over the
  persistence run), and debug fields.

## 6. ERes2NetV2 runtime and detector semantics (pinned)

- Frontend: 80-dim log fbank (torchaudio kaldi defaults: 25 ms/10 ms, povey
  window, preemphasis 0.97, remove DC offset, htk mel, dither 0) at 16 kHz
  with time mean normalization; embedding size 192 for both checkpoints.
- Adjacent-window detector: boundary at `t` inside a VAD utterance scores
  cosine(embedding[t-W, t), embedding[t, t+W)); low score = change candidate;
  windows must fit fully inside the utterance, otherwise the position is
  explicitly skipped (no padding); causal availability `t + W`; confidence =
  `clamp(1 - score)`; confirmation 2 requires two consecutive candidates.
- Stable-anchor detector: anchor = normalized embedding of the first window
  after utterance start; every step a probe window is compared against the
  anchor; confirmation 1 or 2; for 2, the two candidate probes must also be
  mutually similar (mutual threshold 0.5, explicit parameter); on confirmed
  change the candidate probe is promoted to the new anchor; anchor updates
  during stable speech are explicit (`none` = frozen, or EMA alpha 0.9 with
  updates frozen while a candidate is pending); anchors reinitialize at every
  VAD utterance (no identity across turns); per-step safe frontier =
  last scanned position - step + 1.
- Mandated matrix expansion: W 0.50 (steps 0.10/0.25), 0.75 (0.10/0.25), 1.00
  (0.10/0.25/0.50), 1.50 (0.25/0.50), 2.00 (0.50); thresholds 0.30..0.70 in
  0.05; confirmation 1/2 -> 180 adjacent profiles per checkpoint. Anchor
  profiles: W 0.50/0.75/1.00/1.50 x steps 0.10/0.25 x 9 thresholds x
  confirmation 1/2 x update none/ema -> 288 profiles per checkpoint.

## 7. Sweep results (non-selection evidence; corrected run)

**This section supersedes the earlier curated numbers.** A second review found
three metric defects that invalidated the original curated results. All
numbers below were regenerated after the corrections; the full correction list
is in section 7a.

### 7a. Metric corrections applied (authoritative)

1. **GT epoch rebase.** Case-local GT regions were classified with their
   manifest epoch (`0`) while B0/detector cuts used the active experiment
   epoch, so the second manifest case could never match. Regions are now
   rebased to the active epoch before classification (`rebase_regions_to_epoch`),
   in both the LS-EEND and ERes sweep paths (the ERes path also replayed VAD at
   the correct epoch). Regression: an exact detector cut at the second
   `phase1_dev` case's GT sample matches at recall@500 = 1.0
   (`tests/test_sweep_driver.py::test_exact_detector_cut_matches_second_case_gt`).
2. **Matched false cuts.** `false_cuts` was `detector_cut_count`, so a detector
   cut exactly matching GT was counted false. False cuts are now one-to-one
   matched against GT at the pinned 500 ms product tolerance (8000 samples),
   separately for the product (VAD+detector coalesced cuts) and detector-only
   arms. Matching is a deterministic maximum-cardinality per-epoch
   two-pointer matcher (regression includes the review counterexample
   GT `[0, 50]`, cuts `[-50, 40]`, window `50` -> 2 matches).
3. **Detector-only metrics from raw events.** Detector-only recall/false cuts
   are computed from the raw pre-coalescing detector boundaries (converted to
   canonical cuts), independent of product coalescing; a detector event
   absorbed by a VAD boundary still counts for the detector-only arm. Product
   metrics keep using coalesced cuts. `speech_samples` (whole WAV length) was
   removed from `evaluate_case`.
4. **Smoke separation.** Unannotated smoke wavs are excluded from GT recall,
   false-cut, and speech-hour aggregates. They are kept as
   `smoke_epochs`/`smoke_case_count`/`smoke_detector_cut_count_total`
   diagnostics only.
5. **Active-speech denominator.** `false_cuts_per_speech_hour` divides by the
   annotated active-speech sample count (union of non-ambiguous regions with
   active speakers, overlap never double-counted): 89,600 samples for
   `b0_phase0` and 116,800 for `phase1_dev`, stored machine-readably as
   `active_speech_samples` per case, per aggregate, and per profile.
6. **B0 aggregate and incremental metrics.** Every profile result carries the
   true VAD-only B0 aggregate and `incremental_over_b0`
   (`incremental_recall_at_500ms`, `incremental_false_cuts` vs the B0 product
   false-cut total). Sweep result schema is now
   `experiments.speaker_turn_boundary.sweep.v2`; summaries carry
   `experiments.speaker_turn_boundary.sweep_summary.v1`.
7. **ERes confidence clamp.** ERes confidence is clamped to `[0, 1]` at the
   adapter boundary (`AdjacentWindowDetector`/`StableAnchorDetector`) and at
   the runner event boundary (`clamp_confidence`).
8. **ERes progress epoch scoping (independent audit fix).** An independent
   audit found that every one of the 1,872 ERes per-profile artifacts carried
   mixed-epoch progress evidence: each smoke epoch 2/3/4 had progress records
   whose `audio_epoch` set was `[0, current_epoch]`. Root cause: the
   per-profile builder closure is constructed once with a hardcoded
   `audio_epoch=0` and reused across every benchmark/smoke epoch, so the
   per-step `DetectorProgress` records were stamped 0 while only the
   epoch-end record (added by the runner) carried the true epoch. Benchmark
   epoch IDs, detector events, metrics, self-hashes, and the embedding cache
   (keyed by `(start, end)` windows, epoch-agnostic) were correct. Fix:
   `EresDetectorRunner.run_case` now rebases every emitted `DetectorProgress`
   to the runner's current epoch - the same authority that already stamps
   events and the epoch-end record - so every progress record is
   epoch-scoped with no change to detector boundaries or numerical metrics.
   Regression
   `tests/test_eres.py::test_eres_progress_epoch_scoped_across_reused_builder_and_cache`
   exercises a nonzero epoch through the reused builder + shared embedding
   cache and fails on the pre-fix mixed `[0, current]` set. The two
   authoritative ERes sweeps were regenerated (1,872 artifacts, section 9)
   and independently verified; the committed summaries are byte-identical to
   the pre-fix summaries (aggregates/metrics unchanged, as required).

### 7b. Corrected results

B0 (current PuriPuly VAD-only) on `phase1_dev`: **1 VAD cut** (case
`zero_gap_handoff_ab` at sample 44032, 4032 samples = 252 ms after the GT
change at 40000), so B0 product recall@500 = **0.5 (1/2) with 0 product false
cuts** and `incremental_recall_at_500ms = 0` for every profile. On
`b0_phase0` B0 recall@500 = 0.0 (gap-separated change; VAD emits no boundary
for it). **The earlier curated claim of "B0 recall 0/2" is retracted.**

#### Corpus

- `b0_phase0` (committed Phase 0 manifest, 3 clips): 1 GT change, 89,600
  active-speech samples. Degenerate for detector recall: the synthetic tone
  clips do not trigger LS-EEND speaker activity and the only GT change is
  gap-separated, so ERes (within-utterance by design) cannot reach it.
- `phase1_dev` (deterministic, `data/manifests/phase1_dev.json`, hash
  `6dc4939b24f37761055b65553dea4ee076b4adc3e716eec9511100b2872feed0`):
  `zero_gap_handoff_ab` (clean handoff A->B at junction sample 40000) and
  `overlap_300ms_ab` (interruption onset A->A+B at sample 40000), 2 GT changes,
  116,800 active-speech samples. `b0_phase0` manifest hash
  `c6938a08f178d359764b5fc59d37fc978a94e3ff54cb37f4f90e45b888c48d3a`.

#### LS-EEND (4 checkpoints x 108 profiles each = 432 per manifest)

| Manifest | Best product recall@500 | Best detector-only recall@500 | Max product false cuts | GT |
| --- | --- | --- | --- | --- |
| b0_phase0 | 0.0 | 0.0 | 1 | 1 |
| phase1_dev | 0.5 (B0's own VAD cut) | 0.0 | 4 | 2 |

**Detector-only recall is 0.0 at all 432 profiles on both manifests.** On
`phase1_dev`, 162 of 432 profiles emit 1-4 detector events (e.g. L-DIHARD-III
`new_speaker_onset@thr0.70-p1-med1` emits boundaries at 56031 / 65631 / 70431 /
73631, i.e. 1.0 s or more after the GT change at 40000); none lands within
500 ms of a GT change, so every emitted event is a false cut at some profile
(max 4 product false cuts). The earlier curated "0.5/0fc reached by every
checkpoint" figure is retracted: that 0.5 was B0's VAD cut, not LS-EEND
detection. No checkpoint dominance can be claimed from this corpus.

#### ERes2NetV2 (2 checkpoints x 468 profiles each = 936 per manifest)

| Manifest | Best product recall@500 | Best detector-only recall@500 | Max product false cuts | GT |
| --- | --- | --- | --- | --- |
| b0_phase0 | 0.0 | 0.0 | 0 | 1 |
| phase1_dev | 0.5 (B0's own VAD cut) | 0.0 | 0 | 2 |

**ERes emits zero detector events at all 936 profiles on both manifests**
(detector_events_total = 0 everywhere; the earlier curated "adjacent and
anchor each recover 1 of 2 changes at 0 false cuts" is retracted - that 0.5
was B0's VAD cut). The smoke clips (real speech, no GT) do engage the
operating-curve machinery: up to 51 ERes detector cuts and up to 4 LS-EEND
detector cuts per profile across the smoke records, reported separately as
`smoke_detector_cut_count_total` and never in benchmark aggregates. Both
checkpoints behave identically on this corpus.

#### Authoritative takeaways (non-selection)

- On the available synthetic dev corpus, neither detector family adds any
  recall over B0 at the 500 ms product tolerance
  (`incremental_recall_at_500ms = 0` everywhere), and every LS-EEND event
  emitted on `phase1_dev` is a false cut at the 500 ms tolerance
  (incremental product false cuts up to 4; ERes adds none).
- The synthetic Phase 1 dev corpus does not engage either detector family;
  smoke/real-speech evidence shows the harness operates, so this is a corpus
  limitation, not a harness failure. Phase 2 real-corpus work is required
  before any detector conclusion.
- Full per-profile aggregates with per-profile B0 and incremental metrics:
  `results/sweep_ls_eend_summary_{b0_phase0,phase1_dev}.json` (432 profiles
  each) and `results/sweep_eres_summary_{b0_phase0,phase1_dev}.json` (936
  profiles each); regenerable per-profile event traces are git-ignored and
  were written to the external temp scratch during this run (section 9).

### Data correction history (superseded, kept for provenance)

- Early sweep driver filtered manifest GT regions by `audio_epoch ==
  epoch_index`, silently dropping GT for every case after the first epoch.
  Fixed (regression test `test_multi_case_epoch_gt_attribution`) and all
  summaries regenerated.
- The subsequent curated numbers ("B0 recall 0/2", "detectors recover 1 of 2
  changes", "best recall@500 = 0.5" for both detector families on
  `phase1_dev`) were invalidated by the second review's three metric defects
  (epoch rebase for the second case, matched false cuts, raw-event
  detector-only metrics) and the smoke/denominator separation. Section 7b
  above is the only authoritative result set.
- An independent audit then found that all 1,872 ERes per-profile artifacts
  carried mixed `[0, current_epoch]` progress records in every smoke epoch
  (stale builder-closure epoch, section 7a item 8). Fixed, regression-tested,
  and the two authoritative ERes sweeps regenerated into
  `%TEMP%\opencode\stb_phase1_eres_epoch_fix`; aggregate numbers and the four
  committed summary hashes are unchanged (byte-identical, section 9).

### Caching and runtime limitation

- LS-EEND: stateful ONNX step inference is ~1.1 ms/step; per-profile wall
  time is dominated by the Silero B0 replay and event serialization
  (~0.5-1 s/profile; exact `wall_seconds` per profile is recorded in the
  per-profile artifacts, which are regenerable and not committed).
- ERes: embeddings are cached per checkpoint keyed by (start, end) sample
  windows and shared across ALL profiles (thresholds, confirmations, and
  adjacent/anchor families reuse the same window embeddings, e.g. anchor
  probe windows coincide with adjacent right windows). Measured per-window
  compute: E-standard ~40 ms, E-w24s4ep4 ~100 ms for a 1 s window on this
  machine; per-profile `embedding_compute_seconds_mean/p95` and cache sizes
  are recorded in the per-profile artifacts. Consequence: total ERes sweep
  cost scales with unique windows per checkpoint, not with the 936-profile
  count.
- Limitation: per-profile artifacts (full event traces) are git-ignored; only
  the aggregate summaries, parity records, and registry are committed. The
  exact regeneration commands are in section 9. The corrected authoritative
  rerun wrote all per-profile scratch to the external temp cache
  (`%TEMP%\opencode\stb_phase1_rerun\...`), not to the repository. After the
  epoch-progress correction (section 7a item 8), the two authoritative ERes
  sweeps were regenerated into the new scratch root
  `%TEMP%\opencode\stb_phase1_eres_epoch_fix` (1,872 per-profile artifacts,
  one summary per manifest); every artifact was verified for literal `.json`
  suffix, valid `result_sha256`, unique benchmark+smoke epochs, epoch-scoped
  nested GT/VAD/detector/coalesced-cut/progress records, monotonic
  progress with `frontier <= observed <= WAV length`, empty/excluded smoke
  GT, and per-case metric recomputation from raw detector events vs product
  cuts (333,808 checks passed, 0 failed), and both regenerated summaries are
  byte-identical (SHA-256
  `7ABD5FEF8BA275169F49A365F363FBD26B8044528AD189F239F32AEEFEE69E22` for
  `b0_phase0`, `078F37C37E7BA18EDFC283C1336AAEA5D5979A25D175A73FD7638822BF7D6BE7`
  for `phase1_dev`) to the committed
  `results/sweep_eres_summary_{b0_phase0,phase1_dev}.json`.

## 8. What passed and what is blocked

Passed:

- L0 gate: provenance, hashes, licenses, sidecar metadata, state shapes,
  frontend pinning, whole-file vs chunked comparison, frontend parity vs
  official reference, neural parity vs official PyTorch for all four
  checkpoints.
- E0 gate: E-standard and E-w24s4ep4 official artifacts resolved and
  hash-verified; torch/ONNX cosine parity; frontend parity vs torchaudio.
- L2 reducer sweep and E1/E2 matrices executed on both manifests with
  machine-readable records (432 + 936 profiles per manifest; all 2736
  per-profile artifacts self-hash-verified).
- Coalescing + product (VAD+detector) and detector-only metrics with
  deterministic maximum-cardinality one-to-one GT matching at
  250/500/1000/1500/2000 ms deadlines, matched false cuts at the pinned
  500 ms product tolerance, smoke/benchmark separation, annotated
  active-speech denominators, and per-profile B0 + incremental metrics.

Blocked / not claimable:

- No detector recall is claimable on this synthetic corpus: detector-only
  recall@500 is 0.0 at all 432 LS-EEND and 936 ERes profiles on both
  manifests, and every LS-EEND event emitted on `phase1_dev` is a false cut
  at the 500 ms tolerance. Recall@250 is also 0.0 everywhere on both
  manifests (B0's only match is 252 ms past the change, so it qualifies only
  at the 500 ms deadline).
- Threshold freezing and any production finalist choice are Phase 3 items and
  are explicitly not performed.
- Real-corpus generalization (AMI, AliMeeting, LibriSpeech synthetic) is
  Phase 2 and was not started, per instruction.

## 9. Reproducible commands

Tests, lint, and formatting (from the repository root):

```powershell
uv run pytest experiments/speaker_turn_boundary/tests -q
uv run ruff check experiments/speaker_turn_boundary
uv run --extra dev black experiments/speaker_turn_boundary
uv run --extra dev black --check experiments/speaker_turn_boundary
```

Regenerate the Phase 1 dev cases and manifest (byte-identical):

```powershell
uv run python -m experiments.speaker_turn_boundary.build_phase1_cases
```

Frontend/provenance parity (main env; artifact paths are the local cache):

```powershell
uv run python -m experiments.speaker_turn_boundary.run_parity `
  --data-dir experiments/speaker_turn_boundary/data `
  --hf-root <cache>/LS-EEND-ONNX/repo --ckpt-root <cache>/ckpts `
  --eres-std-root <cache>/eres_std --eres-w24-root <cache>/eres_w24
```

Research-env parity (torch/librosa; see `models/README.md` for the env):

```powershell
<cache>/research-venv/Scripts/python.exe experiments/speaker_turn_boundary/research_parity.py `
  --data-dir ... --hf-root ... --ckpt-root ... --eres-std-root ... `
  --eres-w24-root ... --eres-onnx-root ... --fs-eend-root ... `
  --speaker-root ... --cache-dir ... --out <tmp>/parity_results/research_parity.json
```

Sweeps (per manifest; smoke wavs are the hash-pinned ModelScope examples).
Run `--out` into a temp scratch directory, then copy the four
`sweep_*_summary_<manifest>.json` files into `results/`. Per-profile ERes
artifacts are named
`sweep_eres_<manifest>_<checkpoint>_<kind>_<profile_id>.json` with dots in the
profile-derived stem only rewritten to `p` (the literal `.json` suffix is
preserved and regression-tested in `tests/test_eres.py`):

```powershell
uv run python -m experiments.speaker_turn_boundary.run_ls_eend_sweep `
  --hf-root <cache>/LS-EEND-ONNX/repo --manifest <manifest.json> `
  --out <tmp-scratch> --smoke-dir <cache>/parity_cache
uv run python -m experiments.speaker_turn_boundary.run_eres_sweep `
  --eres-onnx-root <cache>/eres_onnx --manifest <manifest.json> `
  --out <tmp-scratch> --smoke-dir <cache>/parity_cache
```

Full sweep: 432 LS-EEND profiles and 936 ERes profiles per manifest, ~5-10
minutes each on this machine.

Epoch-progress-corrected authoritative ERes rerun (exact reproduction
evidence; both manifests into one scratch root, summaries byte-identical to
the committed files, section 7a item 8):

```powershell
uv run python -m experiments.speaker_turn_boundary.run_eres_sweep `
  --eres-onnx-root $env:TEMP/opencode/eres_onnx `
  --manifest experiments/speaker_turn_boundary/data/manifests/b0_phase0.json `
  --out $env:TEMP/opencode/stb_phase1_eres_epoch_fix `
  --smoke-dir $env:TEMP/opencode/parity_cache
uv run python -m experiments.speaker_turn_boundary.run_eres_sweep `
  --eres-onnx-root $env:TEMP/opencode/eres_onnx `
  --manifest experiments/speaker_turn_boundary/data/manifests/phase1_dev.json `
  --out $env:TEMP/opencode/stb_phase1_eres_epoch_fix `
  --smoke-dir $env:TEMP/opencode/parity_cache
```

Pre-fix defect reproduction (builders stamped `audio_epoch=0` into later
epochs' progress records) and the post-fix epoch sets are demonstrated by
`tests/test_eres.py::test_eres_progress_epoch_scoped_across_reused_builder_and_cache`.

## 10. Remaining work for Phase 2/3

- Phase 2 corpus: LibriSpeech-based D1 synthetic (with acoustically verified
  zero-gap rule), AMI and AliMeeting D2 subsets, mixed dev pool D3,
  PuriPuly-like acceptance set D4.
- Freeze reducer/window/threshold parameters on the dev pool before held-out
  reporting; matched false-cut operating curves at 0.5/1/2/5 extra cuts per
  speech-hour; short-turn/no-gap/overlap condition breakdowns.
- Compute/RSS/RTF profiles and a concurrency smoke with local batch ASR.
- Phase 4 provider-policy work (Soniox, Deepgram, Qwen) with oracle
  boundaries; Phase 5/6 replay and conformance.
- Then `detector_results.md`, `provider_oracle_policy_results.md`,
  `end_to_end_results.md`, `experiment_handoff.md` and follow-up
  implementation issues.

## 11. Architecture notes

No production code was touched. The only Phase 0 contract interaction was
read-only reuse (manifest/result schemas, `VadBoundaryReplay`, coalescing).
No architecture drift is suspected in the application; one Phase 0 harness
limitation is confirmed and unchanged: the B0 replay of the committed corpus
emits 0-1 boundaries per clip, which Phase 1 works around with the new
`phase1_dev` manifest rather than by altering Phase 0 artifacts.
