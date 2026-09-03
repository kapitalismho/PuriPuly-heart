# #117 PSEM small-model family probe — factual inventory (read-only, 2026-09-03)

No code changes, no installs, no model downloads. All claims grounded in repo reads
(see paths below). Worktree: `experiment-v2-speaker-change-turn-boundaries-ls`.
#107 (`psem_sortformer_adaptation_depth`) is in progress and independent — its
contract/config were read but not modified.

## 1. Where is PSEM-STRATEGY-DATA-v2?

- Root: `experiments/psem_training_strategy_gate/data/v2/` — the immutable
  natural-data package (authority: issues #76 model experiment / #86 dataset).
- Freeze identity: `data/v2/dataset_freeze.json`
  sha256 `bc7e63bb201c2a33a9b2d69b2364fed8f03839278098f0bd175d6833b330a41e`;
  payload sha256 `f9f1882d0de08a4fcd19e63f1da7ae022f940420863be5bbfc14d1d2a7b0f95e`.
  Corroborated in `contract.json`, `config.json` (#76 gate), `provenance.py`,
  `BASELINE.md`, and #107 `contract.json` (freeze/split/source/annotation/topology
  paths + sha256 pins).
- Format: one JSON object per line (`.jsonl`) manifests + JSON summaries +
  executable generator contract (`data/label_contract.py`, `psem-handoff-v1`).
- Counts: 93 meetings total.
  `source_manifest.jsonl` 93 rows; `annotation_manifest.jsonl` 93 rows;
  `topology_manifest.jsonl` 93 rows; `normalization_manifest.jsonl` 93 rows
  (verified with `wc -l`). `preflight.py` pins `EXPECTED_SOURCE_COUNT = 93`.
- Roles/splits: leakage-safe component-disjoint TRAIN/DEV/EVAL in
  `data/v2/split_manifest.json` (keys: `assignments[]` with
  `component_id/source_ids/role/eval_eligible`, plus `official_roles`,
  `role_summaries`, `leakage_audit`, `selection_order`, `assignment_sha256`).
  Occupancy `config.json:dataset.role_source_counts` gives TRAIN 64 sources;
  #107 `data_split_receipt.json` records DEV AMI 7 + AliMeeting 3,
  EVAL AMI 11 + AliMeeting 8. DEV scored hours ≈ 4.57 h AMI + 1.54 h AliMeeting
  (from `role_summaries`). EVAL was freshness-eligible-only; DEV by seeded
  component-prefix search; remainder TRAIN (`summary_rationale`).
- Source-row fields (sampled first row, AliMeeting R0004_M0012):
  `audio_ref` (e.g. `alimeeting/far_ch0/R0004_M0012.wav`), `corpus`,
  `corpus_version` (M2MeT Train_Ali), `duration_samples`, `channels: 1`,
  16 kHz mono PCM16, `annotation_ref/annotation_sha256`,
  `annotation_coverage_start/end_sample`, `contract_version`,
  `eval_eligible/eval_eligibility_reason`, `component_id`, `license_id`,
  `audio_source_url` (openslr.org/119). Waveform bytes live OUTSIDE the repo
  under external `PSEM_CORPUS_ROOT`; reference checkout under
  `PSEM_REFERENCE_ROOT` (`nttcslab-sp/diar-forced-alignment@9527b7c`).
- Labels (`operational_label_contract.json`, `psem-handoff-v1`):
  16 kHz zero-based half-open unsnapped source samples; grid mapping forbidden
  in dataset labels. Constants: reliable_solo_min 200 ms, annotation jitter
  50 ms, gap/overlap topology min 100 ms, local_continuity_max_gap 1200 ms,
  short backchannel 200–1000 ms. GT speech = commit-pinned Horiguchi et al.
  forced alignments (same-speaker rows unioned only when overlapping/touching;
  cross-speaker rows preserved as overlap; neural/model repair and alignment
  reproduction forbidden). Topology: `official_primary_topology_precedence`
  list, exclusive counting, ambiguity/mask rules, nonlexical mask,
  `boundary_reconciliation`, `forbidden_task_authority`.
- Topology-row fields: `exclusive_primary_episode_count`, `label_result_sha256`,
  `mask_diagnostics` (`actual_transition_count`, `masked_transition_count/fraction`,
  reasons `complex_overlap_transition` / `continuity_unknown` / `mixed`…),
  `ambiguous_samples`. Anchor episodes / C1–C6 strata are NOT stored — they are
  derived at runtime (`derive_relative_occupancy.py` → GT intervals with
  `active_speakers/masked/start_sample/end_sample`).
- Model input convention (README): raw 16 kHz 3 s window `[t−2 s, t+1 s)`,
  30 cells at 100 ms, shared 256-d common head (#76 arms
  FROZEN-WAVLM / FINETUNE-WAVLM / SCRATCH-PSEM).

## 2. Episode-manifest structures reusable for PSEM-SMALL-MODEL-PROBE-v1

- Best templates (all in `experiments/psem_relative_occupancy_gate/`):
  - `results/dev|eval/relative_occupancy_manifest.jsonl` — per-source derived
    view: `source_id, component_id, sample_rate_hz: 16000,
    scored_start/end_sample, audio_ref`. Produced by
    `derive_relative_occupancy.py --roles PSEM-STRATEGY-DEV|EVAL`.
  - `results/dev/gate0_oracle_events.jsonl` — per anchor episode:
    `anchor_episode_id, anchor_id, boundary_source_sample,
    confirmation_ms/samples, decoder_emit_sample,
    model_evidence_frontier_sample, schema psem.relative_occupancy.gate0_event.v1`.
  - `results/dev/gate1_event_ledger.jsonl`, `gate2_event_ledger.jsonl` —
    per-source ledgers with `annotated_episodes[]` (`episode_id,
    opportunity/candidate/anchor_emit/end_emit_sample, expected_anchor_speaker,
    correct_anchor, end_reason`), plus boundary/evidence-frontier/emit/lifecycle/
    fail-closed exposure fields; verifier regenerates every DEV artifact
    (pattern to copy for a Gate-0 manifest freeze).
  - `results/dev/dev_selection_receipt.json` — frozen selection binding
    (`artifact_bindings, config_sha256, gate0/gate1 sha, manifest_sha256,
    causal_enrollment_grid, selected_settings, eval_status/sealed`).
- Prior CAL/MAIN splits: NONE. Repo-wide grep for
  `CAL12|MAIN48|EXT24|ONTOLOGY16|CONTROL24|PSEM-SMALL-MODEL-PROBE` returns zero
  hits (only the V2 id `PSEM-STRATEGY-DATA-v2` matches). No C1–C6 strata fields
  exist in V2. Only reusable split is V2 TRAIN (64 src) / DEV (10 src) /
  EVAL (19 src) component lists; #107 reuses them verbatim
  (`data_split_receipt.json`, `contract.json:dataset`).
- Subset precedent: `psem_frozen_ceiling_gate/frozen_inputs/source_manifest.jsonl`
  (29 old-#97 DEV/EVAL sources, hash-preserving) + `source_evidence_provenance.json`.

## 3. Cached outputs: F0, G, production VAD

- F0 = frozen Sortformer (`diar_streaming_sortformer_4spk-v2.1.nemo`,
  4 slots, native 80 ms grid, Vulkan `low_latency` backend, 8 threads,
  480 ms chunks; earlier CPU traces non-authoritative):
  - #97 `results/dev|eval/sortformer_model_receipt.json` — full-source
    contiguous-coverage receipts (uninterrupted model epoch, `state_reset[0]=true`,
    family/backend/role/source-path bindings, shared external trace root
    per family; traces themselves outside Git).
  - Self-contained copy:
    `psem_frozen_ceiling_gate/frozen_inputs/{dev,eval}_sortformer_model_receipt.json`
    + `posterior_sessions.npz` (8.8 MB; keys
    `sNNN_{episode_ids,speakers,starts,ends,posterior_centers,frontiers,probabilities,alive,reset,valid}`;
    e.g. s000 has 15818 cells) + `issue98_vad_reference.json` +
    `source_evidence_provenance.json`. LSEEND peer traces sit alongside
    (non-authoritative for this probe).
  - Hidden representation (HIDDEN-CEILING-1): 192-d output of 18th post-LN
    temporal Transformer block before `diar.spk_head`; `sortformer_hidden_export.patch`
    + `extract_hidden_features.py` (1e-6 posterior match tolerance); large float32
    features stored outside Git.
- G = GT causal frontier (Gate 0, deterministic decoder, no neural inference):
  `results/dev/gate0_oracle_{metrics.json,events.jsonl,topology_examples.jsonl}` +
  `gate0_verification.json` (+ `GATE0_ONTOLOGY_RESULT.md`). Decoder:
  `decoder.py: ReplacementDecoder(source_id, confirmation_samples)` +
  `_GTSimulator` (enrollment 200 ms, silence-reset 1200 ms, confirm grid
  100/200/300/500 ms — 500 ms is the grid max, matching the probe's persistence
  target). EVAL Gate-0 derivation unconditionally rejected in #97.
- Production VAD: bundled `src/puripuly_heart/data/vad/silero_vad.onnx`
  (Silero 6.2.1, sha `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`).
  Runner `psem_ontology_simplification_gate/run_production_vad.py` (CPU
  `CPUExecutionProvider`, peer profile: threshold 0.5, 512-sample chunks,
  pre_roll 500 ms, hangover 500 ms, max_segment 7000 ms;
  gate = pre-roll + committed chunks through speech end, excl. trailing hangover).
  Cached: `frozen_inputs/{dev,eval}_production_vad_{speech_gate.jsonl,replay_receipt.json}`
  and `psem_ontology_simplification_gate/results/dev|eval/production_vad_*`
  (per-source `speech_spans[start/end_sample], speech_seconds/span_count,
  scored_start/end_sample, audio_sha256`, plus sensitivity arms). Receipts bake in
  absolute local audio paths and external corpus roots (non-portable).

## 4. Reusable adapter/decoder code

- No `PSEMObservationAdapter` exists (grep: zero hits) — must be written.
  Closest streaming patterns:
  - `src/puripuly_heart/core/vad/silero.py: SileroVadOnnx` — `reset()`,
    `speech_probability(samples, sample_rate_hz)`, chunk/context-size helpers;
    `core/vad/bundled.py` (`bundled_silero_vad_onnx_path`, `ensure_...`);
    `core/vad/gating.py`. 16 kHz-mono PCM16 enforcement in
    `psem_relative_occupancy_gate/io_utils.py` (framerate/channels/width check).
  - #107 `nemo_adapter.py: _streaming_step` (FRAME_SAMPLES 1280 = 80 ms,
    state-reset lifecycle validation, NeMo rev `1a3c291b…`, container pin).
  - `psem_relative_occupancy_gate/model_decode.py: CausalAnchorTracker` —
    causal per-cell tracker (unique slot ids, anchor/other thresholds,
    candidate/enrollment, `state_reset` splits, 1200 ms silence reset,
    `evidence_frontier` propagation). Dataclasses `PosteriorCell`,
    `ModelObservation`, `CausalEnrollmentConfig/Event`, `CausalAnchorEpisode`,
    `CausalSessionResult`; builders `posterior_cells`, `oracle_anchor_mapping`,
    `relative_probabilities`, `model_observations`.
- 500 ms persistence decoder: `decoder.py: ReplacementDecoder`
  (confirmation_samples param; emit = boundary + confirm; GT-occupancy path) and
  `CausalAnchorTracker` (posterior path with `replacement_confirmation_samples`).
  Threshold/enrollment grid (95 combos, `other_low_threshold < active_threshold`)
  and duration-weighted PR/confusion/contamination metrics already implemented in
  `model_evaluate.py` / `evaluate.py` / `EVALUATOR.md` (incl. `boundary <=
  evidence frontier <= emit` invariant the verifier enforces).
- #107 causal head (`models.py: PSEMHead`, `GRU(208→64, 1 layer)` over 208-d
  `build_psem_features`, 1.04 s evidence delay, outputs
  `anchor_present/replacement_evidence`) is the head pattern but Sortformer-coupled;
  the probe needs a standalone small-model variant. Occupancy ontology reference:
  `ONTOLOGY.md` (`anchor_present/other_present` → NONE/ANCHOR_ONLY/ANCHOR_PLUS_OTHER/
  OTHER_ONLY; lifecycle UNANCHORED/ANCHORED/UNCERTAIN; cuts only while ANCHORED).

## 5. Dependencies: FireRedChat-pVAD / NeoVAD / ECAPA / ONNX / GRU; downloads

- Present in root `pyproject.toml` / `uv.lock`: `onnxruntime==1.28.0`,
  `sherpa-onnx>=1.13.4` (+ `sherpa-onnx-core==1.13.4`), `huggingface-hub==1.26.0`,
  `soxr==1.1.0`. `torch/torchaudio==2.7.1` only in
  `experiments/speaker_representation_scd/environment/pyproject.toml` (CPU index).
- ABSENT repo-wide (pyproject + uv.lock grep): FireRedChat-pVAD, NeoVAD,
  ECAPA/SpeechBrain, `nemo_toolkit` (root lock; #107 consumes pinned NeMo rev +
  container image via detached GPU runner, not a local install), any pVAD or
  speaker-embedding ONNX other than the Silero VAD blob. GRU exists only as
  `torch.nn.GRU` (used by #107 `PSEMHead`).
- Download policy: no explicit allowlist found in the read files; precedent is
  hash-pinned vendored artifacts (in-tree Silero ONNX, NeMo checkpoint sha,
  Silero model sha) with fail-closed preflight hash checks
  (`preflight.py` in #76/#97/#107 re-hashes manifests, binaries, checkouts and
  refuses on mismatch). Expect Gate-0 to require vendored + sha-pinned
  small-model binaries; fresh HuggingFace downloads at experiment time will
  likely be rejected at review. (Not verified: network egress rules on run hosts.)

## 6. Key risks / blockers for Gate-0 manifest freeze

1. Probe strata undefined — CAL12/MAIN48/EXT24, C1–C6, ONTOLOGY16/CONTROL24
   sampling rules, episode JSON schema, and freeze hashes do not exist; must be
   specified and frozen (copy the Gate-0 regenerate-and-compare verifier pattern).
2. Audio outside repo — manifests reference external `PSEM_CORPUS_ROOT`,
   `PSEM_REFERENCE_ROOT`, `SRSCD_CACHE_ROOT` (+ `PSEM_LSEEND_ROOT`); cached
   receipts bake in absolute machine-local paths; Gate-0 must re-resolve and
   re-hash waveform bytes per machine (`preflight` + `validate_corpus_waveforms`).
3. EVAL sensitivity — #97 `FINAL_DECISION.md` declares all V2 roles
   development-known for the next program (V3 must create a fresh DEV/EVAL
   holdout; V2 may only fold into V3 TRAIN). Probe EVAL reuse needs explicit
   justification or a fresh holdout.
4. Model binaries unpinned — pVAD + embedding ONNX ids/shas, input geometry
   (16 kHz mono, frame/chunk), and reset/bind/step semantics are unspecified;
   freeze them before any fitting (including enrollment/anchor-label rules).
5. Contract coupling — `psem-handoff-v1` constants (1200 ms continuity, 500 ms
   tolerances) and the 100–500 ms confirm grid are baked into Gate-0 verifiers;
   a 500 ms-persistence definition must stay consistent or explicitly version
   the decoder contract. Do not touch #107's `contract.json`/`config.json`.
6. Do-not-modify in flight — #107 files are independent; probe work must not
   alter them or the V2 freeze package (append-only new experiment namespace).
