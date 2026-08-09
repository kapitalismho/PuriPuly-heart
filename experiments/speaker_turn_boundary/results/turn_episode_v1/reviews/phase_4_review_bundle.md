# Phase 4 pre-execution review bundle — raw signal diagnostics

Status: **pending same-reviewer repair verification**. This bundle freezes the Phase 4
signal experiment before any new large neural inference or full diagnostic sweep.
Historical cache inspection and previously committed tiny parity evidence were read to
make the contract concrete; no Phase 4 scientific result has been generated.

Revision: 3.
Candidate: `working-tree` based on `85a8c702c5e18f06e2d1f8ef36ca063056877da1`.

## 1. Authority and accepted entry gate

| Item | Value |
| --- | --- |
| Normative plan | `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md` |
| Current plan Git blob | `cbcf1455651d144df808027183ec8e360752b432` |
| Current plan SHA-256 | `ecd16c765072504f9970cb17fd5cba4f9967715954adead6ebc386fffb8d4f8c` |
| Accepted Phase 3 scientific candidate | `a6403172451b02944e569a9bd94097387aa3adc0` |
| Accepted Phase 3 review-record commit | `85a8c702c5e18f06e2d1f8ef36ca063056877da1` |
| Phase 3 pre-execution bundle SHA-256 | `8dbcd4333297fa1dbc8b26a3ff4d9f0c708a0811588517b91184cabf20d17d36` |
| Integration target | `origin/main` at `848aa0b9f1b35388ded5a250d51a687223eac1c5` |
| Work branch | `experiment-v2-speaker-change-turn-boundaries-ls` |

Phase 3 is accepted with zero verifier mismatch over 82,026 oracle detail rows and
124,803 actions. Its provider-neutral result establishes that exact causal logical
actions can conserve PCM and reduce contamination. It does not establish that either
neural family supplies a useful speaker-specific signal. Phase 4 answers only that
signal question and determines how much Phase 5 compute each family is allowed.

### 1.1 Frozen input and source ledger

| Input | Byte SHA-256 | Bound meaning |
| --- | --- | --- |
| `episode_manifest_dev.json` | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` | content SHA-256 `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68`; 804 episodes |
| `state_equivalence_report.json` | `6e33711632d5f2e3de8e0c22c229b08827d1ccbb873deba2c1681a2ab2c544ec` | B0 source-prefix disposition and predeclared LS/ERes tolerances |
| `proposal_contract.json` | `0448edd933fd1d9d0a0b4d5f9f2631cb0f630c892fc4d46e1a3ec9740e80b7fb` | empty Phase 0 signal registry to be completed by this phase |
| `fusion_contract.json` | `bfda0c3c0ea7b6613ded79e9639692a33449dcf34202b1f2a5e7ec14c45f9873` | action and sentinel clustering contract; no Phase 5 sweep in this phase |
| `oracle_provider_neutral.json` | `be44a6a7764cff4c01064bc506c1d29ab6b4f35dbb48797409e68a610fea82db` | accepted logical-action ceiling |
| `oracle_provider_neutral_verification.json` | `83d7ad3f31777a907c5f1b810259e9e7994f8388f5dc2434c8e26f5477bd31e5` | independent Phase 3 recomputation |
| `frontend.py` | `ca879777c88b51fdf9d720427778e3fb6b2a7594b5b2e019ca5cd59baeb8d881` | LS resampler, source mapping, and feature frontend |
| `reducer.py` | `39c4db5fb5a98ca48bc3bbae982af0f9fa3dfe83c0db9978c29f91eb7ce5a759` | historical LS onset/replacement reducers |
| `phase3_ls.py` | `01ad46cdc3072dbbbe26253b2b0655c941a9cafede27f37db09caa972651ea0f` | historical streaming capture and cache reader |
| `phase3_eres.py` | `5a2bbfed1f427d7e3b631fd5b06747208c24cbf9edf36b3323e25505641236db` | historical exact-window embedding cache reader |
| `phase3_stages.py` | `03bdb0655df367368591c5441330d67a9f1c6df021047343975b39070f078c26` | historical model/frontend identities and replay helpers |
| `adapters/eres2netv2.py` | `7dd9c92d185faf4a0689d727853adc136c666953a883fc18c29ba39e09ec8629` | ERes frontend/export implementation |
| `provenance.py` | `151f864a346bb774d44dffa7c9fdcb25fc070ec62eb5525c918ae9a326ea6f00` | model repositories, revisions, licenses, checkpoint identities |
| `turn_episode/build_episodes.py` | `6deec51274cedf49a70cd299700547f39cbbbc16e200eb8e3056d15887784c7d` | accepted source/reference reconstruction |
| `turn_episode/contracts.py` | `b207d3f8b9720df5dd228aa8bd8b479c54622abb905a9ca04f580820a6fc3c03` | scientific invariants |
| `turn_episode/schemas.py` | `a9fa4571b1bab3cf88d6739a3732c1cc62f753a46d51d59c2db7526468eb8868` | source-coordinate proposal/action schemas |
| `turn_episode/phase4_design.py` | `529edc2887707c34876c2fa7e5fc77bacbadd18c9981509729d9417e0c4585a0` | metadata-only coordinate, candidate, pair, fixture, and forecast generator |
| `tests/test_phase4_design.py` | `04acfead6bb5e45b3aecda1599bce59a856bb93c614cf9a257eee556da557069` | 20 matcher, weight-dominance, and reference-aligned LS acoustic-support fixtures |
| `phase_4_design_ledger.json` | `a3d95083e1262a1d63839415703519c3f3078bfbbce61f39c85136c83f12af79` | content SHA-256 `149fd07404aee0248df97a14d8bf83e79842419937989862355d0c97104bca20`; 186,091 bytes |

The three historical synthetic manifests are frozen at SHA-256
`14347cdb...` (`ls_dev`), `c0aabc5a...` (`ls_held_out_clean`), and `f0d16939...`
(`ls_held_out_other`). The latter two names are historical labels: their content was
already touched by the predecessor experiment and Phase 2 explicitly placed it in
`diagnostic_dev`. They are not the inaccessible `confirmatory_heldout` pool.

## 2. Scope and explicit non-goals

Phase 4 executes raw, threshold-light diagnostics for all verified LS-EEND and
ERes2NetV2 checkpoints on the already accepted `diagnostic_dev` population. It freezes
signal extractors, causal timing, state disposition, matched controls, AUC/EER,
session-block uncertainty, causal neural oracle summaries, runtime evidence, and the
family-level `signal_go`, `signal_limited`, or `signal_stop` result.

Explicit non-goals:

- no `frontier_dev` threshold or policy selection;
- no Phase 5 clustering/refractory/VAD-fusion grid, corrected policy comparison, or
  per-policy product-performance claim;
- no Phase 6 frontier construction or freeze;
- no confirmatory held-out path resolution, listing, annotation read, audio read, or
  aggregate inspection;
- no provider trace, provider credential, network request, paid/live call, or provider
  conclusion;
- no production owner, composition, settings, provider adapter, lifecycle, or public
  entrypoint change;
- no product recommendation, merge, push, deployment, or worktree cleanup.

Every new implementation and result stays below
`experiments/speaker_turn_boundary`. The source trees described by `ARCHITECTURE.md`
remain read-only behavioral context. There is no intended production architecture
change.

## 3. Frozen diagnostic population

Only `episode_manifest_dev.json` rows with `pool == diagnostic_dev` are opened by the
Phase 4 runner. The `frontier_dev` rows remain unexecuted during this phase. The
population is verified before model construction and fails closed on any count, hash,
status, or group-graph drift.

| Population fact | Expected value |
| --- | ---: |
| diagnostic episodes | 695 |
| scorable episodes | 694 |
| diagnostic-only episodes | 1 |
| distinct source-session identifiers | 616 |
| primary clean/gap hard references | 450 |
| same-speaker neutral-pause references | 132 |
| overlap soft references | 313 |
| synthetic episodes | 606 |
| already-opened public episodes | 89 |
| public source sessions | 10 |
| public source-prefix duration | 18,813.025 s |

The synthetic strata are 202 episodes each from `ls_dev`, historical
`ls_held_out_clean`, and historical `ls_held_out_other`. Each contributes 101 primary
hard targets, 21 neutral pauses, and 63 overlap markers. The public diagnostic rows
contain 52 AMI episodes from six sessions and 37 AliMeeting episodes from four sessions.
Their group components come only from the accepted group graph
`7ebf4dffa0af180910007a318d0e3d1e77f7f048dbae852199ddd45f74cce7eb`.

The one `diagnostic_only` row is retained in state/parity and missing-observation
accounting but cannot enter AUC, EER, or a go/limited/stop estimate. Overlap-only targets
are reported as soft diagnostics and cannot satisfy the hard-target signal gate.

## 4. Checkpoints, frontend provenance, and cache identities

### 4.1 Model files

| Family/checkpoint | Model SHA-256 | Sidecar SHA-256 |
| --- | --- | --- |
| LS `L-AMI` | `5a2b813ffe41170e40d0fc08a6eb1699e579e377af30c7962d07885608a6aa77` | `47f29718254995ec017636d5ff31fef8b20bf47dca30d883edcb91e022dc3353` |
| LS `L-CALLHOME` | `b79b1b1cb2a070bfb92543d90af5530681af0e45da8bf5771e515e9c644b6604` | `049084141fb3d7694e4bbdc257024761d6f6e64b8782e47d7426e3f35009dffa` |
| LS `L-DIHARD-II` | `5df89a22ba87989a01217e51d674cc547877ce5b7100dce920ab63adc3258302` | `ecb42ca07888a297b3e0ae277b19e1a1412e1fa68309233f9e6196b06128f0a9` |
| LS `L-DIHARD-III` | `587ad263b46aaa5d4fc7fb9e0524d1990455f7286c3a47b2371d08df8b5671c8` | `ecb42ca07888a297b3e0ae277b19e1a1412e1fa68309233f9e6196b06128f0a9` |
| ERes `E-standard` | `7a6d4f89dcb92a554806bdf6bfb13c7fae0a63e8f992a49b3a503b9a03c705cf` | none; source checkpoint provenance remains pinned in `provenance.py` |
| ERes `E-w24s4ep4` | `3761572a872a29f36af66065075cc9a48adc23c8b26fb0c68488aa3ed8f35f26` | none; source checkpoint provenance remains pinned in `provenance.py` |

The repeated LS DIHARD sidecar is intentional and must verify to the same byte hash.
Model construction is forbidden until all paths exist and every model/sidecar byte hash
matches. ONNX Runtime is pinned in the execution receipt with version, provider,
optimization level, thread counts, CPU, RAM, Python, and operating system.

### 4.2 Historical cache treatment

The inspected legacy v2 cache contains 3,400 files and 183,948,105 bytes. A sorted
ledger over `family|relative-path|size|byte-sha256`, joined by LF without a trailing LF,
has SHA-256 `879ad9cdfe4d45947b819a198c11ce91d087b94631ad31dc3eda91abe0cdf354`.
Its historical temporary parent happens to contain the word `opencode`; this is only an
existing directory name. Phase 4 launches no OpenCode process, terminal, or worker.

The historical cache is read-only and cannot be a direct Phase 4 cache hit because its
manifest identities predate `episode_manifest_dev:deb1713cd...`. Eligible payloads are
legacy imports only. A new neutral cache root
`%TEMP%/puripuly_stb_phase4/turn_episode_v1` binds the full PRD Section 27.2 identity:

- authority SHA and accepted Phase 4 bundle SHA;
- checkpoint and every sidecar hash;
- frontend/resampler contract hash;
- source WAV byte hash;
- `episode_manifest_dev` content hash;
- model input/output tensor names, dtypes, ranks, static dimensions, and their canonical
  contract hash;
- capture payload byte hash and canonical content hash;
- state mode and source-origin mapping;
- for ERes, every absolute `[start,end)` source window and every embedding payload hash.

An imported LS capture is accepted only after shapes, frame centers, availability
frontiers, source length, and WAV identity match and 32 deterministic case captures per
checkpoint are recomputed. Posterior maximum absolute error must be at most `1e-6` and
frame/frontier coordinates must be exact. An imported ERes cache is accepted only after
32 to 256 deterministic windows per checkpoint, selected by the lowest SHA-256 ranks,
are recomputed; maximum absolute embedding error must be at most `1e-5` and cosine at
least `0.99999`. Any sampled failure rejects all legacy imports for that checkpoint and
forces fresh inference after recording the rejection. No tolerance is widened after a
comparison.

Cache files are written to same-directory temporary files and atomically replaced only
after payload count, size, byte hash, and content hash pass. A partial session remains
explicitly partial and cannot enter a complete summary.

## 5. LS source-time, frontend, and causal contract

The authoritative path consumes canonical 16 kHz float PCM in 512-sample chunks. The
resampler is the committed 63-tap Hamming-windowed halfband FIR with center 31, float64
accumulation, float32 output, decimation by two, no artificial right padding, and an
empty flush. Output 8 kHz sample `m` maps to source sample `2m + 31`.

The LS frontend is `logmel23_cummn`: 8 kHz, 200-sample window, 256 FFT, 80-sample hop,
23 mel bins, cumulative mean normalization, context 7, subsampling 10, and convolution
delay 9. For output frame `f`:

```text
center_16k(f)       = 1600*f + 14431
available_count(f)  = 1600*f + 15806
lookback            = 1375 samples = 85.9375 ms
```

`available_count` is the exclusive observed-source frontier required to emit the frame.
The raw posterior extractor may not use a frame until that count is observed. Reducer
confirmation, median lookahead, and any sentinel-cluster lookback are added to the
proposal availability and safe frontier; they are never charged as model compute or
subtracted from measured delay.

Ordinary frames are those with `available_count(f) <= epoch_end_count`; their frontier
is exactly the formula above. Frontend finalization may expose a terminal frame whose
real center is before `epoch_end_count` while its ordinary required count is later. Such
a frame records `observation_frontier = epoch_end_count`, `tail_dependent = true`, and
the ordinary required count. It is retained only in the terminal-tail diagnostic and is
excluded from matched AUC, EER, causal-oracle selection, and every 250/500/1000 ms
horizon. A frame with `center_16k(f) >= epoch_end_count` is dropped. Finalization may
not append new source-audio samples to the timeline or resampler. The committed frontend
may nevertheless use left/right zero padding solely inside STFT analysis framing, zero
feature context outside real base-feature indices, and decode-only zero model-input
frames with `ingest=0, decode=1`. These mechanics do not advance the source-audio count
or ingest new neural feature state. They exist only to complete terminal analysis and
decode state. These rules make the historical `epoch_end_count` frontier explicit
without allowing terminal flush to improve the primary signal result.

The fixed parity set is the six hash-pinned golden/research clips plus deterministic
chunk-edge lengths around 31, 63, 128, 512, 1,600, and terminal partial chunks. Whole
file and streaming must have the same frame count and source coordinates. Resampler
output is exact, feature maximum absolute error is at most `2e-5`, and ONNX posterior
maximum absolute error is at most `1e-5` on the tiny parity set. The streaming path is
authoritative if harmless floating-order differences remain within tolerance. Tail
frames preserve a real source center before epoch end; no future source audio is
invented. The frozen terminal feature-context padding and decode-only flush above are
analysis mechanics and cannot enter AUC, EER, oracle, or horizon selection.

The six-clip ledger SHA-256 is
`e9805b74a0395e5e520d2bfe60c2cdb479981d8e89104b1c39d288edf313e1fb`.
Its three committed clips are `golden_silence.wav` `20eaebff...` (64,044 bytes),
`golden_single_utterance.wav` `325ff307...` (96,044 bytes), and
`golden_two_utterance_gap400.wav` `4dfd5a1a...` (160,044 bytes). Its three
research clips are `speaker1_a_cn_16k.wav` `5f20ce0d...` (118,932 bytes),
`speaker1_b_cn_16k.wav` `20745dc0...` (157,058 bytes), and
`speaker2_a_cn_16k.wav` `8a6cffa4...` (170,028 bytes). Full hashes are in the
design ledger. The bound prior receipts are `parity_frontend.json` byte SHA-256
`537a4cc961387cfd3a72e9fa88d33b2ea695ea6f0b234b49857bc1911db17a1e`
and `parity_research.json` byte SHA-256
`e286446fd06e3a2557a8fcd893022052bfa2559c63215874279466433111c1f3`.

LS reports posterior trajectories, new-track rise, dominant replacement, hysteretic
active-set changes, track flicker per active-speech minute, overlap-state soft
precision/recall, causal-oracle availability, reset versus continuous state, and all
component delays separately.

## 6. ERes frontend, exact-window, and export contract

ERes consumes canonical 16 kHz PCM, so its resampler identity is an exact no-op with
`source_sample == model_sample`. Any non-16-kHz input is rejected rather than silently
resampled. The frontend is 80-bin Kaldi-compatible fbank with 25 ms frames, 10 ms shift,
Povey window, pre-emphasis 0.97, HTK mel scale, and time-mean normalization. The ONNX
embedding dimension is 192.

Every embedding key is an absolute half-open 16 kHz source window `[start,end)`. Windows
must lie inside real source audio, must not cross an audio epoch, and are not padded.
The observation frontier is exactly `end`. Adjacent windows are
`[b-W,b)` and `[b,b+W)`; stable-anchor and prototype windows record their own exact
coordinates and the causal update sequence. Window lengths are 0.50, 0.75, 1.00, 1.50,
and 2.00 seconds for adjacent diagnostics; stable-anchor diagnostics use 0.50, 0.75,
1.00, and 1.50 seconds. Steps are 100 and 250 ms, with 500 ms additionally reported for
the longer adjacent views. Only a 0.50-second right window can naturally supply an
observation by the primary 500 ms horizon; longer-window missingness is expected and
must not search later audio.

The source-time grid origin is absolute sample zero. For an episode with bounds
`[warm_start, scored_start, scored_end, tail_end)`, adjacent grid boundary `b` is every
multiple of the declared step satisfying `b in [scored_start,scored_end]`,
`b-W >= warm_start`, and `b+W <= tail_end`. A causal state probe is trailing window
`[e-W,e)` for every absolute step-grid end `e` satisfying `e-W >= warm_start` and
`e <= tail_end`; its proposal boundary is `e-W` and frontier is `e`. The generator
neither consults B0 VAD nor GT activity to accept a grid window. Windows crossing VAD,
silence, or a reference transition are therefore eligible and are tagged after audio
and annotation reads as `pure_singleton`, `transition_mixed`, `silence`, or `other`.
Those tags stratify reports but cannot change the coordinate universe.

Matched raw-signal measurement adds an explicitly GT-indexed, diagnostic-only boundary
at each positive or negative candidate coordinate `c`. It evaluates exact adjacent
windows `[c-W,c)` and `[c,c+W)` and a read-only anchor/prototype probe `[c,c+W)` when
they fit real episode bounds. The state snapshot contains only regular probes whose end
is at or before `c`; the read-only measurement never updates state and never becomes a
Phase 5 proposal coordinate. This is the only reference-aligned coordinate addition and
prevents the 500 ms view from depending on accidental alignment to a 100 ms grid.

All embeddings are L2-normalized with denominator `max(norm,1e-12)`. A zero-norm
embedding is invalid and produces `missing=true`. For every window/step profile, state
is reset at the accepted episode/source-prefix boundary and processed by increasing
`(frontier,boundary,window_start)` order:

- `stable_no_update`: the first valid trailing probe initializes the anchor and the
  anchor never changes;
- `stable_ema`: the first valid probe initializes the anchor. After scoring a probe,
  cosine at least `0.70` updates `anchor = normalize(0.10*anchor + 0.90*probe)`;
  lower cosine leaves the anchor unchanged;
- `confirmed_anchor`: cosine below `0.50` opens/replaces a pending candidate. The next
  valid probe confirms the first candidate only when it is also below `0.50` against
  the unchanged anchor and mutual cosine with the first probe is at least `0.50`.
  Confirmation emits the first boundary at the second frontier, replaces the anchor
  with the first normalized probe, and clears pending. A mutual failure replaces
  pending with the current probe; a noncandidate clears pending and applies the same
  `0.70`/EMA stable update. Episode end records but does not emit an unconfirmed tail;
- `prototype_memory_4`: the first valid probe creates prototype ordinal zero. Each
  probe selects maximum cosine, ties by smallest creation ordinal. Cosine at least
  `0.70` updates that prototype and its acoustic shadow by the same 0.10/0.90 normalized
  EMA without changing its ordinal. Cosine in `[0.50,0.70)` clears pending and does not
  update. Cosine below `0.50` uses the same two-probe mutual confirmation. A confirmed
  first probe creates a new prototype; at capacity four it replaces the smallest
  creation ordinal and receives the next monotonically increasing ordinal. A mutual
  failure replaces pending with the current probe. No unconfirmed tail is created.

Every transition records pre-state IDs, selected anchor/prototype ordinal, both cosine
values when applicable, the decision, post-state payload hash, and exact source windows.
No profile retains state across episodes.

The six hash-pinned parity clips are rerun after approval, followed by the deterministic
sampled-window import check. Frontend maximum absolute error against the pinned research
export is at most `1e-3`; embedding mean absolute error is at most `1e-3`; embedding
cosine is at least `0.99999`. Exact-window recomputation with the same ONNX export uses
the stricter cache-import limits in Section 4.2.

The ERes artifacts are exactly `eres2netv2.onnx` byte SHA-256 `7a6d4f89...`
(71,441,209 bytes) and `eres2netv2_w24s4ep4.onnx` byte SHA-256 `3761572a...`
(214,043,983 bytes). Both bind float32 `fbank [1,time,80]` to float32
`embedding [1,192]`, dynamic time axis only, opset 17. Their sources are the exact
ModelScope checkpoints and revisions in Section 4.1 and the official 3D-Speaker model
classes at revision `065629c313eaf1a01c65c640c46d77e61e9607b4`. The frozen export
recipe is `model.eval()`, float32 dummy `[1,345,80]`, constant folding enabled,
input/output names `fbank`/`embedding`, opset 17, and dynamic input time axis. Export is
not repeated in Phase 4. If either artifact is absent or drifts, execution stops instead
of creating a new export. `research_parity.py` is bound at `a049e3a4...`,
`run_parity.py` at `4a43a449...`, `run_eres_sweep.py` at `1b56428a...`, and the adapter
at `7dd9c92d...`; full hashes are in Sections 1.1 and the design ledger.

ERes reports same/different cosine distributions, AUC/EER by corpus, language, window,
and stress, pure versus transition-mixed windows, anchor drift, consecutive-candidate
mutual similarity, gain/noise/codec/prosody response, causal-oracle availability, and
sampled frontend/export parity.

## 7. State-equivalence disposition

Before a reset-based result is accepted, Phase 4 reruns the accepted Section 5.4 modes
over every public `diagnostic_dev` episode and deterministic synthetic sentinels:

1. `source_prefix`: replay original source sample 0 through the target region;
2. `episode_reset`: reset at the manifest warm-start and replay only declared warm-up.

LS comparison is per checkpoint and reducer class: raw posterior,
`new_track_onset`, `dominant_replacement`, and the new hysteretic activity-state
diagnostic. ERes comparison is per checkpoint and profile class: adjacent, stable
anchor without update, stable anchor with causal EMA, confirmed stable anchor, and
bounded episode-local prototype memory. Prototype state is always cleared at episode
start and never carries speaker identity across episodes.

At identical absolute source coordinates, LS posterior maximum L1 must be at most
`1e-2`; ERes aligned-window cosine must be at least `0.99`. Proposal count/kind,
boundary coordinate, observation frontier, one frozen no-search sentinel cluster, and
safe-frontier progression must match exactly. A coordinate grid with no common aligned
frames is a failure, not an implicit interpolation pass. The sentinel uses debounce 0,
radius 250 ms, and refractory 0 solely to exercise state propagation; it is not a Phase
5 policy result.

A class passes only if every fixed parity case passes. Otherwise its Phase 4 scored
result and all later uses require deterministic source-prefix replay or a round-trip
snapshot whose resumed raw/proposal/progress trace matches source-prefix exactly. A
failed case remains visible. Warm-up is not enlarged after seeing failure. Synthetic
complete episodes beginning at sample zero may pass trivially, but that does not permit
reset evaluation for the public class.

## 8. Frozen signal extractor registry

The Phase 4 implementation replaces the empty `proposal_contract.json` registry with
the declarations below and a new content hash. Each concrete extractor ID also binds
checkpoint, window/profile parameters, and evaluation horizon (`250`, `500`, or
`1000` ms). All hard-target scores have sign `higher_means_change`. Missing valid
observations produce the finite score `0.0` plus `missing=true`; they never trigger a
later search. The numeric placeholder is never ranked: if either member of a matched
pair is missing for a concrete extractor/horizon, that pair is excluded from that AUC,
EER, and paired acoustic delta and is counted by class and reason.

### 8.1 LS hard-target scalars

For posterior vector `p_t` and the mean `q_t` over the three preceding available frames:

- `ls_new_track_rise.v1`: `max_j max(0, p_t[j] - max(p_{t-3:t}[j]))`; invalid before
  three prior frames;
- `ls_dominant_replacement.v1`: choose prior dominant `a = argmax(q_t)` and best other
  track `b = argmax_{j != a}(p_t[j])`, ties by lowest track index; score
  `max(0,p_t[b]-p_t[a]) * max(0,q_t[a]-q_t[b])`; invalid with fewer than two tracks;
- `ls_activity_set_change.v1`: use frozen low/high hysteresis 0.40/0.60 and score the
  maximum absolute posterior change among tracks whose hysteretic active state changes
  at `t`; zero when no active-state change occurs.

`ls_overlap_strength.v1` is the product of the two largest posterior values and is
secondary/overlap-only. It cannot satisfy the hard-target gate. Track-instability and
flicker are likewise diagnostic-only.

### 8.2 ERes hard-target scalars

- `eres_adjacent_change.v1`: `1 - cosine(left_embedding,right_embedding)` for each
  declared adjacent window;
- `eres_stable_anchor_change.v1`: `1 - cosine(causal_stable_anchor,probe)` with anchor
  coordinates, update decision, and all prior updates recorded;
- `eres_confirmed_anchor_change.v1`: the first candidate's anchor-change score, valid
  only after the next candidate is available and mutual cosine meets the declared 0.50
  confirmation floor;
- `eres_prototype_change.v1`: `1 - max cosine(probe, prototypes)` for a maximum of four
  episode-local normalized prototypes, FIFO tie-breaking, causal updates only, and full
  reset at episode start.

All ERes scores use the PRD-required monotonic `1 - cosine` transform. Stable-anchor or
prototype observations without a valid causal anchor/prototype are missing, not zero
similarity. Pairwise ERes scores do not claim overlap detection.

### 8.3 Acoustic-only controls

The identical source support used by each neural score also produces:

- `acoustic_log_rms_delta.v1`: absolute difference of `log(max(RMS,1e-8))`;
- `acoustic_logmel_flux.v1`: `1 - cosine` between L2-normalized mean 80-bin logmel
  vectors using the pinned ERes frontend without the embedding model.

For adjacent ERes, both acoustic scores use exactly `[b-W,b)` and `[b,b+W)`. For a
stable anchor, confirmed anchor, or prototype, the acoustic path maintains a shadow
state over the identical probe coordinates, initialization, selection, confirmation,
EMA, creation, and replacement decisions. The shadow payload is the L2-normalized mean
80-bin logmel vector; RMS is carried as log RMS and follows the same weighted update.
Thus an averaged neural anchor/prototype is never compared with an unrelated single
audio window. Each row binds neural state hash, acoustic-shadow state hash, and the full
contributing coordinate list.

For each matched candidate coordinate `e` and horizon `H` in `{250,500,1000}` ms, the LS
acoustic support is reference-aligned and diagnostic-only. Its per-side length is
`L = 16*H` source samples, respectively 4,000, 8,000, and 16,000 samples, and the
comparison is `[e-L,e)` versus `[e,e+L)` with frontier `e+L`. The candidate coordinate
is never used as a proposal or product action; it only aligns the already frozen signal
comparison. The same candidate measurement is reused for every LS scalar/checkpoint row
at that horizon. Neural LS selection remains causal and uses the strongest ordinary
frame with center at or after `e` and availability no later than `e+L`; terminal-tail
frames remain ineligible. A matched pair is valid only when both positive and negative
supports are real and each side has at least one eligible ordinary neural frame. The
metadata ledger freezes 295, 313, and 247 paired-valid comparisons at 250, 500, and
1000 ms; preflight aborts if any count becomes zero. Horizon-sized support is declared
before scoring and is not post-hoc tuning.

The acoustic comparator for a family/extractor is the declared acoustic score with the
highest full-sample AUC on the identical matched examples, ties by extractor ID. That
choice is frozen before bootstrap. Each bootstrap replicate compares the neural score
with that same selected acoustic score; it does not reselect a convenient control.

## 9. Examples, matching, AUC, EER, and uncertainty

For horizon `H`, a positive score is the strongest valid scalar with boundary coordinate
at or after the hard target's evidence onset and observation frontier no later than
`evidence_onset + H`. A negative score uses the identically sized causal interval after
its matched same-speaker pseudo-boundary. Sensitivity horizons 250 and 1000 ms are
reported, but only 500 ms controls the gate.

The negative universe is frozen before audio scoring. Every scorable `neutral_pause`
reference contributes its evidence-onset coordinate. A synthetic `same_speaker` or
`gain_variation` case additionally contributes its manifest `splice.b_onset_sample`
when that coordinate is not already represented by a neutral-pause row. Silence and
noise-only cases remain descriptive and never enter matched AUC. Finally, at most one
stable-singleton coordinate is added per episode: enumerate the absolute 100 ms grid
inside an unambiguous exactly-one-speaker region, require 0.50 seconds of that same
region and scored exposure on both sides, require at least 1.00 second from every hard,
soft, neutral, structural, acceptable-interval endpoint, and evidence coordinate, then
choose greatest minimum distance and earliest coordinate on a tie. No fallback relaxes
these guards and no negative is reused.

Synthetic eligibility requires equal synthetic manifest, which is also its accepted
seed-family uncertainty block. Public eligibility requires equal corpus, language, and
accepted recurring-participant group component. Within each eligible group, sorted
positive and negative IDs form one complete matrix and the repository's
`exact_integer_hungarian.v1` returns the global minimum assignment of cardinality
`min(P,N)`. Scalar edge cost is `S*stress_mismatch + D*duration_distance_ms +
G*gap_distance_ms + T`, where `T` is the full 256-bit SHA-256 integer over
`positive_id|negative_id`. `G`, `D`, and `S` are generated arbitrary-precision integers
whose bounds are 450 pairs and `1e9` ms per feature distance; each weight is strictly
greater than the greatest possible aggregate of every lower-order term. This therefore
implements the declared lexicographic total exactly without floating-point tie loss.
Matrix rows and columns are lexical IDs and equal reduced costs prefer the lower column.
The design ledger persists the exact decimal weights. The implementation was checked
against exhaustive assignments for every random 1x1 through 4x4 rectangular fixture.
The pair ledger stores every assignment and all unused/unmatched counts; primary AUC
uses only these pairs.

The metadata-only rev-3 ledger contains 450 positives and 360 negatives: 132 neutral
pauses, 18 additional same-speaker transitions, 9 additional gain transitions, and 201
stable-singleton candidates. It globally matches 313 pairs across 13 blocks, leaving
137 positives unmatched and 47 negatives unused; no eligible group lacks a negative.
Pair rows hash to `fb29fff960932f2840433fa94f1a9e4bade167a6d935a6458dc6e9b191a4f9b9`.
Unmatched rows remain in distribution and missingness reports but not primary matched
AUC. The independent verifier re-enumerates candidates and solves the same matrices
from inputs rather than trusting persisted assignments.

ROC-AUC uses the tie-aware Mann-Whitney definition. EER uses deterministic linear
interpolation between adjacent ROC points, with thresholds ordered from high to low and
stable ties by example ID. Reports stratify by checkpoint, extractor, corpus, language,
window, stress, hard-target type, and public versus synthetic evidence. Strata lacking
both classes report `not_estimable` rather than a fabricated number.

Uncertainty uses 10,000 deterministic whole-block bootstrap replicates. A public source
session uses its accepted recurring-participant group component. Synthetic derivatives
share a block when their manifest source speakers, utterances, or transformation seed
family connect. A resampled block contributes all of its matched pairs. The bootstrap
seed is the first 64 bits of SHA-256 over the Phase 4 contract content hash, extractor
ID, and stratum ID. Percentile bounds use deterministic nearest ranks at 2.5% and 97.5%.
Point estimates, block count, pair count, missing counts by class, and every replicate
input block ID are auditable.

## 10. Signal disposition and compute gate

For every concrete hard-target neural extractor, calculate
`delta_auc = neural_auc - selected_acoustic_auc` on identical examples and bootstrap
the paired difference by block. Each extractor first receives exactly one status:

- `not_estimable`: it lacks either positive or negative valid scores;
- `low_block`: it is otherwise estimable but contributes fewer than eight independent
  blocks; its interval is descriptive and cannot create either `go` or `stop`;
- `eligible_go`: at least eight blocks and 95% lower bound strictly above zero;
- `eligible_stop`: at least eight blocks and 95% upper bound at or below zero;
- `eligible_uncertain`: at least eight blocks and neither eligible condition above.

Family precedence is then mechanical. If no extractor is estimable, the family is
`not_estimable`. Otherwise, discard `low_block` and `not_estimable` rows from directional
voting. If no eligible rows remain, the family is `signal_limited`. If any eligible row
is `eligible_go`, the family is `signal_go`. Otherwise, if every eligible row is
`eligible_stop`, it is `signal_stop`. Every remaining mixture is `signal_limited`.
Thus a low-block positive can neither cause nor veto `go`/`stop`; a mixture of
estimable and non-estimable rows is decided only by eligible rows; and `not_estimable`
is a family outcome only when no extractor supplies both classes.

No multiplicity-adjusted product claim is made at this diagnostic gate. All checkpoints
and extractor results remain visible. In Phase 5, `signal_go` gets the full predeclared
policy grid, `signal_limited` gets the same-proposal ladder plus one sentinel per policy
family, and `signal_stop` gets only B0/B1 and the no-neural control. The runner emits the
allowed Phase 5 compute envelope mechanically from the disposition; the coordinator
cannot expand it after seeing policy results without a new reviewed amendment.

## 11. Causal neural oracle and reducer diagnostics

For each hard target and extractor, the causal signal oracle may choose the strongest
already available observation inside the frozen horizon, but it cannot move the logical
boundary earlier than the observation's own declared boundary coordinate or read later
audio. It reports target coverage, selected boundary error, availability delay,
missingness, and the Phase 3 assembler's contamination ceiling when the selected action
is replayed with the accepted lifecycle semantics. Ground-truth selection makes this an
upper bound only; oracle rows never enter policy selection.

LS continuous-within-source and reset-at-VAD views share the same cached raw posterior
frames. ERes pure-window and transition-mixed views share exact window coordinates.
This ensures differences are reducer/state diagnostics rather than new inference.

## 12. Outputs, completeness, provenance, and size policy

Phase 4 proposes these experiment artifacts under `results/turn_episode_v1/`:

```text
proposal_contract.json
phase_4_design_ledger.json
phase_4_cache_inventory.json
phase_4_frontend_parity.json
phase_4_state_equivalence.json
phase_4_ls_signal_report.json
phase_4_eres_signal_report.json
phase_4_acoustic_controls.json
phase_4_signal_disposition.json
phase_4_signal_details/<family>/<checkpoint>-<shard>.jsonl.gz
phase_4_verification.json
reviews/phase_4_pre_execution.md
```

Every JSON has a canonical content hash; every file has a direct byte SHA-256 in its
parent ledger. Detail shards are deterministic gzip with `mtime=0`, at most 20 MiB each,
and atomically written. Aggregate JSON files must remain below 10 MiB. A single giant
JSON report is forbidden. A shard index binds first/last row key, row count, uncompressed
content hash, compressed byte hash, and size. The independent verifier streams shards
and never requires materializing a 100+ MiB object in memory.

Completeness joins the exact 695-episode population, all six checkpoints, every declared
extractor/horizon, every expected exact window, all parity classes, all pair-ledger rows,
and 10,000 bootstrap replicates for every estimable primary comparison. It rejects
duplicate or missing primary keys, stale code/model/cache hashes, unexpected held-out
labels, nonfinite values, causal-frontier violations, and summaries whose aggregates do
not recompute from detail rows.

The independent verifier must reject at least these mutations through its public entry
point: posterior score change, ERes window-coordinate change, observation frontier moved
earlier, pair/block reassignment, cache payload hash change, AUC summary change, and
family disposition change.

## 13. Runtime, data access, and failure boundaries

Execution uses the declared Windows CPU environment and ONNX Runtime
`CPUExecutionProvider`, one inter-op and one intra-op thread unless the approved receipt
records a reviewed correction. The 10 public source-prefix sessions contain 18,813.025
seconds (5.226 hours) of audio; with synthetic episodes, four LS checkpoints expose
22.595 audio-hours before cache reuse. Historical ERes evidence
measured approximately 30-37 ms per missing window on this machine. The preflight must
forecast expected window count, wall time, peak RSS, and cache bytes from the finalized
coordinate ledger before the full run.

The rev-3 generator deterministically emits 258,543 coordinate declarations with row
SHA-256 `727d96a92fd06c8c020c63275891128c52fbfb73c459067e3d7584dfc8a007b7`,
151,214 deduplicated embedding windows with row SHA-256
`42b316ca0c4e1c89e6aa746db0584d38b27bf75e9b4ed39735edb3f8024b266c`,
and 4,371 deduplicated reference-aligned LS acoustic windows with row SHA-256
`3fe0ffef5a2dc79ec385f89924181ff2e98c6f61f812ca39007eccb6083b148a`.
Two ERes checkpoints therefore expose 302,428 embedding jobs. Four LS checkpoints expose
81,343.7 audio-seconds: the 18,813.025-second public source-prefix set plus 1,522.9
synthetic seconds per checkpoint. Forecast inputs are conservative historical LS RTF
`0.05`, ERes service time `0.037` seconds/window, 4,096 LS cache bytes/audio-second,
2,048 ERes bytes/embedding, and 900 seconds fixed verification overhead. The resulting
forecast is 16,157.021 seconds (4.488061 hours), 957,032,243 new cache bytes, and 6 GiB
peak RSS. All are below the frozen ceilings. Preflight recomputes these values and stops
on any count or bound drift; a favorable cache hit may reduce actual work but cannot
expand coordinates or the ceiling.

Conservative authorization ceiling after approval: six wall-clock hours, 8 GiB new
cache, 16 GiB peak RSS, no GPU, no network, no credential, and zero provider cost. The
runner stops before inference if the forecast exceeds a ceiling. During execution it
records model load/warm-up, mean/p50/p95 service time, RTF, peak RSS, cache size, and
one-stream backlog. Two-stream stress is secondary and may run only from existing raw
caches after the scientific summaries are complete.

The Goal executor may make one approved full execution attempt. A failed full run is
not automatically restarted. Failure, partial outputs, and the exact safe retry boundary
are recorded in STATE before another coordinator-authorized attempt.

Phase-stopping conditions include:

- any authority, accepted Phase 3, input, model, sidecar, cache, or population hash drift;
- any confirmatory-held-out access attempt;
- parity outside frozen tolerance or incomplete state disposition;
- artificial audio padding, future-read, or observation frontier before required input;
- unmatched example silently entering primary AUC;
- fewer than 10,000 valid bootstrap replicates for an estimable comparison;
- self-hash, shard, completeness, recomputation, or mutation-verifier failure;
- summary/detail disagreement or a report exceeding the size policy;
- compute/memory/cache forecast above the authorization ceiling.

## 14. Execution order after approval

1. Persist the approved review artifact and exact accepted bundle byte hash.
2. Implement the registry, preflight, runner, cache writer, verifier, and focused tests
   exactly under the experiment tree.
3. Run formatting, focused pure fixtures, tiny frontend/export parity, and cache-import
   sampling; stop on any mismatch.
4. Materialize the exact window/example/pair/block ledger and runtime forecast without
   opening `frontier_dev` or confirmatory content; stop if counts or ceilings drift.
5. Execute one full source-prefix/raw-cache pass and deterministic diagnostics.
6. Run the independent verifier, mutation tests, focused tests, and the full experiment
   suite.
7. Commit one coherent Phase 4 scientific candidate and obtain a fresh independent
   Phase 4 exit review.
8. Only an accepted Phase 4 exit may authorize preparation of the Phase 5 review bundle.

## 15. Reviewer checklist and verdict space

The reviewer must verify at minimum:

1. authority and accepted Phase 3 entry identities;
2. `diagnostic_dev` population and confirmatory isolation;
3. all checkpoint/sidecar/frontend and reusable-cache identities;
4. LS source mapping, buffering, tail, and availability formulas;
5. ERes no-op resampling, frontend/export parity, and exact windows;
6. state-equivalence classes, fixed tolerances, and fail-to-source-prefix rule;
7. every extractor's formula, sign, horizon, valid-window rule, and missing rule;
8. matched controls, acoustic comparator, AUC/EER, blocks, bootstrap, and disposition;
9. causal oracle boundaries and separation from policy evidence;
10. expected outputs, completeness, independent recomputation, mutation rejection,
    runtime ceiling, and size policy;
11. experiment-only architecture boundary and lack of production/provider access.

Findings are classified `blocking`, `major`, or `minor`. The PRD review artifact uses
`approved`, `approved_with_required_changes`, or `rejected`; execution is authorized
only after a final `approved` verdict. In Goal control, the corresponding reviewer result
is recorded as `accepted`, `repair_required`, `not_reviewable`, or
`needs_user_decision`.

| Finding | Severity | Resolution |
| --- | --- | --- |
| rev-1 candidate commit pin was invalid | blocking | corrected to actual `85a8c702c5e18f06e2d1f8ef36ca063056877da1` |
| ERes grid and state machines were under-specified | blocking | exact absolute grids, measurement anchors, four state machines, thresholds, updates, and tail rules frozen in rev 2 |
| negative universe, matching, and acoustic mappings were under-specified | blocking | guarded candidates, exact-integer global matching, non-reuse, pair ledger, and extractor-specific acoustic shadows frozen in rev 2 |
| low-block disposition overlapped `go`/`stop` | blocking | exclusive extractor statuses and mechanical family precedence frozen in rev 2 |
| LS terminal timing and parity/export provenance were incomplete | important | terminal exclusion, six-clip ledger, code receipts, exact ONNX contract, source revisions, and fail-closed export rule frozen in rev 2 |
| rev-2 LS center-bound acoustic support yielded zero primary 500 ms pairs | blocking | candidate-aligned 250/500/1000 ms diagnostic supports and nonzero paired-valid invariants frozen in rev 3 |
| rev-2 terminal wording forbade committed frontend analysis padding and decode-only flush | important | forbidden source-audio ingestion is separated from frozen STFT/context padding and `ingest=0, decode=1` terminal mechanics in rev 3 |

Final verdict: **pending same-reviewer repair verification**.
