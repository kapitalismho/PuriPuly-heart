# PSEM-SMALL-MODEL-PROBE-v1 — Gate 0 manifest freeze (issue #117)

Regenerate: `python experiments/psem_small_model_probe/manifest/build_manifest.py`
(all asserts fail-closed; hash-stable on rerun).

## Counts

| Split | Rows | Sessions |
|---|---|---|
| CAL12 (6 topology groups x 2; AMI 6 + AliMeeting 6) | 12 | 5 |
| MAIN48 (C1-C6 x 8; per stratum AMI 4 + AliMeeting 4) | 48 | 6 |
| EXT24 reserve (C1-C6 x 4; per stratum AMI 2 + AliMeeting 2) | 24 | 4 |
| ONTOLOGY16 (flag on MAIN48; 8 C3 + 8 C4, AMI 8 + AliMeeting 8) | 16 | subset |
| CONTROL24 (flag on MAIN48; C1+C2+C5, AMI 12 + AliMeeting 12) | 24 | subset |

`causal_bindable`: 84/84 true (no null causal spans; C6 rows still bindable at the
50% 1 s activity rule — bindable rate stays a Gate-1 metric).

## Hashes

- `file_sha256` (sha256 of raw `manifest.jsonl` bytes):
  `e5956ab0fc451c647f5d6bfbdece87f27816de5c5b45bc0423514511c2c36582`
- `freeze_sha256` (sha256 of canonical sorted-by-`episode_id` `{rows, counts}` JSON):
  `91efb276a5bbf523088113ff81afc7f59086294c4e856d2a0f4a4b19532e6d82`
- V2 reference (`experiments/psem_training_strategy_gate/data/v2/dataset_freeze.json`):
  file sha + `freeze_payload_sha256` pinned in `dataset_freeze.json`
  (`v2_freeze_file_sha256`, `v2_freeze_payload_sha256`,
  `v2_freeze_core_payload_sha256`, `v2_dataset_freeze_id`).
## Disjointness proof (build-time asserts, all passing)

- Session sets on `(corpus, session_id)`:
  CAL12 = {alimeeting/R0004_M0012, alimeeting/R1019_M1928,
  ami/EN2004a, ami/EN2006a, ami/EN2009d} (5);
  MAIN48 = {alimeeting/R1019_M1960, alimeeting/R1021_M1940,
  alimeeting/R1021_M1944, ami/ES2002b, ami/ES2009a, ami/ES2009b} (6);
  EXT24 = {alimeeting/R1021_M4073, alimeeting/R1021_M4080,
  ami/ES2009c, ami/ES2009d} (4).
  CAL12 n MAIN48 = {} ; EXT24 n (CAL12 u MAIN48) = {}.
- `episode_id` unique (84/84); evaluation windows non-overlapping within session;
  ONTOLOGY16/CONTROL24 subset of MAIN48 (16/24 exactly);
  native spans exactly 5000 ms; causal spans exactly 1000 ms with
  end <= authoritative transition (or nulls + `causal_bindable=false`).

## Derivation notes (GT only, no model outputs)

- GT intervals/transitions/episodes come from the cached relative-occupancy
  manifests (`results/dev|eval/relative_occupancy_manifest.jsonl`), themselves
  regenerated from the frozen V2 reference checkout (V2 `data/v2/` untouched).
- Native: latest qualifying 5 s ms-aligned window (A-activity > 95%,
  zero other-speaker, unmasked) in the longest A-only run before the transition.
- Causal: latest 1 s ms-aligned A-only unmasked window ending <= transition
  with A-activity >= 50% (builder choice; Gate-1 may revisit), else nulls.
- ONTOLOGY16 tiers: C3/C4 overlap with A-active >= 300 ms first, then C5/C6
  overlap fallback. Final freeze: all 16 from tier 1 (no fallback used).
- C3/C4 candidate order prefers longest A-active overlap (serves the 300 ms
  ontology rule); all other strata take earliest transitions.

## Follow-up (not a Gate-0 blocker)

- Audio bytes live outside the repo (`PSEM_CORPUS_ROOT` etc. unset on this
  machine). Spans are sample/time metadata only, so the freeze is complete,
  but Gate 1 must resolve every row's `audio_ref` against the corpus root and
  re-hash waveform bytes before binding/enrollment.
- Probe episode selection touches V2 DEV *and* EVAL sessions (DEV alone has
  10 sessions — too few for three session-disjoint splits with both corpora).
  Per #97 FINAL_DECISION, V2 roles are development-known; EVAL-session reuse
  here needs explicit program sign-off or a fresh holdout before any
  model-selection claim.
