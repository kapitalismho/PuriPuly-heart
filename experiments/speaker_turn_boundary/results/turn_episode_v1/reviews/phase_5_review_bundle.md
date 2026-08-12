# Phase 5 pre-execution review bundle — compact corrected proposal and fusion replay

Status: **review findings repaired locally; not accepted for execution**. Revision 8
replaces revision 7 after the sole fresh reviewer returned `repair_required`. The owner
directed that the findings be fixed without another review round, so this repaired
candidate is deliberately not represented as independently accepted. No Phase 5
performance execution, scored Phase 5 result, held-out access, provider call,
production change, or Phase 6 preparation has occurred. Only pinned-input validation,
compact design generation, synthetic/frozen-shape benchmarks, and outcome-critical
scientific fixtures have run.

Candidate: `working-tree` on `6d085346ef84165223ec93da37c9d7a54bfaacbb`.

## 1. Authority and accepted entry boundary

| Item | Exact identity |
| --- | --- |
| Normative PRD | `.agents/specs/prd/bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md` |
| PRD SHA-256 | `bad637985e6ea2b82b0ac0e233b99ca7364d324dd2c5a38ec446b95a8604fbc4` |
| Accepted Phase 4 scientific candidate | `5edfa67f7bb73c352b15459fdde018b196b5b5ac` |
| Accepted Phase 4 review record | `b5b1397f5ade6fbdbf0dde26af08318646d7a9b5` |
| Accepted Phase 4 bundle SHA-256 | `a6afa3dc946815c162ee18d09b1c7ad3ad08e252f7286c110b37a685fe2b1759` |
| Phase 4 completion content SHA-256 | `db75772938fc4a59f21784e9fbc279ad3003bffc72b32594d7844fec8a28f14c` |
| Phase 4 verification content SHA-256 | `f8ba0e6498d2bc6d87854b6bdaefb5f7f15a7263ea9f98c399cd8b56d8bab51c` |
| Integration target | `origin/main@848aa0b9f1b35388ded5a250d51a687223eac1c5` |
| Superseded revision-6 bundle SHA-256 | `07643d83dc7092289d8cf6e5a8ecf7a8e8fe2cfe2a1c1753fd4ce46676d810c3` |
| Reviewed revision-7 bundle SHA-256 | `c337d38d76b3188e7c250cff72a991bd3bb936cc9d84ee621fa142dc2da6237b` |
| Revision-7 verdict | `repair_required`, reviewer `/root/phase5_compact_preexec_review` |

Phase 4 accepted ERes2NetV2 `signal_go` and LS-EEND `signal_stop`. Phase 5 therefore
uses only the accepted E-standard ERes evidence. All 432 historical LS rows remain
visible as `not_replayed_signal_stop`; no LS inference or policy replay is allowed.

Current HEAD advanced only through independently owned, non-overlapping
`experiments/speaker_representation_scd` work. Those tracked and untracked changes are
excluded from this review. Every Phase 5 candidate file is under
`experiments/speaker_turn_boundary`; the Goal STATE is operational control state.

## 2. Compact scope and hard exclusions

Exactly four proposal profiles may execute:

| Profile | Policy class | Window | Step | Checkpoint |
| --- | --- | ---: | ---: | --- |
| `phase4_native:adjacent_direct:E-standard:W8000:S1600:T500` | adjacent | 8,000 | 1,600 | E-standard |
| `phase4_native:adjacent_direct:E-standard:W8000:S4000:T500` | adjacent | 8,000 | 4,000 | E-standard |
| `phase4_native:prototype_memory_4:E-standard:W8000:S1600:T500` | prototype-memory-4 | 8,000 | 1,600 | E-standard |
| `phase4_native:prototype_memory_4:E-standard:W8000:S4000:T500` | prototype-memory-4 | 8,000 | 4,000 | E-standard |

The adjacent accepted signal is evaluated at the two declared reference-aligned steps.
The two prototype profiles use the accepted source-prefix four-prototype state contract.
All four retain the 36-point clustering grid, four VAD-fusion modes, five-stage ladder,
and three required frequency controls.

Hard exclusions:

- no new W24 inference, cache import, proposal generation, or policy replay;
- no replay of the 936 legacy compatibility profiles; they remain immutable historical
  evidence only;
- corrected historical scoring uses B0/B1 and the same four compact profiles over the
  exact 204 cases;
- no LS-EEND profile, held-out content, Phase 6 frontier/panel/freeze, provider access,
  production wiring, public entrypoint, credential, network, paid/live call, OpenCode,
  merge, push, deployment, release, or cleanup.

After independently verified and accepted Phase 5 per-policy results, execution must
stop and report to the owner before any Phase 6 preparation. A separate explicit owner
resume is required.

## 3. Frozen inputs and population

| Input | Bytes | Byte SHA-256 |
| --- | ---: | --- |
| `episode_manifest_dev.json` | 2,936,679 | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` |
| `natural_exposure_manifest.json` | 538,834 | `42b21562222f19fc880b93a40a3a999b122ec4afd00a3c7deeab46b9bc482e1c` |
| `state_equivalence_report.json` | 435,052 | `6e33711632d5f2e3de8e0c22c229b08827d1ccbb873deba2c1681a2ab2c544ec` |
| `oracle_provider_neutral.json` | 2,277,473 | `be44a6a7764cff4c01064bc506c1d29ab6b4f35dbb48797409e68a610fea82db` |
| `proposal_contract.json` | 143,200 | `982c54a4164335e9be5e1823deb3f8b51915cf90f07845beb0e11f8abd22fd4b` |
| `fusion_contract.json` | 3,910 | `bfda0c3c0ea7b6613ded79e9639692a33449dcf34202b1f2a5e7ec14c45f9873` |
| `phase_4_completion.json` | 4,378 | `368a5c23a30e10f1884fd3797166b23ee93df0a1d0f84fc7006010b17fdec565` |
| `phase_4_verification.json` | 2,167 | `dda1f1c1d9f51e9eec919e31f31635f07d22e0e84369cf87d6a755186ab12740` |
| historical `dev_rows_v2.jsonl` | pinned | `6fc01ce8f679aad4e4d9c6d5c45a1d0552f0ba030e46306a7a308466155c8f19` |
| historical `mixed_dev_pool.json` | 673,589 | `1221176c92f50a2b096e4cd64d5da0168527918e3fba539273c614eabf07a398` |
| historical `dev_run_contract_v2.json` | 3,001 | `099c7a5dc6ce38916b5937b29527397534549ac28030e73043574ff89c44145f` |
| historical `preflight_v2.json` | 6,797 | `87d684a1bf48b35afc796d390ad33ae814645d05c1f5a80195799e6b707042b4` |
| historical `b0_vad_only.json` | 1,150,135 | `769a987b0b0f05cc1b0b32cf61a20e64472823ad4c91e852589240825600d3c7` |

The current population is 878 episodes from 626 sources: 606 synthetic and 20 public.

| Pool | Episodes |
| --- | ---: |
| `diagnostic_dev` | 695 |
| `frontier_dev` | 109 |
| `natural_exposure_validation` | 74 |

The corrected historical population is exactly 204 cases. Confirmatory-held-out paths
remain unresolved and the held-out episode count is zero.

## 4. Proposal, clustering, fusion, B0/B1, and scoring contract

Adjacent profiles use episode-reset semantics accepted by state equivalence. Prototype
profiles execute each `(profile, source)` once to the maximum required tail and route
content-identical source-prefix state/proposal snapshots into episode epochs. Fixtures
require routed proposals, progress, final-state hashes, and tail records to equal
independent per-episode source-prefix replay.

Every proposal binds family, checkpoint, profile, source, epoch, kind, boundary,
observation frontier, confidence semantics, state provenance, and confirmation evidence.
Future reads, artificial tail audio, cross-source/epoch state, unsafe progress, or absent
cache coordinates fail closed. Confidence is within-semantics mean `1-cosine`; the
threshold is strict `change_score > 0.50`.

The clustering grid is debounce `{0,100,250}` ms, radius `{250,500}` ms, refractory
`{0,250,500}` ms, and representative `{first,max_confidence}`: 36 nodes per profile.
Fusion uses VAD radius `{250,500}` ms and same-silence association `{false,true}`. It
derives silence and forbidden associations only from causally observed lifecycle facts.

B1 must remain exactly equivalent to accepted B0 in action kind, boundary, observation
frontier, lifecycle owner, and final segmentation. B0 matching is fixed before candidate
matching so retained or accelerated B0 success cannot be relabeled as neural recovery.
Matching is globally optimal and source-order preserving. Structural maximum-duration
actions remain segmentation boundaries but enter no benefit or harm attribution.
Unscored intervals are union-normalized and excluded from every exposure and outcome
metric. Clean/gap remains the hard-turn headline; overlap stays separate.
Trusted word timing with no clipped word in an episode is an observable negative
(`word_intervals=[]`), while only unavailable timing (`word_intervals=None`) increments
`lexical_not_observable`.

Uniform VAD-active, causal energy-change, and within-VAD-active shuffle controls match
each tested detector-created hard-action count. They use no ground-truth label or future
audio. Inability to place the exact count is explicit evidence, never a lowered count.

## 5. Exact execution universe

Current 878-episode workload:

| Quantity | Count |
| --- | ---: |
| Logical proposal-profile/episode routes | 3,512 |
| Physical episode-reset traces | 1,756 |
| Physical source-prefix passes | 1,252 |
| Physical proposal probe steps | 591,552 |
| Logical emittable positions | 191,323 |
| Cluster executions | 126,432 |
| Fusion executions | 505,728 |
| Maximum control executions | 1,517,184 |
| Physical execution systems | 2,503 |
| Logical systems | 4,611 |
| Logical system/episode identities | 4,048,458 |
| Logical ladder alias system edges | 2,108 |
| Logical ladder alias episode edges | 1,850,824 |
| Physical execution episode nodes | 2,197,634 |

Corrected 204-case historical workload:

| Quantity | Count |
| --- | ---: |
| Profile/case routes | 816 |
| Proposal probe steps | 68,961 |
| Logical emittable positions | 68,553 |
| Cluster executions | 29,376 |
| Fusion executions | 117,504 |
| Control executions | 352,512 |
| Neural policy/case identities | 940,032 |
| B0/B1 baseline case identities | 408 |
| Total logical case identities | 940,440 |
| Physical policy/case nodes | 510,000 |

The typed historical aggregate contains 4,608 neural systems plus explicit B0 and B1
rows. Each baseline row binds all 204 case identities, ordered identity/action/score
digests, the complete contamination/harm metric field set, and the exhaustive B0/B1
equivalence receipt. The combined ordered logical-identity digest covers exactly
4,988,898 identities.
The 936 legacy ERes rows and 1,369 historical aggregate rows remain unchanged historical
evidence and are not treated as newly replayed Phase 5 results.

## 6. E-standard cache and exact runtime forecast

Only E-standard W8000 coordinates enter the executable cache universe:

| Quantity | Count |
| --- | ---: |
| Unique/checkpoint window jobs | 427,566 |
| Reusable accepted Phase 4 windows | 219,802 |
| New inference windows | 207,764 |
| Accepted cache files | 1,251 |
| Accepted cache bytes | 888,753,697 |

Every accepted source receipt is validated before reuse. Missing windows are inferred
into a distinct Phase 5 cache contract. W24 receipts remain historical and are not
imported.

The benchmark ran on the declared Phase 4 CPUExecutionProvider host with 8 physical and
16 logical cores. Conservative measured floors include 103.779 policy batches/s,
13,964.932 proposal-state steps/s, 172,380.960 control placements/s, 202,942.446 scoring
actions/s, and 608,898.565 framed logical identities/s. The maximum source-prefix shape
is 32,696 steps; the maximum historical proposal shape is 11,392 positions.

| Forecast component | Seconds |
| --- | ---: |
| New E-standard inference | 2,462.070 |
| Accepted cache validation/import | 879.208 |
| Proposal state | 47.298 |
| Policy replay | 77.798 |
| Controls, conservative hard upper | 651.269 |
| Scoring, conservative hard upper | 800.338 |
| Serialization/hash/gzip verification | 9.959 |
| Independent verification | 4,445.499 |
| Total | 9,381.632 (2.60601 h) |

The repaired exact compact forecast is inside the approved 2–3 hour execution envelope and below
the mandatory 3-hour return-to-owner ceiling. Stage A is forecast at 0.94127 hour.
Execution must stop before Stage B if exact observed cardinalities, word-timing receipts,
RSS, shard size, or remaining runtime invalidate the reviewed bounds.

## 7. Independent verification and storage

Verification is exhaustive for input/cache/file/self hashes, system and episode/case
identities, completeness, generation-time causal/schema guards, B0/B1 equivalence,
pool/session/block aggregates, natural-rate labeling, summary arithmetic, and shard
receipts.

Raw and derived trace reconstruction uses the frozen deterministic 2,048-unit stratified
sample plus every mandatory sentinel and deterministic failure example. It covers every
observed checkpoint, proposal-policy class, pool, corpus, ladder stage, fusion mode, and
control kind, then fills by ascending SHA-256 of
`turn-episode-v1-phase5-audit-v1 || canonical_unit_id`. Any mismatch fails the phase.
There is no duplicate full neural/policy/control/scoring replay.

Projected durable result size is 34,170,623.70 bytes (about 32.6 MiB), across nine typed
representations:

| Representation | Rows |
| --- | ---: |
| Physical proposal receipts | 3,824 |
| Logical proposal routes | 4,328 |
| Physical systems | 2,503 |
| Logical systems | 4,611 |
| Alias edges | 2,108 |
| Current aggregates | 4,611 |
| Historical aggregates, including B0/B1 | 4,610 |
| Deterministic failure examples | 420 |
| Independent audit units | 2,048 |

Aggregate JSON is limited to 10 MiB. Detail uses deterministic gzip JSONL shards of at
most 20 MiB with row-count, key-range, rolling-content-hash, compressed-byte-hash, and
size receipts. A monolithic large JSON is forbidden. The compact design ledger itself is
341,798 bytes; the old 165.7 MiB state-equivalence artifact is absent and the accepted
replacement is 435,052 bytes.

## 8. Regenerated pre-execution artifacts

| Artifact | Bytes | Byte SHA-256 | Content SHA-256 |
| --- | ---: | --- | --- |
| `phase_5_design_ledger.json` | 341,798 | `3e0e82fc3d22fbcc4402ee797f805584a83a742209bf1be4c1867c25bea11e91` | `03d72c97c95f25bbb283b9ed935e1bc4173212cbb07890e4a8a405f8c06a797c` |
| `phase_5_policy_benchmark.json` | 14,707 | `f83d6aaa8d920852909b864ddef4ac4c8702d57328a40a5d14a6a76c9dee8f65` | `e89b657c19038160f24c9bf6d36f43a18bab8cc18c7211b0ebc270b4c214732c` |
| `phase_5_storage_benchmark.json` | 22,342 | `8675c01a75ec8d48423569f922b22ab777afa600e45dcb5be545964eca5b2d00` | `798ade66c1dbe5b2f3ed76f5f04e7705ed9d1c01e37a1ac77edb7ab0366ce306` |

Generated-from source hashes are embedded and must equal the live candidate bytes.
The ledger authority hash must equal the current PRD hash. All three JSON artifacts are
self-hashed and deterministic.

## 9. Execution order after approval

1. Persist the approved Phase 5 pre-execution review artifact and exact bundle hash.
2. Preserve the reviewed candidate boundary and verify the approved bytes.
3. Implement only the runner, bounded writers, exact-cardinality gate, and independent
   verifier required by this compact contract.
4. Run the minimum outcome-critical fixtures for cache identity, source-prefix routing,
   proposal causality, cluster/fusion semantics, B0/B1, matching/scoring, controls,
   storage, and verifier mutations.
5. Execute Stage A once and enforce the exact interstage gate.
6. Execute the compact Stage B once only if the gate remains valid.
7. Independently verify results and obtain a fresh Phase 5 exit review.
8. Report accepted per-policy performance results to the owner and stop before Phase 6.

No failed full attempt is automatically restarted or silently grid-reduced. Authority,
input/cache, timing, causality, state, B0/B1, identity, aggregate, storage, or audit drift
returns to the coordinator.

## 10. Sole-review repair disposition

The sole fresh review checked the compact W8-only universe, exclusions, audit, isolation,
and artifact bindings consistently, then required two repairs:

1. `phase5_scoring.py` now preserves observable-empty word timing as an observable
   negative and treats only missing timing as `lexical_not_observable`; the focused
   fixture exercises both states.
2. The typed historical aggregate now includes B0 and B1, 408 baseline case identities,
   full contamination/harm metrics, ordered identity/action/score digests, and the
   exhaustive equivalence receipt. The redundant unprojected historical-correction
   output was removed; dependent storage, identity, runtime, artifact, and bundle counts
   were regenerated.

Per the owner's explicit instruction, no re-review was requested. Therefore the recorded
review verdict remains **`repair_required` on revision 7**, while revision 8 records the
locally verified repairs and remains **unauthorized for Phase 5 performance execution**
under the current accepted-review gate.
