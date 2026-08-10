# R0 Baseline and Decision Ledger

## Status

- Experiment: `speaker_representation_scd_v1`
- Goal: `GOAL-EXPERIMENT-PLAN-EN`
- Authority: `EXPERIMENT_PLAN.en.md`
- Authority SHA-256: `ca46bce33b90c89597b5c9f2092b952a3f76d638c9c5524d4ca7ba23800e9191`
- Audit date: `2026-08-10`
- Ledger status: working R0 input, not an accepted evidence boundary
- Current checkpoint: R0 protocol candidate repair

This ledger maps reusable local assets, forbidden inheritance, environment constraints, model
metadata, and the owner decisions required before an R0 contract can be frozen. It does not
authorize model acquisition, inference, training, confirmatory access, or product changes.

---

## 1. Baseline Git and Runtime Context

| Item | Audited value | Disposition |
| --- | --- | --- |
| Work branch | `experiment-v2-speaker-change-turn-boundaries-ls` | Pinned Goal context |
| Start/current HEAD | `44ca7d9a47fb27e0cf35439fd8069df92862cd07` | No new commit exists |
| Integration target | `origin/main` | Pinned at `848aa0b9f1b35388ded5a250d51a687223eac1c5` |
| New experiment tree | Only the Korean plan, pinned English plan, and this ledger | No implementation exists |
| Existing Phase 4 process | Legacy `phase4_signal run` is active under wrapper PID `33144` and worker PID `50000` | Do not contend for compute or alter its state |
| Production files | No changes | Must remain outside this experiment |

The active legacy process is an operational constraint, not input evidence for this Goal. Its
results, thresholds, and state cannot be inherited into this experiment.

---

## 2. Local Environment Audit

### 2.1 Application environment

| Item | Audited value |
| --- | --- |
| Project Python | `3.12.10` through `uv run` |
| Global Python | `3.14.0`; outside the project requirement and not eligible for the experiment |
| Project Python requirement | `>=3.12,<3.14` |
| NumPy | Available |
| SciPy | Available |
| ONNX Runtime | Available |
| huggingface-hub | Available |
| PyTorch | Not installed |
| Transformers | Not installed |
| scikit-learn | Not installed |
| pandas | Not installed |
| matplotlib | Not installed |

The application dependency lock shall not be expanded merely to conduct this experiment. R1 needs
a separate, locked research environment whose Python, PyTorch, Transformers, accelerator runtime,
and package hashes are recorded independently.

### 2.2 Audited host

| Item | Audited value |
| --- | --- |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 logical processors |
| Physical RAM | 33,933,414,400 bytes |
| CUDA discovery | `nvidia-smi` unavailable |
| Current compute availability | Unavailable for new heavy work while legacy Phase 4 runs |

This host is eligible for metadata checks and later CPU smoke tests. It is not yet an approved full
extraction or training environment for the four-encoder study.

---

## 3. Reusable Contract Assets

Reuse means semantic reuse under a new experiment contract. It does not automatically authorize a
direct import, and it never imports a legacy result or selection.

| Asset | SHA-256 | Reuse decision | Required adaptation |
| --- | --- | --- | --- |
| `speaker_turn_boundary/timeline.py` | `f0f31a3abf8b73c0be6e6cfa78b10a44a6a2f2cab9aa17856003bfa757b1ce21` | Reuse canonical 16 kHz source-position semantics | Add compatibility tests under the new schema namespace |
| `speaker_turn_boundary/events.py` | `2193bda0f06ff9e3d4171402c9ce2296ed273f10994de35332ca070d212b347a` | Reuse boundary/observation separation concept | New event schema must also record compute completion |
| `speaker_turn_boundary/ground_truth.py` | `34d2236595c4fb3e105b1aa5da8b4fa05e513f33979ca63c8c6903299d0f820d` | Reuse `SpeakerRegion` and active-set transition basis only | New taxonomy must separate new-speaker onset, exclusive-new-speaker, backchannel, and handoff |
| `speaker_turn_boundary/phase3_metrics.py` | `c8459bb738b2259cc04e6999a96d830df1d5454db88033e84bc19018ad7b87be` | Reuse deterministic one-to-one causal matching principles | Remove Phase 3/B0-specific selection and add 100/250/500 ms tolerances plus declared deadlines |
| `speaker_turn_boundary/schemas.py` | `62e38828ad245584893efae0d829727264b206ab059f822e27d80b95abf52690` | Reference only | Define new feature, raw observation, normalized event, and run-contract schemas |
| `speaker_turn_boundary/metadata.py` | `4f89ee7451c7decf4d2d994360d7f6c5333070d060147c289bce386fda1fb7d2` | Reuse metadata pattern | Add torch/transformers/accelerator/model-precision fields |
| `speaker_turn_boundary/provenance.py` | `151f864a346bb774d44dffa7c9fdcb25fc070ec62eb5525c918ae9a326ea6f00` | Reuse canonical hashing and identity patterns | Define new registry/cache/run identities; do not import legacy result identities |
| `speaker_turn_boundary/models/registry.json` | `102d5a78698919747c066a666a2842a8eaeacc5497462d6b99606779e8210df1` | Reuse ERes artifact identities only | Exclude LS/EEND artifacts from the representation registry |

### 3.1 Contracts that cannot be copied unchanged

- Legacy `POSITIVE_KINDS` treats `speaker_left` as negative and does not model
  `exclusive_new_speaker` as a separate latency target.
- Legacy Phase 3 localization tolerance is fixed at 500 ms, while the new experiment requires
  ±100/250/500 ms views and separate availability deadlines.
- Legacy metric matching contains product/B0-specific accounting that does not belong in the
  representation experiment.
- Legacy result schemas do not represent model layer, pooling, context mode, feature validity,
  checkpoint artifact, or compute completion.
- Legacy common-VAD output may become a candidate input only after it is re-pinned under the new
  protocol with exact availability timing.

---

## 4. Reusable Dataset Inventory

All listed datasets are `development-known`. None is eligible to become the new untouched
confirmatory test merely because its filename contains `held_out`.

| Manifest | Cases | SHA-256 | New-experiment role |
| --- | ---: | --- | --- |
| `alimeeting_eval_pilot.json` | 8 | `61415f7b5c322cedbdbddd1cc8c97decef993c4c6bea23037c23771e989aed93` | Development-known natural Mandarin/far-field source |
| `ami_dev_pilot.json` | 2 | `009e2a683b7bf15c50f934eabc4f0c52691af1b79e64735393a26ec783089e95` | Development-known natural English source |
| `ami_held_out_pilot.json` | 2 | `86eecb4054ae1709cc21a32075d23a807b1724bf96b80ec1b2056a3170f7e964` | Development-known natural English source; not confirmatory |
| `b0_phase0.json` | 3 | `cd5bb2b9249d502be88acd62c309e4289058cd9d001a4b367655ff8c0333c726` | Deterministic D0 fixture candidate |
| `ls_dev.json` | 202 | `14347cdbdb2eff4cc73489f1b59d6755723d9098089dad66ae222984e90370dd` | Development-known D1 synthetic English |
| `ls_held_out_clean.json` | 202 | `c0aabc5ad8c3f00ec53d45f3b372b8ebca7ca9237720a1bb7a70b8de7dda2581` | Development-known D1 synthetic English; not confirmatory |
| `ls_held_out_other.json` | 202 | `f0d169394a9fdee9e708bc9cad46c0547946bf967799fa4e2e1a398ddb984079` | Development-known D1 synthetic English stress; not confirmatory |
| `mixed_dev_pool.json` | 204 | `1221176c92f50a2b096e4cd64d5da0168527918e3fba539273c614eabf07a398` | Development-only legacy mixture; rebuild new connected blocks |
| `phase1_dev.json` | 2 | `071898f921a28daebce02adefd55279f58ffabcbd85b7a2cc813c0b5dd593486` | Small deterministic change/overlap fixture candidate |
| `puripuly_like_provisional.json` | 0 | `2408f45da68c85bc0332b1534390ec725da0e6f770bf7c25dbcf3f0329a39db5` | No usable data |

The default external legacy corpus root exists and contains LibriSpeech, AMI, AliMeeting, archives,
and regenerated Phase 2 material. R0 may verify file identity and metadata. It shall not treat an
existing legacy split as sealed confirmatory data.

### 4.1 Exact initial ERes/LS-EEND common-GT panel

The current Phase 4 design establishes the exact development-known intersection that can be used
for the first paired representation comparison.

| Item | Frozen identity |
| --- | --- |
| Episode manifest | `results/turn_episode_v1/episode_manifest_dev.json` |
| Manifest byte SHA-256 | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` |
| Manifest content SHA-256 | `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68` |
| Total episodes | 804 |
| Phase 4 diagnostic episodes | 695 |
| Positive/negative candidates | 450 / 360 |
| Matched pairs | 313 |
| Matched-pair rows SHA-256 | `fb29fff960932f2840433fa94f1a9e4bade167a6d935a6458dc6e9b191a4f9b9` |
| Phase 4 coordinate rows SHA-256 | `58cbd9eaf4554761bf71e698bc4b1f251ae722c4281be35d0270dbc0ab285470` |

This is suitable for same-row encoder comparisons and contextual ERes/LS-EEND event comparisons.
It is not confirmatory evidence because its labels, selections, and legacy detector outcomes are
already development-known. Phase 4 remains active, so the new experiment shall not read incomplete
outputs or contend for its compute resources.

### 4.2 Coverage gaps

- No approved public multilingual D4 manifest exists.
- No sealed public D5 confirmatory manifest exists.
- Existing natural data cover English and Mandarin but do not establish Korean, Japanese, or
  same-speaker code-switch claims.
- Existing synthetic data contain clean/gap/overlap/stress cases but do not by themselves establish
  conversational handoff validity.
- A new speaker/session identity ledger is required before any development/confirmatory split can
  be frozen.

### 4.3 Public-only D4/D5 decision matrix

The detailed decision is recorded in `R0_DATASET_DECISION.md`, SHA-256
`777fea6786e823601f0425b98c7c7fa52a844648b0e39ae1165683150c209308`.
It does not authorize download, waveform inspection, or confirmatory-label access. Exact release
file hashes are added to the R0 source ledger when acquisition is separately authorized.

| Candidate | Proposed role | Useful evidence | Limitation / required guard |
| --- | --- | --- | --- |
| Existing Phase 4 common-GT panel | Primary D4 development-known panel | Exact paired ERes/LS-EEND bridge; English/Mandarin natural meetings plus controlled English synthetic episodes | Already observed; never confirmatory |
| All AMI meetings | D4 development-known only | Strong natural English diagnostic evidence | Tracked coverage artifacts already expose annotation hashes, active-speaker summaries, and derived target counts; never confirmatory |
| VoxConverse v0.3 official test | D5 natural in-the-wild candidate | Overlap, diverse speakers, debates, talk shows, news, CC BY 4.0 research release | Entire named test partition; official website currently withholds audio, so acquisition needs official availability or byte-parity proof and language claims remain conditional |
| AISHELL-4 official evaluation/test meetings | D5 natural Mandarin cross-corpus candidate | 211 real meetings, 4–8 speakers, overlap and speaker-activity labels | Data terms are CC BY-SA 4.0; freeze the exact official split and verify no prior local/result exposure |
| Zeroth-Korean reserved speakers | D4 development or D5 controlled Korean candidate | CC BY 4.0, 105 train and 10 test speakers; supports speaker-disjoint same/different and synthetic-boundary probes | Read speech rather than conversation; cannot support natural-handoff claims |
| JVS hash-reserved speakers | D4 development or D5 controlled Japanese/nuisance candidate | 100 speakers with normal, whisper, and falsetto recordings | Audio permits non-commercial research but is not generally product-cleared; read/studio speech cannot support natural-handoff claims |
| Unused AliMeeting official sessions | D5 same-domain Mandarin sensitivity | Natural public meeting domain matching the existing Mandarin panel | Same-corpus generalization is weaker than AISHELL-4; must exclude all eight legacy session IDs and connected speakers |

Frozen decision: use the legacy common-GT panel for development selection; construct R6-Z from
the complete VoxConverse v0.3 official test partition, eight hash-selected AISHELL-4 official test sessions,
all Zeroth-Korean official test speakers, and 20 fixed JVS speakers. Report KO/JA as controlled
representation/SCD evidence, not natural handoff; do not claim VoxConverse as English-only until a
post-lock metadata audit supports that stratum.

Metadata sources:

- VoxConverse project and v0.3 annotations: `https://www.robots.ox.ac.uk/~vgg/data/voxconverse/`
  and `https://github.com/joonson/voxconverse` at
  `24bf60be297701cd7e4ef18550c6d390c1b87365`
- AISHELL-4 official project/data specification: `https://github.com/felixfuyihui/AISHELL-4`
  and `https://www.openslr.org/111/`
- Zeroth-Korean: `https://www.openslr.org/40/`
- JVS corpus: `https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus`
- AliMeeting: `https://www.openslr.org/119/`

---

## 5. Initial Model Registry Candidates

Remote metadata requests retrieved repository/config identities only. No weight file was
downloaded.

### 5.1 SSL encoders

| Model | Frozen repository revision | Preferred artifact candidate | LFS SHA-256 | Bytes | Config facts |
| --- | --- | --- | --- | ---: | --- |
| mHuBERT-147 | `7ad3fc0bc5106c58c9c13526abccad527150d135` | `model.safetensors` | `2359b3e9dc6869cb0855119a2866f056aeb400e46252da9cbcc8e9b7aee50c8b` | 377,510,584 | HuBERT, 12 layers, hidden 768, 12 heads, 16 kHz, normalization enabled |
| WavLM Base+ | `4c66d4806a428f2e922ccfa1a962776e232d487b` | `pytorch_model.bin` | `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51` | 377,617,425 | WavLM, 12 layers, hidden 768, 12 heads, 16 kHz, normalization disabled |
| UniSpeech-SAT Base+ | `74f559583458188867750f1b8cb6710b11f5be41` | `pytorch_model.bin` | `0ebc4dd3edc1e10e21a4d16791ad65b9217033d9205317e999a973304b27eda4` | 382,236,294 | UniSpeech-SAT, 12 layers, hidden 768, 12 heads, 16 kHz, normalization enabled |

Pinned metadata hashes:

| Model | `config.json` SHA-256 | `preprocessor_config.json` SHA-256 |
| --- | --- | --- |
| mHuBERT-147 | `9a67db4e9afb5f02772395bb442a0eb4b0275d5d19e074b98076f4ceada111da` | `e850c9afcd068b4f7f226000ccee047349368a23e1dc5f5784838f822b2d185c` |
| WavLM Base+ | `fea6df1c2700a3954fc07e70588aecc9055eeb28db2ff57151a2db0d19180ed4` | `99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2` |
| UniSpeech-SAT Base+ | `1818b3201e43c032ac763520c6f82820f32b16aaa4edae78e700832bdf116a71` | `4a93853b74278b7c769d07f5a861e5d12ceb5db2bced5620d335f87238cb9e86` |

WavLM and UniSpeech-SAT expose only pickle-based `pytorch_model.bin` artifacts in their pinned
repositories. R1 must verify the exact LFS digest before load, use the official trusted source in
an isolated environment, and record the safe loading/conversion path.

### 5.2 ERes encoder

The legacy local registry already pins the primary standard checkpoint.

| Item | Value |
| --- | --- |
| Model ID | `iic/speech_eres2netv2_sv_zh-cn_16k-common` |
| Revision | `1cf80d41fb3435bd3d8df185b5c423333b2db42a` |
| Checkpoint file | `pretrained_eres2netv2.ckpt` |
| Checkpoint SHA-256 | `0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c` |
| Checkpoint bytes | 71,768,231 |
| Frontend | 80-bin feature input at 16 kHz |
| Final embedding | 192 dimensions |
| Registry license | Apache-2.0 |

The existing ONNX export exposes only the final embedding. It cannot prove or provide the required
pre-pooling tap. R1 needs the exact official PyTorch source graph, a named tap, receptive-field
analysis, and final-embedding reconstruction parity.

---

## 6. License Disposition

| Model | Audited source statement | R0 disposition |
| --- | --- | --- |
| mHuBERT-147 | Model card declares `cc-by-nc-sa-4.0` | `research_allowed`, `product_restricted`; derived-student status requires separate review |
| WavLM Base+ | Model card points to the Microsoft UniSpeech license, whose first line is `Attribution-ShareAlike 3.0 Unported` | `research_allowed`, product status `unknown` pending legal review |
| UniSpeech-SAT Base+ | Model card points to the same Microsoft UniSpeech license | `research_allowed`, product status `unknown` pending legal review |
| ERes2NetV2 standard | Legacy registry records Apache-2.0 | `research_allowed`; product status provisional until exact pre-pooling source license is verified |

The license file referenced by the two Microsoft model cards was read from
`https://raw.githubusercontent.com/microsoft/UniSpeech/main/LICENSE` and had SHA-256
`74295d561f60f770a4aa7525b71c0d119ec70422e9f5c601ee3c77e1b7822c91` at audit time.

This is a technical provenance classification, not legal advice. No model may be called a product
candidate until the corresponding R0/R9 legal gate is accepted.

---

## 7. Forbidden Inheritance

The following boundaries are fail-closed.

| Legacy item | New-experiment disposition |
| --- | --- |
| ERes/LS-EEND thresholds | Forbidden; select only from the new development split |
| Legacy profile winners/shortlists | Forbidden |
| Legacy go/stop or comparative conclusions | Context only, never input evidence |
| Legacy held-out labels/results | Development-known; never new confirmatory evidence |
| Legacy model caches | Forbidden unless the complete extractor/cache identity matches and sampled parity passes; no such cache is currently accepted |
| Phase 3/4 product reducer state | Forbidden in the representation comparison |
| LS-EEND raw posterior as a representation row | Forbidden |
| Full-context SSL feature as streaming evidence | Forbidden |
| GT activity mask as an operational result | Forbidden; diagnostic upper bound only |
| Production application wiring | Outside scope |

---

## 8. Section 30 Decision Register

The owner resumed execution with the recommended D30-3, D30-5, and D30-6 dispositions; those
decisions are now frozen in authority Section 30 and the machine contracts.

| ID | Open decision | Current evidence | Recommendation | Status |
| --- | --- | --- | --- | --- |
| D30-1 | D4 recruitment scale and language combinations | The owner cannot collect private data | Replace recruitment with approved public multilingual development data and limit claims to observed public-language/scenario coverage | Resolved by owner |
| D30-2 | New untouched confirmatory data | AMI is GT-exposed in tracked coverage artifacts; only public data are available | Use the complete VoxConverse v0.3 test partition plus fixed AISHELL-4/Zeroth-Korean/JVS rules in `R0_DATASET_DECISION.md` | Resolved; AMI permanently development-known |
| D30-3 | Primary false-event budget | Existing development sessions are too small for precise rate claims | Use integer false-event Pareto frontier as primary during development; retain `1/hour` as a labeled reference and freeze a rate operating point only if preflight proves estimability | Resolved by owner resume |
| D30-4 | Primary R5 head | The owner limited the current execution to pre-training zero-shot work | Defer all learned probes/heads and their architecture decision to separate approval after R6-Z | Resolved as not applicable to current execution |
| D30-5 | Hardware and ceilings | Local host has no visible CUDA and a legacy CPU experiment is active | Sequential CPU only after legacy release; one model/worker, 8 threads, 24 GiB RAM, 25 GiB source, 20 GiB cache, 50 GiB external storage, 96 total and 24 per-model wall hours; smoke forecast required | Resolved by owner resume |
| D30-6 | Restricted-license teachers | mHuBERT is non-commercial; Microsoft model product status is unresolved | Include restricted/unresolved models as research-only candidates; exclude product claims until a separate legal gate records product_allowed | Resolved by owner resume |
| D30-7 | LS-EEND contextual table | The owner authorized reuse of the exact ERes/LS-EEND measurement data | Use manifest byte SHA-256 `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` for development context; a newly run, pre-locked LS-EEND configuration may enter only the D5 natural-event table | Resolved by owner and dataset decision |

No downstream artifact-acquisition, environment, smoke, parity, or confirmatory-access gate is
implied by these owner decisions.

---

## 9. Proposed First Reviewable R0 Artifact Set

The first coherent R0 checkpoint should contain only deterministic contracts and validation tests,
not neural inference or training.

```text
experiments/speaker_representation_scd/
    README.md
    R0_BASELINE_DECISION_LEDGER.md

    configs/protocol/
        r0_protocol.json
        analysis_contract.json
        compute_ceiling.json
        license_disposition.json

    data/
        source_ledger.json
        split_contract.json
        confirmatory_access_policy.json

    models/
        registry.json

    schemas.py
    provenance.py
    validate_r0.py

    tests/
        test_r0_contracts.py
        test_r0_provenance.py
        test_r0_split_policy.py
```

### 9.1 R0 protocol content

The protocol must freeze at least:

- Canonical 16 kHz source timeline and three-time event contract
- `local_trailing_window`, `left_context_tail_pool`, and `offline_full_context` semantics
- Primary pooling durations and observation hops
- L1/L3/L6/L9/L12 layer convention
- Matched-budget and best-operating-point panels
- Common causal VAD, ungated, and oracle-activity conditions
- GT event taxonomy and primary/secondary targets
- Representation, event, compute, and statistical metrics
- Whole-block bootstrap and deterministic matching
- R3 funnel preserving one sentinel per encoder through R5
- Test-access guard and amendment policy
- Model/license/data identities and explicit unknowns

### 9.2 R0 validation requirements

- Canonical serialization and self-hash fixtures
- Authority/model/data/source identity validation
- Fail-closed mutable model revision rejection
- Existing-known versus confirmatory split rejection
- Speaker/session/transformation connected-component leakage fixtures
- Disallowed legacy-result/cache import fixtures
- Future-access policy fixtures
- Compute ceiling schema and no-execution preflight

---

## 10. R0 Stop Conditions

Stop before R1 if any of the following remains true.

- The owner has not resolved decisions that materially change dataset authority, compute resources,
  model eligibility, or the confirmatory boundary.
- The new confirmatory source cannot be sealed before labels/results are inspected.
- A model artifact, source revision, or license cannot be pinned.
- The ERes source graph needed for pre-pooling parity cannot be legally or technically identified.
- The research environment would alter production dependencies.
- The compute plan would contend with the active legacy experiment.
- The protocol permits future-audio leakage or conflates pooling duration with availability latency.

---

## 11. R0 Recommendation

The local baseline is suitable for building an independent R0 contract without duplicating its
timeline, active-set, hashing, and manifest work. It is not sufficient for immediate R1 execution.

The corrected dataset, common-GT, false-event, license, and resource decisions are frozen. The R0
contract artifact set may be implemented and reviewed without touching the running legacy
experiment or opening confirmatory data. Neural execution remains gated on acquisition, a locked
research environment, legacy resource release, extractor validity, and a passing smoke forecast.
