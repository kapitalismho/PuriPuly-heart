# R0 Public-Only Dataset Decision

## Status

- Experiment: `speaker_representation_scd_v1`
- Decision date: `2026-08-10`
- Authority: `EXPERIMENT_PLAN.en.md`
- Scope: frozen encoders and zero-shot SCD only
- Acquisition boundary: existing or public data only; no participant recruitment or private recording
- Execution state: metadata decision only; no new corpus or model download, feature extraction, or confirmatory access occurred

---

## 1. Decision

The study uses two disjoint evidence tiers.

```text
development-known
  existing ERes/LS-EEND common-GT data
  plus public KO/JA controlled-development speakers

sealed public confirmatory
  VoxConverse v0.3 official test partition
  AISHELL-4 official test sessions
  Zeroth-Korean official test speakers
  hash-reserved JVS speakers
```

All layer, pooling, prototype, threshold, hysteresis, and promotion choices are made on
development-known data. Confirmatory data are opened once after every zero-shot configuration and
the evaluation-code hash are locked.

---

## 2. Development-Known Core

### 2.1 Exact common ERes/LS-EEND panel

| Item | Identity |
| --- | --- |
| Manifest | `experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json` |
| Byte SHA-256 | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` |
| Canonical content SHA-256 | `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68` |
| Total episodes | 804 |
| Phase 4 diagnostic episodes | 695 |
| Candidate inventory | 450 positive, 360 negative |
| Matched pairs | 313 |
| Pair rows SHA-256 | `fb29fff960932f2840433fa94f1a9e4bade167a6d935a6458dc6e9b191a4f9b9` |
| Phase 4 coordinate rows SHA-256 | `58cbd9eaf4554761bf71e698bc4b1f251ae722c4281be35d0270dbc0ab285470` |

This panel is the primary bridge to the existing experiment. It supplies identical audio, GT
events, source coordinates, positive/negative episodes, and matching blocks for the four new
representations. It is not confirmatory because existing ERes/LS-EEND outputs and selection
results have already been observed.

### 2.2 Public multilingual development additions

| Language | Source | Development rule | Permitted claim |
| --- | --- | --- | --- |
| English | Existing AMI and LibriSpeech-derived episodes | Use the common panel without resplitting | Natural EN event and controlled EN pair evidence |
| Mandarin | Existing AliMeeting episodes | Use the common panel without resplitting | Natural ZH event evidence |
| Korean | Zeroth-Korean training partition | After metadata import, sort speaker IDs by `(SHA-256, ID)` and take the first 20 speakers not used by the official test partition | Controlled KO same/different and synthetic-boundary evidence |
| Japanese | JVS | Use the 20 fixed development speakers below | Controlled JA same/different, normal/whisper/falsetto, and synthetic-boundary evidence |

JVS development speakers:

```text
jvs046 jvs095 jvs089 jvs081 jvs064
jvs060 jvs028 jvs009 jvs068 jvs015
jvs030 jvs053 jvs047 jvs078 jvs032
jvs055 jvs048 jvs022 jvs024 jvs097
```

The development objective uses equal language-block weighting where all four languages are
available. Natural and controlled/synthetic rows remain separate.

### 2.3 AMI exposure correction

All AMI meetings are `development-known` for this study. The tracked legacy
`coverage_inventory_details.jsonl` already contains annotation hashes, active-speaker summaries,
and derived hard-target counts for 163 AMI recordings, including the eight meetings previously
proposed as D5. Those eight meetings are permanently disqualified from confirmatory use:

| Exposure artifact | Identity |
| --- | --- |
| Path | `experiments/speaker_turn_boundary/results/turn_episode_v1/coverage_inventory_details.jsonl` |
| Byte SHA-256 | `15b2e4f0efa270985c3bbc6d848ee9ed25496089268e561bff921c5c1be3ef8c` |
| Rows | 171 total, 163 AMI |

```text
EN2004a ES2011d ES2012d ES2013d
TS3010b ES2007d EN2003a TS3011b
```

This correction supersedes the earlier metadata-only assumption. No alternative AMI subset may
be promoted to confirmatory status from this repository because meetings in the same participant
or series components are also exposed.

---

## 3. Sealed Public Confirmatory Panels

### 3.1 Natural in-the-wild SCD: VoxConverse

Use the complete official VoxConverse v0.3 `test` partition at annotation repository revision
`24bf60be297701cd7e4ef18550c6d390c1b87365`. The repository was absent from tracked legacy source,
annotation, coverage, and result artifacts before this selection. No VoxConverse audio, RTTM,
derived GT, feature, score, or aggregate has been opened locally.

The whole named test partition is selected, so no recording is chosen from its event counts,
speaker counts, duration, language, or model behavior. R2 must register exact audio and annotation
bytes and prove audio/RTTM parity before use. The official project currently states that audio is
not available from its website even though the v0.3 repository references download links;
therefore acquisition remains `unacquired` and fail-closed. A third-party mirror may be used only
after its bytes are reconciled to the official release identity. If parity cannot be established,
this source is `not_evaluable`; it is not silently replaced after development results are viewed.

VoxConverse contains in-the-wild debates, talk shows, and news clips rather than controlled
meetings. Language coverage shall be audited only after the R6-Z lock. Report the natural-English
stratum only for recordings whose language identity can be established independently of model
scores; otherwise report the panel as natural in-the-wild speech without an English-only claim.
The official repository describes research availability under CC BY 4.0 while noting that video
copyright remains with the original owners.

### 3.2 Natural Mandarin SCD: AISHELL-4

Use the official AISHELL-4 `test.tar.gz` partition from OpenSLR SLR111. After archive identity and
metadata are registered, create the evaluation subset by sorting official test session IDs by
`(SHA-256(session_id), session_id)` and taking the first eight sessions. The selection may be
expanded only by a protocol amendment made before any model score is observed.

This panel supplies natural meetings, overlap, quick turns, noise, and speaker-activity labels.
The public release declares CC BY-SA 4.0. The eight-session subset preserves a cross-corpus test
relative to the AliMeeting development data while bounding CPU-only confirmatory cost.

### 3.3 Controlled Korean representation/SCD: Zeroth-Korean

Use every speaker in the official Zeroth-Korean test partition as the confirmatory speaker set.
The OpenSLR metadata declares 10 test speakers and CC BY 4.0. Generate controlled same-speaker,
different-speaker, clean-change, gap, and overlap episodes only after the synthesis recipe and
seed are frozen on development data.

This is read speech and supports controlled representation/SCD claims only. It does not support a
natural Korean turn-taking or handoff claim.

### 3.4 Controlled Japanese representation/SCD: JVS

Use the following 20 hash-reserved JVS speakers.

```text
jvs050 jvs003 jvs094 jvs011 jvs052
jvs023 jvs002 jvs016 jvs013 jvs025
jvs093 jvs019 jvs066 jvs058 jvs051
jvs086 jvs059 jvs029 jvs033 jvs077
```

The split is obtained by sorting `jvs001` through `jvs100` by `(SHA-256(speaker_id), speaker_id)`;
the first 20 are confirmatory, the next 20 are development, and the remaining 60 are unused.
Normal, whisper, and falsetto recordings permit real same-speaker nuisance tests. Deterministic
splices permit controlled clean/gap/overlap boundaries.

JVS audio is restricted to academic/non-commercial research and related personal uses under its
published terms. It remains research-only and cannot support a product-eligibility claim.

---

## 4. Confirmatory Claim Structure

The final report shall not collapse all sources into one score.

| Panel | Primary outputs |
| --- | --- |
| Natural in-the-wild VoxConverse | Boundary F1/recall, availability latency, false events/hour, overlap/backchannel strata; English-only claim conditional on post-lock metadata audit |
| Natural Mandarin AISHELL-4 | Same event metrics and cross-corpus robustness |
| Controlled Korean Zeroth | Same/different AUC/EER, nuisance separation where available, synthetic-boundary F1/latency |
| Controlled Japanese JVS | Same/different AUC/EER, normal/whisper/falsetto robustness, synthetic-boundary F1/latency |

The cross-language headline reports per-language values, macro averages, and the worst observed
language. It does not imply natural conversational coverage for Korean or Japanese.

---

## 5. Relationship to Existing ERes/LS-EEND Work

The shared development panel permits a strict paired comparison because every method receives the
same source audio and GT/time contract. The new study reuses timeline, episode, active-speaker,
matching, block, and provenance identities. It does not reuse legacy thresholds, shortlist
conclusions, detector-specific state, or raw feature caches.

Existing ERes/LS-EEND results may appear in a development-known contextual table. After the current
experiment finishes and one ERes-final configuration and one LS-EEND configuration are frozen,
those configurations may also be run once on the sealed natural VoxConverse/AISHELL-4 panels. Such new runs
may enter the confirmatory event-level table because their configuration was fixed before D5
access. LS-EEND never enters representation AUC/EER or layer-ranking tables.

This yields two valid comparisons:

```text
representation comparison
  mHuBERT vs WavLM vs UniSpeech-SAT vs ERes pre-pooling
  identical rows and zero-shot detector logic

event-system comparison
  four zero-shot representation detectors vs ERes final vs LS-EEND
  identical natural audio, GT events, matching, and timing definitions
```

---

## 6. Leakage and Access Rules

- Existing `held_out` names remain development-known.
- AMI is development-known only and cannot enter the confirmatory tier.
- VoxConverse v0.3 test content cannot influence layer, pooling, threshold, hysteresis, or model
  promotion; its complete named partition is the unit of selection.
- AISHELL-4 test, Zeroth test, and reserved JVS speakers cannot influence layer, pooling,
  threshold, hysteresis, or model promotion.
- Confirmatory feature extraction is permitted only after the run-contract hash is frozen.
- Derived confirmatory GT may be materialized automatically after lock, but no model-specific
  setting may change in response to its event counts or results.
- Public-checkpoint pretraining overlap is generally unknowable; report it as `unknown` rather
  than describing D5 as pretraining-unseen.
- A failed/missing model row uses the all-encoder intersection as primary analysis and a separate
  missingness sensitivity.

---

## 7. Acquisition and Storage Consequences

Approximate published archive sizes are:

| Source | Approximate download |
| --- | ---: |
| AISHELL-4 official test | 5.2 GB |
| Zeroth-Korean | 10 GB |
| JVS | 3.5 GB |
| VoxConverse v0.3 official test | Unknown until an official or parity-proven archive is registered; the common storage ceiling applies |

The known archives total approximately 18.7 GB plus the as-yet-unregistered VoxConverse test
archive. Acquisition is allowed only if the registered total remains within the frozen 25 GiB
source-download and 50 GiB external-storage ceilings; otherwise execution fails closed pending an
owner-approved amendment. Downloads remain disabled until the later acquisition gate. Pooled
features shall be cached; full hidden-state tensors shall not be retained across the complete grid.

---

## 8. Source Registry

- VoxConverse project: `https://www.robots.ox.ac.uk/~vgg/data/voxconverse/`
- VoxConverse v0.3 annotations: `https://github.com/joonson/voxconverse`
- AISHELL-4 OpenSLR SLR111: `https://www.openslr.org/111/`
- AISHELL-4 data specification: `https://aishell-4.oss-cn-hangzhou.aliyuncs.com/AISHELL-4%20Data-Specification.pdf`
- Zeroth-Korean OpenSLR SLR40: `https://www.openslr.org/40/`
- JVS project page: `https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus`
- AliMeeting OpenSLR SLR119: `https://www.openslr.org/119/`
