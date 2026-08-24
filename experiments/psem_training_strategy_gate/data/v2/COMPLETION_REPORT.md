# PSEM-STRATEGY-DATA-v2 completion report

## Result

The issue #86 dataset freeze checkpoint is accepted at local commit `fc2d7cc74932aef265649238245b78318c0af2c1` for the dataset prerequisite to issue #76.

- authority: `https://github.com/kapitalismho/PuriPuly-heart/issues/86`
- pinned authority SHA-256: `90078d66026f1374b065a5b9022788c40fac076cd4cf307df87b5027ea3fcb63`

The accepted package is:

- label contract: `psem-handoff-v1`
- dataset freeze: `PSEM-STRATEGY-DATA-v2`
- freeze file SHA-256: `7cfbec9f12981bddd6f66c284cbb6547b53c52137fbcdc52380d6cec5de0433b`
- freeze payload SHA-256: `583c955e6d3a3f1ad2fcdd9156a42718bcf98288bbec4edbda5a661f9bbc9d86`
- preflight file SHA-256: `ded73bb9658b302441f1fd4a424746bb7e82e1e67f5aa94af66196a7e72a7d0a`
- preflight payload SHA-256: `f55a8dc80d2de95864c4caf305a8de11f1d08fe0fa174b3aa5e1f3dd923e8802`

SHA-256 values are integrity receipts for files, contracts, and bound identities. They are not model-selection evidence. AliMeeting Train objectives 1–3 produced one optimum, so the salted SHA-256 tie-break was not exercised.

## Dataset and annotation result

The v2 package contains 93 natural meeting sources:

- 68 retained AMI meetings
- 8 retained AliMeeting Eval meetings
- 17 selected AliMeeting Train meetings from whole identity components

All canonical activity timelines use the released RTTMs from `nttcslab-sp/diar-forced-alignment` commit `9527b7c64846fb38316a610f32e9d3466bd6d8b7`. Original AMI XML and AliMeeting TextGrid metadata remain the authority for transcript, source, speaker, and explicit nonlexical-risk identities, but not literal speech onset and offset.

The split contains 57 identity components and assigns:

| Role | AMI meetings | AliMeeting meetings | AMI scored hours | AliMeeting scored hours |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 50 | 14 | 24.996625 | 7.461752 |
| DEV | 7 | 3 | 4.568083 | 1.537742 |
| EVAL | 11 | 8 | 6.033410 | 3.974088 |

All 37 natural-data, topology, corpus-balance, leakage, prior-exposure, and exact WavLM-overlap gates pass. The final preflight contains the exact 59 required checks, with 59 passing, zero failing, and `ready_for_issue_76=true`.

The freeze binds 20 local dataset artifacts, two inherited calibration artifacts, 27 repository code/config inputs, all 93 source/reference identities, the split and evaluator contracts, and the exact passing preflight result through an acyclic freeze-core binding.

## Scope and limitations

`PSEM-STRATEGY-DATA-v1` remains immutable and auditable but is superseded for official issue #76 model selection. No issue #76 arm was trained, no third corpus was added, and no model prediction, model score, or issue #76 outcome influenced reference, tranche, annotation, contract, or split decisions.

No project-specific manual listening, boundary annotation, gold audit, MFA reproduction, or independent acoustic boundary-accuracy estimate was performed.

> PSEM-STRATEGY-DATA-v2 adopts the commit-pinned forced-aligned AMI/AliMeeting references released by Horiguchi et al. (ASRU 2025) as the common temporal activity reference. This project does not perform additional manual boundary adjudication or independently estimate their acoustic boundary accuracy.

This limitation applies equally to all three issue #76 arms and must be retained in the eventual issue #76 report and publication claims.

## Issue #76 readiness

Official issue #76 material runs may start only from the checked `PSEM-STRATEGY-DATA-v2` freeze and its passing preflight. The model arms, 3-second evidence geometry, two seeds, common head and losses, untouched EVAL discipline, shared threshold sweeps, and anti-leakage controls remain unchanged.

The publication-ready issue #76 update is in `ISSUE_76_HANDOFF.md`. It has not been posted externally because publication requires separate approval.
