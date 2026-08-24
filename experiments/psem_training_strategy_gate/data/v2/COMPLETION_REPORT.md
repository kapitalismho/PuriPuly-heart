# PSEM-STRATEGY-DATA-v2 completion report

## Result

The issue #86 dataset freeze artifact is recorded at local commit `2fbd021598cbb75891d1e0505f66791d164fa47e` for the dataset prerequisite to issue #76.

- authority: `https://github.com/kapitalismho/PuriPuly-heart/issues/86`
- pinned authority SHA-256: `90078d66026f1374b065a5b9022788c40fac076cd4cf307df87b5027ea3fcb63`

The accepted package is:

- label contract: `psem-handoff-v1`
- dataset freeze: `PSEM-STRATEGY-DATA-v2`
- freeze file SHA-256: `080f008af0f3ca089cc44f89fc4a1a33b12ee9345090661955f215e069fb2bf0`
- freeze payload SHA-256: `76e2840d5cbd8f56b7cc65dadc2c40daf26df8020fa065fb1d43514d0ae9b747`
- preflight file SHA-256: `844b52454d91f30106ac14ecbd325d1e11eb4b0b3e2db53e053af88d9f906ebe`
- preflight payload SHA-256: `d2ff76399281c7e9428ea715318ee7683348daa2120d918bb5696a5996f3a195`

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

The freeze binds 20 local dataset artifacts, two inherited calibration artifacts, 29 repository code/config inputs including `pyproject.toml` and `uv.lock`, all 93 source/reference identities, the split and evaluator contracts, and the exact passing preflight result through an acyclic freeze-core binding.

## Scope and limitations

`PSEM-STRATEGY-DATA-v1` remains immutable and auditable but is superseded for official issue #76 model selection. No issue #76 arm was trained, no third corpus was added, and no model prediction, model score, or issue #76 outcome influenced reference, tranche, annotation, contract, or split decisions.

No project-specific manual listening, boundary annotation, gold audit, MFA reproduction, or independent acoustic boundary-accuracy estimate was performed.

> PSEM-STRATEGY-DATA-v2 adopts the commit-pinned forced-aligned AMI/AliMeeting references released by Horiguchi et al. (ASRU 2025) as the common temporal activity reference. This project does not perform additional manual boundary adjudication or independently estimate their acoustic boundary accuracy.

This limitation applies equally to all three issue #76 arms and must be retained in the eventual issue #76 report and publication claims.

## Issue #76 readiness

Official issue #76 material runs may start only from the checked `PSEM-STRATEGY-DATA-v2` freeze and its passing preflight. The model arms, 3-second evidence geometry, two seeds, common head and losses, untouched EVAL discipline, shared threshold sweeps, and anti-leakage controls remain unchanged.

The publication-ready issue #76 update is in `ISSUE_76_HANDOFF.md`. It has not been posted externally because publication requires separate approval.
