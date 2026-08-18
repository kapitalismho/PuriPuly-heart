# PSEM strategy dataset plan

Authority is GitHub issue #77 under the scientific boundary in issue #76.

The package is built in this order: freeze the operational label contract; reconstruct prior exposure; acquire and normalize natural meetings; run annotation-only calibration; census every candidate; group all knowable identities; select EVAL, then DEV, then TRAIN by connected component; freeze hashes and manifests; run the fail-closed dataset preflight.

No WavLM or scratch PSEM training, model-score inspection, model-family selection, operating-threshold selection, or product-readiness claim is part of this package.

Large audio and mutable intermediate artifacts remain outside Git. Repository artifacts contain deterministic source identities, provenance, authorization or license references, hashes, split roles, topology counts, and the final preflight result.

The initial contract version is `psem-handoff-v0`. Annotation-only calibration retained every provisional constant and froze the contract without a version bump. Any later pre-dataset-freeze constant change increments the contract version, is recorded in `AMENDMENTS.md`, and invalidates all affected generated artifacts.

The candidate inventory contains 20 AMI Mix-Headset meetings and all 8 AliMeeting Eval sessions available in the external corpus root. `source_manifest.jsonl` binds canonical 16 kHz mono PCM16 waveforms, corpus speaker identities, meeting-series metadata where known, licenses, authorization status, and annotation identities. `annotation_manifest.jsonl` binds AMI manual segment bundles and AliMeeting participant TextGrids without importing old event labels.

`prior_exposure_manifest.jsonl` reconstructs actual selection exposure from the committed R6, R7, R7-B, R8, R9, and issue #72 configs. It excludes the ten named meetings from future EVAL. Other AMI meetings are not blanket-banned; their EVAL eligibility remains unresolved until connected-identity and pinned-checkpoint overlap audits complete.

`annotation_normalization.py` reads the bound AMI segment XML and AliMeeting TextGrid bytes directly, converts decimal timestamps to the unsnapped 16 kHz half-open source-sample timeline, and sweeps all speech boundaries into complete canonical silence/singleton/overlap intervals. Unknown global identities remain locally traceable but carry `speaker_identity_known: false`; they are not treated as globally disjoint identities. AMI annotation tails beyond decoded audio are clipped only at the scored waveform end and counted explicitly. `normalization_manifest.jsonl` binds each session's canonical interval and new PSEM label result hashes and is regenerated if the frozen contract is later amended.

`annotation_calibration.py` computes annotation-only solo, silence-gap, overlap, intervening-speaker, micro-event, mask, and per-corpus granularity distributions from every accepted canonical timeline. `annotation_calibration.json` and `ANNOTATION_CALIBRATION.md` record the retained-constant decision and explicitly exclude model predictions, model scores, official model results, and model training from calibration authority.
