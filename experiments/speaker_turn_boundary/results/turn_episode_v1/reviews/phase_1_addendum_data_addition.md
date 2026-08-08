# Phase 1 addendum — data-gap list and authorized AMI materialization plan

Status: addendum to the accepted Phase 1 evidence (PRD Section 29 Phase 1 gate: "Any data
addition or sampling-rule change proposed from the inventory must be included in the
accepted Phase 1 evidence and reviewed before it is materialized for Phase 2 scoring").
User decision 2026-08-08: downloading additional sessions from authorized sources is
approved.

## 1. Gap identified by the inventory

`coverage_inventory.json` (commit `49c96aff`) reports:

- AMI: 4 materialized/scorable sessions (ES2003a, ES2004a, IS1008a, IS1009a), all
  previously touched; independent blocks 4; untouched scorable sessions 0.
- AliMeeting: 8 materialized/scorable sessions, all previously touched; untouched 0.
- Confirmatory gate (Section 16.3): at least eight independent contributing blocks per
  corpus. Not satisfiable by local materialization alone.

## 2. Gap-fill plan (frozen before materialization)

### 2.1 AMI audio materialization (authorized)

Source: AMI corpus mirror (`groups.inf.ed.ac.uk/ami/AMICorpusMirror`, HEAD-verified
reachable on 2026-08-08; per-meeting `Mix-Headset` 16 kHz wav via `corpus/ami.py`).

Selection rule (deterministic, frozen here):

1. Eligible: meetings with complete per-participant `words.xml` annotations already present
   locally; not among the 4 materialized meetings; series not in the touched series set
   {ES2003, ES2004, IS1008, IS1009}.
2. Order: ascending by `(sha256(meeting_id) hex, meeting_id)`.
3. Accept a meeting only if its series has not been accepted yet (keep-together rule: one
   meeting per series maximizes independent blocks); stop when 8 accepted.

Selected meetings (8 independent series):

| meeting | series | duration_s | est. wav size | parts |
| --- | --- | --- | --- | --- |
| ES2010d | ES2010 | 973 | ~31 MB | 4 |
| EN2002c | EN2002 | 2972 | ~95 MB | 4 |
| ES2006c | ES2006 | 2189 | ~70 MB | 4 |
| IN1014 | IN1014 | 3703 | ~118 MB | 4 |
| TS3006a | TS3006 | 1253 | ~40 MB | 4 |
| IS1007d | IS1007 | 2005 | ~64 MB | 4 |
| EN2001d | EN2001 | 3546 | ~113 MB | 5 |
| IS1006b | IS1006 | 2149 | ~69 MB | 4 |

Total forecast: ~601 MB audio. Storage available: yes (corpus root has 4+ GB of archives).

Materialization mechanics:

- Download via `corpus/external.py` `download_file` (resume-capable) from
  `ami_mirror_url(meeting_id)`; target `%TEMP%/opencode/stb_phase2_corpora/ami/audio/<meeting>/<meeting>.Mix-Headset.wav`.
- Verification after download: wav decodes as 16 kHz mono PCM16; decoded length within
  +/-(2 s) of the annotation-derived duration; per-file SHA-256 recorded in a
  materialization manifest (`results/turn_episode_v1/ami_materialization_manifest.json`)
  with the selected meeting list and each file hash.
- No official mirror checksum exists; duration/decode verification is the declared check.
- The AMI mirror does not serve a complete audio manifest; the exact file list is the
  frozen selection above.

Post-materialization: rebuild `coverage_inventory.json` (untouched scorable AMI sessions
become 8; independent-block estimate expected 12 AMI components, 8 untouched) and re-run
the Phase 1 exit-gate verification.

### 2.2 AliMeeting gap — unfillable from authorized sources

The authorized AliMeeting source in scope is the Eval set (the local archive
`alimeeting_eval.tar.gz` and the Eval_Ali distribution contain exactly 8 far sessions, all
already touched). No untouched authorized AliMeeting session exists. Therefore:

- AliMeeting cannot contribute confirmatory blocks in this run; its 8 sessions remain
  historical validation.
- A pooled AMI-plus-AliMeeting confirmatory claim (Section 16.3) is **not attainable** and
  is recorded as such; no AliMeeting download is planned.
- AMI-only confirmatory evidence (>= 8 untouched AMI blocks) becomes attainable, with the
  in-domain caveat: AMI-trained LS results on AMI are in-domain model evidence, reported
  stratified (Section 17).

## 3. Non-goals of this addendum

- No AliMeeting downloads; no new corpus sources; no change to sampling rules
  (natural-exposure frame and target-enriched selection stay as approved); no confirmatory
  held-out access; no provider credentials; no production code changes.
- Materialized AMI audio is opened only by the baseline B0 replay and the inventory; no
  scored episode manifests are generated (Phase 2).

## 4. Review request

This addendum and the materialization script
(`turn_episode/materialize_ami_additions.py`) are the review candidate. Approval is
required before the download begins.
