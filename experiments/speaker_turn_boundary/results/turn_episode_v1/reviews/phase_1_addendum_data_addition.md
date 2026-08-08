# Phase 1 addendum (rev 2) — data-gap list and authorized AMI materialization plan

Status: addendum to the accepted Phase 1 evidence (PRD Section 29 Phase 1 gate). User
decision 2026-08-08: downloading additional sessions from authorized sources is approved.
Rev 2 resolves review findings P1-DATA-001..007.

## 1. Gap identified by the inventory

`coverage_inventory.json` (commit `49c96aff`): AMI 4 materialized/scorable sessions (all
previously touched); AliMeeting 8 sessions (all previously touched); untouched 0; the
confirmatory gate (>=8 independent blocks per corpus) is not satisfiable by local
materialization alone.

## 2. Gap-fill plan (frozen before materialization)

### 2.0 Preflight (rev 2 addition, finding P1-DATA-001)

The AMI mirror serves heterogeneous files: some `Mix-Headset` wavs are stereo or have
lengths that differ from the annotation duration by up to hundreds of seconds. A
read-only preflight probed all 155 candidate meetings (annotated locally, series not in
the touched set): **78 pass** the frozen acceptance criterion — WAV header mono, 16 kHz,
16-bit PCM; Content-Length-derived duration within +/-2 s of the annotation duration.
Only preflight-passing meetings are eligible for selection. No format conversion is
performed; non-conforming meetings are excluded (the primary condition is mono 16 kHz).

### 2.1 Selection rule (rev 2, finding P1-DATA-002)

1. Eligible: preflight-passing meetings with complete local `words.xml` annotations.
2. Keep-together components: union-find over global participant ids (meetings.xml
   `nxt_agent`->`global_name`) and AMI series prefixes, over **all** annotated meetings
   (171) — the same rule as the accepted inventory group graph.
3. Exclude any component containing a touched session (ES2003a, ES2004a, IS1008a,
   IS1009a) — the earlier series-only exclusion missed IN1014/IS1006b sharing a
   participant component with IS1008a.
4. Order: ascending by `(sha256(meeting_id) hex, meeting_id)`; accept only meetings whose
   component is new; first 8 accepted form the **development** group, the next 8 form the
   **reserved** group.

Selected meetings (16 new independent components; all preflight-passing):

Development (8, opened for development: B0 replay + inventory detail rows):

| meeting | duration_s | est. wav size |
| --- | --- | --- |
| EN2002c | 2972 | ~95 MB |
| TS3006a | 1253 | ~40 MB |
| EN2001d | 3546 | ~113 MB |
| TS3009b | 2460 | ~79 MB |
| ES2015d | 1931 | ~62 MB |
| TS3007a | 1609 | ~51 MB |
| TS3012c | 2376 | ~76 MB |
| TS3005b | 2440 | ~78 MB |

Reserved (8, materialized but **not opened**; candidates for the Phase 6/7 confirmatory
held-out selection):

| meeting | duration_s | est. wav size |
| --- | --- | --- |
| TS3003b | 2210 | ~71 MB |
| ES2014a | 1149 | ~37 MB |
| TS3004a | 1345 | ~43 MB |
| EN2006a | 3526 | ~113 MB |
| EN2009d | 5324 | ~170 MB |
| TS3008b | 2320 | ~74 MB |
| ES2016a | 1384 | ~44 MB |
| ES2002b | 2281 | ~73 MB |

Total forecast: ~1.22 GB. Storage available (archives already occupy 4+ GB).

### 2.2 Access semantics (rev 2, finding P1-DATA-004)

- **Development group**: opened for development — B0 baseline replay, inventory detail
  rows (regions from words.xml), natural-exposure frame, target-enriched selection.
  These sessions are development material (diagnostic/frontier candidates), never
  confirmatory held-out.
- **Reserved group**: downloaded and decode-QA-verified at materialization, then **not
  opened**: no region extraction, no B0 replay, no inventory detail rows, no
  target-enriched selection. Their metadata (duration, speakers) is already covered by
  the annotation-level inventory counts. Any earlier scored-evaluation access moves them
  to historical validation (Section 17 fail-closed). Materialization decode-QA is
  artifact-integrity verification, not scored evaluation; it is recorded as such in the
  materialization manifest.
- Confirmatory status after this plan: AMI has 8 development blocks (opened) plus 8
  reserved blocks (untouched) available for the Phase 6 freeze / Phase 7 pre-access
  review. The pooled AMI-plus-AliMeeting claim remains **unattainable** (AliMeeting has
  no untouched authorized source; the Eval set contains exactly 8 far sessions, all
  touched). AMI-only confirmatory evidence is attainable from the reserved group,
  reported with the in-domain caveat (AMI-trained LS on AMI).

### 2.3 Materialization mechanics (rev 2, findings P1-DATA-003/005/006/007)

- Script: `turn_episode/materialize_ami_additions.py`:
  - **Recomputes the frozen selection before downloading** (same eligibility, union-find
    component graph, hash order) and fails closed on any mismatch with the frozen lists.
  - Downloads each wav to `<meeting>.part` with Range resume and atomically renames to
    the final name after full validation (canonical decode: mono 16 kHz PCM16; duration
    within +/-2 s of the preflight Content-Length duration; per-file SHA-256).
  - Emits `results/turn_episode_v1/ami_materialization_manifest.json` with an explicit
    ordered `selected_meetings` list, `group` (development|reserved) per meeting,
    per-file sha256/size/decoded duration, and a canonical `content_sha256` over the
    payload.
- Inventory integration (finding P1-DATA-003): `build_coverage_inventory.py` consumes the
  materialization manifest via `--ami-materialization-manifest`:
  - development sessions become scorable sessions (B0 replay included; B0 completeness
    gate becomes 20 sessions: 12 pilot + 8 dev);
  - reserved sessions are recorded in a new `reserved_materialized` inventory section
    (materialized, not opened; no detail rows, no B0 evidence, no target-enriched
    selection);
  - `untouched_scorable_sessions` counts only never-opened sessions (reserved group);
  - natural-exposure frame includes reserved sessions' durations (duration-only, computed
    before label inspection — no leak; the frame is thus frozen before Phase 6 selection).

### 2.4 AliMeeting gap — unfillable from authorized sources

The authorized AliMeeting source in scope is the Eval set (exactly 8 far sessions, all
touched). No AliMeeting download is planned; AliMeeting stays historical validation; the
pooled AMI-plus-AliMeeting claim is recorded as unattainable.

## 3. Non-goals of this addendum

- No AliMeeting downloads; no new corpus sources; no sampling-rule change (natural frame
  and target-enriched rules stay as approved; target-enriched selection excludes reserved
  sessions); no confirmatory held-out access before the Phase 7 pre-access review; no
  provider credentials; no production code changes.
- Reserved sessions receive no scored-evaluation access of any kind until the Phase 7
  review approval.

## 4. Review request

This addendum (rev 2), the revised materializer, and the revised inventory integration
are the review candidate. Approval is required before the download begins.
