# State-corrected adaptation decision

Issue 121 H7301 is **STOP / inconclusive**. Retain F0 operationally. This is the
Director's frozen scientific disposition after the completed CPU product
postprocess; it is not a new GPU request.

The CPU execution contract is verified, but this run is not a formal immutable
candidate and is not terminal acceptance. H7301 is not accepted. No automatic
H7302, T2, TA, or EVAL arm was opened, and no Gate 1 receipt was emitted.

## Ten Issue §31 questions

### 1. Is H7301 accepted, and what is the operational disposition?

No. Only the 100 ms point is jointly useful on the equal-corpus macro C/M
criterion. The 300 ms point is not jointly useful and the 500 ms point is worse.
That is one of three horizons, below the required two of three. Retain F0; do not
promote H7301, open a confirmation arm, or call this a candidate acceptance.

### 2. What are the corrected-H macro results at all horizons?

The table reports contamination in seconds per active-speech hour, miss rate, and
false cuts per hour. `C envelope` and `M envelope` each use one selected global
operating threshold shared across all ten meetings (C and M may differ). The rows
are all-source equal-corpus macro aggregates at the selected global threshold
for that envelope. These global-threshold rows are the decision quantities;
independent corpus-envelope operating points in Question 3 use separate
AliMeeting and AMI thresholds and are descriptive, not substitutes for this
table.

| Horizon | F0 C / miss / false | H at C-envelope C / miss / false | H at M-envelope C / miss / false | Jointly useful |
| --- | --- | --- | --- | --- |
| 100 ms | 2161.675 / 0.775487 / 228.734 | 1561.578 / 0.583874 / 108.642 | 1570.013 / 0.571040 / 151.139 | yes |
| 300 ms | 1555.186 / 0.537089 / 92.282 | 1566.368 / 0.576841 / 89.445 | 1566.368 / 0.576841 / 89.445 | no |
| 500 ms | 1978.648 / 0.687291 / 30.487 | 2232.237 / 0.817277 / 30.452 | 2232.237 / 0.817277 / 30.452 | no; worse |

Corresponding H-F0 macro deltas are:

- 100 ms C: contamination `-600.096946`, false cuts `-120.092529`, miss
  `-0.191614`; M: `-591.662003`, `-77.594974`, `-0.204447`.
- 300 ms C/M: contamination `+11.182169`, false cuts `-2.836675`, miss
  `+0.039752`.
- 500 ms C/M: contamination `+253.589080`, false cuts `-0.035230`, miss
  `+0.129986`.

The selected score thresholds are raw/calibrated `0.588784479/0.061180602`
and `0.363103618/0.031632118` for the 100 ms C/M points;
`0.391021290/0.034480712` for 300 ms C/M; and
`0.723463754/0.092900882` for 500 ms C/M. These are score-coordinate changes,
not evidence of a different event result.

### 3. Is the 100 ms cross-corpus gain real, and is it universal?

The 100 ms gain is real at the two independent corpus-envelope points, but it
must not be described as all-source domination. The following are descriptive
corpus directions (H minus F0; contamination / miss), not a shared macro
threshold or an acceptance rule:

| Horizon / independent point | AliMeeting | AMI |
| --- | --- | --- |
| 100 ms C envelope | `-1094.488 / -0.383355` | `-217.975 / -0.154117` |
| 100 ms M envelope | `-1094.488 / -0.383355` | `-191.428 / -0.167652` |
| 300 ms C envelope | `-205.001 / -0.055630` | `-76.846 / -0.056029` |
| 300 ms M envelope | `-205.001 / -0.055630` | `-62.195 / -0.056825` |
| 500 ms C envelope | `-88.886 / -0.067000` | `+189.849 / +0.075922` |
| 500 ms M envelope | `-88.886 / -0.067000` | `+194.607 / +0.075126` |

At 100 ms, for example, `ami_ES2009a` has positive contamination deltas
(`+172.351` at C and `+109.658` at M), so the result is not dominated-all or
uniformly favorable. Leave-one-meeting-out macro results remain favorable at
100 ms, but that does not erase the per-source and timing residuals.

### 4. What do uncertainty and timing say?

The intervals below are the 2,000-replicate paired-source bootstrap meeting-mean
intervals. They are source-mean intervals, not pooled-rate intervals and not
macro CIs. Each cell is contamination CI / miss CI; raw and calibrated intervals
are shown as `raw; calibrated`.

| Horizon / point | Raw; calibrated C delta CI | Raw; calibrated miss delta CI |
| --- | --- | --- |
| 100 C | `[-723.700, -119.918]; [-723.162, -139.979]` | `[-0.237296, -0.081017]; [-0.248356, -0.085838]` |
| 100 M | `[-717.566, -83.631]; [-707.277, -88.950]` | `[-0.263764, -0.050912]; [-0.270812, -0.053899]` |
| 300 C | `[-39.073, 120.870]; [-36.020, 124.895]` | `[+0.002945, +0.065197]; [+0.003426, +0.066205]` |
| 300 M | `[-43.730, 125.851]; [-42.514, 128.290]` | `[+0.003476, +0.065709]; [+0.004756, +0.066265]` |
| 500 C | `[+70.833, +355.377]; [+72.430, +356.825]` | `[+0.025593, +0.153624]; [+0.025724, +0.150599]` |
| 500 M | `[+67.802, +345.655]; [+74.700, +347.867]` | `[+0.025623, +0.151012]; [+0.025022, +0.145931]` |

The declared timing criterion is p90 delay no more than F0 plus 80 ms. At the
100 ms selected points, p90 violations are AliMeeting R1019 `+157 ms` at C,
AliMeeting R8009 `+302 ms` at C, and AliMeeting R8009 `+199 ms` at M. At 300 ms,
R1019 is `+169 ms` at both C and M. No 500 ms selected point has a violation.
There is no global pooled p90 claim: source-mean timing is descriptive only.
No meeting or topology cutoff was invented.

### 5. Is this only a calibration result?

No. Calibration improves proper scoring on the TRAIN-CALIB material, but the raw
and calibrated event metrics at all selected H points are identical. Therefore
the 100 ms event gain is not a calibration-only artifact.
These are frame-level calibration diagnostics, not product-event metrics. The
original decision report did not have DEV F0 AP; the durable publication bundle
now computes it posthoc and does not retroactively treat it as preregistered
evidence.

On 169,951 TRAIN-CALIB frames:

- candidate AP is `0.497992215`; raw/calibrated NLL is
  `0.402203033 / 0.122659548`; raw/calibrated Brier is
  `0.117269873 / 0.035322329`;
- F0 AP is `0.117604962`; raw/calibrated NLL is
  `1.695953250 / 0.196998168`; raw/calibrated Brier is
  `0.235684938 / 0.050974547`.

The posthoc DEV ranking analysis uses valid-and-mapped frames only. Pooled
candidate AP is `0.487681950383668` and pooled F0 AP is
`0.150603849645612` over 195,375 frames. Candidate source-macro AP is AMI
`0.491683249822344`, AliMeeting `0.430875816411725`, and all ten
`0.473441019799158`; corresponding F0 source-macro AP is AMI
`0.144531097981704`, AliMeeting `0.174723625454627`, and all ten
`0.153588856223581`. Pooled and source-macro quantities are different
estimands, and these ranking summaries are not causal or deployment claims.

The durable [H7301 persistence analysis](results/issue-121-h7301-persistence-v1/PERSISTENCE_ANALYSIS.md)
reproduces matched/unmatched positive event-generating run traces at the exact
selected global thresholds and a held H500 threshold. The traces are
descriptive evidence about persistence/confirmation interaction; unmatched
events can be timing or matching failures, and no source-level cause is proven.

### 6. What changed relative to Issue 107?

The exact Issue 107 negative remains unchanged. Its sealed decision reported H
versus F0 pooled primary metrics as contamination `2154.20` versus `1902.41`,
false cuts/hour `75.01` versus `45.85`, and missed replacements/hour `252.41`
versus `218.80`: H was worse on all three. The unmodified historical record is
`experiments/psem_sortformer_adaptation_depth/results/issue-107-a40-1334720a-01/LEAN_ADAPTATION_DECISION.md`.

H7301 is a separate state-corrected bounded evaluation. Its corrected gain is
limited to 100 ms; it does not revise the Issue 107 negative, establish a general
adaptation direction, or justify a depth progression.

### 7. What is known about train fit, representation, topology, and cause?

TRAIN-FIT AP/NLL/Brier diagnostics are absent. The export records only 2,261 loss
chunks, 142 GPU steps, and `loss_sum=170.385308`. Loss is not ranking evidence;
TRAIN-FIT quality must not be inferred from loss. The 100 ms event result therefore
does not prove an event-objective or representation cause.

All 10 DEV sources mapped 100% of their scored frames. The real AMI EN2009d
indexed path covered 6,428 intervals and 532 oracle ranges with exact equality.
A nested synthetic mismatch was out of contract for the contiguous timeline and is
not a real-data defect. No slot trajectory/drift, fragmentation/continuity
attribution, F0 topology comparator, or leave-one-topology-out result was
established. Meeting, topology, and timing residuals are reported descriptively;
no one-meeting or one-topology dominance cutoff is inferred.

### 8. Is there seed or depth evidence for a follow-up?

No. Seed 7302 was not run. The shallowest supported operational choice remains
F0; no learned candidate was accepted. H7302 was not chosen because the known
horizon and timing failures are not meaningful ordinary seed uncertainty; this
weakness does not imply an automatic depth progression. The
representation-limited-evidence requirement applies to T2 only. The adaptation
depth sequence is closed.

### 9. Were any confirmation, T2, TA, or EVAL arms authorized?

No. No candidate was selected, so EVAL was not opened. No H7302, T2, TA, or
confirmation arm was opened automatically. Confirmation was not chosen because
the known horizon/timing failures and the insufficiently established
representation-limited basis make escalation unjustified; this is not a claim
that H confirmation has the T2-only representation-evidence requirement. No
native causal or KD deployment was chosen. Under Issue 121 §2 and §18.4, the
conservative STOP is cheaper and safer than resolving ambiguous defaults by
inventing another arm.

### 10. What exactly was run and what artifacts bind the result?

The immutable GPU export binding names 53 FIT sources, while the actual frozen
NPZ payload is 21 files: 11 CALIB and 10 DEV. GPU training metadata records 142
steps. The CPU
command was:

```text
uv run python -u -m experiments.psem_state_corrected_adaptation_gate.run_h_arm \
  --command postprocess \
  --export-dir .cache/issue-121-h-profile-staging/rerun-export/gpu_export \
  --out-dir .cache/issue-121-h-profile-staging/rerun-postprocess \
  --workers 8
```

The successful CPU run used one ordered `spawn` pool, eight workers, all thread
caps set to one, 42,857 score tasks, and 8,917,633 reused primitives. The observed
CPU wall time was 29m18s; the wave receipt reports 1,693.8040342 seconds. The
process exited 0 with no retained descendants. There is no invented exact start
or end timestamp.

Immutable/source identities:

- manifest SHA-256:
  `7ff72366ffa6182e5b3ef7824507294f8dc66073c99287744039a6a7701bc131`
- GPU/source code hash:
  `a3d9003a76ea167c33c644f1e3d15862e0181ed633a0378c91cb6d0fccaa263a`
- H postprocess source hash:
  `674a3639e4245d118960875fbc38a091e0ee9e995b9d8137664ae99417779ca5`
- trained head hash:
  `bb5029da54b01a84763d0513a544cdbcd99bb1592f73b9eef71a1916e59aea3f`
- checkpoint hash:
  `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`
- input hash:
  `72e37c1a2612a365cd575bc7ca36646690ac77c1ce04066c1dbf29b0189c0ed8`
- partition hash:
  `e5190d423c2b038668831e558cc6771184d7e7b1c06b6ef6934f595fbffbdf6a`
- weights hash:
  `225f41be27b43450e9b9062349fed52b325d8faedcee23038aa1a511d16166f5`

The canonical postprocess outputs and posthoc persistence evidence are packaged
in the [durable H7301 publication bundle](results/issue-121-h7301-persistence-v1/README.md).
The canonical files and source-byte hashes are:

- [`calibration_metrics.json`](results/issue-121-h7301-persistence-v1/canonical/calibration_metrics.json):
  `c1a6b05ff5da589c793f641f604f3f8150607e58224c65e13f84c7dc4308adf8`
- [`dev_frontier.json.gz`](results/issue-121-h7301-persistence-v1/canonical/dev_frontier.json.gz):
  deterministic gzip of the exact `dev_frontier.json`, whose SHA-256 is
  `11b8195bbcf9a301a1524729a659762e834a89bd2a5ecbeaa365c2343a6f0345`
- [`gate1_diagnostics.json`](results/issue-121-h7301-persistence-v1/canonical/gate1_diagnostics.json):
  `8df0d02a98f996178fc755a5805353a5fbc3399b2b4b6cfa11753b4bdf9dad84`
- [`gate1_decision_evidence.md`](results/issue-121-h7301-persistence-v1/canonical/gate1_decision_evidence.md):
  `6f4dd8eff971076f0a302e366700bf64bfcb21509453d4d558d2be08db48dcab`

The durable bundle also contains the immutable
[`gpu_export_manifest.json`](results/issue-121-h7301-persistence-v1/export/gpu_export/gpu_export_manifest.json),
training metrics, all 11 CALIB and 10 DEV numeric NPZ files, a per-file
SHA-256 manifest, and the
[persistence analysis](results/issue-121-h7301-persistence-v1/PERSISTENCE_ANALYSIS.md).
The compressed frontier is 43.9 MB; its original 1.06 GB size and hash are
recorded in `bundle_manifest.json`, together with the reader recipe. No audio,
transcripts, PII, checkpoints, credentials, or process logs are in the bundle.
All result bindings equal the export manifest binding. The diagnostics say
`human_adjudication_required=true`, `gate_receipt_emitted=false`,
`t2_opened=false`, and `eval_opened=false`.

For cost accounting, earlier failed/aborted CPU attempts are retained as process
receipts only: 6,840 s (about 114 m), 1,346 s (22m26s), 1,335 s (22m15s), and a
reported 1h3m. These are observed process times, approximately 114+22+22+63
minutes cumulative; no dollar cost is inferred. The successful 29m18s run is
separate. No source or result JSON was mutated for this disposition.

**Review status:** execution contract verified; not a formal immutable candidate;
not terminal accepted. The scientific decision is recorded and finalized. Formal
commit review remains outstanding; scientific gate analysis is not pending.

STOP / inconclusive
