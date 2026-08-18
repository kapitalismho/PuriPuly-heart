# Natural topology census

This is the frozen-contract pre-split census of every accepted natural candidate meeting. No model prediction, model score, official model result, or model training participated.

Contract: `psem-handoff-v0` (`frozen_after_annotation_only_calibration`)

Split roles remain unassigned until the identity graph and pretrained-checkpoint overlap audit are complete. Component counts are therefore pending and the raw-pool lower-bound audit is not split feasibility evidence.

## Overall

| Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Masked/ambiguous fraction |
|---:|---:|---:|---:|---:|
| 28 | 15.886353 | 10.728529 | 2.799398 | 0.0 |

## Mask diagnostics

- Actual handoff/relation transitions: `12611`
- Masked handoff/relation transitions: `3992` (`0.31654904`)
- Masked transition reasons: `{"complex_overlap_transition": 1512, "continuity_unknown": 1450, "mixed_unresolved_transition": 1030}`
- Diagnostic masked region counts: `{"complex_overlap_region": 3057, "complex_overlap_transition": 5428, "overlap_to_silence_unresolved": 2}`

## Exclusive primary topology counts

| Primary topology | Count |
|---|---:|
| `short_backchannel_return` | 386 |
| `overlap_takeover` | 1687 |
| `overlap_return` | 3187 |
| `silence_gap_different_speaker_handoff` | 1428 |
| `same_speaker_silence_gap_resume` | 948 |
| `clean_direct_different_speaker_handoff` | 271 |

## By corpus

| Corpus | Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Primary episodes |
|---|---:|---:|---:|---:|---:|
| AMI | 20 | 11.694456 | 7.695746 | 1.96097 | 5137 |
| AliMeeting | 8 | 4.191897 | 3.032783 | 0.838428 | 2770 |

## By meeting

| Corpus | Session | Hours | Direct | Gap handoff | Same gap | Overlap return | Overlap takeover | Short return | Micro gap | Micro overlap | Masked T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AliMeeting | R8001_M8004 | 0.43082 | 2 | 18 | 16 | 169 | 59 | 6 | 3 | 20 | 116 |
| AliMeeting | R8003_M8001 | 0.574364 | 9 | 52 | 55 | 161 | 49 | 16 | 7 | 11 | 139 |
| AliMeeting | R8007_M8010 | 0.514975 | 5 | 18 | 13 | 141 | 51 | 5 | 4 | 70 | 281 |
| AliMeeting | R8007_M8011 | 0.516387 | 16 | 47 | 15 | 135 | 80 | 15 | 11 | 38 | 163 |
| AliMeeting | R8008_M8013 | 0.620972 | 17 | 137 | 21 | 143 | 84 | 28 | 9 | 11 | 141 |
| AliMeeting | R8009_M8018 | 0.458746 | 11 | 86 | 83 | 103 | 37 | 39 | 6 | 12 | 33 |
| AliMeeting | R8009_M8019 | 0.545742 | 8 | 30 | 258 | 55 | 53 | 20 | 0 | 3 | 56 |
| AliMeeting | R8009_M8020 | 0.529891 | 11 | 36 | 191 | 102 | 35 | 29 | 6 | 4 | 33 |
| AMI | EN2001d | 0.985105 | 14 | 109 | 0 | 143 | 127 | 7 | 9 | 16 | 172 |
| AMI | EN2002c | 0.825627 | 13 | 59 | 1 | 178 | 107 | 6 | 6 | 13 | 163 |
| AMI | EN2006a | 0.979304 | 19 | 160 | 31 | 160 | 135 | 25 | 16 | 43 | 434 |
| AMI | EN2009d | 1.478993 | 12 | 79 | 4 | 380 | 175 | 13 | 17 | 40 | 287 |
| AMI | ES2002b | 0.633265 | 9 | 15 | 6 | 171 | 43 | 2 | 4 | 19 | 98 |
| AMI | ES2003a | 0.316601 | 1 | 15 | 4 | 21 | 10 | 4 | 1 | 9 | 55 |
| AMI | ES2004a | 0.291487 | 2 | 13 | 3 | 58 | 17 | 7 | 5 | 15 | 98 |
| AMI | ES2014a | 0.31917 | 0 | 20 | 10 | 37 | 15 | 2 | 2 | 16 | 77 |
| AMI | ES2015d | 0.536545 | 14 | 38 | 18 | 87 | 61 | 15 | 2 | 48 | 204 |
| AMI | ES2016a | 0.384497 | 6 | 16 | 13 | 48 | 16 | 4 | 3 | 10 | 107 |
| AMI | IS1008a | 0.262176 | 5 | 24 | 0 | 45 | 16 | 9 | 4 | 2 | 27 |
| AMI | IS1009a | 0.233009 | 3 | 18 | 1 | 48 | 23 | 6 | 4 | 17 | 55 |
| AMI | TS3003b | 0.613973 | 4 | 38 | 68 | 52 | 31 | 15 | 4 | 17 | 124 |
| AMI | TS3004a | 0.373701 | 8 | 49 | 6 | 47 | 38 | 8 | 7 | 15 | 136 |
| AMI | TS3005b | 0.677754 | 20 | 85 | 27 | 127 | 81 | 18 | 14 | 40 | 173 |
| AMI | TS3006a | 0.348018 | 9 | 35 | 13 | 79 | 44 | 11 | 5 | 21 | 117 |
| AMI | TS3007a | 0.44704 | 9 | 51 | 23 | 58 | 47 | 23 | 11 | 18 | 123 |
| AMI | TS3008b | 0.644711 | 10 | 50 | 30 | 135 | 71 | 17 | 3 | 26 | 140 |
| AMI | TS3009b | 0.683348 | 8 | 65 | 19 | 114 | 78 | 20 | 4 | 41 | 213 |
| AMI | TS3012c | 0.66013 | 26 | 65 | 19 | 190 | 104 | 16 | 14 | 38 | 227 |

## Raw candidate-pool lower bound

| Criterion | Observed raw pool | Combined hard role minimum | Raw deficit/status |
|---|---:|---:|---:|
| Scored natural hours | 15.886353 hours | 33.0 hours | 17.113647 hours |
| Independent meetings | 28 upper bound | 22 | component audit pending |
| Stable singleton hours | 10.728529 hours | 10.0 hours | 0.0 hours |
| Ongoing overlap minutes | 167.963883 minutes | 75.0 minutes | 0.0 minutes |
| `short_backchannel_return` | 386 | 100 | 0 |
| `overlap_takeover` | 1687 | 120 | 0 |
| `overlap_return` | 3187 | 120 | 0 |
| `silence_gap_different_speaker_handoff` | 1428 | 240 | 0 |
| `same_speaker_silence_gap_resume` | 948 | 240 | 0 |
| `clean_direct_different_speaker_handoff` | 271 | 120 | 0 |

The scored-hour deficit is an acquisition blocker. The meeting count is only an upper bound until connected identity components are audited. Zero raw-pool deficits do not prove component-safe role allocation. No topology substitutes for another, and no threshold, count, or natural-data requirement is weakened.

Topology manifest SHA-256: `9ffe8d6f6e17bf0fc4130c4b40dcb4a9da1a1dc034f854d2ff0d77619613c6f4`
