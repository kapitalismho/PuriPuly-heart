# Natural topology census

This is the frozen-contract pre-split census of every accepted natural candidate meeting. No model prediction, model score, official model result, or model training participated.

Contract: `psem-handoff-v0` (`frozen_after_annotation_only_calibration`)

Split roles remain unassigned until the identity graph and pretrained-checkpoint overlap audit are complete. Component counts are therefore pending and the raw-pool lower-bound audit is not split feasibility evidence.

## Overall

| Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Masked/ambiguous fraction |
|---:|---:|---:|---:|---:|
| 76 | 39.790016 | 27.179784 | 6.194047 | 0.0 |

## Mask diagnostics

- Actual handoff/relation transitions: `28782`
- Masked handoff/relation transitions: `9759` (`0.33906608`)
- Masked transition reasons: `{"complex_overlap_transition": 3423, "continuity_unknown": 4089, "mixed_unresolved_transition": 2247}`
- Diagnostic masked region counts: `{"complex_overlap_region": 6754, "complex_overlap_transition": 11913, "overlap_to_silence_unresolved": 7}`

## Exclusive primary topology counts

| Primary topology | Count |
|---|---:|
| `short_backchannel_return` | 857 |
| `overlap_takeover` | 3564 |
| `overlap_return` | 7115 |
| `silence_gap_different_speaker_handoff` | 3297 |
| `same_speaker_silence_gap_resume` | 1772 |
| `clean_direct_different_speaker_handoff` | 667 |

## By corpus

| Corpus | Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Primary episodes |
|---|---:|---:|---:|---:|---:|
| AMI | 68 | 35.598119 | 24.147 | 5.355619 | 14502 |
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
| AMI | EN2003a | 0.622299 | 6 | 58 | 23 | 136 | 42 | 12 | 4 | 16 | 139 |
| AMI | EN2004a | 0.957132 | 16 | 37 | 15 | 231 | 99 | 12 | 9 | 52 | 251 |
| AMI | EN2006a | 0.979304 | 19 | 160 | 31 | 160 | 135 | 25 | 16 | 43 | 434 |
| AMI | EN2009d | 1.478993 | 12 | 79 | 4 | 380 | 175 | 13 | 17 | 40 | 287 |
| AMI | ES2002b | 0.633265 | 9 | 15 | 6 | 171 | 43 | 2 | 4 | 19 | 98 |
| AMI | ES2003a | 0.316601 | 1 | 15 | 4 | 21 | 10 | 4 | 1 | 9 | 55 |
| AMI | ES2004a | 0.291487 | 2 | 13 | 3 | 58 | 17 | 7 | 5 | 15 | 98 |
| AMI | ES2005a | 0.132744 | 1 | 6 | 3 | 20 | 10 | 2 | 0 | 5 | 25 |
| AMI | ES2005b | 0.642572 | 12 | 35 | 21 | 136 | 50 | 24 | 5 | 32 | 144 |
| AMI | ES2005c | 0.637751 | 5 | 66 | 17 | 134 | 73 | 18 | 4 | 49 | 163 |
| AMI | ES2005d | 0.480883 | 8 | 44 | 16 | 89 | 44 | 8 | 7 | 29 | 120 |
| AMI | ES2006a | 0.356761 | 6 | 13 | 9 | 56 | 19 | 7 | 0 | 17 | 88 |
| AMI | ES2006b | 0.606424 | 6 | 25 | 11 | 101 | 51 | 6 | 3 | 28 | 126 |
| AMI | ES2006c | 0.606024 | 9 | 23 | 12 | 141 | 51 | 8 | 3 | 31 | 140 |
| AMI | ES2006d | 0.546513 | 8 | 55 | 14 | 100 | 68 | 9 | 11 | 70 | 207 |
| AMI | ES2007a | 0.335084 | 8 | 34 | 10 | 35 | 31 | 13 | 8 | 20 | 117 |
| AMI | ES2007b | 0.468151 | 3 | 35 | 19 | 84 | 43 | 7 | 4 | 15 | 95 |
| AMI | ES2007c | 0.660391 | 12 | 61 | 19 | 100 | 49 | 18 | 4 | 33 | 172 |
| AMI | ES2007d | 0.347087 | 14 | 39 | 12 | 65 | 38 | 14 | 5 | 32 | 106 |
| AMI | ES2008a | 0.289822 | 3 | 19 | 19 | 43 | 9 | 4 | 2 | 4 | 65 |
| AMI | ES2008b | 0.619905 | 7 | 54 | 31 | 81 | 32 | 18 | 6 | 23 | 138 |
| AMI | ES2008c | 0.584061 | 9 | 38 | 31 | 80 | 44 | 6 | 4 | 11 | 110 |
| AMI | ES2008d | 0.729396 | 14 | 110 | 40 | 132 | 62 | 24 | 14 | 46 | 216 |
| AMI | ES2009a | 0.389502 | 13 | 37 | 15 | 73 | 29 | 5 | 5 | 29 | 94 |
| AMI | ES2009b | 0.398702 | 7 | 19 | 8 | 65 | 20 | 2 | 5 | 15 | 79 |
| AMI | ES2009c | 0.543591 | 16 | 42 | 24 | 106 | 34 | 5 | 1 | 43 | 95 |
| AMI | ES2009d | 0.587484 | 23 | 65 | 14 | 121 | 53 | 9 | 5 | 45 | 171 |
| AMI | ES2010a | 0.178969 | 5 | 15 | 2 | 25 | 16 | 3 | 1 | 10 | 38 |
| AMI | ES2010b | 0.486661 | 18 | 37 | 8 | 68 | 38 | 10 | 7 | 28 | 107 |
| AMI | ES2010c | 0.510489 | 11 | 33 | 23 | 94 | 42 | 6 | 6 | 35 | 108 |
| AMI | ES2011a | 0.309401 | 5 | 17 | 6 | 37 | 21 | 3 | 2 | 15 | 90 |
| AMI | ES2011b | 0.439241 | 3 | 22 | 18 | 63 | 40 | 4 | 2 | 19 | 104 |
| AMI | ES2011d | 0.550646 | 15 | 54 | 22 | 60 | 80 | 22 | 6 | 26 | 147 |
| AMI | ES2012a | 0.306815 | 4 | 8 | 6 | 37 | 13 | 0 | 2 | 6 | 53 |
| AMI | ES2012b | 0.62213 | 9 | 22 | 42 | 77 | 33 | 7 | 3 | 14 | 120 |
| AMI | ES2012c | 0.613526 | 3 | 23 | 26 | 105 | 43 | 6 | 4 | 31 | 126 |
| AMI | ES2012d | 0.263793 | 2 | 12 | 12 | 37 | 14 | 5 | 0 | 26 | 84 |
| AMI | ES2013a | 0.229215 | 3 | 19 | 3 | 23 | 5 | 6 | 1 | 2 | 65 |
| AMI | ES2013b | 0.591286 | 9 | 43 | 25 | 76 | 25 | 11 | 2 | 9 | 131 |
| AMI | ES2013c | 0.655034 | 11 | 36 | 31 | 80 | 40 | 8 | 4 | 23 | 129 |
| AMI | ES2013d | 0.527378 | 8 | 53 | 23 | 63 | 31 | 9 | 4 | 17 | 138 |
| AMI | ES2014a | 0.31917 | 0 | 20 | 10 | 37 | 15 | 2 | 2 | 16 | 77 |
| AMI | ES2015d | 0.536545 | 14 | 38 | 18 | 87 | 61 | 15 | 2 | 48 | 204 |
| AMI | ES2016a | 0.384497 | 6 | 16 | 13 | 48 | 16 | 4 | 3 | 10 | 107 |
| AMI | IS1007a | 0.268333 | 7 | 28 | 10 | 45 | 29 | 8 | 2 | 10 | 70 |
| AMI | IS1007b | 0.361852 | 4 | 31 | 15 | 80 | 33 | 12 | 4 | 14 | 88 |
| AMI | IS1007c | 0.586991 | 7 | 25 | 23 | 120 | 46 | 9 | 6 | 21 | 110 |
| AMI | IS1007d | 0.563627 | 7 | 50 | 16 | 111 | 50 | 15 | 4 | 24 | 140 |
| AMI | IS1008a | 0.262176 | 5 | 24 | 0 | 45 | 16 | 9 | 4 | 2 | 27 |
| AMI | IS1009a | 0.233009 | 3 | 18 | 1 | 48 | 23 | 6 | 4 | 17 | 55 |
| AMI | TS3003b | 0.613973 | 4 | 38 | 68 | 52 | 31 | 15 | 4 | 17 | 124 |
| AMI | TS3004a | 0.373701 | 8 | 49 | 6 | 47 | 38 | 8 | 7 | 15 | 136 |
| AMI | TS3005b | 0.677754 | 20 | 85 | 27 | 127 | 81 | 18 | 14 | 40 | 173 |
| AMI | TS3006a | 0.348018 | 9 | 35 | 13 | 79 | 44 | 11 | 5 | 21 | 117 |
| AMI | TS3007a | 0.44704 | 9 | 51 | 23 | 58 | 47 | 23 | 11 | 18 | 123 |
| AMI | TS3008b | 0.644711 | 10 | 50 | 30 | 135 | 71 | 17 | 3 | 26 | 140 |
| AMI | TS3009b | 0.683348 | 8 | 65 | 19 | 114 | 78 | 20 | 4 | 41 | 213 |
| AMI | TS3010a | 0.289173 | 5 | 24 | 4 | 22 | 12 | 1 | 0 | 3 | 81 |
| AMI | TS3010b | 0.579301 | 2 | 40 | 28 | 39 | 23 | 8 | 7 | 10 | 132 |
| AMI | TS3010c | 0.595 | 5 | 76 | 17 | 55 | 39 | 15 | 6 | 15 | 167 |
| AMI | TS3010d | 0.533528 | 9 | 73 | 19 | 51 | 38 | 16 | 4 | 15 | 187 |
| AMI | TS3011a | 0.419378 | 4 | 23 | 7 | 70 | 28 | 5 | 3 | 8 | 78 |
| AMI | TS3011b | 0.615058 | 14 | 48 | 16 | 116 | 60 | 11 | 9 | 24 | 134 |
| AMI | TS3011c | 0.665278 | 11 | 68 | 17 | 144 | 66 | 16 | 8 | 14 | 121 |
| AMI | TS3011d | 0.599281 | 9 | 74 | 22 | 101 | 61 | 24 | 10 | 33 | 158 |
| AMI | TS3012c | 0.66013 | 26 | 65 | 19 | 190 | 104 | 16 | 14 | 38 | 227 |

## Raw candidate-pool lower bound

| Criterion | Observed raw pool | Combined hard role minimum | Raw deficit/status |
|---|---:|---:|---:|
| Scored natural hours | 39.790016 hours | 33.0 hours | 0.0 hours |
| Independent meetings | 76 upper bound | 22 | component audit pending |
| Stable singleton hours | 27.179784 hours | 10.0 hours | 0.0 hours |
| Ongoing overlap minutes | 371.6428 minutes | 75.0 minutes | 0.0 minutes |
| `short_backchannel_return` | 857 | 100 | 0 |
| `overlap_takeover` | 3564 | 120 | 0 |
| `overlap_return` | 7115 | 120 | 0 |
| `silence_gap_different_speaker_handoff` | 3297 | 240 | 0 |
| `same_speaker_silence_gap_resume` | 1772 | 240 | 0 |
| `clean_direct_different_speaker_handoff` | 667 | 120 | 0 |

The aggregate scored-hour lower bound passes; role-specific allocation remains unproven until component assignment. The meeting count is only an upper bound until connected identity components are audited. Zero raw-pool deficits do not prove component-safe role allocation. No topology substitutes for another, and no threshold, count, or natural-data requirement is weakened.

Topology manifest SHA-256: `235e0c6995041d1b967907c100dc4460cbf42fc2b4574a91e7eb765f3a3f89f2`
