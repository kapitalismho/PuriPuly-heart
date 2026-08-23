# Natural topology census and component split

This is the frozen-contract census of every accepted natural candidate meeting plus the deterministic connected-component assignment selected for TRAIN, DEV, and EVAL. No model prediction, model score, official model result, or model training participated.

Contract: `psem-handoff-v1` (`forced_alignment_reference_contract`)

The identity graph and pinned WavLM overlap audit cover all 93 sources in 57 components. `split_manifest.json` assigns every component exactly once in EVAL, then DEV, then TRAIN order. The split reaches the integer global upper bound for minimum normalized topology slack and passes all 37 role-specific hard gates. Dataset freeze `PSEM-STRATEGY-DATA-v2` will bind these artifacts at the final freeze checkpoint; final preflight is recorded separately.

## Overall

| Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Masked/ambiguous fraction |
|---:|---:|---:|---:|---:|
| 93 | 48.571701 | 29.633316 | 3.848209 | 0.0 |

## Mask diagnostics

- Actual handoff/relation transitions: `75116`
- Masked handoff/relation transitions: `22563` (`0.30037542`)
- Masked transition reasons: `{"ambiguous_nonlexical_vocalization_crossing": 6821, "complex_overlap_transition": 1644, "continuity_unknown": 5328, "initial_start": 15, "mixed_unresolved_transition": 8755}`
- Diagnostic masked region counts: `{"ambiguous_nonlexical_vocalization_region": 7730, "complex_overlap_region": 5185, "complex_overlap_transition": 8995, "overlap_to_silence_unresolved": 15}`

## Exclusive primary topology counts

| Primary topology | Count |
|---|---:|
| `short_backchannel_return` | 3330 |
| `overlap_takeover` | 3297 |
| `overlap_return` | 5372 |
| `silence_gap_different_speaker_handoff` | 6160 |
| `same_speaker_silence_gap_resume` | 28652 |
| `clean_direct_different_speaker_handoff` | 1181 |

## By corpus

| Corpus | Sessions | Scored hours | Stable singleton hours | Ongoing overlap hours | Primary episodes |
|---|---:|---:|---:|---:|---:|
| AMI | 68 | 35.598119 | 20.672817 | 2.137753 | 32420 |
| AliMeeting | 25 | 12.973582 | 8.960499 | 1.710456 | 15572 |

## By selected role

| Role | Components | Meetings | Scored h | Stable singleton h | Ongoing overlap h | Known speakers | Corpora |
|---|---:|---:|---:|---:|---:|---:|---|
| TRAIN | 37 | 64 | 32.458378 | 19.516395 | 2.846832 | 141 | AMI + AliMeeting |
| DEV | 7 | 10 | 6.105825 | 4.034645 | 0.419253 | 22 | AMI + AliMeeting |
| EVAL | 13 | 19 | 10.007499 | 6.082276 | 0.582124 | 40 | AMI + AliMeeting |

| Role | Direct | Gap handoff | Same gap | Overlap return | Overlap takeover | Short return |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 855 | 4057 | 18506 | 3745 | 2423 | 2162 |
| DEV | 129 | 631 | 4282 | 709 | 375 | 377 |
| EVAL | 197 | 1472 | 5864 | 918 | 499 | 791 |

The exact source and component membership, input hashes, search seed/version, objective order, leakage audit, hard-gate observations, and assignment hash are authoritative in `split_manifest.json`.

## By meeting

| Corpus | Session | Hours | Direct | Gap handoff | Same gap | Overlap return | Overlap takeover | Short return | Micro gap | Micro overlap | Masked T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AliMeeting | R0004_M0012 | 0.537536 | 42 | 167 | 166 | 131 | 88 | 151 | 46 | 94 | 386 |
| AliMeeting | R0005_M0035 | 0.578611 | 38 | 128 | 293 | 115 | 59 | 89 | 38 | 75 | 323 |
| AliMeeting | R0015_M0138 | 0.483549 | 41 | 106 | 114 | 88 | 142 | 61 | 32 | 125 | 373 |
| AliMeeting | R0015_M0139 | 0.400812 | 26 | 70 | 51 | 83 | 117 | 24 | 30 | 165 | 424 |
| AliMeeting | R0020_M0168 | 0.536289 | 38 | 141 | 182 | 119 | 111 | 85 | 21 | 86 | 340 |
| AliMeeting | R1019_M1928 | 0.49925 | 15 | 72 | 334 | 55 | 50 | 31 | 5 | 5 | 125 |
| AliMeeting | R1019_M1946 | 0.473483 | 13 | 111 | 254 | 27 | 64 | 30 | 8 | 8 | 136 |
| AliMeeting | R1019_M1950 | 0.499617 | 6 | 44 | 387 | 19 | 40 | 10 | 2 | 3 | 55 |
| AliMeeting | R1019_M1960 | 0.495433 | 6 | 94 | 409 | 14 | 22 | 28 | 9 | 5 | 51 |
| AliMeeting | R1021_M1940 | 0.491283 | 17 | 111 | 378 | 60 | 40 | 88 | 20 | 16 | 109 |
| AliMeeting | R1021_M1944 | 0.487542 | 5 | 88 | 376 | 36 | 42 | 35 | 7 | 11 | 97 |
| AliMeeting | R1021_M4073 | 0.49275 | 2 | 42 | 580 | 7 | 9 | 4 | 1 | 0 | 48 |
| AliMeeting | R1021_M4080 | 0.492608 | 5 | 82 | 391 | 51 | 14 | 31 | 6 | 3 | 100 |
| AliMeeting | R2001_M2205 | 0.656793 | 47 | 144 | 217 | 199 | 144 | 70 | 29 | 76 | 295 |
| AliMeeting | R2001_M2206 | 0.645189 | 52 | 141 | 160 | 217 | 160 | 77 | 30 | 60 | 367 |
| AliMeeting | R2105_M3318 | 0.513434 | 11 | 74 | 445 | 67 | 35 | 80 | 18 | 23 | 146 |
| AliMeeting | R2108_M3206 | 0.497506 | 3 | 70 | 404 | 18 | 11 | 15 | 2 | 1 | 42 |
| AliMeeting | R8001_M8004 | 0.43082 | 8 | 36 | 201 | 193 | 49 | 25 | 12 | 55 | 218 |
| AliMeeting | R8003_M8001 | 0.574364 | 18 | 62 | 358 | 127 | 48 | 48 | 22 | 45 | 219 |
| AliMeeting | R8007_M8010 | 0.514975 | 29 | 35 | 95 | 167 | 94 | 35 | 16 | 184 | 469 |
| AliMeeting | R8007_M8011 | 0.516387 | 19 | 58 | 238 | 113 | 86 | 39 | 25 | 73 | 245 |
| AliMeeting | R8008_M8013 | 0.620972 | 34 | 163 | 337 | 89 | 52 | 80 | 18 | 22 | 232 |
| AliMeeting | R8009_M8018 | 0.458746 | 8 | 101 | 317 | 80 | 24 | 50 | 11 | 11 | 81 |
| AliMeeting | R8009_M8019 | 0.545742 | 3 | 38 | 614 | 49 | 37 | 36 | 8 | 8 | 119 |
| AliMeeting | R8009_M8020 | 0.529891 | 5 | 55 | 534 | 74 | 25 | 30 | 9 | 5 | 89 |
| AMI | EN2001d | 0.985105 | 20 | 152 | 519 | 73 | 78 | 60 | 15 | 22 | 368 |
| AMI | EN2002c | 0.825627 | 19 | 120 | 432 | 146 | 73 | 74 | 46 | 75 | 463 |
| AMI | EN2003a | 0.622299 | 11 | 62 | 379 | 60 | 23 | 39 | 13 | 23 | 297 |
| AMI | EN2004a | 0.957132 | 18 | 80 | 519 | 136 | 60 | 67 | 35 | 96 | 566 |
| AMI | EN2006a | 0.979304 | 21 | 173 | 320 | 95 | 69 | 57 | 36 | 62 | 653 |
| AMI | EN2009d | 1.478993 | 33 | 133 | 926 | 279 | 119 | 90 | 46 | 113 | 663 |
| AMI | ES2002b | 0.633265 | 8 | 45 | 449 | 83 | 31 | 63 | 24 | 39 | 242 |
| AMI | ES2003a | 0.316601 | 3 | 16 | 177 | 4 | 3 | 7 | 1 | 7 | 100 |
| AMI | ES2004a | 0.291487 | 6 | 21 | 154 | 37 | 16 | 28 | 12 | 20 | 148 |
| AMI | ES2005a | 0.132744 | 0 | 6 | 56 | 5 | 8 | 7 | 6 | 7 | 46 |
| AMI | ES2005b | 0.642572 | 15 | 63 | 433 | 39 | 29 | 47 | 16 | 40 | 348 |
| AMI | ES2005c | 0.637751 | 15 | 81 | 362 | 73 | 48 | 58 | 22 | 67 | 356 |
| AMI | ES2005d | 0.480883 | 12 | 53 | 230 | 33 | 23 | 33 | 22 | 41 | 255 |
| AMI | ES2006a | 0.356761 | 3 | 24 | 209 | 17 | 6 | 11 | 5 | 19 | 177 |
| AMI | ES2006b | 0.606424 | 12 | 43 | 418 | 57 | 42 | 31 | 12 | 41 | 289 |
| AMI | ES2006c | 0.606024 | 15 | 44 | 332 | 55 | 30 | 48 | 23 | 53 | 360 |
| AMI | ES2006d | 0.546513 | 14 | 70 | 220 | 57 | 60 | 33 | 25 | 63 | 364 |
| AMI | ES2007a | 0.335084 | 8 | 47 | 137 | 12 | 13 | 22 | 9 | 26 | 189 |
| AMI | ES2007b | 0.468151 | 2 | 57 | 282 | 36 | 16 | 17 | 5 | 16 | 244 |
| AMI | ES2007c | 0.660391 | 11 | 66 | 375 | 45 | 30 | 35 | 11 | 36 | 334 |
| AMI | ES2007d | 0.347087 | 6 | 61 | 174 | 31 | 17 | 20 | 11 | 24 | 228 |
| AMI | ES2008a | 0.289822 | 2 | 23 | 206 | 12 | 5 | 7 | 8 | 6 | 114 |
| AMI | ES2008b | 0.619905 | 13 | 71 | 427 | 42 | 12 | 35 | 17 | 31 | 214 |
| AMI | ES2008c | 0.584061 | 8 | 52 | 365 | 53 | 23 | 32 | 18 | 32 | 221 |
| AMI | ES2008d | 0.729396 | 15 | 120 | 426 | 67 | 39 | 44 | 29 | 43 | 362 |
| AMI | ES2009a | 0.389502 | 11 | 40 | 210 | 37 | 24 | 28 | 14 | 34 | 211 |
| AMI | ES2009b | 0.398702 | 6 | 33 | 252 | 46 | 8 | 13 | 6 | 13 | 170 |
| AMI | ES2009c | 0.543591 | 10 | 59 | 399 | 46 | 20 | 34 | 6 | 28 | 211 |
| AMI | ES2009d | 0.587484 | 16 | 80 | 303 | 49 | 28 | 39 | 19 | 43 | 326 |
| AMI | ES2010a | 0.178969 | 2 | 22 | 104 | 7 | 7 | 11 | 4 | 11 | 78 |
| AMI | ES2010b | 0.486661 | 11 | 59 | 323 | 28 | 15 | 24 | 15 | 32 | 204 |
| AMI | ES2010c | 0.510489 | 15 | 66 | 322 | 41 | 22 | 26 | 13 | 39 | 215 |
| AMI | ES2011a | 0.309401 | 2 | 29 | 160 | 18 | 10 | 13 | 3 | 21 | 168 |
| AMI | ES2011b | 0.439241 | 5 | 35 | 256 | 39 | 30 | 19 | 11 | 28 | 182 |
| AMI | ES2011d | 0.550646 | 9 | 87 | 239 | 29 | 36 | 27 | 14 | 30 | 296 |
| AMI | ES2012a | 0.306815 | 4 | 20 | 261 | 10 | 5 | 4 | 5 | 6 | 139 |
| AMI | ES2012b | 0.62213 | 2 | 44 | 459 | 46 | 6 | 15 | 9 | 16 | 265 |
| AMI | ES2012c | 0.613526 | 11 | 40 | 392 | 83 | 17 | 25 | 23 | 49 | 283 |
| AMI | ES2012d | 0.263793 | 4 | 25 | 149 | 38 | 9 | 11 | 10 | 29 | 147 |
| AMI | ES2013a | 0.229215 | 3 | 24 | 126 | 3 | 6 | 6 | 4 | 4 | 93 |
| AMI | ES2013b | 0.591286 | 10 | 48 | 454 | 35 | 16 | 25 | 10 | 21 | 256 |
| AMI | ES2013c | 0.655034 | 12 | 55 | 465 | 35 | 19 | 13 | 13 | 25 | 292 |
| AMI | ES2013d | 0.527378 | 6 | 53 | 354 | 28 | 18 | 25 | 6 | 24 | 242 |
| AMI | ES2014a | 0.31917 | 2 | 27 | 177 | 9 | 4 | 9 | 3 | 12 | 151 |
| AMI | ES2015d | 0.536545 | 25 | 89 | 215 | 58 | 49 | 39 | 35 | 60 | 340 |
| AMI | ES2016a | 0.384497 | 4 | 27 | 276 | 20 | 10 | 16 | 4 | 9 | 198 |
| AMI | IS1007a | 0.268333 | 6 | 35 | 117 | 17 | 10 | 13 | 8 | 9 | 140 |
| AMI | IS1007b | 0.361852 | 11 | 46 | 193 | 36 | 13 | 19 | 9 | 20 | 194 |
| AMI | IS1007c | 0.586991 | 12 | 56 | 360 | 70 | 15 | 38 | 21 | 27 | 250 |
| AMI | IS1007d | 0.563627 | 13 | 72 | 293 | 43 | 28 | 40 | 16 | 34 | 291 |
| AMI | IS1008a | 0.262176 | 4 | 21 | 149 | 15 | 6 | 19 | 10 | 8 | 88 |
| AMI | IS1009a | 0.233009 | 2 | 16 | 94 | 23 | 13 | 17 | 6 | 17 | 98 |
| AMI | TS3003b | 0.613973 | 7 | 56 | 562 | 27 | 11 | 11 | 9 | 13 | 254 |
| AMI | TS3004a | 0.373701 | 5 | 53 | 153 | 21 | 23 | 25 | 7 | 18 | 187 |
| AMI | TS3005b | 0.677754 | 17 | 93 | 422 | 75 | 48 | 59 | 27 | 56 | 334 |
| AMI | TS3006a | 0.348018 | 7 | 63 | 145 | 28 | 20 | 29 | 17 | 28 | 226 |
| AMI | TS3007a | 0.44704 | 10 | 51 | 222 | 28 | 19 | 37 | 16 | 9 | 206 |
| AMI | TS3008b | 0.644711 | 18 | 71 | 429 | 84 | 45 | 38 | 22 | 32 | 292 |
| AMI | TS3009b | 0.683348 | 23 | 84 | 400 | 69 | 58 | 39 | 24 | 75 | 452 |
| AMI | TS3010a | 0.289173 | 0 | 18 | 108 | 3 | 0 | 6 | 2 | 2 | 117 |
| AMI | TS3010b | 0.579301 | 6 | 45 | 372 | 18 | 1 | 14 | 4 | 9 | 239 |
| AMI | TS3010c | 0.595 | 7 | 70 | 295 | 18 | 13 | 26 | 6 | 13 | 280 |
| AMI | TS3010d | 0.533528 | 4 | 63 | 224 | 16 | 10 | 24 | 10 | 10 | 322 |
| AMI | TS3011a | 0.419378 | 8 | 24 | 303 | 21 | 10 | 32 | 10 | 16 | 176 |
| AMI | TS3011b | 0.615058 | 18 | 77 | 453 | 77 | 35 | 44 | 22 | 28 | 261 |
| AMI | TS3011c | 0.665278 | 17 | 84 | 441 | 62 | 32 | 64 | 22 | 31 | 295 |
| AMI | TS3011d | 0.599281 | 12 | 82 | 308 | 45 | 25 | 51 | 15 | 42 | 304 |
| AMI | TS3012c | 0.66013 | 23 | 102 | 351 | 126 | 67 | 46 | 19 | 66 | 400 |

## Raw candidate-pool lower bound

| Criterion | Observed raw pool | Combined hard role minimum | Raw deficit/status |
|---|---:|---:|---:|
| Scored natural hours | 48.571701 hours | 33.0 hours | 0.0 hours |
| Independent meetings | 93 accepted sessions | 22 | 0 |
| Stable singleton hours | 29.633316 hours | 10.0 hours | 0.0 hours |
| Ongoing overlap minutes | 230.892517 minutes | 75.0 minutes | 0.0 minutes |
| `short_backchannel_return` | 3330 | 100 | 0 |
| `overlap_takeover` | 3297 | 120 | 0 |
| `overlap_return` | 5372 | 120 | 0 |
| `silence_gap_different_speaker_handoff` | 6160 | 240 | 0 |
| `same_speaker_silence_gap_resume` | 28652 | 240 | 0 |
| `clean_direct_different_speaker_handoff` | 1181 | 120 | 0 |

The raw-pool lower bounds and the selected connected-component assignment both pass. EVAL uses only freshness-eligible components; TRAIN+DEV and EVAL pass every exclusive topology and negative-exposure minimum; TRAIN, DEV, and EVAL pass their scored-hour and meeting minima. No topology substitutes for another, and no threshold, count, or natural-data requirement is weakened.

Topology manifest SHA-256: `728c33d17d239dedf08eed9e014cd7e42f4b980c9bcb5b7826c67449f897d7cd`

Split manifest SHA-256: `dce084ca8394f70e4f7fe4c72687bbfd95998d26e9ce43e600ef2eb8a65490b4`
