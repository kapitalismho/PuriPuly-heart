# mHuBERT-A error decomposition results

Evidence status: **no-training analysis of existing development-known out-of-fold raw predictions**.

The input is the existing `mhubert-147 / A-FROZEN-DIRECT` prediction artifact from issue #72. The analysis added no feature extraction, training, model selection, context change, or event-label change.

Input prediction SHA-256: `4460e11c4689bb14afc6516da9c04ec8a7a5f1a1090eac08da41bf6eb9603b61`

Analysis artifact: `%SRSCD_CACHE_ROOT%/results/mhubert_a_error_decomposition_v1/analysis.json`

Analysis SHA-256: `c4502c2dca6128c042a90fd223786188d96c94116910ef04bc0686c1587653a3`

## Operating point

The threshold `0.6912562847` maximizes mean F1 across the 100/250/500 ms collars over the complete score range. FE/h did not select it. There are 7,042 selected peaks and 4,619 GT events.

| Collar | TP | FP | Miss | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 ms | 2,042 | 5,000 | 2,577 | 29.00% | 44.21% | 35.02% |
| 250 ms | 2,691 | 4,351 | 1,928 | 38.21% | 58.26% | 46.15% |
| 500 ms | 3,017 | 4,025 | 1,602 | 42.84% | 65.32% | 51.75% |

## Timing error

Among the 3,017 predictions matched at 500 ms, signed error has mean `-3.9 ms` and median `0 ms`. There is no material global early/late bias. Absolute error has median `60 ms` and p90 `330 ms`.

| Signed error interval | Matches |
| --- | ---: |
| −500 to −400 ms | 85 |
| −400 to −300 ms | 121 |
| −300 to −200 ms | 137 |
| −200 to −100 ms | 220 |
| −100 to 0 ms | 790 |
| 0 to +100 ms | 1,087 |
| +100 to +200 ms | 300 |
| +200 to +300 ms | 124 |
| +300 to +400 ms | 80 |
| +400 to +500 ms | 73 |

Timing is a real error source:

- 657 predictions that are FP at 100 ms become TP at 250 ms.
- 1,040 predictions that are FP at 100 ms become TP at 500 ms, or 20.8% of all 100 ms FP.
- 1,023 GT events missed at 100 ms are matched at 500 ms, or 39.7% of all 100 ms misses.
- 406 predictions that are FP at 250 ms become TP at 500 ms.

The absence of a signed offset means this is peak spread/localization inconsistency rather than one fixed lag that could be removed by shifting every prediction.

## Duplicate peaks

| Measure | Count |
| --- | ---: |
| Selected raw local maxima | 7,268 |
| Selected peaks after established 200 ms NMS | 7,042 |
| Peaks removed by NMS | 226 |
| GT events with multiple selected peaks within 500 ms after NMS | 435 |
| Excess selected peaks in those GT windows after NMS | 441 |
| Evaluated 500 ms FP still within 500 ms of some GT | 180 |

Duplicate clustering exists, but it is not the main FP source. Only 180 of 4,025 evaluated 500 ms FP, or 4.5%, remain within 500 ms of a GT event. The other 95.5% are remote from every GT event.

## Remote false positives

At the 500 ms collar, 3,845 FP are more than 500 ms from every GT event.

| GT-derived state category | Count | Share of remote FP | Median score | Score p90 |
| --- | ---: | ---: | ---: | ---: |
| Continuous same-speaker singleton | 2,744 | 71.4% | 0.842 | 0.942 |
| Overlap continuation | 711 | 18.5% | 0.801 | 0.928 |
| Same-speaker pause/resume | 177 | 4.6% | 0.891 | 0.968 |
| Silence continuation | 170 | 4.4% | 0.853 | 0.940 |
| Overlap end | 34 | 0.9% | 0.782 | 0.922 |
| Other speech/silence transition | 9 | 0.2% | — | — |

The largest meeting contributes 754 remote FP, or 19.6%; the error is not confined to one meeting. High scores occur in every major category, including a median of 0.891 for same-speaker pause/resume.

The frozen GT can identify activity state and speaker continuity, but it has no laughter or prosody annotation. Those two acoustic categories cannot be inferred honestly from this artifact. The analysis JSON retains the 30 highest-scoring session timestamps for an optional audio audit.

## Candidate coverage without a score threshold

| Candidate set | ±100 ms | ±250 ms | ±500 ms |
| --- | ---: | ---: | ---: |
| Raw local maxima | 64.45% | 92.40% | 99.39% |
| After 200 ms NMS | 61.14% | 91.12% | 99.29% |

After NMS, only 33 of 4,619 GT events have no local peak at any score within 500 ms. This rejects candidate geometry absence as the dominant explanation. It does not prove that those candidates carry discriminative speaker evidence: many have scores too low to survive the operating threshold, while remote same-speaker peaks often score very highly.

## Event-stratum recall

| GT stratum | Count | Recall@100 | Recall@250 | Recall@500 |
| --- | ---: | ---: | ---: | ---: |
| Overlap onset | 3,420 | 38.33% | 52.92% | 61.64% |
| Silence-gap change | 1,192 | 61.24% | 73.49% | 75.84% |
| Short backchannel/return | 5 | 20.00% | 60.00% | 60.00% |
| Clean change | 2 | 0.00% | 100.00% | 100.00% |

The collar sensitivity is larger for overlap onset than silence-gap change. This supports a localization component, particularly around overlap boundaries.

## Decision

The evidence is not a pure timing case.

- **Timing/localization is material:** 1,040 100 ms false predictions become 500 ms true positives, and 1,023 100 ms missed references are recovered at 500 ms.
- **Remote false activation is the larger FP burden:** 3,845 of 4,025 500 ms FP are remote, dominated by continuous same-speaker speech and overlap continuation.
- **Duplicate eventization is secondary:** only 180 evaluated 500 ms FP are GT-proximal after one-to-one matching.
- **Candidate absence is not the dominant limitation:** 99.29% of GT events have some post-NMS peak within 500 ms when the score threshold is removed.

The strongest current diagnosis is a combination of temporal localization error and poor score discrimination against same-speaker/ongoing-overlap hard negatives, with the latter dominating FP count. The data do not justify a localization-only training experiment. Before selecting any new ML experiment, the highest-scoring remote FP timestamps should be audited for unannotated laughter, prosody, and label-semantic cases. If that audit confirms ordinary same-speaker acoustic excursions, a bounded hard-negative direct-detector experiment is better supported than structured-state repair or broader representation/context changes.
