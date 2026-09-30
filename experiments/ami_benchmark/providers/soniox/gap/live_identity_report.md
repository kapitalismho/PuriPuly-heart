# #204 live A-D native identity and text/coverage diagnosis

Status: **completed**.

Read-only consumption of factorial-owned immutable outputs. No provider calls, production edits, new reference alignment rules, native-width identity scoring, or timestamp correction.

## Runnable command

```text
experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/live_diagnose.py
experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/live_diagnose.py --schema-check-only
```

A missing result/projection/score is blocked/pending, never a zero score. Completed cases may be diagnosed while other runs remain pending. Exit 0 means every planned result/projection/score was available; exit 2 means indispensable inputs are pending or a provider run failed. Runtime exceptions mean receipt/configuration failure.

## Frozen protocol

Identity rules: `independent-lexical-anchor-v1` imported unchanged from diagnose.py (SHA-256 `fa6b2393ebebd9db60a4661be78cbeae9e71c520987a93b97337f04b96fc01b7`). Reference inventory SHA-256 `09d1c0c7951074b1dbf7c88ea6d8cba8a3e7ef48d2a218de8d914a6c5423784a`.
Diagnostic execution revision: `e8739a267a14a4688498f32cda3d41bd641e6615` at `2026-09-30T15:10:22.930639+00:00`; Python `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`; scipy `1.18.1`; numpy `2.5.3`. Native inference runtime/configuration is retained separately in each immutable provider result.
Lowercase ASCII lexicalization with apostrophes; concatenate native text before finding words; conservative exact lexical five-word/unique-gap anchors; explicit ambiguous/unmatched words. One maximum-count speaker assignment for each entire case/arm, never per turn. Same published AMI word and turn membership, overlap, <=0.7-s interjection and missing-turn strata as the retained diagnosis. This remains conditional native recognized-word attribution, not complete ASR accuracy or application color accuracy.
A/B use original complete PCM, C/D frozen app PCM with replay/S200 positions preserved; A/C terminal finalize only, B/D frozen app-cut finalize. After long_context/A completed while B was in flight, Director authorized sender v2 before the first C/D call: nominal800 coalescing across incidental 512-sample VAD frames. Exact C/D bytes/cuts/maps are preserved; packet availability is the maximum included availability, with cuts/S200/discontinuities/availability windows still split. A/B mechanics remain unchanged and original evidence is not rerun. Per-arm source-version binding and actual packet signatures are disclosed below. Comparisons are not archived ACK-gated segmented adapter parity.
AB/CD vary finalization under each declared paired PCM/packetization policy. AC/BD still vary combined waveform, source availability, packet boundaries and remaining added buffering: no pure-PCM causal claim is inferred from the amendment or DER alone.


## Sender versions and applicability

The original and amended plans/configurations/sender histories are retained and hashed in evidence. Owner-provided dry mechanics/byte proofs are contextual evidence, not live model accuracy. Original v1 results lacking sender metadata are explicitly bound to original A/B plan/history; no version field is fabricated.

| Case | Arm | Sender source version | Actual PCM writes | Packet-size histogram (samples) | Added availability spread maximum s |
|---|---|---:|---:|---|---:|
| long_context | A | 1 | 12113 | `{"128": 12, "160": 13, "192": 10, "224": 14, "256": 7, "288": 6, "32": 10, "320": 7, "352": 7, "384": 6, "416": 6, "448": 7, "480": 7, "512": 6, "544": 7, "576": 14, "608": 10, "64": 12, "640": 13, "672": 12, "704": 9, "736": 12, "768": 10, "800": 11887, "96": 9}` | None |
| long_context | B | 1 | 12113 | `{"128": 12, "160": 13, "192": 10, "224": 14, "256": 7, "288": 6, "32": 10, "320": 7, "352": 7, "384": 6, "416": 6, "448": 7, "480": 7, "512": 6, "544": 7, "576": 14, "608": 10, "64": 12, "640": 13, "672": 12, "704": 9, "736": 12, "768": 10, "800": 11887, "96": 9}` | None |
| long_context | C | 2 | 12551 | `{"224": 33, "512": 59, "544": 14, "608": 5, "640": 3, "672": 43, "704": 11, "736": 5444, "768": 6, "800": 6933}` | 0.032 |
| long_context | D | 2 | 12551 | `{"224": 33, "512": 59, "544": 14, "608": 5, "640": 3, "672": 43, "704": 11, "736": 5444, "768": 6, "800": 6933}` | 0.032 |
| brief_interjections | A | 2 | 2424 | `{"160": 1, "192": 2, "224": 3, "256": 3, "288": 4, "32": 3, "320": 1, "352": 1, "384": 3, "416": 3, "448": 1, "480": 1, "512": 4, "544": 3, "576": 3, "608": 2, "64": 2, "640": 1, "704": 1, "736": 2, "768": 3, "800": 2376, "96": 1}` | None |
| brief_interjections | B | 2 | 2424 | `{"160": 1, "192": 2, "224": 3, "256": 3, "288": 4, "32": 3, "320": 1, "352": 1, "384": 3, "416": 3, "448": 1, "480": 1, "512": 4, "544": 3, "576": 3, "608": 2, "64": 2, "640": 1, "704": 1, "736": 2, "768": 3, "800": 2376, "96": 1}` | None |
| brief_interjections | C | 2 | 2609 | `{"224": 6, "512": 12, "544": 2, "640": 1, "672": 6, "704": 1, "736": 1154, "768": 1, "800": 1426}` | 0.032 |
| brief_interjections | D | 2 | 2609 | `{"224": 6, "512": 12, "544": 2, "640": 1, "672": 6, "704": 1, "736": 1154, "768": 1, "800": 1426}` | 0.032 |
| low_overlap | A | 2 | 2424 | `{"128": 1, "160": 1, "192": 3, "224": 3, "256": 3, "288": 2, "32": 1, "320": 1, "352": 3, "384": 3, "416": 3, "448": 3, "480": 1, "512": 2, "544": 3, "576": 3, "608": 3, "64": 2, "640": 1, "672": 1, "704": 1, "736": 2, "768": 1, "800": 2376, "96": 1}` | None |
| low_overlap | B | 2 | 2424 | `{"128": 1, "160": 1, "192": 3, "224": 3, "256": 3, "288": 2, "32": 1, "320": 1, "352": 3, "384": 3, "416": 3, "448": 3, "480": 1, "512": 2, "544": 3, "576": 3, "608": 3, "64": 2, "640": 1, "672": 1, "704": 1, "736": 2, "768": 1, "800": 2376, "96": 1}` | None |
| low_overlap | C | 2 | 2634 | `{"224": 10, "512": 11, "544": 1, "608": 3, "640": 2, "672": 7, "704": 3, "736": 1165, "800": 1432}` | 0.032 |
| low_overlap | D | 2 | 2634 | `{"224": 10, "512": 11, "544": 1, "608": 3, "640": 2, "672": 7, "704": 3, "736": 1165, "800": 1432}` | 0.032 |
| long_return_gap | A | 2 | 3633 | `{"128": 2, "160": 3, "192": 2, "224": 1, "256": 1, "288": 2, "32": 3, "320": 7, "352": 3, "384": 2, "416": 2, "448": 3, "480": 7, "512": 2, "544": 1, "576": 1, "608": 2, "64": 3, "640": 3, "672": 2, "704": 4, "736": 3, "768": 3, "800": 3567, "96": 4}` | None |
| long_return_gap | B | 2 | 3633 | `{"128": 2, "160": 3, "192": 2, "224": 1, "256": 1, "288": 2, "32": 3, "320": 7, "352": 3, "384": 2, "416": 2, "448": 3, "480": 7, "512": 2, "544": 1, "576": 1, "608": 2, "64": 3, "640": 3, "672": 2, "704": 4, "736": 3, "768": 3, "800": 3567, "96": 4}` | None |
| long_return_gap | C | 2 | 3346 | `{"224": 11, "512": 14, "544": 2, "608": 4, "672": 19, "736": 1440, "768": 2, "800": 1854}` | 0.032 |
| long_return_gap | D | 2 | 3346 | `{"224": 11, "512": 14, "544": 2, "608": 4, "672": 19, "736": 1440, "768": 2, "800": 1854}` | 0.032 |

## Artifact availability

| Case | Arm | Result |
|---|---|---|
| long_context | A | passed |
| long_context | B | passed |
| long_context | C | passed |
| long_context | D | passed |
| brief_interjections | A | passed |
| brief_interjections | B | passed |
| brief_interjections | C | passed |
| brief_interjections | D | passed |
| low_overlap | A | passed |
| low_overlap | B | passed |
| low_overlap | C | passed |
| low_overlap | D | passed |
| long_return_gap | A | passed |
| long_return_gap | B | passed |
| long_return_gap | C | passed |
| long_return_gap | D | passed |

## Native identity alongside DER and text/coverage

| Case | Arm | DER % | Ref / recognized words | Anchored / ambiguous | Unmatched ref / recognized | Wrong % known anchored | Correct reference coverage % |
|---|---|---:|---:|---:|---:|---:|---:|
| long_context | A | 42.74 | 2331 / 2054 | 1645 / 134 | 552 / 275 | 1.82 | 69.28 |
| long_context | B | 73.67 | 2331 / 2041 | 1606 / 144 | 581 / 291 | 1.12 | 68.13 |
| long_context | C | 43.13 | 2331 / 1918 | 1530 / 141 | 660 / 247 | 0.26 | 65.47 |
| long_context | D | 71.80 | 2331 / 2077 | 1601 / 149 | 581 / 327 | 1.50 | 67.65 |
| brief_interjections | A | 62.60 | 455 / 363 | 275 / 36 | 144 / 52 | 33.09 | 40.44 |
| brief_interjections | B | 69.84 | 455 / 346 | 259 / 45 | 151 / 42 | 39.00 | 34.73 |
| brief_interjections | C | 58.45 | 455 / 362 | 279 / 34 | 142 / 49 | 20.79 | 48.57 |
| brief_interjections | D | 83.26 | 455 / 369 | 276 / 40 | 139 / 53 | 36.23 | 38.68 |
| low_overlap | A | 15.39 | 377 / 381 | 343 / 1 | 33 / 37 | 0.29 | 90.72 |
| low_overlap | B | 18.04 | 377 / 375 | 338 / 1 | 38 / 36 | 0.30 | 89.39 |
| low_overlap | C | 17.83 | 377 / 386 | 343 / 1 | 33 / 42 | 0.29 | 90.72 |
| low_overlap | D | 20.44 | 377 / 382 | 338 / 1 | 38 / 43 | 0.30 | 89.39 |
| long_return_gap | A | 44.62 | 451 / 429 | 402 / 3 | 46 / 24 | 0.25 | 88.91 |
| long_return_gap | B | 47.68 | 451 / 432 | 406 / 3 | 42 / 23 | 0.25 | 89.80 |
| long_return_gap | C | 33.65 | 451 / 431 | 397 / 4 | 50 / 30 | 0.00 | 88.03 |
| long_return_gap | D | 43.66 | 451 / 433 | 402 / 3 | 46 / 28 | 9.45 | 80.71 |

## Paired causal contrasts

| Case | Contrast right-minus-left | DER pp | Conditional native wrong pp | Correct coverage pp | Recognized words | Actual packet/availability signatures matched |
|---|---|---:|---:|---:|---:|---|
| long_context | AB | 30.93 | -0.70 | -1.16 | -13 | True |
| long_context | CD | 28.67 | 1.24 | 2.19 | 159 | True |
| long_context | AC | 0.39 | -1.56 | -3.82 | -136 | False |
| long_context | BD | -1.88 | 0.38 | -0.47 | 36 | False |
| brief_interjections | AB | 7.24 | 5.91 | -5.71 | -17 | True |
| brief_interjections | CD | 24.82 | 15.44 | -9.89 | 7 | True |
| brief_interjections | AC | -4.16 | -12.30 | 8.13 | -1 | False |
| brief_interjections | BD | 13.42 | -2.76 | 3.96 | 23 | False |
| low_overlap | AB | 2.65 | 0.00 | -1.33 | -6 | True |
| low_overlap | CD | 2.61 | 0.00 | -1.33 | -4 | True |
| low_overlap | AC | 2.44 | 0.00 | 0.00 | 5 | False |
| low_overlap | BD | 2.40 | 0.00 | 0.00 | 7 | False |
| long_return_gap | AB | 3.06 | -0.00 | 0.89 | 3 | True |
| long_return_gap | CD | 10.01 | 9.45 | -7.32 | 2 | True |
| long_return_gap | AC | -10.97 | -0.25 | -0.89 | 2 | False |
| long_return_gap | BD | -4.01 | 9.21 | -9.09 | 1 | False |

AB/CD vary finalization under matched declared PCM; AC/BD vary combined PCM policy, not isolated replay/padding/silence removal. One run per arm, exposed diagnostic subset: no statistical-significance claim. Native identity uses each arm's global mapping and explicit coverage; unequal lexical matches constrain causal attribution.

## Difficult strata

| Case | Arm | Stratum | Reference denominator | Outcomes |
|---|---|---|---:|---|
| long_context | A | overlap | 1030 | `{"ambiguous_alignment": 97, "correct": 510, "unmatched_reference": 401, "wrong": 22}` |
| long_context | A | nonoverlap | 1301 | `{"ambiguous_alignment": 37, "correct": 1105, "unmatched_reference": 151, "wrong": 8}` |
| long_context | A | short_interjection | 34 | `{"ambiguous_alignment": 3, "correct": 10, "unmatched_reference": 21}` |
| long_context | A | turn_reference_missing | 0 | `{}` |
| long_context | B | overlap | 1030 | `{"ambiguous_alignment": 104, "correct": 502, "unmatched_reference": 414, "wrong": 10}` |
| long_context | B | nonoverlap | 1301 | `{"ambiguous_alignment": 40, "correct": 1086, "unmatched_reference": 167, "wrong": 8}` |
| long_context | B | short_interjection | 34 | `{"ambiguous_alignment": 4, "correct": 10, "unmatched_reference": 20}` |
| long_context | B | turn_reference_missing | 0 | `{}` |
| long_context | C | overlap | 1030 | `{"ambiguous_alignment": 90, "correct": 474, "unmatched_reference": 462, "wrong": 4}` |
| long_context | C | nonoverlap | 1301 | `{"ambiguous_alignment": 51, "correct": 1052, "unmatched_reference": 198}` |
| long_context | C | short_interjection | 34 | `{"ambiguous_alignment": 2, "correct": 8, "unmatched_reference": 24}` |
| long_context | C | turn_reference_missing | 0 | `{}` |
| long_context | D | overlap | 1030 | `{"ambiguous_alignment": 109, "correct": 486, "unmatched_reference": 423, "wrong": 12}` |
| long_context | D | nonoverlap | 1301 | `{"ambiguous_alignment": 40, "correct": 1091, "unmatched_reference": 158, "wrong": 12}` |
| long_context | D | short_interjection | 34 | `{"ambiguous_alignment": 3, "correct": 9, "unmatched_reference": 22}` |
| long_context | D | turn_reference_missing | 0 | `{}` |
| brief_interjections | A | overlap | 238 | `{"ambiguous_alignment": 27, "correct": 78, "unmatched_reference": 105, "wrong": 28}` |
| brief_interjections | A | nonoverlap | 217 | `{"ambiguous_alignment": 9, "correct": 106, "unmatched_reference": 39, "wrong": 63}` |
| brief_interjections | A | short_interjection | 17 | `{"ambiguous_alignment": 2, "correct": 2, "unmatched_reference": 13}` |
| brief_interjections | A | turn_reference_missing | 0 | `{}` |
| brief_interjections | B | overlap | 238 | `{"ambiguous_alignment": 33, "correct": 66, "unmatched_reference": 109, "wrong": 30}` |
| brief_interjections | B | nonoverlap | 217 | `{"ambiguous_alignment": 12, "correct": 92, "unmatched_reference": 42, "wrong": 71}` |
| brief_interjections | B | short_interjection | 17 | `{"ambiguous_alignment": 1, "correct": 2, "unmatched_reference": 14}` |
| brief_interjections | B | turn_reference_missing | 0 | `{}` |
| brief_interjections | C | overlap | 238 | `{"ambiguous_alignment": 27, "correct": 93, "unmatched_reference": 102, "wrong": 16}` |
| brief_interjections | C | nonoverlap | 217 | `{"ambiguous_alignment": 7, "correct": 128, "unmatched_reference": 40, "wrong": 42}` |
| brief_interjections | C | short_interjection | 17 | `{"ambiguous_alignment": 3, "correct": 1, "unmatched_reference": 11, "wrong": 2}` |
| brief_interjections | C | turn_reference_missing | 0 | `{}` |
| brief_interjections | D | overlap | 238 | `{"ambiguous_alignment": 30, "correct": 72, "unmatched_reference": 102, "wrong": 34}` |
| brief_interjections | D | nonoverlap | 217 | `{"ambiguous_alignment": 10, "correct": 104, "unmatched_reference": 37, "wrong": 66}` |
| brief_interjections | D | short_interjection | 17 | `{"ambiguous_alignment": 3, "correct": 2, "unmatched_reference": 11, "wrong": 1}` |
| brief_interjections | D | turn_reference_missing | 0 | `{}` |
| low_overlap | A | overlap | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | A | nonoverlap | 375 | `{"ambiguous_alignment": 1, "correct": 341, "unmatched_reference": 33}` |
| low_overlap | A | short_interjection | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | A | turn_reference_missing | 0 | `{}` |
| low_overlap | B | overlap | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | B | nonoverlap | 375 | `{"ambiguous_alignment": 1, "correct": 336, "unmatched_reference": 38}` |
| low_overlap | B | short_interjection | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | B | turn_reference_missing | 0 | `{}` |
| low_overlap | C | overlap | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | C | nonoverlap | 375 | `{"ambiguous_alignment": 1, "correct": 341, "unmatched_reference": 33}` |
| low_overlap | C | short_interjection | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | C | turn_reference_missing | 0 | `{}` |
| low_overlap | D | overlap | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | D | nonoverlap | 375 | `{"ambiguous_alignment": 1, "correct": 336, "unmatched_reference": 38}` |
| low_overlap | D | short_interjection | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | D | turn_reference_missing | 0 | `{}` |
| long_return_gap | A | overlap | 52 | `{"ambiguous_alignment": 2, "correct": 32, "unmatched_reference": 18}` |
| long_return_gap | A | nonoverlap | 399 | `{"ambiguous_alignment": 1, "correct": 369, "unmatched_reference": 28, "wrong": 1}` |
| long_return_gap | A | short_interjection | 1 | `{"wrong": 1}` |
| long_return_gap | A | turn_reference_missing | 0 | `{}` |
| long_return_gap | B | overlap | 52 | `{"ambiguous_alignment": 2, "correct": 35, "unmatched_reference": 15}` |
| long_return_gap | B | nonoverlap | 399 | `{"ambiguous_alignment": 1, "correct": 370, "unmatched_reference": 27, "wrong": 1}` |
| long_return_gap | B | short_interjection | 1 | `{"wrong": 1}` |
| long_return_gap | B | turn_reference_missing | 0 | `{}` |
| long_return_gap | C | overlap | 52 | `{"ambiguous_alignment": 3, "correct": 29, "unmatched_reference": 20}` |
| long_return_gap | C | nonoverlap | 399 | `{"ambiguous_alignment": 1, "correct": 368, "unmatched_reference": 30}` |
| long_return_gap | C | short_interjection | 1 | `{"unmatched_reference": 1}` |
| long_return_gap | C | turn_reference_missing | 0 | `{}` |
| long_return_gap | D | overlap | 52 | `{"ambiguous_alignment": 2, "correct": 20, "unmatched_reference": 17, "wrong": 13}` |
| long_return_gap | D | nonoverlap | 399 | `{"ambiguous_alignment": 1, "correct": 344, "unmatched_reference": 29, "wrong": 25}` |
| long_return_gap | D | short_interjection | 1 | `{"unmatched_reference": 1}` |
| long_return_gap | D | turn_reference_missing | 0 | `{}` |

## Native timing and available final-word evidence

| Case | Arm | Outside support tokens | Max native end minus support ms | Processed-counter response rows | Word arrival minus reference end: median / p90 s |
|---|---|---:|---:|---:|---|
| long_context | A | 0 | -360.0 | 3228 | 75.89499999999998 / 143.81999999999925 |
| long_context | B | 66 | 6720.0 | 2976 | 72.58399999999938 / 140.10899999999992 |
| long_context | C | 0 | -388.0 | 3050 | 88.55950000000075 / 149.6110000000008 |
| long_context | D | 73 | 8012.0 | 3130 | 89.66700000000037 / 151.9699999999989 |
| brief_interjections | A | 0 | -4200.0 | 601 | 24.475000000000023 / 34.26799999999827 |
| brief_interjections | B | 1 | 720.0 | 550 | 20.79400000000095 / 31.44900000000098 |
| brief_interjections | C | 0 | -3924.0 | 594 | 30.3920000000017 / 45.33800000000133 |
| brief_interjections | D | 0 | -2124.0 | 583 | 26.291999999998666 / 41.63599999999906 |
| low_overlap | A | 0 | -360.0 | 616 | 24.184000000000935 / 34.55300000000193 |
| low_overlap | B | 7 | 1320.0 | 568 | 21.55700000000172 / 31.810999999998444 |
| low_overlap | C | 0 | -456.0 | 621 | 30.406999999998106 / 45.10900000000038 |
| low_overlap | D | 9 | 1464.0 | 598 | 27.42250000000155 / 42.413000000001375 |
| long_return_gap | A | 0 | -360.0 | 754 | 34.99900000000066 / 49.43999999999943 |
| long_return_gap | B | 19 | 1680.0 | 714 | 31.777999999998286 / 48.37500000000057 |
| long_return_gap | C | 0 | -456.0 | 719 | 20.978000000000748 / 40.49800000000073 |
| long_return_gap | D | 20 | 1824.0 | 712 | 17.7295000000002 / 39.185999999998444 |

### Separating sender backlog from post-input final-word delay

| Case | Arm | Sender/write minus reference word end: median / p90 s | Final word minus actual source-word-end write: median / p90 s | Source-write association counts |
|---|---|---|---|---|
| long_context | A | 69.22899999999936 / 136.59899999999925 | 6.984000000000378 / 8.07799999999952 | `{"single_source_write": 1645}` |
| long_context | B | 69.07149999999933 / 136.55299999999988 | 3.389999999999418 / 6.343999999999141 | `{"single_source_write": 1606}` |
| long_context | C | 82.71150000000057 / 142.31900000000087 | 7.078000000000429 / 8.188000000000102 | `{"repeated_source_writes_ambiguous": 50, "single_source_write": 1478, "source_write_unavailable": 2}` |
| long_context | D | 86.58099999999968 / 147.7239999999997 | 3.702999999997701 / 6.453000000001339 | `{"repeated_source_writes_ambiguous": 42, "single_source_write": 1555, "source_write_unavailable": 4}` |
| brief_interjections | A | 17.40599999999904 / 27.76399999999734 | 7.03099999999904 / 8.109000000000378 | `{"single_source_write": 275}` |
| brief_interjections | B | 17.785999999999035 / 28.635999999999058 | 3.2809999999990396 / 6.453000000001339 | `{"single_source_write": 259}` |
| brief_interjections | C | 22.786000000002673 / 37.711000000002684 | 7.312999999998283 / 8.40599999999904 | `{"repeated_source_writes_ambiguous": 11, "single_source_write": 267, "source_write_unavailable": 1}` |
| brief_interjections | D | 22.771999999998684 / 37.664000000000385 | 3.937999999998283 / 6.73399999999674 | `{"repeated_source_writes_ambiguous": 10, "single_source_write": 265, "source_write_unavailable": 1}` |
| low_overlap | A | 17.155 / 27.984000000001515 | 6.930000000000291 / 8.0 | `{"single_source_write": 342, "source_write_unavailable": 1}` |
| low_overlap | B | 17.292000000001735 / 28.094000000000392 | 3.3280000000013388 / 6.10899999999674 | `{"single_source_write": 337, "source_write_unavailable": 1}` |
| low_overlap | C | 23.080500000001337 / 38.420999999999594 | 7.108999999998559 / 8.170999999998457 | `{"repeated_source_writes_ambiguous": 5, "single_source_write": 336, "source_write_unavailable": 2}` |
| low_overlap | D | 23.136000000002696 / 38.37400000000093 | 3.7180000000007567 / 6.4209999999984575 | `{"repeated_source_writes_ambiguous": 5, "single_source_write": 331, "source_write_unavailable": 2}` |
| long_return_gap | A | 28.14500000000001 / 43.57700000000227 | 6.875 / 8.046999999998661 | `{"single_source_write": 402}` |
| long_return_gap | B | 28.505000000000592 / 44.03699999999867 | 3.3515000000006694 / 6.264999999999418 | `{"single_source_write": 406}` |
| long_return_gap | C | 14.018000000001337 / 34.46700000000169 | 7.234000000000378 / 8.343000000000757 | `{"repeated_source_writes_ambiguous": 7, "single_source_write": 389, "source_write_unavailable": 1}` |
| long_return_gap | D | 14.105500000001058 / 34.677000000001726 | 3.7264999999970314 / 6.546999999998661 | `{"repeated_source_writes_ambiguous": 7, "single_source_write": 394, "source_write_unavailable": 1}` |

## Matched-input native clock observations

| Case | Contrast | Additional requested finalizations | Final observed total_audio_proc_ms difference ms | Anchored / ambiguous word matches | Native-start offset quartile medians ms |
|---|---|---:|---:|---|---|
| long_context | AB | 121 | 7080 | 1953 / 2 | `[1080.0, 2940, 4500.0, 6240.0]` |
| long_context | CD | 121 | 8400 | 1808 / 27 | `[1140.0, 3240.0, 5220.0, 7320.0]` |
| brief_interjections | AB | 24 | 1080 | 329 / 0 | `[120.0, 180.0, 600.0, 1020]` |
| brief_interjections | CD | 24 | 2040 | 351 / 0 | `[240, 720.0, 1470.0, 1740.0]` |
| low_overlap | AB | 25 | 1680 | 374 / 0 | `[60, 600.0, 1020.0, 1260]` |
| low_overlap | CD | 25 | 1920 | 379 / 0 | `[180, 660, 1020, 1380.0]` |
| long_return_gap | AB | 35 | 2040 | 423 / 0 | `[120.0, 720.0, 1140.0, 1800]` |
| long_return_gap | CD | 35 | 2280 | 426 / 1 | `[240, 960.0, 1620.0, 1980]` |

## Interpretation of completed evidence

- Manual app-cut finalization raises declared DER in 8/8 observed AB/CD comparisons; actual packet/availability signatures match in 8/8. This is the single-run controlled replay result, not application-adapter parity or a pure native-identity error percentage.
- long_context: AB adds 30.93 DER points while conditional native identity error changes -0.70 points; CD adds 28.67 DER points with native error +1.24 points and correct reference coverage +2.19 points. The large DER penalty therefore does not uniformly represent native recognized-word ID loss.
- For the same additional 121 requested controls, final total_audio_proc_ms changes by 7080 ms in AB and 8400 ms in CD. Actual native counter differences plus progressive word offsets strengthen the clock-discrepancy observation beyond transcript matching alone, but do not support a fixed per-requested-finalize rule or identify acoustic timestamp correction.
- brief_interjections shows native conditional error increases of 5.91 points in AB and 15.44 points in CD, with correct reference coverage changes -5.71/-9.89 points. A timing-only explanation cannot replace these native-attribution/coverage failures.
- long_return_gap/C: returning reference speaker ES2004b.D has anchored outcomes `{'correct': 38}`, native IDs `{'2': 38}`, global mapping `{'2': 'ES2004b.D'}`, and published-turn outcomes `{'ES2004b.sync.722': {'correct': 23}, 'ES2004b.sync.724': {'correct': 15}}`. These literal recognized-word witnesses distinguish a native label merge/attribution failure from displaced native timing; they are not re-mapped separately for the returning turn.
- long_return_gap/D: returning reference speaker ES2004b.D has anchored outcomes `{'wrong': 38}`, native IDs `{'2': 23, '1': 15}`, global mapping `{'2': 'ES2004b.C', '1': 'ES2004b.B'}`, and published-turn outcomes `{'ES2004b.sync.722': {'wrong': 23}, 'ES2004b.sync.724': {'wrong': 15}}`. These literal recognized-word witnesses distinguish a native label merge/attribution failure from displaced native timing; they are not re-mapped separately for the returning turn.
- Raw source-end-to-final-word receipt includes substantial sender backlog. The separately reported single-occurrence post-write medians are about seven seconds for terminal-only A/C and about three to four seconds for cut-finalized B/D in these runs; this combined processing/wire/finalization difference is a quality/readiness frontier, not an application publication measurement or a proposed fixed color wait.


Exact native tokens, processed counters as actually received, reference IDs, SDK-group indices and source-map intersections are traceable through immutable artifact receipts and representative examples in [live_identity_evidence.json](live_identity_evidence.json). Supporting native-clock and mapped-source offset quartiles are retained for every available contrast, including ambiguity and source-point support limits. No reference-fitted shift or 120-ms-per-finalize subtraction.
Provider word receipt requires every contributing final token. The evidence records published-turn arrivals and per-frozen-cut exposure separately at source-cut time and actual finalize-write time. Unmatched, ambiguous, missing/mixed and not-yet-final words remain in denominators. Negative arrival-minus-reference-end values are retained rather than clipped. These replay/reference clocks are not independently verified acoustic anchors.

Raw word-arrival minus published speech end includes sender backlog/context policy. A separate source-map association locates actual PCM writes for the last source sample before each anchored reference word end. Single-occurrence sender/write delay and subsequent word-receipt delay are retained separately; repeated-context occurrences are explicitly ambiguous. The post-write part combines provider, wire, finalization and receiver behavior, not pure model processing.
Archived segmented wall_time_s began before source-real-time offline app trace generation; new factorial active-WebSocket timing begins after frozen traces. Therefore archived long_context wall time versus new D WebSocket duration is not an adapter latency, processing, or billing improvement. This report compares only explicitly identified current replay/input/receipt clocks.
## Passed / blocked / not run

- Passed when available: frozen-helper and pinned-reference identities, native/raw/plan receipts, independent lexical/global speaker mapping, explicit coverage/strata, actual final-token receive observations.
- Passed model-free schema mechanics: half-open source/padding boundaries, subword final-word receipt barrier, unavailable receipt timestamps, separate source-cut/control-write exposure, literal overrun and actually present processed-counter retention. These are not live model-quality or caption-publication evidence.
- Result/projection/score gaps: none in a completed report; pending or failed arms remain explicitly listed in incomplete reports and are never zero measurements.
- R4 independent acoustic clock validation remains unresolved even with actual processed counters: published word times and transcript matches are not new waveform/event calibration.
- Not run here: provider generation/scoring (factorial owner), application normalization/scope and first-readable publication (consumption owner), physical microphone/VRChat/HMD, production enablement and billing.

## Disposition

Use native identity, correct reference coverage, omissions/ambiguity and provider-word arrival observations beside DER, not as substitutes. Defer a live-quality operating-point recommendation while indispensable comparisons or application publication evidence remain unavailable. No additional wait, recoloring, dropping short turns, retry, or production change is proposed by this diagnostic.
