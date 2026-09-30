# #204 retained offline diagnosis

## Disposition

**Defer causal/candidate adoption.** Retained corrected DER and independent lexical identity diagnostics are reproducible offline. They do not isolate finalization from PCM/ACK policy, calibrate native clocks, or establish first-readable usefulness. Live factorial and consumption evidence belong to separate owners. No production edits, provider calls, guessed timestamp correction, reference-informed clipping, or native 60-ms identity support were used.

## Execution and preservation

Execution revision: `4dc46fdac8c771d3bb58e46bd44304b86fde9c47`. Original baseline: `4dc46fdac8c771d3bb58e46bd44304b86fde9c47`.
Runtime: `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`; `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`; Node `v18.19.1`; pyannote.metrics `4.0.0`.
Archived inference: stt-rt-v5 at `wss://stt-rt.soniox.com/transcribe-websocket`, en hint, diarization enabled, language identification and endpoint detection disabled, pcm_s16le / 16 kHz / mono, one WS per case; async stt-async-v5 at `https://api.soniox.com/v1`. Those archived settings are not asserted to match the current application.
Archived sender: 800-sample (50-ms) PCM chunks with post-write sleep; forced/segmented wait for each requested fin acknowledgement before successor PCM. Padding is a separately sent all-zero S200 write without equivalent sleep. Archived segmented policy uses Silero-v6.2.1, 500-ms ring/pre-roll/hangover, onset 0.3 / derived exit 0.2, SmartTurn threshold 0.75, delivery profile on; source-language ownership snapshot zh-CN / expected zh-CN,ja differs from provider en hint. All frozen segments/source pieces are retained; no labels enter segmentation.
The exact retained comparison command ran inside a temporary copy with the pinned scoring venv symlinked; the generator never wrote original baseline artifacts. Generated comparison JSON was checked semantically against retained JSON, and original comparison JSON/Markdown hashes were unchanged.

```text
experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/diagnose.py --fetch-word-references
experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/diagnose.py
```

All exact subprocess argv/stdout/stderr, package versions, manifest/SDK/source identities, native receipt identities, source-to-provider sample maps, and alignment rows are in [offline_evidence.json](offline_evidence.json). Public annotation acquisition URLs/hashes and licensing are in [ami_word_reference_inventory.json](ami_word_reference_inventory.json).

## Checks

| Check | Result |
|---|---|
| Corrected baseline offline command | passed |
| Original generator output preservation | passed |
| Seven clip WAV/PCM/full-source identities | passed |
| 28 native receipt identities | passed |
| 28 actual SDK group/source projection reproductions | passed |
| Reconstructed PCM duration matches retained support | passed |
| Synthetic diagnostic mechanics only | passed |
| Published independent lexical/turn reference diagnostic | passed with declared alignment limits |
| Independently verified waveform clock calibration/correction | blocked: no retained event calibration/counters; no guessed correction |
| Live controlled factorial comparison | not run by retained-only owner; separate owner |
| Application normalization/first-readable consumption | not run by retained-only owner; separate owner |
| Current integrated app endpoint parity | not run; no archived-parity claim |
| Production enablement/VRChat/HMD and Korean/Japanese | not run |

Failed during local harness development (repaired, not provider failures): the acquisition run completed all case diagnostics but report rendering hit a distribution-name KeyError; the next run exposed diagnostic-only floating-boundary slivers in SDK verification. The final harness uses importlib.metadata.version and the existing projection numerical tolerance (<1 microsecond), without changing any baseline output or improving provider scores.


## Corrected declared-output DER

Full-clip UEM; zero collar; overlap included; pyannote.metrics 4.0.0; one optimal global mapping per case. SDK 2.3.0 speaker grouping happens before source mapping. Existing silence bridges and false alarms are retained. These are not word/person error percentages.

| Arm | Six-primary DER % | Miss s | False alarm s | Confusion s | Far-field DER % |
|---|---:|---:|---:|---:|---:|
| continuous | 48.27 | 276.091 | 203.559 | 85.154 | 61.44 |
| forced | 70.28 | 330.085 | 246.213 | 246.055 | 80.88 |
| segmented | 68.15 | 314.356 | 255.076 | 227.966 | 80.17 |
| async_full_file | 48.27 | 290.322 | 203.270 | 71.231 | 69.49 |

| Case | Arm | DER % | Miss s | False alarm s | Confusion s | Predicted / reference speakers |
|---|---|---:|---:|---:|---:|---:|
| low_overlap | continuous | 15.39 | 0.534 | 15.349 | 0.000 | 2 / 3 |
| rapid_turn_taking | continuous | 64.54 | 62.568 | 12.608 | 19.220 | 2 / 4 |
| overlap_heavy | continuous | 70.90 | 46.603 | 15.803 | 28.990 | 2 / 4 |
| long_return_gap | continuous | 44.57 | 5.055 | 50.274 | 0.063 | 3 / 4 |
| brief_interjections | continuous | 62.60 | 34.229 | 16.768 | 22.627 | 3 / 4 |
| long_context | continuous | 42.58 | 127.102 | 92.757 | 14.254 | 3 / 3 |
| far_field_pair | continuous | 61.44 | 45.558 | 18.478 | 15.159 | 3 / 4 |
| low_overlap | forced | 17.53 | 0.900 | 16.675 | 0.516 | 2 / 3 |
| rapid_turn_taking | forced | 78.31 | 71.208 | 16.988 | 26.337 | 2 / 4 |
| overlap_heavy | forced | 82.46 | 55.043 | 22.083 | 29.169 | 2 / 4 |
| long_return_gap | forced | 51.59 | 6.721 | 50.800 | 6.591 | 3 / 4 |
| brief_interjections | forced | 80.31 | 39.841 | 19.860 | 34.749 | 3 / 4 |
| long_context | forced | 77.28 | 156.372 | 119.807 | 148.693 | 3 / 3 |
| far_field_pair | forced | 80.88 | 56.630 | 24.750 | 22.879 | 3 / 4 |
| low_overlap | segmented | 25.31 | 1.260 | 24.295 | 0.564 | 2 / 3 |
| rapid_turn_taking | segmented | 76.96 | 69.722 | 18.610 | 24.235 | 2 / 4 |
| overlap_heavy | segmented | 77.23 | 50.803 | 24.823 | 23.918 | 3 / 4 |
| long_return_gap | segmented | 41.43 | 6.485 | 40.068 | 4.939 | 3 / 4 |
| brief_interjections | segmented | 84.90 | 34.480 | 24.979 | 40.389 | 3 / 4 |
| long_context | segmented | 74.18 | 151.606 | 122.301 | 133.921 | 3 / 3 |
| far_field_pair | segmented | 80.17 | 57.719 | 21.983 | 23.639 | 3 / 4 |
| low_overlap | async_full_file | 15.52 | 0.723 | 15.298 | 0.000 | 2 / 3 |
| rapid_turn_taking | async_full_file | 72.90 | 65.878 | 10.158 | 30.581 | 3 / 4 |
| overlap_heavy | async_full_file | 68.15 | 46.158 | 17.698 | 23.994 | 3 / 4 |
| long_return_gap | async_full_file | 45.16 | 5.363 | 50.402 | 0.362 | 3 / 4 |
| brief_interjections | async_full_file | 49.67 | 39.282 | 13.901 | 5.234 | 4 / 4 |
| long_context | async_full_file | 43.61 | 132.918 | 95.813 | 11.060 | 3 / 3 |
| far_field_pair | async_full_file | 69.49 | 47.698 | 17.138 | 24.742 | 3 / 4 |

Far-field pair is dependent on overlap_heavy (same meeting/window, different microphone) and is never pooled into six-primary results. Six primary windows are exposed annotation-selected diagnostics, not an independent participant sample or holdout.

## Independent reference-aligned native identity diagnostic

Published AMI v1.6.2 word XML provides lexical content, speaker identity, and original-meeting time; segment XML provides independent turn membership. Inputs are pinned to the existing audio mirror revision and CC BY 4.0. These annotations are independent of Soniox outputs, but not new manually audited acoustic/event anchors. References never enter inference.
Rules: lowercase ASCII alphanumeric/apostrophe lexical words, punctuation removed; native token text concatenated before lexicalization, so subword tokens are conserved. A word with conflicting contributing native speakers is mixed; absent metadata is missing. Reference words are sorted by published start, speaker, and ID; overlap linearization can misalign concurrent words and is a limitation, not fixed with native timing.
SequenceMatcher(auto junk off) yields monotone exact lexical blocks. A match is anchored only by a unique shared contiguous five-word phrase, or by a word occurring exactly once in each gap between those anchors (including clip-edge gaps). All other exact matches are ambiguous; nonmatches remain explicit. No token widths or timing shifts determine identity. One maximum-count one-to-one label assignment for the complete case/arm is fitted on anchored known-speaker words; it is not re-fitted per turn. This conditional diagnostic is not full ASR accuracy or realtime color accuracy.
Overlap means any positive reference-word intersection with a different speaker; short interjection means published full turn duration <=0.7 s, not native token duration. Separate denominators and per-turn counts are retained, including unknown-turn membership.

| Case | Arm | Ref / recognized words | Anchored / ambiguous | Unmatched ref / recognized | Wrong % known anchored | Correct reference coverage % |
|---|---|---:|---:|---:|---:|---:|
| low_overlap | continuous | 377 / 381 | 344 / 1 | 32 / 36 | 0.29 | 90.98 |
| low_overlap | forced | 377 / 378 | 339 / 1 | 37 / 38 | 0.29 | 89.66 |
| low_overlap | segmented | 377 / 404 | 341 / 3 | 33 / 60 | 0.29 | 90.19 |
| low_overlap | async_full_file | 377 / 381 | 347 / 1 | 29 / 33 | 0.00 | 92.04 |
| rapid_turn_taking | continuous | 536 / 395 | 258 / 54 | 224 / 83 | 26.74 | 35.26 |
| rapid_turn_taking | forced | 536 / 393 | 220 / 82 | 234 / 91 | 40.45 | 24.44 |
| rapid_turn_taking | segmented | 536 / 406 | 265 / 57 | 214 / 84 | 28.30 | 35.45 |
| rapid_turn_taking | async_full_file | 536 / 384 | 254 / 61 | 221 / 69 | 45.28 | 25.93 |
| overlap_heavy | continuous | 586 / 440 | 309 / 64 | 213 / 67 | 41.10 | 31.06 |
| overlap_heavy | forced | 586 / 428 | 302 / 64 | 220 / 62 | 38.41 | 31.74 |
| overlap_heavy | segmented | 586 / 438 | 301 / 64 | 221 / 73 | 19.93 | 41.13 |
| overlap_heavy | async_full_file | 586 / 432 | 291 / 64 | 231 / 77 | 32.30 | 33.62 |
| long_return_gap | continuous | 451 / 427 | 401 / 3 | 47 / 23 | 0.25 | 88.69 |
| long_return_gap | forced | 451 / 427 | 396 / 4 | 51 / 27 | 0.25 | 87.58 |
| long_return_gap | segmented | 451 / 436 | 383 / 13 | 55 / 40 | 0.00 | 84.92 |
| long_return_gap | async_full_file | 451 / 435 | 404 / 5 | 42 / 26 | 0.50 | 89.14 |
| brief_interjections | continuous | 455 / 367 | 279 / 36 | 140 / 52 | 32.62 | 41.32 |
| brief_interjections | forced | 455 / 361 | 272 / 36 | 147 / 53 | 27.94 | 43.08 |
| brief_interjections | segmented | 455 / 379 | 253 / 60 | 142 / 66 | 41.50 | 32.53 |
| brief_interjections | async_full_file | 455 / 351 | 278 / 33 | 144 / 40 | 6.83 | 56.92 |
| long_context | continuous | 2331 / 2054 | 1636 / 128 | 567 / 290 | 1.83 | 68.90 |
| long_context | forced | 2331 / 2023 | 1607 / 124 | 600 / 292 | 1.43 | 67.91 |
| long_context | segmented | 2331 / 2117 | 1582 / 164 | 585 / 371 | 1.01 | 67.18 |
| long_context | async_full_file | 2331 / 2040 | 1588 / 149 | 594 / 303 | 0.31 | 67.91 |
| far_field_pair | continuous | 586 / 434 | 277 / 78 | 231 / 79 | 14.80 | 40.27 |
| far_field_pair | forced | 586 / 420 | 273 / 77 | 236 / 70 | 17.22 | 38.57 |
| far_field_pair | segmented | 586 / 436 | 267 / 78 | 241 / 91 | 16.48 | 38.05 |
| far_field_pair | async_full_file | 586 / 449 | 285 / 69 | 232 / 95 | 33.68 | 32.25 |

### Diagnostic interpretation

The long_context native conditional wrong-speaker fraction is only 1.83% continuous, 1.43% forced, and 1.01% segmented among conservatively anchored known-speaker words, while corrected DER is 42.58%, 77.28%, and 74.18%. Its correct reference-word coverage remains 68.90%, 67.91%, and 67.18%. Thus the large DER gap does not establish a large native recognized-word identity loss in this case; timing/support/projection and uncovered words remain distinct factors, not a proven unique cause.
Overlap_heavy segmented has lower conditional identity error (19.93%) than continuous (41.10%) despite worse DER (77.23% vs 70.90%); independently linearized overlap and uncovered/ambiguous words limit interpretation. Brief_interjections segmented has worse conditional identity (41.50% vs 32.62%) and lower correct reference coverage (32.53% vs 41.32%). These exposed-set observations rule out treating the pooled DER change as a uniform voice-identity gain/loss.
Representative metadata, zero-duration, overrun and largest envelope-bridge events are retained beside each arm, including exact native token indices and source-piece mappings. Synthetic metric self-checks are separate from provider evidence.

### Explicit difficult strata

| Case | Arm | Stratum | Reference denominator | Outcome counts |
|---|---|---|---:|---|
| low_overlap | continuous | overlap | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | continuous | short_interjection | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | continuous | turn_reference_missing | 0 | `{}` |
| low_overlap | forced | overlap | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | forced | short_interjection | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | forced | turn_reference_missing | 0 | `{}` |
| low_overlap | segmented | overlap | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | segmented | short_interjection | 2 | `{"correct": 1, "wrong": 1}` |
| low_overlap | segmented | turn_reference_missing | 0 | `{}` |
| low_overlap | async_full_file | overlap | 2 | `{"correct": 2}` |
| low_overlap | async_full_file | short_interjection | 2 | `{"correct": 1, "unmatched_reference": 1}` |
| low_overlap | async_full_file | turn_reference_missing | 0 | `{}` |
| rapid_turn_taking | continuous | overlap | 383 | `{"ambiguous_alignment": 44, "correct": 85, "unmatched_reference": 194, "wrong": 60}` |
| rapid_turn_taking | continuous | short_interjection | 18 | `{"ambiguous_alignment": 1, "correct": 3, "unmatched_reference": 6, "wrong": 8}` |
| rapid_turn_taking | continuous | turn_reference_missing | 0 | `{}` |
| rapid_turn_taking | forced | overlap | 383 | `{"ambiguous_alignment": 65, "correct": 54, "unmatched_reference": 202, "wrong": 62}` |
| rapid_turn_taking | forced | short_interjection | 18 | `{"ambiguous_alignment": 1, "correct": 3, "unmatched_reference": 7, "wrong": 7}` |
| rapid_turn_taking | forced | turn_reference_missing | 0 | `{}` |
| rapid_turn_taking | segmented | overlap | 383 | `{"ambiguous_alignment": 44, "correct": 88, "unmatched_reference": 185, "wrong": 66}` |
| rapid_turn_taking | segmented | short_interjection | 18 | `{"ambiguous_alignment": 1, "correct": 3, "unmatched_reference": 6, "wrong": 8}` |
| rapid_turn_taking | segmented | turn_reference_missing | 0 | `{}` |
| rapid_turn_taking | async_full_file | overlap | 383 | `{"ambiguous_alignment": 50, "correct": 61, "unmatched_reference": 188, "wrong": 84}` |
| rapid_turn_taking | async_full_file | short_interjection | 18 | `{"correct": 2, "unmatched_reference": 9, "wrong": 7}` |
| rapid_turn_taking | async_full_file | turn_reference_missing | 0 | `{}` |
| overlap_heavy | continuous | overlap | 383 | `{"ambiguous_alignment": 51, "correct": 82, "unmatched_reference": 180, "wrong": 70}` |
| overlap_heavy | continuous | short_interjection | 24 | `{"ambiguous_alignment": 1, "correct": 5, "unmatched_reference": 18}` |
| overlap_heavy | continuous | turn_reference_missing | 0 | `{}` |
| overlap_heavy | forced | overlap | 383 | `{"ambiguous_alignment": 53, "correct": 87, "unmatched_reference": 183, "wrong": 60}` |
| overlap_heavy | forced | short_interjection | 24 | `{"ambiguous_alignment": 1, "correct": 5, "unmatched_reference": 17, "wrong": 1}` |
| overlap_heavy | forced | turn_reference_missing | 0 | `{}` |
| overlap_heavy | segmented | overlap | 383 | `{"ambiguous_alignment": 50, "correct": 104, "unmatched_reference": 187, "wrong": 42}` |
| overlap_heavy | segmented | short_interjection | 24 | `{"ambiguous_alignment": 2, "correct": 5, "unmatched_reference": 16, "wrong": 1}` |
| overlap_heavy | segmented | turn_reference_missing | 0 | `{}` |
| overlap_heavy | async_full_file | overlap | 383 | `{"ambiguous_alignment": 52, "correct": 63, "unmatched_reference": 196, "wrong": 72}` |
| overlap_heavy | async_full_file | short_interjection | 24 | `{"ambiguous_alignment": 1, "correct": 3, "unmatched_reference": 18, "wrong": 2}` |
| overlap_heavy | async_full_file | turn_reference_missing | 0 | `{}` |
| long_return_gap | continuous | overlap | 52 | `{"ambiguous_alignment": 2, "correct": 32, "unmatched_reference": 18}` |
| long_return_gap | continuous | short_interjection | 1 | `{"wrong": 1}` |
| long_return_gap | continuous | turn_reference_missing | 0 | `{}` |
| long_return_gap | forced | overlap | 52 | `{"ambiguous_alignment": 2, "correct": 32, "unmatched_reference": 18}` |
| long_return_gap | forced | short_interjection | 1 | `{"wrong": 1}` |
| long_return_gap | forced | turn_reference_missing | 0 | `{}` |
| long_return_gap | segmented | overlap | 52 | `{"ambiguous_alignment": 6, "correct": 26, "unmatched_reference": 20}` |
| long_return_gap | segmented | short_interjection | 1 | `{"unmatched_reference": 1}` |
| long_return_gap | segmented | turn_reference_missing | 0 | `{}` |
| long_return_gap | async_full_file | overlap | 52 | `{"ambiguous_alignment": 3, "correct": 29, "unmatched_reference": 19, "wrong": 1}` |
| long_return_gap | async_full_file | short_interjection | 1 | `{"wrong": 1}` |
| long_return_gap | async_full_file | turn_reference_missing | 0 | `{}` |
| brief_interjections | continuous | overlap | 238 | `{"ambiguous_alignment": 27, "correct": 79, "unmatched_reference": 104, "wrong": 28}` |
| brief_interjections | continuous | short_interjection | 17 | `{"ambiguous_alignment": 2, "correct": 2, "unmatched_reference": 13}` |
| brief_interjections | continuous | turn_reference_missing | 0 | `{}` |
| brief_interjections | forced | overlap | 238 | `{"ambiguous_alignment": 27, "correct": 80, "unmatched_reference": 108, "wrong": 23}` |
| brief_interjections | forced | short_interjection | 17 | `{"ambiguous_alignment": 2, "correct": 2, "unmatched_reference": 12, "wrong": 1}` |
| brief_interjections | forced | turn_reference_missing | 0 | `{}` |
| brief_interjections | segmented | overlap | 238 | `{"ambiguous_alignment": 41, "correct": 72, "unmatched_reference": 101, "wrong": 24}` |
| brief_interjections | segmented | short_interjection | 17 | `{"ambiguous_alignment": 2, "correct": 2, "unmatched_reference": 13}` |
| brief_interjections | segmented | turn_reference_missing | 0 | `{}` |
| brief_interjections | async_full_file | overlap | 238 | `{"ambiguous_alignment": 22, "correct": 97, "unmatched_reference": 100, "wrong": 19}` |
| brief_interjections | async_full_file | short_interjection | 17 | `{"ambiguous_alignment": 2, "correct": 4, "unmatched_reference": 11}` |
| brief_interjections | async_full_file | turn_reference_missing | 0 | `{}` |
| long_context | continuous | overlap | 1030 | `{"ambiguous_alignment": 91, "correct": 506, "unmatched_reference": 411, "wrong": 22}` |
| long_context | continuous | short_interjection | 34 | `{"ambiguous_alignment": 3, "correct": 10, "unmatched_reference": 21}` |
| long_context | continuous | turn_reference_missing | 0 | `{}` |
| long_context | forced | overlap | 1030 | `{"ambiguous_alignment": 87, "correct": 507, "unmatched_reference": 424, "wrong": 12}` |
| long_context | forced | short_interjection | 34 | `{"ambiguous_alignment": 3, "correct": 9, "unmatched_reference": 21, "wrong": 1}` |
| long_context | forced | turn_reference_missing | 0 | `{}` |
| long_context | segmented | overlap | 1030 | `{"ambiguous_alignment": 114, "correct": 487, "unmatched_reference": 418, "wrong": 11}` |
| long_context | segmented | short_interjection | 34 | `{"ambiguous_alignment": 2, "correct": 8, "unmatched_reference": 24}` |
| long_context | segmented | turn_reference_missing | 0 | `{}` |
| long_context | async_full_file | overlap | 1030 | `{"ambiguous_alignment": 103, "correct": 495, "unmatched_reference": 428, "wrong": 4}` |
| long_context | async_full_file | short_interjection | 34 | `{"ambiguous_alignment": 3, "correct": 8, "unmatched_reference": 21, "wrong": 2}` |
| long_context | async_full_file | turn_reference_missing | 0 | `{}` |
| far_field_pair | continuous | overlap | 383 | `{"ambiguous_alignment": 57, "correct": 112, "unmatched_reference": 190, "wrong": 24}` |
| far_field_pair | continuous | short_interjection | 24 | `{"correct": 5, "unmatched_reference": 19}` |
| far_field_pair | continuous | turn_reference_missing | 0 | `{}` |
| far_field_pair | forced | overlap | 383 | `{"ambiguous_alignment": 55, "correct": 102, "unmatched_reference": 195, "wrong": 31}` |
| far_field_pair | forced | short_interjection | 24 | `{"correct": 4, "unmatched_reference": 18, "wrong": 2}` |
| far_field_pair | forced | turn_reference_missing | 0 | `{}` |
| far_field_pair | segmented | overlap | 383 | `{"ambiguous_alignment": 58, "correct": 100, "unmatched_reference": 199, "wrong": 26}` |
| far_field_pair | segmented | short_interjection | 24 | `{"correct": 5, "unmatched_reference": 19}` |
| far_field_pair | segmented | turn_reference_missing | 0 | `{}` |
| far_field_pair | async_full_file | overlap | 383 | `{"ambiguous_alignment": 52, "correct": 84, "unmatched_reference": 192, "wrong": 55}` |
| far_field_pair | async_full_file | short_interjection | 24 | `{"correct": 3, "unmatched_reference": 19, "wrong": 2}` |
| far_field_pair | async_full_file | turn_reference_missing | 0 | `{}` |

## Trace and classification findings

- Native attribution: representative lexically anchored wrong-speaker words retain public reference IDs, native token indices/content/speaker/time, contributing SDK groups, and mapped source-piece intersections in `cases.CASE.arms.ARM.representative_wrong_native_attribution`. The native disagreement is visible independently of 60-ms support; mappings are global. Missing/ambiguous words are not silently scored correct.
- Native content/metadata: all reference and recognized word counts, unmatched words, mixed/missing contributing metadata, and ambiguous alignments are explicit. Unmatched lexical words are not definitive ASR omissions: transcription conventions, overlap order, contractions, and segmentation can also break exact matching. Native missing speaker counts are separate from timing omissions.
- Projection: SDK groups and source-projected intervals are regenerated using the pinned actual SDK and compared to retained artifacts. Large same-speaker envelopes bridge silence; false alarm cost is preserved. Native zero-duration tokens remain recognized lexical evidence where text exists, without positive diarization support. Historical embedded conversion metadata describes old pulse projection, not the corrected score.
- PCM/source policy: the artifact retains sample-coordinate maps and reconstructs PCM identities from source pieces plus all-zero padding. Replay, skipped support, discontinuities and padding are counted separately. These are reconstructed from retained sender maps, **not observed per-write network receipts**. Original app-cut segment metadata is retained; the offline script does not regenerate app segmentation or claim current integrated parity.
- Normalization/deadline: final native receipts do not retain first-readable publication, native-token arrival timestamps, consumer scope or text conservation. No caption attribution or live-ready claim follows from this diagnosis. The independent application-consumption owner handles those fixtures/traces.

## R4 retained timestamp observations — unresolved, no correction

Exact text anchors between continuous and forced final transcripts compare first contributing native token starts, grouped by continuous-native-time quartile. This supporting text comparison is separate from identity scoring, with no fitted subtraction. Native processed-audio counters were not retained in the simplified message envelopes. Actual PCM/control/receive write timestamps are absent. A native clock, sender behavior, matching ambiguity and source/output conversion cannot be uniquely distinguished from these receipts.

| Case | Forced finalizations | Max native overrun ms (historical diagnostic) | Quartile matched-word median offsets ms |
|---|---:|---:|---|
| low_overlap | 20 | 1920.0 | `[240, 840.0, 1440, 2040.0]` |
| rapid_turn_taking | 20 | 1920.0 | `[240, 840.0, 1440.0, 2040.0]` |
| overlap_heavy | 20 | 1920.0 | `[240.0, 840, 1440.0, 2040]` |
| long_return_gap | 30 | 3120.0 | `[360.0, 1440, 2160.0, 3120]` |
| brief_interjections | 20 | 0.0 | `[240, 840, 1440.0, 2040]` |
| long_context | 100 | 11520.0 | `[1560.0, 4440.0, 7320, 10560.0]` |
| far_field_pair | 20 | 1920.0 | `[240.0, 840.0, 1440, 2100.0]` |

These progressive offsets and overrun values reproduce supporting retained observations; they do not prove 120 ms of synthetic PCM per finalize. No per-finalize subtraction, reference-fitted calibration, silence clipping, or correction was applied. Published AMI lexical times were not promoted to independently verified waveform clock anchors. A small licensed/consented event-anchor fixture, exact PCM/control receipts and actual processed-audio counters (where returned) remain necessary for causal calibration and separately frozen correction validation.

## Limits and documentation proposals for Director

- Recommend documenting corrected DER and the conditional lexical identity diagnostic together, with unmatched/ambiguous/overlap/interjection denominators rather than calling DER voice-ID accuracy.
- Record the timestamp finding as bounded unresolved: retained messages omit counters/receive times; no correction is supported. Preserve literal native timing.
- Record this execution as replacement WSL offline scoring/SDK environment, not the historical Windows provider runtime. Retained Python/provider SDK and model server versions cannot be retroactively recovered; current model names/config and pinned projection/scorer versions are retained.
- Current production endpoint/config parity is not established by archived endpoint-off inference. No historical endpoint behavior was restored. This task did not exercise physical microphone, VRChat/HMD, Korean/Japanese or billing.
- No new Soniox requests or private audio/credentials were consumed by this script. Public input acquisition was only pinned AMI XML/license metadata. Factorial live-run quota and candidate readiness evidence are outside this retained-only assignment.
- Maintainer decisions still concern causal live results and supported quality/latency/cost tradeoffs; this offline evidence does not authorize production adoption. No supported negative model-quality conclusion can replace an indispensable missing controlled comparison.


## Public reference availability

No missing published word/turn reference inputs or untimed selected lexical reference words were observed. This does not certify the acoustic calibration or correctness of the published reference timings.
