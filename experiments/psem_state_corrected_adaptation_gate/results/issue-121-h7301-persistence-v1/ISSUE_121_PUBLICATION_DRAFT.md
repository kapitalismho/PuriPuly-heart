# Issue 121 publication draft: H7301 state-corrected adaptation

## Scientific decision

**STOP / inconclusive; retain F0.** H7301 is not accepted. The 100 ms point is jointly useful on the declared equal-corpus macro C/M criterion, but the 300 ms point is not jointly useful and the 500 ms point is worse. That is one of three horizons, below the required two-of-three rule. No H7302, T2, TA, EVAL, or confirmation arm is authorized by this result.

This is a bounded H7301 state-corrected evaluation. It does not revise the unchanged Issue 107 negative and does not establish a general adaptation direction, depth progression, or source-level cause.

## §31 answers and result interpretation

1. **Acceptance and disposition:** H7301 is not accepted; retain F0 and stop escalation.
2. **Corrected-H global results:** The decision table uses one selected global threshold across all ten DEV meetings for each C or M envelope (C and M may differ). The raw metrics are contamination seconds per active-speech hour / miss rate / false cuts per hour:

   | Horizon | F0 | H at global C envelope | H at global M envelope | Jointly useful |
   | --- | --- | --- | --- | --- |
   | 100 ms | 2161.675 / 0.775487 / 228.734 | 1561.578 / 0.583874 / 108.642 | 1570.013 / 0.571040 / 151.139 | yes |
   | 300 ms | 1555.186 / 0.537089 / 92.282 | 1566.368 / 0.576841 / 89.445 | 1566.368 / 0.576841 / 89.445 | no |
   | 500 ms | 1978.648 / 0.687291 / 30.487 | 2232.237 / 0.817277 / 30.452 | 2232.237 / 0.817277 / 30.452 | no; worse |

   The corresponding H-minus-F0 deltas are 100 ms C contamination `-600.096946`, false cuts/hour `-120.092529`, miss `-0.191614`; 100 ms M `-591.662003`, `-77.594974`, `-0.204447`; 300 ms C/M `+11.182169`, `-2.836675`, `+0.039752`; and 500 ms C/M `+253.589080`, `-0.035230`, `+0.129986`.
3. **Independent corpus points:** Corpus-specific envelope thresholds are separate descriptive operating points, not replacements for the global decision table. H-minus-F0 contamination / miss deltas are AliMeeting and AMI respectively: 100 ms C `-1094.488 / -0.383355`, `-217.975 / -0.154117`; 100 ms M `-1094.488 / -0.383355`, `-191.428 / -0.167652`; 300 ms C `-205.001 / -0.055630`, `-76.846 / -0.056029`; 300 ms M `-205.001 / -0.055630`, `-62.195 / -0.056825`; 500 ms C `-88.886 / -0.067000`, `+189.849 / +0.075922`; 500 ms M `-88.886 / -0.067000`, `+194.607 / +0.075126`.
4. **Uncertainty and timing:** The canonical intervals are 2,000-replicate paired source/meeting-mean intervals, not pooled-rate or macro confidence intervals. Timing p90 values are per meeting, not pooled. The 100 ms timing residuals include AliMeeting R1019 `+157 ms` at C, R8009 `+302 ms` at C and `+199 ms` at M; at 300 ms R1019 is `+169 ms` at C and M; no selected 500 ms point violates the timing criterion.
5. **Calibration and ranking:** Raw and calibrated event metrics are identical at selected points; calibration changes score coordinates only. TRAIN-CALIB candidate AP is `0.497992215` and F0 AP `0.117604962`. A posthoc DEV analysis over valid-and-mapped frames computes pooled candidate AP `0.487681950383668` and F0 AP `0.150603849645612` over 195,375 frames. Candidate source-macro AP is AMI `0.491683249822344`, AliMeeting `0.430875816411725`, all ten `0.473441019799158`; corresponding F0 source-macro AP is AMI `0.144531097981704`, AliMeeting `0.174723625454627`, all ten `0.153588856223581`. DEV F0 AP was unavailable in the earlier report; these values are posthoc and are not retroactively preregistered evidence.
6. **Issue 107:** The sealed Issue 107 negative remains unchanged: H versus F0 pooled contamination `2154.20` versus `1902.41`, false cuts/hour `75.01` versus `45.85`, missed replacements/hour `252.41` versus `218.80`.
7. **Training and cause:** The export records 2,261 loss chunks and 142 GPU steps. TRAIN-FIT AP/NLL/Brier, slot trajectory/drift, topology comparator, and causal attribution are absent. No event-objective or representation cause is proven.
8. **Seed and depth:** Seed 7302 was not run. No depth progression is authorized.
9. **Arms:** No confirmation, H7302, T2, TA, or EVAL arm was opened or authorized by this result.
10. **Artifacts and binding:** The durable bundle contains the canonical postprocess outputs, deterministic compressed frontier, immutable export manifest, training metrics, all 11 CALIB and 10 DEV numeric NPZ files, reproducible analysis JSON/Markdown, and per-file hashes. It excludes audio, transcripts, checkpoints, credentials, and process logs. Dataset source identifiers and numeric label/prediction arrays remain in the reproducibility bundle; this is not a blanket assertion that all metadata is anonymous.

## Persistence exploration

The exact selected global C-envelope thresholds are H100 `0.5887844788775033`, H300 `0.391021290491487`, and H500 `0.7234637539961835`; H100 M is `0.3631036176874745`, while H300/H500 M equal their C thresholds. Positive runs use speech-present frames with score at or above threshold, contiguous source coordinates, invalid/speech-absent/below-threshold closure, masked-frame skips without duration or previous-end updates, and duration equal to the sum of each eligible `(end-start)` interval divided by 16 ms, including continuation after emit. Boundary matching uses the configured 500 ms product-event tolerance; that tolerance is not the horizon.

At the exact selected C thresholds:

- H100: matched `n=599`, median/p90/max `480/589/960 ms`; unmatched `n=663`, `300/1000/4700 ms`.
- H300: matched `n=655`, `500/668.2/995 ms`; unmatched `n=545`, `400/1400/4700 ms`.
- H500: matched `n=344`, `500/694.5/972 ms`; unmatched `n=196`, `892/2131/4700 ms`.

Holding the exact H500 score threshold and changing only the horizon yields unmatched `300/1000/4700 ms` at H100, `400/1300/4700 ms` at H300, and `892/2131/4700 ms` at H500. These traces are descriptive evidence of a persistence/confirmation interaction. Unmatched runs can be timing or matching failures and are not categorical ground-truth false positives. No gate criterion is changed and no causal source-level explanation is proven.

## Reproducibility and provenance

The frozen CPU postprocess took an observed `29m18s`. The preceding GPU run covered 53 FIT sources and used 142 training steps; its numeric export contains 11 CALIB plus 10 DEV sources. GPU/source binding SHA-256 is `a3d9003a76ea167c33c644f1e3d15862e0181ed633a0378c91cb6d0fccaa263a`; CPU postprocess source SHA-256 is `674a3639e4245d118960875fbc38a091e0ee9e995b9d8137664ae99417779ca5`; export manifest SHA-256 is `7ff72366ffa6182e5b3ef7824507294f8dc66073c99287744039a6a7701bc131`; trained-head SHA-256 is `bb5029da54b01a84763d0513a544cdbcd99bb1592f73b9eef71a1916e59aea3f`.

Reproduce from the repository root with the command in [`README.md`](README.md). The durable analysis output is [`PERSISTENCE_ANALYSIS.md`](PERSISTENCE_ANALYSIS.md), machine-readable results are [`persistence_analysis.json`](persistence_analysis.json), and the full per-file binding is [`bundle_manifest.json`](bundle_manifest.json). The parent decision is [`../../STATE_CORRECTED_ADAPTATION_DECISION.md`](../../STATE_CORRECTED_ADAPTATION_DECISION.md).

Preparation preflight used branch `experiment-v2-speaker-change-turn-boundaries-ls` at HEAD `eb743e073b05d6f56c36166f7f7684697247ba62`, upstream `origin/experiment-v2-speaker-change-turn-boundaries-ls`, and repository `kapitalismho/PuriPuly-heart`. Director must pin any posted links to the final reviewed commit at the commit barrier; this draft intentionally records the preparation preflight separately from that future commit identity.

Formal commit review is outstanding. This draft is local publication preparation only; no commit, push, release, or issue comment has been performed.
