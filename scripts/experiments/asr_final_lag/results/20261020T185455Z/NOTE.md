# Notes on this run

## Input

- The same 125 s conversation windows as `../20261020T070225Z` (`ko_conv`, `ja_conv`, `en_conv`). References are uploaded caption lines, so sentence columns measure caption line ends.
- Qwen Audio 3.1 cut variants open one task per local segment on two alternating connections (see `README.md`, "Cut schedules and Smart Turn"). Gemini end variants send `audio_stream_end` at the cut times of a local policy. Smart Turn is scored 200 ms into each pause.
- One run per variant. Run-to-run noise is not measured here; the Gemini interim replay of two natural runs (`../interim-20261020T190834Z/NOTE.md`) shows CER differences of up to 8 points between runs.

## Analyzer fix

`analyze.py` searched the whole reference for the first partial text. In `en_conv` and `ja_conv`, the first Qwen Audio partial of the cut variants ("Work" for "Welcome") matched text far into the reference, so `partial first`, `partial stable`, `release diff` and `punct` showed values such as -19.6 s. The first search now starts at the beginning of the reference. Re-analysis of every earlier results folder gives identical summaries; only the Qwen Audio cut rows of this folder changed.

## Qwen Audio cut variants

Cells: CER, then `final` p50 / p90 after the caption line end.

| variant | ko_conv | ja_conv | en_conv |
| --- | --- | --- | --- |
| natural (server endpointing) | 0.094, 14.8 / 25.1 s | 0.291, 6.5 / 27.5 s | 0.089, 10.7 / 23.9 s |
| cut_pause | 0.105, 2.9 / 7.9 s | 0.313, 2.5 / 6.3 s | 0.083, 3.6 / 8.7 s |
| cut_smart | 0.109, 3.2 / 7.9 s | 0.328, 2.7 / 13.9 s | 0.080, 4.0 / 8.7 s |
| cut_smart_ctx | 0.107, 3.1 / 7.8 s | 0.315, 3.2 / 6.9 s | 0.064, 3.8 / 8.1 s |
| cut_app | 0.135, 3.1 / 5.7 s | 0.330, 3.1 / 8.2 s | 0.079, 2.6 / 4.7 s |
| cut_fixed3 | 0.216, 2.2 / 3.8 s | 0.372, 2.7 / 6.4 s | 0.106, 1.9 / 3.1 s |

- Cutting at pauses brings Qwen Audio finals from 6.5-14.8 s (p50) to 2.5-4.0 s. CER stays within about 2 points of the natural run (`cut_pause`: +1.1 ko, +2.2 ja, -0.6 en).
- Smart Turn cuts (`cut_smart`) are not better than cutting at any 200 ms pause after 2 s (`cut_pause`).
- Cuts during speech lose words. `cut_fixed3` cuts during speech 38 times in Korean and loses 60 more characters than `cut_pause` (`deleted` 104 vs 44). That is roughly 1.6 characters per cut; Japanese and English lose roughly 2 and 0.8.
- `cut_app` (6 s cut with 300 ms replay) duplicates and loses text in Korean (`inserted` 20 vs 8, `deleted` 57 vs 44 against `cut_pause`).
- p90 stays at 6-9 s for pause cuts. English has few pauses of 200 ms or more: the median `cut_pause` segment is 9.0 s long, and 8 of 17 segments end at the 10 s fallback.
- `finish-task` to `task-finished` is 0.25-0.45 s (p50). No task failed. `usage` is not reported for this model.

### Previous-text context (`cut_smart_ctx`)

- `payload.input.context` was accepted without errors.
- `cut_smart` and `cut_smart_ctx` send the same audio segments. The first task, which has no context, gives the same final text in both runs for all three languages. Later finals differ in 8 of 24 Korean and 12 of 16 English finals; the Japanese runs produced 38 and 36 finals, of which only 6 match in order. The context therefore changes the output.
- Deleted characters fell in every language (44 → 37, 91 → 67, 51 → 28). CER fell by 0.2-1.6 points. This is one run per variant.

### Releasing Qwen Audio partial text inside cut tasks

Offline replay with `interim_release.simulate` on these runs (not a script output). Latency is per character from its Soniox end timestamp, p50 / p90.

| variant | ko_conv | ja_conv | en_conv |
| --- | --- | --- | --- |
| cut_smart, finals only | 3.5 / 7.9 s, CER 0.109 | 1.9 / 4.5 s, 0.328 | 4.3 / 8.3 s, 0.080 |
| cut_smart, partial W=0.6 + final remainder | 1.8 / 3.2 s, 0.213 | 1.8 / 4.3 s, 0.334 | 3.3 / 6.0 s, 0.080 |
| cut_smart, partial W=1.0 + final remainder | 2.3 / 3.6 s, 0.164 | 1.9 / 4.4 s, 0.328 | 3.5 / 6.3 s, 0.080 |
| cut_pause, partial W=0.6 + final remainder | 2.0 / 4.0 s, 0.183 | 1.9 / 4.7 s, 0.321 | 3.0 / 5.5 s, 0.091 |

- Korean partial text of Qwen Audio is not safe to release: CER rises 5-10 points.
- Japanese gains nothing, because short tasks already finish quickly.
- English gains about 1 s at p50 and 2-3 s at p90 with no CER change for `cut_smart`.

## Gemini end variants

Cells: CER, `deleted` characters, then `final` p50 / p90 after the caption line end.

| variant | ko_conv | ja_conv | en_conv |
| --- | --- | --- | --- |
| natural | 0.111, 17, 8.5 / 43.9 s | 0.349, 95, 3.1 / 12.1 s | 0.093, 39, 8.6 / 40.5 s |
| end_app (24 / 30 / 22 ends) | 0.126, 32, 2.4 / 4.7 s | 0.355, 102, 2.6 / 12.1 s | 0.101, 64, 2.6 / 5.1 s |
| end_smart (16 / 37 / 4 ends) | 0.112, 23, 3.5 / 11.5 s | 0.370, 85, 2.5 / 9.2 s | 0.081, 37, 6.8 / 21.9 s |

- `end_app` (the current application: an end at every 4-6 s cut) makes finals fast but loses words: `deleted` +15 in Korean and +25 in English.
- `end_smart` keeps CER near natural but sends few ends in English, so English finals stay slow.
- Finalize acknowledgements sometimes take seconds (Japanese p90 4.7 s for `end_app` and 9.3 s for `end_smart`).
- Interim release on these runs is in `../interim-20261020T190834Z/NOTE.md`.
