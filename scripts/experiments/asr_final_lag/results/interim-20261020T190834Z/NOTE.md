# Notes on this replay

- Sources: the natural and end runs of `../20261020T185455Z`. The earlier replay `../interim-20261020T185034Z` used the natural runs of `../20261020T070225Z` (same audio).
- Cells: latency p50 / p90 per character, then CER. "Interim" is `interim W=0.6 + final remainder`.

## Two natural runs of the same audio

| data | run | finals only | interim |
| --- | --- | --- | --- |
| ko_conv | 070225Z | 10.4 / 43.1 s, 0.111 | 2.3 / 4.6 s, 0.119 |
| ko_conv | 185455Z | 8.2 / 42.9 s, 0.111 | 2.1 / 4.3 s, 0.130 |
| ja_conv | 070225Z | 2.4 / 7.3 s, 0.349 | 1.8 / 4.2 s, 0.420 |
| ja_conv | 185455Z | 2.2 / 4.6 s, 0.349 | 1.7 / 3.3 s, 0.341 |
| en_conv | 070225Z | 10.8 / 43.6 s, 0.093 | 3.9 / 11.1 s, 0.099 |
| en_conv | 185455Z | 10.7 / 43.6 s, 0.093 | 3.6 / 10.9 s, 0.091 |

- Gemini finals had the same CER in both runs; interim text did not.
- The 7-point Japanese CER rise of the first replay did not repeat (-0.8 points this time). Korean rose by 0.8 and 1.9 points, English changed by +0.6 and -0.2. Treat the interim CER cost as 0-2 points with occasional bad runs, and measure more runs before relying on a per-language figure.
- English p90 stays near 11 s: long run-on speech has no sentence-final punctuation in the interim text, so it waits for a final. The split policies in `../split-20261020T194622Z` address this case.

## Interim release with forced ends

| data | end_smart + interim | end_app + interim | end_app finals only |
| --- | --- | --- | --- |
| ko_conv | 1.9 / 4.1 s, 0.124 | 1.7 / 3.3 s, 0.180 | 2.6 / 5.0 s, 0.126 |
| ja_conv | 1.5 / 3.4 s, 0.385 | 1.5 / 2.7 s, 0.367 | 1.7 / 3.5 s, 0.355 |
| en_conv | 3.5 / 10.8 s, 0.089 | 2.6 / 5.5 s, 0.130 | 2.9 / 5.5 s, 0.101 |

- Ends at 4-6 s cuts break interim release: the final after a forced end no longer matches the interim text, and CER rises to 0.180 in Korean and 0.130 in English.
- Ends at Smart Turn cuts and turn ends work with interim release, at a cost similar to the natural runs.
