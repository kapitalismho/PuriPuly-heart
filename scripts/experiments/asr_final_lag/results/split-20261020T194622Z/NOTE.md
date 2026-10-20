# Notes on this probe

- Two runs of the probe exist:
  - `../split-20261020T190859Z` used Soniox and Gemini from `../20261020T185455Z`.
  - This run used Soniox, Deepgram and Gemini from `../20261020T193903Z`.
- Both runs used the same audio, trigger (3.0 s), translator and judge.
- The policies below have 23-66 judged utterances each, pooled over all data. Per-language and per-provider cells have far fewer, so only pooled figures are compared.

## Pooled results of both runs

Cells: first run / this run.

| policy | latency long p90 | same | changed | premature |
| --- | --- | --- | --- | --- |
| punctuation only | 8.6 / 8.1 s | - | - | - |
| time cut (latest word) | 4.0 / 4.1 s | 17% / 20% | 31% / 36% | 8% / 17% |
| provider comma | 7.0 / 6.6 s | 57% / 52% | 17% / 24% | 4% / 3% |
| longest pause | 4.8 / 4.8 s | 44% / 37% | 16% / 27% | 12% / 10% |
| pplx-decider Q2 p>=0.5 | 5.1 / 5.5 s | 33% / 32% | 44% / 24% | 22% / 8% |
| pplx-decider Q2 p>=0.7 | 5.7 / 6.5 s | 50% / 48% | 35% / 23% | 12% / 6% |
| mercury-decide Q2 p>=0.5 | 5.0 / 5.1 s | 38% / 21% | 34% / 46% | 10% / 15% |

- Splitting long utterances moves `latency long` p50 only from about 3.4 s to 2.5-3.3 s, because a split can happen at the earliest 3.4 s after the text starts. The gain is at p90: 8.1-8.6 s down to 4-5 s for pause and time cuts.
- No decision-model policy beats the simple rules in both runs.
  - Q2 results swing between runs: pplx-decider at p>=0.5 had 44% `changed` in the first run and 24% in this one.
  - Q1 alone rarely passes its threshold (0.1-0.8 splits per minute).
- The provider comma has the fewest premature subtitles (3-4%) and the most `same` verdicts in both runs, but it splits rarely, so p90 stays at 6.6-7.0 s.
- The longest pause reaches p90 4.8 s with 10-12% premature subtitles. A blind time cut is about as fast and has the most `changed` verdicts (31-36%).
- Japanese splits are judged `changed` far more often than Korean or English splits under most policies, but the Japanese cells hold only 2-14 utterances.
- Cost of this run: $0.43 (decisions $0.017, translation $0.032, judge $0.38).
