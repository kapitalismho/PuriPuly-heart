# Long-utterance split probe

Replays recorded partial transcripts. Punctuation in the partial text releases text as usual. When no punctuation release happens for 3.0 s after new text starts, a split policy may release an earlier part of the unreleased text. A split costs a 0.4 s finalize allowance plus, for decision models, the slowest round trip of that trigger's calls.

- `time cut`: the latest word end with right context (a stand-in for the acoustic 4-6 s rules). `provider comma`: the latest comma in the partial text. `longest pause`: the word end with the largest gap of at least 0.25 s in Soniox word timing.
- `Q1`: sentence or clause end question. `Q2`: translation question with the target language. Each word end with right context is asked once per question. A policy releases the latest word end at or above the threshold; `+forced` releases the highest-scored word end at the second trigger when none passes.
- `sentence` / `clause` / `mid`: share of splits within 1 character of a Soniox sentence-final punctuation label, a comma label, or neither.
- `latency long`: seconds from each character's Soniox end timestamp to its release, for characters in utterances that reached a trigger. `latency all`: every character. Cells show `p50 / p90`.
- `judged`: split utterances translated chunk by chunk with the application prompt and `google/gemma-4-26b-a4b-it`, then compared with the whole-utterance translation by `openai/gpt-6.1-sol`. `premature`: an earlier subtitle misleads before the next one.

## All data

| policy | splits/min | sentence | clause | mid | latency long | latency all | judged | same | minor | changed | premature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| punctuation only | 0.0 | - | - | - | 3.43 / 8.64 | 2.06 / 6.37 | 0 | - | - | - | - |
| time cut (latest word) | 5.76 | 0% | 6% | 94% | 2.48 / 3.96 | 1.88 / 3.68 | 52 | 17% | 52% | 31% | 8% |
| provider comma | 2.08 | 8% | 92% | 0% | 3.24 / 6.96 | 2.04 / 5.37 | 23 | 57% | 26% | 17% | 4% |
| longest pause | 4.88 | 0% | 16% | 84% | 2.94 / 4.75 | 1.99 / 4.07 | 43 | 44% | 40% | 16% | 12% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.16 | 0% | 50% | 50% | 3.42 / 8.18 | 2.06 / 6.15 | 2 | 50% | 0% | 50% | 50% |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 1.28 | 6% | 19% | 75% | 3.42 / 7.15 | 2.06 / 5.85 | 9 | 11% | 44% | 44% | 22% |
| pplx-decider-v1.2 Q2 p>=0.3 | 3.44 | 0% | 5% | 95% | 3.01 / 4.90 | 2.03 / 4.46 | 29 | 24% | 45% | 31% | 10% |
| pplx-decider-v1.2 Q2 p>=0.5 | 3.28 | 0% | 15% | 85% | 3.09 / 5.10 | 2.04 / 4.59 | 27 | 33% | 22% | 44% | 22% |
| pplx-decider-v1.2 Q2 p>=0.7 | 2.64 | 0% | 24% | 76% | 3.19 / 5.66 | 2.04 / 4.80 | 26 | 50% | 15% | 35% | 12% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 3.28 | 0% | 15% | 85% | 3.09 / 5.10 | 2.04 / 4.59 | 27 | 33% | 22% | 44% | 22% |
| mercury-decide Q1 p>=0.5 | 0.72 | 0% | 22% | 78% | 3.36 / 7.12 | 2.06 / 5.64 | 8 | 25% | 38% | 38% | 12% |
| mercury-decide Q1 p>=0.5 +forced | 1.76 | 0% | 27% | 73% | 3.32 / 6.35 | 2.06 / 5.25 | 15 | 33% | 40% | 27% | 7% |
| mercury-decide Q2 p>=0.3 | 3.84 | 0% | 6% | 94% | 3.00 / 4.63 | 2.04 / 4.32 | 33 | 30% | 30% | 39% | 21% |
| mercury-decide Q2 p>=0.5 | 3.36 | 2% | 7% | 90% | 3.12 / 5.04 | 2.05 / 4.50 | 29 | 38% | 28% | 34% | 10% |
| mercury-decide Q2 p>=0.7 | 2.4 | 3% | 7% | 90% | 3.19 / 5.53 | 2.05 / 4.73 | 20 | 40% | 25% | 35% | 10% |
| mercury-decide Q2 p>=0.5 +forced | 3.52 | 2% | 7% | 91% | 3.12 / 5.00 | 2.05 / 4.47 | 30 | 40% | 27% | 33% | 10% |

## en_conv

| policy | splits/min | sentence | clause | mid | latency long | latency all | judged | same | minor | changed | premature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| punctuation only | 0.0 | - | - | - | 3.73 / 10.17 | 2.78 / 8.73 | 0 | - | - | - | - |
| time cut (latest word) | 9.36 | 0% | 0% | 100% | 2.57 / 4.02 | 2.19 / 3.89 | 22 | 14% | 59% | 27% | 9% |
| provider comma | 3.6 | 13% | 87% | 0% | 3.50 / 7.71 | 2.77 / 6.93 | 12 | 75% | 17% | 8% | 0% |
| longest pause | 8.64 | 0% | 8% | 92% | 3.06 / 4.94 | 2.52 / 4.56 | 20 | 25% | 50% | 25% | 20% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.24 | 0% | 100% | 0% | 3.68 / 9.78 | 2.78 / 8.30 | 1 | 100% | 0% | 0% | 0% |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 3.36 | 7% | 21% | 71% | 3.67 / 7.68 | 2.78 / 7.15 | 7 | 14% | 57% | 29% | 14% |
| pplx-decider-v1.2 Q2 p>=0.3 | 6.96 | 0% | 0% | 100% | 3.14 / 5.12 | 2.62 / 4.82 | 16 | 25% | 50% | 25% | 12% |
| pplx-decider-v1.2 Q2 p>=0.5 | 6.72 | 0% | 11% | 89% | 3.25 / 5.38 | 2.65 / 5.05 | 15 | 33% | 20% | 47% | 33% |
| pplx-decider-v1.2 Q2 p>=0.7 | 4.8 | 0% | 25% | 75% | 3.42 / 6.34 | 2.70 / 5.72 | 14 | 64% | 7% | 29% | 14% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 6.72 | 0% | 11% | 89% | 3.25 / 5.38 | 2.65 / 5.05 | 15 | 33% | 20% | 47% | 33% |
| mercury-decide Q1 p>=0.5 | 1.44 | 0% | 0% | 100% | 3.66 / 7.83 | 2.77 / 7.05 | 5 | 40% | 40% | 20% | 0% |
| mercury-decide Q1 p>=0.5 +forced | 4.32 | 0% | 22% | 78% | 3.59 / 6.64 | 2.77 / 6.21 | 11 | 36% | 45% | 18% | 0% |
| mercury-decide Q2 p>=0.3 | 7.2 | 0% | 0% | 100% | 3.08 / 4.75 | 2.59 / 4.55 | 16 | 25% | 38% | 38% | 31% |
| mercury-decide Q2 p>=0.5 | 6.24 | 0% | 0% | 100% | 3.29 / 5.28 | 2.69 / 4.98 | 14 | 29% | 50% | 21% | 14% |
| mercury-decide Q2 p>=0.7 | 4.56 | 0% | 5% | 95% | 3.37 / 5.99 | 2.71 / 5.36 | 10 | 30% | 40% | 30% | 20% |
| mercury-decide Q2 p>=0.5 +forced | 6.72 | 0% | 0% | 100% | 3.29 / 5.17 | 2.69 / 4.88 | 15 | 33% | 47% | 20% | 13% |

## ja_conv

| policy | splits/min | sentence | clause | mid | latency long | latency all | judged | same | minor | changed | premature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| punctuation only | 0.0 | - | - | - | 3.44 / 5.85 | 1.51 / 3.73 | 0 | - | - | - | - |
| time cut (latest word) | 2.88 | 0% | 0% | 100% | 2.41 / 3.94 | 1.51 / 3.13 | 11 | 0% | 45% | 55% | 9% |
| provider comma | 1.44 | 0% | 100% | 0% | 2.90 / 4.43 | 1.51 / 3.41 | 6 | 50% | 17% | 33% | 17% |
| longest pause | 1.92 | 0% | 88% | 12% | 3.14 / 4.74 | 1.51 / 3.48 | 7 | 57% | 29% | 14% | 14% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.24 | 0% | 0% | 100% | 3.44 / 5.54 | 1.51 / 3.73 | 1 | 0% | 0% | 100% | 100% |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 0.24 | 0% | 0% | 100% | 3.44 / 5.54 | 1.51 / 3.73 | 1 | 0% | 0% | 100% | 100% |
| pplx-decider-v1.2 Q2 p>=0.3 | 1.2 | 0% | 40% | 60% | 3.19 / 4.82 | 1.51 / 3.61 | 5 | 0% | 40% | 60% | 20% |
| pplx-decider-v1.2 Q2 p>=0.5 | 0.96 | 0% | 75% | 25% | 3.35 / 4.88 | 1.51 / 3.62 | 4 | 25% | 25% | 50% | 25% |
| pplx-decider-v1.2 Q2 p>=0.7 | 0.96 | 0% | 75% | 25% | 3.42 / 4.95 | 1.51 / 3.63 | 4 | 25% | 25% | 50% | 25% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 0.96 | 0% | 75% | 25% | 3.35 / 4.88 | 1.51 / 3.62 | 4 | 25% | 25% | 50% | 25% |
| mercury-decide Q1 p>=0.5 | 0.48 | 0% | 100% | 0% | 3.31 / 5.05 | 1.51 / 3.61 | 2 | 0% | 0% | 100% | 50% |
| mercury-decide Q1 p>=0.5 +forced | 0.48 | 0% | 100% | 0% | 3.31 / 5.05 | 1.51 / 3.61 | 2 | 0% | 0% | 100% | 50% |
| mercury-decide Q2 p>=0.3 | 1.2 | 0% | 60% | 40% | 3.12 / 4.59 | 1.51 / 3.45 | 5 | 40% | 20% | 40% | 20% |
| mercury-decide Q2 p>=0.5 | 0.96 | 0% | 75% | 25% | 3.19 / 4.65 | 1.51 / 3.52 | 4 | 50% | 0% | 50% | 25% |
| mercury-decide Q2 p>=0.7 | 0.48 | 0% | 50% | 50% | 3.29 / 5.42 | 1.51 / 3.61 | 2 | 50% | 0% | 50% | 0% |
| mercury-decide Q2 p>=0.5 +forced | 0.96 | 0% | 75% | 25% | 3.19 / 4.65 | 1.51 / 3.52 | 4 | 50% | 0% | 50% | 25% |

## ko_conv

| policy | splits/min | sentence | clause | mid | latency long | latency all | judged | same | minor | changed | premature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| punctuation only | 0.0 | - | - | - | 2.66 / 5.07 | 1.65 / 4.05 | 0 | - | - | - | - |
| time cut (latest word) | 5.04 | 0% | 19% | 81% | 2.29 / 3.87 | 1.61 / 3.52 | 19 | 32% | 47% | 21% | 5% |
| provider comma | 1.2 | 0% | 100% | 0% | 2.56 / 5.03 | 1.65 / 3.98 | 5 | 20% | 60% | 20% | 0% |
| longest pause | 4.08 | 0% | 0% | 100% | 2.52 / 4.05 | 1.64 / 3.65 | 16 | 62% | 31% | 6% | 0% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.0 | - | - | - | 2.66 / 5.07 | 1.65 / 4.05 | 0 | - | - | - | - |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 0.24 | 0% | 0% | 100% | 2.66 / 5.06 | 1.65 / 4.05 | 1 | 0% | 0% | 100% | 0% |
| pplx-decider-v1.2 Q2 p>=0.3 | 2.16 | 0% | 0% | 100% | 2.65 / 4.40 | 1.65 / 3.86 | 8 | 38% | 38% | 25% | 0% |
| pplx-decider-v1.2 Q2 p>=0.5 | 2.16 | 0% | 0% | 100% | 2.65 / 4.41 | 1.65 / 3.86 | 8 | 38% | 25% | 38% | 0% |
| pplx-decider-v1.2 Q2 p>=0.7 | 2.16 | 0% | 0% | 100% | 2.65 / 4.41 | 1.65 / 3.86 | 8 | 38% | 25% | 38% | 0% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 2.16 | 0% | 0% | 100% | 2.65 / 4.41 | 1.65 / 3.86 | 8 | 38% | 25% | 38% | 0% |
| mercury-decide Q1 p>=0.5 | 0.24 | 0% | 0% | 100% | 2.66 / 5.07 | 1.65 / 4.04 | 1 | 0% | 100% | 0% | 0% |
| mercury-decide Q1 p>=0.5 +forced | 0.48 | 0% | 0% | 100% | 2.66 / 5.00 | 1.65 / 4.04 | 2 | 50% | 50% | 0% | 0% |
| mercury-decide Q2 p>=0.3 | 3.12 | 0% | 0% | 100% | 2.60 / 4.32 | 1.65 / 3.80 | 12 | 33% | 25% | 42% | 8% |
| mercury-decide Q2 p>=0.5 | 2.88 | 8% | 0% | 92% | 2.64 / 4.40 | 1.65 / 3.87 | 11 | 45% | 9% | 45% | 0% |
| mercury-decide Q2 p>=0.7 | 2.16 | 11% | 0% | 89% | 2.64 / 4.49 | 1.65 / 3.92 | 8 | 50% | 12% | 38% | 0% |
| mercury-decide Q2 p>=0.5 +forced | 2.88 | 8% | 0% | 92% | 2.64 / 4.40 | 1.65 / 3.87 | 11 | 45% | 9% | 45% | 0% |

## Calls

| model | role | calls | errors | cost | RTT p50 |
|---|---|---|---|---|---|
| perplexity/pplx-decider-v1.2 | decision | 1210 | 0 | $0.0063 | 0.569 |
| inception/mercury-decide | decision | 1210 | 0 | $0.0057 | 0.428 |
| google/gemma-4-26b-a4b-it | translation | 485 | 0 | $0.0249 | 0.492 |
| openai/gpt-6.1-sol | judge | 185 | 0 | $0.2960 | 2.769 |
