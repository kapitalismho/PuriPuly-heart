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
| punctuation only | 0.0 | - | - | - | 3.37 / 8.07 | 2.04 / 6.03 | 0 | - | - | - | - |
| time cut (latest word) | 4.91 | 0% | 10% | 90% | 2.58 / 4.07 | 1.92 / 3.77 | 66 | 20% | 44% | 36% | 17% |
| provider comma | 2.03 | 5% | 87% | 8% | 3.24 / 6.63 | 2.04 / 5.20 | 33 | 52% | 24% | 24% | 3% |
| longest pause | 4.0 | 0% | 21% | 79% | 2.97 / 4.79 | 2.02 / 4.15 | 52 | 37% | 37% | 27% | 10% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.11 | 0% | 0% | 100% | 3.37 / 7.96 | 2.04 / 5.95 | 2 | 0% | 0% | 100% | 100% |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 1.17 | 5% | 27% | 68% | 3.37 / 6.75 | 2.05 / 5.65 | 16 | 56% | 12% | 31% | 19% |
| pplx-decider-v1.2 Q2 p>=0.3 | 3.04 | 0% | 7% | 93% | 3.16 / 5.10 | 2.04 / 4.51 | 39 | 26% | 44% | 31% | 13% |
| pplx-decider-v1.2 Q2 p>=0.5 | 2.88 | 0% | 20% | 80% | 3.22 / 5.46 | 2.04 / 4.70 | 37 | 32% | 43% | 24% | 8% |
| pplx-decider-v1.2 Q2 p>=0.7 | 2.03 | 0% | 29% | 71% | 3.31 / 6.54 | 2.04 / 5.13 | 31 | 48% | 29% | 23% | 6% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 2.93 | 0% | 20% | 80% | 3.22 / 5.40 | 2.04 / 4.67 | 37 | 32% | 43% | 24% | 11% |
| mercury-decide Q1 p>=0.5 | 0.8 | 0% | 13% | 87% | 3.35 / 6.99 | 2.05 / 5.46 | 11 | 18% | 27% | 55% | 36% |
| mercury-decide Q1 p>=0.5 +forced | 1.6 | 3% | 23% | 73% | 3.34 / 6.51 | 2.05 / 5.34 | 20 | 35% | 30% | 35% | 20% |
| mercury-decide Q2 p>=0.3 | 3.41 | 0% | 8% | 92% | 3.06 / 4.79 | 2.04 / 4.34 | 44 | 25% | 32% | 43% | 14% |
| mercury-decide Q2 p>=0.5 | 3.15 | 0% | 10% | 90% | 3.15 / 5.09 | 2.05 / 4.50 | 39 | 21% | 33% | 46% | 15% |
| mercury-decide Q2 p>=0.7 | 2.45 | 0% | 13% | 87% | 3.22 / 5.58 | 2.05 / 4.73 | 34 | 26% | 35% | 38% | 18% |
| mercury-decide Q2 p>=0.5 +forced | 3.15 | 0% | 10% | 90% | 3.15 / 5.09 | 2.05 / 4.50 | 39 | 21% | 33% | 46% | 15% |

## en_conv

| policy | splits/min | sentence | clause | mid | latency long | latency all | judged | same | minor | changed | premature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| punctuation only | 0.0 | - | - | - | 3.75 / 9.37 | 2.66 / 7.76 | 0 | - | - | - | - |
| time cut (latest word) | 8.48 | 0% | 2% | 98% | 2.68 / 4.13 | 2.21 / 3.97 | 30 | 17% | 57% | 27% | 10% |
| provider comma | 3.68 | 9% | 78% | 13% | 3.54 / 7.25 | 2.65 / 6.35 | 18 | 61% | 22% | 17% | 0% |
| longest pause | 7.36 | 0% | 11% | 89% | 3.17 / 5.05 | 2.47 / 4.60 | 25 | 24% | 48% | 28% | 16% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.0 | - | - | - | 3.75 / 9.37 | 2.66 / 7.76 | 0 | - | - | - | - |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 3.04 | 5% | 32% | 63% | 3.72 / 7.22 | 2.67 / 6.64 | 13 | 62% | 15% | 23% | 8% |
| pplx-decider-v1.2 Q2 p>=0.3 | 5.92 | 0% | 5% | 95% | 3.40 / 5.44 | 2.63 / 4.99 | 21 | 24% | 52% | 24% | 10% |
| pplx-decider-v1.2 Q2 p>=0.5 | 5.76 | 0% | 19% | 81% | 3.46 / 5.92 | 2.64 / 5.32 | 21 | 33% | 48% | 19% | 10% |
| pplx-decider-v1.2 Q2 p>=0.7 | 3.52 | 0% | 36% | 64% | 3.65 / 7.38 | 2.65 / 6.43 | 17 | 71% | 18% | 12% | 0% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 5.92 | 0% | 19% | 81% | 3.46 / 5.80 | 2.64 / 5.25 | 21 | 33% | 48% | 19% | 14% |
| mercury-decide Q1 p>=0.5 | 1.44 | 0% | 0% | 100% | 3.71 / 7.59 | 2.68 / 6.80 | 6 | 33% | 33% | 33% | 17% |
| mercury-decide Q1 p>=0.5 +forced | 3.68 | 4% | 22% | 74% | 3.69 / 6.95 | 2.68 / 6.39 | 14 | 50% | 29% | 21% | 7% |
| mercury-decide Q2 p>=0.3 | 7.04 | 0% | 2% | 98% | 3.23 / 4.95 | 2.57 / 4.66 | 26 | 27% | 35% | 38% | 15% |
| mercury-decide Q2 p>=0.5 | 6.56 | 0% | 5% | 95% | 3.36 / 5.32 | 2.60 / 4.94 | 23 | 17% | 39% | 43% | 17% |
| mercury-decide Q2 p>=0.7 | 4.96 | 0% | 6% | 94% | 3.46 / 6.11 | 2.62 / 5.40 | 20 | 25% | 45% | 30% | 15% |
| mercury-decide Q2 p>=0.5 +forced | 6.56 | 0% | 5% | 95% | 3.36 / 5.32 | 2.60 / 4.94 | 23 | 17% | 39% | 43% | 17% |

## ja_conv

| policy | splits/min | sentence | clause | mid | latency long | latency all | judged | same | minor | changed | premature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| punctuation only | 0.0 | - | - | - | 2.98 / 5.80 | 1.54 / 3.77 | 0 | - | - | - | - |
| time cut (latest word) | 2.4 | 0% | 13% | 87% | 2.38 / 3.94 | 1.54 / 3.17 | 14 | 0% | 36% | 64% | 36% |
| provider comma | 1.12 | 0% | 100% | 0% | 2.81 / 4.87 | 1.54 / 3.52 | 7 | 29% | 14% | 57% | 14% |
| longest pause | 1.6 | 0% | 90% | 10% | 2.80 / 4.54 | 1.54 / 3.54 | 9 | 33% | 44% | 22% | 0% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.32 | 0% | 0% | 100% | 2.98 / 5.50 | 1.54 / 3.72 | 2 | 0% | 0% | 100% | 100% |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 0.32 | 0% | 0% | 100% | 2.98 / 5.50 | 1.54 / 3.72 | 2 | 0% | 0% | 100% | 100% |
| pplx-decider-v1.2 Q2 p>=0.3 | 1.28 | 0% | 12% | 88% | 2.74 / 4.64 | 1.54 / 3.57 | 7 | 0% | 43% | 57% | 43% |
| pplx-decider-v1.2 Q2 p>=0.5 | 0.96 | 0% | 67% | 33% | 2.93 / 4.77 | 1.54 / 3.68 | 5 | 0% | 60% | 40% | 20% |
| pplx-decider-v1.2 Q2 p>=0.7 | 0.96 | 0% | 50% | 50% | 2.96 / 4.90 | 1.54 / 3.71 | 5 | 0% | 40% | 60% | 40% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 0.96 | 0% | 67% | 33% | 2.93 / 4.77 | 1.54 / 3.68 | 5 | 0% | 60% | 40% | 20% |
| mercury-decide Q1 p>=0.5 | 0.8 | 0% | 40% | 60% | 2.88 / 4.83 | 1.54 / 3.63 | 4 | 0% | 0% | 100% | 75% |
| mercury-decide Q1 p>=0.5 +forced | 0.8 | 0% | 40% | 60% | 2.88 / 4.83 | 1.54 / 3.63 | 4 | 0% | 0% | 100% | 75% |
| mercury-decide Q2 p>=0.3 | 1.28 | 0% | 50% | 50% | 2.86 / 4.59 | 1.54 / 3.54 | 7 | 14% | 14% | 71% | 29% |
| mercury-decide Q2 p>=0.5 | 0.96 | 0% | 50% | 50% | 2.88 / 4.70 | 1.54 / 3.59 | 5 | 20% | 0% | 80% | 40% |
| mercury-decide Q2 p>=0.7 | 0.8 | 0% | 60% | 40% | 2.91 / 4.80 | 1.54 / 3.62 | 4 | 0% | 0% | 100% | 75% |
| mercury-decide Q2 p>=0.5 +forced | 0.96 | 0% | 50% | 50% | 2.88 / 4.70 | 1.54 / 3.59 | 5 | 20% | 0% | 80% | 40% |

## ko_conv

| policy | splits/min | sentence | clause | mid | latency long | latency all | judged | same | minor | changed | premature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| punctuation only | 0.0 | - | - | - | 2.62 / 4.96 | 1.66 / 3.97 | 0 | - | - | - | - |
| time cut (latest word) | 3.84 | 0% | 25% | 75% | 2.32 / 3.93 | 1.62 / 3.54 | 22 | 36% | 32% | 32% | 14% |
| provider comma | 1.28 | 0% | 100% | 0% | 2.53 / 4.80 | 1.66 / 3.87 | 8 | 50% | 38% | 12% | 0% |
| longest pause | 3.04 | 0% | 11% | 89% | 2.51 / 4.08 | 1.66 / 3.63 | 18 | 56% | 17% | 28% | 6% |
| pplx-decider-v1.2 Q1 p>=0.5 | 0.0 | - | - | - | 2.62 / 4.96 | 1.66 / 3.97 | 0 | - | - | - | - |
| pplx-decider-v1.2 Q1 p>=0.5 +forced | 0.16 | 0% | 0% | 100% | 2.62 / 4.96 | 1.66 / 3.97 | 1 | 100% | 0% | 0% | 0% |
| pplx-decider-v1.2 Q2 p>=0.3 | 1.92 | 0% | 8% | 92% | 2.62 / 4.38 | 1.66 / 3.80 | 11 | 45% | 27% | 27% | 0% |
| pplx-decider-v1.2 Q2 p>=0.5 | 1.92 | 0% | 0% | 100% | 2.62 / 4.39 | 1.66 / 3.81 | 11 | 45% | 27% | 27% | 0% |
| pplx-decider-v1.2 Q2 p>=0.7 | 1.6 | 0% | 0% | 100% | 2.62 / 4.43 | 1.66 / 3.82 | 9 | 33% | 44% | 22% | 0% |
| pplx-decider-v1.2 Q2 p>=0.5 +forced | 1.92 | 0% | 0% | 100% | 2.62 / 4.39 | 1.66 / 3.81 | 11 | 45% | 27% | 27% | 0% |
| mercury-decide Q1 p>=0.5 | 0.16 | 0% | 0% | 100% | 2.62 / 4.96 | 1.66 / 3.96 | 1 | 0% | 100% | 0% | 0% |
| mercury-decide Q1 p>=0.5 +forced | 0.32 | 0% | 0% | 100% | 2.62 / 4.91 | 1.66 / 3.96 | 2 | 0% | 100% | 0% | 0% |
| mercury-decide Q2 p>=0.3 | 1.92 | 0% | 0% | 100% | 2.62 / 4.35 | 1.66 / 3.78 | 11 | 27% | 36% | 36% | 0% |
| mercury-decide Q2 p>=0.5 | 1.92 | 0% | 8% | 92% | 2.62 / 4.42 | 1.66 / 3.82 | 11 | 27% | 36% | 36% | 0% |
| mercury-decide Q2 p>=0.7 | 1.6 | 0% | 10% | 90% | 2.62 / 4.43 | 1.66 / 3.84 | 10 | 40% | 30% | 30% | 0% |
| mercury-decide Q2 p>=0.5 +forced | 1.92 | 0% | 8% | 92% | 2.62 / 4.42 | 1.66 / 3.82 | 11 | 27% | 36% | 36% | 0% |

## Calls

| model | role | calls | errors | cost | RTT p50 |
|---|---|---|---|---|---|
| perplexity/pplx-decider-v1.2 | decision | 1724 | 0 | $0.0089 | 0.572 |
| inception/mercury-decide | decision | 1724 | 0 | $0.0080 | 0.438 |
| google/gemma-4-26b-a4b-it | translation | 624 | 0 | $0.0320 | 0.518 |
| openai/gpt-6.1-sol | judge | 243 | 0 | $0.3807 | 2.768 |
