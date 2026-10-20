# Decision model release probe

Replays recorded partial-transcript streams through decision models on OpenRouter and measures where and when each policy would release text for translation. Text sent to the models is lowercased and has punctuation removed.

- Labels are sentence-final punctuation in the Soniox final transcript of the same audio (hindsight), mapped onto each provider's transcript. Commas count as clause labels.
- `sentence` / `clause` / `mid`: share of releases within 1 characters of a sentence label, of a comma label, or neither.
- `recall`: share of sentence labels that some release hits.
- `latency`: seconds from the audio end of a labeled sentence's last character (Soniox token timestamp) to the first release that covers it. Cells show `p50 / p90`.
- Proposal: a choice question over numbered markers placed after each new word (each new character in Japanese, at most 12); the chosen marker becomes a candidate when its probability is at least 0.5.
- `immediate p>=x`: release a candidate as soon as the proposal answer arrives, when its probability is at least x.
- `verify v>=x`: every candidate is checked once more after 3 English / 2 Korean / 3 Japanese characters of right context arrive; release when that answer is at least x. `+silence` also releases a candidate when the transcript stays unchanged for 0.8 s.
- `punct rule` releases at sentence-final punctuation in the partial transcript, which is close to how the labels were made. `natural final` releases final text as it arrives.

## Read speech (FLEURS)

| policy | releases | sentence | clause | mid | recall | latency |
|---|---|---|---|---|---|---|
| natural final | 303 | 20% | 13% | 67% | 26% | 5.81 / 56.15 |
| punct rule | 257 | 77% | 9% | 14% | 98% | 0.97 / 2.28 |
| debounce 0.3s | 1723 | 11% | 11% | 78% | 89% | 1.06 / 1.66 |
| debounce 0.6s | 809 | 15% | 14% | 71% | 58% | 1.57 / 9.51 |

| model | policy | releases | sentence | clause | mid | recall | latency |
|---|---|---|---|---|---|---|---|
| typesafe/jev-1.14 | immediate p>=0.5 | 646 | 17% | 12% | 71% | 46% | 2.28 / 11.96 |
| typesafe/jev-1.14 | immediate p>=0.7 | 245 | 31% | 10% | 59% | 32% | 4.01 / 43.12 |
| typesafe/jev-1.14 | immediate p>=0.9 | 28 | 46% | 25% | 29% | 6% | 13.61 / 36.00 |
| typesafe/jev-1.14 | verify v>=0.5 | 105 | 55% | 12% | 32% | 29% | 7.11 / 54.97 |
| typesafe/jev-1.14 | verify v>=0.5 +silence | 204 | 32% | 15% | 53% | 32% | 5.22 / 23.33 |
| typesafe/jev-1.14 | verify v>=0.7 | 39 | 87% | 3% | 10% | 18% | 13.57 / 46.48 |
| typesafe/jev-1.14 | verify v>=0.7 +silence | 156 | 30% | 15% | 54% | 23% | 6.58 / 35.25 |
| typesafe/jev-1.14 | verify v>=0.9 | 2 | 100% | 0% | 0% | 1% | 36.31 / 65.00 |
| typesafe/jev-1.14 | verify v>=0.9 +silence | 132 | 20% | 18% | 61% | 12% | 10.20 / 59.52 |
| inception/mercury-decide | immediate p>=0.5 | 868 | 16% | 10% | 74% | 60% | 1.94 / 4.19 |
| inception/mercury-decide | immediate p>=0.7 | 753 | 17% | 10% | 73% | 57% | 2.03 / 5.14 |
| inception/mercury-decide | immediate p>=0.9 | 517 | 20% | 10% | 71% | 46% | 2.29 / 7.97 |
| inception/mercury-decide | verify v>=0.5 | 203 | 49% | 16% | 35% | 51% | 3.78 / 17.19 |
| inception/mercury-decide | verify v>=0.5 +silence | 364 | 28% | 14% | 58% | 48% | 2.89 / 11.97 |
| inception/mercury-decide | verify v>=0.7 | 193 | 51% | 16% | 33% | 50% | 4.04 / 17.57 |
| inception/mercury-decide | verify v>=0.7 +silence | 355 | 28% | 14% | 58% | 47% | 2.92 / 11.97 |
| inception/mercury-decide | verify v>=0.9 | 158 | 61% | 13% | 26% | 48% | 4.48 / 17.73 |
| inception/mercury-decide | verify v>=0.9 +silence | 338 | 29% | 14% | 57% | 47% | 2.96 / 12.89 |
| perplexity/pplx-decider-v1.2 | immediate p>=0.5 | 427 | 39% | 14% | 47% | 76% | 1.95 / 9.84 |
| perplexity/pplx-decider-v1.2 | immediate p>=0.7 | 285 | 48% | 13% | 39% | 65% | 2.20 / 13.86 |
| perplexity/pplx-decider-v1.2 | immediate p>=0.9 | 100 | 66% | 12% | 22% | 33% | 9.03 / 34.89 |
| perplexity/pplx-decider-v1.2 | verify v>=0.5 | 156 | 79% | 6% | 14% | 63% | 3.65 / 17.51 |
| perplexity/pplx-decider-v1.2 | verify v>=0.5 +silence | 256 | 57% | 13% | 30% | 69% | 2.70 / 11.38 |
| perplexity/pplx-decider-v1.2 | verify v>=0.7 | 104 | 91% | 4% | 5% | 49% | 6.94 / 27.14 |
| perplexity/pplx-decider-v1.2 | verify v>=0.7 +silence | 226 | 58% | 13% | 29% | 62% | 3.28 / 15.36 |
| perplexity/pplx-decider-v1.2 | verify v>=0.9 | 42 | 95% | 2% | 2% | 21% | 20.59 / 74.77 |
| perplexity/pplx-decider-v1.2 | verify v>=0.9 +silence | 193 | 53% | 15% | 32% | 48% | 4.72 / 29.41 |
| perplexity/pplx-decider-flash-v1.2 | immediate p>=0.5 | 316 | 37% | 10% | 53% | 53% | 2.45 / 20.24 |
| perplexity/pplx-decider-flash-v1.2 | immediate p>=0.7 | 100 | 60% | 10% | 30% | 30% | 6.84 / 39.38 |
| perplexity/pplx-decider-flash-v1.2 | immediate p>=0.9 | 16 | 75% | 6% | 19% | 6% | 22.68 / 95.70 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.5 | 60 | 90% | 2% | 8% | 28% | 9.96 / 58.47 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.5 +silence | 135 | 61% | 11% | 27% | 39% | 7.23 / 38.80 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.7 | 26 | 96% | 4% | 0% | 13% | 12.77 / 85.85 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.7 +silence | 110 | 57% | 14% | 29% | 29% | 12.47 / 69.15 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.9 | 12 | 100% | 0% | 0% | 6% | 16.18 / 34.09 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.9 +silence | 104 | 55% | 14% | 31% | 26% | 12.89 / 69.23 |

## Conversation (YODAS2)

| policy | releases | sentence | clause | mid | recall | latency |
|---|---|---|---|---|---|---|
| natural final | 331 | 46% | 10% | 44% | 29% | 5.07 / 17.85 |
| punct rule | 581 | 82% | 6% | 11% | 89% | 1.00 / 3.35 |
| debounce 0.3s | 1440 | 28% | 11% | 61% | 74% | 1.07 / 2.16 |
| debounce 0.6s | 785 | 35% | 12% | 53% | 53% | 1.71 / 4.93 |

| model | policy | releases | sentence | clause | mid | recall | latency |
|---|---|---|---|---|---|---|---|
| typesafe/jev-1.14 | immediate p>=0.5 | 565 | 39% | 11% | 50% | 38% | 2.86 / 15.14 |
| typesafe/jev-1.14 | immediate p>=0.7 | 165 | 59% | 8% | 33% | 18% | 7.73 / 31.09 |
| typesafe/jev-1.14 | immediate p>=0.9 | 5 | 80% | 0% | 20% | 1% | 21.92 / 58.17 |
| typesafe/jev-1.14 | verify v>=0.5 | 167 | 70% | 8% | 22% | 22% | 8.02 / 36.53 |
| typesafe/jev-1.14 | verify v>=0.5 +silence | 247 | 58% | 11% | 31% | 27% | 5.04 / 24.03 |
| typesafe/jev-1.14 | verify v>=0.7 | 58 | 86% | 2% | 12% | 9% | 10.14 / 36.41 |
| typesafe/jev-1.14 | verify v>=0.7 +silence | 165 | 59% | 10% | 31% | 20% | 6.40 / 27.65 |
| typesafe/jev-1.14 | verify v>=0.9 | 3 | 100% | 0% | 0% | 1% | 25.92 / 56.85 |
| typesafe/jev-1.14 | verify v>=0.9 +silence | 130 | 52% | 12% | 35% | 14% | 8.70 / 32.05 |
| inception/mercury-decide | immediate p>=0.5 | 578 | 35% | 12% | 53% | 38% | 2.79 / 7.43 |
| inception/mercury-decide | immediate p>=0.7 | 449 | 34% | 13% | 53% | 30% | 3.54 / 9.73 |
| inception/mercury-decide | immediate p>=0.9 | 222 | 39% | 11% | 50% | 17% | 6.70 / 20.78 |
| inception/mercury-decide | verify v>=0.5 | 202 | 64% | 14% | 22% | 25% | 6.50 / 16.78 |
| inception/mercury-decide | verify v>=0.5 +silence | 299 | 50% | 14% | 36% | 29% | 4.56 / 11.26 |
| inception/mercury-decide | verify v>=0.7 | 190 | 66% | 13% | 21% | 24% | 6.91 / 17.49 |
| inception/mercury-decide | verify v>=0.7 +silence | 289 | 51% | 13% | 36% | 28% | 4.58 / 12.35 |
| inception/mercury-decide | verify v>=0.9 | 147 | 70% | 13% | 17% | 20% | 8.38 / 21.32 |
| inception/mercury-decide | verify v>=0.9 +silence | 259 | 50% | 14% | 36% | 26% | 4.79 / 14.29 |
| perplexity/pplx-decider-v1.2 | immediate p>=0.5 | 511 | 52% | 14% | 35% | 48% | 2.82 / 11.17 |
| perplexity/pplx-decider-v1.2 | immediate p>=0.7 | 312 | 58% | 11% | 31% | 34% | 3.96 / 22.38 |
| perplexity/pplx-decider-v1.2 | immediate p>=0.9 | 122 | 72% | 7% | 20% | 16% | 10.00 / 48.05 |
| perplexity/pplx-decider-v1.2 | verify v>=0.5 | 229 | 81% | 11% | 8% | 35% | 6.35 / 17.81 |
| perplexity/pplx-decider-v1.2 | verify v>=0.5 +silence | 318 | 69% | 13% | 19% | 41% | 4.10 / 13.71 |
| perplexity/pplx-decider-v1.2 | verify v>=0.7 | 155 | 89% | 6% | 5% | 26% | 8.62 / 28.26 |
| perplexity/pplx-decider-v1.2 | verify v>=0.7 +silence | 278 | 69% | 12% | 19% | 36% | 4.92 / 16.49 |
| perplexity/pplx-decider-v1.2 | verify v>=0.9 | 64 | 92% | 5% | 3% | 11% | 18.45 / 43.41 |
| perplexity/pplx-decider-v1.2 | verify v>=0.9 +silence | 229 | 66% | 12% | 22% | 28% | 5.99 / 21.14 |
| perplexity/pplx-decider-flash-v1.2 | immediate p>=0.5 | 264 | 56% | 11% | 34% | 26% | 5.57 / 37.77 |
| perplexity/pplx-decider-flash-v1.2 | immediate p>=0.7 | 73 | 78% | 4% | 18% | 11% | 9.65 / 54.59 |
| perplexity/pplx-decider-flash-v1.2 | immediate p>=0.9 | 5 | 100% | 0% | 0% | 1% | 24.01 / 58.47 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.5 | 78 | 76% | 8% | 17% | 12% | 12.70 / 41.97 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.5 +silence | 129 | 72% | 9% | 19% | 18% | 8.14 / 38.40 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.7 | 23 | 78% | 9% | 13% | 4% | 25.40 / 66.10 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.7 +silence | 93 | 75% | 10% | 15% | 13% | 9.88 / 42.93 |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.9 | 0 | - | - | - | 0% | - |
| perplexity/pplx-decider-flash-v1.2 | verify v>=0.9 +silence | 75 | 75% | 11% | 15% | 10% | 12.12 / 47.93 |

## Calls

| model | calls | errors | retries | calls/min of audio | RTT p50 / p90 | tokens/call | cost | cost per audio hour |
|---|---|---|---|---|---|---|---|---|
| typesafe/jev-1.14 | 5883 | 0 | 0 | 117 | 0.377 / 0.425 | 534 | $0.1321 | $0.158 |
| inception/mercury-decide | 5922 | 0 | 4 | 118 | 0.436 / 0.615 | 267 | $0.0316 | $0.038 |
| perplexity/pplx-decider-v1.2 | 4316 | 0 | 0 | 86 | 0.567 / 0.627 | 290 | $0.0251 | $0.030 |
| perplexity/pplx-decider-flash-v1.2 | 4995 | 0 | 152 | 99 | 0.473 / 0.498 | 290 | $0.0145 | $0.017 |

Per-language and per-provider rows are in `summary.json`.
