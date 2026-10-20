# ASR partial/final timing

Times are seconds after the reference sentence's speech end (audio time = wall time since the first chunk was sent). Cells show `p50 / p90`. Sentence rows exclude the last sentence of each recording; word rows exclude the last 2 s.

- `partial first`: the partial text first covers the sentence end.
- `partial stable`: the partial text covers the sentence end and never retracts.
- `final`: final text covers the sentence end (no forced finalize unless the variant says so).
- `word final` / `word partial`: provider word/token end timestamps to final or first partial arrival.
- `fin ack`: finalize request to the provider's finalize acknowledgement.

## en_us

| provider | variant | error | CER | finals | partial first | partial stable | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.065 | 33 | 0.04 / 0.38 | 0.04 / 0.38 | 0.37 / 0.39 (0) | 2.80 / 4.38 | 0.68 / 1.09 | 0.26 / 0.27 | 0.8541 |
| deepgram | finalize_mid |  | 0.085 | 49 | -0.08 / 0.54 | -0.08 / 0.54 | 0.48 / 1.73 (0) | 1.34 / 2.36 | 0.69 / 1.08 | 0.26 / 0.28 | 0.9071 |
| deepgram | natural |  | 0.070 | 31 | -0.05 / 0.91 | -0.05 / 0.91 | 2.68 / 3.30 (0) | 3.13 / 4.66 | 0.64 / 1.07 | - | 0.8778 |
| deepgram | nointerim |  | 0.070 | 31 | 2.69 / 3.31 | 2.69 / 3.31 | 2.69 / 3.31 (0) | 3.12 / 4.68 | - | - | 0.8924 |
| gemini | natural |  | 0.018 | 12 | 0.25 / 4.45 | 0.25 / 4.45 | 0.25 / 4.45 (0) | - | - | - | 0.5059 |
| qwen | natural |  | 0.018 | 11 | 0.03 / 0.93 | 0.03 / 0.93 | 1.52 / 7.03 (0) | 5.92 / 11.64 | 0.79 / 2.42 | - | 0.5831 |
| qwen | semantic |  | 0.021 | 11 | -0.06 / 1.02 | -0.06 / 1.02 | 1.42 / 6.97 (0) | 6.06 / 11.66 | 0.82 / 2.38 | - | 0.3454 |
| soniox | finalize_gap |  | 0.043 | 42 | -0.12 / 0.37 | -0.12 / 0.37 | 0.35 / 0.39 (0) | 4.13 / 5.64 | 0.18 / 0.68 | 0.23 / 0.27 | 0.7079 |
| soniox | finalize_mid |  | 0.054 | 49 | -0.14 / 0.64 | -0.14 / 0.64 | 0.44 / 1.69 (0) | -0.52 / 1.28 | -1.34 / 0.23 | 0.26 / 0.27 | 0.7063 |
| soniox | natural |  | 0.040 | 58 | -0.12 / 0.55 | -0.12 / 0.55 | 5.11 / 5.78 (0) | 5.71 / 6.58 | 0.72 / 0.74 | - | 0.6138 |

## ja_jp

| provider | variant | error | CER | finals | partial first | partial stable | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.106 | 31 | -1.73 / -1.13 | -1.73 / 0.16 | 0.21 / 0.33 (0) | 2.71 / 4.32 | 0.61 / 1.06 | 0.20 / 0.21 | 0.5908 |
| deepgram | finalize_mid |  | 0.154 | 44 | -1.28 / -0.22 | -1.28 / -0.22 | -0.71 / -0.10 (0) | 1.29 / 2.24 | 0.65 / 1.08 | 0.20 / 0.22 | 0.5892 |
| deepgram | natural |  | 0.099 | 30 | -1.35 / -0.67 | -1.33 / 0.07 | 0.04 / 3.12 (0) | 2.96 / 4.58 | 0.61 / 1.09 | - | 0.5695 |
| deepgram | nointerim |  | 0.099 | 30 | 0.07 / 3.12 | 0.07 / 3.12 | 0.07 / 3.12 (0) | 2.93 / 4.59 | - | - | 0.5966 |
| gemini | natural |  | 0.048 | 12 | -1.13 / -0.74 | -1.13 / -0.74 | -1.13 / -0.74 (0) | - | - | - | 0.5328 |
| qwen | natural |  | 0.023 | 15 | -1.42 / -1.13 | -1.42 / -1.13 | -0.51 / 0.23 (0) | 5.05 / 8.71 | 1.09 / 2.37 | - | 0.2698 |
| qwen | semantic |  | 0.018 | 15 | -1.35 / -0.89 | -1.35 / -0.89 | -0.24 / 0.25 (0) | 5.10 / 8.65 | 1.14 / 2.27 | - | 0.2542 |
| soniox | finalize_gap |  | 0.034 | 39 | -1.58 / -1.15 | -1.58 / -1.15 | 0.36 / 0.39 (0) | 4.68 / 5.96 | 0.34 / 0.64 | 0.23 / 0.27 | 0.7172 |
| soniox | finalize_mid |  | 0.054 | 44 | -1.58 / -1.11 | -1.58 / -1.11 | -0.67 / -0.07 (0) | -0.14 / 1.51 | -0.99 / 0.34 | 0.27 / 0.28 | 0.73 |
| soniox | natural |  | 0.039 | 54 | -1.61 / -1.12 | -1.61 / -1.12 | 3.67 / 4.16 (0) | 5.80 / 6.58 | 0.76 / 0.77 | - | 0.7147 |

## ko_kr

| provider | variant | error | CER | finals | partial first | partial stable | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.119 | 30 | -0.35 / 0.39 | -0.35 / 0.39 | 0.40 / 0.44 (0) | 2.94 / 4.53 | 0.77 / 1.20 | 0.28 / 0.33 | 1.178 |
| deepgram | finalize_mid |  | 0.142 | 48 | -0.30 / 0.64 | -0.30 / 0.64 | 0.67 / 1.12 (0) | 1.34 / 2.27 | 0.64 / 1.07 | 0.29 / 0.31 | 1.2526 |
| deepgram | natural |  | 0.112 | 30 | 0.25 / 0.82 | 0.25 / 0.83 | 0.93 / 2.62 (0) | 2.90 / 4.51 | 0.73 / 1.23 | - | 1.0701 |
| deepgram | nointerim |  | 0.112 | 30 | 0.99 / 2.68 | 0.99 / 2.68 | 0.99 / 2.68 (0) | 2.94 / 4.53 | - | - | 1.1781 |
| gemini | natural |  | 0.032 | 12 | 0.07 / 8.30 | 0.07 / 8.30 | 0.07 / 8.30 (0) | - | - | - | 0.5175 |
| qwen | natural |  | 0.049 | 10 | 0.04 / 1.16 | 0.04 / 1.16 | 0.55 / 10.39 (0) | 7.13 / 13.91 | 1.37 / 2.79 | - | 0.4749 |
| qwen | semantic |  | 0.268 | 10 | 0.40 / 2.03 | 0.40 / 2.03 | 2.76 / 10.83 (0) | 7.74 / 13.80 | 1.91 / 6.28 | - | 0.5214 |
| soniox | finalize_gap |  | 0.032 | 41 | -0.35 / 0.35 | -0.35 / 0.35 | 0.34 / 0.36 (0) | 4.49 / 5.68 | 0.19 / 0.68 | 0.21 / 0.25 | 0.9247 |
| soniox | finalize_mid |  | 0.047 | 48 | -0.37 / 0.48 | -0.37 / 0.48 | 0.59 / 1.06 (0) | -0.18 / 1.54 | -0.90 / 0.44 | 0.25 / 0.27 | 0.9406 |
| soniox | natural |  | 0.032 | 58 | -0.36 / 0.41 | -0.36 / 0.41 | 4.12 / 5.77 (0) | 5.74 / 6.53 | 0.75 / 0.77 | - | 0.9794 |

## Final-only vs partial decision + forced finalize

`estimated partial path` = max(partial stable, 0.10 s) + decider RTT (0.25 s) + median finalize ack of the `finalize_gap` run. `finalize at gap` is the measured final lag when finalize is sent 0.10 s after each sentence end.

| provider | lang | natural final | finalize at gap | estimated partial path | estimated saving |
|---|---|---|---|---|---|
| deepgram | en_us | 2.68 / 3.30 | 0.37 / 0.39 | 0.60 / 1.42 | 1.45 / 2.52 |
| deepgram | ja_jp | 0.04 / 3.12 | 0.21 / 0.33 | 0.55 / 0.55 | -0.51 / 1.46 |
| deepgram | ko_kr | 0.93 / 2.62 | 0.40 / 0.44 | 0.78 / 1.36 | 0.10 / 1.59 |
| soniox | en_us | 5.11 / 5.78 | 0.35 / 0.39 | 0.58 / 1.02 | 4.54 / 5.18 |
| soniox | ja_jp | 3.67 / 4.16 | 0.36 / 0.39 | 0.58 / 0.58 | 3.09 / 3.58 |
| soniox | ko_kr | 4.12 / 5.77 | 0.34 / 0.36 | 0.56 / 0.87 | 3.55 / 4.92 |
