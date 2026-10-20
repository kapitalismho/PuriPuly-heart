# ASR partial/final timing

Times are seconds after the reference sentence's speech end (audio time = wall time since the first chunk was sent). The speech end is the last Silero VAD speech frame of the sentence; provider word end timestamps are typically 0.1-0.3 s earlier. Cells show `p50 / p90`. Sentence rows exclude the last sentence of each recording; word rows exclude the last 2 s.

- `partial first`: the partial text first covers the sentence end.
- `partial stable`: the partial text covers the sentence end and never retracts.
- `final`: final text covers the sentence end (no forced finalize unless the variant says so).
- `word final` / `word partial`: provider word/token end timestamps to final or first partial arrival.
- `release diff`: character error rate of the partial text of a sentence at its `partial stable` time against the final text of the same sentence; mean and the number of sentences whose partial differs.
- `punct (missing)`: the partial text first shows terminal punctuation at the sentence end, within 3 characters after the aligned reference end (searched for 8 s after `partial first`).
- `extra ends`: terminal punctuation in the final text that is not at a reference sentence end or at terminal punctuation inside a reference sentence.
- `fin ack`: finalize request to the provider's finalize acknowledgement.

## en_us

Silence between sentences (speech end to next speech start): 1.06 / 2.10 s.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.065 | 33 | 0.41 / 0.70 | 0.41 / 0.70 | 0.000 (0/13) | 0.41 / 0.72 (2) | 1 | 0.78 / 1.73 (0) | 2.80 / 4.38 | 0.68 / 1.09 | 0.26 / 0.27 | 0.8541 |
| deepgram | finalize_mid |  | 0.085 | 49 | 0.38 / 0.91 | 0.38 / 0.91 | 0.005 (1/13) | 0.36 / 0.69 (3) | 1 | 1.40 / 1.82 (0) | 1.34 / 2.36 | 0.69 / 1.08 | 0.26 / 0.28 | 0.9071 |
| deepgram | natural |  | 0.070 | 31 | 0.52 / 0.99 | 0.52 / 0.99 | 0.000 (0/13) | 0.56 / 1.04 (2) | 2 | 3.01 / 3.91 (0) | 3.13 / 4.66 | 0.64 / 1.07 | - | 0.8778 |
| deepgram | nointerim |  | 0.070 | 31 | 3.00 / 3.92 | 3.00 / 3.92 | 0.000 (0/13) | 3.00 / 3.77 (2) | 2 | 3.00 / 3.92 (0) | 3.12 / 4.68 | - | - | 0.8924 |
| gemini | natural |  | 0.018 | 12 | 0.92 / 4.47 | 0.92 / 4.47 | 0.000 (0/13) | 0.92 / 4.47 (0) | 1 | 0.92 / 4.47 (0) | - | - | - | 0.5059 |
| qwen | natural |  | 0.018 | 11 | 0.58 / 1.23 | 0.58 / 1.23 | 0.013 (4/13) | 1.62 / 1.93 (1) | 2 | 1.76 / 7.49 (0) | 5.92 / 11.64 | 0.79 / 2.42 | - | 0.5831 |
| qwen | semantic |  | 0.021 | 11 | 0.62 / 1.34 | 0.62 / 1.34 | 0.017 (5/13) | 1.65 / 2.05 (1) | 2 | 1.81 / 7.42 (0) | 6.06 / 11.66 | 0.82 / 2.38 | - | 0.3454 |
| soniox | finalize_gap |  | 0.043 | 42 | 0.40 / 0.61 | 0.40 / 0.61 | 0.001 (1/13) | 0.53 / 0.62 (0) | 0 | 0.90 / 1.69 (0) | 4.13 / 5.64 | 0.18 / 0.68 | 0.23 / 0.27 | 0.7079 |
| soniox | finalize_mid |  | 0.054 | 49 | 0.46 / 0.66 | 0.46 / 0.66 | 0.001 (1/13) | 0.57 / 0.77 (0) | 37 | 1.34 / 1.79 (0) | -0.52 / 1.28 | -1.34 / 0.23 | 0.26 / 0.27 | 0.7063 |
| soniox | natural |  | 0.040 | 58 | 0.51 / 0.58 | 0.51 / 0.58 | 0.001 (1/13) | 0.55 / 0.70 (0) | 0 | 5.64 / 6.42 (0) | 5.71 / 6.58 | 0.72 / 0.74 | - | 0.6138 |

## ja_jp

Silence between sentences (speech end to next speech start): 3.25 / 4.24 s.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.106 | 31 | 0.26 / 0.63 | 0.26 / 0.66 | 0.002 (1/11) | 0.40 / 0.66 (0) | 2 | 2.08 / 3.07 (0) | 2.71 / 4.32 | 0.61 / 1.06 | 0.20 / 0.21 | 0.5908 |
| deepgram | finalize_mid |  | 0.154 | 44 | 0.52 / 0.96 | 0.52 / 0.96 | 0.030 (3/11) | 0.52 / 0.96 (0) | 1 | 1.42 / 1.69 (0) | 1.29 / 2.24 | 0.65 / 1.08 | 0.20 / 0.22 | 0.5892 |
| deepgram | natural |  | 0.099 | 30 | 0.62 / 0.86 | 0.65 / 0.93 | 0.001 (1/11) | 0.65 / 0.91 (0) | 1 | 2.73 / 3.65 (0) | 2.96 / 4.58 | 0.61 / 1.09 | - | 0.5695 |
| deepgram | nointerim |  | 0.099 | 30 | 2.73 / 3.70 | 2.73 / 3.70 | 0.000 (0/11) | 2.73 / 3.70 (0) | 1 | 2.73 / 3.70 (0) | 2.93 / 4.59 | - | - | 0.5966 |
| gemini | natural |  | 0.048 | 12 | 0.85 / 1.00 | 0.85 / 1.00 | 0.000 (0/11) | 0.85 / 1.00 (0) | 0 | 0.85 / 1.00 (0) | - | - | - | 0.5328 |
| qwen | natural |  | 0.023 | 15 | 0.60 / 0.85 | 0.60 / 0.85 | 0.010 (2/11) | 1.51 / 1.91 (0) | 0 | 1.51 / 1.91 (0) | 5.05 / 8.71 | 1.09 / 2.37 | - | 0.2698 |
| qwen | semantic |  | 0.018 | 15 | 0.62 / 0.85 | 0.62 / 0.85 | 0.010 (2/11) | 1.55 / 2.01 (0) | 0 | 1.55 / 2.01 (0) | 5.10 / 8.65 | 1.14 / 2.27 | - | 0.2542 |
| soniox | finalize_gap |  | 0.034 | 39 | 0.39 / 0.45 | 0.39 / 0.45 | 0.000 (0/11) | 0.58 / 0.68 (0) | 1 | 2.42 / 2.80 (0) | 4.68 / 5.96 | 0.34 / 0.64 | 0.23 / 0.27 | 0.7172 |
| soniox | finalize_mid |  | 0.054 | 44 | 0.41 / 0.48 | 0.41 / 0.48 | 0.000 (0/11) | 0.63 / 0.69 (0) | 31 | 1.45 / 1.71 (0) | -0.14 / 1.51 | -0.99 / 0.34 | 0.27 / 0.28 | 0.73 |
| soniox | natural |  | 0.039 | 54 | 0.42 / 0.45 | 0.42 / 0.45 | 0.000 (0/11) | 0.61 / 0.68 (0) | 1 | 5.73 / 6.00 (0) | 5.80 / 6.58 | 0.76 / 0.77 | - | 0.7147 |

## ko_kr

Silence between sentences (speech end to next speech start): 1.27 / 2.66 s.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.119 | 30 | 0.40 / 0.90 | 0.40 / 0.90 | 0.002 (1/12) | 0.40 / 0.90 (0) | 2 | 0.99 / 1.64 (0) | 2.94 / 4.53 | 0.77 / 1.20 | 0.28 / 0.33 | 1.178 |
| deepgram | finalize_mid |  | 0.142 | 48 | 0.49 / 0.76 | 0.49 / 0.76 | 0.011 (3/12) | 0.49 / 0.76 (0) | 1 | 1.04 / 1.50 (0) | 1.34 / 2.27 | 0.64 / 1.07 | 0.29 / 0.31 | 1.2526 |
| deepgram | natural |  | 0.112 | 30 | 0.79 / 1.03 | 0.80 / 1.03 | 0.002 (1/12) | 0.79 / 1.03 (0) | 3 | 1.60 / 3.79 (0) | 2.90 / 4.51 | 0.73 / 1.23 | - | 1.0701 |
| deepgram | nointerim |  | 0.112 | 30 | 1.62 / 3.81 | 1.62 / 3.81 | 0.002 (1/12) | 1.62 / 3.24 (1) | 3 | 1.62 / 3.81 (0) | 2.94 / 4.53 | - | - | 1.1781 |
| gemini | natural |  | 0.032 | 12 | 0.94 / 8.33 | 0.94 / 8.33 | 0.000 (0/12) | 0.94 / 8.33 (0) | 0 | 0.94 / 8.33 (0) | - | - | - | 0.5175 |
| qwen | natural |  | 0.049 | 10 | 0.93 / 1.89 | 0.93 / 1.89 | 0.026 (3/12) | 1.11 / 5.64 (1) | 1 | 1.90 / 10.44 (0) | 7.13 / 13.91 | 1.37 / 2.79 | - | 0.4749 |
| qwen | semantic |  | 0.268 | 10 | 1.02 / 2.12 | 1.02 / 2.12 | 0.026 (3/12) | 1.23 / 4.06 (2) | 1 | 3.26 / 10.85 (0) | 7.74 / 13.80 | 1.91 / 6.28 | - | 0.5214 |
| soniox | finalize_gap |  | 0.032 | 41 | 0.39 / 0.49 | 0.39 / 0.49 | 0.000 (0/12) | 0.39 / 0.49 (0) | 0 | 1.20 / 2.68 (0) | 4.49 / 5.68 | 0.19 / 0.68 | 0.21 / 0.25 | 0.9247 |
| soniox | finalize_mid |  | 0.047 | 48 | 0.48 / 0.50 | 0.48 / 0.50 | 0.002 (1/12) | 0.48 / 0.50 (0) | 36 | 0.97 / 1.49 (0) | -0.18 / 1.54 | -0.90 / 0.44 | 0.25 / 0.27 | 0.9406 |
| soniox | natural |  | 0.032 | 58 | 0.44 / 0.50 | 0.44 / 0.50 | 0.000 (0/12) | 0.45 / 0.54 (0) | 0 | 5.59 / 6.16 (0) | 5.74 / 6.53 | 0.75 / 0.77 | - | 0.9794 |

## Final-only vs partial decision

- `partial release` = max(partial stable, 0.10 s) + decider RTT (0.25 s): translation starts from the partial text of the sentence.
- `partial + finalize` = `partial release` + median finalize ack of the `finalize_gap` run: translation waits for a forced final.
- `finalize at gap` = measured final lag when finalize is sent 0.10 s after each sentence's speech end.
- `punct rule` = `punct` of the natural run: translation starts when the partial text punctuates the sentence end, without a decider.
- `saving` = natural final - partial release.

| provider | lang | natural final | finalize at gap | punct rule | partial release | partial + finalize | saving |
|---|---|---|---|---|---|---|---|
| deepgram | en_us | 3.01 / 3.91 | 0.78 / 1.73 | 0.56 / 1.04 | 0.77 / 1.24 | 1.02 / 1.49 | 1.71 / 3.33 |
| deepgram | ja_jp | 2.73 / 3.65 | 2.08 / 3.07 | 0.65 / 0.91 | 0.90 / 1.18 | 1.10 / 1.38 | 1.66 / 2.77 |
| deepgram | ko_kr | 1.60 / 3.79 | 0.99 / 1.64 | 0.79 / 1.03 | 1.05 / 1.28 | 1.33 / 1.56 | 0.75 / 2.05 |
| qwen | en_us | 1.76 / 7.49 | - | 1.62 / 1.93 | 0.83 / 1.48 | - | 1.12 / 7.11 |
| qwen | ja_jp | 1.51 / 1.91 | - | 1.51 / 1.91 | 0.85 / 1.10 | - | 0.96 / 1.06 |
| qwen | ko_kr | 1.90 / 10.44 | - | 1.11 / 5.64 | 1.18 / 2.14 | - | 1.07 / 9.31 |
| soniox | en_us | 5.64 / 6.42 | 0.90 / 1.69 | 0.55 / 0.70 | 0.76 / 0.83 | 0.99 / 1.06 | 4.90 / 5.64 |
| soniox | ja_jp | 5.73 / 6.00 | 2.42 / 2.80 | 0.61 / 0.68 | 0.67 / 0.70 | 0.90 / 0.94 | 5.03 / 5.38 |
| soniox | ko_kr | 5.59 / 6.16 | 1.20 / 2.68 | 0.45 / 0.54 | 0.69 / 0.75 | 0.91 / 0.97 | 4.91 / 5.49 |
