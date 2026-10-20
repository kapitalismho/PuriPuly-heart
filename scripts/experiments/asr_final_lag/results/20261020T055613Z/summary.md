# ASR partial/final timing

Times are seconds after the reference sentence's speech end (audio time = wall time since the first chunk was sent). The speech end is the last Silero VAD speech frame of the sentence; provider word end timestamps are typically 0.1-0.3 s earlier. Cells show `p50 / p90`. Sentence rows exclude the last sentence of each recording; word rows exclude the last 2 s.

- `partial first`: the partial text first covers the sentence end.
- `partial stable`: the partial text covers the sentence end and never retracts.
- `final`: final text covers the sentence end (no forced finalize unless the variant says so).
- `word final` / `word partial`: provider word/token end timestamps to final or first partial arrival.
- `release diff`: character error rate of the partial text of a sentence at its `partial stable` time against the final text of the same sentence; mean and the number of sentences whose partial differs.
- `fin ack`: finalize request to the provider's finalize acknowledgement.

## en_us

Silence between sentences (speech end to next speech start): 0.20 / 0.20 s.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.064 | 35 | 0.24 / 0.33 | 0.24 / 0.33 | 0.000 (0/15) | 0.31 / 0.34 (0) | 2.72 / 4.37 | 0.63 / 1.00 | 0.19 / 0.20 | 0.6121 |
| deepgram | finalize_mid |  | 0.085 | 50 | 0.16 / 0.86 | 0.16 / 0.86 | 0.000 (0/15) | 0.94 / 1.56 (0) | 1.25 / 2.21 | 0.57 / 0.96 | 0.18 / 0.20 | 0.6293 |
| deepgram | natural |  | 0.055 | 32 | 0.28 / 0.69 | 0.28 / 0.69 | 0.003 (1/15) | 2.36 / 3.96 (0) | 3.01 / 4.54 | 0.60 / 1.01 | - | 0.639 |
| deepgram | nointerim |  | 0.055 | 32 | 2.35 / 3.92 | 2.35 / 3.92 | 0.000 (0/15) | 2.35 / 3.92 (0) | 3.01 / 4.52 | - | - | 0.635 |
| gemini | finalize_gap |  | 0.016 | 16 | 0.45 / 0.60 | 0.45 / 0.60 | 0.007 (5/15) | 0.50 / 0.60 (0) | - | - | 0.37 / 0.46 | 0.5689 |
| gemini | finalize_mid |  | 0.166 | 48 | 0.66 / 1.47 | 0.66 / 1.47 | 0.047 (4/15) | 1.42 / 2.37 (0) | - | - | 0.30 / 0.42 | 0.5681 |
| gemini | natural |  | 0.021 | 2 | 0.54 / 0.79 | 0.54 / 0.79 | 0.019 (5/15) | 68.19 / 105.35 (0) | - | - | - | 0.5962 |
| qwen | natural |  | 0.021 | 8 | 0.53 / 0.97 | 0.53 / 0.97 | 0.005 (2/15) | 8.13 / 12.62 (0) | 8.21 / 15.61 | 0.88 / 2.39 | - | 0.3188 |
| qwen | semantic |  | 0.012 | 8 | 0.54 / 1.13 | 0.54 / 1.13 | 0.005 (2/15) | 8.07 / 13.01 (0) | 8.48 / 15.93 | 0.85 / 2.21 | - | 0.288 |
| soniox | finalize_gap |  | 0.036 | 38 | 0.33 / 0.36 | 0.33 / 0.36 | 0.000 (0/15) | 0.33 / 0.36 (0) | 3.54 / 5.38 | 0.05 / 0.58 | 0.19 / 0.24 | 0.6147 |
| soniox | finalize_mid |  | 0.042 | 50 | 0.55 / 0.61 | 0.55 / 0.61 | 0.001 (1/15) | 1.25 / 2.09 (0) | -0.43 / 1.24 | -1.30 / 0.30 | 0.26 / 0.27 | 0.7281 |
| soniox | natural |  | 0.040 | 59 | 0.51 / 0.63 | 0.51 / 0.63 | 0.001 (1/15) | 5.43 / 6.02 (0) | 5.80 / 6.59 | 0.76 / 0.77 | - | 0.7325 |

## ja_jp

Silence between sentences (speech end to next speech start): 0.20 / 0.20 s.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.086 | 35 | 0.36 / 0.41 | 0.36 / 0.41 | 0.003 (2/15) | 0.40 / 0.43 (0) | 2.39 / 4.30 | 0.69 / 1.13 | 0.27 / 0.31 | 0.8632 |
| deepgram | finalize_mid |  | 0.165 | 52 | 0.28 / 0.80 | 0.28 / 0.91 | 0.008 (4/15) | 1.02 / 2.17 (0) | 1.35 / 2.28 | 0.67 / 1.09 | 0.25 / 0.28 | 0.8824 |
| deepgram | natural |  | 0.097 | 32 | 0.78 / 1.20 | 0.78 / 1.20 | 0.000 (0/15) | 2.81 / 3.98 (0) | 3.07 / 4.60 | 0.70 / 1.15 | - | 0.8675 |
| deepgram | nointerim |  | 0.097 | 32 | 2.80 / 4.00 | 2.80 / 4.00 | 0.000 (0/15) | 2.80 / 4.00 (0) | 3.07 / 4.61 | - | - | 0.8405 |
| gemini | finalize_gap |  | 0.039 | 18 | 0.42 / 0.49 | 0.42 / 0.49 | 0.006 (2/15) | 0.47 / 0.58 (0) | - | - | 0.34 / 0.44 | 0.8121 |
| gemini | finalize_mid |  | 0.213 | 53 | 0.42 / 1.43 | 0.42 / 1.43 | 0.019 (4/15) | 1.02 / 2.20 (0) | - | - | 0.33 / 0.41 | 0.5572 |
| gemini | natural |  | 0.032 | 2 | 0.62 / 0.83 | 0.62 / 0.87 | 0.037 (7/15) | 56.42 / 94.10 (0) | - | - | - | 0.6966 |
| qwen | natural |  | 0.028 | 6 | 0.91 / 1.66 | 0.91 / 1.66 | 0.011 (3/15) | 11.20 / 21.61 (0) | 12.36 / 26.13 | 1.64 / 2.89 | - | 0.9009 |
| qwen | semantic |  | 0.028 | 6 | 0.91 / 1.22 | 0.91 / 1.22 | 0.011 (3/15) | 11.25 / 21.98 (0) | 12.62 / 26.59 | 1.34 / 2.66 | - | 0.2625 |
| soniox | finalize_gap |  | 0.033 | 41 | 0.34 / 0.37 | 0.34 / 0.37 | 0.000 (0/15) | 0.35 / 0.40 (0) | 3.67 / 5.45 | 0.17 / 0.66 | 0.22 / 0.27 | 0.7044 |
| soniox | finalize_mid |  | 0.051 | 52 | 0.39 / 0.45 | 0.39 / 0.45 | 0.000 (0/15) | 0.79 / 1.74 (0) | -0.27 / 1.51 | -1.07 / 0.33 | 0.26 / 0.27 | 0.7053 |
| soniox | natural |  | 0.028 | 62 | 0.41 / 0.48 | 0.41 / 0.48 | 0.000 (0/15) | 5.36 / 6.15 (0) | 5.80 / 6.58 | 0.75 / 0.77 | - | 0.7246 |

## ko_kr

Silence between sentences (speech end to next speech start): 0.20 / 0.20 s.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | finalize_gap |  | 0.110 | 33 | 0.31 / 1.07 | 0.31 / 1.07 | 0.002 (1/13) | 0.31 / 4.28 (0) | 2.62 / 4.41 | 0.63 / 1.55 | 0.20 / 0.21 | 0.9328 |
| deepgram | finalize_mid |  | 0.166 | 49 | 0.35 / 1.57 | 0.35 / 1.57 | 0.007 (1/13) | 0.92 / 2.04 (0) | 1.20 / 2.17 | 0.60 / 1.04 | 0.18 / 0.20 | 0.9135 |
| deepgram | natural |  | 0.119 | 31 | 0.47 / 0.86 | 0.47 / 0.86 | 0.002 (1/13) | 2.61 / 3.75 (0) | 2.83 / 4.47 | 0.61 / 1.06 | - | 0.9467 |
| deepgram | nointerim |  | 0.119 | 31 | 2.60 / 3.75 | 2.60 / 3.75 | 0.000 (0/13) | 2.60 / 3.75 (0) | 2.86 / 4.47 | - | - | 0.9352 |
| gemini | finalize_gap |  | 0.026 | 16 | 0.47 / 0.81 | 0.47 / 0.81 | 0.010 (3/13) | 0.49 / 4.62 (0) | - | - | 0.37 / 4.49 | 0.6009 |
| gemini | finalize_mid |  | 0.146 | 49 | 0.40 / 0.57 | 0.40 / 0.57 | 0.071 (5/13) | 1.04 / 2.19 (0) | - | - | 0.30 / 0.36 | 0.5891 |
| gemini | natural |  | 0.029 | 2 | 0.33 / 0.49 | 0.35 / 0.57 | 0.021 (5/13) | 31.62 / 59.57 (0) | - | - | - | 0.7671 |
| qwen | natural |  | 0.072 | 8 | 1.27 / 2.31 | 1.27 / 2.31 | 0.019 (3/13) | 9.68 / 18.99 (0) | 9.82 / 17.35 | 1.93 / 4.46 | - | 0.6163 |
| qwen | semantic |  | 0.072 | 8 | 1.14 / 2.18 | 1.14 / 2.18 | 0.019 (3/13) | 9.82 / 18.17 (0) | 9.16 / 16.77 | 1.92 / 4.14 | - | 2.2909 |
| soniox | finalize_gap |  | 0.032 | 40 | 0.33 / 0.36 | 0.33 / 0.36 | 0.000 (0/13) | 0.33 / 0.36 (0) | 4.02 / 5.56 | 0.17 / 0.60 | 0.22 / 0.24 | 0.943 |
| soniox | finalize_mid |  | 0.050 | 50 | 0.42 / 0.52 | 0.42 / 0.52 | 0.002 (1/13) | 0.95 / 2.09 (0) | -0.31 / 1.46 | -1.16 / 0.38 | 0.23 / 0.24 | 0.9252 |
| soniox | natural |  | 0.027 | 59 | 0.44 / 0.54 | 0.44 / 0.54 | 0.000 (0/13) | 5.49 / 6.09 (0) | 5.75 / 6.58 | 0.76 / 0.77 | - | 1.0774 |

## Final-only vs partial decision

- `partial release` = max(partial stable, 0.10 s) + decider RTT (0.25 s): translation starts from the partial text of the sentence.
- `partial + finalize` = `partial release` + median finalize ack of the `finalize_gap` run: translation waits for a forced final.
- `finalize at gap` = measured final lag when finalize is sent 0.10 s after each sentence's speech end.
- `saving` = natural final - partial release.

| provider | lang | natural final | finalize at gap | partial release | partial + finalize | saving |
|---|---|---|---|---|---|---|
| deepgram | en_us | 2.36 / 3.96 | 0.31 / 0.34 | 0.53 / 0.94 | 0.71 / 1.12 | 1.75 / 3.53 |
| deepgram | ja_jp | 2.81 / 3.98 | 0.40 / 0.43 | 1.03 / 1.45 | 1.30 / 1.72 | 1.77 / 2.77 |
| deepgram | ko_kr | 2.61 / 3.75 | 0.31 / 4.28 | 0.72 / 1.11 | 0.91 / 1.31 | 1.89 / 2.77 |
| gemini | en_us | 68.19 / 105.35 | 0.50 / 0.60 | 0.79 / 1.04 | 1.16 / 1.41 | 67.15 / 104.55 |
| gemini | ja_jp | 56.42 / 94.10 | 0.47 / 0.58 | 0.87 / 1.12 | 1.21 / 1.46 | 55.80 / 93.10 |
| gemini | ko_kr | 31.62 / 59.57 | 0.49 / 4.62 | 0.60 / 0.82 | 0.97 / 1.20 | 30.94 / 59.05 |
| qwen | en_us | 8.13 / 12.62 | - | 0.78 / 1.22 | - | 6.84 / 11.82 |
| qwen | ja_jp | 11.20 / 21.61 | - | 1.16 / 1.91 | - | 10.43 / 19.86 |
| qwen | ko_kr | 9.68 / 18.99 | - | 1.52 / 2.56 | - | 7.07 / 16.69 |
| soniox | en_us | 5.43 / 6.02 | 0.33 / 0.36 | 0.76 / 0.88 | 0.95 / 1.07 | 4.67 / 5.32 |
| soniox | ja_jp | 5.36 / 6.15 | 0.35 / 0.40 | 0.67 / 0.73 | 0.89 / 0.95 | 4.79 / 5.57 |
| soniox | ko_kr | 5.49 / 6.09 | 0.33 / 0.36 | 0.69 / 0.79 | 0.91 / 1.01 | 4.78 / 5.36 |
