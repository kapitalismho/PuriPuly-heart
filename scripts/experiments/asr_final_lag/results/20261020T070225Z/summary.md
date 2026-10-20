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

## en_conv

Reference units are caption lines, not sentences: sentence columns measure caption line ends, and the CER reference is the uploaded caption text.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | natural |  | 0.088 | 31 | 0.60 / 1.01 | 0.66 / 1.02 | 0.055 (12/52) | 0.35 / 0.88 (41) | 14 | 2.94 / 4.24 (0) | 2.98 / 4.55 | 0.58 / 1.01 | - | 0.5972 |
| gemini | natural |  | 0.093 | 7 | 0.72 / 1.22 | 0.72 / 1.22 | 0.054 (12/52) | 0.61 / 1.14 (42) | 7 | 8.73 / 40.51 (0) | - | - | - | 0.5671 |
| qwen | natural |  | 0.076 | 6 | 1.20 / 3.07 | 1.22 / 3.07 | 0.078 (21/52) | 2.82 / 4.39 (42) | 7 | 12.36 / 26.73 (0) | 13.47 / 27.50 | 1.44 / 3.64 | - | 3.4517 |
| soniox | natural |  | 0.103 | 60 | 0.60 / 0.75 | 0.61 / 0.75 | 0.060 (7/52) | 0.68 / 0.75 (42) | 12 | 5.51 / 6.54 (0) | 5.74 / 6.58 | 0.72 / 0.78 | - | 0.6963 |

## ja_conv

Reference units are caption lines, not sentences: sentence columns measure caption line ends, and the CER reference is the uploaded caption text.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | natural |  | 0.411 | 31 | 0.91 / 4.05 | 1.39 / 4.68 | 0.003 (2/43) | 0.99 / 2.47 (5) | 6 | 3.11 / 6.75 (2) | 3.25 / 4.53 | 0.87 / 1.66 | - | 0.963 |
| gemini | natural |  | 0.349 | 25 | 1.04 / 6.09 | 1.20 / 6.09 | 0.158 (15/43) | 0.93 / 1.68 (11) | 6 | 3.73 / 12.21 (2) | - | - | - | 0.5501 |
| qwen | natural |  | 0.295 | 10 | 0.99 / 3.48 | 0.99 / 3.48 | 0.105 (16/45) | 1.92 / 4.12 (12) | 10 | 7.03 / 27.71 (0) | 6.98 / 14.52 | 1.61 / 2.89 | - | 0.2716 |
| soniox | natural |  | 0.300 | 59 | 0.54 / 1.36 | 0.54 / 1.36 | 0.131 (12/45) | 0.64 / 1.16 (6) | 11 | 5.81 / 7.08 (0) | 5.73 / 6.58 | 0.75 / 0.76 | - | 0.6577 |

## ko_conv

Reference units are caption lines, not sentences: sentence columns measure caption line ends, and the CER reference is the uploaded caption text.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | natural |  | 0.108 | 32 | 0.37 / 1.04 | 0.43 / 1.22 | 0.009 (5/48) | 0.36 / 0.94 (14) | 10 | 2.87 / 4.22 (0) | 3.14 / 4.51 | 0.63 / 1.17 | - | 0.9115 |
| gemini | natural |  | 0.111 | 7 | 0.42 / 0.95 | 0.42 / 0.95 | 0.069 (24/48) | 0.44 / 1.08 (13) | 20 | 11.98 / 44.01 (0) | - | - | - | 0.5663 |
| qwen | natural |  | 0.087 | 6 | 1.19 / 2.57 | 1.19 / 2.57 | 0.030 (10/48) | 2.34 / 3.25 (16) | 8 | 15.06 / 25.39 (0) | 13.91 / 26.89 | 1.71 / 3.32 | - | 0.6116 |
| soniox | natural |  | 0.081 | 59 | 0.47 / 0.70 | 0.47 / 0.70 | 0.011 (3/48) | 0.52 / 0.66 (18) | 11 | 5.47 / 6.49 (0) | 5.75 / 6.59 | 0.72 / 0.78 | - | 1.0756 |

## Final-only vs partial decision

- `partial release` = max(partial stable, 0.10 s) + decider RTT (0.25 s): translation starts from the partial text of the sentence.
- `partial + finalize` = `partial release` + median finalize ack of the `finalize_gap` run: translation waits for a forced final.
- `finalize at gap` = measured final lag when finalize is sent 0.10 s after each sentence's speech end.
- `punct rule` = `punct` of the natural run: translation starts when the partial text punctuates the sentence end, without a decider.
- `saving` = natural final - partial release.

| provider | lang | natural final | finalize at gap | punct rule | partial release | partial + finalize | saving |
|---|---|---|---|---|---|---|---|
