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
| deepgram | natural |  | 0.083 | 31 | 0.71 / 1.10 | 0.72 / 1.10 | 0.048 (10/52) | 0.47 / 0.98 (41) | 13 | 3.05 / 4.37 (0) | 3.07 / 4.64 | 0.67 / 1.12 | - | 0.8529 |
| gemini | natural |  | 0.093 | 8 | 0.64 / 1.02 | 0.64 / 1.02 | 0.053 (12/52) | 0.44 / 1.05 (42) | 7 | 7.23 / 21.90 (0) | - | - | - | 0.5682 |
| soniox | natural |  | 0.103 | 60 | 0.56 / 0.71 | 0.57 / 0.71 | 0.060 (7/52) | 0.65 / 0.71 (42) | 12 | 5.47 / 6.49 (0) | 5.71 / 6.55 | 0.69 / 0.74 | - | 0.5907 |

## ja_conv

Reference units are caption lines, not sentences: sentence columns measure caption line ends, and the CER reference is the uploaded caption text.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | natural |  | 0.411 | 31 | 0.86 / 4.01 | 1.30 / 4.62 | 0.003 (2/43) | 0.99 / 2.46 (5) | 6 | 3.10 / 6.70 (2) | 3.19 / 4.48 | 0.81 / 1.66 | - | 0.6273 |
| gemini | natural |  | 0.349 | 26 | 0.73 / 6.03 | 0.85 / 6.03 | 0.112 (13/43) | 0.68 / 1.20 (11) | 6 | 3.13 / 12.11 (2) | - | - | - | 0.5422 |
| soniox | natural |  | 0.302 | 60 | 0.54 / 1.37 | 0.54 / 1.37 | 0.113 (11/45) | 0.67 / 1.17 (6) | 11 | 5.82 / 7.07 (0) | 5.70 / 6.59 | 0.75 / 0.78 | - | 0.701 |

## ko_conv

Reference units are caption lines, not sentences: sentence columns measure caption line ends, and the CER reference is the uploaded caption text.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | natural |  | 0.108 | 32 | 0.46 / 1.11 | 0.50 / 1.30 | 0.009 (5/48) | 0.46 / 1.06 (14) | 10 | 2.92 / 4.25 (0) | 3.18 / 4.58 | 0.71 / 1.28 | - | 1.5202 |
| gemini | natural |  | 0.111 | 10 | 0.33 / 1.02 | 0.33 / 1.02 | 0.062 (20/48) | 0.33 / 1.28 (11) | 20 | 6.50 / 27.83 (0) | - | - | - | 0.6493 |
| soniox | natural |  | 0.081 | 59 | 0.49 / 0.69 | 0.49 / 0.69 | 0.004 (2/48) | 0.51 / 0.73 (18) | 11 | 5.47 / 6.48 (0) | 5.75 / 6.59 | 0.74 / 0.77 | - | 1.3626 |

## Final-only vs partial decision

- `partial release` = max(partial stable, 0.10 s) + decider RTT (0.25 s): translation starts from the partial text of the sentence.
- `partial + finalize` = `partial release` + median finalize ack of the `finalize_gap` run: translation waits for a forced final.
- `finalize at gap` = measured final lag when finalize is sent 0.10 s after each sentence's speech end.
- `punct rule` = `punct` of the natural run: translation starts when the partial text punctuates the sentence end, without a decider.
- `saving` = natural final - partial release.

| provider | lang | natural final | finalize at gap | punct rule | partial release | partial + finalize | saving |
|---|---|---|---|---|---|---|---|
