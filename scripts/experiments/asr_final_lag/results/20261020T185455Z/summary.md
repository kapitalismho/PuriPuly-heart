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
| gemini | end_app |  | 0.101 | 22 | 0.62 / 1.02 | 0.62 / 1.02 | 0.034 (9/52) | 0.53 / 2.66 (42) | 8 | 2.64 / 5.14 (0) | - | - | 0.28 / 0.32 | 0.5656 |
| gemini | end_smart |  | 0.081 | 8 | 0.61 / 0.87 | 0.61 / 0.87 | 0.048 (13/52) | 0.58 / 2.46 (42) | 5 | 6.84 / 21.86 (0) | - | - | 0.34 / 0.41 | 0.5617 |
| gemini | natural |  | 0.093 | 7 | 0.59 / 0.89 | 0.59 / 0.89 | 0.055 (13/52) | 0.46 / 0.74 (42) | 7 | 8.63 / 40.45 (0) | - | - | - | 0.5722 |
| qwen | cut_app |  | 0.079 | 23 | 0.57 / 0.88 | 0.57 / 0.88 | 0.057 (17/52) | 0.73 / 1.30 (41) | 16 | 2.56 / 4.69 (0) | 3.04 / 5.40 | 0.61 / 1.46 | 0.40 / 0.59 | 0.6007 |
| qwen | cut_fixed3 |  | 0.106 | 42 | 0.44 / 0.85 | 0.46 / 1.15 | 0.021 (14/52) | 0.66 / 2.09 (37) | 29 | 1.91 / 3.13 (0) | 1.61 / 2.82 | 0.53 / 1.14 | 0.33 / 0.50 | 0.4892 |
| qwen | cut_pause |  | 0.083 | 17 | 0.54 / 1.18 | 0.56 / 1.18 | 0.042 (13/52) | 1.02 / 2.49 (39) | 14 | 3.64 / 8.73 (0) | 4.18 / 8.65 | 0.70 / 1.65 | 0.38 / 0.51 | 0.5094 |
| qwen | cut_smart |  | 0.080 | 16 | 0.65 / 1.25 | 0.65 / 1.25 | 0.032 (11/52) | 1.08 / 2.99 (37) | 11 | 4.01 / 8.70 (0) | 4.19 / 8.26 | 0.67 / 1.66 | 0.43 / 0.58 | 0.5011 |
| qwen | cut_smart_ctx |  | 0.064 | 16 | 0.57 / 1.21 | 0.57 / 1.21 | 0.056 (14/52) | 0.68 / 2.17 (38) | 13 | 3.77 / 8.06 (0) | 4.19 / 8.33 | 0.69 / 1.77 | 0.45 / 0.51 | 0.4203 |
| qwen | natural |  | 0.089 | 6 | 0.62 / 1.10 | 0.64 / 1.11 | 0.081 (17/52) | 1.40 / 2.83 (42) | 8 | 10.65 / 23.89 (0) | 11.86 / 25.10 | 0.85 / 2.14 | - | 0.2651 |
| soniox | natural |  | 0.103 | 60 | 0.57 / 0.74 | 0.57 / 0.74 | 0.060 (7/52) | 0.65 / 0.71 (42) | 12 | 5.47 / 6.50 (0) | 5.71 / 6.55 | 0.69 / 0.75 | - | 0.5949 |

## ja_conv

Reference units are caption lines, not sentences: sentence columns measure caption line ends, and the CER reference is the uploaded caption text.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini | end_app |  | 0.355 | 34 | 0.94 / 8.68 | 0.95 / 8.68 | 0.061 (9/43) | 0.61 / 1.95 (8) | 7 | 2.64 / 12.11 (2) | - | - | 0.27 / 4.69 | 0.7076 |
| gemini | end_smart |  | 0.370 | 35 | 0.58 / 6.70 | 0.88 / 6.70 | 0.089 (10/43) | 0.58 / 2.88 (6) | 8 | 2.47 / 9.15 (2) | - | - | 0.29 / 9.27 | 0.575 |
| gemini | natural |  | 0.349 | 26 | 0.69 / 4.64 | 0.88 / 4.64 | 0.134 (15/43) | 0.67 / 1.16 (10) | 6 | 3.12 / 12.10 (2) | - | - | - | 0.6175 |
| qwen | cut_app |  | 0.330 | 30 | 0.81 / 3.50 | 1.08 / 3.68 | 0.050 (10/43) | 0.96 / 2.42 (20) | 6 | 3.13 / 8.22 (2) | 2.40 / 4.84 | 1.02 / 2.25 | 0.30 / 0.50 | 0.5251 |
| qwen | cut_fixed3 |  | 0.372 | 40 | 1.00 / 4.33 | 1.41 / 4.33 | 0.060 (10/43) | 1.43 / 2.98 (20) | 12 | 2.74 / 6.42 (2) | 1.56 / 2.83 | 0.91 / 1.88 | 0.27 / 0.33 | 0.4952 |
| qwen | cut_pause |  | 0.313 | 28 | 0.68 / 3.64 | 0.68 / 3.64 | 0.068 (9/43) | 1.29 / 3.39 (15) | 5 | 2.52 / 6.29 (2) | 2.05 / 4.80 | 1.01 / 2.19 | 0.30 / 0.41 | 0.5151 |
| qwen | cut_smart |  | 0.328 | 38 | 0.81 / 5.31 | 0.81 / 5.31 | 0.047 (8/43) | 1.26 / 2.33 (16) | 11 | 2.69 / 13.85 (2) | 1.93 / 4.60 | 1.01 / 2.06 | 0.25 / 0.38 | 0.4764 |
| qwen | cut_smart_ctx |  | 0.315 | 36 | 0.70 / 3.56 | 0.70 / 3.56 | 0.040 (6/45) | 1.45 / 3.10 (9) | 11 | 3.18 / 6.87 (0) | 1.93 / 4.56 | 1.05 / 2.19 | 0.27 / 0.38 | 0.4424 |
| qwen | natural |  | 0.291 | 10 | 0.87 / 3.38 | 0.87 / 3.38 | 0.123 (17/45) | 1.69 / 3.32 (12) | 10 | 6.49 / 27.46 (0) | 6.63 / 13.92 | 1.44 / 2.55 | - | 0.3152 |
| soniox | natural |  | 0.304 | 60 | 0.54 / 1.41 | 0.54 / 1.41 | 0.130 (12/45) | 0.62 / 1.14 (6) | 11 | 5.79 / 7.05 (0) | 5.71 / 6.55 | 0.72 / 0.75 | - | 0.5787 |

## ko_conv

Reference units are caption lines, not sentences: sentence columns measure caption line ends, and the CER reference is the uploaded caption text.

| provider | variant | error | CER | finals | partial first | partial stable | release diff | punct (missing) | extra ends | final (missing) | word final | word partial | fin ack | connect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini | end_app |  | 0.126 | 25 | 0.36 / 1.30 | 0.36 / 1.30 | 0.092 (20/48) | 0.30 / 0.48 (15) | 24 | 2.40 / 4.73 (0) | - | - | 0.31 / 2.85 | 0.768 |
| gemini | end_smart |  | 0.112 | 21 | 0.33 / 0.91 | 0.34 / 1.02 | 0.058 (20/48) | 0.30 / 1.34 (12) | 15 | 3.50 / 11.49 (0) | - | - | 0.27 / 3.73 | 0.597 |
| gemini | natural |  | 0.111 | 9 | 0.33 / 0.75 | 0.33 / 0.75 | 0.066 (22/48) | 0.32 / 1.04 (13) | 20 | 8.49 / 43.87 (0) | - | - | - | 0.6216 |
| qwen | cut_app |  | 0.135 | 24 | 0.51 / 2.11 | 0.51 / 2.11 | 0.077 (19/48) | 0.63 / 2.79 (10) | 16 | 3.13 / 5.74 (0) | 3.12 / 5.34 | 0.84 / 2.21 | 0.40 / 0.51 | 0.9251 |
| qwen | cut_fixed3 |  | 0.216 | 41 | 0.67 / 1.94 | 0.72 / 2.62 | 0.070 (15/48) | 0.92 / 2.46 (14) | 20 | 2.20 / 3.77 (0) | 1.65 / 2.91 | 0.74 / 2.05 | 0.31 / 0.43 | 0.4634 |
| qwen | cut_pause |  | 0.105 | 22 | 0.46 / 1.55 | 0.51 / 1.55 | 0.057 (20/48) | 0.88 / 2.76 (10) | 12 | 2.93 / 7.89 (0) | 3.40 / 7.46 | 0.89 / 2.38 | 0.36 / 0.60 | 0.9245 |
| qwen | cut_smart |  | 0.109 | 24 | 0.47 / 1.71 | 0.47 / 1.71 | 0.056 (15/48) | 0.77 / 2.12 (13) | 13 | 3.15 / 7.85 (0) | 3.62 / 8.01 | 0.95 / 2.33 | 0.35 / 0.46 | 0.5152 |
| qwen | cut_smart_ctx |  | 0.107 | 24 | 0.44 / 1.71 | 0.44 / 1.71 | 0.080 (18/48) | 0.79 / 2.13 (13) | 13 | 3.11 / 7.83 (0) | 3.53 / 7.95 | 0.91 / 2.31 | 0.31 / 0.39 | 0.4327 |
| qwen | natural |  | 0.094 | 6 | 0.91 / 2.33 | 0.91 / 2.33 | 0.030 (10/48) | 1.78 / 3.20 (16) | 9 | 14.80 / 25.07 (0) | 13.70 / 26.59 | 1.43 / 2.82 | - | 0.743 |
| soniox | natural |  | 0.085 | 59 | 0.44 / 0.67 | 0.44 / 0.67 | 0.009 (4/48) | 0.51 / 0.68 (16) | 12 | 5.44 / 6.46 (0) | 5.72 / 6.55 | 0.69 / 0.74 | - | 0.5959 |

## Cut and end variants

- `deleted` / `inserted`: character edit operations against the reference text (lost or duplicated words show up here).
- `cuts`: finalize requests (`finish-task` for Qwen Audio, `audio_stream_end` for Gemini). `tasks`, `audio sent` and `task finish` (finish-task to task-finished, p50 / p90) are Qwen Audio only.

| data | provider | variant | CER | deleted | inserted | cuts | tasks | audio sent s | task finish | usage |
|---|---|---|---|---|---|---|---|---|---|---|
| en_conv | gemini | end_app | 0.101 | 64 | 56 | 22 | - | - | - | - |
| en_conv | gemini | end_smart | 0.081 | 37 | 54 | 4 | - | - | - | - |
| en_conv | gemini | natural | 0.093 | 39 | 76 | 0 | - | - | - | - |
| en_conv | qwen | cut_app | 0.079 | 46 | 52 | 23 | 23 | 128.79 | 0.40 / 0.59 | - |
| en_conv | qwen | cut_fixed3 | 0.106 | 87 | 45 | 42 | 42 | 124.89 | 0.33 / 0.50 | - |
| en_conv | qwen | cut_pause | 0.083 | 58 | 34 | 17 | 17 | 124.5 | 0.38 / 0.51 | - |
| en_conv | qwen | cut_smart | 0.080 | 51 | 31 | 16 | 16 | 124.5 | 0.43 / 0.58 | - |
| en_conv | qwen | cut_smart_ctx | 0.064 | 28 | 45 | 16 | 16 | 124.5 | 0.45 / 0.51 | - |
| en_conv | qwen | natural | 0.089 | 56 | 53 | 0 | - | - | - | - |
| ja_conv | gemini | end_app | 0.355 | 102 | 39 | 30 | - | - | - | - |
| ja_conv | gemini | end_smart | 0.370 | 85 | 46 | 37 | - | - | - | - |
| ja_conv | gemini | natural | 0.349 | 95 | 40 | 0 | - | - | - | - |
| ja_conv | qwen | cut_app | 0.330 | 104 | 32 | 31 | 31 | 114.64 | 0.30 / 0.50 | - |
| ja_conv | qwen | cut_fixed3 | 0.372 | 126 | 40 | 40 | 40 | 115.37 | 0.27 / 0.33 | - |
| ja_conv | qwen | cut_pause | 0.313 | 77 | 17 | 28 | 28 | 112.11 | 0.30 / 0.41 | - |
| ja_conv | qwen | cut_smart | 0.328 | 91 | 13 | 42 | 42 | 110.77 | 0.25 / 0.38 | - |
| ja_conv | qwen | cut_smart_ctx | 0.315 | 67 | 17 | 42 | 42 | 110.77 | 0.27 / 0.38 | - |
| ja_conv | qwen | natural | 0.291 | 60 | 82 | 0 | - | - | - | - |
| ko_conv | gemini | end_app | 0.126 | 32 | 40 | 24 | - | - | - | - |
| ko_conv | gemini | end_smart | 0.112 | 23 | 38 | 16 | - | - | - | - |
| ko_conv | gemini | natural | 0.111 | 17 | 43 | 0 | - | - | - | - |
| ko_conv | qwen | cut_app | 0.135 | 57 | 20 | 25 | 25 | 126.34 | 0.40 / 0.51 | - |
| ko_conv | qwen | cut_fixed3 | 0.216 | 104 | 17 | 42 | 42 | 124.54 | 0.31 / 0.43 | - |
| ko_conv | qwen | cut_pause | 0.105 | 44 | 8 | 23 | 23 | 124.06 | 0.36 / 0.60 | - |
| ko_conv | qwen | cut_smart | 0.109 | 44 | 9 | 26 | 26 | 123.44 | 0.35 / 0.46 | - |
| ko_conv | qwen | cut_smart_ctx | 0.107 | 37 | 13 | 26 | 26 | 123.44 | 0.31 / 0.39 | - |
| ko_conv | qwen | natural | 0.094 | 41 | 12 | 0 | - | - | - | - |

## Final-only vs partial decision

- `partial release` = max(partial stable, 0.10 s) + decider RTT (0.25 s): translation starts from the partial text of the sentence.
- `partial + finalize` = `partial release` + median finalize ack of the `finalize_gap` run: translation waits for a forced final.
- `finalize at gap` = measured final lag when finalize is sent 0.10 s after each sentence's speech end.
- `punct rule` = `punct` of the natural run: translation starts when the partial text punctuates the sentence end, without a decider.
- `saving` = natural final - partial release.

| provider | lang | natural final | finalize at gap | punct rule | partial release | partial + finalize | saving |
|---|---|---|---|---|---|---|---|
