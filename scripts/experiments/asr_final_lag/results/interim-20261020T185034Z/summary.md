# Gemini interim release replay

Replays recorded Gemini runs. `finals only` releases each authoritative final as it arrives (the current application). The interim policies release interim text up to its last sentence-final punctuation once that prefix has been unchanged for W seconds. When a final arrives, they release only what is left:

- `final remainder`: the final after the already released text (located by text alignment). When the released text cannot be found in the final, the rest of the last interim is used instead (`unmatched`).
- `interim remainder`: the rest of the last interim, plus any final text after the end of that interim.

`latency` is seconds from each character's Soniox end timestamp to its release (`p50 / p90`). CER, `deleted` and `inserted` compare all released text with the reference text. Conversation references are uploaded captions.

| data | variant | policy | releases | latency | CER | deleted | inserted | turns | unmatched |
|---|---|---|---|---|---|---|---|---|---|
| ko_conv | natural | finals only | 7 | 10.44 / 43.08 | 0.111 | 17 | 43 | - | - |
| ko_conv | natural | interim W=0.0 + final remainder | 62 | 1.53 / 3.95 | 0.137 | 30 | 36 | 7 | 0 |
| ko_conv | natural | interim W=0.6 + final remainder | 52 | 2.32 / 4.62 | 0.119 | 27 | 33 | 7 | 0 |
| ko_conv | natural | interim W=1.0 + final remainder | 48 | 2.66 / 4.94 | 0.122 | 27 | 34 | 7 | 0 |
| ko_conv | natural | interim W=0.0 + interim remainder | 62 | 1.53 / 3.95 | 0.131 | 30 | 29 | 7 | 0 |
| ko_conv | natural | interim W=0.6 + interim remainder | 52 | 2.32 / 4.62 | 0.113 | 27 | 26 | 7 | 0 |
| ko_conv | natural | interim W=1.0 + interim remainder | 48 | 2.66 / 4.94 | 0.113 | 27 | 26 | 7 | 0 |
| ja_conv | natural | finals only | 25 | 2.36 / 7.29 | 0.349 | 95 | 40 | - | - |
| ja_conv | natural | interim W=0.0 + final remainder | 59 | 1.33 / 3.72 | 0.429 | 105 | 84 | 25 | 1 |
| ja_conv | natural | interim W=0.6 + final remainder | 47 | 1.79 / 4.21 | 0.420 | 104 | 83 | 25 | 1 |
| ja_conv | natural | interim W=1.0 + final remainder | 41 | 2.03 / 4.68 | 0.420 | 104 | 83 | 25 | 3 |
| ja_conv | natural | interim W=0.0 + interim remainder | 60 | 1.36 / 4.13 | 0.425 | 103 | 84 | 25 | 3 |
| ja_conv | natural | interim W=0.6 + interim remainder | 48 | 1.85 / 4.82 | 0.425 | 105 | 84 | 25 | 3 |
| ja_conv | natural | interim W=1.0 + interim remainder | 42 | 2.12 / 5.27 | 0.425 | 105 | 84 | 25 | 3 |
| en_conv | natural | finals only | 7 | 10.79 / 43.64 | 0.093 | 39 | 76 | - | - |
| en_conv | natural | interim W=0.0 + final remainder | 29 | 2.99 / 10.34 | 0.094 | 43 | 71 | 7 | 0 |
| en_conv | natural | interim W=0.6 + final remainder | 25 | 3.85 / 11.12 | 0.099 | 43 | 80 | 7 | 0 |
| en_conv | natural | interim W=1.0 + final remainder | 24 | 4.24 / 14.01 | 0.099 | 43 | 80 | 7 | 0 |
| en_conv | natural | interim W=0.0 + interim remainder | 29 | 2.99 / 10.34 | 0.091 | 43 | 67 | 7 | 0 |
| en_conv | natural | interim W=0.6 + interim remainder | 25 | 3.85 / 11.12 | 0.091 | 43 | 67 | 7 | 0 |
| en_conv | natural | interim W=1.0 + interim remainder | 24 | 4.24 / 14.01 | 0.091 | 43 | 67 | 7 | 0 |
| ko_kr | natural | finals only | 2 | 32.21 / 60.37 | 0.029 | 9 | 0 | - | - |
| ko_kr | natural | interim W=0.0 + final remainder | 21 | 3.39 / 8.67 | 0.035 | 10 | 1 | 2 | 0 |
| ko_kr | natural | interim W=0.6 + final remainder | 18 | 4.36 / 9.79 | 0.030 | 10 | 1 | 2 | 0 |
| ko_kr | natural | interim W=1.0 + final remainder | 17 | 4.95 / 10.17 | 0.030 | 10 | 1 | 2 | 0 |
| ko_kr | natural | interim W=0.0 + interim remainder | 21 | 3.39 / 8.67 | 0.037 | 10 | 1 | 2 | 0 |
| ko_kr | natural | interim W=0.6 + interim remainder | 18 | 4.36 / 9.79 | 0.032 | 10 | 1 | 2 | 0 |
| ko_kr | natural | interim W=1.0 + interim remainder | 17 | 4.95 / 10.17 | 0.032 | 10 | 1 | 2 | 0 |
| ja_jp | natural | finals only | 2 | 52.78 / 100.16 | 0.032 | 4 | 2 | - | - |
| ja_jp | natural | interim W=0.0 + final remainder | 23 | 3.35 / 7.88 | 0.047 | 7 | 6 | 2 | 0 |
| ja_jp | natural | interim W=0.6 + final remainder | 19 | 4.33 / 9.09 | 0.046 | 8 | 5 | 2 | 0 |
| ja_jp | natural | interim W=1.0 + final remainder | 19 | 4.70 / 9.43 | 0.046 | 8 | 5 | 2 | 0 |
| ja_jp | natural | interim W=0.0 + interim remainder | 23 | 3.35 / 7.88 | 0.047 | 7 | 6 | 2 | 0 |
| ja_jp | natural | interim W=0.6 + interim remainder | 19 | 4.33 / 9.09 | 0.046 | 7 | 6 | 2 | 0 |
| ja_jp | natural | interim W=1.0 + interim remainder | 19 | 4.70 / 9.43 | 0.046 | 7 | 6 | 2 | 0 |
| en_us | natural | finals only | 2 | 60.54 / 109.02 | 0.021 | 20 | 9 | - | - |
| en_us | natural | interim W=0.0 + final remainder | 29 | 2.79 / 5.77 | 0.029 | 28 | 7 | 2 | 0 |
| en_us | natural | interim W=0.6 + final remainder | 21 | 4.25 / 8.89 | 0.029 | 28 | 7 | 2 | 0 |
| en_us | natural | interim W=1.0 + final remainder | 21 | 4.62 / 9.22 | 0.025 | 26 | 6 | 2 | 0 |
| en_us | natural | interim W=0.0 + interim remainder | 29 | 2.79 / 5.77 | 0.029 | 28 | 7 | 2 | 0 |
| en_us | natural | interim W=0.6 + interim remainder | 21 | 4.25 / 8.89 | 0.029 | 28 | 7 | 2 | 0 |
| en_us | natural | interim W=1.0 + interim remainder | 21 | 4.62 / 9.22 | 0.029 | 28 | 7 | 2 | 0 |
