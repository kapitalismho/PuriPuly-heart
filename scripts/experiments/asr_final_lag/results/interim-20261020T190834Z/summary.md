# Gemini interim release replay

Replays recorded Gemini runs. `finals only` releases each authoritative final as it arrives (the current application). The interim policies release interim text up to its last sentence-final punctuation once that prefix has been unchanged for W seconds. When a final arrives, they release only what is left:

- `final remainder`: the final after the already released text (located by text alignment). When the released text cannot be found in the final, the rest of the last interim is used instead (`unmatched`).
- `interim remainder`: the rest of the last interim, plus any final text after the end of that interim.

`latency` is seconds from each character's Soniox end timestamp to its release (`p50 / p90`). CER, `deleted` and `inserted` compare all released text with the reference text. Conversation references are uploaded captions.

| data | variant | policy | releases | latency | CER | deleted | inserted | turns | unmatched |
|---|---|---|---|---|---|---|---|---|---|
| ko_conv | natural | finals only | 9 | 8.17 / 42.93 | 0.111 | 17 | 43 | - | - |
| ko_conv | natural | interim W=0.0 + final remainder | 66 | 1.44 / 3.69 | 0.145 | 34 | 38 | 9 | 0 |
| ko_conv | natural | interim W=0.6 + final remainder | 56 | 2.13 / 4.32 | 0.130 | 31 | 36 | 9 | 0 |
| ko_conv | natural | interim W=1.0 + final remainder | 50 | 2.52 / 4.67 | 0.131 | 26 | 37 | 9 | 0 |
| ko_conv | natural | interim W=0.0 + interim remainder | 66 | 1.44 / 3.69 | 0.141 | 34 | 32 | 9 | 0 |
| ko_conv | natural | interim W=0.6 + interim remainder | 56 | 2.13 / 4.32 | 0.124 | 31 | 29 | 9 | 0 |
| ko_conv | natural | interim W=1.0 + interim remainder | 50 | 2.52 / 4.67 | 0.124 | 29 | 27 | 9 | 0 |
| ko_conv | end_smart | finals only | 21 | 4.14 / 12.15 | 0.112 | 23 | 38 | - | - |
| ko_conv | end_smart | interim W=0.0 + final remainder | 72 | 1.33 / 3.27 | 0.137 | 26 | 34 | 21 | 1 |
| ko_conv | end_smart | interim W=0.6 + final remainder | 55 | 1.91 / 4.12 | 0.124 | 23 | 35 | 21 | 1 |
| ko_conv | end_smart | interim W=1.0 + final remainder | 52 | 2.19 / 4.30 | 0.124 | 22 | 35 | 21 | 1 |
| ko_conv | end_smart | interim W=0.0 + interim remainder | 72 | 1.33 / 3.27 | 0.134 | 26 | 28 | 21 | 1 |
| ko_conv | end_smart | interim W=0.6 + interim remainder | 55 | 1.95 / 4.14 | 0.126 | 25 | 27 | 21 | 1 |
| ko_conv | end_smart | interim W=1.0 + interim remainder | 52 | 2.19 / 4.36 | 0.122 | 25 | 27 | 21 | 1 |
| ko_conv | end_app | finals only | 25 | 2.63 / 5.04 | 0.126 | 32 | 40 | - | - |
| ko_conv | end_app | interim W=0.0 + final remainder | 73 | 1.20 / 2.94 | 0.184 | 34 | 71 | 25 | 0 |
| ko_conv | end_app | interim W=0.6 + final remainder | 67 | 1.70 / 3.33 | 0.180 | 33 | 71 | 25 | 0 |
| ko_conv | end_app | interim W=1.0 + final remainder | 58 | 1.94 / 3.65 | 0.175 | 31 | 66 | 25 | 1 |
| ko_conv | end_app | interim W=0.0 + interim remainder | 73 | 1.20 / 2.94 | 0.194 | 37 | 72 | 25 | 0 |
| ko_conv | end_app | interim W=0.6 + interim remainder | 67 | 1.70 / 3.33 | 0.190 | 37 | 72 | 25 | 0 |
| ko_conv | end_app | interim W=1.0 + interim remainder | 58 | 1.94 / 3.65 | 0.189 | 35 | 70 | 25 | 0 |
| ja_conv | natural | finals only | 26 | 2.17 / 4.58 | 0.349 | 95 | 40 | - | - |
| ja_conv | natural | interim W=0.0 + final remainder | 64 | 1.12 / 2.79 | 0.362 | 100 | 50 | 26 | 2 |
| ja_conv | natural | interim W=0.6 + final remainder | 47 | 1.65 / 3.29 | 0.341 | 92 | 45 | 26 | 1 |
| ja_conv | natural | interim W=1.0 + final remainder | 40 | 1.86 / 3.64 | 0.343 | 91 | 42 | 26 | 1 |
| ja_conv | natural | interim W=0.0 + interim remainder | 64 | 1.13 / 3.02 | 0.360 | 99 | 50 | 26 | 4 |
| ja_conv | natural | interim W=0.6 + interim remainder | 48 | 1.71 / 3.79 | 0.362 | 104 | 49 | 26 | 4 |
| ja_conv | natural | interim W=1.0 + interim remainder | 41 | 1.98 / 4.14 | 0.370 | 104 | 49 | 26 | 4 |
| ja_conv | end_smart | finals only | 35 | 1.76 / 4.49 | 0.370 | 85 | 46 | - | - |
| ja_conv | end_smart | interim W=0.0 + final remainder | 61 | 1.21 / 2.94 | 0.391 | 98 | 50 | 35 | 3 |
| ja_conv | end_smart | interim W=0.6 + final remainder | 45 | 1.53 / 3.39 | 0.385 | 88 | 48 | 35 | 0 |
| ja_conv | end_smart | interim W=1.0 + final remainder | 44 | 1.68 / 3.49 | 0.382 | 86 | 47 | 35 | 0 |
| ja_conv | end_smart | interim W=0.0 + interim remainder | 61 | 1.21 / 2.94 | 0.401 | 95 | 56 | 35 | 3 |
| ja_conv | end_smart | interim W=0.6 + interim remainder | 45 | 1.59 / 3.54 | 0.382 | 93 | 54 | 35 | 3 |
| ja_conv | end_smart | interim W=1.0 + interim remainder | 44 | 1.75 / 3.65 | 0.380 | 92 | 53 | 35 | 3 |
| ja_conv | end_app | finals only | 34 | 1.73 / 3.46 | 0.355 | 102 | 39 | - | - |
| ja_conv | end_app | interim W=0.0 + final remainder | 66 | 1.03 / 2.23 | 0.377 | 103 | 48 | 34 | 3 |
| ja_conv | end_app | interim W=0.6 + final remainder | 47 | 1.50 / 2.70 | 0.367 | 103 | 41 | 34 | 0 |
| ja_conv | end_app | interim W=1.0 + final remainder | 44 | 1.61 / 3.00 | 0.364 | 103 | 39 | 34 | 0 |
| ja_conv | end_app | interim W=0.0 + interim remainder | 66 | 1.05 / 2.40 | 0.378 | 111 | 46 | 34 | 5 |
| ja_conv | end_app | interim W=0.6 + interim remainder | 47 | 1.56 / 3.26 | 0.362 | 108 | 43 | 34 | 5 |
| ja_conv | end_app | interim W=1.0 + interim remainder | 44 | 1.69 / 3.39 | 0.362 | 108 | 43 | 34 | 5 |
| en_conv | natural | finals only | 7 | 10.72 / 43.59 | 0.093 | 39 | 76 | - | - |
| en_conv | natural | interim W=0.0 + final remainder | 30 | 2.78 / 9.97 | 0.090 | 44 | 65 | 7 | 0 |
| en_conv | natural | interim W=0.6 + final remainder | 26 | 3.63 / 10.94 | 0.091 | 44 | 67 | 7 | 0 |
| en_conv | natural | interim W=1.0 + final remainder | 24 | 4.11 / 13.68 | 0.095 | 44 | 74 | 7 | 0 |
| en_conv | natural | interim W=0.0 + interim remainder | 30 | 2.78 / 9.97 | 0.087 | 44 | 61 | 7 | 0 |
| en_conv | natural | interim W=0.6 + interim remainder | 26 | 3.63 / 10.94 | 0.087 | 44 | 61 | 7 | 0 |
| en_conv | natural | interim W=1.0 + interim remainder | 24 | 4.11 / 13.68 | 0.087 | 44 | 61 | 7 | 0 |
| en_conv | end_smart | finals only | 8 | 9.15 / 24.12 | 0.081 | 37 | 54 | - | - |
| en_conv | end_smart | interim W=0.0 + final remainder | 34 | 2.64 / 7.83 | 0.092 | 40 | 63 | 8 | 1 |
| en_conv | end_smart | interim W=0.6 + final remainder | 27 | 3.45 / 10.75 | 0.089 | 41 | 61 | 8 | 1 |
| en_conv | end_smart | interim W=1.0 + final remainder | 24 | 4.10 / 13.37 | 0.089 | 41 | 61 | 8 | 1 |
| en_conv | end_smart | interim W=0.0 + interim remainder | 34 | 2.64 / 7.83 | 0.090 | 40 | 59 | 8 | 0 |
| en_conv | end_smart | interim W=0.6 + interim remainder | 27 | 3.45 / 10.75 | 0.089 | 41 | 60 | 8 | 0 |
| en_conv | end_smart | interim W=1.0 + interim remainder | 24 | 4.10 / 13.37 | 0.089 | 41 | 60 | 8 | 0 |
| en_conv | end_app | finals only | 22 | 2.91 / 5.49 | 0.101 | 64 | 56 | - | - |
| en_conv | end_app | interim W=0.0 + final remainder | 39 | 1.96 / 4.89 | 0.131 | 126 | 25 | 22 | 0 |
| en_conv | end_app | interim W=0.6 + final remainder | 34 | 2.55 / 5.50 | 0.130 | 136 | 20 | 22 | 0 |
| en_conv | end_app | interim W=1.0 + final remainder | 33 | 2.71 / 5.53 | 0.124 | 134 | 16 | 22 | 0 |
| en_conv | end_app | interim W=0.0 + interim remainder | 39 | 1.96 / 4.89 | 0.129 | 126 | 22 | 22 | 0 |
| en_conv | end_app | interim W=0.6 + interim remainder | 34 | 2.55 / 5.50 | 0.129 | 138 | 19 | 22 | 0 |
| en_conv | end_app | interim W=1.0 + interim remainder | 33 | 2.71 / 5.53 | 0.124 | 136 | 17 | 22 | 0 |
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
