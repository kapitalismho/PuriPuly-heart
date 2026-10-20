# Smart Turn at 200 ms pauses

Silero VAD finds pauses in the probe audio. Smart Turn v3.2 (the bundled application model) scores each pause of at least 200 ms with up to 8 s of audio ending 200 ms into the pause; the context starts at the speech onset after the last pause of at least 0.5 s. Each pause is labeled with the Soniox final token that ended within 0.2 s after or 0.6 s before the pause start: `sentence` when sentence-final punctuation follows it, `clause` for a comma, `mid` otherwise, `none` when no token ended there.

- `cut at p >= x`: the pauses a cut rule would use. `x = 0` means every 200 ms pause.
- `sentence recall`: share of Soniox sentence ends with such a cut.
- `200 ms pause at sentence ends`: share of Soniox sentence ends followed by any pause of at least 200 ms (the ceiling for any 200 ms pause rule).

## read speech

- 200 ms pauses: 52 in 6.31 min.
- 200 ms pause at sentence ends: 27% of 51.
- AUC sentence vs mid: 0.911
- Mean score: {'sentence': 0.858, 'clause': 0.625, 'mid': 0.266}

| data | p >= | cuts/min | sentence | clause | mid | none | sentence recall |
|---|---|---|---|---|---|---|---|
| all | 0.0 | 8.25 | 27% | 41% | 31% | 1 | 0.275 |
| all | 0.3 | 4.76 | 40% | 47% | 13% | 0 | 0.235 |
| all | 0.5 | 4.6 | 41% | 45% | 14% | 0 | 0.235 |
| all | 0.75 | 4.44 | 43% | 46% | 11% | 0 | 0.235 |
| all | 0.9 | 4.12 | 46% | 42% | 12% | 0 | 0.235 |
| ko_kr | 0.0 | 12.08 | 21% | 42% | 38% | 1 | 0.278 |
| ko_kr | 0.5 | 4.35 | 44% | 44% | 11% | 0 | 0.222 |
| ko_kr | 0.75 | 3.86 | 50% | 50% | 0% | 0 | 0.222 |
| ja_jp | 0.0 | 6.0 | 38% | 38% | 23% | 0 | 0.278 |
| ja_jp | 0.5 | 5.08 | 36% | 45% | 18% | 0 | 0.222 |
| ja_jp | 0.75 | 5.08 | 36% | 45% | 18% | 0 | 0.222 |
| en_us | 0.0 | 6.76 | 29% | 43% | 29% | 0 | 0.267 |
| en_us | 0.5 | 4.34 | 44% | 44% | 11% | 0 | 0.267 |
| en_us | 0.75 | 4.34 | 44% | 44% | 11% | 0 | 0.267 |

## conversation

- 200 ms pauses: 77 in 6.25 min.
- 200 ms pause at sentence ends: 35% of 140.
- AUC sentence vs mid: 0.725
- Mean score: {'sentence': 0.826, 'clause': 0.539, 'mid': 0.545}

| data | p >= | cuts/min | sentence | clause | mid | none | sentence recall |
|---|---|---|---|---|---|---|---|
| all | 0.0 | 12.32 | 59% | 12% | 29% | 4 | 0.3 |
| all | 0.3 | 9.44 | 68% | 9% | 23% | 3 | 0.264 |
| all | 0.5 | 9.28 | 67% | 9% | 24% | 3 | 0.257 |
| all | 0.75 | 8.16 | 69% | 10% | 20% | 2 | 0.236 |
| all | 0.9 | 6.88 | 71% | 12% | 17% | 1 | 0.207 |
| ko_conv | 0.0 | 12.0 | 52% | 0% | 48% | 4 | 0.244 |
| ko_conv | 0.5 | 8.64 | 67% | 0% | 33% | 3 | 0.222 |
| ko_conv | 0.75 | 6.72 | 75% | 0% | 25% | 2 | 0.2 |
| ja_conv | 0.0 | 21.12 | 64% | 20% | 16% | 0 | 0.422 |
| ja_conv | 0.5 | 17.28 | 67% | 14% | 19% | 0 | 0.359 |
| ja_conv | 0.75 | 16.32 | 65% | 15% | 21% | 0 | 0.328 |
| en_conv | 0.0 | 3.84 | 50% | 0% | 50% | 0 | 0.129 |
| en_conv | 0.5 | 1.92 | 75% | 0% | 25% | 0 | 0.097 |
| en_conv | 0.75 | 1.44 | 100% | 0% | 0% | 0 | 0.097 |

## Cut policy preview

The same cut schedules the live Qwen Audio and Gemini variants use (`segmenter.py`). `app` approximates the current Peer rules (Smart Turn complete at 0.75 then 512 ms, otherwise 800 ms; 4 s/192 ms; 5 s/128 ms; 6 s hard cut with 0.3 s replay). `pause` cuts at a 200 ms pause after 2 s. `smart` cuts at a 200 ms pause when Smart Turn is at least 0.5. `fixed3` cuts every 3 s. All but `app` end a turn at an 800 ms pause, cut at an 80 ms pause after 6 s and cut in speech after 10 s.

| data | policy | segments/min | mean s | max s | cuts |
|---|---|---|---|---|---|
| ko_kr | app | 11.11 | 5.56 | 6.32 | in speech 8, pause clause 5, pause mid 3, pause sentence 6 |
| ko_kr | pause | 13.04 | 4.6 | 10.14 | in speech 1, pause clause 8, pause mid 9, pause none 1, pause sentence 7 |
| ko_kr | smart | 9.18 | 6.54 | 10.14 | in speech 1, pause clause 4, pause mid 7, pause sentence 6 |
| ko_kr | fixed3 | 19.81 | 3.03 | 3.42 | in speech 34, pause clause 1, pause mid 3, pause sentence 2 |
| ja_jp | app | 10.62 | 5.82 | 6.32 | in speech 13, pause clause 2, pause mid 2, pause sentence 5 |
| ja_jp | pause | 11.08 | 5.41 | 8.64 | pause clause 8, pause mid 7, pause sentence 8 |
| ja_jp | smart | 10.62 | 5.65 | 8.64 | pause clause 7, pause mid 7, pause sentence 8 |
| ja_jp | fixed3 | 20.32 | 2.95 | 3.07 | in speech 42, pause clause 1 |
| en_us | app | 10.62 | 5.87 | 6.32 | in speech 16, pause clause 1, pause mid 3, pause sentence 1 |
| en_us | pause | 9.65 | 6.21 | 10.27 | in speech 3, pause clause 6, pause mid 6, pause sentence 4 |
| en_us | smart | 10.14 | 5.92 | 10.27 | in speech 3, pause clause 6, pause mid 5, pause none 1, pause sentence 5 |
| en_us | fixed3 | 20.28 | 2.96 | 3.26 | in speech 38, pause clause 2, pause sentence 1 |
| ko_conv | app | 12.0 | 5.05 | 6.32 | in speech 7, pause mid 6, pause none 3, pause sentence 8 |
| ko_conv | pause | 11.04 | 5.39 | 10.14 | in speech 3, pause mid 9, pause none 3, pause sentence 7 |
| ko_conv | smart | 12.48 | 4.75 | 10.11 | in speech 4, pause mid 8, pause none 3, pause sentence 10 |
| ko_conv | fixed3 | 20.16 | 2.97 | 3.51 | in speech 38, pause mid 1, pause none 1, pause sentence 1 |
| ja_conv | app | 14.88 | 3.7 | 6.52 | in speech 4, pause clause 4, pause mid 4, pause sentence 18 |
| ja_conv | pause | 13.44 | 4.0 | 8.6 | pause clause 4, pause mid 5, pause sentence 18 |
| ja_conv | smart | 20.16 | 2.64 | 6.85 | pause clause 7, pause mid 8, pause sentence 26 |
| ja_conv | fixed3 | 19.2 | 2.88 | 3.51 | in speech 22, pause clause 3, pause mid 3, pause sentence 11 |
| en_conv | app | 11.04 | 5.6 | 6.43 | in speech 12, pause mid 5, pause sentence 5 |
| en_conv | pause | 8.16 | 7.32 | 10.52 | in speech 8, pause mid 4, pause sentence 4 |
| en_conv | smart | 7.68 | 7.78 | 10.52 | in speech 6, pause mid 6, pause sentence 3 |
| en_conv | fixed3 | 20.16 | 2.97 | 3.51 | in speech 38, pause mid 2, pause sentence 1 |
