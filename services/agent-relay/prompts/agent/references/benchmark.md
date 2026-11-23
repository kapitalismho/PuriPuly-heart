# Model comparison evidence

Source: the Korean multi-turn translation benchmark linked from the README, recorded here 2026-11-07. Korean dialogue translated into English, Japanese and Simplified Chinese, scored as GEMBA-MQM penalty by a GPT-6 Astra judge at high effort; lower is better. The README reports 216 multi-turn samples for Experiment 2. These are quality results for that language set and these exact pipelines only, not speed, cost, other input languages or the product's recommendation policy, which the guide states separately. Name those conditions when citing a result.

## Experiment 1: Translation

| Translation option | Penalty |
| --- | ---: |
| GPT-6 Luna | 0.189 |
| DeepSeek V4.1 Flash, thinking disabled | 0.452 |
| Gemma 4 26B A4B | 0.639 |
| DeepSeek V4 Flash 0731 | 0.727 |
| Index-Translate 35B-A3B | 1.090 |
| Gemma 4 E4B QAT Q4, local llama.cpp | 1.927 |

## Experiment 2: Recognition → translation

| Pipeline | Penalty |
| --- | ---: |
| Gemini 3.5 Transcribe → GPT-6 Luna | 0.608 |
| Soniox STT → GPT-6 Luna | 0.795 |
| Gemini 3.5 Transcribe → Gemma 4 26B A4B | 1.083 |
| Soniox STT → Gemma 4 26B A4B | 1.317 |
| Qwen3-ASR 1.7B → Gemma 4 26B A4B | 1.320 |
| Fun-ASR-MLT-Nano Q8_0 → Gemma 4 26B A4B | 1.494 |
| Gemini 3.5 Transcribe → Index-Translate 35B-A3B | 1.567 |
| Fun-ASR-MLT-Nano Q8_0 → Index-Translate 35B-A3B | 2.154 |
