# Model comparison evidence

Korean dialogue → English/Japanese/Simplified Chinese; Astra-high MQM penalty, lower is better. Quality evidence, not speed or default recommendations.

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
| Saved Gemini 3.5 Transcribe → Index-Translate 35B-A3B | 1.567 |
| Fun-ASR-MLT-Nano Q8_0 → Index-Translate 35B-A3B | 2.154 |
