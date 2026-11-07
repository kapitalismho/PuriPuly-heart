# Model comparison evidence

[September 2026 benchmark](https://github.com/kapitalismho/korean-llm-context-translation-benchmark/blob/cb967002c845b80ca133097a4011898e89b87d0a/README.md), checked 2026-10-30: 216 Korean conversations × three targets, 648 cells. Gemini 3.7 Flash GEMBA-MQM mean penalty; lower is better. Quality evidence, not latency or default recommendation. Follow the guide's access-aware policy.

## Translation

All valid cells; 612-common-cell ordering agrees. Speech rows include recognition error.

| System | Input | Penalty | Valid cells |
| --- | --- | ---: | ---: |
| GPT-6 Luna | Text | 0.130 | 648 |
| Gemma 4 31B | Text | 0.353 | 648 |
| Gemma 4 26B A4B | Text | 0.387 | 648 |
| DeepSeek V4 Flash 0731 | Text | 0.571 | 648 |
| Gemma 4 12B QAT Q4 | Text | 0.855 | 648 |
| Gemma 4 E4B fp16 | Text | 1.353 | 648 |
| Gemma 4 E4B QAT Q4 | Text | 1.577 | 648 |
| Hy-MT2 7B | Text | 1.863 | 648 |
| Qwen 3.8 Live Translate | Speech | 2.124 | 647 |
| Papago Web | Text | 2.699 | 648 |
| MiLMMT 46-4B | Text | 3.087 | 647 |
| Gemini 3.5 Live Translate | Speech | 3.723 | 638 |
| DeepL API | Text | 3.914 | 642 |
| Soniox Translate | Speech | 5.010 | 630 |
| Google Cloud Translation Basic | Text | 5.731 | 648 |

Speech subsets with current-utterance CER ≤5%: Qwen 1.392/286 cells, Gemini 2.991/223, Soniox 3.473/355. Different subsets; context-audio CER unfiltered, not ASR-free. Product experience rates Soniox more favorably.

## Full speech-to-translation pipeline

Means: 618 matched cells; coverage: valid/648. Separate ASR/LLM measurements, no integrated latency.

| Pipeline | Penalty | Coverage |
| --- | ---: | ---: |
| Gemini 3.5 Transcribe → Luna, effort none | 0.676 | 648 |
| Soniox STT → Luna, effort none | 0.942 | 648 |
| Qwen3-ASR 1.7B → Gemma 4 26B | 1.024 | 648 |
| Gemini 3.5 Transcribe → Gemma 4 26B | 1.084 | 648 |
| Soniox STT → Gemma 4 26B | 1.293 | 648 |
| Qwen 3.8 Live Translate, direct speech | 2.108 | 647 |
| Qwen3-ASR 0.6B → Gemma 4 26B | 2.374 | 647 |
| Gemini 3.5 Live Translate, direct speech | 3.754 | 638 |
| Soniox Translate, direct speech | 4.989 | 630 |

Version limits: retired Qwen CPU ≠ CPU Auto; individual Gemma ≠ combined route; DeepSeek 0731 ≠ V4.1; Transcribe/STT ≠ Live Translate/Translate.

E4B QAT Q4 + policy/history: 2.118 → 1.452 on 642 pairs (31.5%; 236 improved, 134 worsened, 272 tied). Not history's isolated effect or permission to send voice history to Puri.

Source authors: Korean Multi-turn Context Translation Benchmark contributors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); revision in the source link.
