# Keys and costs

Use approximate **API translation requests per US$1**, not utterances or Puri chat turns. Each model invocation, including a retry/speculative request, counts. Recognition and subscription limits are separate.

| Translation route | Approximate translations / US$1 |
| --- | ---: |
| Gemma 4 26B A4B + 31B | 16,700 |
| DeepSeek V4.1 Flash, official API | 20,200 |
| GPT-6 Luna, OpenAI API | 10,400 |
| DeepSeek V4 Flash, OpenRouter | 20,400 |
| Gemini 3.8 Flash | 1,400 |
| Qwen 3.8 Flash, Beijing | 9,000 |

Basis: README local-ASR estimates, 2026-10-17 (checked 2026-10-30), ×1.2 requests/utterance and rounded; 900 input + 12 output tokens/request. Cache hits: DeepSeek V4.1 70%, OpenRouter V4 60%, Luna 0%. Not account balances. Index/public and local Gemma have no API charge; ChatGPT uses Codex subscription allowance.

**Free credit or generous allowance:** Deepgram $200 trial, no expiry/card; ElevenLabs Scribe 10,000 credits/month, no card; Gemini Transcribe default Free tier. These dated offers do not establish remaining credit or account billing. Alibaba: Singapore 1M tokens/model for 90 days; eligible students in China ¥300/year.

Puri: 50 submitted prompt turns/installation/UTC day. D/M/tools/notices share the submitted turn's slot. Restart/reset does not renew it; no global monthly cap. Settings, Cancel, static help and completed results remain usable.
