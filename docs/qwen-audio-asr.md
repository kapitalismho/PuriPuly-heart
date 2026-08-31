# Qwen Audio ASR

PuriPuly supports two Qwen cloud speech-to-text models:

- `qwen3-asr-flash-realtime` uses the Qwen Realtime adapter and endpoint.
- `qwen-audio-3.0-asr-flash-streaming` uses the dedicated Qwen Audio duplex WebSocket adapter and `/api-ws/v1/inference` endpoint.

Both models use the Qwen API key for the selected Beijing or Singapore region. Select the model in Settings under the Qwen ASR provider. The language hint is resolved from the selected source language; unsupported language values fall back to the provider's documented default.

The Qwen Audio adapter keeps one WebSocket connection while rotating provider tasks at each local speech boundary. Local VAD speech-end events are authoritative: each boundary produces exactly one final transcript event, including an empty event when no sentence was recognized. Audio arriving during task transitions is queued and sent to the next task.

When custom vocabulary is enabled, source-language terms are sent as Qwen vocabulary entries with the default weight `4`. Terms are applied to newly started tasks. If Qwen rejects vocabulary parameters, the adapter reports a diagnostic task failure instead of silently dropping the vocabulary.
