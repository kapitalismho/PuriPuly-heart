# PuriPuly <3

VRChat real-time speech translation pipeline (STT → LLM → OSC).

## Install

- `pip install -e .`
- (Dev) `pip install -e '.[dev]'`

## Dev Tools

- Format: `black src tests`
- Lint: `ruff check src tests`
- Pre-commit: `pre-commit install` (run `pre-commit run --all-files` to verify)

## Secrets

Provider API keys are read from `SecretStore` first, then fall back to environment variables.

- Secret keys:
  - `google_api_key` (Gemini)
  - `alibaba_api_key` (Qwen, Alibaba STT)
  - `deepgram_api_key` (Deepgram STT)
  - `soniox_api_key` (Soniox STT)
- Env vars:
  - `GOOGLE_API_KEY`
  - `ALIBABA_API_KEY`
  - `DEEPGRAM_API_KEY`
  - `SONIOX_API_KEY`

If `settings.secrets.backend = "encrypted_file"`, set `PURIPULY_HEART_SECRETS_PASSPHRASE` before running.

## CLI

- Send one message: `python3 -m puripuly_heart.main osc-send "hello"`
- Stream stdin to VRChat chatbox:
  - Without LLM: `python3 -m puripuly_heart.main run-stdin`
  - With LLM: `GOOGLE_API_KEY=... python3 -m puripuly_heart.main run-stdin --use-llm`
- Capture mic audio (VAD→STT→(LLM)→OSC):
  - `python3 -m puripuly_heart.main run-mic`
  - With LLM: `python3 -m puripuly_heart.main run-mic --use-llm`

## Integration Tests (Opt-in)

Integration tests are skipped by default. Run with:

- `INTEGRATION=1 python3 -m pytest`

Test audio (for E2E/STT+LLM tests):

- Set `TEST_AUDIO_PATH` to a local WAV file, or place it at `.test_audio/test_speech.wav` (gitignored).

### Qwen ASR (DashScope)

Required env vars:

- `ALIBABA_API_KEY` (or `DASHSCOPE_API_KEY`)

Optional:

- `QWEN_ASR_MODEL` (default: `qwen3-asr-flash-realtime`)
- `QWEN_ASR_ENDPOINT` (default: `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime`)
- `QWEN_ASR_LANGUAGE` (default: `ko`)
- `QWEN_ASR_SAMPLE_RATE` (default: `16000`)

### Soniox STT (WebSocket)

Required env vars:

- `SONIOX_API_KEY`

Optional:

- `SONIOX_STT_MODEL` (default: `stt-rt-v3`)
- `SONIOX_STT_ENDPOINT` (default: `wss://stt-rt.soniox.com/transcribe-websocket`)
- `SONIOX_STT_LANGUAGE` (default: `ko`)
- `SONIOX_STT_SAMPLE_RATE` (default: `16000`)
- `SONIOX_STT_KEEPALIVE` (default: `10`)
- `SONIOX_STT_TRAILING_SILENCE_MS` (default: `100`)
