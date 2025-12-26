# A-Jebal-Daera-Translator

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
- Env vars:
  - `GOOGLE_API_KEY`
  - `ALIBABA_API_KEY`

If `settings.secrets.backend = "encrypted_file"`, set `AJEBAL_SECRETS_PASSPHRASE` before running.

## CLI

- Send one message: `python3 -m ajebal_daera_translator.main osc-send "hello"`
- Stream stdin to VRChat chatbox:
  - Without LLM: `python3 -m ajebal_daera_translator.main run-stdin`
  - With LLM: `GOOGLE_API_KEY=... python3 -m ajebal_daera_translator.main run-stdin --use-llm`
- Capture mic audio (VAD→STT→(LLM)→OSC):
  - `python3 -m ajebal_daera_translator.main run-mic`
  - With LLM: `python3 -m ajebal_daera_translator.main run-mic --use-llm`

## Integration Tests (Opt-in)

Integration tests are skipped by default. Run with:

- `INTEGRATION=1 python3 -m pytest`

### Google STT v2

Required env vars:

- `GOOGLE_SPEECH_RECOGNIZER` (e.g. `projects/.../locations/.../recognizers/...`)

Optional:

- `GOOGLE_SPEECH_ENDPOINT` (default: `speech.googleapis.com`)
- `GOOGLE_SPEECH_LANGUAGE` (default: `en-US`)

### Alibaba Model Studio STT (DashScope)

Required env vars:

- `ALIBABA_API_KEY` (or `DASHSCOPE_API_KEY`)

Optional:

- `ALIBABA_STT_MODEL` (default: `fun-asr-realtime`)
- `ALIBABA_STT_ENDPOINT` (default: `wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference`)
- `ALIBABA_STT_SAMPLE_RATE` (default: `16000`)
