# VRChat OSC controls

- Purpose: exchange PuriPuly dashboard state with VRChat through OSC
- PuriPuly setting: **Settings > General > VRChat OSC**
- VRChat setting: **Action Menu > Settings > OSC > Enable**
- Automatic: OSCQuery discovery and dynamic receive port
- Manual: send `9000`, receive `9001` by default
- Off: stop control traffic and preserve manual ports

## Expression Parameters

- Add every required parameter to the avatar's Expression Parameters.
- Match names, capitalization, and types exactly.

| Parameter | Meaning | Type | Default | Saved | Synced |
| --- | --- | --- | ---: | --- | --- |
| `PuriPuly_Talk` | Self capture | Bool | False | Off | Off |
| `PuriPuly_Listen` | Peer capture | Bool | False | Off | Off |
| `PuriPuly_Trans` | Translation | Bool | False | Off | Off |
| `PuriPuly_Captions` | Captions | Bool | False | Off | Off |
| `PuriPuly_PeerAuto` | Peer source auto-detection | Bool | False | Off | Off |
| `PuriPuly_MuteSync` | VRChat mute synchronization | Bool | False | Off | Off |
| `PuriPuly_ChatboxSource` | Include source text in Chatbox output | Bool | False | Off | Off |
| `PuriPuly_SelfSrcLang` | Self source language | Int | 17 | Off | Off |
| `PuriPuly_SelfDstLang` | Self target language | Int | 7 | Off | Off |
| `PuriPuly_PeerSrcLang` | Peer source language | Int | 7 | Off | Off |
| `PuriPuly_PeerDstLang` | Peer target language | Int | 17 | Off | Off |
| `PuriPuly_SelfASR` | Self ASR provider | Int | 0 | Off | Off |
| `PuriPuly_PeerASR` | Peer ASR provider | Int | 0 | Off | Off |
| `PuriPuly_Translator` | Translation model | Int | 0 | Off | Off |
| `PuriPuly_Fallback` | Translation fallback | Int | 0 | Off | Off |

- Default: PuriPuly fresh runtime value
- Saved: Off; PuriPuly owns persistence and republishes its current state
- Synced: Off; local application controls do not require avatar network sync
- Remote visual effects: drive separate synced visual parameters
- `MuteSelf`: VRChat-provided; do not add as a custom Expression Parameter
- Bool menu controls: use Toggle
- Int menu controls: use Bool proxies with Avatar Parameter Drivers
- Avoid Button/Sub-Menu for Int values; they reset to zero when deactivated

## Parameter ABI

- Address: `/avatar/parameters/<name>`
- Bool payload: OSC boolean
- Int payload: fixed ABI ID
- Int ID policy: append-only; never reuse an ID
- Int IDs are the OSC payload; the tables below map each ID to its human-readable name
- Bool commands: absolute and idempotent

### Language IDs

| ID | Language | ID | Language | ID | Language |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | Arabic | 12 | Hindi | 24 | Portuguese |
| 1 | Bulgarian | 13 | Hungarian | 25 | Romanian |
| 2 | Catalan | 14 | Indonesian | 26 | Russian |
| 3 | Czech | 15 | Italian | 27 | Slovak |
| 4 | Danish | 16 | Japanese | 28 | Swedish |
| 5 | German | 17 | Korean | 29 | Thai |
| 6 | Greek | 18 | Lithuanian | 30 | Turkish |
| 7 | English | 19 | Latvian | 31 | Ukrainian |
| 8 | Spanish | 20 | Malay | 32 | Vietnamese |
| 9 | Estonian | 21 | Dutch | 33 | Chinese (Simplified) |
| 10 | Finnish | 22 | Norwegian | 34 | Chinese (Traditional) |
| 11 | French | 23 | Polish | | |

- Used by: `PuriPuly_SelfSrcLang`, `PuriPuly_SelfDstLang`, `PuriPuly_PeerSrcLang`, `PuriPuly_PeerDstLang`

### ASR IDs

| ID | Provider |
| ---: | --- |
| 0 | Auto Select (CPU Inference) |
| 1 | Parakeet TDT 0.6B v3 |
| 2 | Parakeet TDT-CTC 0.6B (ja) |
| 3 | Qwen3 ASR 0.6B |
| 4 | Qwen3 ASR 1.7B |
| 5 | Deepgram |
| 6 | Qwen3 ASR Flash Realtime |
| 7 | Soniox |
| 8 | Custom Speech Recognition (Offline) |
| 9 | Custom Speech Recognition (Realtime) |

- Used by: `PuriPuly_SelfASR`, `PuriPuly_PeerASR`

### Translation model IDs

| ID | Model |
| ---: | --- |
| 0 | Gemma 4 26B + 31B |
| 1 | Gemma 4 31B |
| 2 | Gemma 4 26B A4B |
| 3 | DeepSeek V4 Flash |
| 5 | Gemini 3.7 Flash |
| 6 | Gemini 3.1 Flash-Lite |
| 7 | Qwen 3.5 Plus |
| 8 | OpenAI-compatible API |
| 9 | Custom HTTP API |

- Used by: `PuriPuly_Translator`
- Gemma 4 31B on a Cerebras connection is published as ID `1`; select the connection in PuriPuly.
- ID `9` selects the currently configured custom HTTP API and does not select an individual extension.

### Fallback IDs

| ID | Fallback |
| ---: | --- |
| 0 | Off |
| 1 | DeepSeek V4 Flash (Official API) |
| 2 | DeepSeek V4 Flash (OpenRouter) |
| 3 | Gemma 4 26B A4B (OpenRouter) |
| 4 | Gemma 4 26B + 31B (OpenRouter) |
| 5 | Gemma 4 31B (OpenRouter) |
| 6 | Gemma 4 26B + 31B (Managed) |
| 7 | Gemma 4 31B (Managed) |
| 8 | Gemma 4 31B (Cerebras) |

- Used by: `PuriPuly_Fallback`
