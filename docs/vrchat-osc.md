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
| `PuriPuly_SelfSrcLang` | Self source language | Int | 17 (`ko`) | Off | Off |
| `PuriPuly_SelfDstLang` | Self target language | Int | 7 (`en`) | Off | Off |
| `PuriPuly_PeerSrcLang` | Peer source language | Int | 7 (`en`) | Off | Off |
| `PuriPuly_PeerDstLang` | Peer target language | Int | 17 (`ko`) | Off | Off |
| `PuriPuly_SelfASR` | Self ASR provider | Int | 0 (`local_cpu_auto`) | Off | Off |
| `PuriPuly_PeerASR` | Peer ASR provider | Int | 0 (`local_cpu_auto`) | Off | Off |
| `PuriPuly_Translator` | Translation model | Int | 0 (`gemma4_26b_31b`) | Off | Off |
| `PuriPuly_Fallback` | Translation fallback | Int | 0 (`none`) | Off | Off |

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
- Bool commands: absolute and idempotent

### Language IDs

| ID | Language | ID | Language | ID | Language |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | `ar` | 12 | `hi` | 24 | `pt` |
| 1 | `bg` | 13 | `hu` | 25 | `ro` |
| 2 | `ca` | 14 | `id` | 26 | `ru` |
| 3 | `cs` | 15 | `it` | 27 | `sk` |
| 4 | `da` | 16 | `ja` | 28 | `sv` |
| 5 | `de` | 17 | `ko` | 29 | `th` |
| 6 | `el` | 18 | `lt` | 30 | `tr` |
| 7 | `en` | 19 | `lv` | 31 | `uk` |
| 8 | `es` | 20 | `ms` | 32 | `vi` |
| 9 | `et` | 21 | `nl` | 33 | `zh-CN` |
| 10 | `fi` | 22 | `no` | 34 | `zh-TW` |
| 11 | `fr` | 23 | `pl` | | |

- Used by: `PuriPuly_SelfSrcLang`, `PuriPuly_SelfDstLang`, `PuriPuly_PeerSrcLang`, `PuriPuly_PeerDstLang`

### ASR IDs

| ID | Value |
| ---: | --- |
| 0 | `local_cpu_auto` |
| 1 | `local_parakeet_v3` |
| 2 | `local_parakeet_ja` |
| 3 | `local_qwen` |
| 4 | `local_qwen_gpu` |
| 5 | `deepgram` |
| 6 | `qwen_asr` |
| 7 | `soniox` |

- Used by: `PuriPuly_SelfASR`, `PuriPuly_PeerASR`

### Translation model IDs

| ID | Value |
| ---: | --- |
| 0 | `gemma4_26b_31b` |
| 1 | `gemma4_31b` |
| 2 | `gemma4` |
| 3 | `deepseek_v4_flash` |
| 4 | `deepseek_v4_pro` |
| 5 | `gemini3_flash` |
| 6 | `gemini31_flash_lite` |
| 7 | `qwen35_plus` |
| 8 | `local_llm` |
| 9 | `custom_http` |

- Used by: `PuriPuly_Translator`
- Gemma 4 31B on a Cerebras connection is published as ID `1`; select the connection in PuriPuly.
- ID `9` selects the currently configured custom HTTP API and does not select an individual extension.

### Fallback IDs

| ID | Value |
| ---: | --- |
| 0 | `none` |
| 1 | `deepseek_v4_flash_official` |
| 2 | `openrouter_deepseek_v4_flash` |
| 3 | `openrouter_gemma4_26b_a4b` |
| 4 | `openrouter_gemma4_26b_31b` |
| 5 | `openrouter_gemma4_31b` |
| 6 | `managed_gemma4_26b_31b` |
| 7 | `managed_gemma4_31b` |
| 8 | `cerebras_gemma4_31b` |

- Used by: `PuriPuly_Fallback`
