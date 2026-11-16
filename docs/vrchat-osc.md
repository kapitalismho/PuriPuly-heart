# VRChat OSC controls

- Purpose: control PuriPuly from VRChat and synchronize dashboard state
- PuriPuly setting: **Settings > General > VRChat OSC**
- VRChat setting: **Action Menu > Settings > OSC > Enable**
- Automatic: OSCQuery discovery and dynamic receive port
- Manual: send `9000`, receive `9001` by default
- Off: stop control traffic and preserve manual ports

## Progressive Self chatbox translations

- Self translations update the chatbox as speech is recognized, even when the VR overlay is off.
- Later results can correct earlier previews.
- Not every intermediate update or page is shown. Text already read by others cannot be retracted.
- A successful local OSC send does not confirm that VRChat received or displayed the text.

## Expression Parameters

- Add only the parameters you need to the avatar's Expression Parameters.
- Match names, capitalization, and types exactly.

| Parameter | Meaning | Type | Default |
| --- | --- | --- | ---: |
| `PuriPuly_Talk` | Self capture | Bool | False |
| `PuriPuly_Listen` | Peer capture | Bool | False |
| `PuriPuly_Trans` | Translation | Bool | False |
| `PuriPuly_Captions` | Captions | Bool | False |
| `PuriPuly_PeerAuto` | Peer source auto-detection | Bool | False |
| `PuriPuly_MuteSync` | VRChat mute synchronization | Bool | False |
| `PuriPuly_ChatboxSource` | Include source text in Chatbox output | Bool | False |
| `PuriPuly_SelfSrcLang` | Self source language | Int | 17 |
| `PuriPuly_SelfDstLang` | Self primary target language | Int | 7 |
| `PuriPuly_SelfDstLang2` | Self secondary target language; 255 disables it | Int | 255 |
| `PuriPuly_PeerSrcLang` | Peer source language | Int | 7 |
| `PuriPuly_PeerDstLang` | Peer target language | Int | 17 |
| `PuriPuly_SelfASR` | Self ASR provider | Int | 0 |
| `PuriPuly_PeerASR` | Peer ASR provider | Int | 0 |
| `PuriPuly_Translator` | Translation model | Int | 16 |

- Defaults are fresh runtime values.
- Set **Saved** and **Synced** to **Off** for all controls. PuriPuly stores and republishes state.
- Use separate synced parameters for remote visual effects.
- `MuteSelf`: VRChat-provided; do not add as a custom Expression Parameter
- Bool menu controls: use Toggle
- Int menu controls: use Bool proxies with Avatar Parameter Drivers
- Avoid Button/Sub-Menu for Int values; they reset to zero when deactivated

- Remove retired `PuriPuly_Fallback` and `PuriPuly_SmartTurn` parameters.

## Parameter ABI

- Address: `/avatar/parameters/<name>`
- Bool payload: OSC boolean
- Int payload: fixed ABI ID
- Int ID policy: append-only; never reuse an ID
- Bool controls request absolute enabled states, not toggles.
- The app publishes current state and corrections.
- Delivery is unacknowledged; delayed Bool packets can act as new commands.

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

- Used by: `PuriPuly_SelfSrcLang`, `PuriPuly_SelfDstLang`, `PuriPuly_SelfDstLang2`, `PuriPuly_PeerSrcLang`, `PuriPuly_PeerDstLang`
- `PuriPuly_SelfDstLang2`: `255` means Off. Selecting the primary target also disables the secondary target.

### ASR IDs

| ID | Provider |
| ---: | --- |
| 0 | Auto Select (CPU Inference) |
| 1 | Legacy CPU ASR input → Auto Select (CPU Inference) |
| 2 | Parakeet TDT-CTC 0.6B (ja) |
| 3 | Legacy CPU Qwen input → Auto Select (CPU Inference) |
| 4 | Qwen3 ASR 1.7B |
| 5 | Deepgram |
| 7 | Soniox |
| 8 | Custom Speech Recognition (Offline) |
| 9 | Custom Speech Recognition (Realtime) |
| 10 | Gemini Transcribe (Live) |
| 11 | ElevenLabs Scribe |
| 12 | Qwen Audio 3.1 ASR Flash Streaming |
| 13 | Free cloud ASR auto-selection (ElevenLabs Scribe → Gemini → Deepgram) |
| 14 | Fun-ASR-Nano-2512 (Chinese) |
| 15 | Fun-ASR-MLT-Nano-2512 (31 languages) |
| 16 | Granite Speech 5.0 Multilingual (7 languages) |
| 17 | GigaAM-v3 e2e-CTC (Russian only) |

- Used by: `PuriPuly_SelfASR`, `PuriPuly_PeerASR`
- ID `6` is retired and rejected as an unknown ASR value. Qwen Audio uses ID `12`.
- Legacy inputs `1` and `3` normalize to Auto ID `0` on both channels.

### Translation model IDs

| ID | Model |
| ---: | --- |
| 0 | Gemma 4 26B + 31B |
| 3 | DeepSeek V4 Flash |
| 7 | Qwen 3.8 Flash |
| 8 | OpenAI-compatible API |
| 9 | Custom HTTP API |
| 10 | Gemma 4 E4B CPU |
| 11 | Gemma 4 E4B GPU |
| 13 | DeepSeek V4.1 Flash |
| 14 | GPT-6 Luna |
| 15 | Gemma 4 26B A4B GPU |
| 16 | Index-Translate 35B |

- Used by: `PuriPuly_Translator`
- IDs `5` and `6` are retired and rejected. Existing IDs are not renumbered.

Implementation: [parameter schema](../src/puripuly_heart/core/osc/control_schema.py), [control runtime](../src/puripuly_heart/app/services/osc/control_runtime.py).
