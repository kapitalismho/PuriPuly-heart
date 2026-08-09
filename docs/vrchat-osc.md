# VRChat OSC controls

PuriPuly exchanges its dashboard state with VRChat through OSC. Enable it at **Settings > General > VRChat OSC**; VRChat must have OSC enabled at **Action Menu > Settings > OSC > Enable**. Connection modes: Automatic (OSCQuery discovery, dynamic ports), Manual (send `9000`, receive `9001`), Off (stops control traffic, preserves saved ports).

## Expression Parameters

Add the fixed names below to the avatar's Expression Parameters with the shown types; keep names and types unchanged when updating an avatar.

| Parameter group | Parameters | Expression type |
| --- | --- | --- |
| Dashboard and behavior | `PuriPuly_Talk`, `PuriPuly_Listen`, `PuriPuly_Trans`, `PuriPuly_Captions`, `PuriPuly_PeerAuto`, `PuriPuly_MuteSync`, `PuriPuly_ChatboxSource` | Bool |
| Language selection | `PuriPuly_SelfSrcLang`, `PuriPuly_SelfDstLang`, `PuriPuly_PeerSrcLang`, `PuriPuly_PeerDstLang` | Int |
| Engine and fallback selection | `PuriPuly_SelfASR`, `PuriPuly_PeerASR`, `PuriPuly_Translator`, `PuriPuly_Fallback` | Int |

In the Action Menu, use Toggle controls for the Booleans. Menu Button/Sub-Menu controls reset their parameter to zero when deactivated, so keep the Int parameters out of the menu and drive them through Bool proxy toggles with Avatar Parameter Drivers.

## Parameter ABI

All parameters use the address `/avatar/parameters/<name>`. Boolean values are OSC booleans. Integer values are fixed IDs; IDs are append-only and must not be reused.

### Boolean controls

| Parameter | Meaning |
| --- | --- |
| `PuriPuly_Talk` | Self capture |
| `PuriPuly_Listen` | Peer capture |
| `PuriPuly_Trans` | Translation |
| `PuriPuly_Captions` | Captions |
| `PuriPuly_PeerAuto` | Peer source auto-detection |
| `PuriPuly_MuteSync` | VRChat mute synchronization |
| `PuriPuly_ChatboxSource` | Include source text in Chatbox output |

Boolean commands are absolute and idempotent; sending the same value twice causes no second transition.

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

Language IDs are used by `PuriPuly_SelfSrcLang`, `PuriPuly_SelfDstLang`, `PuriPuly_PeerSrcLang`, and `PuriPuly_PeerDstLang`.

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

These IDs are used by `PuriPuly_SelfASR` and `PuriPuly_PeerASR`.

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
| 9 | `gemma4_31b_cerebras` |

These IDs are used by `PuriPuly_Translator`.

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

These IDs are used by `PuriPuly_Fallback`.
