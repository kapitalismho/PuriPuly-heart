# VRChat OSC controls

PuriPuly can exchange its dashboard state with VRChat through OSC. Open **Settings > General > VRChat OSC > Connection** and choose one of these modes:

| Mode | Behavior |
| --- | --- |
| Automatic | Discovers VRChat through OSCQuery, uses a dynamically assigned receive port, advertises PuriPuly's parameter tree, and follows VRChat's discovered OSC send port. |
| Manual | Sends to port `9000` and listens on port `9001` by default. Both ports can be changed. |
| Off | Stops PuriPuly OSC control traffic while preserving the saved port values. |

VRChat must have OSC enabled at **Action Menu > Settings > OSC > Enable**. Automatic mode uses the local mDNS services `_oscjson._tcp` and `_osc._udp`; manual mode does not require OSCQuery discovery.

## Avatar and Action Menu setup example

In the avatar's Expression Parameters, add the fixed names below with the shown OSC types. Add all seven Boolean parameters as `Bool` parameters and all eight selection parameters as `Int` parameters; keep the names and types unchanged when updating an avatar.

| Parameter group | Parameters | Expression type |
| --- | --- | --- |
| Dashboard and behavior | `PuriPuly_Talk`, `PuriPuly_Listen`, `PuriPuly_Trans`, `PuriPuly_Captions`, `PuriPuly_PeerAuto`, `PuriPuly_MuteSync`, `PuriPuly_ChatboxSource` | Bool |
| Language selection | `PuriPuly_SelfSrcLang`, `PuriPuly_SelfDstLang`, `PuriPuly_PeerSrcLang`, `PuriPuly_PeerDstLang` | Int |
| Engine and fallback selection | `PuriPuly_SelfASR`, `PuriPuly_PeerASR`, `PuriPuly_Translator`, `PuriPuly_Fallback` | Int |

In the Action Menu, use Toggle controls for the Boolean parameters. Keep the integer `PuriPuly_*` parameters out of the menu controls: Button and Sub-Menu controls reset their associated parameter to zero when they deactivate, so a direct menu binding would turn a nonzero selection into ID `0`.

Use persistent proxy parameters for integer selections:

1. Add one `Bool` proxy Expression Parameter per selectable ID, such as `PuriPuly_Menu_SelfSrcLang_16`. These proxy names are avatar-local menu state, not part of the PuriPuly OSC ABI.
2. Add a Toggle for each proxy in a **PuriPuly** submenu. Use one submenu per selection family and keep the toggles mutually exclusive.
3. In the avatar's FX or another local playable layer, create one state for each proxy. On entry to a state, use an **Avatar Parameter Driver** to set the matching `PuriPuly_*` Int parameter to its fixed ID and clear the other proxies in that family. Enable **Local Only** for the driver. The driver sets ID `0` explicitly when the ID `0` proxy is selected.
4. Initialize one proxy in each family as enabled. The proxy remains enabled until another selection clears it, while the driver keeps the canonical `PuriPuly_*` Int parameter at the selected value for OSC publication.
5. Add reverse states for each canonical `PuriPuly_*` Int target and ID, with a condition such as `PuriPuly_SelfSrcLang == 16`. Their **Avatar Parameter Driver** sets the matching proxy Bool to `true` and clears the other proxies in that family. Enable **Local Only**. These reverse drivers run when a desktop or OSC change publishes a canonical Int value, so the Action Menu indicator follows the actual selected ID.

This proxy-and-driver arrangement avoids transient menu resets and keeps an absolute Int selection stable after the Action Menu closes. Puppet controls are not suitable for these parameters because VRChat Puppet controls drive Float values. Initialize Boolean values to `false` and the proxy-driven integer values to the IDs for the desired initial state from the tables below.

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

Boolean commands are absolute and idempotent. Sending the same value twice does not create a second state transition.

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

## Synchronization behavior

PuriPuly sends the complete state when OSC starts, when VRChat is discovered, and after `/avatar/change`. Ordinary changes send only changed values. Sent values are remembered so an echoed value from VRChat does not cause a publication loop. Incoming integer changes are serialized and coalesced per parameter so rapid slider or menu updates do not apply stale expensive settings repeatedly.

The receiver shares one UDP socket for PuriPuly controls, `/avatar/parameters/MuteSelf`, and `/avatar/change`. Existing Chatbox output remains on the configured send destination.

If automatic discovery is unavailable, PuriPuly keeps its receiver available and falls back to the saved manual send destination until VRChat is discovered.
