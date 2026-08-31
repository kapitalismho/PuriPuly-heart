# Qwen Audio operational evidence — 2026-08-31

Evidence was collected with the opt-in harness `scripts/qwen_audio_live.py` using the existing Windows keyring service `puripuly-heart`. The harness is dry-run by default; every provider call below included explicit `--live`. Key material was never printed or persisted. The region-specific key names were `alibaba_api_key_beijing` and `alibaba_api_key_singapore`; both were present. Environment variables were not used.

Commands (all credentials resolved internally; no key argument was supplied):

- `uv run python scripts/qwen_audio_live.py --live --region both --audio .tmp/korean-sapi.wav --report .tmp/qwen-audio-full-v9.json --realtime-delay 0.001 --task-timeout 20`
- `uv run python scripts/qwen_audio_live.py --live --comparison-only --region both --audio .tmp/korean-sapi.wav --report .tmp/qwen-audio-comparison-v8.json --task-timeout 20`
- `uv run python scripts/qwen_audio_live.py --live --reconnect-only --region both --audio .tmp/korean-sapi.wav --report .tmp/qwen-audio-reconnect-v10.json --task-timeout 20`

The retained JSON reports are schema version 2 and contain no task IDs, protocol headers, credentials, or transcripts by default. Transcript retention requires the explicit `--retain-transcripts` opt-in; this operational report does not reproduce provider transcript payloads.

### Reproducible local SAPI fixture

From PowerShell on Windows, this exact command selects `Microsoft Heami Desktop - Korean`, speaks the fixed utterance, and writes a mono 16-bit 22,050 Hz WAV without any external service:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command '$ErrorActionPreference="Stop"; $out=Join-Path (Get-Location) ".tmp\korean-sapi.wav"; New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null; $voice=New-Object -ComObject SAPI.SpVoice; $token=$voice.GetVoices() | Where-Object { $_.GetDescription() -eq "Microsoft Heami Desktop - Korean" } | Select-Object -First 1; if(-not $token){throw "Microsoft Heami Desktop - Korean not installed"}; $voice.Voice=$token; $stream=New-Object -ComObject SAPI.SpFileStream; $stream.Format.Type=22; $stream.Open($out,3,$false); $voice.AudioOutputStream=$stream; [void]$voice.Speak("안녕하세요 PuriPuly"); $stream.Close()'
```

The harness reads that WAV, validates mono/16-bit PCM, and deterministically resamples it to 16 kHz with its local linear-interpolation conversion before sending PCM. The same command with `$out` set to `.tmp\korean-hotword-sapi.wav` is the hotword fixture.

## Live Qwen Audio smoke

Model: `qwen-audio-3.0-asr-flash-streaming`; language hint: `ko`; sample rate: 16 kHz mono PCM; endpoints were the region defaults.

| Region | Tasks | Cases | Usage duration sum | Usage fields observed | Boundary → terminal (mean, 11 cases) | Final drain | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| Beijing | 12 | 0.3 s x2, 0.8 s x2, 1.2 s x2, 2 s, 4 s, 7 s, 7.0+0.2 s split, 0.3 s drain | 24 s | 12/12 | 0.604 s | terminal event received before close | measured |
| Singapore | 12 | 0.3 s x2, 0.8 s x2, 1.2 s x2, 2 s, 4 s, 7 s, 7.0+0.2 s split, 0.3 s drain | 24 s | 12/12 | 0.898 s | terminal event received before close | measured |

The 7-second split sent the immediate 0.2-second post-boundary replay while the next task was being admitted. Each region produced two ordered terminal events for that split. The final 0.3-second stream was drained with `stop()` before the socket closed.

Split conservation is exact in both regions: expected and sent identities match for `[224000, 6400]` bytes and `[70, 2]` frames. Whole-stream SHA-256 values are `631d7b36c3c645551881519fb9664267ba1edc1831a9868f0486f2afd7a8aba6` and `dfacf3681aab1b059fe59eb259ee53b48e17646bec5441afa9a8bba926756098`; ordered per-frame SHA-256 arrays are also equal.

Terminal transcript payloads are excluded from the persisted report by default. The short clipped cases are therefore recorded only by duration, event ordering, usage fields, and PCM identities; no transcription or accuracy claim is made.

Schema v2 region `status` is an aggregate: normal mode is measured only when its required Qwen3 and both hotword transport children are measured; hotword-only is measured only when both hotword children are `measured_transport`. Children not run by comparison, split, or reconnect-only modes remain `not_run` and do not gate that mode's region status.

## Provider event shape and usage evidence

- `task-started`: 12; payload present; no `usage.duration` field.
- `result-generated`: 72; sentence/output fields present; `usage.duration` appeared at both `payload.usage.duration` and `payload.output.usage.duration` in the result stream. The harness recursively extracts either nested path. Exactly 12 terminal sentence events were retained after duplicate sentence records were suppressed.
- `task-finished`: 12; payload/output fields present; no `usage.duration` field.

The task records report provider `usage.duration` values of `0, 0, 1, 1, 1, 1, 2, 4, 7, 7, 0, 0` seconds in each region, summing to 24 seconds. A late `result-generated` event after `task-finished` cannot overwrite the finished task's usage or terminal result. These are provider-reported values, not inferred billing units. No console/account billing view was available, so billing rounding, price, and cost remain **BLOCKED**.

## Qwen3 comparison

The comparison command sent exactly the same 25,600-byte 0.8-second PCM buffer to both providers in each region and measured from the same local speech-boundary call until the first terminal event:

| Region | Qwen Audio common metric | Qwen3 common metric | Audio − Qwen3 delta |
| --- | ---: | ---: | ---: |
| Beijing | 0.328 s | 0.078 s | +0.250 s |
| Singapore | 0.781 s | 0.109 s | +0.672 s |

The values are one paired smoke sample per region, not a benchmark. Both providers use the same boundary-to-terminal metric and the same PCM bytes; the comparison is separate from the 12-task transition smoke above.

## Korean voice and hotword toggle

Windows SAPI inventory found `Microsoft Heami Desktop - Korean` (`MSTTS_V110_koKR_HeamiM` in the OneCore voice registry). A local-only SAPI fixture was generated with the utterance `안녕하세요 PuriPuly`; its original format was mono, 16-bit, 22,050 Hz and the harness resampled it locally to 16 kHz. No external TTS service was used.

The full v9 live run exercised the hotword transport toggle in both regions:

- Enabled: requested hotword `PuriPuly`, outgoing vocabulary `{PuriPuly: 4}`, usage `2` seconds.
- Disabled: outgoing vocabulary absent, usage `2` seconds.

The toggle transport contract is measured (`hotword_transport: measured`). The provider did not distinguish this synthetic SAPI pronunciation with or without the vocabulary, so a recognition-quality difference is **BLOCKED** rather than inferred.

## Reconnect and protocol/error paths

An intentional mid-lifecycle `abort_for_toggle_off()` was followed by a fresh session and a completed 0.8-second task in each region:

- Beijing: abort elapsed `1.672 s`; fresh-session recovery elapsed `0.703 s`; terminal event received.
- Singapore: abort elapsed `0.656 s`; fresh-session recovery elapsed `1.141 s`; terminal event received.

The v10 report records `abort_status: measured` separately from recovery status. The scripted adapter protocol/error tests were run from the accepted adapter implementation. The timeout test was made deterministic by opening with the normal five-second startup send timeout, then setting the short `0.01 s` timeout only after startup before injecting a blocked PCM send. No production adapter code was changed.

## Blocked items

- Model Studio console/account access was unavailable: no cost or billing-rounding comparison is claimed; machine status is `billing: BLOCKED`.
- No independent repeated utterance fixture was available; the local SAPI fixture was reused and clipped/padded for deterministic duration cases. Korean transcription was exercised, but accuracy/generalization is not claimed.
- Hotword enabled-vs-disabled recognition difference was not demonstrated by this synthetic fixture; transport and vocabulary fields are measured, while machine status remains `hotword_recognition_delta: BLOCKED`.

## Harness and focused checks

- `uv run --extra dev pytest tests/providers/test_qwen_audio.py tests/integration/test_qwen_audio_live_harness.py -q` — `39 passed`.
- `uv run python -m py_compile scripts/qwen_audio_live.py tests/integration/test_qwen_audio_live_harness.py` — passed.
