# Issue 177 Outcome C native evidence repair

Repairs the Outcome A packaged-command failures in the evidence harnesses only. Production owner/backend/runtime sources were not changed.

## CPU real decode

A packaged `local-cpu-real-model-check` obtained a nonempty backend final, then raised `StopIteration` at `local_cpu_real_decode.py:190`. `_AttemptHandler` still required a removed `[LocalASR][Attempt] key=value` log. Production now emits human Recognition lines (and may route `basic_log_sink`); that text is not an evidence contract.

The harness now:

- uses `create_local_cpu_backend` and the public session contract (`send_audio_f32`, `on_speech_end`, `STTBackendTranscriptEvent`)
- accepts only a nonempty final
- measures load and decode timing with `perf_counter`
- records model/provider/CPU identity, fixture hashes, `text_length`, and RTF
- does not parse logs or store transcript text

Obsolete log-format fixture tests were deleted.

Source command (isolated fixtures/models; do not overwrite A reports):

```text
.venv-b\Scripts\python.exe -m puripuly_heart.release_evidence.local_cpu_real_decode --model-root .tmp\issue-177-native-workloads\appdata\puripuly-heart\models --audio-root .tmp\issue-177-native-workloads\fixtures --report .tmp\issue-177-native-workloads\cpu-source-repair-result.json
```

Packaged command (C owner artifact; machine-assessable exit 0/1 plus report `status`):

```text
PuriPulyHeart.exe local-cpu-real-model-check --model-root <owned-model-root> --audio-root <owned-fixtures> --report <owned-report>
```

Source proof on `.venv-b` CPython 3.12.10: `.tmp/issue-177-native-workloads/cpu-source-repair-result.json`, SHA-256 `64c113fbf42967fb3ead31a1c9dead82fb6bfe1a621692ba842ff195eee86980`, `status=passed`. Nonempty finals: Parakeet v3 79, Parakeet JA 50, Qwen 105 characters. The final CPython 3.14.7 packaged command also passed all three real models: `C:/c177/cpu-packaged-3147.json`.

## GPU production composition

The original packaged command asserted shared residency before lazy sessions existed. Its first repair then stalled because the evidence flow did not keep each production boundary observable through handoff and recovery. Recovery also rebuilds canonical providers in `ready` state, so the harness must restart the channel handles, use the replacement requests' exact scoped identities, and open a real session before expecting a new worker.

The harness now:

- applies Self and Peer through `replace_provider(..., start=True)` and opens sessions only through the production owned scoped path (`PeerAudioSegmentLedger` plus the Self/Peer VAD owners)
- gives each blocking production stage a finite error with the active task stack, cancels a timed-out operation, and emits stage progress without parsing log wording
- asserts initial shared PID, `{self, peer}` residency, configured physical device, model residency, and matching model identities only after both sessions exist
- starts handoff after Self `SpeechStart`, commits at `SpeechEnd`, accepts the in-flight terminal from the current or retired scoped sink, and proves the replacement terminal without interrupting Peer
- kills the shared worker, observes `retry_required`, runs Controller recovery, restarts the recovered ready channel handles, and dispatches Peer then Self with the canonical post-recovery request scopes
- requires the recovered runtime's requested `auto` identity and a post-failure `activation_ready` resolving the exact selected `vulkan-index-0` physical device
- requires nonempty final `STTProviderTurnTerminal` results, a fresh shared recovered PID, Self release retaining Peer, last release removing the worker, and close leaving no named owned task

Final packaged execution from an unrelated CWD with the isolated `LOCALAPPDATA` passed:

```text
C:/c177/output/dist-final/PuriPulyHeart/PuriPulyHeart.exe local-asr-production-composition-evidence --audio <repo>/.tmp/issue-177-native-workloads/fixtures/qwen-de.wav --report C:/c177/gpu-production-composition-definitive.json --candidate issue-177-cpython-3.14.7-definitive --expected-gpu-name "Radeon RX 7900 XTX"
```

Report: `C:/c177/gpu-production-composition-definitive.json`, SHA-256 `2332ee6e0d7979ce07f0a7c38e85ac9ccff02926e1c64f968c32262a375c8c95`, `status=passed`, elapsed 10.564 s. Initial Self and Peer shared PID 28964 on Radeon RX 7900 XTX; Self handoff kept that PID and produced both in-flight and replacement finals. The deliberate crash removed PID 28964 and produced `frame_reader_failed`/`retry_required`. Controller recovery was lazy, then post-failure activation resolved `auto` to `vulkan-index-0`; recovered Peer and Self shared fresh PID 30304 and both produced finals. Releasing Self retained Peer and PID 30304. Releasing Peer removed the worker, and final close reported `closed=true`, no active channels, no worker PID, and no remaining named task.

The bound packaged executable SHA-256 was `40b31d85497755447ed7b81f7bb678d768f488d15615e789b04a231686785cd3`; the packaged harness module matched source SHA-256 `f6fd7061ac0ee822c7a750618d7f8e1bf1aa5577645e41dfd96fb55ffa29ff9f`. After GPU evidence and final GUI `WM_CLOSE`, a CIM scan for executable paths under the package root returned zero rows; both GUI reports also recorded `surviving_owned_processes=[]`.

## Focused tests

```text
C:/c177/source/.venv-c/Scripts/python.exe -m pytest tests/release_evidence/test_local_cpu_real_decode.py tests/release_evidence/test_local_asr_production_composition.py
```

CPython 3.14.7 result: `19 passed`. Targeted Ruff check of the repaired module and its test passed.
