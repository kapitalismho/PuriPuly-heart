# Issue 177 Outcome A native workload reference

Executed 2026-09-19 against the frozen `be59afc` Outcome A artifact. This supplements, and does not replace, `windows-baseline-be59afc.md`. No installer, registry, product source, frozen environment, user model cache, credentials, private audio, microphone, or cloud inference was touched.

## Identity, capacity, and isolation

- Frozen executable: `dist/PuriPulyHeart/PuriPulyHeart.exe`, SHA-256 `bf6b0d878c6b41e6e57d764a2d8b9dac945206894f72d5e90047e2750c1b46a0`, 28,469,029 bytes.
- Source files exercised by the fallback CPU decode, provisioning, process capture, and packaged composition were checked with `git diff --exit-code be59afc -- <targeted paths>`; both checks returned 0 with no diff.
- C: free space before download was 426,121,318,400 bytes. The four repository manifests declare 4,006,268,496 bytes, so the bounded download fit with more than 100x headroom.
- Owned root: `.tmp/issue-177-native-workloads/`. `LOCALAPPDATA` was set to its `appdata` child. The reusable model root is `.tmp/issue-177-native-workloads/appdata/puripuly-heart/models`; fixtures are in `.tmp/issue-177-native-workloads/fixtures`.
- `.venv-win` was executed but not modified. The packaged executable was run from unrelated CWD `C:\Windows\Temp`.

## Public fixtures and model identities

The speech WAV URLs are the revision-pinned public `DECODE_CASES` URLs in `local_cpu_real_decode.py`:

| Fixture | Bytes | SHA-256 |
| --- | ---: | --- |
| `parakeet-v3-en.wav` | 184,608 | `148b936b43ce7c546a866e64da059f0458aee2d65e617f16e9d94f06e8d99ed6` |
| `parakeet-ja.wav` | 1,248,044 | `09abd330ce706a6e6969fe6bbc8275314af631fcee4e536afd0626570d263fbf` |
| `qwen-de.wav` | 215,084 | `80bb10c44085a7ce01a17abaf6a2095ed37e1695fca41cc0ea9733f1f24a749c` |

Repository provisioning performed strict size and SHA-256 validation. Installed immutable model files were:

| Model/file | Bytes | SHA-256 |
| --- | ---: | --- |
| Parakeet v3 `decoder.int8.onnx` | 11,845,275 | `179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e` |
| Parakeet v3 `encoder.int8.onnx` | 652,184,281 | `acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247` |
| Parakeet v3 `joiner.int8.onnx` | 6,355,277 | `3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3` |
| Parakeet v3 `tokens.txt` | 93,939 | `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |
| Parakeet JA `model.int8.onnx` | 655,542,604 | `3addd00ef5bd1742078389e540b77394e4a508bdf2f4c9ad1b4a76d93e76598e` |
| Parakeet JA `tokens.txt` | 28,557 | `732f64c53909f2620c713f4106b487d92e6f54a6915b3cd3d1dbd32f9f4f392a` |
| Qwen CPU `conv_frontend.onnx` | 44,148,281 | `d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e` |
| Qwen CPU `decoder.int8.onnx` | 756,563,239 | `61e5f8249f9e7c82d5e01e1938c79fb3f5b3135f91664928033029e42451bd18` |
| Qwen CPU `encoder.int8.onnx` | 182,491,662 | `60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9` |
| Qwen CPU `tokenizer/merges.txt` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| Qwen CPU `tokenizer/tokenizer_config.json` | 12,487 | `4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c` |
| Qwen CPU `tokenizer/vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| Qwen GPU `Qwen3-ASR-1.7B-Q6_K.gguf` | 1,692,554,208 | `c75a961b7134a6c952d89797865cb0d0376876185aee04ef6d12c31c2952e4e1` |

## Provisioning, Xet cancellation, and restart: passed

`.venv-win\Scripts\python.exe .tmp\issue-177-native-workloads\provision_and_fixture.py` used `LocalASRProvisioningOwner` and `HuggingFaceXetDownloadAdapter`. Its worker command was the frozen packaged executable's `hf-xet-download-worker`, not a direct `huggingface_hub` call in the controller.

The first GPU install was cancelled after packaged Xet reported 1% progress. The owner returned `cancelled=true` in 0.062 s, installed no model, and left no `*.staging-*` directory. A new owner generation restarted the same pinned asset and reached `ready`; strict checksum inspection then reported all three CPU models and the GPU model `ready`. Owner close completed. The measured transfer durations are operational observations only, not comparative performance evidence.

Primary report: `.tmp/issue-177-native-workloads/provision-report.json`, SHA-256 `65c96939f4e1e083d607aac503f2a89a6f12e93db9858f84a9ee53438c640c11`.

## CPU real speech decode

The requested packaged command was executed twice from `C:\Windows\Temp`:

```text
PuriPulyHeart.exe local-cpu-real-model-check --model-root <owned-model-root> --audio-root <owned-fixtures> --report <owned-report>
```

Both attempts exited 1 and created no native report. The packaged evidence harness obtained a nonempty final from its first decode, then raised `StopIteration` at `local_cpu_real_decode.py:190` because its `_AttemptHandler` found no expected `[LocalASR][Attempt]` diagnostic. This is a harness/diagnostic-contract failure, so the packaged CPU evidence command is **failed**, not passed.

To separate that reporting defect from native inference, `cpu_decode_observable.py` used the unchanged production `create_local_cpu_backend` factory and strict model catalog validation. All three real speech cases returned a nonempty final `STTBackendTranscriptEvent`:

| Production provider | Fixture | Audio | Observable result |
| --- | --- | ---: | --- |
| `local_parakeet_v3` | English Parakeet | 3.845 s | final, 79 characters |
| `local_parakeet_ja` | Japanese Parakeet | 13.000 s | final, 50 characters |
| `local_qwen` | German Qwen | 6.720 s | final, 105 characters |

This proves the A source production CPU backends and native model closure, but it does not convert the failed packaged command into a pass. Reports:

- `cpu-decode-observable.json`, SHA-256 `3767499659f11372687408a85d2836c690b282038c45968b077b5e0ba58dfd1f`.
- `cpu-packaged-command-result.json`, SHA-256 `2c69094bc49eb4544255e498758b0dfc07667a4ad1bfc82861895b0d39d8d674`.

## Packaged production composition and Radeon GPU: failed before inference

The frozen executable was run twice from `C:\Windows\Temp` with isolated `LOCALAPPDATA`, the validated Q6_K model, the 6.72 s German speech fixture, candidate `A-be59afc`, and expected GPU name `Radeon RX 7900 XTX`.

Both runs discovered and selected `vulkan-index-0`, description `AMD Radeon RX 7900 XTX`, and reported 25,753,026,560 bytes total GPU memory. They composed the real `UiApplicationBoundary`, `LocalASRProviderRuntimeFactory`, `LocalASRProviderRuntimeOwner`, Self and Peer channel owners, disabled external LLM, and applied both provider replacements. Both then failed the production invariant `production Self and Peer did not share one worker` before audio inference. This is a repeatable A production-owner failure, not a missing model/GPU prerequisite and not a direct-engine result. Both shutdowns passed: the owner closed, active channels became empty, worker PID became null, and no named GPU/local-ASR/provider task remained.

Reports:

- `gpu-production-composition.json`, SHA-256 `a193d43b0e0f3ffe3fce52c0193083dc04e4543c90d6b75823b95ae38fdad1fc`.
- `gpu-production-composition-retry.json`, SHA-256 `f7d1421828dd2ec6a9f10dbee1dd3441aba0f1988f3e9e8965022e0b9e8cdd99`.

## Controlled process-audio capture: passed; other capture remains absent

A lawful generated-tone capture was safe because capture was targeted by process PID, not by microphone or whole-device loopback. The existing production process-capture owner launched a 700 Hz target process and an unrelated 1300 Hz control process. It used three native process sources and zero device-loopback sources. The captured target amplitude was `0.1799695454`; control amplitude was `4.5273e-7`, ratio `2.5156e-6`, proving the unrelated process was excluded. Target exit, ordered source/provider/task teardown, typed warning, manual retry with a fresh PID, and final close passed. No ambient/user audio was recorded.

Command:

```text
.venv-win\Scripts\python.exe -m puripuly_heart.release_evidence.windows_process_isolation --evidence .tmp\issue-177-native-workloads\process-isolation.json --thresholds scripts\release\windows-process-isolation-thresholds.json
```

Report SHA-256: `1e8b6d8bd6be5d13f2546e36cdbbacb91f473676f3e05cec8f1b13974556efb7`.

A dedicated physical/virtual endpoint was not provisioned. Microphone, generic device loopback, live VRChat/SteamVR/HMD, and cloud paths remain **not run**. The process-specific result must not be generalized to those paths.

## Outcome A delta

Public prerequisites that were previously absent are now durably available and hash-bound. Repository provisioning, packaged Xet transfer/cancellation/restart, strict model validation, source production CPU decodes, and isolated production process capture passed. Two actual A product gaps remain: the packaged CPU evidence command's stale attempt-diagnostic parser and the packaged GPU production composition's failure to retain one shared worker across Self and Peer. GPU decoding, handoff, worker-crash recovery, and post-inference generation/receipt assertions remain unmet because the production invariant failed first. No fabricated downstream pass is claimed.
