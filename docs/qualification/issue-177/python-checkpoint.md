# Issue 177 — Outcome C and source/PyInstaller boundary checkpoint

## Result

**Ready for committed independent review, but not blanket issue/release approval.** The independently assembled Windows CPython 3.14.7 candidate now passes the software, native-runtime, packaged CPU/GPU, GUI, and isolated installer lifecycle gates assigned to Outcome C. The remaining gaps require external hardware/services or a safe published-upgrade environment: physical microphone/loopback, HMD/OpenVR, VRChat, cloud-provider credentials/endpoints, and upgrade of an actual published installer identity.

## Candidate and identity

- Source snapshot: `C:/c177/source`; final package: `C:/c177/output/dist-final/PuriPulyHeart` (1,145 files, 585,777,608 bytes).
- Ordinary-GIL official CPython 3.14.7 x64 (`C:/c177/Python3147Nuget/tools`), PyInstaller 6.21.0, Flet/Flet Desktop 1.0.0, Inno Setup 6.6.1.
- `PuriPulyHeart.exe`: SHA-256 `40b31d85497755447ed7b81f7bb678d768f488d15615e789b04a231686785cd3`.
- `PuriPulyHeartOverlay.exe`: `8892adab3af227f15bfca90fe9d7df754c08464c7384a5647b94eb90001b65c8`.
- `PuriPulyHeartGpuWorker.exe`: `73470cddfb8452a8290f81c2e7390d2379131eea8045b836cbb9136a54a3015a`.
- Final isolated installer: `C:/c177/output/installer-final/PuriPulyHeart-Setup-2.7.0.exe`, SHA-256 `5f02626aa5451e9ad1d7a41e212ed720f14d77051519390003fed475e14d48df`.
- Repository and snapshot bytes match for `build.spec`, `pyproject.toml`, `uv.lock`, and both native evidence modules. Final GPU module SHA-256 is `f6fd7061ac0ee822c7a750618d7f8e1bf1aa5577645e41dfd96fb55ffa29ff9f`.
- `C:/c177/artifact-inventory-definitive.json` records package/native DLL/PYD hashes and the 12 relevant locked wheel filename/hash sets; its SHA-256 is `d80677d0099f8293cf9b17b6a611386cb6735ce1342ca25c14f21ab96d6abec5`.

## Runtime boundary and diagnostics

`RuntimeLayout` is the single immutable source/PyInstaller/native-host boundary for resources, native runtimes, executable roots, user data, models, and logs. Source resolution is repository-relative; packaged resolution is `_MEIPASS`/executable-relative; native-host resolution requires explicit bootstrap roots. Runtime callers no longer depend on the launch CWD.

The shutdown owner exposes an on-demand `ApplicationShutdownStallDiagnostic` with CPython 3.14 named-task/await graphs, owner state and generation, active native work, child-process state, and the honest marker `native_stack_available=false`. The focused retained-native-work diagnostic test passed and repeated close returns the same terminal snapshot.

## Definitive evidence

| Check | Result and evidence |
|---|---|
| Full suite | **PASS:** 6,042 passed, 37 skipped, 37 warnings in 192.81 s on CPython 3.14.7. JUnit `C:/c177/full-suite-definitive.xml`, SHA-256 `5928786e8f63e1f83a4aadfa58a890d195d9e14a325e0216920ed1cfdd4fee29`. |
| Real CPython native workload | **PASS:** ordinary GIL, AMD64; ORT 1.28.0 CPU/Azure, NumPy 2.5.1; five real Silero VAD inferences, three SmartTurn ONNX inferences, and native soxr 48 kHz→16 kHz resample. `C:/c177/native-probe-314.json`, SHA-256 `c494d7a616a431530c5d6bc0fd500369b794a2d36ad9b1f3d09692e95f118b3d`. |
| Packaged CPU ASR | **PASS:** three installed real models/public WAVs produced nonempty finals of 79/50/105 characters, CPU RTF 0.032/0.028/0.122. Report binds frozen CPython 3.14.7 and the exact final EXE hash. `C:/c177/cpu-packaged-definitive.json`, SHA-256 `5cbf5ae2ce24c504fcc57252a2740f342a186d36148f5dbc8a731687031f004e`. |
| Packaged GPU production composition | **PASS:** Radeon RX 7900 XTX (`vulkan-index-0`); initial Self/Peer shared PID 28964; in-flight handoff preserved Peer; SIGTERM produced `retry_required`; production controller retry lazily opened fresh PID 30304, resolved `auto` to the physical device, and returned authoritative Self/Peer finals; Self release retained Peer/PID; final Peer release removed the worker; shutdown closed all owners with no named tasks or package-root survivors. `C:/c177/gpu-production-composition-definitive.json`, SHA-256 `2332ee6e0d7979ce07f0a7c38e85ac9ccff02926e1c64f968c32262a375c8c95`. |
| Release headless gates | **PASS:** from unrelated Unicode/space CWD, exact workflow commands `--version`, `gui-startup-check`, local-Qwen, soxr, hf-xet, and Unicode/space config telemetry-disable all exited 0. `C:/c177/headless-gates-definitive.json`. |
| Process capture helper | **PASS:** release-only helper loaded packaged `_native.cp314-win_amd64.pyd` SHA-256 `98cdb81f…35755`; `native_process_specific=true`, no fallback, credentials, or network. |
| GUI | **PASS:** source, final package, and final installed main/desktop-overlay previews were bound through exact PID ancestry to their Flet HWND, captured directly, closed through `WM_CLOSE`, exited 0, and left no owned survivors. Final package: `C:/c177/gui-package-definitive.{json,png}` and `overlay-package-definitive.{json,png}`; installed: `gui-installed-definitive.{json,png}` and `overlay-installed-definitive.{json,png}`. Images visibly show the localized main dashboard and Korean overlay preview without missing glyphs. |
| Native/package inventory | **PASS:** CPython 3.14 and cp314/abi3 native extensions, ONNX Runtime, sherpa, ProcTap, hf-xet, soxr plus sibling DLL, standard PortAudio, OpenVR, CPU/Vulkan llama.cpp, overlay, and GPU worker are present. Unsupported ASIO DLL is absent. Recursive PyInstaller archive inspection found no `pytest`, `PyInstaller`, or `flet_cli` package (only NumPy's runtime `numpy._pytesttester` module name). |
| Isolated installer | **PASS:** alternate AppId `{94459ABA-E8C3-4F97-8C02-86CA158B00F6}`, isolated app-data/dir/group, current-user silent fresh install to a Unicode path, installed headless and GUI checks, deliberate `soxr.dll` corruption repaired by reinstall, telemetry-disable persisted through another reinstall, and silent uninstall removed install dir, isolated app data, and uninstall registry key. Production identity was untouched. |

Both packaged native binaries also return version `2.7.0`. The final post-run CIM scan found zero processes whose executable path was under the package root.

## Release workflow mapping

`.github/workflows/release.yml` builds with the pinned interpreter, runs the same packaged headless gates after PyInstaller and before installer construction, and builds/runs the release-only process-capture helper. Its `ProcessStartInfo.ArgumentList` invocation preserves Unicode and spaces. The final local runner exercised that command set against the exact candidate above.

## External gaps retained

- **Published upgrade:** deliberately not attempted. The production AppId is occupied and this workstation has no disposable clean VM/snapshot; forcing it would risk the real installation. Fresh isolated install/reinstall/uninstall evidence is not represented as published-upgrade evidence.
- **Physical surfaces:** no controlled physical microphone/loopback source, HMD/OpenVR runtime, or VRChat session was supplied.
- **Cloud surfaces:** no qualification credentials/endpoints were supplied.

Outcome C's independently reproducible software/native checkpoint is therefore ready for a fresh committed review. These external gaps remain explicit and prevent treating this checkpoint as blanket completion of issue #177 or final release approval.
