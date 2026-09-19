# Issue 177 Outcome A: Windows baseline at `be59afc`

Executed on 2026-09-19. This is an engineering baseline, not release approval and not an assertion that all Outcome A gates passed.

## Identity and isolation

- Source: `be59afc041122eaa0a1da41d807de454140023c6` (`git rev-parse HEAD`). `git status --short` was empty before execution.
- Product: `2.7.0`.
- OS: Windows 11 Education 10.0.22631, build 22631, x64.
- CPU: AMD Ryzen 7 9800X3D.
- GPU: AMD Radeon RX 7900 XTX, driver `32.0.31041.1004`. Virtual display adapters were also present.
- Dedicated execution environment: `.venv-win`, CPython `3.12.10`, created with:

  ```powershell
  $env:UV_PROJECT_ENVIRONMENT = ".venv-win"
  uv sync --locked --extra build --python "C:\Users\salee\AppData\Local\Programs\Python\Python312\python.exe"
  ```

- Isolated GUI profiles were under `%TEMP%`; no existing application configuration, credentials, conversation audio, or private model contents were read. No cloud or paid calls were made. No microphone or private audio was recorded.
- Build outputs are ignored local artifacts under `build/`, `dist/`, and `installer_output/`. No artifact was published or released.

## Initial execution status (historical)

The rows below preserve the first execution's results. The installer continuation below and [native workload reference](native-workload-reference.md) record later attempts, resolved prerequisites, and newly observed failures; the initial blocked rows are not the final evidence inventory.

| Area | Status | Executed result |
| --- | --- | --- |
| Clean source identity | passed | Exact requested commit and initially clean worktree confirmed. |
| Source dependency composition | passed | Python 3.12.10 with Flet/Flet Desktop/Flet CLI 0.86.1 and PyInstaller 6.21.0 resolved from `uv.lock`. |
| Source GUI | passed | Real Flet window created in an isolated profile; controls, Korean locale text, fonts, icons, navigation, debug-preview marker and window chrome were visually observed. Normal window close returned 0 and left no owned child. |
| Packaged GUI | passed | Real `dist/PuriPulyHeart/PuriPulyHeart.exe` launched from unrelated CWD `C:\Windows\Temp`; Flet window was visually observed; close returned 0 and left no owned child. |
| Desktop overlay process/start/close | passed | Preview launched as a separate Python plus Flet viewer process, window title `PuriPuly Overlay`, then normal close returned 0 with no owned process retained. |
| Desktop overlay visual-detail parity | not run | The window is transparent, so a screen-copy would include unrelated user desktop content. No durable capture was retained and no visual-detail claim is made. |
| Source maintenance command | passed | Isolated `installer-telemetry-preference disable` returned 0 and wrote canonical telemetry-off state; an invalid directory-as-config input returned the specified failure code 23. |
| Source native imports/resampling | passed with limit | NumPy, ONNX Runtime, sherpa-onnx, ProcTap, psutil, PyAudioWPatch, sounddevice, stock soxr, hf-xet, cryptography and CFFI imported. Stock soxr resampled 48 kHz synthetic sine data to 16 kHz. This is not packaged custom-soxr proof. |
| Bundled Silero VAD workload | passed with limit | The real bundled ONNX model ran five inferences on lawful synthetic sine input. This proves model/session/native execution, not speech accuracy or capture. |
| Bundled SmartTurn workload | passed with limit | The real bundled ONNX model ran three inferences on lawful synthetic sine input. This proves feature/model/native execution, not endpoint quality. |
| Custom soxr preparation | passed | Pinned source inputs were downloaded and verified; the system-linked `cp312-abi3-win_amd64` wheel and sibling `soxr.dll` were built. |
| PyInstaller onedir build | passed | Main onedir, release-only process-capture helper, Rust overlay, Rust GPU worker, Flet runtime and llama.cpp CPU/Vulkan runtimes built/staged. |
| Packaged smoke before install | passed | Version, GUI import/startup check, Local Qwen DLL bootstrap, custom-soxr load/resample/report, native ProcTap capture-start report, overlay startup contract, GPU worker version, llama.cpp CPU/Vulkan launch and packaged license/source-bundle verification passed. |
| Inno compile | passed | Inno Setup 6.6.1 compiled `PuriPulyHeart-Setup-2.7.0.exe`. The installer was not executed. |
| Isolated installer install/reinstall/uninstall | blocked | The script refused the already occupied alternate smoke AppId registry key `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}_is1`. It was left untouched. No installer/install/uninstall was run. |
| Real microphone/loopback/process audio | not run | Windows audio APIs and devices were enumerable, including WASAPI input/output, but no authorized controlled audio fixture/device session was supplied for recording/capture. The ProcTap result proves native process-specific capture startup only. |
| Real local ASR decode | blocked | Bundled VAD/SmartTurn models exist, but no complete local ASR model plus controlled speech fixture was available. One discovered Qwen GGUF cache candidate was a zero-byte file; this is not a claim about all model locations. |
| GPU ASR model workload | blocked | GPU and Vulkan toolchain were present, and the worker built, but no valid GGUF model/audio pair was available. |
| Cloud translation/ASR | not run | Credentials and paid-call authorization were intentionally not accessed. |
| VRChat/SteamVR/HMD sustained workload | not run | No authorized live VRChat/SteamVR/HMD session was established. |
| Upgrade, rollback and uninstall | blocked | Requires an owned isolated installer identity; the configured alternate identity was already occupied. |

Outcome A remains incomplete. The installer continuation resolves the occupied-AppId obstacle for an isolated install/reinstall/uninstall, and the native workload reference supplies public models, real CPU decode, download cancellation/restart, and isolated process-capture evidence. Packaged CPU/GPU evidence commands failed, published-version upgrade and physical-device/VR evidence remain unperformed, and the first installer probe exposed an orphaned viewer after host termination. These limits do not invalidate the recorded source/packaged reference for isolated B work.

Continuation 2026-09-19 (new AppId, production installer not executed): isolated install/maintenance/GUI/unrelated-CWD/Unicode path/reinstall/rollback/uninstall evidence is recorded in [Continuation: isolated installer lifecycle](#continuation-isolated-installer-lifecycle). Occupied `{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}` and production `{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}` were left untouched. This is still not release approval.

## Toolchain and dependency inventory

Installed tools were found by their installed locations, not only `PATH`:

- `uv 0.9.17`.
- Visual Studio Build Tools 2022 `17.14.36804.6`, complete and launchable according to `vswhere`; MSVC tools `14.44.35207`, compiler `19.44.35222`, MSBuild `17.14.23`.
- Windows SDK `10.0.26100.0` selected by CMake.
- Vulkan SDK `1.4.350.0` at `C:\VulkanSDK\1.4.350.0`.
- Rust `1.97.1` and Cargo `1.97.1`.
- CMake `4.4.0` from the dedicated environment.
- Inno Setup `6.6.1`.

Resolved baseline distributions of special interest:

```text
flet==0.86.1
flet-desktop==0.86.1
flet-cli==0.86.1
pyinstaller==6.21.0
numpy==2.5.1
onnxruntime==1.28.0
sherpa-onnx==1.13.4
sherpa-onnx-core==1.13.4
proc-tap==1.1.1
psutil==7.2.2
pyaudiowpatch==0.2.12.8
sounddevice==0.5.5
soxr==1.1.0
hf-xet==1.5.2
cryptography==46.0.7
cffi==2.0.0
```

Selected installed native identities from `.venv-win`:

| File | SHA-256 |
| --- | --- |
| `numpy/_core/_multiarray_umath.cp312-win_amd64.pyd` | `a764477b092cead9514d848d68a302966038c6735a0c746b940a2adc422589cb` |
| `onnxruntime/capi/onnxruntime.dll` | `3d6bd02dc137b9a1dfdac045a52b147ffb920bb06ab812b08fc8e14279859855` |
| `onnxruntime/capi/onnxruntime_providers_shared.dll` | `7ef797110a893f820db67a3f6cf38650110ccb3373567e3048fa8f3af1790040` |
| `sherpa_onnx/lib/_sherpa_onnx.cp312-win_amd64.pyd` | `f52edb4c489f9dcbc8d6e5a6fa0e9c38a00b7ea9fc216c5e39a424bc1a1545a1` |
| `proctap/_native.cp312-win_amd64.pyd` | `e3aa3153aac358ef22d0d3ebf19fefa3b1e68da88c5e1e20ca10b5dc2bf032dc` |
| `_portaudiowpatch.cp312-win_amd64.pyd` | `b4c83f7b3535914f49e0692c99533d4a0851c64ee018d424fd015606ece909ee` |
| `_sounddevice_data/portaudio-binaries/libportaudio64bit.dll` | `ec080194f01e4095c7fb43dbd7ed05af922c5b34295056a9ff56782741d65481` |
| `hf_xet/hf_xet.pyd` | `1d7086190c8f67568f670d4e1e1493097ce31058a634d9ce5c7f2922875e13e6` |
| `cryptography/hazmat/bindings/_rust.pyd` | `fe5c4cdb5512f792fb0f5b3ce0cd8c1484614851d56ff5d1e7215213eb9e0bca` |
| `_cffi_backend.cp312-win_amd64.pyd` | `53de34afc3939d81bf2498b77425c52e2c3f3164f8cfe2543d3acc8f24134488` |

The source sounddevice wheel contained both the standard and ASIO PortAudio DLLs. The assembled package passed the build rule that requires the standard DLL and excludes the ASIO DLL.

## Executed runtime measurements

### Source GUI startup and close

A throwaway probe launched three fresh isolated profiles from `%TEMP%` and an unrelated CWD, polled for the real owned visible window, sampled the full Python/Flet process tree at that point, sent normal `WM_CLOSE`, waited for the root process, and checked the original PIDs for survivors.

| Run | Visible window | Process tree RSS | Processes | Threads | Exit | Survivors |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2.2236 s | 328,818,688 B | 3 | 99 | 0 | none |
| 2 | 1.6622 s | 327,426,048 B | 3 | 100 | 0 | none |
| 3 | 1.4943 s | 309,506,048 B | 3 | 100 | 0 | none |

Median visible-window time was 1.6622 s; observed maximum was 2.2236 s. Median sampled RSS was 327,426,048 B. Run 1 was process/OS-cold relative to these three runs; runs 2-3 may benefit from OS file cache. All profiles were fresh. These are visible-window timings, not first-interactive, capture-ready, or first-translation timings. Three runs do not support a tail-percentile claim.

A separately supervised visual run showed a three-process tree (`python.exe` supervisor, application `python.exe`, `flet.exe`) with approximately 350 MiB combined RSS at observation. Normal close logged ordered application shutdown and removed all three processes.

### Packaged GUI

`dist/PuriPulyHeart/PuriPulyHeart.exe` launched from `C:\Windows\Temp` with an isolated `--config` path. At visual observation the owned tree was:

```text
PuriPulyHeart.exe  RSS 121,765,888 B  threads 28
└─ flet.exe        RSS 215,752,704 B  threads 64
```

The rendered surface matched the source run at the inspected state. The isolated profile lacked a provisioned local Qwen ASR model, and the UI visibly reported that absence; this is an honest prerequisite failure, not a GUI failure. Normal window close returned 0. Both observed PIDs were gone and no process named `flet.exe`, `PuriPulyHeart.exe`, `PuriPulyHeartOverlay.exe`, or `PuriPulyHeartGpuWorker.exe` remained.

### Desktop overlay preview

The actual preview command produced this observed tree:

```text
python.exe  RSS 48,050,176 B  threads 11
└─ flet.exe RSS 154,316,800 B threads 70
```

The owned window existed with title `PuriPuly Overlay` and bounds 1344 by 420. Normal close returned 0. Detailed transparency/placement rendering was not assessed because a capture would include unrelated user desktop content.

### Bundled native inference on synthetic input

Assets were read from the source package and hashes matched their pinned constants:

| Model | Size | SHA-256 |
| --- | ---: | --- |
| `data/vad/silero_vad.onnx` | 2,327,524 B | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| `data/models/smart-turn-v3.2-cpu.onnx` | 8,679,182 B | `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f` |

Silero VAD inference times on five 512-sample synthetic sine chunks were `0.654, 0.168, 0.102, 0.088, 0.087 ms`. SmartTurn inference times on three one-second synthetic sine inputs were `55.174, 43.989, 43.046 ms`; the returned score was stable at `0.8447034358978271`. These timings are only a repeatable native execution reference. Synthetic tones do not certify speech detection, endpoint accuracy, or audio continuity.

NumPy reported OpenBLAS `0.3.33.112.0`, `USE64BITINT`, `DYNAMIC_ARCH`, `NO_AFFINITY`, Haswell target and `MAX_THREADS=24`. No `OPENBLAS_NUM_THREADS` override was applied in these source probes.

## Build and artifact record

The long worktree path first caused MSBuild FileTracker `FTK1011` while CMake tested libsoxr. Retrying the unchanged preparation through a temporary `Q:` mapping to the same worktree succeeded. That mapping then exposed a strict path-identity limitation in the monolithic script: the first full build reached packaged soxr smoke but rejected `Q:\...` versus the canonical path as `expected_extension_path` mismatch. Running the unchanged full build again from the canonical worktree path passed every pre-installer build and packaged smoke gate.

The canonical run stopped before installer construction because the alternate installer-smoke AppId was already occupied. The production installer was then compiled directly with unchanged `installer.iss`; it was not executed.

Artifact identities:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `dist/PuriPulyHeart/PuriPulyHeart.exe` | 28,469,029 B | `bf6b0d878c6b41e6e57d764a2d8b9dac945206894f72d5e90047e2750c1b46a0` |
| `dist/PuriPulyHeart/PuriPulyHeartOverlay.exe` | 2,604,544 B | `4ea5b54efa894784f2d672149ff318bbfa7685aa797718421bbd17f2a38a6bf4` |
| `dist/PuriPulyHeart/PuriPulyHeartGpuWorker.exe` | 77,886,464 B | `831cadf7d91cdc3eb5dffea1618ccd782c9f5f5574722d62b4a1012fa2ffc486` |
| `installer_output/PuriPulyHeart-Setup-2.7.0.exe` | 180,486,494 B | `b1f6c781381f65c55ad6eaa57916aebb0d7e8be84e2f725cfc59889c4a851b44` |
| `build/flet/flet-windows.zip` | 40,104,260 B | `2cf0865b31bd0e394a24a6c2d270e084cf9dad9c711e0b5d0cf9fa9bfac31e14` |
| custom `soxr-1.1.0-cp312-abi3-win_amd64.whl` | 83,822 B | `d49069416db1e334b930575c87f75f80f4740e442342915c63ebdd038c0d5827` |

The onedir contained 1,137 files totaling 566,877,451 bytes. This is a directory-content sum, not NTFS allocated size and not installed size.

Additional proved identities/contracts:

- Rust overlay startup contract reported app `2.7.0`, contract `9`, execution revision `r2`.
- Packaged ProcTap smoke: status `passed`, version `1.1.1`, native process-specific capture started, no device fallback, credentials or network; `_native.cp312-win_amd64.pyd` hash `e3aa3153aac358ef22d0d3ebf19fefa3b1e68da88c5e1e20ca10b5dc2bf032dc`.
- Packaged custom soxr imported `soxr/soxr_ext.pyd` and loaded its exact sibling `soxr/soxr.dll`.
- llama.cpp release `b10423`, commit `a94d563ed801d1da1b8c2432946de07d0231bb3d`; both CPU and Vulkan packaged `llama-server.exe` paths launched during verification.
- llama CPU archive: 18,456,396 B, SHA-256 `b5a396f113a344578c0766331704bd541fd743c4c8e92858bea18440ee0ab19a`.
- llama Vulkan archive: 34,563,676 B, SHA-256 `510447fb021c80a264b2181c885b5f2ce9cc5b66c65d447cd1f9ce7ba81dc222`.
- soxr compliance source bundle: 483,041 B, SHA-256 `e5e14702d91e05a06d1bc894879f277cd1474ce9a1c2e46d24f90395153b8d5a`; it binds python-soxr 1.1.0, libsoxr 0.1.3 and zeroconf 0.150.0 source hashes.
- Packaged `hf-xet-runtime-check` executed from `C:\Windows\Temp` and returned 0.

## Executable and path contracts frozen for comparison

The current host is PyInstaller onedir `PuriPulyHeart.exe`. Default invocation and `run-gui` are GUI. The same executable also dispatches meaningful non-GUI commands: `--version`, `verify-desktop-overlay-repro`, `installer-telemetry-preference`, `local-qwen-runtime-check`, `soxr-runtime-check`, `hf-xet-runtime-check`, `gui-startup-check`, `local-cpu-real-model-check`, `local-asr-production-composition-evidence`, and `hf-xet-download-worker`. `run-desktop-overlay`, its preview, and its repro are windowed presentation/diagnostic commands.

Frozen callers that must be migrated together include:

- Inno runs `{app}\PuriPulyHeart.exe --config <settings> installer-telemetry-preference <enable|disable>` with CWD `{app}` and requires exit 0.
- Frozen hf-xet workers self-execute `PuriPulyHeart.exe`; source workers use the Python interpreter plus `-m puripuly_heart.main`.
- Frozen desktop overlay supervision executes `PuriPulyHeart.exe run-desktop-overlay --config <manifest>`; source supervision uses Python and the overlay module.
- Development process-isolation harnesses use `sys._base_executable` or `sys.executable`; this is not the installed host contract.

Current writable roots derive from `%LOCALAPPDATA%` (fallback `%APPDATA%`, then the user profile) plus `puripuly-heart`: settings, logs, models, HTTP extensions and other owned state. Secrets are keyring-backed or an encrypted file beside the configured settings path. Bundled prompts/data/fonts/VAD/SmartTurn, native executables, llama.cpp, Local Qwen DLLs, PortAudio and licenses are read-only payload. Existing prompt and llama development resolution still contain CWD fallbacks; native work must not accidentally preserve those as installed requirements.

Release workflows are not equivalent: `.github/workflows/release.yml` builds the release artifact but does not execute the app or install the installer, while `scripts/ci/build-release-artifacts.ps1` owns packaged/installed/reinstalled GUI, soxr and ProcTap smokes and isolated install/reinstall/uninstall. The latter is not invoked by the release workflow. This baseline executed its pre-install packaged gates, but the occupied installer-smoke identity prevented installed gates.

## Environment incident and recovery record

The initial root directory listing did not expose hidden directories. The first baseline sync used the repository default and `uv` explicitly reported `Creating virtual environment at: .venv`; early probes used it. After the direction to use a separate environment, `.venv-win` was created. An attempted removal of the apparently newly-created `.venv` removed files until Windows refused the locked `Scripts/python.exe`. Inspection showed an unidentified live Python process parented by `bun.exe`; it was not one of the supervised baseline GUI/overlay process trees. No unidentified process was terminated.

Recovery was limited to files removed by that attempt:

1. Recreated `.venv/pyvenv.cfg` with CPython 3.12.10 and uv 0.9.17 metadata matching the separately created environment.
2. Re-ran the exact locked sync into `.venv`.
3. Compared installed `*.dist-info` names read-only: `.venv` and `.venv-win` each contained 108 distributions with empty set differences.

No further `.venv` writes were made after that recovery. All subsequent probes/builds used `UV_PROJECT_ENVIRONMENT=.venv-win`. This incident did not modify product source, dependencies, build scripts, Git state, model cache or the unidentified process state.

## Commands for reproduction

Use a clean checkout at the recorded commit and a dedicated environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
uv sync --locked --extra build --python "C:\Path\To\Python312\python.exe"

# Source checks
.\.venv-win\Scripts\python.exe -m puripuly_heart.main --version
.\.venv-win\Scripts\python.exe -m puripuly_heart.main gui-startup-check
.\.venv-win\Scripts\python.exe -m puripuly_heart.main hf-xet-runtime-check

# Build inputs. A short checkout path is required on this machine for MSBuild FileTracker.
pwsh -NoProfile -File scripts/ci/prepare-soxr-release-inputs.ps1

# Full local release qualification; it performs an isolated installer lifecycle only
# if its alternate AppId namespace is unoccupied.
$env:PURIPULY_HEART_RELEASE_BUILD_ROOT = Join-Path $env:TEMP "PuriPulyHeart-A-ReleaseBuild"
pwsh -NoProfile -File scripts/ci/build-release-artifacts.ps1 `
  -AppVersion 2.7.0 -InnoSetupVersion 6.6.1

# Compile production installer without installing it.
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" installer.iss
```

## Source mutation and candidate-reuse barrier

Outcome A is frozen and remains blocked, not accepted. No further baseline build or product execution should use or mutate `.venv-win`, `build/`, `dist/`, `installer_output/`, or `%TEMP%\PuriPulyHeart-A-ReleaseBuild`. Do not remove the occupied installer-smoke registry key and do not delete either local environment.

The baseline reference is sufficiently recorded for a separate owner to begin isolated Outcome B compatibility implementation: exact source, environment, GUI/process behavior, command/path contracts, selected native-library identities, build artifacts, packaged smokes, measurements, and blockers are recorded above.

Outcome B must use a new `.venv-b` environment and separate build/output roots. At minimum, set:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-b"
$env:PURIPULY_HEART_RELEASE_BUILD_ROOT = Join-Path $env:TEMP "PuriPulyHeart-B-ReleaseBuild"
```

Because `build.spec`, `prepare-soxr-release-inputs.ps1`, `prepare-flet-runtime.ps1`, and `installer.iss` still use repository-relative `build/`, `dist/`, and `installer_output/`, merely changing `PURIPULY_HEART_RELEASE_BUILD_ROOT` does not isolate every output. The B owner must either use a separate worktree or first adapt/invoke those existing scripts with candidate-specific repository-relative output directories; B must not overwrite A's ignored artifacts in this worktree.

Useful unchanged baseline commands and required inputs for comparison are:

```powershell
# Dedicated A environment; reference only, now frozen.
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

# The custom soxr preparation required a short checkout path on this machine.
pwsh -NoProfile -File scripts/ci/prepare-soxr-release-inputs.ps1

# Release qualification inputs.
$env:PURIPULY_HEART_RELEASE_BUILD_ROOT = Join-Path $env:TEMP "PuriPulyHeart-A-ReleaseBuild"
pwsh -NoProfile -File scripts/ci/build-release-artifacts.ps1 `
  -AppVersion 2.7.0 -InnoSetupVersion 6.6.1

# Installer compilation only; this does not install.
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" installer.iss
```

The required local tool inputs were CPython 3.12.10, uv 0.9.17, VS Build Tools 2022 17.14.36804.6 with MSVC 14.44.35207 and Windows SDK 10.0.26100.0, Vulkan SDK 1.4.350.0, Rust/Cargo 1.97.1, CMake 4.4.0, and Inno Setup 6.6.1. The comparison authority is the artifact identity table above, not reuse of a mutable path. If an exact A artifact is unavailable, rebuild the unchanged commit in a separate short-path worktree and record the new hashes rather than treating a different hash as the same candidate.

## Continuation: isolated installer lifecycle

Executed later on 2026-09-19 without rebuilding A, without executing `installer_output/PuriPulyHeart-Setup-2.7.0.exe`, and without modifying `installer.iss`, product source, or shared envs. Throwaway scripts and logs live under `%TEMP%\PuriPulyHeart-A177-InstallerEvidence\` and `%TEMP%\PuriPulyHeart-A177-InstallerSmoke\20260919T084410Z-d3b00fb4\`.

Frozen A payload hashes were re-checked before compile and after compile; production setup remained `b1f6c781381f65c55ad6eaa57916aebb0d7e8be84e2f725cfc59889c4a851b44`.

| Area | Status | Executed result |
| --- | --- | --- |
| Occupied C2E4 smoke AppId | passed | `HKCU\...\Uninstall\{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}_is1` still `PuriPuly <3 2.6.1` at `...\PuriPulyHeart-Korean-Privacy-Validation\`. Left untouched. |
| Production AppId | passed | `HKCU\...\Uninstall\{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}_is1` still `PuriPuly <3 2.7.0` at `...\Programs\PuriPulyHeart\`. Left untouched. Production installer was not executed. |
| New AppId preflight | passed | `{D3B00FB4-4E73-4801-8E66-EBF7C53D11FE}` was unoccupied before first execute. |
| Distinct smoke compile | passed | Unchanged `installer.iss` + exact A `dist\PuriPulyHeart` and `build\overlay`, Inno 6.6.1, `/O` override, `/DSkipLocalSttProvisioning=1`. Output `C:\Users\salee\AppData\Local\Temp\PuriPulyHeart-A177-InstallerSmoke\20260919T084410Z-d3b00fb4\PuriPulyHeart-Setup-2.7.0.exe` size 180486536, SHA-256 `cb986e6b5fcdfc51a1bb983976316764124c0ec332625a82e366abdf4307d000`. |
| Isolated install | passed | `/CURRENTUSER /VERYSILENT` into Unicode `{localappdata}\Programs\PuriPulyHeart-A177-설치-20260919T084410Z-d3b00fb4`. Installed exe SHA-256 matched frozen A. Log marker `Local STT provisioning skipped for isolated installer smoke.` Fresh settings telemetry enabled. |
| Maintenance | passed | Installed `--version` from `C:\Windows\Temp` printed `2.7.0`. `installer-telemetry-preference disable` returned 0 and persisted canonical OFF. `soxr-runtime-check` returned 0. |
| GUI startup check / unrelated CWD | passed | Installed `gui-startup-check` from `C:\Windows\Temp` returned 0. |
| Installed GUI / Unicode CWD | passed with limit | Later owned install at `...\PuriPulyHeart-A177-설치2-...\` launched with `--config` from Unicode CWD `...\PuriPulyHeart-A177-cwd2-한글-...\`. Its Flet PID 26728 showed `PuriPuly <3`; `WM_CLOSE` returned 0 with no survivors in that launch tree. PrintWindow failed (`OverflowError` on hwnd), so no installed visual-parity claim is made. The first probe missed orphaned Flet PID 6900; see reconciliation below. |
| Reinstall | passed | Mutated owned `soxr\soxr.dll`, re-ran the smoke installer to the same Unicode dir; bundled soxr hash restored; telemetry opt-out preserved; STT skip marker present. |
| Rollback | passed | Silent retry into the first leftover Unicode dir hit RestartManager in-use Flet, aborted (exit 5), logged `Rolling back changes` and `Uninstallation process succeeded`; new AppId registry stayed absent. An earlier post-install telemetry exception did **not** undo copied files (silent CurStepChanged exception, exit 0); that path is not claimed as rollback. Production Flet was not closed. |
| Uninstall | passed | Owned `unins000.exe /VERYSILENT` on the second Unicode install removed that dir, the isolated AppData root `puripuly-heart-a177-20260919T084410Z-d3b00fb4`, and `{D3B00FB4-4E73-4801-8E66-EBF7C53D11FE}_is1`. C2E4 and production identities unchanged. |
| Published v2.7 upgrade | not run | `Downloads\PuriPulyHeart-Setup-2.7.0.exe` SHA-256 `10218a4e08b14a3fb5bde3106553fb446c9a06f59da6e14f5f1e64328c56bf15` uses occupied production AppId. Same-build reinstall is not an upgrade. |
| Process-capture helper in installer | not run | Not part of the frozen A payload compile; `/DProcessCaptureSmokeArtifactRoot` omitted. |

ISCC command (working directory = this worktree):

```text
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
  /DMyAppId={{D3B00FB4-4E73-4801-8E66-EBF7C53D11FE}
  /DMyAppDataDirName=puripuly-heart-a177-20260919T084410Z-d3b00fb4
  /DInstallerSmokeAppDataRoot=C:/Users/salee/AppData/Local/puripuly-heart-a177-20260919T084410Z-d3b00fb4
  /DMyAppGroupName=PuriPulyHeart-A177-20260919T084410Z-d3b00fb4
  /DMyAppDirName=PuriPulyHeart-A177-20260919T084410Z-d3b00fb4
  /DSkipLocalSttProvisioning=1
  /OC:\Users\salee\AppData\Local\Temp\PuriPulyHeart-A177-InstallerSmoke\20260919T084410Z-d3b00fb4
  installer.iss
```

Owned leftover not blanket-deleted: `C:\Users\salee\AppData\Local\Programs\PuriPulyHeart-A177-설치-20260919T084410Z-d3b00fb4\dbghelp.dll` remained after the first uninstall (access denied). The second Unicode install dir was fully removed. Product owner approval for release/deployment is not granted here.

### First-launch process reconciliation

The first installer GUI probe terminated its host and incorrectly reported an empty survivor set after matching only executable paths. Flet PID 6900 outlived parent PID 17060 and retained the first isolated install's `dbghelp.dll`. Its creation time and loaded-module path bound it to that probe. This is failed abrupt-host viewer containment/accounting evidence, not a passed shutdown case.

The separately owned production host PID 13440 and Flet PID 3424 predated this installer probe and were never A-owned. They were left untouched. The later isolated GUI PID 26728 had already exited normally.

For the verified orphan's window, `WM_CLOSE`, `SC_CLOSE`, and Alt+F4 had no effect. `WM_QUIT` to its verified thread ended PID 6900 within 0.5 seconds. Final inspection found no surviving A-owned host/viewer or A-install module mapping; only the pre-existing production tree remained. The leftover `dbghelp.dll` became readable and was not deleted. Manual recovery does not convert abrupt-host containment into a pass.
