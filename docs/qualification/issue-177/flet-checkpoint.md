# Issue 177 Outcome B: Flet 1.0 checkpoint

Date: 2026-09-19  
Source baseline: `be59afc041122eaa0a1da41d807de454140023c6`  
Scope: Flet-only checkpoint on Windows; Python remains 3.12 and the existing PyInstaller/Inno, repository layout, installer, native-host, and architecture contracts remain unchanged.

## Candidate composition

The locked candidate uses CPython `3.12.10`, `flet==1.0.0`, `flet-desktop==1.0.0`, `flet-cli==1.0.0`, `flet-platform-assets==1.0.0`, and PyInstaller `6.21.0`. Dependency work was limited to the three requested Flet pins and their necessary lock closure (`flet-platform-assets==1.0.0`, `pillow==12.3.0`). Python bounds and markers were not changed.

The pinned Flet Windows archive is 40,287,115 bytes with SHA-256 `758f21506fbb9ad180bd93c7460a2ca55630401c6026a9bc9e2273444014491d`. The same version and hash are enforced by `build.spec`, `scripts/ci/prepare-flet-runtime.ps1`, and the release dependency guards. Contract-file hashes at the candidate boundary are:

| File | SHA-256 |
| --- | --- |
| `pyproject.toml` | `296ee2789d0f05c1ccbc6bd5512b7efd74db76d6ac876d47340aff362ce7ba43` |
| `uv.lock` | `4faaa84f128b4c48a1c82987b83923bf90c713f228ea552fb4ec4a12184af43e` |
| `build.spec` | `a06d70062e5b7590b2fb8fe9342dc9443764b5c895c382881460058abb6bb877` |
| `scripts/ci/prepare-flet-runtime.ps1` | `0ecb3d0c642b5c76dc91a656add117d0845e30d345f1f1a8f2f100466dd6fdcc` |

Direct inspection under `.venv-b` confirmed the two private viewer hooks intentionally retained by the frozen package integration are still callable in Flet 1.0.0:

```text
flet_desktop.__locate_and_unpack_flet_view(page_url, assets_dir, hidden)
flet_desktop.open_flet_view_async(page_url, assets_dir, hidden)
```

## Compatibility changes

Imperative APIs and their test doubles were migrated to the Flet 1.0 contract. Main-window readiness now awaits `wait_until_ready_to_show()` and `center()` directly. Dialog dismissal uses `Page.pop_dialog()` directly; obsolete Flet-only missing-method and synchronous readiness fallbacks were removed while existing product-level error recovery was retained. Overlay preview buttons use the Flet 1.0 `Button` control.

Outlined text fields now use `InputBorder` state maps. Normal, focused, error, and disabled states are explicit `OutlineInputBorder` values; focused uses the primary color and width 2, error uses the error color, and normal/disabled use divider styling. Borderless display/port fields use explicit four-state `NoInputBorder` maps. The shared helper is used by API-key, custom-vocabulary, prompt-editor, Discord-referral, and QQ settings fields.

## Source and visual qualification

All real-window checks ran from an unrelated working directory with isolated configuration roots. Source main UI, source overlay preview, packaged onedir main UI, and packaged overlay preview each opened a real Flet desktop window. Captures showed the application chrome, fonts, icons, Korean dashboard content, debug marker, overlay controls, and sample caption. Each owned window was closed through `WM_CLOSE`; its supervised root exited 0. No B-owned process survived. The concurrently running production host PID 13440 / Flet PID 3424 was never A-owned and was not touched or used as B evidence. A first-installer A orphan, Flet PID 6900, was separately discovered and closed by the installer owner.

A real Flet 1.0 probe rendered the product border helper in normal, focused, error, and disabled states. The capture showed the normal outline, width-emphasized focused outline, error outline plus error message, and muted disabled outline. Source/package captures and throwaway capture/probe scripts remain outside the repository under `%TEMP%\PuriPulyHeart-B-Flet\`:

- `source-main.png`
- `overlay-preview.png`
- `border-states.png`
- `packaged-main.png`
- `packaged-overlay.png`

The earlier `installed-main.png` was title-selected while another `PuriPuly <3` window existed; it is not used to attribute installed behavior. The replacement installed capture and machine-readable ownership record are `%TEMP%\PuriPulyHeart-B177-VisualRepair\installed-main-path-bound.png` and `installed-main-path-bound.json`.

## Tests and build

Focused compatibility and affected UI/application tests passed in successive runs: `161 passed, 2 skipped`; the broader relevant UI/application run passed `259`; and the final focused fallback run passed `137`. The complete UI suite passed `1169 passed, 2 skipped`. The final repository-wide `.venv-b\Scripts\python.exe -m pytest` run passed `6028 passed, 37 skipped` in 198.12 seconds. Ruff passed for every changed Python file. Pytest emitted one pre-existing warning about an un-awaited async test function; no test failed.

A byte-for-byte source snapshot was made at `C:\b177` immediately before the release build; a `robocopy /MIR /L` comparison reported 1,318 files, 0 copies, 0 mismatches, 0 failures, and 0 extras. The snapshot used its own `C:\b177\.venv-b` and `%TEMP%\PuriPulyHeart-B-ReleaseBuild`; no A environment, build tree, artifact, installer identity, or process was mutated.

The unchanged `scripts/ci/build-release-artifacts.ps1 -AppVersion 2.7.0 -InnoSetupVersion 6.6.1` built both Rust executables and the PyInstaller onedir, then passed staged overlay, packaged executable, packaged overlay, packaged GPU worker, Flet archive, llama CPU/Vulkan, soxr provenance/runtime, and other pre-installer package gates. It stopped only at its safety guard because the script's hardcoded alternate AppId `{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}` was already occupied. That identity was left untouched. Unchanged `installer.iss` then compiled successfully with Inno Setup 6.6.1.

The onedir contains 1,146 files totaling 582,100,047 bytes. Artifact identities:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `PuriPulyHeart.exe` | 29,947,281 B | `ff39bb024ee1604b8bab6bc8f973b27298789a48e2a8b8e644c4b2bc59e56342` |
| `PuriPulyHeartOverlay.exe` | 2,604,544 B | `5122120d65e91f0d0e067a5966d1a2c6bb3aaf48bb8a4a0572e31cca7ee39488` |
| `PuriPulyHeartGpuWorker.exe` | 77,886,464 B | `0e958d683d3ba02a6a624ce2ea2e73dd56e99ec948874af22f8e238fc52fbca8` |
| production `PuriPulyHeart-Setup-2.7.0.exe` | 185,658,747 B | `96fe07320b3744c9d028d4a907e83243bcd2a778a6e4e82304026d3d7f4125fd` |
| `flet-windows.zip` | 40,287,115 B | `758f21506fbb9ad180bd93c7460a2ca55630401c6026a9bc9e2273444014491d` |

## Isolated installer lifecycle

The production installer and occupied production/smoke identities were never executed or changed. For installed qualification, unchanged `installer.iss` was compiled again with a fresh AppId `{C633B26F-721B-489C-9230-9CBBB5D82643}`, distinct app-data/group/directory names, a distinct `/O` output, and `/DSkipLocalSttProvisioning=1`. The resulting isolated installer is 185,660,909 bytes with SHA-256 `2dc537da6c0074e3e35cb436be3525ab139d155922d75a491824e3a624032a16`.

A current-user very-silent install into Unicode path `...\Programs\PuriPulyHeart-B177-설치-c633b26f` passed and logged the intentional local-model provisioning skip. From `C:\Windows\Temp`, installed `--version` returned `2.7.0`; `gui-startup-check`, `hf-xet-runtime-check`, `local-qwen-runtime-check`, and `soxr-runtime-check` returned 0, with runtime paths resolving inside the Unicode install. Those command results remain applicable. Its title-only GUI capture is superseded by the path-bound visual repair below.

Installed telemetry was set OFF through the product command. The owned installed `soxr.dll` was then truncated, the isolated installer was rerun to the same directory, and reinstall restored its exact SHA-256 `f5b759d52304f6d33da18394ed3741288abf0e2ee1b30adad2a8732775f82552` while preserving telemetry OFF. Silent uninstall removed the install directory, isolated app-data root, and fresh AppId registry key; all three tested absent afterward.

### Path-bound installed visual repair

The retained B payload was not rebuilt: `C:\b177\dist\PuriPulyHeart\PuriPulyHeart.exe` still had SHA-256 `ff39bb024ee1604b8bab6bc8f973b27298789a48e2a8b8e644c4b2bc59e56342`. Unchanged `installer.iss` was compiled against that exact payload with fresh AppId `{E60E72D2-157D-4496-B88E-E4E34AC1F131}`, distinct profile/directory/output names, and the same intentional model-download skip. The repair installer is 185,660,886 bytes with SHA-256 `c2d7ccf1ab376c95ae1898703dacf0c7f1dfa7281a97c26f4b5f0af4ade7a509`.

The fresh Unicode install root was `...\Programs\PuriPulyHeart-B177-검증-e60e72d2`. The captured HWND `2034784` belonged exclusively to Flet PID `25108`, image `C:\Users\salee\.flet\client\flet-desktop-full-1.0.0-758f21506fbb\flet\flet.exe`. Its recorded command line names the fresh install's `puripuly_heart\data` directory, and its direct parent was installed host PID `27548`, whose exact image was `...\PuriPulyHeart-B177-검증-e60e72d2\PuriPulyHeart.exe` and whose SHA-256 matched the retained candidate. The verified parent chain is `25108 -> 27548`; no title-only process selection was used.

Before capture, that exact HWND was moved to `(80,100)` to distinguish the run. The inspected 1136x850 PNG is 32,443 bytes with SHA-256 `8f1624c5afca3ce589cdcac810fd56717161a83c3311c791527f000242cc7952`. It visibly shows the full coral four-tile dashboard, material icons and fonts, `DBG`, `Zzz...`, the Korean message prompt, and both Korean-to-English and English-to-Korean rows without clipping or rendering corruption.

After inspection, `WM_CLOSE` was posted to that exact HWND. Flet PID 25108 and installed host PID 27548 both exited normally; the app and capture supervisors returned 0. An earlier repair-probe tree owned by the same fresh install (host PID 6824 / Flet PID 6864) was also closed by its exact verified HWND, and both processes exited. A path/profile-scoped survivor query found none; it showed only the unrelated production Flet PID 3424. Silent uninstall then removed only the fresh repair install directory, app-data root, and E60E72D2 registry identity; the uninstall log reported success and `Removed all? Yes`. All three owned paths/identities tested absent afterward, while the production registry identity remained present.

This isolated lifecycle intentionally skipped local model downloads. It does not claim a published-version upgrade, real microphone/audio continuity, cloud credentials, model accuracy, or VR presentation. Those are outside the Flet-only B checkpoint. Real Windows UI, build, packaged, installed, reinstall, and uninstall behavior are qualified here on the baseline machine (Windows 11 Education x64, Ryzen 7 9800X3D, Radeon RX 7900 XTX).

## Candidate barrier

This document records the implemented B candidate; independent review and Director acceptance are separate gates. Later source, dependency, build-input, package, or installer changes require reassessment of affected evidence rather than invalidating unchanged checks automatically. `.venv-b`, `C:\b177`, candidate outputs, and evidence directories are retained rather than deleted. Outcome A documentation and artifacts remain separate.
