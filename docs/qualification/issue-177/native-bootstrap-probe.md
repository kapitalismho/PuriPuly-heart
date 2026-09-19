# Issue 177 preparatory native bootstrap probe

The original isolated feasibility experiment below was executed on 2026-09-19 at source `285be5104886c2e1169de63b775726ea6ff4451c`. It is not a D pass, native-adoption decision, or release approval. The later product-host extension is recorded separately at the end. D acceptance remains gated on the comparator and required qualification; independent experiments do not require unrelated external checks to close first.

## Pinned inputs and toolchain

The probe used the public tuple from issue 177: Flet/template `1.0.0`, Flutter `3.44.8` (Dart `3.12.2`), Serious Python `4.7.0`, dart-bridge `1.9.0`, Python-build release `20260908`, and ordinary GIL CPython `3.14.7` Windows x64.

Downloaded bytes were checked before staging, including cache inputs:

| Input | SHA-256 |
| --- | --- |
| `flet-build-template.zip` | `d82972b6438e6b9c03f0741a1a307e5baa1448503886e97da62805a08c1facee` |
| `python-windows-for-dart-3.14.7.zip` | `a36cd03d9bcd68de30f3ff27de78e0f4eba31ac6ccfc0597ffa17ef663155e6e` |
| `dart_bridge-windows-x86_64.dll` | `70f111699c3a8f4ef96ebf997228fe68f9f6de84c48573a7eb6578b368fe5ec8` |
| debug bridge cache input | `c4165c69121668e5c0a9d312028474de65dbd4a45c9fd8a1d1fade21ba29efda` |
| Flet wheel | `7e61fffc7f2ee70de1d0729ee39177b838897f10258fb92d534e4a0ef8564f48` |
| official Flutter `3.44.8` archive | `095c108a08e0377d8a6501fed65aeb288908a070ed3f135e525dc6431c7686e4` |

Flutter's official Windows release manifest identifies framework revision `058e0af2c2b57e369d905a03ac9748b0ebf543c6`. The isolated SDK reported Flutter `3.44.8` and Dart `3.12.2`. The build used VS Build Tools 2022 `17.14.36804.6`, MSVC `14.44.35207`, and Windows SDK `10.0.26100.0`, located through `vswhere`; no toolchain was installed or configured globally.

Representative integrity commands:

```powershell
sha256sum flet-build-template.zip python-windows-for-dart-3.14.7.zip dart_bridge-windows-x86_64.dll
sha256sum flutter_windows_3.44.8-stable.zip
cmp C:/f177/gui-output/dart_bridge.dll dart_bridge-windows-x86_64.dll
cmp C:/f177/gui-output/python314.dll installed-layout/python314.dll
```

Both final comparisons were byte-identical. Serious Python's CMake downloader itself accepts an existing cache entry without checking a digest; pre-verification is therefore required, not optional.

## Standalone headless API proof: passed

A throwaway 75 KiB Windows-subsystem C++ host (`PuriPulyProbe.exe`, SHA-256 `d2d133220a8bd5e036b2b5d94353eee7a14dbf1d66979a9f7b0ee2551684d06f`) was compiled in ignored scratch space. It:

1. parses the original command line with `CommandLineToArgvW`;
2. resolves its installed directory from `GetModuleFileNameW`;
3. replaces ambient `PYTHONHOME`, `PYTHONPATH`, and `PATH` with the installed root, `app`, `DLLs`, and System32;
4. constrains DLL loading with `SetDefaultDllDirectories`, `AddDllDirectory`, and an absolute `LoadLibraryExW`;
5. preserves the upstream `serious_python_is_mp_invocation_w()` branch;
6. recognizes internal `--headless` before any COM or Flutter initialization and invokes `serious_python_main_w()` as `-m native_probe` with untouched remaining wide arguments.

The installed fixture contained one CPython/stdlib/site-packages/app distribution beside the host and bridge. It did not use system Python, duplicate a runtime, fake `sys.frozen`/`sys._MEIPASS`, fork dart-bridge, or initialize Flutter.

From unrelated CWD `C:\Windows\Temp`, installed and result paths containing spaces and Korean characters, and deliberately polluted `PYTHONHOME`, `PYTHONPATH`, and `PATH`:

- fixed module returned requested exit `0` and `23`; invocation without `--headless` returned `2`;
- Unicode result paths and the single quoted argument `quoted value 한글` survived exactly in `sys.argv`;
- `sys.prefix`, `PYTHONHOME`, `PYTHONPATH`, `sys.executable`, loaded `python314.dll`, and loaded bridge all resolved under the installed path;
- `ctypes.WinDLL()` returned the already-loaded bridge handle, and `GetModuleFileNameW` identified the exact pinned DLL;
- runtime was CPython `3.14.7`, `sys._is_gil_enabled()` was true, and `Py_GIL_DISABLED` was `0`;
- no window existed, `flutter_windows.dll` was absent, Flutter was absent before Python, and `CoGetApartmentType` returned `0x800401F0` (`CO_E_NOTINITIALIZED`);
- the multiprocessing-shaped `from multiprocessing.spawn ...` invocation returned `0` and wrote `multiprocessing interception preserved`;
- ambient `OPENBLAS_NUM_THREADS=7` remained `7`, `sys.exit.__module__` remained `sys`, and stdout was the normal `_io.TextIOWrapper`: the headless route did not inherit the Flet GUI bootstrap's exit/tee/BLAS policy.

Reproduction entry points are the ignored scratch `compile.cmd`, `stage.ps1`, and `run-probe.ps1`; the latter prints the full machine-readable report. The fixed module is deliberately probe-only, not a product command dispatcher.

## Minimal real native GUI: bounded pass

The verified Flet template and isolated Flutter SDK built a minimal Flet application with the same pinned runtime DLL bytes. A short physical build path was required: the repository worktree path caused Dart process command line error 206 and MSBuild FileTracker `FTK1011`; a drive mapping alone did not help because the CLI canonicalized the Python project path. Building the isolated fixture at `C:\f177` succeeded.

Executed build command shape:

```powershell
<absolute-scratch-python> -m flet.cli build windows C:\f177\fixture `
  --output C:\f177\gui-output --template <verified-template> `
  --python-version 3.14 --skip-flutter-doctor --no-rich-output --yes
```

The real output launched from `C:\Windows\Temp`, created a visible 720x360 Flutter window titled `Issue 177 Native GUI Probe`, rendered the three expected text controls, and loaded `flutter_windows.dll`, the pinned `dart_bridge.dll`, and the pinned `python314.dll` from its own output. Embedded Python reported CPython `3.14.7`, ordinary GIL, `sys.prefix` at the output, and `OPENBLAS_NUM_THREADS=1`. Normal `WM_CLOSE` returned `0`; the observed host PID did not survive. The output executable hash was `0f15a9a73deb4af4ce34139d3808418e0cfec2ad88e6904a68fbf9b66b0f5558`; generated `pubspec.lock` hash was `77beabcb449fd2f8946f7cbad0bbd48520ab454e6491495579cf7f95944526cc`.

The assembled x64 CRT files came from the build machine and all reported file version `14.51.36247.0`: `msvcp140.dll` `7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca`, `vcruntime140.dll` `d1f4225df2cd877dbf130d5668a021dce3f94118455ff5ec952061c30afc9ce7`, and `vcruntime140_1.dll` `a7146c08f89fe5b04541ab507cdb59ff7b44534d4ba3c668a426c6450a03434e`.

A material failure was also reproduced: the stock GUI host launched with polluted ambient `PYTHONHOME`/`PYTHONPATH` used the poisoned values during core initialization and exited `1` because it could not import `encodings`. The small headless bootstrap fixes this for its path; it has not yet been integrated into the GUI runner. Thus the GUI run proves the real Flutter/bridge/Python surface only under a clean parent environment, not the complete R3 installed boundary.

The Serious Python packaging step additionally downloaded `cpython-3.14.7+20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz` as a temporary compile/install tool; its observed SHA-256 was `ca3c33ca924dfcab3b74205a7a58a88b0255135c53f95497b26b5e60700fd66d`. The shipped `python3.dll` and `python314.dll` were nevertheless byte-identical to the verified `20260908` runtime ZIP. This extra build input needs an explicit checksum/provenance rule before candidate qualification.

## Bootstrap side effects and limits

Pinned `python.dart` sets `CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1` and `OPENBLAS_NUM_THREADS=1`, wraps part of ctypes lookup/loading, tees stdout/stderr to native logging plus an error-capture file, and replaces `sys.exit` with bridge-mediated exit. The GUI run confirmed the BLAS environment change. Its runner uses `TerminateProcess` after the message loop, so post-window Python finalizers/`atexit` are not graceful-shutdown evidence. The explicit headless route bypasses those GUI policies and retains ordinary process exit codes.

No product module, dependency file, lock, build, installer, model/cache, credential, audio, or cloud path was used. This probe did not exercise product commands, product native libraries, real ASR/audio/GPU/VR workloads, desktop overlay, ordered product shutdown, installation/upgrade/rollback/uninstall, or a shared product GUI/headless executable. Those remain blocked/not run for D until C is stable and the host adaptation is integrated and qualified.

## Tool-launch isolation incident

One scratch `flet.exe ... --help` invocation was mistakenly run from the repository worktree. Flet's tiny launcher selected the repository `.venv` rather than its own scratch interpreter, upgraded `flet`/`flet-cli` to `1.0.0`, and added `flet-platform-assets 1.0.0` plus Pillow `12.3.0`. A mistaken current-lock sync then temporarily removed the captured 27 build-extra packages and changed `flet-desktop`; the immediately following build-extra sync restored those same 27 packages at the exact versions printed by the removal output.

Recovery did not use the changed project lock for the Flet baseline: absolute `uv pip` commands restored `flet`, `flet-desktop`, and `flet-cli` to the captured pre-incident `0.86.1`, then removed only the two proven additions. Final read-only verification reported all three `0.86.1` and both additions absent. No further `.venv` writes occurred; `.venv-win`, `.venv-b`, repository `build`, and repository `dist` were not touched. Subsequent work used the absolute scratch interpreter with `python -m flet.cli`, an isolated CWD/project, and no ambient `UV_PROJECT_ENVIRONMENT`. This launcher-selection trap is itself relevant R3/R7 source-development supportability evidence.

## Combined product-host extension and measured shutdown failure

The resumed experiment used product source `e04cc2b6dce21bad20b2004668bcfc81aa87cc6f` and the same pinned upstream tuple. All host/template adaptations, copied product changes, staging and execution remained in ignored `.tmp/issue-177-native-product/` and `C:/d177`; no native production build path or cutover was introduced by this experiment.

The final experimental host, `C:/d177/설치 공간/PuriPulyHeart 제품/PuriPulyHeartNative.exe`, has SHA-256 `e7b20c2591ecf85a6ed6b71a5c521e9a4e263faf919fecccaab7979493043350`. Its reported inventory contains 21,593 files and 723,973,473 bytes. This prototype inventory is not a qualified runtime-only dependency closure or a matched C-versus-D resource result.

Executed observations:

- The same host's fixed headless product dispatcher printed version `2.7.0` and returned 0; actual telemetry disable/enable calls used explicit isolated encrypted-file configurations and returned 0, while an intentionally invalid configuration preserved failure exit 23. The install and unrelated working directory contained spaces and Korean characters; ambient Python paths were deliberately poisoned before host sanitation.
- The actual product GUI rendered in one host-owned PID/HWND, with Python, bridge and Flutter DLLs loaded from the installed root and installed-only Python paths. Normal and repeated `WM_CLOSE` returned 0; forced host loss returned 1. The observed owned-process survivor sets were empty without manual cleanup. This does not by itself prove ordered teardown.

The legacy `C:/c177/gui_probe_v2.py` and `.tmp/issue-177-native-product/run_gui_probe.py` survivor checks enumerated descendants only after the root exited. Their empty arrays therefore do not establish process containment: a descendant that detached or was reparented before that enumeration could have escaped observation. New D evidence must retain every owned process identity as `(PID, create_time, image)` while the root is alive, test those exact identities after root exit, and add a post-exit image/path sweep. This limitation does not invalidate the separate terminal-before-window-destroy shutdown markers.
- The separately invoked product desktop renderer authenticated using contract v9 and execution v1/r2, accepted the initial snapshot/bounds, emitted `desktop_first_visible` and `overlay_ready`, rendered bilingual revision-2 text, and exited 0 on bridge shutdown without survivors. This renderer probe is not a claim of complete main-application overlay integration or visual parity.
- The native renderer initially failed `window_reveal_lost`. A copied native-only `ShowWindow` adaptation made the probe visible. Its root cause and a narrow production implementation remain to be assessed; the probe workaround is not an accepted permanent repair.

The pinned native Python runtime contains no interpreter executable. A separately identified official PSF NuGet CPython 3.14.7 `python.exe` (106,208 bytes; SHA-256 `4942b86a6597e5aee0128daa00050ed79bc21f6e709a78eb19cbfeb0c2f39ac9`) was staged without a second DLL or standard library. It executed against the existing native `python314.dll` (`4c59183e519b728be1997b5bef38c2cda0dff5ff3d5ad665ff7f063bac587e79`), with installed-only prefixes/import paths. The host now supplies this actual interpreter through `PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE`; no unrestricted host `-m` route was added. This proves binary/layout feasibility, not every interpreter-shaped production caller.

### R5 failure: shutdown orchestration cancels itself

Absence of a Python-main return after native teardown was initially only an observation, not evidence that graceful cleanup had failed. Narrow copied-source markers subsequently established the actual failure:

1. `WM_CLOSE` reaches the real window-close callback and schedules `_close_after_window_request`.
2. The coordinator completes `freeze_ingress` and enters `stop_external_producers`.
3. `FletFoundationRuntime.cancel_page_tasks` closes its ordinary page-task set, which includes the task awaiting that same coordinator.
4. That await receives `CancelledError` while the coordinator remains `shutting_down`, `terminal=false`, `failure_count=0`.
5. The existing `finally` calls `page.window.destroy()` before later shutdown phases run.

The failure reproduced for normal and repeated close. Exit 0 and no surviving process therefore masked incomplete critical cleanup. The shared source repair separates shutdown orchestration from ordinary cancel-on-close UI jobs; it does not attempt Python finalizers after native `TerminateProcess`.

The immutable pre-repair evidence is `.tmp/issue-177-native-product/evidence/pre-repair-r5-self-cancellation/`: its manifest hashes read-only copied source, exact diffs, normal/repeated reports and `gui-shutdown-markers.jsonl`. Other retained reports include `headless-evidence.json`, `overlay-evidence.json`, `interpreter-evidence.json`, owned-window screenshots, distribution inventory and pinned-input hashes; exact C++/Dart/bootstrap inputs are in `final-inputs/`. Copied GUI code also suppresses after-launch tasks for the isolated probe and contains fsync instrumentation. Those probe-only changes are not product semantics to adopt.

Post-repair native validation is retained separately in `.tmp/issue-177-native-product/evidence/post-repair-r5-native-validation/`. Its hashed manifest binds exact tracked and probe-adapted source. Normal and repeated close now observe coordinator `completed`, `terminal=true`, zero failures and outstanding tasks, and all six phases through log flush before `window.destroy()`. The repeated close is deduplicated while shutdown is active. Both hosts exit 0 without survivors or manual cleanup; headless version, telemetry persistence and failure exit 23 remain passing. These idle/repeated-close observations close the measured self-cancellation reproduction, not the remaining active-work/native qualification matrix. The prototype inventory above describes the pre-repair payload; the post-repair bundle separately identifies its changed Python sources.

Full native workload coverage, runtime-only payload closure, permanent build/entry adaptation, production caller migration, remaining native shutdown scenarios, Inno qualification and matched resource/maintainability evidence remain separate obligations. Existing physical/cloud/VR and published-AppId upgrade limitations remain unchanged.
