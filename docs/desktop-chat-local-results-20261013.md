# Floating chat: local development results, 2026-10-13

## Baseline and protection

- Fork branch: `fix/desktop-chat-toggle-source-handoff-20261013`.
- Fetched starting SHA: `4bf0d93fc4339d4b4e44220c924757d5f26d4fb7`; re-fetched unchanged before preparing commits.
- Independent worktree: `D:/CODE/PuriPuly-heart-2.6.1/desktop-chat-handoff-20261013`.
- The original investigation worktree's uncommitted changes remain untouched. QQ commit `677a9ee837b7e19abfa758998e86f8bce6a28b3a` remains an ancestor. No changes to the QQ branch/PR, main, production Broker, formal installation, or production secrets.
- The old tool `D:/CODE/PuriPuly-heart-2.6.1/PuriPuly-heart-2.6.1/scripts/qq_issue_api_key.py` is unchanged; SHA256 `D439BEC49A6E4A99E010B31BD26DCD7D6CD3D666484CB47A1DCB578AC5D38FFE`. Neither it nor its configuration is included.

## Implementation and changed files

| Files under `src/puripuly_heart` | Change |
| --- | --- |
| `app/services/quick_chat.py` (new) | Bounded SELF/PEER records, role-qualified canonical IDs, accepted-event sequence, late updates and safe failure projection. |
| `app/services/quick_translate.py` | Explicit show/hide, visibility callback, bounded/coalesced non-blocking child delivery, history replay on child replacement, delivery cleanup and bounded shutdown writes. |
| `app/services/manual_typing.py` | Preserve canonical utterance ID when a submitted translation fails, so its original row is updated rather than duplicated. |
| `app/adapters/ui_runtime.py`, `app/services/ui_application.py` | Existing shared manual path for translation-off original publication; effective runtime language/translation snapshots; canonical window preferences. No second provider or independent language settings. |
| `app/services/application_control.py` | Mixed provider + language/OSC updates previously reported success while silently discarding non-provider changes. Use full settings application for mixed changes; provider-only requests retain their existing route. Reproduced FAIL before fix and PASS after fix. |
| `ui/app.py`, `ui/views/dashboard.py`, `ui/components/display_card.py` | Real switch next to the composer, child-hide feedback, safe context synchronization, readable localized label. |
| `ui/event_dispatch.py`, `ui/presentation_adapter.py` | Fan out existing accepted UI events after delivery-authority validation. A child error cannot interrupt normal dashboard/output processing. Refresh child context through existing presentation hooks. |
| `ui/quick_translate_window.py` | SELF/PEER cards with distinct light backgrounds and role labels, display modes, preserved draft/history, copy, pin, original/failure fallback, 40–100% opacity with opaque default, latest-message scrolling after send/completion. Scrollback reading is not forcibly displaced by unrelated incoming updates. |
| `config/settings_vnext/schema.py` | Portrait defaults 420×680 and additive display-mode/opacity preferences. Existing custom geometry remains respected. |
| `data/i18n/{en,zh-CN,ja,ko,ru}.json` | Floating-chat labels, modes, opacity, roles, effective translation state and more accurate readiness text. |
| `core/local_translation/{assets,provisioning,public_download}.py` | Reuse an already complete partial without requesting an invalid Range; retain hash checks. Cooperative checksum cancellation and join owned inspection/verification/promotion workers before cleanup. Existing-model checksum inspection no longer blocks the event loop. |

Updated tests: `tests/app/test_quick_translate.py`, `tests/app/test_application_control_contract.py`, `tests/core/local_translation/test_model_sources.py`; new `tests/app/test_quick_chat.py`. System boundary update: `docs/architecture.md`.

## Tests and exact commands

Interpreter: existing locked development environment, Python 3.14.3 / Flet 1.0.0. Set `PYTHONPATH` to this worktree's `src`; tests also select it through their existing `conftest.py`. No dependency upgrades.

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
& ../desktop-cn-download-audio-test/.venv/Scripts/python.exe -m pytest tests/app/test_quick_chat.py tests/app/test_quick_translate.py tests/test_desktop_quick_toggle_contract.py tests/ui/test_app_branches.py tests/ui/test_event_bridge.py tests/ui/test_presentation_adapter.py tests/app/test_manual_typing.py tests/app/test_ui_application_boundary.py tests/app/test_application_control_contract.py tests/core/output/test_router.py tests/core/runtime/test_output_runtime.py tests/core/test_self_translation_channel_owner.py tests/core/test_peer_translation_channel_owner.py tests/core/test_translation_turn_owner.py tests/core/local_translation/test_model_sources.py tests/core/local_translation/test_gemma_assets.py tests/core/local_translation/test_gemma_provisioning.py tests/config/test_settings_vnext_migration_serialization.py tests/config/test_settings_vnext_schema.py tests/ui/test_qq_auth_log_lifecycle.py tests/architecture -o addopts='' -q --tb=short --basetemp=D:/CODE/PuriPuly-heart-2.6.1/chat-final-tests-02
```

PASS: **861 tests**, 129.12 seconds; one existing `asyncio.iscoroutinefunction` deprecation warning. The final invocation used the same command with `--basetemp=D:/CODE/PuriPuly-heart-2.6.1/chat-final-tests-04`. This is the relevant regression selection, not the full repository suite. Mock/small-asset tests are not real audio/model-download acceptance. An intermediate new scroll test had a misplaced assertion (`KeyError: id`); correcting its setup and rerunning the full selection produced the final clean result.

```powershell
& ../desktop-cn-download-audio-test/.venv/Scripts/python.exe -m compileall -q src/puripuly_heart/app src/puripuly_heart/ui src/puripuly_heart/core/local_translation
$files = @(git diff --name-only -- '*.py') + @('src/puripuly_heart/app/services/quick_chat.py','tests/app/test_quick_chat.py')
& ../investigation-v282/.venv/Scripts/python.exe -m ruff check $files
foreach ($file in $files) { & ../investigation-v282/.venv/Scripts/python.exe -m black --check --fast $file }
git diff --check
```

PASS: compilation, whole-GUI imports, changed-file Ruff/Black and whitespace checks. No original user configuration was loaded for GUI tests.

## Windows GUI and real API evidence

Test profile: `chat-smoke-20261013`; data root `D:/CODE/PuriPuly-heart-2.6.1/chat-gui-test-data`; separate test SecretStore namespace. The small external GUI harness `D:/CODE/PuriPuly-heart-2.6.1/chat_gui_smoke_20261013.py` is not shipped or committed. It injects clearly labeled mock history through the existing presentation callback; ordinary user submissions still use the real application pipeline.

PASS observed in real Windows Flet windows:

- Composer switch opens an independent portrait chat window; close command hides it and synchronizes the main switch; reopening retains chat rows and a multi-line draft.
- Pin toggles and saves; main minimization does not destroy the chat window.
- Main-page language swap propagated live to the child. Separately applied CLI languages propagated to both effective runtime and child.
- Microsoft Chinese Pinyin composition: first Enter confirms composition without submission; subsequent Enter sends. Shift+Enter inserts a newline without sending. This does not prove all Chinese/Japanese IMEs.
- Translation-off original submission completed without an LLM. Canonical transcript and request result produced one row, not two.
- Opacity adjusted through the actual slider, saved **66%**, and restored after restarting the isolated application. The user also explicitly reported manual transparency testing as working.
- The final GUI displayed the last injected mock history item rather than the beginning. After scrolling upward to old records, sending the local original-only sample `q` explicitly returned the viewport to the newest row, which was fully visible. Flet's suspended automatic-follow state required a bounded native `scroll_to(offset=-1)` call on a non-virtualized, bounded history column; toggling `auto_scroll` alone failed the initial real-window probe. The final source and GUI probe passed. Scroll failures have the safe `scroll_failed` log/UI code.

The user subsequently explicitly authorized one short real DeepSeek translation and provided a test Key. It was entered using the existing CLI hidden credential prompt into the isolated protected SecretStore, not process arguments, plaintext files, source, or Git. Selected route: `deepseek_v4_flash_41` / `official_byok`; actual readiness query returned `translation_enabled=true` and `translation_runtime_ready=true`. OSC was separately confirmed `off` before sending. No Managed Key issuance was performed.

**PASS (real API, not Mock):** typed `你好` in the actual child; provider returned `Hi`; the child showed `你好 → Hi`, completion state and cleared input. The existing accepted-conversation log recorded the original and English translation at **13:10:27 local time**, with no Key. No additional cloud calls were needed for the final scroll test.

Log persistence/privacy test: simulated `RuntimeError` containing private-text and fake-Key sentinels. Closing/hiding the quick window and detaching/reopening Logs retained the ordinary ERROR record; the rotating file retained `stage=translation code=translation_failed exception_type=RuntimeError`. Sentinel text/Key were absent. Additional literal-Key-pattern scans of code/tests/docs and isolated logs/configuration returned no matches. Conversations remain on the project's existing accepted-conversation path, not duplicated in technical diagnostics.

## Model-source review and remaining work

PASS (Mock/small assets): bounded timeout retries, external redirect rejection before contacting it, correct Range/restart behavior, exact-complete partial reuse, wrong/missing imports, SHA256/size validation, disk-full handling, cancellation before downloading and during verification, staging cleanup after worker completion, existing-install protection and no duplicate download after valid import.

The mirror remains the scoped `https://hf-mirror.com` candidate. It does not change QQ/OpenRouter/ASR routes, global environment/proxy settings, TLS verification or the pinned Gemma contract. Prior feature evidence reported offshore redirects; this stage did not repeat that network probe or establish any new mirror as usable. NOT_TESTED here: current mirror reachability, mainland direct/TUN-off full transfer, real 4.275 GB model hashes/load, alternate byte-identical ModelScope copies.

DSH should inspect lease-acquisition/atomic-promotion cancellation boundaries, slow/stalled child recovery and cross-process preference races; check intended cancellation behavior once atomic promotion has already committed. Cross-session download resume is not established by the in-task partial-file test. No permanent source fallback or silent overseas rerouting was added.

## Handoff limits

- NOT_TESTED: Japanese IME, other Chinese IMEs, game/borderless-fullscreen focus, exclusive fullscreen, multi-monitor removal/per-monitor DPI, actual microphone/STT/VRChat traffic, full model loading, and official native distribution packaging on this follow-up.
- CodeRabbit NOT_TESTED: CLI missing; download endpoint connection failed (`curl: (7)`, port 443 through the current local proxy). No independent review approval is claimed.
- Existing SELF/PEER/output/translation-turn and QQ-log tests passed; real production authentication was not attempted.
- No DSH task was launched, no upstream PR was created, no branch was merged and no Release was published. This is development plus minimum targeted validation, not complete acceptance.
