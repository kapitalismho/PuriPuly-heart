# Desktop chat and model-source handoff - 2026-10-13

## Status and scope

Repository: RICHARDwuxiaofei/PuriPuly-heart

Source branch: feat/desktop-cn-download-audio-test

Base commit: c040aa6d5f76c576780cd64058233b312ba873b4

Follow-up branch: fix/desktop-chat-toggle-source-handoff-20261013

Status: PARTIAL IMPLEMENTATION. This is not a completed two-way chat window, a completed independent code review, or a verified domestic model-download release. Do not merge, publish, install over production, or claim physical acceptance from these changes.

The user is in class and cannot perform manual tests now. Finish the remaining implementation and run available tests without asking them to operate their computer. Separately list genuinely manual checks for later.

## Changes in this follow-up

- QuickTranslateOwner.open now toggles visibility, instead of always showing the window. Existing callers of open therefore change behavior intentionally. Audit callers and existing tests when integrating.
- One child is reused while alive. Child-initiated hiding reports visibility back to the owner. Revision numbers reject acknowledgements from an older show/hide request. Admitted translation work remains active while hidden.
- The initial window is 420 by 680. The exact previous default 480 by 520 migrates to the portrait default. Other saved user dimensions remain unchanged. Geometry calculations are separated into ui/quick_window_layout.py.
- A failed submission-pipe write updates the displayed pending row to a connection error and retains input.
- The architecture still uses the existing owned presentation child and private pipes. No new translation backend, output destination, model endpoint, credential store, or runtime owner was added.

## What is NOT implemented here

- The main-page toggle's requested placement and selected-state synchronization.
- Received PEER messages and canonical SELF messages in one chronological chat stream.
- Original plus translation / translation only / original only display options.
- Distinct SELF and PEER bubble presentation.
- Live language/status synchronization from the main page.
- Changing the quick submission API to guarantee original-only sending while translation is disabled. The existing submit_quick_text binding remains unchanged.
- Any additional model source, full model download, model loading, or Windows GUI acceptance.

Do not report those items as completed merely because the popup opens or the new isolated tests pass.

## Required final user experience

Place a real on/off control beside the main dashboard text composer. On shows one independent, opaque, portrait floating window; off hides it. Closing that child with X must turn the control off without closing the main application. Reopening preserves the session draft and bounded history. Keep the main window minimizable independently, and stop owned children on application exit. Do not inject into games or bypass anti-cheat.

The upper area is a Discord-style chronological SELF/PEER conversation; the composer is at the bottom. Use role labels as well as distinct colors. Subscribe through an appropriate existing presentation/output boundary rather than re-transcribing audio, re-translating completed turns, or polling rendered labels. Key records by canonical utterance/request identity, not text: duplicate delivery must not duplicate rows, while intentionally sending the same text twice must remain possible. A late translation updates its original row without moving it to the end. Bound memory/history and child delivery so a hidden or stalled window cannot block normal output.

For PEER display, support original plus translation, translation only, and original only. Display options do not independently enable a model. With peer translation disabled, show original text even if the window's stored preference is translation only. With translation enabled but pending or failed, make that state explicit and retain an intelligible original fallback. Turning off peer capture entirely is different: it may produce no new transcripts.

For outgoing text, reuse the main application's typed manual-submission operation and its admission/output policy. Translation enabled means translate using the main source/target selections; translation disabled means send original without requiring an LLM. Do not use a separate provider, force OSC on, or route PEER text into the VRChat chatbox. Preserve existing multi-target settings and provider-switch cancellation/late-result rules. A failed request must not silently send original. Keep input on failure and clear only the draft actually submitted on success.

Languages and effective translation state come from the main application's canonical state, including changes made outside the dashboard. Do not create a second language setting. Prefer one-way presentation synchronization; any editing in the child must use the same existing application intent. Include provider-not-ready, disconnected child, composing IME, and app-shutdown states.

## Model-source inventory at the base commit

| Model artifact | Engine/purpose | Sources declared at this commit |
| --- | --- | --- |
| Gemma 4 E4B IT QAT UD-Q4_K_XL plus MTP draft | Local translation / GGUF | Official Hugging Face, scoped hf-mirror selection, verified offline import. Existing branch evidence reports the mirror redirecting offshore and being rejected safely. |
| Qwen3-ASR 0.6B INT8 | CPU speech recognition / sherpa-onnx | Hugging Face AND ModelScope already exist in the manifest. |
| Parakeet TDT 0.6B v3 INT8 | CPU speech recognition / sherpa-onnx | Hugging Face only in its manifest. |
| Parakeet TDT-CTC 0.6B Japanese INT8 | CPU Japanese speech recognition / sherpa-onnx | Hugging Face only in its manifest. |
| Qwen3-ASR 1.7B Q6_K | GPU speech recognition / transcribe.cpp-vulkan | Hugging Face only in its manifest; this is a specific GGUF, not the upstream safetensors checkpoint. |

CPU Auto requires all THREE CPU model installations to be valid. A working Qwen 0.6B download alone does not make CPU Auto ready.

Use core/local_asr/local_stt_catalog.py, core/local_asr/local_stt_assets.py, data/models/*.manifest.json, and core/local_translation/assets.py as the actual contracts. Paths are relative to src/puripuly_heart.

Qwen 0.6B's existing ModelScope repository is zengshuishui/Qwen3-ASR-onnx at revision c69fb1666ccb59a82c09840c511a6c894e6a2482. Its ONNX files use model_0.6B path overrides. Do not replace this with raw Qwen weights or omit tokenizer assets.

The Gemma repository is unsloth/gemma-4-E4B-it-qat-GGUF at revision 8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265. Both gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf and mtp-gemma-4-E4B-it.gguf must match their existing sizes and SHA-256 values. The combined size is 4,275,373,792 bytes. Do not change quantization, remove the draft requirement, or weaken hashes just to make an unrelated mirror pass.

A new source should be a per-artifact mapping to verified byte-identical assets. ModelScope is a candidate hosting ecosystem, not a universal Hugging Face URL replacement. No new universal, byte-verified source was established during this session. In particular, existing Qwen 0.6B support is not a newly added source, and a model-card page is not proof of a working binary endpoint.

## Validation actually performed in this session

Twelve isolated tests passed under Python 3.13.5 / pytest 9.0.2. They import the modified owner and geometry modules through a temporary namespace-package harness, with feature_diagnostics.log_feature replaced by a no-op and subprocesses represented by fake pipes. The harness is not installed in the application or committed as a runtime component.

Covered: show/hide/show reuse; child hide followed by reopen; stale hide acknowledgement rejection; admitted result delivery while hidden; concurrent opens using one child; closed owner refusing reopen; exited child replacement; portrait default; legacy-default migration; custom dimensions; offscreen coordinates; small-work-area geometry.

Compilation of the three changed/new source modules also passed. This does NOT prove whole-project imports, normal logging, Flet behavior, Windows process lifecycle, screen geometry, IME input, game interaction, real translation, or model downloads. Those remain NOT_TESTED here.

CodeRabbit did not run: its CLI was missing and installation failed with `curl: (6) Could not resolve host: cli.coderabbit.ai`. No CodeRabbit issue count or clean-review claim is available. Run the review in an environment with working DNS/network and authenticated CodeRabbit; do not substitute this test result for that review.

## DSH execution and acceptance

1. Inspect git status and preserve local uncommitted work. Fetch the follow-up branch and use a dedicated worktree/local branch. Do not reset or clean the user's working tree. Read AGENTS.md and docs/architecture.md. Confirm the checked-out commit and the five-file follow-up diff against c040aa6d5f76c576780cd64058233b312ba873b4.
2. Use the repository's required Python and locked dependencies, not this session's reduced test harness. Run `uv run pytest tests/test_desktop_quick_toggle_contract.py -q`, then locate and run existing quick-window, manual-typing, application-lifecycle, settings, output/event-bridge, model-source and installer regressions. Run the complete appropriate project suite before calling it regression-clean. Investigate failures rather than deleting assertions to match changed behavior.
3. Complete all requirements marked NOT implemented. Add behavior tests for exactly-once submission/publication, interleaved SELF/PEER order, late translations, all display modes, translation-off no-LLM sending, main language changes, provider replacement, retries, visibility state, and bounded queues/history.
4. Run CodeRabbit against the original feature-branch base after implementation. If unavailable, report the actual review blocker instead of claiming approval.
5. In an isolated Windows preview profile, launch the real GUI and record the build commit. Verify the toggle beside the composer, portrait proportions, selected-state synchronization, actual messages, input retention, main minimization, close/hide/reopen, clipboard, Chinese and Japanese IME Enter versus Shift+Enter, screen DPI, monitor removal, and clean exit without orphan processes. Use authorized local/mock backends for automation; do not borrow production keys or spend cloud quota silently.
6. Verify sources per model and per file. First test metadata, bounded Range data and redirect chains; treat this only as a probe. For actual download acceptance, retrieve every required byte once where feasible, validate all existing hashes/sizes, install atomically, and load the matching recognizer/translator. Keep verified existing caches; do not wastefully redownload them. Test interruption, cancellation, disk-full/permission errors, HTML/404 responses, corrupt files, and preservation of a previously working installation. State explicitly whether the test ran on the user's mainland network with proxy/TUN off; do not change their VPN/firewall globally without permission.
7. Produce a small report with commit, exact commands, test outcomes, GUI evidence, and a per-model source/probe/full-download/hash/load matrix. Keep IMPLEMENTED, automated PASS, manual PASS, FAIL, BLOCKED and NOT_TESTED distinct. No full packet captures, unlimited logs, transcript dumps, tokens or signed download URLs in reports.

Manual checks to leave for the user after class: a short real conversation; translation-off original sending; game/window focus and input behavior; their actual network's model download performance. Do not wait for these before finishing the automatable work, and do not mark them PASS in advance.
