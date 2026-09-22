# Overlay speaker-transition implementation and visual comparison

Status: repaired implementation and working designs A2/C1/E1, based on candidate `4ac0f34b6683f793e94bcbb36bd6f037ac5a9ce3`. Browser comparison, deterministic production replay, live Soniox synthetic-audio plumbing, actual retained desktop controls and Windows native texture checks are recorded separately below. Human interpretation and actual SteamVR/HMD evaluation remain not run; this is not completed issue acceptance.

- Issue: https://github.com/kapitalismho/PuriPuly-heart/issues/178
- Repair baseline: `4ac0f34b6683f793e94bcbb36bd6f037ac5a9ce3`; input baseline pinned in scenario: `13274569769d3c1ec7a896a2d15b919b76136a6e`
- Evidence paths: `experiments/overlay_speaker_transition/**` and this document.
- Director decisions applied: default absent/invalid is A; simple localized selector/persistence with no elaborate migration; A/C/E all retained in current scope; settings are disposable owner-comparison controls that the owner plans to remove before official release, so public UX effort is minimized.
- Director-selected working designs pending real native/HMD verification: **A2, C1, E1**. C2/E2/A1/baseline remain in the prototype as comparators only. Working selection is not final validation and not a global winner claim.
- Design adjustment U2: A2/E1 gap 28→18 reference px fits existing native 32px top padding (14px height + 18px gap); geometry scales with text, length 196px is capped to the first rendered line's ink width, and the marker follows that line's centered text origin. Text positions and slot sizes are unchanged.

## 1. What was built

Runnable browser prototype plus deterministic replay/contrast inputs, all on the same fixed synthetic sequence and the existing two-block window:

- `experiments/overlay_speaker_transition/prototype.html` — runnable comparison. Mode switcher (baseline + A1/A2/C1/C2/E1/E2), event scrubber t01..t14, dark/bright/busy scenes, none/P/D/T in-page CVD simulation, side-by-side same-event grid. Repair state: centered text, transparent strip (effective alpha 0), 18px-gap rules with first-line cap.
- `experiments/overlay_speaker_transition/scenario.json` — fixed input v2: 14 logical events, evidence kinds, two-block window policy, native layout constraints, candidate geometry/palette table, background definitions.
- `experiments/overlay_speaker_transition/replay.py` — deterministic replay of every mode over the fixed sequence; writes `expected_results.json`; enforces 13 invariants (segmentation, single toggle/marker, SELF interposition, consecutive changes, unknown/reset/late withholding, revision identity, multilingual whole-turn color, E expiry keeping its marker).
- `experiments/overlay_speaker_transition/contrast_check.py` — independent text-contrast table against raw scene samples (effective alpha 0) plus color-role distinction with protanopia/deuteranopia/tritanopia simulation; writes `contrast_analysis.json`.
- `experiments/overlay_speaker_transition/screenshots/*.webp` — 13 browser-captured stage images named `<mode>-idx<event>-<scene>-<cvd>.webp`; durable manifest with SHA256 in `experiments/overlay_speaker_transition/manifest.json`.
- `experiments/overlay_speaker_transition/manifest.json` — durable manifest: actual filenames, bytes, SHA256 for every screenshot and input/generator file.

Native constraints used as far as available without HMD access: 4096x1056 surface rendered at 0.22 scale (900x232 stage), centered text (native DWRITE center alignment), primary 132px, secondary 82px, 5px black quad outline, effective background alpha 0 (native `effective_background_alpha` returns 0.0 — preexisting text-only transparent overlay; configured 0.24 is not applied; production background policy unchanged), 48/40px padding, 32px strip top padding, 2-block window, no flashing, no prefixes, no person labels. CJK secondary lines render through system fallback fonts in the browser; native Shaper/collection behavior is not claimed.

Run:

```text
python experiments/overlay_speaker_transition/replay.py
python experiments/overlay_speaker_transition/contrast_check.py
python -m http.server 8137 --directory experiments/overlay_speaker_transition
http://127.0.0.1:8137/prototype.html
```

Reproduction: `replay.py` prints 13/13 PASS; scenario sha256 `b61d553e3667…` (full hash in `expected_results.json` meta and `manifest.json`).

## 2. Shared fixed sequence (speaker letters are test annotations, never product labels)

`t01 A initial` fresh peer, no claim. `t02 A same` delivery cut, no cue. `t03 B confirmed` first boundary/toggle. `t04 B same` continuation, no duplicate. `t05 SELF` white everywhere; C keeps run; E emphasis expires. `t06 B same across SELF` valid comparison, no new boundary. `t07 C confirmed`, `t08 A confirmed` (toggle once each; no lookup of earlier hue). `t09 UNKNOWN mixed/overlap` withheld; hue retained without claiming continuity. `t10 RESET` empty window, fresh Gold start implying nothing. `t11 revision` of expired `peer-a3` withheld. `t12 late` expired generation withheld. `t13 multilingual wrapping KO/EN confirmed` whole turn takes one arrival color; secondary keeps smaller size, never a second meaning. `t14 short EN/JA reply, same` E emphasis from t13 expires; marker persists on t13; t14 itself carries no marker.

## 3. Candidates (exact parameters)

All modes: SELF `#FFFFFF`. C1 peer base Gold `#FFD700`, third hue Sky `#33D6FF`. C2 peer base Gold `#FFD700` (F3 repair: corrected from `#FFC21A` so the only deliberate C difference is the third hue Mint `#2DE1A8`). Whole-turn coloring in C/E covers primary and secondary lines.

| ID | Mode | Palette | Marker geometry (native reference px at text_scale 1.0; prototype x0.22) |
|----|------|---------|-----------------------------------------------------------|
| baseline | — | SELF `#FFFFFF`, PEER `#FFD700` | none (comparator only) |
| A1 | A boundary dash | same as baseline | inline leading `U+2014` em dash, 1.0x primary size, 0.28em gap, Gold, only on first primary line; persists while block survives (comparator only) |
| A2 ★ | A boundary rule | SELF `#FFFFFF`, PEER `#FFD700`, marker Gold `#FFD700` | upper rule 196x14, 18px gap (14h+18gap fits native existing 32 top padding), aligned to centered first rendered line origin, width min(196, first-line ink width), scales with text; persists while block survives; never a full-panel break |
| C1 ★ | C alternate sky | SELF `#FFFFFF`, Gold `#FFD700`, Sky `#33D6FF` | none (color only) |
| C2 | C alternate mint | SELF `#FFFFFF`, Gold `#FFD700`, Mint `#2DE1A8` | none (color only; comparator only) |
| E1 ★ | E rule + sky | SELF `#FFFFFF`, body Gold `#FFD700` / emphasis Sky `#33D6FF`, persistent marker Gold `#FFD700` | same 196x14/18px centered-origin rule as A2, always Gold; body Sky on arrival, reverts Gold on first new readable logical turn (SELF or PEER); rule stays Gold throughout; no timer |
| E2 | E dash + mint | SELF `#FFFFFF`, Gold `#FFD700`, emphasis Mint `#2DE1A8` | leading em dash in arrival hue on the transition block, persists after body expires; same event-based expiry as E1 (comparator only) |

Production F2 basis (ProductionRepair; implementation owned there): marker Gold `#FFD700` (`cache_peer_text_brush`), top=0, height=14*text_scale; existing first-line layout origin_y=32*text_scale so marker-bottom to line-origin gap=18*text_scale without moving text; x=strip padding + DirectWrite cached first nonempty rendered line ink bounds.left (not `ResolvedLineLayout.origin_x`); width=min(196*text_scale, cached first-line ink width) so short replies cap the rule; applies to first nonempty primary or secondary line; text positions/window/slots unchanged. The prototype mirrors this with centered text and a first-line-length width cap; it does not reimplement DirectWrite ink measurement.

Explanatory copy used in the prototype (proposed, not finalized product copy): boundary = local speaker-change evidence at the start of that Peer block; same color never means the same person; no marker never means the same speaker.

## 4. Per-scenario results (browser prototype + replay, repair recapture)

Replay invariants: 13/13 PASS across baseline/A1/A2/C1/C2/E1/E2 on scenario v2 (sha `b61d553e3667…`). Key observable points: t02 no marker/toggle in any mode; t03 exactly one marker (A/E) or one toggle (C); t04 carry; t05 SELF white with C run retained and E emphasis expired; t06 no boundary across SELF; t07/t08 one toggle each with each marker kept on its own block; t08 color from toggle parity only (C2 now Gold `#FFD700`/Mint `#2DE1A8` verified in replay: t03/t13 windows carry Gold base with Mint run color); t09 withheld; t10 fresh Gold single-block window; t11/t12 withheld with window byte-identical to t10; t13 one arrival color across both lines; t14 E body expired to Gold with t13 Gold marker intact, C run unchanged. A synthetic same-key revision check confirms E emphasis survives rewording. `expected_results.json` records `marker_hex` per entry (E1 `#FFD700`).

Browser visual inspection, repaired prototype (stage DOM verified plus screenshot read-back; human-readable preliminary evidence, not HMD evidence):

- Centered layout: A2 t03 dark vision read-back confirms both blocks horizontally centered and the Gold rule centered on the block midpoint, not left-aligned. Short-reply cap: A2 t14 dark DOM shows the t13 marker on the t13 block and no marker on the t14 `OK.` block (correct per contract — t14 is `same` evidence); the prototype width cap renders a visibly shorter centered rule on short first lines.
- C1 vs C2 at t08 dark: both pairs clearly distinguishable — vision read-back confirms C2 upper Gold vs lower Mint. Whole-block coloring including secondary lines confirmed in DOM and images.
- E1 at t03 dark: Gold rule above, Sky body (DOM `background:#FFD700` + `color:#33D6FF`).
- E1 at t05 dark: SELF white block, prior emphasis expired, no stray marker on SELF. E1 at t14 dark: expired t13 body correctly Gold with its Gold rule marker still present above it; short reply readable.
- Transparent strip (F5): C1 t13 busy vision read-back confirms no dark strip panel behind the text — text sits directly on the gradient with dark outlines. A2 t13 bright: Gold text plus rule legible via outline on bright background; see honest bright/busy risk in §5.
- Corrected CVD in-page simulations at t13 busy: C1 deuteranopia renders Gold `#FADF24`-ish vs light blue `#A2BDFF`-ish (no longer purple); C2 deuteranopia Gold-derived `#EBD228`-ish vs warm gray `#C7C1AC`-ish. All pairs remain separable in the recaptured images; numbers in §5 are the corrected ones.

Screenshots (13 files, webp, full manifest in §8): baseline-idx02-dark-none; A2 idx02 dark, idx13 dark, idx12 bright; C1 idx07 dark, idx12 busy-none, idx12 busy-deuteranopia; C2 idx07 dark, idx12 busy-deuteranopia; E1 idx02 dark, idx04 dark, idx13 dark, idx12 busy-none. In-page CVD toggles for protanopia/tritanopia remain exercisable in the prototype but are not recaptured as files; corrected numbers cover all three in §5.
## 5. Contrast and color-role analysis (independent axes)

Method (F5 repair): WCAG relative luminance on sRGB against the raw scene sample — effective background alpha is 0 because native `effective_background_alpha` returns 0.0 (preexisting text-only transparent overlay; `native/overlay/src/renderer/types.rs`; configured 0.24 is not applied and production background policy is unchanged). The 5px black outline is not modeled numerically, so bright/busy ratios below are a lower bound before the outline contribution. Busy gradient bounded by four samples; fog sample approximates the worst case. Generator: `contrast_check.py`; full table: `contrast_analysis.json`.

CVD audit (U1 follow-up): the first version applied its 3x3 matrices directly to gamma-encoded sRGB, which is the documented-wrong order and produced the suspicious deuteranopia Sky→purple rendering (`#7064F3`-ish). Both `contrast_check.py` and the prototype now use Machado-Oliveira-Fernandes-2009 severity-1.0 matrices in linear light: sRGB decode → matrix → clamp → sRGB encode. References: Machado et al. 2009 IEEE TVCG 15(6) 1291–1298, DOI 10.1109/TVCG.2009.113; R colorspace maintainers' note that Machado transforms belong in linear RGB; Vienot/Brettel/Mollon pipeline descriptions requiring the same linear-light order. All CVD numbers and in-page simulations below are the corrected ones; the old gamma-space numbers are withdrawn.

Text vs raw scene, effective alpha 0 (higher better; F5 recalculation — values shifted from the withdrawn 0.24-strip table):

| Text | Dark | Bright | Busy navy/ochre/teal/fog |
|------|------|--------|--------------------------|
| White | 19.57 | 1.15 | 14.22 / 6.18 / 3.03 / 1.23 |
| Gold | 13.95 | 1.22 | 10.14 / 4.41 / 2.16 / 1.14 |
| Sky | 11.37 | 1.49 | 8.26 / 3.59 / 1.76 / 1.40 |
| Mint | 11.60 | 1.46 | 8.43 / 3.66 / 1.79 / 1.38 |

Role distinction (linear-RGB distance; larger = more separable), corrected CVD rows (unchanged by F5):

- Gold–Sky: 1.391 (P 1.063 / D 1.166 / T 1.065); luminance delta 0.138 (P 0.016 / D 0.218 / T 0.036).
- Gold–Mint: 1.052 (P 0.405 / D 0.587 / T 1.034); luminance delta 0.126 (P 0.016 / D 0.200 / T 0.056).
- White–Gold 1.050, White–Sky 1.021, White–Mint 1.174 for reference.
- Corrected simulated swatches: Gold→ P `#F0D300` / D `#FADF24` / T `#FFC4B7`; Sky→ P `#BDD0FF` / D `#A2BDFF` / T `#00E3E3`; Mint→ P `#DCD0A5` / D `#C7C1AC` / T `#00E1D1`.

Honest bright/busy risk: without a dimming strip, Gold drops to 1.22 on bright and 1.14 on busy-fog; Sky/Mint sit at 1.38–1.49 there. Text legibility on bright/busy scenes therefore relies on the black outline plus scene variation, not on fill-vs-scene luminance. This is recorded as a risk for HMD verification, not as a pass. Under the corrected simulation, Sky keeps the larger Gold separation in every CVD row, and the gap over Mint is widest under protanopia/deuteranopia (D 1.166 vs 0.587). Do not claim color-only C provides the redundant cue available in A/E.

## 6. Working selection (Director-selected; not final validation)

- A: working design **A2 (compact upper Gold rule, 196x14/18px gap, centered-origin, first-line cap)**. Rationale from actual repaired-prototype observation: detectable without reading the first word; centered attachment matches the native centered layout (vision-confirmed centered rule on t03 dark); 14h+18gap fits the native 32 top padding without moving text; gold-on-gold keeps the palette intact with zero new-hue risk. Exact spec: §3 A2 row. Risk: at very small text scales the 14px rule is thinner than a dash stroke; Gold-rule-on-Gold-body legibility at HMD scale is a pending native check.
- C: working design **C1 (Gold `#FFD700` / Sky `#33D6FF`)**. Rationale: largest measured Gold–third-hue separation overall and under every corrected simulated CVD (P 1.063 / D 1.166 / T 1.065 vs Mint 0.405 / 0.587 / 1.034); whole-turn single color keeps primary/secondary hierarchy clean; C2 comparator now controlled (Gold base corrected) so the Sky-vs-Mint comparison is valid. Exact spec: §3 C1 row; no marker. Risk: color-only channel carries no redundant cue; protan/deutan luminance deltas are small (0.016/0.218) — pair with the mandatory copy line that equal hues never imply the same person. C2 remains a tested fallback with identical semantics.
- E: working design **E1 (persistent Gold rule + Sky emphasis, same 196x14/18px geometry)**. Rationale: reuses the A2 boundary geometry with the most separable emphasis hue; the Gold rule persists after the Sky body expires to Gold, preserving a cue the original Cyan-only design lost. Exact spec: §3 E1 row; event-based expiry only. Risk: two simultaneous changes (marker + full-body recolor) is the most salient option; consecutive transitions produce back-to-back Sky bodies, each with its own Gold marker — correct per contract but visually busy; whether the Gold rule reads clearly against a Gold body at HMD scale is a pending native check.

Production policy: a shared semantic interpreter consumes bounded session-scoped, source-timed speaker evidence before renderer projection; renderers receive style and boundary fields, not raw provider speaker IDs. Unknown, overlapping, invalid, reordered, or incomparable evidence cannot confirm a change. Claims attach only on the logical turn's first readable Peer output, translated or original-only; later revisions cannot replay a cue. C retains its hue through uncertainty when surviving content makes a reset misleading; a fresh empty context starts Gold. Entering E does not replay historical emphasis. The simple localized selector persists A/C/E; absent/invalid values resolve to A. These are disposable owner-comparison controls (minimal UX, planned removal before official release—not a waiver for HMD/interpretation checks), not a permanent public preference design.

## 7. What was observed vs what is not claimed

Measurable/observable evidence includes replay invariants, browser DOM/screenshots and contrast calculations, the actual retained desktop control tree, live synthetic Soniox production plumbing, and Windows D3D/DirectWrite texture rendering. These are distinct surfaces; none establishes HMD readability. Historical invalidation: the first candidate's desktop evidence exercised a non-production builder and missed the absent retained-tree marker. That claim is withdrawn; §12 records the repaired real retained path. Human interpretation, distraction and actual HMD visual evaluation remain not run.

## 8. Durable manifest (repair revision; F4 fix)

Full manifest: `experiments/overlay_speaker_transition/manifest.json` (actual filenames, bytes, SHA256; 13 screenshots + 6 input/generator files). Summary:

- `screenshots/A2-idx02-dark-none.webp` 18284 bytes, sha256 `3dfe7eea206de2f782152ca2f815527aa9384a54b06bd034eee4b1d2dfe3b0cb`
- `screenshots/A2-idx12-bright-none.webp` 37676 bytes, sha256 `d5316a9198687b0bae4e1633a5495a24ba0f8513bdc65414b136d78969b6ae1d`
- `screenshots/A2-idx13-dark-none.webp` 27634 bytes, sha256 `9e3a91129c5d5b769f704fbd15051213ffdd1d7a17125268a67c803d87742402`
- `screenshots/baseline-idx02-dark-none.webp` 18688 bytes, sha256 `cb235296583bffc8e3cf4ebd596b5cd5229fdf06ab0a8ce6d6424ada29286577`
- `screenshots/C1-idx07-dark-none.webp` 11562 bytes, sha256 `ea79695da786d5bca58b3a061773514476c30e4dca88e62301279dabc6bfaba1`
- `screenshots/C1-idx12-busy-deuteranopia.webp` 31920 bytes, sha256 `431d62d5c915a8f8e6aca851db2e7279aecc26327a8eee18d3254f579cd43917`
- `screenshots/C1-idx12-busy-none.webp` 34646 bytes, sha256 `dd5c31727afa46a6d93bde52dac6af9b6e16b8c296a01e527ab42060d52be1c5`
- `screenshots/C2-idx07-dark-none.webp` 11198 bytes, sha256 `97f2837169f6ff6323382bba26353be46270b964076bf5d04d6afaeeec062ae8`
- `screenshots/C2-idx12-busy-deuteranopia.webp` 32484 bytes, sha256 `924791ca9db881e7eab1581785e60deeef5305102f55b681ae64348d5d021ad9`
- `screenshots/E1-idx04-dark-none.webp` 12770 bytes, sha256 `32ee3efdaf4342eebacfa1d3e5b6f8c0fd724116202d3b68d1798379a1093814`
- `screenshots/E1-idx12-busy-none.webp` 34766 bytes, sha256 `494eb871fe7bad8d87faa91cc1558f1b9df35a3b881502eb1d23bb945f2524fb`
- `screenshots/E1-idx13-dark-none.webp` 27634 bytes, sha256 `9e3a91129c5d5b769f704fbd15051213ffdd1d7a17125268a67c803d87742402`
- `scenario.json` 16708 bytes, sha256 `b61d553e3667728109b39e3c2e9b37f25b16f5682b0795bb86359f0de32abea2`
- `expected_results.json` 98867 bytes, sha256 `7285358185de57648b8dc6c3744335467b5c410aa897839c96779a05482bcb3c`
- `contrast_analysis.json` 3390 bytes, sha256 `39104e694fd7ba4acffe4ac38b3c0c3c630b54d119dab792a68d3f5f0a10d6e9`
- `prototype.html` 21351 bytes, sha256 `0001cdf51bf92832ab4a28f06aee5aeae02e7473f8612a1bb256054e9146e86d`
- `replay.py` 11963 bytes, sha256 `11c2a11dddf5263256229f836f1cb03fd1164834ce6881d76be4252c1f4b32ca`
- `contrast_check.py` 5185 bytes, sha256 `9ca572c1da4b65d4a29d90267ab5432131bacbaff9e9d1675aa2bdc878879197`
## 9. Explicitly pending (not run / blocked / reserved)

- Actual Windows/SteamVR/HMD visual pass for working designs A2/C1/E1 and UI switching: NOT RUN. Includes boundary legibility and small-text rule thinness.
- Live two-voice Soniox production plumbing and retained desktop controls: exercised; see §11–§12. Actual desktop window capture was attempted but failed to resolve a window handle; no desktop screenshot is claimed.
- Small engineering/owner comprehension evaluation (before/after explanation questions, distraction notes): NOT RUN; protocol proposed in §10.
- Production mode lifecycle, source/generation reset, overlapping synthetic evidence, persistence and protocol checks: exercised as listed below. HMD clipping, gaze-away/return, distraction and actual UI-to-HMD switching remain unverified.
- Default (absent/invalid A), disposable selector scope, and any strategy elimination: Director-decided as noted in the header; final removal of disposable controls belongs to the owner before official release.

## 10. Suggested owner validation protocol (handoff, not executed)

For each of A2/C1/E1 on HMD plus baseline: show t03 boundary/toggle, t05 SELF interposition, t08 consecutive change, t09 unknown, t10 reset, t13 multilingual wrap, t14 short reply, each on dark/bright/busy. Ask before explanation: where did the speaker change; do equal hues imply the same person; does no cue prove continuity; does the marker read as topic split. Then explain the cue meaning once and re-ask. Record errors and distraction, not preference scores.

## 11. Initial production integration evidence

The checks below were first run on candidate `4ac0f34b6683f793e94bcbb36bd6f037ac5a9ce3`; §12 records repairs and affected rechecks. Historical native binary identity and initial results remain here for traceability, not as proof of the repaired renderer.
Integration baseline: `13274569769d3c1ec7a896a2d15b919b76136a6e`. App/native version remains 2.7.0; settings schema is 47, overlay contract is 10, execution contract remains r2, and native retry ownership remains exclusive. Startup explicitly advertises speaker-transition presentation version 1 and all three modes. OBS #176 is absent from this baseline; no OBS feature was added.

The existing Audio/content segmentation, translation-child identities, two-block visible window, output routing and single presenter expiry owner remain the intended architecture. The new shared speaker interpretation and presenter projection are described in `docs/architecture.md`; no new speaker registry, expiry owner or PSEM implementation is introduced.

### Executed checks

| Surface | Command or durable artifact | Observed result |
|---|---|---|
| Overlay/output/translation semantics | `uv run pytest -q tests/core tests/core/runtime/test_output_runtime.py tests/integration/test_speaker_transition_pipeline.py -k "overlay or peer_translation or translation_output or translation_turn or soniox or speaker_transition" -rs` | 469 passed; 2 real-subprocess tests skipped |
| Soniox/provider replay | `uv run pytest -q tests/providers/test_soniox_backend.py tests/providers/test_soniox_reuse.py tests/integration/test_soniox_stt_integration.py tests/core/test_soniox_multilingual_release_readiness.py tests/integration/test_speaker_transition_pipeline.py -rs` | 61 passed; 1 opt-in live test skipped; separate live smoke below was run |
| Settings/config/app/UI | `uv run pytest -q tests/config tests/app tests/ui -k "settings and not logs"` | 847 passed |
| Native suite | `cmd /c "set CARGO_TARGET_DIR=C:\t\ovr178&& cargo test --manifest-path native\overlay\Cargo.toml"` | 270 passed; 1 opt-in memory/performance probe ignored |
| Native format | `cargo fmt --manifest-path native/overlay/Cargo.toml -- --check` | Passed |
| Native executable | `cmd /c "set CARGO_TARGET_DIR=C:\t\ovr178&& cargo build --locked --manifest-path native\overlay\Cargo.toml --bin PuriPulyHeartOverlay"` then `--check-startup-contract` | Built and reported contract 10 with A/C/E capability |
| Settings/runtime smoke | `experiments/overlay_speaker_transition/runtime_validation/runtime_evidence.json` | Actual SettingsView A→C→E→A intents; persisted C/E/A reload; retained text/visible age; unchanged STT signature; both desktop text lines share body hue and slot height remains unchanged |

Native debug executable SHA256: `aa18e9e9f0745329d51ddafff98550ac661976885286e9161a9cc6ecc5a09d6e`. This identifies the startup-tested binary, not HMD rendering. Short `CARGO_TARGET_DIR` avoids the observed Windows CMake long-path failure.

### Deterministic production replay

`tests/integration/test_speaker_transition_pipeline.py` routes real Soniox JSON parsing through scoped terminal handling, `PeerTranslationChannelOwner`, translation children, output publication and the presenter. The deterministic translation provider is a test substitute; speaker plumbing is production code. A→A→B→B→SELF→B→C→A produces Gold→Gold→Sky→Sky→White→Sky→Gold→Sky in C. Additional cases withhold changes after generation/session reset and overlapping source spans, and reject late cue attachment without refreshing content age.

### Live Soniox synthetic-audio smoke

Durable runner: `experiments/overlay_speaker_transition/runtime_validation/live_soniox_production_path_smoke.py`. Result: sibling `live_production_path_result.json`. Invocation:

```text
uv run python experiments/overlay_speaker_transition/runtime_validation/live_soniox_production_path_smoke.py --audio-dir <local-synthetic-wav-directory>
```

Two synthetic Windows SAPI voices (Microsoft David and Zira) supplied nonprivate English test sentences; no microphone or private speech was used. Credentials were loaded in memory from the configured secret store and are not included in artifacts. One provider session returned speaker 1 at 60–2520 ms (confidence 0.885), then speaker 2 at 3420–5160 ms (confidence 0.967), with no overlap. Live scoped terminals went through receipts, actual Peer translation ownership, translation turns/output projection and `OverlayPresenter`; only translation text used a deterministic echo provider. The first claim was context reset; the second was transition. A produced Gold plus rule, C Sky without rule, E Sky plus rule.

The earlier direct interpreter/presenter injection recorded in `runtime_evidence.json` is provider-capability evidence only; the separate production-path smoke closes that wiring gap. Neither check establishes PSEM readiness, live overlapping-speech accuracy, native pixels or HMD readability.

### Remaining acceptance blockers

SteamVR was not running during validation and no human HMD observation was available. Required actual Windows/SteamVR/HMD checks (including UI switching, dark/bright/busy scenes, scale, multilingual wrapping, rapid changes and gaze-away/return) and before/after-explanation interpretation observations are not run. A desktop control-tree check or successful native startup cannot replace them. All three modes remain available; the issue must not be closed on this evidence alone.

## 12. Independent review repairs and affected rechecks

Checkpoint review of `13274569769d3c1ec7a896a2d15b919b76136a6e..4ac0f34b6683f793e94bcbb36bd6f037ac5a9ce3` found the following defects, all accepted for repair:

- S1: original-only/failed-translation Peer output dropped transition claims. Claims now travel independently of translated text and attach once on first readable Peer output. Translation-off and failure fallback are covered for A/C/E, including subsequent same-speaker content and SELF expiration of E.
- F1: the actual retained desktop renderer lacked boundary controls. Its reusable slots now create, update and clear markers for boundary-off, SELF and empty reuse without shifting text or changing card sizes.
- F2: native marker alignment and scale diverged from the prototype. DirectWrite cached ink metrics now locate the first nonempty rendered primary/secondary line; geometry follows text scale, and U2 fits the marker in existing top padding.
- F3–F5: C2 palette disagreement, screenshot inventory/hashes and fictitious backing-strip contrast were corrected. The manifest and repaired comparison results above supersede the initial comparison evidence.

The proposed S2 defect (invalid intra-run timing resets the comparison reference) was rejected: conservative withholding is required when no valid comparison remains. Provider timing sensitivity is unmeasured. The live artifacts contain merged speaker runs, not raw token counts; one run does not imply one token.

Affected checks after repair:

| Check | Result |
|---|---|
| `uv run pytest -q tests/integration/test_speaker_transition_pipeline.py` | 7 passed, including A/C/E × disabled/failed translation |
| Peer owner, translation output/streaming, presenter/modes, output wiring/runtime and integration replay selection | Passed; same named files in the repair commit's tests |
| `uv run pytest -q tests/config/test_overlay_settings.py tests/ui/test_desktop_overlay_renderer.py` | Passed |
| `tests/ui/test_desktop_overlay_renderer.py -k retained_surface_applies_and_clears_speaker_boundary` | Passed through actual `FletDesktopRendererWindow` retained controls |
| `cargo test --manifest-path native/overlay/Cargo.toml --target-dir C:/t/ovr178` | 272 passed; 1 opt-in probe ignored. Includes scaled short/wide line geometry and a real Windows D3D/DirectWrite texture smoke |
| `cargo build --release --manifest-path native/overlay/Cargo.toml --target-dir C:/t/ovr178` | Passed |
| Release executable `--check-startup-contract` | App 2.7.0, overlay 10, execution r2, exclusive retry, A/C/E presentation v1 |
| `uv run ruff check .` | Passed |
| Live Soniox production-path runner, rerun after repair | Passed for A/C/E; sanitized result updated in `runtime_validation/live_production_path_result.json` |

Repaired native release executable SHA256: `06ce2b35869933073af72a7a71d87815ac226796263f49da15dfd1d6fe4a3667`, size 2,629,120 bytes. The original debug identity in §11 is superseded for renderer evidence. No packaged deployment, push or release was performed.

All three modes remain implemented and selectable. Required HMD and human-interpretation observations remain blocked; successful renderer textures, tests and provider plumbing do not close those criteria.
