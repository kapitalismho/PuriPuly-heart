# Overlay speaker-transition implementation and visual comparison

Status: A/C/E implemented with working designs A2/C1/E1. Browser comparison, production-path replay, live Soniox synthetic-audio plumbing, settings/runtime checks, and native startup checks are recorded below. Actual HMD readability and human interpretation remain blocked; this is not completed issue acceptance.

- Issue: https://github.com/kapitalismho/PuriPuly-heart/issues/178
- Implementation baseline pinned in input: `13274569769d3c1ec7a896a2d15b919b76136a6e`
- Evidence paths: `experiments/overlay_speaker_transition/**` and this document.
- Director decisions applied: default absent/invalid is A; simple localized selector/persistence with no elaborate migration; A/C/E all retained in current scope; settings are disposable owner-comparison controls that the owner plans to remove before official release, so public UX effort is minimized.
- Director-selected working designs pending real native/HMD verification: **A2, C1, E1**. C2/E2/A1/baseline remain in the prototype as comparators only. Working selection is not final validation and not a global winner claim.

## 1. What was built

Runnable browser prototype plus deterministic replay/contrast inputs, all on the same fixed synthetic sequence and the existing two-block window:

- `experiments/overlay_speaker_transition/prototype.html` — runnable comparison. Mode switcher (baseline + A1/A2/C1/C2/E1/E2), event scrubber t01..t14, dark/bright/busy scenes, none/P/D/T in-page CVD simulation, side-by-side same-event grid.
- `experiments/overlay_speaker_transition/scenario.json` — fixed input: 14 logical events, evidence kinds, two-block window policy, native layout constraints, candidate geometry/palette table, background definitions.
- `experiments/overlay_speaker_transition/replay.py` — deterministic replay of every mode over the fixed sequence; writes `expected_results.json`; enforces 13 invariants (segmentation, single toggle/marker, SELF interposition, consecutive changes, unknown/reset/late withholding, revision identity, multilingual whole-turn color, E expiry keeping its marker).
- `experiments/overlay_speaker_transition/contrast_check.py` — independent text-contrast table plus color-role distinction with protanopia/deuteranopia/tritanopia simulation; writes `contrast_analysis.json`.
- `experiments/overlay_speaker_transition/screenshots/*.webp` — 11 browser-captured stage images named `<mode>-idx<event>-<scene>-<cvd>.webp`.

Native constraints used as far as available without HMD access: 4096x1056 surface rendered at 0.22 scale (900x232 stage), primary 132px, secondary 82px, 5px black quad outline, strip `rgba(0,0,0,0.24)`, 48/40px padding, 2-block window, no flashing, no prefixes, no person labels. CJK secondary lines render through system fallback fonts in the browser; native Shaper/collection behavior is not claimed.

Run:

```text
python experiments/overlay_speaker_transition/replay.py
python experiments/overlay_speaker_transition/contrast_check.py
python -m http.server 8137 --directory experiments/overlay_speaker_transition
http://127.0.0.1:8137/prototype.html
```

Reproduction: `replay.py` prints 13/13 PASS; scenario sha256 `246a0672a758f52d83164ee7aa3ad12912d467ceb1ab97b15290ad0dcdadbf9b`.

## 2. Shared fixed sequence (speaker letters are test annotations, never product labels)

`t01 A initial` fresh peer, no claim. `t02 A same` delivery cut, no cue. `t03 B confirmed` first boundary/toggle. `t04 B same` continuation, no duplicate. `t05 SELF` white everywhere; C keeps run; E emphasis expires. `t06 B same across SELF` valid comparison, no new boundary. `t07 C confirmed`, `t08 A confirmed` (toggle once each; no lookup of earlier hue). `t09 UNKNOWN mixed/overlap` withheld; hue retained without claiming continuity. `t10 RESET` empty window, fresh Gold start implying nothing. `t11 revision` of expired `peer-a3` withheld. `t12 late` expired generation withheld. `t13 multilingual wrapping KO/EN confirmed` whole turn takes one arrival color; secondary keeps smaller size, never a second meaning. `t14 short EN/JA reply, same` E emphasis from t13 expires; marker persists; C run unchanged.

## 3. Candidates (exact parameters)

All modes: SELF `#FFFFFF`. Peer base Gold `#FFD700` (C2 also tested `#FFC21A` in an early variant; the prototype pins C2 peer base to Gold and Mint `#2DE1A8` so the only deliberate C difference is the third hue; see §6). Whole-turn coloring in C/E covers primary and secondary lines.
| ID | Mode | Palette | Marker geometry (native px at 4096 wide; prototype x0.22) |
|----|------|---------|-----------------------------------------------------------|
| baseline | — | SELF `#FFFFFF`, PEER `#FFD700` | none (comparator only) |
| A1 | A boundary dash | same as baseline | inline leading `U+2014` em dash, 1.0x primary size, 0.28em gap, Gold, only on first primary line; persists while block survives (comparator only) |
| A2 ★ | A boundary rule | SELF `#FFFFFF`, PEER `#FFD700`, marker Gold `#FFD700` | compact upper rule 196x14, 28px gap, left-aligned to block text origin, Gold; persists while block survives; never a full-panel break |
| C1 ★ | C alternate sky | SELF `#FFFFFF`, Gold `#FFD700`, Sky `#33D6FF` | none (color only) |
| C2 | C alternate mint | SELF `#FFFFFF`, Gold `#FFD700`, Mint `#2DE1A8` | none (color only; comparator only) |
| E1 ★ | E rule + sky | SELF `#FFFFFF`, body Gold `#FFD700` / emphasis Sky `#33D6FF`, persistent marker Gold `#FFD700` | upper 196x14 Gold rule on the transition block, left-aligned, 28px gap; body Sky on arrival, reverts Gold on first new readable logical turn (SELF or PEER); rule stays Gold throughout; no timer |
| E2 | E dash + mint | SELF `#FFFFFF`, Gold `#FFD700`, emphasis Mint `#2DE1A8` | leading em dash in arrival hue on the transition block, persists after body expires; same event-based expiry as E1 (comparator only) |

Production policy synced with implementation owner: A2/E1 rule is left-aligned compact upper rule at native 4096 convention 196x14 with 28px gap; the E1 rule remains Gold `#FFD700` while the body is Sky on the initial readable transition and stays Gold when the body returns to Gold. C1/E1 body Sky is `#33D6FF`; SELF White.
Explanatory copy used in the prototype (proposed, not finalized product copy): boundary = local speaker-change evidence at the start of that Peer block; same color never means the same person; no marker never means the same speaker.

## 4. Per-scenario results (browser prototype + replay, 2026-09-23)

Replay invariants: 13/13 PASS across baseline/A1/A2/C1/C2/E1/E2 after the E1 Gold-marker sync (scenario sha `246a0672a758…`). Key observable points: t02 no marker/toggle in any mode; t03 exactly one marker (A/E) or one toggle (C); t04 carry; t05 SELF white with C run retained and E emphasis expired; t06 no boundary across SELF; t07/t08 one toggle each with each marker kept on its own block; t08 color from toggle parity only; t09 withheld; t10 fresh Gold single-block window; t11/t12 withheld with window byte-identical to t10; t13 one arrival color across both lines; t14 E body expired to Gold with t13 Gold marker intact, C run unchanged. A synthetic same-key revision check confirms E emphasis survives rewording. `expected_results.json` now records `marker_hex` per entry (E1 `#FFD700`).

Browser visual inspection (stage DOM verified plus screenshot read-back; this is human-readable preliminary evidence, not HMD evidence):

- A2 at t03 dark: Gold rule above the incoming Gold block, explicitly not a full-width divider. E1 at t03 dark re-verified after sync: Gold rule above, Sky body (DOM `background:#FFD700` + `color:#33D6FF`); vision read-back confirms the rule matches the first block's Gold, not the Sky body.
- C1 vs C2 at t08 dark: both pairs clearly distinguishable (Gold vs Sky; Gold vs Mint). Whole-block coloring including secondary lines confirmed in DOM and images.
- E1 at t05 dark: SELF white block, prior emphasis expired, no stray marker on SELF. E1 at t14 dark: expired t13 body correctly Gold with its Gold rule marker still present above it; short reply readable.
- A2 at t13 bright: rule marker and Gold text legible on bright background with outline; bright compresses all contrast (see §5).
- Corrected CVD in-page simulations at t13 busy: C1 deuteranopia renders Gold `#FADF24`-ish vs light blue `#A2BDFF`-ish (no longer purple); C1 protanopia Gold `#F0D300`-ish vs `#BDD0FF`-ish; C2 deuteranopia Gold-derived `#EBD228`-ish vs warm gray `#C7C1AC`-ish. All pairs remain separable in the recaptured images; numbers in §5 are the corrected ones.

Screenshots (13 files, webp, hashes in §8): baseline/A2 at idx02 dark; C1 at idx07 dark; E1 at idx02/idx04 dark plus idx13 dark; A2 idx12 bright; C1/C2 idx12 busy deuteranopia and protanopia. Stale pre-sync captures (A1 dash, old Sky-marker E1/E2, old gamma-space CVD) removed.
## 5. Contrast and color-role analysis (independent axes)

Method: WCAG relative luminance on sRGB; strip `rgba(0,0,0,0.24)` blended over scene samples; 5px black outline not modeled numerically. Busy gradient bounded by four samples; fog sample approximates the worst case. Generator: `contrast_check.py`; full table: `contrast_analysis.json`.

CVD audit (U1 follow-up): the first version applied its 3x3 matrices directly to gamma-encoded sRGB, which is the documented-wrong order and produced the suspicious deuteranopia Sky→purple rendering (`#7064F3`-ish). Both `contrast_check.py` and the prototype now use Machado-Oliveira-Fernandes-2009 severity-1.0 matrices in linear light: sRGB decode → matrix → clamp → sRGB encode. References: Machado et al. 2009 IEEE TVCG 15(6) 1291–1298, DOI 10.1109/TVCG.2009.113; R colorspace maintainers' note that Machado transforms belong in linear RGB; Vienot/Brettel/Mollon pipeline descriptions requiring the same linear-light order. All CVD numbers and in-page simulations below are the corrected ones; the old gamma-space numbers are withdrawn.

Text vs backdrop (higher better; unchanged by the CVD fix):

| Text | Dark | Bright | Busy navy/ochre/teal/fog |
|------|------|--------|--------------------------|
| White | 19.91 | 2.04 | 16.23 / 8.90 / 4.92 / 2.16 |
| Gold | 14.19 | 1.45 | 11.57 / 6.34 / 3.51 / 1.54 |
| Sky | 11.57 | 1.19 | 9.43 / 5.17 / 2.86 / 1.26 |
| Mint | 11.80 | 1.21 | 9.62 / 5.27 / 2.92 / 1.28 |

Role distinction (linear-RGB distance; larger = more separable), corrected CVD rows:

- Gold–Sky: 1.391 (P 1.063 / D 1.166 / T 1.065); luminance delta 0.138 (P 0.016 / D 0.218 / T 0.036).
- Gold–Mint: 1.052 (P 0.405 / D 0.587 / T 1.034); luminance delta 0.126 (P 0.016 / D 0.200 / T 0.056).
- White–Gold 1.050, White–Sky 1.021, White–Mint 1.174 for reference.
- Corrected simulated swatches: Gold→ P `#F0D300` / D `#FADF24` / T `#FFC4B7`; Sky→ P `#BDD0FF` / D `#A2BDFF` / T `#00E3E3`; Mint→ P `#DCD0A5` / D `#C7C1AC` / T `#00E1D1`.

Reading: text contrast and role distinction are separate. All four inks are strong on dark and compressed on bright/fog (1.2–2.2), where the black outline carries legibility. Under the corrected simulation, Sky keeps the larger Gold separation in every CVD row, and the gap over Mint is widest under protanopia/deuteranopia (D 1.166 vs 0.587). Do not claim color-only C provides the redundant cue available in A/E.

## 6. Working selection (Director-selected; not final validation)

- A: working design **A2 (compact upper Gold rule)**. Rationale: detectable without reading the first word; attached to the incoming block so SELF interposition cannot misattribute it; short 196x14 geometry avoids the topic-break reading; gold-on-gold keeps the palette intact with zero new-hue risk. Exact spec: §3 A2 row. Risk: at very small text scales the 14px rule is thinner than a dash stroke; native small-scale check pending.
- C: working design **C1 (Gold `#FFD700` / Sky `#33D6FF`)**. Rationale: largest measured Gold–third-hue separation overall and under every corrected simulated CVD (P 1.063 / D 1.166 / T 1.065 vs Mint 0.405 / 0.587 / 1.034); whole-turn single color keeps primary/secondary hierarchy clean. Exact spec: §3 C1 row; no marker. Risk: color-only channel carries no redundant cue; protan/deutan luminance deltas are small (0.016/0.218) — pair with the mandatory copy line that equal hues never imply the same person. C2 remains a tested fallback with identical semantics.
- E: working design **E1 (persistent Gold rule + Sky emphasis)**. Rationale: reuses the A2 boundary geometry with the most separable emphasis hue; the Gold rule persists after the Sky body expires to Gold, preserving a cue the original Cyan-only design lost. Exact spec: §3 E1 row; event-based expiry only. Risk: two simultaneous changes (marker + full-body recolor) is the most salient option; consecutive transitions produce back-to-back Sky bodies, each with its own Gold marker — correct per contract but visually busy; whether the Gold rule reads clearly against a Gold body at HMD scale is a pending native check.

Production policy: a shared semantic interpreter consumes bounded session-scoped, source-timed speaker evidence before renderer projection; renderers receive style and boundary fields, not raw provider speaker IDs. Unknown, overlapping, invalid, reordered, or incomparable evidence cannot confirm a change. Claims attach only on first readable translation; later revisions cannot replay a cue. C retains its hue through uncertainty when surviving content makes a reset misleading; a fresh empty context starts Gold. Entering E does not replay historical emphasis. The simple localized selector persists A/C/E; absent/invalid values resolve to A. These are disposable owner-comparison controls, not a permanent public preference design.

## 7. What was observed vs what is not claimed

Measurable/observable in this record: replay invariant outcomes, DOM color/marker attachment per event, screenshot pixels at browser scale, computed contrast/role numbers. Human interpretability (where did the speaker change, does equal hue imply identity, does no cue prove continuity, does the boundary read as topic split) and any HMD/SteamVR readability claim are not run. No flashing, prefixes, labels, or hue==person claims were introduced anywhere.

## 8. Build/file hashes (this worktree, 2026-09-23, post-sync revision)

- `scenario.json` 13606 bytes, sha256 `246a0672a758f52d83164ee7aa3ad12912d467ceb1ab97b15290ad0dcdadbf9b`
- `expected_results.json` 98867 bytes, sha256 `76d6683b7fe88fde9476cb733b7ce87c14558c8cd961c08d38f1d17a0df33929`
- `prototype.html` 20518 bytes, sha256 `cebf396ec929ae442f145a94dfcd373e4a262fd2045ecc3b8d8968d8d639e36b`
- `replay.py` 11301 bytes, sha256 `de28f8fa68c31e24e709c296b06c72d2e824bdb790612069cb61166a9a505cd9`
- `contrast_check.py` 4730 bytes, sha256 `4fc99f9ab3783aac98adbb4efedfe80b6a2991c6b4ea91222b1836f3664c749d`
- `contrast_analysis.json` 3019 bytes, sha256 `aef6c6d431c89d89bb13e62556cc36052977dea45e4784d27e26d76245f70957`
- Screenshots: 11 `.webp` files listed in §4.

## 9. Explicitly pending (not run / blocked / reserved)

- Actual Windows/SteamVR/HMD visual pass for working designs A2/C1/E1 and UI switching flow: NOT RUN (no device access from this worker). Includes Gold-rule-on-Gold-body legibility at HMD scale and small-text rule thinness.
- Live Soniox-to-presenter production plumbing: PASS for two sequential synthetic SAPI voices in one provider session; details in §11. Live overlapping speech was not exercised.
- Desktop compatibility: PASS for real settings controls and runtime renderer plans/surfaces; an actual Flet preview launch was attempted, but own-window capture failed (window lookup returned no handle). No desktop screenshot or pixel-level visual pass is claimed.
- Small engineering/owner comprehension evaluation (before/after explanation questions, distraction notes): NOT RUN; protocol proposed in §10.
- Production mode lifecycle, source/generation reset, overlapping synthetic evidence, settings persistence and native protocol checks: exercised as listed in §11. HMD clipping, gaze-away/return, distraction and actual UI-to-HMD switching remain unverified.
- Default (absent/invalid A), disposable selector scope, and any strategy elimination: Director-decided as noted in the header; final removal of disposable controls belongs to the owner before official release.

## 10. Suggested owner validation protocol (handoff, not executed)

For each of A2/C1/E1 on HMD plus baseline: show t03 boundary/toggle, t05 SELF interposition, t08 consecutive change, t09 unknown, t10 reset, t13 multilingual wrap, t14 short reply, each on dark/bright/busy. Ask before explanation: where did the speaker change; do equal hues imply the same person; does no cue prove continuity; does the marker read as topic split. Then explain the cue meaning once and re-ask. Record errors and distraction, not preference scores.

## 11. Production integration evidence

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
