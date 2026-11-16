# Photo encoding and quad ownership experiment — issue 438

## Decision

Retain JPEG quality 85 and the current rotation-capable AABB pipeline. Higher-quality transmission produced individual wins but no consistent practical improvement on this corpus. Automatic quad ownership produced three localized wins, twenty practical ties, and two mixed tradeoffs. Do not adopt the experimental candidate generally on this evidence.

This is an experiment result, not a production rollout. No settings, production defaults, input resolution, or live provider contracts were changed. Perspective compositing, manually corrected corners, and a thickness-only control were not evaluated. There is no evidence here to recommend adopting a thickness-only change.

The owner requested a private explanation before GitHub publication. Results publication, issue closure, Project completion, push, PR, and deployment remain unperformed and outside the current delivery authorization. The issue's Project item was moved to In progress when work began.

## Authority, corpus, and configuration

- Authority: [issue 438](https://github.com/kapitalismho/PuriPuly-heart/issues/438).
- The owner selected their November VRChat screenshot folder before implementation or execution, then explicitly increased the comparison to 25 images.
- One fixed representative subset covers 11 landscape and 14 portrait images: large titles, small explanatory text, vertical writing, comics, and tilted posters. Both experiments use the same 25 images.
- Private `build/issue-438/corpus.json` records exact selected filenames, original paths, and SHA-256 hashes. The sanitized [measurement CSV](issue-438-measurements.csv) records each image ID and source hash without personal paths or image content.
- Baseline: `850d8b4ddab0436b62513efdc81758c1d655f029`.
- Initial experiment checkpoint: `98d8f22f94afcdfab0a7c409cfb2ef85f66433ef`.
- Phase-corrected code checkpoint: `6e1a3a5b4d9389e32f16b6cfc13f53ca023e4a74`. Recorded source hashes establish applicability of captures made before the commits were labeled.
- Candidate executable SHA-256: `8e4dd8add4c07d7743cc83538096528ab2973f8de7161a772bb2083fbcb75012`.
- Windows x64; Python 3.14.10; rebuilt native release executable with the opt-in `photo-experiment` Cargo feature.
- Provider: owner-authorized direct OpenAI API, `https://api.openai.com/v1/responses`; requested and returned model `gpt-6-luna`; target `ko`.
- Fixed request settings: `reasoning.effort=none`, `store=false`, `stream=true`, `max_output_tokens=8192`; other model settings left unspecified consistently. No automatic retries.
- Maximum request concurrency three, except the first single-request usability check. Encoding order rotates by image; geometry extraction follows that image's compression requests.
- The normal Managed route remains selected in product settings. Its hosted JPEG-only/fixed-schema contract could not exercise PNG or quad output without deployment, so the owner explicitly authorized the billable direct API route instead. No managed credentials were borrowed.
- Native segmentation asset SHA-256: `bfe95d72db1eea72101074293c8d27b77eb87663fa48a23e8258e59249510bcf`. Font: bundled Noto Sans CJK KR Medium. Existing ONNX segmentation, Telea inpainting, and GPU composition were exercised.

## Execution and measurement boundaries

Completed 25 native preparations, 100 actual provider requests, and 100 native replay subprocesses with exit code zero. There were no provider retries. All raw responses and actual provider usage are retained privately. Total reported usage across both experiments was 222,608 tokens.

Transmission remains 1280×720 or 720×1280. All three encodings are made once from the same uncompressed resized RGB canvas using the production Lanczos3/letterbox helper. JPEG 85 and 95 use 4:4:4 sampling; PNG is RGB8 without palette quantization. Outgoing MIME matches the encoding. The independently prepared local working image retains its existing sizing and orientation behavior; it is not decoded from the transmission JPEG. No input-resolution comparison was made.

The request-size field is the exact JSON body size, excluding HTTP headers, framing, and TLS. API completion runs from the adapter's monotonic request start to the completed response, separately from transport close. Preparation, requests, and replays ran in separate phases. The composed serial-stage estimate adds measured preparation, request completion, and replay process time; B also adds its separate quad request. It excludes Python file/base64 setup, client construction, analysis/cleanup, and batch scheduling gaps. It is **not observed continuous end-to-end latency**. Live capture-to-display, headset presentation, and production streaming latency are unavailable.

Native replay process time includes model startup, segmentation, inpainting, typesetting, GPU composition, and artifact writes. Per-stage times remain separately available. These measurements do not establish pure algorithm overhead or repeatable performance differences.

## A. Transmission results

Bytes use decimal units. Each arm has 25 successful requests and 25 completed replays.

| Measurement | JPEG 85 | JPEG 95 | RGB PNG |
|---|---:|---:|---:|
| Total encoded bytes | 5,928,704 | 10,557,538 | 28,106,723 |
| Total request JSON bytes | 7,961,822 | 14,133,602 | 37,532,493 |
| Mean encoding time, ms | 13.391 | 16.599 | 4.646 |
| Mean preparation time, ms | 136.652 | 139.859 | 127.907 |
| Mean API completion time, ms | 6,087.004 | 5,945.102 | 6,586.019 |
| Median API completion time, ms | 5,056.070 | 4,893.382 | 6,383.154 |
| Median first-block time, ms | 1,503.508 | 1,391.953 | 1,957.279 |
| Median replay process time, ms | 1,052.774 | 1,049.285 | 1,039.588 |
| Median composed serial-stage estimate, ms | 6,313.276 | 6,039.956 | 7,317.718 |

JPEG 95 transmits about 1.78 times the encoded bytes of JPEG 85; PNG about 4.74 times. PNG's fast local encoding does not eliminate its larger transfer. The small observed JPEG 95 latency advantage is not a demonstrated encoding speedup: one generation per image/arm, changing output lengths, request concurrency, and ordering remain confounders.

### Actual token usage

| Reported field, 25-image sum | JPEG 85 | JPEG 95 | RGB PNG |
|---|---:|---:|---:|
| Input | 37,550 | 37,550 | 37,550 |
| Output | 15,862 | 14,962 | 15,453 |
| Total | 53,412 | 52,512 | 53,003 |
| Cached input | 0 | 0 | 0 |
| Cache-write subset | 37,475 | 37,475 | 37,475 |
| Reasoning subset | 0 | 0 | 0 |

Every individual A request reported 1,502 input tokens. Equality is an observation, not an inference from equal resolution. Cache-write tokens were reported in the 30-minute subset; the 12-hour subset was zero. These subsets are not added to input or total usage. Image-specific token usage was not exposed and remains unavailable. Output-token differences reflect different generated results; no fixed token saving from a format is established.

The [measurement CSV](issue-438-measurements.csv) includes every image/arm's actual usage, cache details, bytes, timings, replay state, and source hash. Empty fields mean unavailable or not applicable, never assumed zero. In particular, image-specific tokens and observed continuous end-to-end time are unavailable; quad extraction has no photo-protocol first-block metric.

### Paired visual assessment

All 25 source canvases and all 75 composites were inspected, with targeted original-resolution, mask, erased-image, and composite closeups. Recognition/localization checks covered missed text, obvious transcription mistakes, physical-line counts, clipping, and neighboring text. Downstream checks covered remaining source strokes, background damage, readability, and layout.

Practical preferences were JPEG 85 on six images, JPEG 95 on four, PNG on two, four ties, and nine mixed outcomes. This is a bounded visual judgment, not an OCR score or causal compression estimate. Returned grouping, wording, translations, and line counts vary between A arms.

The [quality CSV](issue-438-quality.csv) records affected-image/category lower-bound flags for every arm. `1` means at least one confirmed defect in inspected regions; `0` means none confirmed there, not certified absence; an empty cell means insufficient evidence. Do not add category flags into a score: severity and unassessed categories differ. Tiny peripheral lettering was not exhaustively labeled. Quad recognition flags inherit the fixed JPEG 85 observations; they are not independent quad OCR results.

Representative observations:

| Image | Observed result |
|---|---|
| 02 | JPEG 95 preserves the decorative logo and translates the small subtitle; JPEG 85 creates giant overlapping Korean columns and damages neighboring artwork. PNG introduces another unnecessary logo translation. |
| 13 | PNG mislocalizes a vertical quote onto the character, damaging artwork and placing oversized text there. Lossless pixels do not prevent a wrong box. |
| 19 | JPEG 95 produces severe oversized/overlapping comic-panel text; JPEG 85 and PNG are substantially more usable. |
| 25 | PNG improves small explanatory text coverage and avoids the dangling source character visible in the JPEG outputs. |

These opposite-direction examples support retaining JPEG 85 rather than changing defaults on a single-generation comparison.

## B. Genuine quad ownership results

One additional JPEG 85 request per image asks for ordered four-point geometry for the canonical baseline block IDs. The prompt explicitly asks the model to inspect visible direction/extent rather than convert each bbox to corners. The exact prompt/schema are retained in `provider-configuration.json` and the experimental script. Coordinates are send-image pixels, transformed back through the existing scale/offset into original-photo pixels. Raw corner predictions and overlays remain available, including bad predictions.

All 25 final answers supplied validated geometry for all 419 canonical blocks. Of these, 383 non-kept blocks actually used quad ownership; 36 retained their original content under the existing keep-original rule. No partial-AABB or mixed-fallback replay was used. Canonical source, translation, order, bboxes, and physical-line counts were identical between each baseline/quad pair.

The candidate changes convex pixel-center membership for both core-mask seeding and recovered-line ownership. It retains segmented foreground, component recovery, dilation, inpainting, and existing mask-derived rotation/thickness/typesetting. It neither fills the polygon nor immediately discards geometry into an AABB. Bbox-derived padded ROI and glyph sizing remain fixed to isolate this operation.

Observed paired outcomes: twenty practical ties, three localized quad wins, and two mixed tradeoffs. This is not a percentage improvement in OCR or masking quality.

| Image | Fixed-text geometry observation |
|---|---|
| 08 | Small bubble font increases from 19.2 to 25.42 working pixels; visibly more readable, while neighboring problems remain. |
| 14 | A title that wraps across the subtitle becomes a readable slanted single line with unchanged wording. |
| 17 | Adjacent V logos survive instead of being erased. Damage to neighboring Booth artwork and other placement problems persists. |
| 06 | One author/title collision improves, but a shallow subtitle quad newly leaves visible blue source characters. A genuine regression accompanies the gain. |
| 05 | Incorrect corners cut through the broad slogan; the recovered mask remains identical to baseline in the inspected region and source strokes remain. |

Two automatic quads extend beyond the unchanged padded ROI: image 05 block 0 and image 20 block 7. Coverage outside that ROI is unevaluated. Other retained bad examples include clipped subtitle/ending/punctuation regions and broad neighboring-text ownership. Valid convex corners are not necessarily accurate corners.

### Acquisition and processing cost

The separate geometry requests reported 52,442 input, 11,239 output, and 63,681 total tokens. Cached-input and reasoning subsets were zero. Median geometry request completion was 4,685.408 ms. This is the cost of this fixed-text diagnostic acquisition strategy, not a claim that a future combined translation/quad schema must require a second request.

Median local replay process time was 1,096.183 ms for quads versus 1,052.774 ms for the fixed JPEG 85 baseline. Mean mask-stage sums per image were 566.464 ms versus 566.602 ms; erase/type-layout stage sums were also of similar scale. Single runs and artifact writing prevent a precise overhead claim. Median composed serial-stage estimate, including canonical OCR plus separate geometry acquisition, was 11,309.031 ms; it is not production end-to-end latency.

Implementation cost is additional feature-gated sidecar validation, coordinate mapping, ownership threading, diagnostics, and offline artifact support, plus the bounded provider adapter. No live protocol or settings UI was added. The observed benefits do not justify promoting this maintenance surface into production yet.

Perspective rectification/warping was deliberately not implemented. Existing mask rotation remains active in both arms. Region-selection improvements do not establish perspective-compositing quality. No manual-corner upper bound or separate thickness-only control was run.

## Phase-handling repair and verification

One B response contained an assistant commentary message followed by an explicit `final_answer`. Concatenating both phases incorrectly made valid final JSON look malformed. Independent whole-checkpoint review identified this as adapter defect R1, not a bad geometry prediction.

The repair selects exactly one explicit final-answer message consistently for A/B and rejects missing/ambiguous finals. It never chooses an arbitrary last valid JSON fragment. Raw commentary, SSE, usage, and the superseded failure remain preserved. Re-extracting all 100 captured terminal responses changed no A result, no other B result, and no usage record. It recovered image 01's nine quads without another API call or secret access. Only its previously missing replay was then executed. The independent reviewer verified closure at the phase-corrected checkpoint.

Executed checks:

- Default and `photo-experiment` native release builds passed.
- 39 focused native tests passed; four existing external-fixture tests remained ignored.
- 18 focused Python tests passed, including 14 new phase-regression cases and four existing provider cases.
- Actual native preparation on an owner image confirmed dimensions, both JPEG 4:4:4 headers, RGB PNG without palette, and source identity.
- All 100 actual requests completed and all 100 replay subprocesses exited zero; all saved composites were visually assessed.

The first native build target encountered MSBuild `FTK1011` due to a long intermediate path. A short isolated target resolved it. Five existing native warnings remain. An initial synthetic adapter-proof fixture was excluded by the `image-*` discovery glob; the corrected fixture passed without a product change. These initial failures are retained in the local evidence rather than erased.

## Local evidence index and limits

Private evidence lives under `build/issue-438/`; original images, base64 request bodies, full OCR/translations, credentials, and personal paths are not included in this report or its CSVs. Credentials were never retained in request artifacts.

| Evidence | Local path |
|---|---|
| Corpus and configuration | `corpus.json`, `provider-configuration.json` |
| Source/executable identity | `run-provenance.json`, `phase-repair-provenance.json` |
| Provider measurements | `measurements.json`, `provider-analysis.json` |
| Downstream timings and invariants | `replay-analysis.json`, `quad-roi-diagnostics.json` |
| Phase repair and tests | `phase-reextraction.json`, `phase-repair-pytest.txt` |
| Per-request originals | `image-NN/requests/ARM/attempt-01/` |
| Masks, erased images, composites, layout metadata | `image-NN/replay/ARM/` |
| All visual observations | `quality-first/observations.json`, `quality-second/observations.json` |
| JPEG 95 win | `quality-first/image-02-logo-detail.png` |
| Quad gain and regression | `quality-first/image-08-bottom-readability.png`, `quality-first/image-06-subtitle-placement.png` |
| Fixed-text title placement | `quality-second/image-14-title-closeup.png` |
| Neighboring-logo preservation | `quality-second/image-17-icons-erased.png` |
| PNG small-text improvement | `quality-second/image-25-latency-closeup.png` |

The representative outputs are local-only, not publicly accessible evidence. This report and CSVs allow measurement/result inspection without publishing the owner-selected images. They do not make the private corpus externally reproducible.

Protected README/architecture/privacy/CLI/OSC/release documents were left unchanged. Production contracts and defaults remain unchanged; feature-only commands and evidence are documented here rather than added to those protected guides.

The decision has sufficient bounded evidence without additional generations: keep JPEG 85 and current geometry, record localized quad opportunities and failures, and treat any production adoption as a separate decision.
