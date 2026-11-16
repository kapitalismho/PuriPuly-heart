# Lossless WebP follow-up — issue 438

## Decision

Keep JPEG 85 as the production default. This lossless WebP configuration preserves the PNG pixels and reduces encoded bytes by 29.90% versus PNG, but remains 3.32 times the JPEG 85 size. There is no observed input-token saving. The selected Pillow/libwebp configuration adds substantial encoding time; that is not a conclusion about all WebP encoders or settings.

The owner asked to execute the recommended lossless WebP follow-up quickly. The same 25 owner-selected images and fixed provider settings were reused. This adds exactly 25 WebP requests and 25 AABB replays, without repeating the original JPEG/PNG or quad requests. All new requests completed and parsed; all new replay subprocesses exited zero. No retries or production changes were made. GitHub publication and issue closure remain held at the owner's prior request.

## Reproducible scope

- Base: `fd47717576448a44d15b42dc1e8857c8a5fa9e32`; script: `scripts/bench_photo_438_webp.py`.
- Executed script SHA-256: `334135679238757b8f11f0c0e509f70dc03306ecb24e572a29700afa89d199f0`.
- Encoder: Pillow 13.0.0 / libwebp 1.6.0, RGB, `lossless=True`, `method=4`, `quality=100`, `exact=True`. In lossless mode, quality controls compression effort, not JPEG-equivalent visual quality.
- Input: the existing RGB PNG of the shared resized canvas. No baseline JPEG was decoded; no capture, resize, letterbox, send dimensions, or working-image sizing changed.
- All 25 decoded WebP images exactly matched the corresponding PNG RGB bytes and dimensions. Per-image PNG, RGB, and WebP hashes are retained.
- Each request used `image/webp`, the existing OpenAI API route, model `gpt-6-luna`, target `ko`, reasoning `none`, `store=false`, streaming, and 8192 maximum output tokens. Local proof compared non-image request parameters and mappings with the existing arm.
- Concurrency limit three after the first usability request, which is included in the 25. No automatic retry.
- Native executable SHA-256 was checked before replay: `8e4dd8add4c07d7743cc83538096528ab2973f8de7161a772bb2083fbcb75012`. Existing per-image model/runtime/font and AABB replay settings were reused.
- Existing experiment helpers provide transport capture, explicit final-answer selection, actual usage, parsing, and native process execution. No native rebuild or production provider-contract change was needed.

The earlier comparison remains documented in [the original report](issue-438-report.md). The follow-up deliberately retains that pinned experimental baseline; it is not a newly synchronized production/release candidate.

## Measurements

Old arms are reused observations, not reruns. Bytes are decimal. Each column contains 25 images.

| Measurement | JPEG 85 | JPEG 95 | RGB PNG | Lossless WebP |
|---|---:|---:|---:|---:|
| Encoded bytes, total | 5,928,704 | 10,557,538 | 28,106,723 | 19,703,238 |
| Request JSON bytes, total | 7,961,822 | 14,133,602 | 37,532,493 | 26,327,870 |
| Encoded bytes, mean | 237,148 | 422,302 | 1,124,269 | 788,130 |
| Encode time, median ms | 12.965 | 16.556 | 4.726 | 487.483 |
| API completion, median ms | 5,056.070 | 4,893.382 | 6,383.154 | 6,402.304 |
| Replay process, median ms | 1,052.774 | 1,049.285 | 1,039.588 | 1,077.594 |
| Composed serial-stage estimate, median ms | 6,313.276 | 6,039.956 | 7,317.718 | 8,267.373 |

WebP's cached PNG load took a median 11.500 ms, separately from encoding. Its composed preparation estimate is 618.852 ms: prior native source-read/decode/resize time plus the new measured WebP encode time. It does not describe the actual cached-input preparation path. The composed serial estimate adds that estimate, actual API completion time, and native process elapsed time. These are not observed continuous capture-to-display timings; Python pre-request file/base64 setup, analysis, and scheduling gaps are excluded.

The old JPEG/PNG timings use the native Rust encoders; WebP uses Pillow/libwebp. The WebP batch ran later, without interleaved repetitions. Service load, output length, and timing variability remain confounders. The similar PNG/WebP request medians do not establish speed equivalence or a causal format effect. No Rust WebP performance or alternative encoder-effort result is claimed.

### Actual provider usage

| Reported tokens, 25-image total | JPEG 85 | JPEG 95 | RGB PNG | Lossless WebP |
|---|---:|---:|---:|---:|
| Input | 37,550 | 37,550 | 37,550 | 37,550 |
| Output | 15,862 | 14,962 | 15,453 | 15,424 |
| Total | 53,412 | 52,512 | 53,003 | 52,974 |

All 25 WebP calls reported 1,502 input tokens each. Cached input and reasoning subsets were zero; cache-write tokens totaled 37,475 in the 30-minute subset, with zero in the 12-hour subset. These are retained subsets, never additional totals. Image-specific usage was unavailable. The 25 follow-up calls consumed 52,974 reported tokens; no old-call usage is charged again in this count.

[Per-image WebP measurements](issue-438-webp-measurements.csv) contain the new 25 rows. Empty image-specific-token and observed-continuous-end-to-end columns mean unavailable, not zero. Old per-image rows remain in [the original CSV](issue-438-measurements.csv).

## Quality interpretation and examples

Exact transmitted RGB parity with PNG was verified for every image. This establishes losslessness, not identical model outputs or better recognition. The provider may generate different grouping, words, boxes, and line counts on separate requests. No full-corpus qualitative scoring pass was repeated for this quick follow-up.

Selected direct inspections:

- Image 01: schedule text is readable, but stylized-logo recognition, broad erasure artifacts, small overlapping labels, and residual source strokes remain. This is not a general recognition/masking fix.
- Image 09: headline and author lines remain readable in both lossless arms; smaller labels and placement vary, and dark erasure artifacts remain in both.
- Image 19: the Director observed oversized and overlapping WebP comic-panel text, whereas the earlier PNG composite was more usable. Pixel-identical transmission did not guarantee the same downstream layout.
- Image 25: the Director observed a dangling source character beside the translated explanatory text in WebP; the previously inspected PNG result avoided that residue.

The last two are additional selected visual observations, not evidence that WebP compression caused regressions. There is no measured quality advantage supporting a default switch. Lossy WebP, alternative effort levels, native Rust WebP encoding, and interleaved repeated trials were not evaluated.

## HTML delivery repair

The owner reported that images and boxes did not appear through the original HTML delivery. This was accepted as a failure; prior verification through an absolute file URL did not prove the owner's preview route.

The private viewer now supplies a local HTTP delivery address, anchors the source stage to its intrinsic image aspect ratio, hides geometry until its source image loads, and displays actionable image-load errors instead of detached boxes. The existing 25 images and 419 corner predictions are unchanged.

Current local URL: `http://127.0.0.1:8438/quad-comparison.html`. The named `photo-438-comparison-server` binds loopback only and is intentionally left running for owner viewing. The HTML still depends on its existing adjacent image tree; copying the HTML alone is not a portable package. If the process stops, the address is unavailable until the local server is restarted.

Actual HTTP browser proof covered all 25 image selections: images loaded and image/SVG displayed rectangles aligned. A browser-only blocked source request produced an explicit error and hidden overlay; removing the block restored normal loading and alignment. No provider call was involved. The controlled failure check initially waited for a different Korean phrase; the actual alert was inspected and retained. This was an observation-selector error, not an application failure.

Repaired HTML SHA-256: `48ddcc2836c099078bcc8dc8769cc722fa5a184a7468d0c8a165eee4e13835bd`. Private proof: `build/issue-438/quad-html-http-verification.json`.

## Evidence and verification

Private root: `build/issue-438-webp/`.

- `preparation-summary.json`: 25 exact-pixel proofs, encoder options, source/output hashes and timing boundaries.
- `provider-configuration.json`: fixed request configuration and source/executable provenance.
- `measurements.json`, `attempts.jsonl`: actual requests, usage, parsing, and replay measurements.
- `provider-execution-report.json`: commands, outcomes, comparisons, selected visual observations, and limits.
- `image-NN/requests/webp/attempt-01/`: actual body, raw SSE, terminal response, final text, parsed blocks, measurement.
- `image-NN/replay/webp/`: job, process result, logs, masks, erased and final images, metadata.

Executed commands used the existing repository interpreter and `scripts/bench_photo_438_webp.py --phase prepare`, `--phase provider --director-go --concurrency 3`, and `--phase replay`. The narrow feature proof was the 25 pixel/MIME/settings/mapping checks plus actual provider and native execution. No permanent tests, broad suites, or unsupported type-check claim were added.

No public image, original OCR/translation, personal path, credential, or request body is included in the maintained CSV/report. Raw evidence and the HTML remain local. Production adoption, publication, and closure remain separate owner decisions.
