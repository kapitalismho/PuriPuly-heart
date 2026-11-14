# Photo mask model delivery record

Execution date: 2026-11-13. This is a bounded publication and installation verification record, not the architecture map or an application release declaration.

## Published artifact

- Repository: [salee1233/mobilenetv4-text-seg-onnx](https://huggingface.co/salee1233/mobilenetv4-text-seg-onnx/tree/12c0ab260ba27a0bffb5d7a51ced06e36b04a1c0).
- Revision: `12c0ab260ba27a0bffb5d7a51ced06e36b04a1c0`.
- File: `mbnv4_text_seg.onnx`, 48,103,781 bytes.
- SHA-256: `bfe95d72db1eea72101074293c8d27b77eb87663fa48a23e8258e59249510bcf`.
- Only the existing ONNX, model card and attribution notice were uploaded. No training, new conversion, test-image or recognized-text publication occurred.
- Full anonymous downloads from Hugging Face, hf-mirror.com and the application's hf-mirror.net endpoint matched the local artifact. This hash comparison was development verification, not installer behavior.

## Installation behavior

[Photo provisioning](../installer/photo-model.iss) installs the required model at `{app}\assets\photo\mbnv4_text_seg.onnx` after CPU ASR provisioning. Chinese installer languages prefer hf-mirror.net; other languages prefer Hugging Face. Download failure or wrong size tries the other pinned source. Both failing blocks installation without replacing an existing model.

Setup checks model-file presence and exact size, not hashes. A matching installed file is reused; the existing AI-model re-download task forces replacement. Same-size corruption is not detected by this policy. The [photo process adapter](../src/puripuly_heart/core/photo/process.py) supplies installed model and canonical packaged ONNX Runtime paths before writing the child manifest. Source execution retains its existing native defaults. Attribution and license declarations are in the [product notices](../src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt).

The owner retained the existing legacy-cleanup ownership checks: hashes used to avoid deleting modified old application files are separate from model validation. Build-time packaged-metadata checks are also unchanged.

## Verification

- Three compiled Inno Setup 7.1.0 checks passed: full product installer compilation/metadata and English/Chinese provisioning behavior. Controlled local sources covered both-source failure, fallback, repair, reuse, forced replacement, preservation on size failure and same-size acceptance without hashing.
- Two photo-process lifecycle tests and scoped Ruff checks passed.
- The production photo installer include downloaded the actual public model into an isolated installation. Reinstall reused the file unchanged.
- The installed model and packaged ORT path processed all original 23 images / 229 fixed blocks, with no provider requests or source-image changes.
- The current Python process supervisor launched the installed native executable twice. Both launches rendered the original u01 photo's 12 blocks, acknowledged shutdown and exited with code 0. The second rendering was visually inspected; existing overlap and source-remnant defects remain.
- The product installer wizard displayed the 48.10 MB photo download on its Ready page with ASR provisioning isolated out. The preview was cancelled before installation.

## Limits

Native executable SHA-256: `e5db619de80998d48a454991e0b46dcf296ab9dc6138924ec7f3396f6eadc5bb`. This was the isolated executable used for the earlier original-23 verification, not a rebuild of concurrent memory/bbox work. The evidence covers installed-model/native compatibility, not a complete rebuilt application release, provider accuracy or physical headset legibility. Private images, text and rendered evidence remain unpublished. No GitHub issue update, application release or Relay deployment occurred.
