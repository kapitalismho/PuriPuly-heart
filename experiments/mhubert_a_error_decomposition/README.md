# mHuBERT-A error decomposition

This no-training diagnostic uses only the existing `mhubert-147 / A-FROZEN-DIRECT` out-of-fold raw predictions from the issue #72 experiment.

It measures signed GT-to-peak timing error, promotions between the 100/250/500 ms collars, duplicate and GT-proximal false peaks, false peaks more than 500 ms from every GT event, GT-state-derived false-positive categories, and threshold-independent candidate coverage.

The operating threshold maximizes mean F1 across the three collars over the complete score range. FE/h does not select it. Candidate coverage is also reported with no score threshold, both before and after the established 200 ms duplicate suppression.

Laughter and prosody are not annotated in the frozen GT. The diagnostic therefore does not infer those categories from audio and retains representative timestamps for later listening if needed.

```powershell
$env:SRSCD_CACHE_ROOT = 'C:\Users\salee\AppData\Local\puripuly-heart-research\speaker_representation_scd_v1'
python -m experiments.mhubert_a_error_decomposition.analyze --output "$env:SRSCD_CACHE_ROOT\results\mhubert_a_error_decomposition_v1\analysis.json"
```

