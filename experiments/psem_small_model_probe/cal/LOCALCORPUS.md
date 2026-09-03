# Local corpus resolution (psem-corpus)

Corpus root (outside repo; no audio copied into the repo):

- `PSEM_CORPUS_ROOT=C:\Users\salee\.psem-corpus`
- `PSEM_REFERENCE_ROOT` unset (primary covers all 15 frozen sessions).

Layout under the root follows `cal/audio_resolve.py` conventions:
`<root>/<audio_ref>` with `ami/audio/<SID>/<SID>.Mix-Headset.wav` and
`alimeeting/far_ch0/<SID>.wav`.

## Session -> wav table (15/15 resolved, 0 missing)

All files verified: exist, mono 16 kHz int16, frame count equals the V2
`source_manifest.jsonl` `duration_samples`, and every episode's native (5 s)
+ causal (1 s) spans plus the session max `evaluation_end_ms` load via
`load_span` without overrun (`check_corpus.py`: 15/15 PASS).

| corpus | session | audio_ref | frames | V2 match | file sha256[:16] | span-check |
|---|---|---|---|---|---|---|
| ami | EN2004a | ami/audio/EN2004a/EN2004a.Mix-Headset.wav | 55130795 | yes | d188219aff66264a | PASS |
| ami | EN2006a | ami/audio/EN2006a/EN2006a.Mix-Headset.wav | 56407894 | yes | 4e10dd0209c661af | PASS |
| ami | EN2009d | ami/audio/EN2009d/EN2009d.Mix-Headset.wav | 85189974 | yes | eeb4a5ff47cadba8 | PASS |
| ami | ES2002b | ami/audio/ES2002b/ES2002b.Mix-Headset.wav | 36476075 | yes | 977fbf6cd473cfb1 | PASS |
| ami | ES2009a | ami/audio/ES2009a/ES2009a.Mix-Headset.wav | 22435328 | yes | 472adcae2cff535a | PASS |
| ami | ES2009b | ami/audio/ES2009b/ES2009b.Mix-Headset.wav | 22965248 | yes | 7bef2999aa6dc63c | PASS |
| ami | ES2009c | ami/audio/ES2009c/ES2009c.Mix-Headset.wav | 31310848 | yes | 350661f98c86a8a5 | PASS |
| ami | ES2009d | ami/audio/ES2009d/ES2009d.Mix-Headset.wav | 33839104 | yes | b5ef423361ce67c8 | PASS |
| alimeeting | R0004_M0012 | alimeeting/far_ch0/R0004_M0012.wav | 31514608 | yes | 23e93504d0f1b154 | PASS |
| alimeeting | R1019_M1928 | alimeeting/far_ch0/R1019_M1928.wav | 28771752 | yes | 0bbafa526c8c0b42 | PASS |
| alimeeting | R1019_M1960 | alimeeting/far_ch0/R1019_M1960.wav | 28686122 | yes | a842b222d0fda40b | PASS |
| alimeeting | R1021_M1940 | alimeeting/far_ch0/R1021_M1940.wav | 28329539 | yes | a70684449b115658 | PASS |
| alimeeting | R1021_M1944 | alimeeting/far_ch0/R1021_M1944.wav | 28113929 | yes | c96b364c2f0e6ee1 | PASS |
| alimeeting | R1021_M4073 | alimeeting/far_ch0/R1021_M4073.wav | 28426437 | yes | 01891117bc955ea0 | PASS |
| alimeeting | R1021_M4080 | alimeeting/far_ch0/R1021_M4080.wav | 28435200 | yes | d61134736b67d965 | PASS |

Missing sessions: none.

## Exact sources

- AMI (8 sessions): direct per-session download from the Edinburgh mirror
  recorded in V2 `source_manifest.jsonl` `audio_source_url`:
  `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/<SID>/audio/<SID>.Mix-Headset.wav`.
- AliMeeting (7 sessions): no per-file mirror exists (OpenSLR 119, ModelScope
  `modelscope/AliMeeting`, and HuggingFace mirrors only carry whole tarballs;
  Internet Archive has nothing). Stream-extracted the 8-channel source wavs
  from `https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz`
  (78,639,309,701 bytes) and derived `far_ch0` with the exact V2 rule
  (`alimeeting_train_materialization.py::_materialize_channel_zero`:
  8ch PCM16 16 kHz assert, channel 0 written mono, frame-count assert):
  R0004_M0012<-MS005, R1019_M1928<-MS110, R1019_M1960<-MS108,
  R1021_M1940<-MS116, R1021_M1944<-MS115, R1021_M4073<-MS104,
  R1021_M4080<-MS105. 8ch staging deleted after derivation.
- V2 `source_manifest.jsonl` carries no audio sha/size (`audio_sha256: null`),
  so identity is pinned by exact frame-count match + file sha256 above.

## Real-run env

```cmd
set PSEM_CORPUS_ROOT=C:\Users\salee\.psem-corpus
PYTHONPATH=. python experiments/psem_small_model_probe/cal/run_cal.py
```

Re-verify any time with:

```cmd
set PSEM_CORPUS_ROOT=C:\Users\salee\.psem-corpus
PYTHONPATH=. python experiments/psem_small_model_probe/cal/check_corpus.py
```
