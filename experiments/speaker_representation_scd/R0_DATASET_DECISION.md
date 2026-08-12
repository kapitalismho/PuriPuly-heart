# R0 Legacy-Common-GT-Only Dataset Decision

## Status

- Experiment: `speaker_representation_scd_v1`
- Owner amendment date: `2026-08-11`
- Authority: `EXPERIMENT_PLAN.en.md`
- Current sequence: `R2-L -> reduced R3 -> reduced R4 -> candidate selection`
- Evidence level: exploratory and `development-known`
- New corpus acquisition: forbidden
- Model or detector training: forbidden

## 1. Decision

The current pre-training experiment uses only the data already used by the legacy ERes2NetV2 and
LS-EEND comparison. Zeroth-Korean, JVS, D5, and every other newly acquired public corpus are outside
the current executable scope.

```text
current experimental data
  exact legacy ERes/LS-EEND common-GT panel

not current inputs
  Zeroth-Korean
  JVS
  VoxConverse/AISHELL-4 or another D5 reserve
  private or newly recorded audio
```

Previously downloaded Zeroth bytes, partial JVS attempts, and their receipts are historical
provenance only. They must not be consumed, resumed, or deleted by this experiment, and their state
does not block R2-L.

## 2. Sole Experimental Dataset

| Item | Identity |
| --- | --- |
| Manifest | `experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json` |
| Byte SHA-256 | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` |
| Canonical content SHA-256 | `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68` |
| Total episodes | 804 |
| Diagnostic episodes | 695 |
| Source identities | 616 |
| Unique WAV bytes | 600 |
| Candidate inventory | 450 positive and 360 negative rows |
| Existing matched pairs | 313 |
| Pair rows SHA-256 | `fb29fff960932f2840433fa94f1a9e4bade167a6d935a6458dc6e9b191a4f9b9` |
| Phase-4 coordinate rows SHA-256 | `58cbd9eaf4554761bf71e698bc4b1f251ae722c4281be35d0270dbc0ab285470` |

The panel contains existing LibriSpeech-derived controlled episodes, AMI conversations, and
AliMeeting conversations. R2-L must revalidate all referenced WAV and annotation identities before
using them. It must not modify the legacy experiment's files, results, or caches.

## 3. Scientific Claim Boundary

All current rows have already been inspected through the legacy experiment. They are suitable for:

- paired representation comparison on identical audio and GT;
- layer/context screening;
- zero-shot detector development and candidate selection;
- comparison with already measured ERes-final and LS-EEND event outputs;
- error analysis for available overlap, backchannel, gap, stress, English, and Mandarin strata.

They are not suitable for a fresh confirmatory or final generalization claim. In particular, the
current study shall not claim Korean, Japanese, broad multilingual, true code-switch, whisper, or
unseen-corpus performance when the required rows are absent.

An untouched public validation/test decision is deferred until a learned SCD head is separately
approved. No future dataset needs to be chosen or acquired before current candidate selection.

## 4. Relationship to Existing ERes/LS-EEND Work

The experiment reuses the exact source timeline, GT events, episode identities, matching blocks,
and metric definitions. This supports a strict paired comparison because each new encoder sees the
same audio and event coordinates.

The following are reused only as already measured contextual baselines:

- ERes2NetV2 final-embedding event results;
- LS-EEND event results on the exact shared GT subset.

Neither model is rerun. Their thresholds, detector states, raw feature caches, and shortlist
conclusions are not imported into representation ranking. LS-EEND remains outside cosine/AUC/EER
tables because it produces speaker activity rather than a comparable representation.

## 5. Next Data Action: R2-L

R2-L performs only the following:

1. Revalidate the manifest, WAV, annotation, event, pair, and block identities.
2. Resolve the 600 existing unique WAV byte identities without downloading a corpus.
3. Freeze shared R3 anchors and an R4 source subset before observing new encoder scores.
4. Generate 100, 300, and 500 ms trailing-window coordinates at a 1,600-sample hop.
5. Produce a measured reduced R3/R4 wall-time and storage forecast.
6. Report exact R3/R4 inputs, configurations, commands, and cost to the owner and stop for approval.

R2-L performs no neural inference and does not authorize R3 or R4.
