# Notes on this replay

Sources: FLEURS run `20261020T055613Z` and conversation run `20261020T070225Z`, all four ASR providers, four decision models, 24 streams per model.

## Result

- No decision model policy came close to the punctuation rule. Conversation data:

  | policy | sentence | mid | recall | latency p50 / p90 |
  | --- | --- | --- | --- | --- |
  | punct rule | 82% | 11% | 89% | 1.00 / 3.35 s |
  | pplx-decider-v1.2, immediate p>=0.5 | 52% | 35% | 48% | 2.82 / 11.17 s |
  | pplx-decider-v1.2, verify v>=0.5 | 81% | 8% | 35% | 6.35 / 17.81 s |
  | pplx-decider-v1.2, verify v>=0.5 +silence | 69% | 19% | 41% | 4.10 / 13.71 s |
  | natural final | 46% | 44% | 29% | 5.07 / 17.85 s |

- Other models were worse than pplx-decider-v1.2 overall.
- Round trips from the replay machine were 0.38-0.57 s (p50). Cost was $0.017-0.158 per audio hour. Flash needed 152 retries.

## Diagnosis from the call logs

The analysis scripts are not committed. The numbers come from `calls/*.jsonl.gz` with the labels and alignment in `decide.py`.

1. The proposal question is the bottleneck. When a labeled sentence end was offered as a marker that already had words after it, the models chose it in this share of calls (conversation / read speech):
   - pplx-decider-v1.2: 33% / 43%.
   - Mercury: 40% / 43%.
   - Jev: 6% / 9%.
   - Flash: 5% / 13%.

   Most other answers were `none`. Jev and Flash mostly judge only whether the transcript end is a boundary.
2. Replay design flaw: each proposal marks only words added since the previous proposal, and the previous transcript end is not marked again. A boundary first offered at the transcript end, without right context, is never offered again. With pplx-decider-v1.2, 12-25% of conversation sentence labels were offered only that way (Soniox 25%, Deepgram 15%, Gemini and Qwen 12%). The `--fake-decider` oracle did not expose this because it always accepts the boundary at the transcript end.
3. The right-context check discriminates boundaries. Verify-call AUC against the labels (read / conversation):

   | model | AUC read / conversation | rates at v>=0.5 (conversation) |
   | --- | --- | --- |
   | pplx-decider-v1.2 | 0.87 / 0.88 | TPR 0.71, FPR 0.12 |
   | Mercury | 0.83 / 0.81 | answers mostly near 0 or 1 |
   | Jev | 0.77 / 0.79 | - |
   | Flash | 0.88 / 0.76 | - |

   True boundaries get low probabilities (pplx-decider-v1.2 mean 0.62-0.65), so 0.5 is a strict threshold.
4. Labels are noisy, so these values are lower bounds. Examples from sampled disagreements:
   - The model said 0.98 for "where people can find you online | yeah absolutely", and there is no label.
   - Labels within the 1-character slack can cut a word in Korean and Japanese ("무슨 말이에요 이 | 렇게").
   - Soniox commas inside lists ("content and stories | audio dramas") count as clause labels.
5. A right-context check has a latency floor. Partial text first shows 3 English, 2 Korean or 3 Japanese characters after a labeled sentence end. Adding a 0.57 s round trip gives this p50:
   - Soniox, Deepgram and Gemini: 2.0-2.3 s.
   - Qwen Audio: 2.7-3.2 s.

   Under the same metric, the punctuation rule's p50 is 0.5-1.4 s for Soniox, Deepgram and Gemini.

## Punctuation rule per provider (conversation)

Deepgram, Gemini and Qwen Audio punctuation is independent of the Soniox labels. Soniox rows are circular.

| provider | recall | mid | latency p50 |
| --- | --- | --- | --- |
| Deepgram | 91-98% | 7-24% | 0.56-0.88 s |
| Gemini | 77-91% | 17-19% | 0.46-0.96 s |
| Qwen Audio | 65-91% | 5-23% | 2.5-2.9 s |

## Implication for the next replay

- Ask the binary right-context question at every word boundary once enough right context has arrived, instead of the marker choice. This removes the never-re-asked flaw.
- Expect a p50 floor near 2 s.
- Limit the models to pplx-decider-v1.2 and Mercury.
