# 화자 변화 실험 중지 요약

## 상태

2026-08-17 사용자 결정에 따라 이 실험은 중지한다.

앞으로 이 실험 범위에서는 다음 작업을 하지 않는다.

- 12-arm clean regeneration
- feature 재추출
- 추가 학습
- adapter 강화
- structured formulation 수정
- localization, hard-negative, fine-tuning, scratch 후속 실험
- 추가 오류 분석

현재 결과는 기존 out-of-fold raw prediction을 전체 score 범위에서 재평가한 **development-known provisional evidence**다. 깨끗한 최종 커밋에서 12개 arm을 다시 생성하는 공식 재현은 시작하지 않았으며, 독립 review도 완료하지 않았다.

## 원래 알고 싶었던 것과 실제 실험의 차이

상위 연구 질문은 다음과 같았다.

> 화자 변화 모델을 pretrained encoder fine-tuning으로 갈 것인가, 아니면 scratch 학습으로 갈 것인가?

이번 실험은 그 질문을 답하지 못했다.

실제로 비교한 것은 고정된 pretrained embedding 위에서 direct target, output adapter, R7-B-style structured target의 차이였다.

```text
audio
  -> pretrained encoder
  -> 고정 embedding
  -> task head 또는 output adapter + task head
```

모든 arm에서 pretrained encoder parameter는 고정됐다. `B-TRAINABLE-DIRECT`와 `D-TRAINABLE-STATE`라는 이름의 `TRAINABLE`은 encoder fine-tuning을 뜻하지 않는다. 이 두 arm은 encoder 출력 뒤에 붙인 64차원 residual bottleneck adapter와 task head만 학습했다.

```text
pretrained encoder parameters trainable = 0
```

Scratch/random-init 모델은 실행하지 않았다.

따라서 이 실험으로 다음을 판단할 수 없다.

- 실제 encoder fine-tuning의 효과
- partial fine-tuning과 full fine-tuning의 차이
- pretrained 모델과 scratch 모델의 차이
- fine-tuning과 scratch 중 최종 방향

## 실제로 수행한 실험

### 모델

- `eres2netv2-standard-prepool`
- `wavlm-base-plus`
- `mhubert-147`

### Arm

| Arm | 실제 의미 | 학습된 부분 |
| --- | --- | --- |
| `A-FROZEN-DIRECT` | 고정 pretrained embedding에서 speaker-change event 직접 예측 | direct temporal/task head |
| `B-TRAINABLE-DIRECT` | 고정 pretrained embedding 뒤 output adapter를 붙이고 event 직접 예측 | output adapter + direct head |
| `C-FROZEN-STATE` | 고정 pretrained embedding에서 state/relation을 예측하고 event로 변환 | state/relation head |
| `D-TRAINABLE-STATE` | 고정 pretrained embedding 뒤 output adapter를 붙이고 state/relation을 예측한 뒤 event로 변환 | output adapter + state/relation head |

세 모델에 네 arm을 적용해 총 12개 arm을 비교했다.

모든 arm은 동일한 10개 자연 연속 회의, 5개 out-of-fold split, source-time grid, context, event 정의, duplicate suppression, one-to-one matching을 사용했다.

### 평가 정정

초기 보고는 10 FE/h와 250 ms collar를 사실상 대표 operating point처럼 사용했다. 이 선택은 precision이 높고 recall이 극단적으로 낮은 일부 threshold를 과대평가했다.

최종 재평가는 다음 원칙을 사용했다.

- 전체 prediction score 범위를 끝까지 평가
- 100/250/500 ms collar를 병렬로 제시
- 각 arm에서 세 collar F1 평균이 최대인 threshold를 사용
- FE/h는 참고 annotation과 호환성 표로만 유지
- FE/h를 제품 정책이나 threshold 선택 제약으로 사용하지 않음

## 12-arm 결과

아래 `Macro F1`은 각 arm의 최적 operating point에서 100/250/500 ms F1의 평균이다.

| 모델 | A: Frozen Direct | B: Output-adapter Direct | C: Frozen Structured | D: Output-adapter Structured |
| --- | ---: | ---: | ---: | ---: |
| ERes2NetV2 | 37.34% | **38.32%** | 28.88% | 28.89% |
| WavLM Base+ | **41.59%** | 41.31% | 30.23% | 30.51% |
| mHuBERT-147 | **44.31%** | 44.30% | 34.38% | 34.18% |

관측된 최고 arm은 `mhubert-147 / A-FROZEN-DIRECT`였다.

| Collar | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 100 ms | 29.00% | 44.21% | 35.02% |
| 250 ms | 38.21% | 58.26% | 46.15% |
| 500 ms | 42.84% | 65.32% | 51.75% |

이 결과의 정확한 표현은 다음과 같다.

> 시험한 고정 pretrained-feature 파이프라인 중에서는 mHuBERT-147 + direct head가 가장 좋았다.

이 결과를 다음처럼 표현하면 안 된다.

> mHuBERT fine-tuning이 가장 좋았다.

mHuBERT encoder를 fine-tune하지 않았기 때문이다.

## Output adapter 결과

Direct arm에서 A에서 B로 바뀐 Macro F1은 다음과 같다.

| 모델 | A | B | B−A |
| --- | ---: | ---: | ---: |
| ERes2NetV2 | 37.34% | 38.32% | +0.98%p |
| WavLM Base+ | 41.59% | 41.31% | −0.28%p |
| mHuBERT-147 | 44.31% | 44.30% | −0.01%p |

현재 output adapter는 세 모델에서 일관된 개선을 만들지 못했다.

이 결과가 지지하는 결론:

> 현재 post-encoder residual bottleneck adapter는 도움이 되지 않았다.

이 결과가 지지하지 않는 결론:

> Encoder fine-tuning은 도움이 되지 않는다.

실제 encoder fine-tuning을 실행하지 않았기 때문이다.

## Structured 결과

현재 R7-B-style structured arm은 모든 모델에서 direct arm보다 최종 event Macro F1이 낮았다.

| 비교 | ERes2NetV2 | WavLM Base+ | mHuBERT-147 |
| --- | ---: | ---: | ---: |
| A→C: frozen에서 structured end-to-end 효과 | −8.46%p | −11.35%p | −9.93%p |
| B→D: output adapter에서 structured end-to-end 효과 | −9.43%p | −10.80%p | −10.11%p |

이 결과가 지지하는 결론:

> 현재 구현된 R7-B-style state/relation-to-event 파이프라인은 direct event head보다 나빴다.

이 결과가 지지하지 않는 결론:

> Structured representation 자체가 나쁘다.

Structured arm은 다음 두 단계를 함께 포함한다.

```text
audio -> state/relation prediction -> speaker-change event projection
```

내부 state/relation 품질과 event projection 손실이 최종 event 결과에 섞여 있다.

Structured 진단은 완전한 학습 실패와 다른 모습을 보였다.

- state macro-F1: 69.61~73.69%
- decoder relation balanced accuracy: 74.47~83.81%
- decoder different-speaker recall: 50.49~68.99%

반면 relation 경로별 편차가 컸고, 곱셈식 event projection은 여러 component probability를 곱해 score와 recall을 낮출 수 있다. 따라서 structured arm의 최종 열세를 structured representation 자체의 실패로 분리할 수 없다.

## mHuBERT-A 오류 분석

추가 학습 없이 기존 `mhubert-147 / A-FROZEN-DIRECT` raw prediction만 분석했다.

### Timing/localization

- 100 ms에서 FP인 prediction 1,040개가 500 ms에서는 TP가 됨
- 100 ms에서 놓친 GT 1,023개가 500 ms에서는 검출됨
- 500 ms TP의 절대 시간오차 중앙값: 60 ms
- 500 ms TP의 절대 시간오차 p90: 330 ms
- signed error 평균: −3.9 ms
- signed error 중앙값: 0 ms

Timing 문제는 실제로 존재하지만, 모든 prediction을 한 방향으로 이동해 고칠 수 있는 고정 latency offset은 아니었다.

### Remote false positives

500 ms collar에서도 FP 4,025개가 남았다.

- 모든 GT에서 500 ms 이상 떨어진 remote FP: 3,845개, 95.5%
- GT 500 ms 이내의 duplicate/proximal FP: 180개, 4.5%

Remote FP의 GT state 분류:

| 분류 | 개수 | Remote FP 비율 |
| --- | ---: | ---: |
| Continuous same-speaker singleton | 2,744 | 71.4% |
| Overlap continuation | 711 | 18.5% |
| Same-speaker pause/resume | 177 | 4.6% |
| Silence continuation | 170 | 4.4% |
| Overlap end | 34 | 0.9% |

따라서 timing만이 아니라 same-speaker speech와 ongoing overlap에 높은 change score를 주는 문제도 크다. FP 개수 기준으로는 remote false activation이 더 큰 오류원이다.

### Candidate coverage

Score threshold를 제거한 post-NMS local-peak coverage:

| Collar | Coverage | Candidate 없는 GT |
| ---: | ---: | ---: |
| 100 ms | 61.14% | 1,795 |
| 250 ms | 91.12% | 410 |
| 500 ms | 99.29% | 33 |

500 ms candidate geometry 부재는 주된 한계가 아니다. 그러나 모든 score의 local peak를 포함한 값이므로 representation의 discriminative evidence가 충분하다는 뜻도 아니다.

## 한계점

### 1. 원래의 fine-tuning 대 scratch 질문을 답하지 못함

가장 큰 한계다. 실제 encoder fine-tuning과 scratch arm이 모두 없다.

### 2. `TRAINABLE` 이름이 실제 구현보다 강한 의미를 가짐

Pretrained encoder 내부 parameter는 학습되지 않았다. 결과를 `adaptation` 또는 `fine-tuning` 일반의 실패로 확대하면 안 된다.

### 3. Clean official regeneration 미실시

전체 score-range evaluator로 기존 raw prediction을 재평가했지만, 수정된 최종 코드와 깨끗한 commit에서 12개 arm을 다시 생성하지 않았다. 사용자가 재생성을 중지했다.

### 4. Development-known evidence

동일한 기존 자연 연속 회의와 out-of-fold split을 사용한 방향 선택용 결과다. 별도 untouched test panel, 제품 도메인, 다국어 일반화를 증명하지 않는다.

### 5. Operating-point optimism

각 arm의 threshold는 같은 development-known 결과에서 Macro F1이 최대가 되도록 선택됐다. 절대 성능을 배포 성능으로 해석하면 안 된다.

### 6. Structured representation과 event projection이 confounded

최종 event 결과만으로 state/relation 학습과 state/relation-to-event 변환 중 어느 단계가 주원인인지 분리할 수 없다.

### 7. Timing collar는 latency 정책이 아님

100/250/500 ms는 event matching 허용 오차다. 제품 latency나 lookahead 정책을 결정한 실험이 아니다. Context와 latency sweep도 하지 않았다.

### 8. FP 의미 분류의 한계

GT state와 speaker continuity로 자동 분류했지만 laughter와 prosody annotation은 없다. Remote FP가 웃음, 억양 변화, pitch 변화인지 판단하려면 별도 audio audit가 필요하다. 이 audit는 사용자 중지 결정에 따라 실행하지 않는다.

### 9. Scratch의 의미도 아직 고정되지 않음

Scratch가 동일 pretrained architecture의 random initialization인지, 별도 compact PSEM architecture인지, 다른 학습 데이터와 objective를 뜻하는지 실험 계약이 없다. 따라서 향후 비교를 하더라도 먼저 비교 단위를 명시해야 한다.

## 최종 결론

이번 실험에서 확인된 사실:

1. 시험한 고정 pretrained feature 중 mHuBERT-147가 가장 좋은 direct baseline이었다.
2. 현재 post-encoder output adapter는 일관된 개선을 만들지 못했다.
3. 현재 R7-B-style structured-to-event pipeline은 direct head보다 최종 event 성능이 낮았다.
4. mHuBERT direct의 오류에는 timing/localization 문제가 존재한다.
5. 더 큰 FP 개수는 same-speaker speech와 overlap continuation에서 발생하는 remote false activation이었다.
6. 500 ms 범위의 candidate geometry 부재는 주된 문제가 아니었다.

이번 실험에서 확인되지 않은 것:

1. 실제 encoder fine-tuning의 효과
2. Scratch 학습의 효과
3. Fine-tuning과 scratch 중 최종 방향
4. Structured representation 자체의 가치
5. Localization training과 hard-negative training 중 어느 것이 더 나은지
6. 제품 도메인과 다국어 환경에서의 성능

따라서 이 실험의 가장 정확한 한 문장 결론은 다음과 같다.

> 고정 pretrained embedding 기반 비교에서는 mHuBERT direct head가 가장 좋았고, 현재 output adapter와 현재 structured-to-event 구현은 개선을 만들지 못했다. 그러나 실제 encoder fine-tuning과 scratch를 시험하지 않았으므로 원래의 fine-tuning 대 scratch 결정은 여전히 미해결이다.

## 보존된 상세 문서와 artifact

- 12-arm 상세 결과: `experiments/psem_trainable_formulation_gate/RESULTS.md`
- 실험 계약과 실행 방법: `experiments/psem_trainable_formulation_gate/README.md`
- mHuBERT-A 오류 분석: `experiments/mhubert_a_error_decomposition/RESULTS.md`
- mHuBERT-A 분석 코드: `experiments/mhubert_a_error_decomposition/analyze.py`
- Raw prediction SHA-256: `4460e11c4689bb14afc6516da9c04ec8a7a5f1a1090eac08da41bf6eb9603b61`
- Error analysis SHA-256: `c4502c2dca6128c042a90fd223786188d96c94116910ef04bc0686c1587653a3`

