# ASR 입력 품질 후퇴 원인 규명 실험 계획서

## 1. 문서의 성격

- 작성일: 2026-08-12
- 기준 커밋: `31008444` (`revert local ASR trailing silence trimming`)
- 대상: 셀프 마이크 채널과 피어 오디오 채널
- 목적: 기능이나 실험 기반을 완성하는 것이 아니라, ASR 품질 후퇴의 원인과 유효한 개선 수단을 실제 결과로 판별한다.

이 문서는 실험 구현 명세가 아니다. 문서를 받은 에이전트가 저장 위치, 스크립트 구조, provider 호출 방법과 세부 코드를 결정한다. 다만 아래의 비교 조건, 데이터 경계, 측정 지표와 판정 규칙은 실험 결과의 해석 가능성을 위해 유지한다.

Cloud ASR의 final 이벤트 해석, 요청-응답 매핑과 세션 종료 정책은 이 문서의 범위가 아니다. 해당 작업은 [Cloud ASR Finalization 정책 교정 계획서](cloud-asr-finalization-policy-plan.md)에서 독립적으로 다룬다.

## 2. 최종적으로 답해야 할 질문

1. 동일한 음성을 ASR에 직접 전달했을 때보다 현재 앱 경로를 거쳤을 때 품질이 낮아지는가?
2. 차이가 있다면 캡처, 리샘플링, VAD/발화 경계, chunk 전송 중 어디에서 생기는가?
3. 편하게 말한 음성을 디지털로 증폭하는 것만으로 실제 크게 말했을 때의 전사 품질에 얼마나 가까워지는가?
4. 고정 gain, 발화 단위 정규화 상한선, 동적 AGC 중 추가 검토할 가치가 있는 방식은 무엇인가?
5. 피어 스테레오의 현재 산술평균 다운믹스가 레벨 또는 스펙트럼 손상을 유발하는가?
6. Soniox에서 관찰된 말끝 글자와 중간 단어 오류가 ASR에 실제 전달된 말끝 PCM과 연관되는가?
7. 위 효과가 Soniox 하나의 특성인지, 로컬 Qwen·Deepgram·Qwen ASR Flash에서도 같은 방향으로 나타나는 공통 입력 문제인지?
8. 실제 공간 음성 정답 데이터가 부족한 상태에서도 공간 단서를 더 연구할 근거가 남는가?

## 3. 현재까지 확인된 사실

### 3.1 셀프 채널

- 현재 PC의 `마이크(Virtual Desktop Audio)` WASAPI endpoint는 `48 kHz / 1 channel`로 노출된다.
- 실제 마이크 테스트 로그도 `requested_channels=1`, `opened_channels=1`, `frame_channels=1`이다.
- 따라서 현재 환경의 셀프 채널에서는 `(L + R) / 2`에 의한 손실이 없다.
- 마이크 테스트 UI 값은 raw capture frame의 순간 peak에 기반한다. 진단 로그의 `rms_db`와 직접 같은 수치가 아니다.
- 최근 발성 구간에서 약 `RMS -33.7 dBFS`, `peak -17.2 dBFS`가 관찰됐다. RMS만으로 디지털 신호가 극단적으로 작다고 단정할 수 없다.

### 3.2 피어 채널

- process capture는 코드 계약상 `48 kHz float32 stereo`이다.
- loopback capture는 출력 endpoint가 보고하는 채널 수를 사용하며, 최근 실제 peer pipeline 로그는 반복해서 `48 kHz / 2 channels`를 기록했다.
- 두 경로 모두 현재 `MonoFirstStreamingResampler`에서 채널 산술평균 후 16 kHz mono로 바뀐다.
- 2채널 포맷이 곧 유효한 공간 스테레오라는 뜻은 아니다. dual mono, 레벨 차이가 있는 stereo, HRTF/위상차가 있는 stereo를 구분해야 한다.
- 현재 설정의 loopback은 특정 Virtual Desktop 출력에 고정된 것이 아니라 기본 출력 장치를 따른다. 실행 시 실제 해결된 출력 endpoint를 기록해야 한다.

### 3.3 발화 경계

- 로컬 ASR 말끝 무음 제거를 롤백한 뒤 사용자가 체감한 로컬 인식률이 이전 수준으로 회복됐다.
- 이 결과는 발화 끝의 PCM 보존과 VAD 종료 조건을 독립적으로 확인해야 한다는 직접적인 근거다.
- Soniox에서는 전체 누락뿐 아니라 마지막 글자와 중간 단어의 대치·삭제도 관찰됐다. 삭제 오류만 측정해서는 안 된다.

### 3.4 보유 데이터의 한계

- 기존 speaker-turn-boundary 고정 fixture와 생성 데이터는 주로 16 kHz mono이며 공간 다운믹스의 실제 효과를 증명하지 못한다.
- LibriSpeech, AMI, AliMeeting 자료는 파이프라인 및 정상 레벨 대조군으로 사용할 수 있지만, 한국어 Quest Pro/Virtual Desktop 셀프 발화의 대체재가 아니다.
- 현재 확보된 AMI `Mix-Headset`와 AliMeeting `far_ch0`만으로 VRChat/Steam Audio 계열의 binaural 공간 음성을 대표할 수 없다.

## 4. 실험 원칙

### 4.1 결과 우선

- 재사용 가능한 benchmark framework, dashboard, UI, 범용 fixture generator를 만들지 않는다.
- 실험을 실행하고 결과표를 만드는 데 직접 필요한 최소 스크립트만 작성한다.
- 실험 때문에 production audio architecture를 리팩터링하지 않는다.
- 실험 전에 production AGC, 새 다운믹스 정책, 공간 분리 기능을 구현하지 않는다.
- 기존 테스트 스위트 정비나 광범위한 회귀 테스트 추가를 실험 성과로 간주하지 않는다.

### 4.2 paired comparison

- 가능한 모든 비교는 같은 원본 waveform에서 파생한다.
- 편한 발화와 큰 발화처럼 다시 말해야 하는 조건 외에는 재녹음으로 조건을 만들지 않는다.
- gain, 다운믹스, tail padding 후보는 같은 원본에서 결정적으로 생성한다.
- 한 실험에서 두 개 이상의 원인을 동시에 변경하지 않는다.

### 4.3 두 개의 실행 경로

각 주요 조건은 가능하면 아래 두 경로로 나눈다.

1. **Direct ASR**: 정답 waveform을 provider 요구 형식으로만 변환하여 VAD 없이 전달
2. **Production replay**: 현재 앱의 다운믹스, streaming resample, VAD와 chunking 경로를 통과

Direct는 좋은데 production replay만 나쁘면 모델 문제가 아니라 공통 앱 경로 문제로 분류한다.

### 4.4 레벨 bucket 분리

- 충분한 레벨의 기존 데이터에서 개선이 없다는 결과를 낮은 레벨에서의 무효 근거로 사용하지 않는다.
- `low`, `middle`, `normal/high` bucket을 active-speech RMS 기준으로 별도 집계한다.
- 전체 평균만 보고 gain의 유효성을 판정하지 않는다.

### 4.5 결과 보존

- 빈 전사, timeout, 연결 실패, 환각도 결과이며 삭제하지 않는다.
- 실패 사례를 임의로 재실행해 좋은 결과로 대체하지 않는다. 재실행은 별도 attempt로 남긴다.
- provider model ID, language, vocabulary/hint, endpoint option과 chunking 설정을 run manifest에 고정한다.
- 사용자 음성 원본과 API key는 Git에 커밋하지 않는다.

## 5. 데이터 계획

### 5.1 S-SELF: 실제 한국어 셀프 발화

가장 중요한 데이터다. 외부 녹음 앱이 아니라 우리 앱이 사용하는 Virtual Desktop Audio capture 경로를 기준으로 한다.

#### 문장

- 총 30문장
- 일반 대화문 10개
- 1~3초의 짧은 반응과 짧은 문장 5개
- 5초 이상의 긴 문장 5개
- 숫자, 고유명사, 외래어가 포함된 문장 5개
- 마지막 음절·조사·어미가 정답 판정에 중요한 문장 5개

#### 발성 조건

- `comfortable`: 평소처럼 편하게 말하기
- `deliberately_loud`: 의식적으로 더 크고 또렷하게 말하기

같은 30문장을 두 조건으로 녹음하여 60개 실제 발화를 만든다. 가능하면 같은 착용 상태, Virtual Desktop 설정, Windows 입력 설정과 주변 소음을 유지한다.

#### 연속 발화

- 동일 환경에서 자연스럽게 이어 말하는 3~5분짜리 세션 1개
- 짧은 pause, 긴 pause, 문장 끝 받침과 어미를 포함
- isolated utterance에서는 드러나지 않는 VAD와 streaming chunk 경계 확인에만 사용

#### 저장 지점

필요하면 최소한의 실험용 tap을 두어 다음을 저장한다.

- capture 직후 48 kHz mono 원본
- 16 kHz streaming resample 직후
- VAD/발화 조립 후 provider에 실제 전달된 PCM

앱 내부의 세 지점을 얻을 수 있다면 외부 녹음 앱 자료는 주 데이터에 필요하지 않다.

### 5.2 S-HIGH: 정상 레벨 실제 음성 대조군

- transcript가 있는 LibriSpeech 실제 음성 중 20~30개
- 짧은 문장과 긴 문장을 혼합
- 원본 active-speech RMS가 충분한 자료를 우선 선택
- 영어 pipeline control 및 전처리의 no-harm 확인에 사용

현재 외부 자료 후보:

- `C:/Users/salee/Documents/dev/puripuly_heart/.worktrees/puripuly_heart/experiment-v2-speaker-change-turn-boundaries-ls`
- `%TEMP%/opencode/stb_phase2_corpora`

이 데이터에서 gain 효과가 작거나 없는 것은 예상 가능한 결과다. 저레벨 발화 결과와 평균내지 않는다.

### 5.3 S-STEREO-SYNTH: 결정적 스테레오 파생 데이터

transcript가 있는 실제 mono 발화에서 소규모로 생성한다.

- `L=R` dual mono
- `L=speech, R=0`
- `L=0, R=speech`
- 좌우 레벨 차이 6/12/18 dB
- 한 채널 지연 0.1/0.3/0.7 ms
- 극성 반전 stress case
- 서로 다른 두 화자를 좌우에 배치
- 필요 시 소수의 HRTF 방향 `-60/-30/0/+30/+60°`

목적은 VRChat 현실성을 증명하는 것이 아니라, 현재 평균 다운믹스의 수학적 손실과 공간 연구의 최소 가능성을 선별하는 것이다.

### 5.4 S-PEER-OBS: 실제 VRChat 피어 관찰 데이터

확보 가능한 경우에만 사용한다.

- 정면, 왼쪽, 오른쪽에서 각 5~10초
- process capture와 Virtual Desktop output loopback의 raw stereo
- transcript가 없더라도 채널 통계 확인에는 사용 가능
- transcript 또는 알려진 prompt가 있을 때만 ASR 정확도 판정에 사용

실제 labeled peer 자료가 없으면 공간 실험의 최종 결론은 `inconclusive`로 남긴다. 합성 결과만으로 production 공간 기능을 채택하지 않는다.

### 5.5 외부 녹음 앱 자료

- 주 ASR corpus로 사용하지 않는다.
- 우리 앱과 외부 앱의 capture 차이를 확인할 필요가 있을 때만 동시 녹음한 진단 대조군으로 사용한다.
- 같은 Virtual Desktop Audio endpoint, 48 kHz mono, lossless WAV를 우선한다.
- 두 파일을 정렬한 뒤 일정 gain 차이, 시간 가변 compression/AGC, noise gate, 주파수 응답 차이를 구분한다.

## 6. ASR 대상

### 한국어 필수 대상

- 로컬 CPU: `qwen3-asr-0.6b-int8-sherpa`
- Soniox: 실행 시점에 해결된 모델 ID 기록. 현재 설정 기준 `stt-rt-v5`
- Deepgram: 실행 시점에 해결된 모델 ID 기록. 현재 설정 기준 `nova-3`
- Qwen ASR Flash Realtime: 실행 시점의 정확한 dated model ID 기록

### 선택 대상

- 로컬 GPU Qwen 1.7B: 모델 용량 효과를 구분할 필요가 있을 때만 사용
- 영어 S-HIGH에는 현재 CPU Auto가 선택하는 Parakeet 계열을 pipeline control로 추가 가능

한국어에서는 현재 로컬 CPU 후보가 사실상 Qwen 0.6B 하나이므로, 언어를 지원하지 않는 Parakeet을 억지로 한국어 비교군에 포함하지 않는다.

## 7. 실험 목록

## E0. 기준선 고정과 최소 smoke

### 목적

실험 실행 자체가 명백히 잘못된 상태에서 대량 추론하는 것을 막는다.

### 실행

- 기준 커밋, Python/runtime, 모델 파일 identity, CPU/GPU, provider 설정을 기록한다.
- S-SELF 3개와 S-HIGH 3개만 Direct와 Production replay로 실행한다.
- waveform duration, sample count, transcript와 error 기록이 정상인지 육안 확인한다.

### 중단 조건

- 같은 입력 ID가 서로 다른 원본을 가리킴
- provider payload duration이 예상과 다름
- 결과 파일이 기존 run을 덮어씀
- API/model/language 설정을 manifest에서 확인할 수 없음

E0를 통과하면 실험 코드 정비를 멈추고 본 실험으로 이동한다.

## E1. Direct ASR 대 Production replay

### 가설

모든 모델에서 체감 품질이 함께 낮아진 원인이 공통 오디오 경로라면, 같은 waveform의 Production replay 결과가 Direct보다 나빠진다.

### 입력

- S-SELF comfortable 30개
- S-HIGH 20~30개

### 비교

- Direct ASR
- 현재 Production replay

### 결과 해석

- Direct 우수, Production 열세: 공통 파이프라인 원인 지지
- 둘 다 저레벨에서 열세: provider/model의 레벨 민감성 가능
- 특정 provider만 Production에서 열세: provider adapter 또는 chunk 전송 가능
- 차이 없음: 후퇴 원인을 capture 이전 환경, 발성, provider 변경 또는 체감 표본 차이에서 찾는다.

## E2. 입력 레벨 반응 곡선

### 가설

ASR 또는 VAD가 특정 입력 레벨 아래에서 비선형적으로 악화된다.

### 입력 변환

동일한 깨끗한 실제 발화를 active-speech RMS 기준으로 다음 수준에 배치한다.

- -18 dBFS
- -24 dBFS
- -30 dBFS
- -36 dBFS
- -42 dBFS
- 필요할 때만 -48 dBFS

peak clipping이 생기지 않는 원본을 사용하며, 소음과 음성의 상대비인 SNR은 바꾸지 않는다.

### 비교

- Direct ASR level curve
- Production replay level curve

### 결과

- provider별 CER/삭제/대치/삽입 곡선
- VAD miss와 empty transcript 곡선
- 품질이 급락하기 시작하는 level bucket

이 결과가 gain 실험의 대상 범위를 정한다.

## E3. 고정 gain과 정규화 상한선

### 목적

처음부터 `+6 dB` 또는 동적 AGC를 결정하지 않고, 디지털 레벨 보정으로 얻을 수 있는 상한선을 측정한다.

### 입력

- E2에서 실제 품질 저하가 확인된 low/middle bucket
- 정상 레벨 no-harm 대조군

### 처리

- 0 dB
- +3 dB
- +6 dB
- +9 dB
- +12 dB
- 전체 발화를 본 뒤 목표 active-speech RMS로 맞추는 offline oracle normalization

offline oracle은 production 후보가 아니라 실시간 AGC가 얻을 수 있는 이론적 상한선의 근사다.

### 판정

- oracle도 개선하지 못하면 동적 AGC 구현을 중단한다.
- 고정 gain이 oracle 개선의 대부분을 얻고 정상 레벨에서 악화하지 않으면 단순 gain 후보를 남긴다.
- oracle만 개선하고 고정 gain이 불안정하면 제한적 동적 보정을 후속 검토한다.
- 정상 레벨 회귀, clipping, 환각 증가가 있으면 전역 고정 gain을 기각한다.

## E4. 실제 큰 발성과 디지털 증폭의 분해

### 가설

의식적으로 크게 말할 때의 개선은 단순 amplitude뿐 아니라 조음, 자음 에너지, mic SNR과 발성 방식 변화의 합일 수 있다.

### paired condition

1. comfortable 원본
2. comfortable을 deliberately_loud의 RMS에 디지털로 맞춘 버전
3. deliberately_loud 원본
4. deliberately_loud를 comfortable RMS로 감쇠한 버전

### 해석

- 2와 3이 비슷함: 디지털 gain의 기여가 큼
- 3과 4가 계속 우수함: 발성/명료도/SNR 기여가 큼
- 2가 개선하지만 3보다 낮음: 두 효과가 함께 존재

이 실험은 사용자가 Soniox에서 느낀 전사 품질 개선이 앱 gain으로 재현 가능한 비율을 추정한다.

## E5. 발화 끝 PCM 보존과 VAD 경계

### 가설

말끝 무음 제거 또는 VAD 종료가 ASR에 실제 전달되는 PCM을 잘라 마지막 음절과 일부 단어의 품질을 악화시킨다.

### 입력

- 마지막 음절·조사·어미가 중요한 S-SELF 문장 10개
- 연속 발화 세션 중 실제 말끝 오류가 생긴 구간

### PCM tail 조건

- 0 ms
- 100 ms
- 200 ms
- 400 ms
- 800 ms

### 비교 축

- Direct 또는 VAD 우회
- 현재 VAD/utterance 경로
- 실제 PCM에 남은 tail

ASR model과 provider option은 모든 PCM tail 조건에서 고정하며, 실제 PCM tail 이외의 조건은 바꾸지 않는다.

### 지표

- 마지막 글자 삭제율
- 마지막 단어 삭제율
- 전체 character deletion/substitution
- 완성 transcript latency

가장 긴 tail을 선택하는 것이 아니라, 품질 개선이 plateau에 도달하는 가장 짧은 조건을 찾는다.

## E6. 리샘플링·청크 전달 조건부 실험

### 실행 조건

E1에서 Direct와 Production replay의 차이가 확인됐지만 E2/E5로 설명되지 않을 때만 실행한다.

### 비교

- 48 kHz 원본에서 고품질 one-shot 16 kHz 변환
- 현재 streaming resampler와 실제 chunk size
- provider에 실시간 속도로 전송
- 동일 PCM을 가능한 한 빠르게 전송
- resampler flush와 마지막 PCM chunk 전달 완전성

### 측정

- 총 샘플 수와 duration
- 마지막 sample 손실
- chunk 경계 불연속
- 실제 전송 byte 수
- 완성 transcript 차이

E1에 차이가 없다면 이 실험을 수행하지 않는다.

## E7. 피어 모노 다운믹스

### 가설

현재 `(L + R) / 2`가 한쪽에 치우친 음성에는 레벨 손실을, 시간·위상차가 있는 음성에는 스펙트럼 손상을 만든다.

### 입력

- S-STEREO-SYNTH
- 확보 가능한 S-PEER-OBS raw stereo

### 비교

- 현재 산술평균
- L only
- R only
- 발화 단위 active RMS가 큰 채널 선택
- 정답을 보고 L/R 중 좋은 쪽을 고르는 oracle channel upper bound

각 조건은 두 번 비교한다.

1. 실제 출력 레벨 그대로
2. active-speech RMS를 같게 맞춘 뒤

RMS를 맞춘 뒤 차이가 사라지면 순수 레벨 문제다. 차이가 남으면 위상/스펙트럼 또는 간섭 문제다.

### 실제 peer 자료의 제한

- transcript가 없으면 채널 RMS, peak, correlation, delay, mean-downmix loss만 보고한다.
- labeled real peer가 없으면 ASR 개선 주장은 합성 조건에만 한정한다.

## E8. 공간 단서 선별 실험

### 실행 조건

E7에서 L/R 차이가 실제로 존재하고, 현재 평균보다 더 나은 결과를 얻을 여지가 확인될 때만 실행한다.

### 소규모 조건

- 알려진 방향의 HRTF 렌더링
- 서로 다른 화자의 방향 분리
- overlap 100/300/500 ms
- 위치가 유지되는 경우와 이동하는 경우

### 목표

- 공간 정보가 ASR waveform 개선에 직접 쓰일 가능성
- waveform은 그대로 두고 화자 전환/겹침 판단의 side information으로만 쓸 가능성

### 금지된 결론

- 합성 HRTF 성능을 VRChat production 성능으로 일반화하지 않는다.
- 실제 labeled VRChat 데이터 없이 neural beamforming, separation 또는 공간 기반 production 정책을 도입하지 않는다.

## E9. provider 교차 확인

### 실행 순서

1. 로컬 CPU Qwen으로 E1~E7의 넓은 조건을 선별한다.
2. 각 가설별 baseline과 유망 후보 최대 1~2개만 남긴다.
3. Soniox, Deepgram, Qwen ASR Flash Realtime에서 동일 paired subset을 확인한다.
4. cloud 결과가 provider 비결정성으로 의심될 때만 finalist를 한 번 추가 반복한다.

모든 조합을 모든 cloud provider에 보내는 full factorial은 수행하지 않는다.

## 8. 측정 지표

### 8.1 오디오

- sample rate, channels, sample count, duration
- active-speech RMS dBFS
- 전체 구간 RMS dBFS
- peak dBFS
- clipping sample ratio
- silence/noise floor와 추정 SNR
- L/R RMS와 peak 차이
- L/R correlation과 inter-channel delay
- 현재 mean-downmix가 강한 채널 대비 잃는 dB
- 발화 시작과 끝에서 제거 또는 보충된 시간

UI의 음량 퍼센트는 보조 관찰로만 남기고 판정 지표로 사용하지 않는다.

### 8.2 한국어 ASR

- CER를 주 지표로 사용
- character insertion/deletion/substitution을 별도 보고
- whitespace와 일반 punctuation을 정규화한 CER와 raw exact match를 함께 보존
- 문장 완전 일치율
- 마지막 음절/단어 보존율
- 숫자·고유명사·외래어 정확도
- empty transcript와 알려진 환각 비율
- utterance별 win/tie/loss

### 8.3 실행 특성

- local inference time과 RTF
- cloud first partial latency와 final latency
- timeout/reconnect/error
- 처리 후보 자체의 CPU time

정확도 원인 규명이 우선이며 latency는 후보가 유효하다고 판명된 뒤의 2차 판단 자료다.

## 9. 사전 판정 규칙

작은 pilot이므로 통계적 유의성을 production 보장처럼 표현하지 않는다. 대신 동일 발화 paired effect와 provider 간 방향 일관성을 본다.

### 9.1 공통 파이프라인 원인

- Direct가 Production replay보다 여러 provider에서 같은 방향으로 우수하면 지지
- 한 provider에서만 나타나면 provider adapter/config 문제로 분류
- 차이가 없으면 공통 PCM 경로 원인 가설은 약화

### 9.2 gain

후보를 남기려면 다음을 모두 만족해야 한다.

- low bucket의 CER 또는 핵심 오류가 반복적으로 개선됨
- normal/high bucket에서 의미 있는 회귀가 없음
- clipping과 환각이 증가하지 않음
- 로컬 Qwen만이 아니라 cloud provider 최소 2개에서 방향이 재현됨

oracle normalization에 효과가 없으면 동적 AGC는 기각한다. 고정 gain이 oracle 효과의 대부분을 얻으면 복잡한 AGC보다 고정 또는 제한 gain을 우선한다.

### 9.3 발성

실제 큰 발화를 normal level로 감쇠해도 우수하면 digital gain의 설명력은 제한적이라고 결론낸다. 이 경우 사용자가 느낀 개선을 전부 앱 gain으로 재현할 수 있다고 주장하지 않는다.

### 9.4 발화 끝

- tail 증가에 따라 마지막 음절 오류가 감소하면 경계 가설 지지
- Direct에서도 동일하면 provider/model의 말끝 PCM 민감성 가능
- Production에서만 동일하면 VAD 또는 마지막 PCM 전달 가능
- 품질 plateau 이후의 추가 tail은 latency 비용만 늘리는 것으로 본다.

### 9.5 다운믹스와 공간

- dual mono 또는 실제 peer에서 L/R이 사실상 같으면 downmix 변경을 중단한다.
- 현재 mean의 열세가 RMS matching 후 사라지면 spatial processing보다 level-safe downmix 문제로 한정한다.
- RMS matching 후에도 열세면 위상/간섭 가설을 남긴다.
- 합성에서만 개선하고 실제 데이터가 없으면 결과는 `promising but unvalidated`이며 production 변경 근거가 아니다.

## 10. 단계별 중단 규칙

1. E0 실패 시 대량 실행 금지
2. E1에서 Direct/Production 차이가 없으면 E6 생략
3. E2에서 레벨 민감성이 없으면 E3의 gain sweep 축소 또는 중단
4. E3 oracle이 무효면 동적 AGC 연구 중단
5. E7에서 L/R이 동일하면 E8 중단
6. 실제 labeled peer가 없으면 공간 결과를 production 권고로 승격하지 않음
7. cloud에서는 가설별 baseline과 finalist만 실행

이 규칙은 결과가 없을 때 더 많은 harness와 변형을 만들어 실험을 계속하는 것을 막기 위한 것이다.

## 11. 최소 산출물

실험별로 다음만 남기면 된다.

### Run manifest

- run ID와 일시
- Git commit과 dirty 상태
- 입력 audio ID와 hash
- 실제 모델 ID와 provider options
- 변환 조건과 관측된 실제 PCM tail
- 실행 경로 `direct` 또는 `production_replay`
- 성공, 실패, timeout 상태

### Per-utterance result

- reference text
- transcript 원문과 정규화문
- CER와 insertion/deletion/substitution
- audio metrics
- latency/RTF/error

### 최종 `RESULTS.md`

각 가설에 대해 다음 네 값만 명확히 적는다.

- `supported`
- `not supported`
- `inconclusive`
- `blocked by missing real-domain data`

그리고 effect table, 대표 실패 사례, 한계, 다음 제품 결정을 기록한다. 대시보드나 별도 시각화 앱은 만들지 않는다.

## 12. 구현 범위 제한

### 허용

- raw capture/replay/transform/provider call/scoring을 위한 작은 실험 스크립트
- 필요한 경우 production source를 감싸는 좁은 실험용 tap
- CSV/JSON/Markdown 결과 생성
- 이미 설치된 모델과 provider adapter 재사용

### 제외

- 범용 오디오 benchmark framework
- production 설정/UI에 gain 또는 downmix 옵션 추가
- production AGC/limiter/beamformer 구현
- speaker-turn-boundary 실험 체계의 대규모 확장
- 실험과 무관한 architecture refactor
- 광범위한 test suite 정비
- 대규모 corpus 다운로드 자동화

실험용 코드가 production owner나 lifecycle을 변경하면 architecture drift 가능성을 결과 문서에 별도로 보고한다.

## 13. 실행 순서와 비용 제어

1. S-SELF 녹음과 S-HIGH subset 선정
2. E0 최소 smoke
3. E1 Direct/Production 비교
4. 로컬 CPU worker에서 E2~E5 선별
5. 필요할 때만 E6
6. E7 stereo downmix
7. 가설별 finalist 선정
8. cloud provider 교차 확인
9. E7 결과가 유망할 때만 E8
10. 결과표와 `RESULTS.md` 작성

cloud 실행 전 provider별 예상 입력 오디오 분량과 반복 수를 기록한다. pilot에서는 조건 수를 늘리기보다 paired utterance 수와 실패 사례 보존을 우선한다.

## 14. CPU 장시간 작업의 Orca/OpenCode worker 정책

실험 설계, 코드 구현, smoke와 결과 해석은 문서를 받은 주 에이전트가 소유한다. 로컬 CPU ASR 대량 추론처럼 오래 걸리는 실행만 Orca CLI를 통해 별도 OpenCode 세션에 맡긴다.

### 역할 경계

- 주 에이전트: 실험 코드 구현, 입력/조건 검증, run manifest 고정, 결과 해석
- OpenCode worker: 이미 준비된 명령 실행, 진행 감시, 지정된 run directory에 원시 결과 저장
- worker는 실험 설계를 바꾸거나 production 코드를 수정하지 않는다.
- CPU contention이 결과에 영향을 주므로 로컬 CPU inference worker를 여러 개 동시에 돌리지 않는다.
- latency/RTF를 측정하는 run 동안 다른 CPU-heavy 작업을 병렬 실행하지 않는다.

### 실행 절차

Orca CLI 사용 전 설치된 버전의 가이드를 다시 읽는다.

```powershell
orca skills get orca-cli
orca status --json
```

현재 worktree에 OpenCode worker terminal을 만든다.

```powershell
orca terminal create --worktree active --title "ASR CPU experiment worker" --command "opencode" --json
orca terminal wait --terminal <handle> --for tui-idle --timeout-ms 60000 --json
```

주 에이전트는 정확한 실행 명령, 고정된 입력 manifest와 유일한 output directory를 포함한 bounded prompt를 보낸다.

```powershell
orca terminal send --terminal <handle> --text "<bounded experiment-run prompt>" --enter --json
```

worker prompt에는 최소한 다음을 명시한다.

- 코드를 수정하지 말 것
- 지정된 명령과 조건만 실행할 것
- 결과를 덮어쓰지 말 것
- stdout/stderr, exit code, 시작/종료 시각을 보존할 것
- 실패를 숨기거나 임의 재실행하지 말 것
- 완료 시 run manifest와 결과 경로를 보고할 것

주 에이전트는 Orca terminal을 통해 완료를 확인하고 결과를 직접 검토한다.

```powershell
orca terminal wait --terminal <handle> --for tui-idle --timeout-ms 3600000 --json
orca terminal read --terminal <handle> --json
```

OpenCode worker가 같은 worktree를 사용하므로, worker 실행 중 주 에이전트는 해당 실험 코드와 run manifest를 수정하지 않는다. 수정이 필요하면 run을 중단하거나 끝낸 뒤 새 run ID로 다시 시작한다.

## 15. 결과별 다음 결정

| 결과 | 다음 행동 |
|---|---|
| Production replay만 나쁨 | 손실이 처음 나타나는 오디오 경계를 좁혀 수정 후보 설계 |
| low level에서 gain/oracle이 유효 | 가장 단순한 안전 후보부터 별도 production 설계 |
| 실제 큰 발성만 유효 | digital gain의 기대 효과를 제한하고 SNR/발성 조건 중심으로 해석 |
| tail이 유효 | 가장 짧은 유효 PCM padding 또는 VAD 보존 정책 검토 |
| mean downmix만 열세 | self는 유지하고 peer에만 안전한 downmix 후보 검토 |
| 합성 공간만 유효 | 실제 labeled peer 확보 전까지 연구 결과로만 보존 |
| 어떤 후보도 일관되지 않음 | AGC/공간 처리 도입 없이 provider·발화·환경 변화 조사로 전환 |

## 16. 연구 결과 해석 시 참고할 경계

- 다채널 neural filtering이 single-channel보다 상대 WER를 개선한 연구가 있지만, 물리적 microphone array와 공동 학습 모델을 전제로 한다. 현재 VRChat stereo downmix에 같은 효과를 직접 기대할 수 없다.
- binaural 공간 단서가 간섭 화자 조건에서 도움이 된 연구도 통제된 HRTF/array 조건의 결과다.
- Windows capture는 앱이 선택한 processing mode와 driver APO에 따라 AGC, noise suppression, AEC 등의 영향을 받을 수 있다. 외부 녹음 앱 자료를 우리 앱 raw capture와 동일시하지 않는다.

참고 자료:

- [Google Research: Multichannel Signal Processing with Deep Neural Networks for ASR](https://research.google/pubs/multichannel-signal-processing-with-deep-neural-networks-for-automatic-speech-recognition/)
- [ISCA: Binaural spatially aware hearing-aid ASR](https://www.isca-archive.org/interspeech_2015/kayser15_interspeech.html)
- [Microsoft: Audio Signal Processing Modes](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-signal-processing-modes)

이 연구들은 실험 가설을 세우는 근거일 뿐이며, 본 프로젝트의 production 효과에 대한 증거는 이 계획에서 수집한 paired 결과로만 판단한다.
