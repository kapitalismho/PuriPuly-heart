# OpenRouter 라우팅 정책 변경 핸드오프 (order → only+latency)

## 개요

OpenRouter 호출의 provider preferences를 `order` 기반 강제 순서에서 `only` + `sort: latency` 조합으로 전환한다. `order`와 `sort`는 OpenRouter API에서 상호 배타적이지만, `only`/`ignore`는 `sort`와 완벽히 호환된다. `only`로 후보 프로바이더를 화이트리스트로 제한하고, `sort: latency`로 그 후보들 중 OpenRouter가 실시간 레이턴시 기반 자동 선택을 하게 한다.

**참조 커밋(vnext):** `71d5518 refactor(provider): replace OpenRouter order routing with only+latency sort`

## 변경 배경

- 기존 `order` 분기(parasail_first, novita_first, model별 order)는 프로바이더 순서를 강제했으나, 레이턴시 최적화가 아니었다.
- `only` + `sort: latency` 조합은 이미 `GOOGLE_GEMINI_LATENCY` 분기에서 검증된 패턴이다.
- `PARASAIL_FIRST`/`NOVITA_FIRST` routing_mode는 UI 노출이 없고 설정 파일/테스트에서만 쓰였으므로 제거해도 안전하다.
- variant 강제(`baidu/fp8`, `parasail/bf16`, `parasail/fp8`, `wafer/fp8`)는 `only`가 프로바이더 이름 단위로만 제한하므로 자연스럽게 제거된다.

## 소스 변경 (3개 파일)

### 1. `src/puripuly_heart/providers/llm/openrouter.py`

`_build_provider_preferences` 함수를 재설계한다.

**변경 전 (dev 기준, vnext와 다를 수 있음):**
- `routing_mode` 파라미터를 받아 parasail_first/novita_first order 분기를 가짐
- model별 order 분기에 variant(`baidu/fp8`, `parasail/bf16` 등) 포함
- `DEEPSEEK_ONLY`가 `order` + `only` 혼합

**변경 후 (목표 형태):**

```python
def _build_provider_preferences(
    provider_routing: OpenRouterProviderRouting = OpenRouterProviderRouting.DEFAULT,
    *,
    model: str | None = None,
) -> dict[str, object]:
    if provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY:
        return {
            "sort": "latency",
            "only": ["deepseek", "baidu"],
            "allow_fallbacks": True,
        }
    if provider_routing == OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY:
        return {
            "sort": "latency",
            "only": ["google-vertex", "google-ai-studio"],
            "allow_fallbacks": True,
            "data_collection": "deny",
        }
    if model == "google/gemma-4-26b-a4b-it":
        return {
            "sort": "latency",
            "only": ["cloudflare", "parasail", "wafer"],
            "allow_fallbacks": True,
        }
    if model == "deepseek/deepseek-v4-flash":
        return {
            "sort": "latency",
            "only": ["deepseek", "parasail", "Fireworks", "Baidu Qianfan"],
            "allow_fallbacks": True,
        }
    return {
        "sort": "latency",
        "allow_fallbacks": True,
        "ignore": ["venice", "deepinfra", "google-vertex"],
    }
```

**주의사항:**
- `routing_mode` 파라미터를 제거한다. 단, `OpenRouterLLMProvider.routing_mode` / `HttpxOpenRouterClient.routing_mode` 필드는 설정 호환성을 위해 유지한다(필드는 존재하지만 라우팅 결정에 영향 안 줌).
- `_build_request_body`에서 `_build_provider_preferences` 호출부도 `self.routing_mode` 인자를 제거한다:
  ```python
  "provider": _build_provider_preferences(
      self.provider_routing,
      model=self.model,
  ),
  ```
- `OpenRouterRoutingMode` import는 routing_mode 필드 타입으로 여전히 필요하므로 유지.
- dev 브랜치에 `DEEPSEEK_ONLY`/`GOOGLE_GEMINI_LATENCY`가 없다면 해당 분기는 생략하거나 dev의 기존 분기를 동일한 `only`+`sort: latency` 패턴으로 변환한다.
- dev의 model 분기가 vnext와 다를 수 있으니, dev의 실제 model 문자열을 확인 후 `only` 리스트를 적용할 것.

### 2. `src/puripuly_heart/config/settings.py`

`OpenRouterRoutingMode` enum에서 `PARASAIL_FIRST`/`NOVITA_FIRST` 제거, `LATENCY`만 유지.

```python
class OpenRouterRoutingMode(str, Enum):
    LATENCY = "latency"
```

**마이그레이션:** `_parse_openrouter_routing_mode`(동일 파일 내)가 이미 invalid 값을 `LATENCY`로 fallback하므로, 기존 settings.json의 `parasail_first`/`novita_first` 값은 자동으로 `latency`로 정규화된다. 별도 마이그레이션 코드 불필요.

### 3. `src/puripuly_heart/config/runtime_resolution.py`

`_OPENROUTER_ROUTING_MODES` 허용 목록에서 `parasail_first`/`novita_first` 제거.

```python
_OPENROUTER_ROUTING_MODES: Final[tuple[str, ...]] = ("latency",)
```

`_normalize_allowed`가 허용 목록에 없는 값을 default(`latency`)로 정규화하므로, runtime intent 생성 시 legacy 값도 자동 변환된다.

## 테스트 변경 (6개 파일)

dev 브랜치의 테스트 구성이 vnext와 다를 수 있으니, 아래 원칙만 따른다.

### 원칙

1. **`OpenRouterRoutingMode.PARASAIL_FIRST`/`NOVITA_FIRST` 직접 참조** → `OpenRouterRoutingMode.LATENCY`로 변경 (enum에서 제거됐으므로 AttributeError 방지).
2. **문자열 `"parasail_first"`/`"novita_first"`**:
   - 입력값으로 쓰이면 그대로 유지 → 마이그레이션 정규화 검증("legacy 값이 latency로 변환됨").
   - assertion으로 쓰이면 `"latency"`로 변경.
3. **`_build_provider_preferences` 결과 검증**(order 배열 비교):
   - `order` 키를 `sort: latency` + `only` 형태로 변경.
   - variant(`baidu/fp8`, `parasail/bf16` 등) 제거.
   - gemma4: `only: ["cloudflare", "parasail", "wafer"]`
   - deepseek v4 flash: `only: ["deepseek", "parasail", "Fireworks", "Baidu Qianfan"]`
   - DEEPSEEK_ONLY: `sort: latency` + `only: ["deepseek", "baidu"]`
4. **routing_mode 변경 감지 테스트**(controller): routing_mode가 LATENCY 단일값이라 변경 감지가 안 되므로, `provider_routing` 변경으로 대체. `_build_llm_provider_signature`가 `provider_routing`을 포함하므로 rebuild가 트리거됨.

### 파일별 (vnext 기준, dev는 구조가 다를 수 있음)

| 파일 | 수정 |
|---|---|
| `tests/providers/test_openrouter_provider.py` | provider assertion을 새 only+latency 형태로. parasail_first order 테스트는 latency 기반으로 재활용. |
| `tests/config/test_runtime_resolution.py` | assertion `== "parasail_first"` → `== "latency"`(입력값은 유지해 마이그레이션 검증). |
| `tests/app/test_wiring_providers.py` | `OpenRouterRoutingMode.PARASAIL_FIRST` → `LATENCY` (replaceAll). |
| `tests/ui/test_controller_branch_paths.py` | `PARASAIL_FIRST`/`NOVITA_FIRST` → `LATENCY` (replaceAll). routing_mode 변경 감지 테스트는 `provider_routing = DEEPSEEK_ONLY` 변경으로 대체. |
| `tests/config/test_config_and_secrets.py` | `PARASAIL_FIRST` → `LATENCY`. `NOVITA_FIRST` round-trip은 마이그레이션 검증으로(입력 `"novita_first"` 리터럴, 결과 `LATENCY`). |
| `tests/config/settings_migration_fixtures.py` | `PARASAIL_FIRST` → `LATENCY`. |

## 검증 방법

Windows 환경, `.venv` 사용:

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/providers/test_openrouter_provider.py -q
& ".venv\Scripts\python.exe" -m pytest tests/config/test_runtime_resolution.py -q
& ".venv\Scripts\python.exe" -m pytest tests/app/test_wiring_providers.py -q
& ".venv\Scripts\python.exe" -m pytest tests/config/test_config_and_secrets.py -q
& ".venv\Scripts\python.exe" -m pytest tests/app/test_wiring_llm_factory.py -q
```

`test_controller_branch_paths.py`는 큰 파일이므로 수정한 테스트만 선택 실행 권장.

## 주의사항

- **dev 브랜치 코드 확인**: dev의 `_build_provider_preferences`, `OpenRouterRoutingMode`, `OpenRouterProviderRouting` enum, 허용 목록이 vnext와 다를 수 있다. 적용 전 dev의 실제 코드를 먼저 읽을 것.
- **pre-existing 실패와 구분**: vnext에서 `test_settings_migration_fixtures.py`(6개), `test_public_compatibility_surfaces.py`(2개), `tests/architecture/`(2개)는 이 변경과 무관한 pre-existing 실패였다. dev에서도 동일한 pre-existing 실패가 있을 수 있으니, 내 변경으로 인한 실패인지 `git stash`로 확인할 것.
- **black 포맷**: 단일 원소 tuple `_OPENROUTER_ROUTING_MODES = ("latency",)`는 한 줄로 포맷해야 black 통과.
- **Rust 재컴파일 불필요**: 이 변경은 Python만 수정하므로 Rust overlay 재컴파일 없음.
- **호환성 표면 유지**: `routing_mode` 필드/설정 키는 호환성을 위해 유지(제거하지 않음). `public_compatibility_surfaces_snapshot.json`의 `routing_mode` 항목도 유지.
