# nmrbindfit 코드리뷰 + 설계점검 제안서 (for Opus cross-check)

작성일: 2026-02-11  
대상: `nmrbindfit` (현재 main 작업본)  
목적: 신뢰성, 통계 일관성, 유지보수성 중심의 실행 가능한 개선안 제시

---

## 1) 검토 범위와 방법

- 범위
  - 소스: `nmrbindfit/*.py` (12개 모듈)
  - 테스트: `tests/*.py` (10개 파일, 29 tests)
- 방법
  - 정적 코드 검토(모듈 경계, 실패 처리, 수치 안정성)
  - 실행 검증: `pytest --cov=nmrbindfit --cov-report=term`
- 현재 스냅샷
  - 테스트: 29 passed
  - 커버리지: 총 83%
  - 상대적으로 낮은 영역: `nmrbindfit/report_pipeline.py` 61%

---

## 2) 총평 (요약)

프로젝트는 이미 실사용 가능한 구조와 품질을 갖추고 있다. 특히 모델 피팅 파이프라인, 실패 허용 정책, 리포트 산출 흐름은 명확하다.  
다만 외부 검증(논문/타 소프트웨어 비교, 고강도 데이터)까지 고려하면, 아래 3가지가 핵심 리스크다.

1. 피팅 과정의 실패 의미론(광범위 예외 마스킹 + 고정 페널티)
2. 결측치 정책의 기본값(strict vs mask)과 재현성/실용성 균형
3. 수치 안정성 이슈(`solve_11`)는 즉시 공식 교체보다 경계조건 테스트로 위험도 계량이 우선

---

## 3) 현재 강점 (유지 권장)

- 모듈 분리가 명확함: `io -> fit -> report_pipeline/report` 경로가 선명함 (`nmrbindfit/cli.py:15`, `nmrbindfit/cli.py:18`)
- 비선형 해에서 Newton + bisection fallback 조합으로 안정성 방어 (`nmrbindfit/equilibrium.py:237`, `nmrbindfit/equilibrium.py:333`)
- multistart에서 "낮은 RSS 실패"보다 "수렴 성공"을 우선하는 정책이 테스트로 고정됨 (`tests/test_fit_multistart.py:33`)
- 정보기준 정의(`k = p + 1`, shared sigma^2 포함)가 코드/테스트/문서에서 정합 (`nmrbindfit/fit.py:324`, `tests/test_fit_information_criteria.py:27`, `README.md:103`)
- 실패 모델을 전체 파이프라인 중단 없이 결과에 기록하고 계속 진행 (`nmrbindfit/fit.py:648`, `tests/test_fit_multistart.py:107`)

---

## 4) 우선순위별 이슈 및 제안

## P0 (즉시 권장)

### P0-1. `solve_11` 수치 안정성은 "즉시 교체"보다 "테스트 기반 검증" 우선

- 상태: Partially confirmed
- 근거
  - 현재 식: `hg = 0.5 * (term - sqrt(discr))` (`nmrbindfit/equilibrium.py:46`)
  - 판별식 계산: `term**2 - ...` (`nmrbindfit/equilibrium.py:44`)
  - 음수 판별식 방어는 이미 존재: `np.maximum(discr, 0.0)` (`nmrbindfit/equilibrium.py:45`)
  - 테스트 실행 중 overflow warning은 관찰됨(`equilibrium.py:44`)
- 영향
  - 강결합/극단 농도비에서 유효숫자 손실 가능성은 있으나, 일반 입력 범위에서 항상 치명적이라고 단정하기 어려움
- 제안
  - 1단계: 극단 K/농도비 sweep 회귀테스트를 먼저 추가해 위험도를 수치로 확인
  - 2단계(조건부): 오차 임계 초과 구간에서만 안정형 계산식(분자/분모형 또는 분기식) 도입
  - 3단계: 변경 전/후 결과를 mass balance와 BIC 순위 일관성 기준으로 비교
- 수용 기준
  - sweep 테스트에서 경계 입력 오차가 사전 정의 임계치 이내
  - 공식을 바꿀 경우 기존 정상 구간 정확도/성능 회귀 없음

### P0-2. `_residual_vector`의 광범위 예외 마스킹 + 고정 페널티

- 상태: Confirmed
- 근거
  - `except Exception`으로 모든 예외를 잡아 페널티 잔차로 대체 (`nmrbindfit/fit.py:107`)
  - 고정 페널티 `1e6` 사용 (`nmrbindfit/fit.py:111`, `nmrbindfit/fit.py:115`, `nmrbindfit/fit.py:119`)
- 영향
  - 진짜 버그(코드 결함)와 수치 실패가 동일하게 취급되어 디버깅 신호가 약해짐
  - 데이터 스케일과 무관한 페널티는 최적화 지형을 불필요하게 왜곡할 수 있음
- 제안
  - 예외를 분리: 예상 수치오류만 페널티, 예상 밖 예외는 즉시 raise
  - 페널티를 데이터 스케일 기반으로 전환(고정치 제거)
  - "페널티 적용 횟수"를 `FitResult`/리포트 경고로 노출
- 수용 기준
  - 비예상 예외가 테스트에서 즉시 드러남
  - 동일 데이터셋에서 수렴성/재현성 악화 없이 동작

## P1 (단기 개선)

### P1-1. 결측치 처리의 정보 손실 완화(정책 선택형)

- 상태: Partially confirmed
- 근거
  - ppm 컬럼에 NaN 1개만 있어도 해당 컬럼 전체 드롭 (`nmrbindfit/io.py:146`, `nmrbindfit/io.py:154`)
- 영향
  - 일부 포인트 결측이 전체 peak 소실로 확대되어 정보 손실 및 불확실성 증가
- 제안
  - 정책 분리: `drop-column`과 `mask-missing`을 모두 지원
  - 기본값(strict vs mask)은 사용자/도메인 기준으로 결정하고, 선택된 정책을 `report.html`/`decision.txt`에 명시
  - 잔차 계산에서 finite mask 기반 처리 경로 도입
- 수용 기준
  - 결측 일부 데이터에서 strict/mask 두 모드 모두 재현 가능
  - 산출물에 실제 적용된 결측치 정책이 명시됨

### P1-2. `solve_12`/`solve_21`의 abort-on-first-failure 의미를 정책화 필요

- 상태: Confirmed (동작은 의도적)
- 근거
  - 첫 실패 시 루프 `break` (`nmrbindfit/equilibrium.py:386`, `nmrbindfit/equilibrium.py:418`)
- 영향
  - 한 점 실패가 이후 전 구간 NaN으로 이어질 수 있음
  - 반대로 fail-fast는 "의심스러운 해 전체 배제"라는 장점도 있음
- 제안
  - 현재 기본값은 유지(fail-fast), 옵션으로 continue 모드 제공
  - continue 모드에서는 `g_prev`를 마지막 성공 해로 유지하여 다음 점 초기값 안정성 확보
  - 보고서에 실패 인덱스/비율을 명시해 해석 신뢰도 보강
- 수용 기준
  - fail-fast/continue 두 모드의 동작이 테스트로 고정

### P1-3. `report_pipeline`의 낮은 테스트 커버리지 + 느슨한 타입 경계

- 상태: Confirmed
- 근거
  - 커버리지 61%
  - `object` 기반 인자/맵이 많아 정적 안정성 낮음 (`nmrbindfit/report_pipeline.py:80`, `nmrbindfit/report_pipeline.py:111`, `nmrbindfit/report_pipeline.py:367`)
- 영향
  - 리포트 포맷 변경 시 런타임 오류가 늦게 발견될 가능성
- 제안
  - `FitResultLike`/`DatasetLike` 기반 Protocol 타입 정리
  - 요약행/결정문/경고 생성 로직 단위테스트 확장
- 수용 기준
  - `report_pipeline.py` 커버리지 75%+ 목표

## P2 (중기)

### P2-1. 테스트 구성의 경계값/통합 시나리오 보강

- 상태: Confirmed
- 근거
  - `test_equilibrium.py`는 mass balance 중심 3개 테스트 (`tests/test_equilibrium.py:6`)
  - E2E는 단일 CSV + bootstrap=0 경로 중심 (`tests/test_cli_e2e_smoke.py:6`)
  - `models.py` 직접 테스트는 1개 (`tests/test_models.py:7`)
- 영향
  - 극단 K, replicate+bootstrap 조합, 보고서 구조 변경 리스크를 사전 포착하기 어려움
- 제안
  - stress matrix 테스트 추가(K sweep, 농도비 sweep)
  - replicate + bootstrap > 0 E2E 추가
  - HTML/CSV 출력 스냅샷(또는 핵심 필드) 검증

---

## 5) "바꾸지 말아야 할 것" (명시)

- 정보기준 `k` 정의는 "고정 불가침" 항목으로 두지 않는다. 현재 `k=p+1`은 유지 가능하나, 타 소프트웨어/문헌 비교를 위해 리포트·문서에 정의를 명시하는 것을 필수로 둔다 (`nmrbindfit/fit.py:324`, `nmrbindfit/stats.py:10`, `README.md:103`)
- 모델 실패를 결과에 기록하고 파이프라인을 계속 진행하는 graceful degradation 유지 (`nmrbindfit/fit.py:648`)
- multistart에서 success 우선 선택 정책 유지 (`nmrbindfit/fit.py:267`, `tests/test_fit_multistart.py:33`)
- 의사결정 문구의 "provisional" 톤 유지(과대확신 방지) (`nmrbindfit/report_pipeline.py:462`, `tests/test_report_language.py:34`)

---

## 6) 실행 로드맵 (제안)

### Phase 1 (1-2일)
- P0-1, P0-2 처리
- 경계값 sweep 테스트(수치안정성) + 예외 분리 테스트 추가

### Phase 2 (2-3일)
- P1-1 결측치 정책 확장(옵션화)
- P1-2 solver failure 모드 정책화 + 리포트 노출

### Phase 3 (2일)
- P1-3/P2-1 테스트 확장 + 타입 경계 정리
- 커버리지 목표 상향(`report_pipeline` 75%+)

---

## 7) Opus 4.6 크로스체크 요청 질문 (권장)

1. `solve_11`의 안정형 근 공식 전환이 본 도메인 입력 분포에서 실제로 유의미한가?
2. `_residual_vector`의 예외 분리(예상 vs 비예상) 설계가 과도한 복잡도 없이 적용 가능한가?
3. 결측치 정책을 옵션화할 때 기본값을 strict로 유지하는 것이 재현성 측면에서 최선인가?
4. `solve_12/21` fail-fast 기본 유지 + continue 옵션 추가가 도메인 해석에 혼선을 줄 가능성은 없는가?
5. 리포트 파이프라인 타입 강화 시 최소 변경으로 가장 큰 안정성을 얻는 경계는 어디인가?

---

## 8) 참고 지표

- 테스트: `29 passed`
- 커버리지 요약
  - `nmrbindfit/report_pipeline.py`: 61%
  - `nmrbindfit/models.py`: 80%
  - `nmrbindfit/plots.py`: 82%
  - 전체: 83%

이 제안서는 "전면 재작성"이 아니라, 현재 아키텍처를 유지하면서 리스크를 줄이는 최소-고효율 개선안을 목표로 한다.

---
---

# Antigravity 크로스체크 코멘터리

**작성일**: 2026-02-11  
**작성자**: Antigravity (CODE_REVIEW.md 원저자)  
**대상**: 위 Codex 제안서에 대한 항목별 동의/비동의 판정

---

## 동의/비동의 요약표

| 항목 | Codex 제안 | 판정 | 비고 |
|------|-----------|------|------|
| P0-1 | solve_11 catastrophic cancellation | ⚠️ **부분 동의** | 이미 방어 구현 존재, 보강은 동의 |
| P0-2 | _residual_vector 예외 마스킹 + 1e6 | ✅ **동의** | 예외 분리 + 스케일 기반 페널티 모두 타당 |
| P1-1 | 결측치 처리 과도한 정보 손실 | ⚠️ **부분 동의** | mask-missing 추가는 좋으나 기본값 strict 유지 의문 |
| P1-2 | abort-on-first-failure 정책화 | ✅ **동의** | continue 옵션 추가 + g_prev 유지 필요 |
| P1-3 | report_pipeline 커버리지 + 타입 | ✅ **동의** | FitResultLike Protocol 도입 동의 |
| P2-1 | 테스트 경계값/통합 보강 | ✅ **동의** | stress matrix + E2E 확장 모두 필요 |
| §5 유지사항 | σ² 카운팅 k=p+1 유지 | ❌ **비동의** | 명확화는 필요하나 "유지 권장"은 성급 |
| §5 유지사항 | graceful degradation 유지 | ✅ **동의** | — |
| §5 유지사항 | multistart success 우선 유지 | ✅ **동의** | — |
| §5 유지사항 | provisional 톤 유지 | ✅ **동의** | — |

---

## 항목별 상세 판정

### P0-1. solve_11 catastrophic cancellation — ⚠️ 부분 동의

Codex가 지적한 `term**2 - 4*h_tot*g_tot` (L44)의 수치 손실 가능성은 **이론적으로 타당**하다. 그러나 현재 코드를 면밀히 보면:

```python
# equilibrium.py L42-46
term = h_tot + g_tot + 1.0/k
discr = term**2 - 4.0 * h_tot * g_tot
discr = np.maximum(discr, 0.0)        # 음수 판별식 방어
hg = 0.5 * (term - np.sqrt(discr))    # 작은 근 선택
```

이 구현은 **이미 "작은 근" 형태를 사용**하고 있다. `term - sqrt(discr)` 형태에서 cancellation이 발생하는 것은 `discr ≈ term²`, 즉 `4*h_tot*g_tot ≈ 0` (매우 묽은 게스트)일 때인데, 이 경우 `hg ≈ 0`이 물리적으로 올바르다.

진정한 cancellation 위험은 **`1/k`가 `h_tot + g_tot`에 비해 극도로 작을 때** (즉 강결합) `term ≈ h_tot + g_tot`이 되어 `term² - 4*h_tot*g_tot` 계산에서 유효숫자 손실이 생기는 경우이다.

**동의하는 부분**: 극단 K/농도비 sweep 회귀테스트 추가. 이것은 반드시 필요하다.

**유보하는 부분**: "수치안정형으로 변경" 제안의 구체적 형태가 명시되지 않았다. 분자/분모 형태(`4*h_tot*g_tot / (term + sqrt(discr))`)로 전환하면 `term + sqrt(discr) ≈ 0`인 경우(물리적으로 불가능하지만)에 대한 새로운 방어가 필요하므로, 변경 시 반드시 비교 테스트 필요.

---

### P0-2. _residual_vector 예외 마스킹 + 고정 페널티 — ✅ 동의

내 CODE_REVIEW에서는 이 이슈를 별도로 식별하지 않았으나, **Codex의 지적이 정확하다**.

```python
# fit.py L107-119
except Exception:          # 모든 예외를 동일하게 처리
    res = np.full_like(ds.y, 1e6)  # 데이터 스케일 무관한 고정 페널티
```

**동의 근거**:
- `except Exception`은 `IndexError`, `TypeError` 등 코드 결함도 삼켜버림
- `1e6`은 ppm 스케일(~0.01–10)에 비해 과도하게 크지만, 데이터가 다른 단위를 사용하면 상대적 비율이 달라짐
- 페널티 적용 횟수를 `FitResult`에 노출하는 것은 디버깅에 큰 도움

**구체적 구현 제안**:
```python
except (RuntimeError, ValueError):  # 예상 수치 오류만 포착
    scale = max(1.0, np.nanmax(np.abs(ds.y)))
    res = np.full_like(ds.y, scale * 1e3)  # 스케일 기반 페널티
```

---

### P1-1. 결측치 처리 — ⚠️ 부분 동의

**동의하는 부분**: `mask-missing` 옵션 추가 자체는 좋은 제안이다. NMR 적정에서 특정 포인트의 한 peak만 결측인 경우 전체 peak 드롭은 정보 손실이 크다.

**비동의하는 부분**: "기본값을 strict(drop-column)로 유지"라는 Codex 제안에는 의문이 있다. 논문용 도구로서 보수적 기본값이 맞긴 하나, **실제 NMR 데이터에서 부분 결측은 매우 흔하다** (피크 overlap으로 특정 포인트에서 한 peak만 측정 불가). 기본값을 `mask-missing`으로 하고, strict를 옵션으로 제공하는 것이 더 실용적일 수 있다. 이 부분은 도메인 전문가(사용자)의 판단이 필요하다.

---

### P1-2. abort-on-first-failure 정책화 — ✅ 동의

내 CODE_REVIEW(§4.3)에서도 동일하게 식별한 이슈다.

Codex가 제안한 "기본값 fail-fast 유지 + continue 옵션 추가"에 동의한다. 추가로:

- `continue` 모드에서 `g_prev`를 **마지막 성공 값으로 유지**하는 것이 핵심 (Codex 제안서에서 이 부분이 누락됨)
- 실패 인덱스/비율 리포트 노출은 반드시 필요

---

### P1-3. report_pipeline 커버리지 + 타입 — ✅ 동의

`object` 타입 인자가 많은 것은 내 리뷰에서도 확인했다. `FitResultLike` Protocol 도입으로 정적 안정성을 높이는 것에 동의한다. 75%+ 커버리지 목표도 합리적이다.

---

### P2-1. 테스트 보강 — ✅ 동의

내 CODE_REVIEW(§4.10)에서도 동일하게 식별. stress matrix 테스트와 replicate + bootstrap E2E 추가에 전적으로 동의한다.

---

### §5 "바꾸지 말아야 할 것" — σ² 카운팅에 대한 ❌ 비동의

Codex는 "shared σ²를 별도 1개 모수로 카운트하는 현재 정책(k=p+1)은 유지 권장"이라고 했으나, **이것을 "바꾸지 말아야 할 것"으로 고정하는 것은 성급하다**.

**근거**:
1. 내 CODE_REVIEW(§4.1)에서 이를 CRITICAL로 분류한 이유는, 화학 분야의 표준 참고문헌(Thordarson 2011, Hibbert & Thordarson 2016)에서 BIC 정의가 σ²를 포함/미포함 두 가지로 사용되기 때문
2. 모든 모델이 동일하게 σ² 1개를 추가하므로 **모델 간 상대 순위에는 영향이 없다** — 이 점에서는 Codex의 "유지" 판단이 실질적으로 맞다
3. 그러나 **타 소프트웨어(예: Bindfit, HypNMR)와의 BIC 절대값 비교** 시 불일치가 발생할 수 있으므로, 최소한 이 정의를 문서화해야 한다

**결론**: "유지"는 실용적으로 수용 가능하나, "바꾸지 말아야 할 것"으로 명시하기보다는 **"현재 정의를 유지하되, 논문/리포트에 명확히 문서화 필요"**로 격하해야 한다.

---

## Codex가 누락한 이슈 (CODE_REVIEW에서 식별, 제안서에 없음)

| 이슈 | CODE_REVIEW 번호 | 우선순위 |
|------|-----------------|---------|
| `_GridDataset` 중복 정의 | §4.2 | CRITICAL — Protocol 불일치 위험 |
| Bootstrap `max_nfev=2000` 하드코딩 | §4.4 | MEDIUM |
| `svd_diagnosis` / `quadratic_nonlinearity` 미사용 | §4.5 | MEDIUM |
| R² 음수 가능성 무경고 | §4.6 | MINOR |
| nb 모델 SpeciesResult 의미 혼란 | §4.7 | MINOR |
| `_points_bootstrap` 중복 농도 포인트 | §4.9 | MINOR |

특히 **`_GridDataset` 중복**(§4.2)은 Codex 제안서에서 전혀 언급되지 않았으나, Protocol 기반 설계에서 동기화 누락 시 런타임 오류로 이어질 수 있으므로 P1 이상으로 추가 검토 권장.

---

## §7 크로스체크 질문에 대한 답변

### Q1. solve_11의 안정형 근 공식 전환이 실제로 유의미한가?

**대부분의 실제 NMR 적정 데이터에서는 유의미하지 않다.** 일반적 K 범위(10¹–10⁸)와 농도 범위(mM–μM)에서는 현재 구현으로 충분하다. 다만 K > 10¹⁰이거나 농도비가 1000:1을 넘는 극단적 케이스에서는 유효숫자 손실이 발생할 수 있으므로, **회귀 테스트를 추가하여 방어 수준을 확인**하는 것이 현실적 접근이다. 전면 공식 전환보다는 조건부 분기(큰 K일 때만 안정형 사용)가 비용 대비 효과적이다.

### Q2. _residual_vector의 예외 분리가 과도한 복잡도 없이 적용 가능한가?

**가능하다.** `except Exception` → `except (RuntimeError, ValueError)`로 변경하는 것은 1줄 수정이며, 기존 테스트 전체 통과를 확인하면 된다. 스케일 기반 페널티도 `np.nanmax(np.abs(ds.y))` 한 줄 추가로 구현 가능하다. 복잡도 증가는 최소.

### Q3. 결측치 정책의 기본값을 strict로 유지하는 것이 최선인가?

**도메인에 따라 다르다.** NMR 적정에서 피크 overlap으로 인한 부분 결측은 흔하므로, `mask-missing`이 기본값으로 더 실용적일 수 있다. 하지만 논문 재현성을 최우선으로 한다면 strict가 안전하다. **사용자에게 선택을 맡기되, 어느 쪽이든 리포트에 어떤 정책을 사용했는지 명시**하는 것이 핵심이다.

### Q4. fail-fast + continue 옵션이 도메인 해석에 혼선을 줄 가능성은?

**혼선 가능성은 낮다**, 단 리포트에 실패 포인트 수/비율을 명확히 표시하는 조건 하에서. 화학자는 "10개 중 2개 포인트에서 수치 해가 발산"이라는 정보를 보고 해당 모델의 신뢰도를 판단할 수 있다. 오히려 현재의 fail-fast가 "왜 이 모델이 실패했는지" 정보를 주지 않으므로 해석에 더 큰 혼선을 줄 수 있다.

### Q5. report_pipeline 타입 강화 시 최소 변경으로 최대 효과를 얻는 경계는?

**`build_report_artifacts`와 `build_decisions`의 인자 타입**이다. 이 두 함수가 `Dict[str, Dict[str, object]]`로 받고 있는 `results_by_key`를 `Dict[str, Dict[str, FitResultLike]]`로 바꾸면, 내부에서 `res.model`, `res.bic`, `res.bootstrap` 등에 접근하는 모든 코드에 정적 타입 안전성이 전파된다. `FitResultLike` Protocol은 `FitResult`의 공개 필드만 정의하면 되므로 10줄 이내로 작성 가능하다.
