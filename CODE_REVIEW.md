# nmrbindfit — 코드 분석 및 설계 점검 보고서

**작성일**: 2026-02-11  
**대상**: nmrbindfit v0.1.0  
**범위**: 소스 12개 파일, 테스트 9개 파일 (26 tests, 전체 PASS)

---

## 1. 프로젝트 개요

NMR chemical shift titration binding fit 패키지.  
호스트–게스트 결합 시스템의 NMR 화학적 이동 적정 데이터를 해석하여, 결합 화학량론과 결합 상수를 결정한다.

### 지원 모델

| 모델 | 설명 | 평형식 | logK 수 | peak당 δ 수 |
|------|------|--------|---------|-------------|
| `11` | 1:1 결합 | H + G ⇌ HG | 1 | 2 |
| `12` | 1:2 결합 | H + G ⇌ HG; HG + G ⇌ HG₂ | 2 | 3 |
| `21` | 2:1 결합 | H + G ⇌ HG; H + HG ⇌ H₂G | 2 | 3 |
| `nb` | 비결합 (선형 드리프트) | — | 0 | 2 |

### 기술 스택

- Python ≥ 3.9
- numpy, pandas, scipy, matplotlib
- 빌드: setuptools (pyproject.toml)
- 테스트: pytest

---

## 2. 아키텍처

### 모듈 의존성 구조

```
cli.py ─────────────┬──> fit.py ──────> models.py ──> equilibrium.py
                    │                              └──> types.py (DatasetLike Protocol)
                    ├──> io.py (Dataset 로딩)
                    ├──> report_pipeline.py ──> plots.py ──> models.py
                    │                       └──> report.py
                    └──> report.py (HTML/CSV/TXT 출력)
```

### 핵심 데이터 흐름

1. **입력**: CSV/XLSX → `io.load_datasets()` → `Dataset` dataclass
2. **피팅**: `fit.fit_models()` → multistart least_squares → `FitResult` dataclass
3. **평형해석**: `equilibrium.solve_11/12/21()` — 1:1은 해석적, 1:2/2:1은 Newton-Raphson + bisection fallback
4. **부트스트랩**: `fit.bootstrap_params()` — residual/parametric/points 리샘플링
5. **리포트**: `report_pipeline.build_report_artifacts()` + `report.write_report_html()`
6. **출력**: 타임스탬프 디렉토리에 summary.csv, decision.txt, report.html, 모델별 plots

### 주요 설계 패턴

- **`DatasetLike` Protocol** (`types.py`): 모듈 간 느슨한 결합을 위한 구조적 서브타이핑
- **Multistart 최적화**: log₁₀K 공간에서 그리드 탐색 후 최적 수렴 결과 선택
- **실패 허용(graceful degradation)**: 개별 모델 실패 시 `FitResult(success=False)`로 기록하고 계속 진행

---

## 3. 모듈별 분석

### 3.1 `equilibrium.py` (426 lines)

평형 농도 계산기. 세 가지 해석기 제공:

- `solve_11`: 이차방정식 해석해 (catastrophic cancellation 방지 구현)
- `solve_12` / `solve_21`: 포인트별 Newton-Raphson → bisection fallback
- `SolverStats`: 솔버 성능 통계 수집 (newton 성공/실패, fallback 사용)

**수치 안정성 기법**:
- log-space 종 스케일링 (`_scale_species_from_logs`)
- logsumexp 안정 합산
- k₁·k₂ 오버플로우 시 rescaled polynomial 사용 (`solve_12_point` L203)
- 자유 게스트 하한 `1e-18` 적용

### 3.2 `fit.py` (695 lines)

핵심 피팅 엔진:

- `fit_model`: 단일 모델 피팅 (multistart → best selection → 통계 → bootstrap)
- `fit_models`: 전체 모델 반복 (단일 데이터셋 or replicate 동시 피팅)
- `bootstrap_params`: 리샘플링 → refit → CI 계산

**매개변수 벡터 구조**: `[logK₁, logK₂, ..., δ_peak1_species1, δ_peak1_species2, ..., δ_peak2_species1, ...]`

**BIC/AICc 계산**: `_information_criteria()` 함수에서 전체 잔차를 하나의 공유 분산(σ²)으로 추정 후, 가우시안 로그우도를 기반으로 계산.  
여기서 **p** = 모델의 추정 매개변수 총 수 (logK 개수 + 모든 peak의 δ 매개변수 수 + σ² 1개).

### 3.3 `models.py` (178 lines)

모델 정의와 예측:

- `ModelSpec` dataclass: 모델 메타데이터
- `predict_dataset`: 모델별 예측 shift 계산 (species fraction × δ)
- `fraction_bound`: 호스트 기준 결합 분율

### 3.4 `io.py` (234 lines)

데이터 입출력:

- 퍼지 컬럼명 매칭 (`_norm_col`)으로 "Host Conc.", "host_concentration" 등 자동 인식
- 결측 농도 행 드롭, 결측 ppm 컬럼 드롭
- 농도 검증 (양수, 유한)

### 3.5 `stats.py` (112 lines)

통계 진단:

- `gaussian_loglik`: 공유 분산 가우시안 로그우도
- `bic_from_loglik` / `aicc_from_loglik`: 정보 기준 계산
- `quadratic_nonlinearity`: 곡률 진단 (구현됨, 미사용)
- `svd_diagnosis`: SVD 기반 유의 성분 수 진단 (구현됨, 미사용)

### 3.6 `cli.py` (285 lines)

CLI 진입점:

- argparse 기반 인터페이스
- glob 패턴 확장, 중복 파일 검출
- 자동 타임스탬프 출력 디렉토리

### 3.7 `report_pipeline.py` (492 lines)

리포트 조립:

- 피팅 결과 → summary row, model entry, decision entry 변환
- 플롯 생성 (isotherm, residual, fraction bound, bootstrap histogram)
- 상관행렬 CSV 출력
- BIC 기반 모델 선택 결정문 생성 (provisional language)

### 3.8 `report.py` (196 lines)

출력 형식:

- `write_summary_csv`: 모델 비교 요약 테이블
- `write_decision_txt`: 텍스트 결정문
- `write_report_html`: 단일 HTML 리포트 (인라인 CSS)

### 3.9 `plots.py` (201 lines)

시각화:

- `plot_isotherms`: 데이터 + 피팅 곡선
- `plot_residuals`: 잔차 산점도
- `plot_fraction_bound`: 결합 분율
- `plot_bootstrap_hist`: K bootstrap 히스토그램
- PNG + PDF 동시 출력

---

## 4. 발견된 이슈

### 4.1 [CRITICAL] BIC/AICc에서 σ² 매개변수 카운팅 주의

**파일**: `fit.py` L313-329 (`_information_criteria`)

현재 BIC 계산 시 `bic_p = p + n_sigma` (L324)로, σ²를 추가 추정 매개변수로 카운트합니다. `gaussian_loglik`에서 `σ² = RSS/n` (MLE)으로 추정하므로 이는 통계적으로 정당하지만, 화학 분야의 일부 참고문헌에서는 σ²를 별도로 카운트하지 않는 관행이 있습니다.

```python
# fit.py L324
bic_p = p + n_sigma  # n_sigma=1 → σ²를 한 개의 추가 매개변수로 카운트
```

**영향**: 모델 간 매개변수 수가 동일하면 영향 없으나, 다른 소프트웨어와 BIC 값 비교 시 불일치 가능.

**해결방안**: 논문에서 사용할 BIC 정의를 명확히 하고, `METHODS_TEXT`에 "p는 모델 매개변수 수 + σ² 1개를 포함"이라고 명시하거나, σ² 미포함 방식으로 전환.

---

### 4.2 [CRITICAL] `_GridDataset`과 `Dataset`의 중복 정의

**파일**: `plots.py` L30-48

`_GridDataset`이 `Dataset`과 동일한 필드와 프로퍼티를 수동 재구현합니다.

```python
# plots.py — _GridDataset
@dataclass
class _GridDataset:
    name: str
    path: Path
    h_tot: np.ndarray
    # ... Dataset과 동일한 필드 반복
```

**위험**: `Dataset`에 필드 추가 시 `_GridDataset`을 동기화하지 않으면 `DatasetLike` Protocol 불일치 발생.

**해결방안**: `Dataset` 생성자를 직접 사용:

```python
def _grid_dataset(ds, n=200):
    ...
    return Dataset(name=ds.name, path=ds.path, h_tot=h_vals, g_tot=g_vals,
                   x=x_vals, y=np.zeros(...), y_cols=ds.y_cols, dropped_peaks=ds.dropped_peaks)
```

---

### 4.3 [MEDIUM] `solve_12`/`solve_21`의 abort-on-first-failure

**파일**: `equilibrium.py` L385-386 (solve_12), L417-418 (solve_21)

```python
except RuntimeError:
    break  # 하나의 포인트 실패 시 전체 중단
```

**문제**: 데이터가 등비급수적(geometric) 농도로 나열된 경우, 중간의 한 포인트 실패가 이후 전체를 NaN으로 만듦.

**해결방안**:
- `break` → `continue`로 변경하여 실패 포인트만 NaN 유지
- 단, `g_prev` carry-forward 로직이 `None`으로 리셋되므로 다음 포인트에서 x0 초기 추측이 사라지는 부작용 고려 필요
- 대안: `g_prev`를 마지막 성공 값으로 유지

---

### 4.4 [MEDIUM] Bootstrap `max_nfev` 하드코딩

**파일**: `fit.py` L588

```python
params_fit, res = _fit_with_initial(model, boot_datasets, params0, max_nfev=2000, ...)
```

메인 fit은 `max_nfev=5000` (기본값)이지만 bootstrap refit은 2000으로 고정. 복잡한 1:2/2:1 모델에서 수렴 실패율 증가 가능.

**해결방안**: `bootstrap_params`의 인자로 `max_nfev`를 전달받거나, 메인 값의 일정 비율(예: 40%)을 사용.

---

### 4.5 [MEDIUM] `svd_diagnosis`와 `quadratic_nonlinearity` 미사용

**파일**: `stats.py` L64-111

두 함수 모두 구현되어 있지만 `cli.py`, `report_pipeline.py`, `fit.py` 어디에서도 호출되지 않음.

**해결방안**:
- 리포트에 진단 결과를 추가하여 활용 (SVD 기반 유의 성분 수 → 모델 복잡도 추천)
- 또는 미래 사용 계획이 없으면 제거하여 코드 정리

---

### 4.6 [MINOR] R² 음수 가능성 무경고

**파일**: `fit.py` L151-165

비선형 모델에서 R² < 0이 가능하나, 리포트에서 별도 경고 없이 출력됨.

**해결방안**: `report_pipeline._build_model_warnings()`에 R² < 0 경고 추가.

---

### 4.7 [MINOR] nb 모델의 SpeciesResult 의미 혼란

**파일**: `models.py` L155-159

`nb` 모델이 `SpeciesResult(h=h_tot, g=g_tot, hg=zeros)`를 반환하는데, 비결합 모델에서 species 개념이 적용되지 않으므로 의미적으로 혼란. `hg2`, `h2g`는 `None`(기본값).

**해결방안**: 주석으로 "nb 모델은 선형 드리프트이므로 species는 형식적 반환"임을 명시.

---

### 4.8 [MINOR] `_compute_equivalents`의 도달 불가능 분기

**파일**: `io.py` L166-169

```python
return np.where(h_tot != 0, g_tot / h_tot, 0.0)
```

`_validate_concentration_arrays`에서 `h_tot > 0` 검증을 거치므로 `h_tot == 0` 분기는 도달 불가. 방어적 코드로 유지 가능.

---

### 4.9 [MINOR] `_points_bootstrap`에서 중복 농도 포인트 발생

**파일**: `fit.py` L497-512

`points` bootstrap은 `h_tot`, `g_tot`, `x`, `y`를 동시 리샘플링하므로 중복 농도 포인트가 발생. 플로팅 시 비정상적 커브 가능.

**해결방안**: points bootstrap 결과 플로팅 시 원본 데이터셋의 x축 사용.

---

### 4.10 [MINOR] 테스트 커버리지 미비

| 미테스트 영역 | 위험도 | 비고 |
|-------------|--------|------|
| `plots.py` 전체 | Low | 시각 출력이므로 functional test 어려움 |
| `report.py` HTML 생성 | Low | 구조적 테스트 가능 |
| `io.py` XLSX 경로 | Medium | openpyxl 의존성 |
| end-to-end 통합 테스트 | **High** | CSV → fit → report 전체 흐름 |

**해결방안**: 최소한 작은 CSV → `run_fit()` → 출력 파일 존재 확인하는 통합 테스트 추가.

---

## 5. 전체 평가

| 영역 | 평가 | 비고 |
|------|------|------|
| 모듈 구조 | ✅ 우수 | 명확한 관심사 분리, 단방향 의존성 |
| Protocol 활용 | ✅ 우수 | `DatasetLike` 구조적 서브타이핑 |
| 수치 안정성 | ✅ 양호 | log-space, bisection fallback, overflow guard |
| 에러 처리 | ⚠️ 보통 | abort-on-first-failure, 방어적 코드 일부 미비 |
| 통계 정확성 | ⚠️ 주의 | BIC p-count 정의 명확화 필요 |
| 테스트 | ⚠️ 보통 | 단위 테스트 양호 (26/26 pass), integration 부족 |
| 코드 중복 | ⚠️ 보통 | `_GridDataset` 중복 |
| 미사용 코드 | 🔵 참고 | `svd_diagnosis`, `quadratic_nonlinearity` |

---

## 6. 용어 정리

| 용어 | 정의 |
|------|------|
| **p** (BIC 공식) | 모델의 총 추정 매개변수 수. `logK 개수 + 전체 peak의 δ 매개변수 수 + σ²(1개)`. 예: 1:1 모델, 2 peaks → p = 1(logK) + 2×2(δ) + 1(σ²) = 6 |
| **n** (BIC 공식) | 관측치 총 수. `데이터 포인트 수 × peak 수` |
| **BIC** | Bayesian Information Criterion = -2·logL + p·ln(n). 낮을수록 선호 |
| **AICc** | 소표본 보정 AIC = -2·logL + 2p + 2p(p+1)/(n-p-1) |
| **RSS** | Residual Sum of Squares = Σ(observed - predicted)² |
| **RMSE** | Root Mean Square Error = √(RSS/n) |
| **logK** | log₁₀(K). 최적화는 logK 공간에서 수행 |
| **δ (delta)** | 각 chemical species의 고유 화학적 이동값 (ppm) |
| **equivalents** | x축 = [G]_total / [H]_total |
