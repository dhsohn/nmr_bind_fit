# NMR Bind Fit 코드 수정계획서

## 개요

초분자화학 논문 출판을 위한 코드 결점 수정계획.
심각도 순으로 정리하며, 각 항목에 수정 대상 파일, 현재 문제, 수정 방향을 기술한다.

---

## 1. Methods 텍스트의 solver 기술 오류 (치명적)

**파일:** `core/report_pipeline.py` (470행 부근)

**현재 문제:**
자동 생성되는 Methods 섹션에 "Newton–Raphson iteration with bisection fallback"이라고
기재되어 있으나, 실제 코드는 `scipy.optimize.brentq`를 사용한다.
커밋 `daddc08`에서 Newton+bisection을 brentq로 교체했지만 텍스트를 갱신하지 않았다.

**수정 방향:**
- `_compose_methods_sections` 함수의 "Parameter Estimation" 섹션에서
  "Newton–Raphson iteration with bisection fallback"을
  "Brent's method (`scipy.optimize.brentq`)"로 교체한다.
- 허용 오차(xtol=1e-50, rtol=1e-15)도 함께 기재한다.

**수정 범위:** 문자열 1개 교체

---

## 2. R² 계산의 inter-peak 부풀림 (높음)

**파일:** `core/fit.py` (204-221행)

**현재 문제:**
모든 peak의 관측값을 하나의 배열로 합쳐 grand mean 기준 SS_tot을 계산한다.
Peak 간 baseline 차이(예: 7.0 ppm vs 8.5 ppm)가 SS_tot에 포함되어
R²가 실제 fitting 품질과 무관하게 ~1.0으로 부풀려진다.

**수정 방향:**
- `_r2_score` 함수를 수정하여 **per-peak R²를 계산**한 뒤 평균을 취한다.
- 각 peak별로 SS_res와 SS_tot을 독립 계산한다.
- `FitResult`에 `r2_per_peak: List[float]` 필드를 추가하여 per-peak R²도 보존한다.
- 기존 `r2` 필드는 per-peak 평균 R²로 대체한다.

**수정 예시:**
```python
def _r2_score(datasets, y_pred_list):
    r2_peaks = []
    for ds, y_pred in zip(datasets, y_pred_list):
        for j in range(ds.n_peaks):
            col_obs = ds.y[:, j]
            col_pred = y_pred[:, j]
            mask = np.isfinite(col_obs) & np.isfinite(col_pred)
            if not np.any(mask):
                continue
            obs = col_obs[mask]
            pred = col_pred[mask]
            ss_res = np.sum((obs - pred) ** 2)
            ss_tot = np.sum((obs - np.mean(obs)) ** 2)
            if ss_tot > 0:
                r2_peaks.append(1.0 - ss_res / ss_tot)
    if not r2_peaks:
        return float("nan")
    return float(np.mean(r2_peaks))
```

**수정 범위:** `fit.py` 함수 1개 수정, `FitResult` 필드 추가, 보고서 출력 코드 반영

---

## 3. K 표준오차(SE) 계산 방식 개선 (높음)

**파일:** `core/report_pipeline.py` (151-182행)

**현재 문제:**
bootstrap logK 샘플을 10^logK로 변환한 뒤 선형 스케일에서 std를 계산한다.
K는 log-정규분포를 따르므로 선형 스케일 SE는 비대칭 분포를 대칭 오차로 잘못 요약한다.

**수정 방향:**
- logK 스케일에서의 SE를 **추가로** 보고한다.
- `ParamEntry`에 `se_log: float` 필드를 추가하여 log-space SE를 저장한다.
- 보고서/summary에서 K의 SE 대신 **95% CI만 주력으로 보고**하고,
  SE는 보조 정보로 남긴다. CI는 이미 percentile로 계산되므로
  비대칭 분포를 올바르게 반영한다.
- summary.csv에 `bootstrap_logK_SE` 열을 추가한다.

**수정 범위:** `report_pipeline.py`의 `_build_param_entries`, `_build_summary_row` 수정,
`report.py`의 `ParamEntry` 수정

---

## 4. 절대적 적합도 검정 추가 (중간)

**파일:** `core/stats.py` (신규 함수), `core/fit.py` (결과 저장)

**현재 문제:**
BIC/AICc로 상대적 모델 비교만 수행하며, 최선 모델의 절대적 fit 품질을 평가하지 않는다.

**수정 방향:**
- `stats.py`에 잔차 정규성 검정 함수를 추가한다:
  - Shapiro-Wilk 검정 (`scipy.stats.shapiro`)
  - Durbin-Watson 자기상관 통계량 (`Σ(eᵢ - eᵢ₋₁)² / Σeᵢ²`)
- `FitResult`에 `residual_diagnostics: dict` 필드를 추가한다.
- 보고서에 잔차 진단 결과를 포함한다.
- 검정 결과가 유의하면 경고를 출력한다.

**수정 예시:**
```python
from scipy.stats import shapiro

def residual_diagnostics(residuals: np.ndarray) -> dict:
    flat = residuals[np.isfinite(residuals)]
    result = {}
    if flat.size >= 3:
        stat, p = shapiro(flat[:5000])  # shapiro는 5000개 이하 권장
        result["shapiro_stat"] = stat
        result["shapiro_p"] = p
    if flat.size >= 2:
        diff = np.diff(flat)
        dw = float(np.sum(diff ** 2) / np.sum(flat ** 2))
        result["durbin_watson"] = dw
    return result
```

**수정 범위:** `stats.py` 함수 추가, `fit.py` 결과 저장, 보고서 출력

---

## 5. Smooth curve의 [H]t 처리 개선 (중간)

**파일:** `core/plots.py` (57-74행)

**현재 문제:**
fitted curve를 그릴 때 모든 grid point에서 `median([H]t)`를 사용한다.
[H]t가 희석으로 변하는 실험에서는 실제 데이터 점의 예측값과 curve가 불일치한다.

**수정 방향:**
- [H]t의 변동계수(CV)를 확인한다.
- CV가 임계값(예: 1%) 이하이면 현재 방식 유지 (상수 [H]t 적정).
- CV가 임계값 초과이면, equivalents 대신 **[G]t를 x축으로** 사용하고,
  각 grid point의 [H]t를 선형 보간하여 curve를 생성한다.
- 또는 더 단순하게: curve 대신 **실제 데이터 점에서의 예측값을 연결선으로** 표시한다.

**수정 예시 (단순 접근):**
```python
def _grid_dataset(ds, n=200):
    cv = np.std(ds.h_tot) / np.mean(ds.h_tot)
    if cv < 0.01:
        # 기존 방식: 상수 [H]t
        h_ref = float(np.median(ds.h_tot))
        eq_vals = np.linspace(np.min(ds.x), np.max(ds.x), n)
        h_vals = np.full_like(eq_vals, h_ref)
        g_vals = eq_vals * h_ref
    else:
        # 보간 방식: [H]t를 equivalents에 대해 보간
        eq_vals = np.linspace(np.min(ds.x), np.max(ds.x), n)
        h_vals = np.interp(eq_vals, ds.x, ds.h_tot)
        g_vals = eq_vals * h_vals
    ...
```

**수정 범위:** `plots.py`의 `_grid_dataset` 수정

---

## 6. BCa Bootstrap CI 옵션 추가 (낮음)

**파일:** `core/fit_bootstrap.py` (220-224행)

**현재 문제:**
percentile 방법만 사용한다. 소표본이나 skewed 분포에서 coverage가 부족할 수 있다.

**수정 방향:**
- BCa (bias-corrected and accelerated) CI 계산 함수를 추가한다.
- bias correction factor z₀와 acceleration factor a를 계산한다.
- `BootstrapResult`에 `ci_low_bca`, `ci_high_bca` 필드를 추가한다.
- 기본 CI 방법을 BCa로 변경하되, percentile CI도 유지한다.

**수정 예시:**
```python
def _bca_ci(samples, original_stat, jackknife_stats, alpha=0.05):
    from scipy.stats import norm
    z = norm.ppf
    n = len(samples)
    # bias correction
    z0 = z(np.mean(samples < original_stat))
    # acceleration
    theta_hat = np.mean(jackknife_stats)
    diffs = theta_hat - jackknife_stats
    a = np.sum(diffs ** 3) / (6.0 * np.sum(diffs ** 2) ** 1.5)
    # adjusted percentiles
    z_alpha = z(alpha / 2)
    z_1alpha = z(1 - alpha / 2)
    p_low = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_high = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))
    return np.percentile(samples, 100 * p_low), np.percentile(samples, 100 * p_high)
```

**주의:** BCa는 jackknife 추정이 필요하므로 계산 비용이 증가한다.
논문에서 percentile CI를 사용한 이유를 명시하는 것도 대안이다.

**수정 범위:** `fit_bootstrap.py` 함수 추가, `BootstrapResult` 확장, CLI 옵션 추가

---

## 7. K 단위 자동 감지 또는 경고 (낮음)

**파일:** `core/report_pipeline.py` (473행), `core/cli.py`

**현재 문제:**
보고서에 "K ∈ [1, 10¹²] M⁻¹"로 고정 기재하지만, 실제 K 단위는 입력 농도 단위에 의존한다.

**수정 방향:**
- CLI에 `--concentration-unit` 옵션을 추가한다 (기본값: "M").
- 보고서에 사용자가 지정한 단위를 반영한다.
- 또는, 단위 가정을 보고서에서 제거하고
  "K units are the reciprocal of the input concentration units"로 대체한다.

**수정 범위:** `cli.py` 옵션 추가, `report_pipeline.py` 문자열 수정

---

## 8. Dead code 정리: solver의 x0 파라미터 (낮음)

**파일:** `core/equilibrium.py` (97-98행, 161-166행)

**현재 문제:**
`solve_12_point`과 `solve_21_point`의 `x0` 파라미터가 선언만 되고 사용되지 않는다.
`solve_12`와 `solve_21`에서 `x0=g_prev`를 전달하지만 무시된다.

**수정 방향:**
- `solve_12_point`과 `solve_21_point`에서 `x0` 파라미터를 제거한다.
- `solve_12`와 `solve_21`에서 `x0=g_prev` 전달을 제거한다.
- `g_prev` 변수 관련 코드도 제거한다.

**수정 범위:** `equilibrium.py` 시그니처 및 호출부 정리

---

## 9. RMSE 정의 명시 (낮음)

**파일:** `core/fit.py` (375행), `core/report_pipeline.py`

**현재 문제:**
`rmse = sqrt(RSS / n)`으로 계산하며, n은 총 관측 수이다.
Unbiased 추정치 `sqrt(RSS / (n - p))`와 구별되지만 어떤 정의를 사용했는지 명시하지 않는다.

**수정 방향:**
- 보고서 Methods 섹션에 RMSE 정의를 명시한다:
  "RMSE was calculated as sqrt(RSS/n) using the MLE convention."
- 또는, `sqrt(RSS / (n - p))`로 변경하고 이를 문서화한다.

**수정 범위:** 문자열 추가 또는 수식 변경 1줄

---

## 10. 잔차 cross-peak 상관관계 진단 추가 (낮음)

**파일:** `core/stats.py`, `core/report_pipeline.py`

**현재 문제:**
대각 공분산을 가정하지만 (Methods에 명시됨), 이 가정의 타당성을 검증하지 않는다.

**수정 방향:**
- 잔차 행렬의 peak 간 상관계수를 계산하여 보고서에 포함한다.
- 상관이 유의미하면 경고를 출력한다.

**수정 예시:**
```python
def cross_peak_correlation(residuals: np.ndarray) -> np.ndarray:
    finite_rows = residuals[np.all(np.isfinite(residuals), axis=1)]
    if finite_rows.shape[0] < 3:
        return np.full((residuals.shape[1], residuals.shape[1]), np.nan)
    return np.corrcoef(finite_rows, rowvar=False)
```

**수정 범위:** `stats.py` 함수 추가, 보고서 출력

---

## 수정 우선순위 및 일정

| 순서 | 항목 | 심각도 | 예상 난이도 |
|------|------|--------|-------------|
| 1 | Methods solver 기술 수정 | 치명적 | 단순 |
| 2 | R² per-peak 계산 | 높음 | 보통 |
| 3 | K SE log-space 보고 | 높음 | 보통 |
| 4 | 적합도 검정 추가 | 중간 | 보통 |
| 5 | Smooth curve [H]t 처리 | 중간 | 보통 |
| 6 | BCa bootstrap CI | 낮음 | 높음 |
| 7 | K 단위 처리 | 낮음 | 단순 |
| 8 | Dead code 정리 | 낮음 | 단순 |
| 9 | RMSE 정의 명시 | 낮음 | 단순 |
| 10 | Cross-peak 상관 진단 | 낮음 | 보통 |

1-3번은 논문 제출 전 필수, 4-5번은 강력 권장, 6-10번은 선택적이다.
