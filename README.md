# nmr_bind_fit

NMR chemical shift titration binding fits with model comparison (1:1, 1:2, 2:1, non-binding).

## Install

This package is pure Python. Install dependencies with pip:

```
pip install numpy pandas scipy matplotlib
```

If you need Excel support:

```
pip install openpyxl
```

## Quick Start

Single file fit with all models and 1000 bootstrap iterations:

```
python -m nmrbindfit.cli fit --input data.csv --bootstrap 1000
```

Multiple replicate files with shared binding constants:

```
python -m nmrbindfit.cli fit --input data1.xlsx data2.xlsx --replicates
```

If you see optimizer overflow warnings, constrain K:

```
python -m nmrbindfit.cli fit --input data.csv --k-min 1e0 --k-max 1e12
```

## Input Format

CSV or XLSX with:
- Host Conc. (total host concentration, M)
- Guest Conc. (total guest concentration, M)
- one or more ppm columns (e.g., ppm, ppm_H1)
- optional sigma column for weighting

Rows are dropped only when host/guest (and sigma if provided) are missing. If any ppm column contains missing
values, that peak column is excluded while remaining peaks are retained. The x-axis is always equivalents ([G]<sub>t</sub>/[H]<sub>t</sub>).

Example (CSV):

```
Host Conc.,Guest Conc.,ppm_H1,ppm_H2,sigma
1.0e-3,0.0,7.10,8.22,0.005
1.0e-3,2.5e-4,7.15,8.25,0.005
1.0e-3,5.0e-4,7.20,8.28,0.005
1.0e-3,1.0e-3,7.32,8.34,0.005
```

## Outputs

The output directory is auto-created as `YYYYMMDD_HHMMSS_mmm_<input_name>` (or `..._replicates`) and contains:
- summary.csv (model comparison table)
- decision.txt (recommended model and diagnostics)
- report.html (plots + summary)
- model_*/ (plots, bootstrap histograms, correlation matrix)

## Workflow (flowchart)

```mermaid
flowchart TD
    A[CLI 실행: nmrbindfit fit] --> B[입력 파일 로딩<br/>CSV/XLSX]
    B --> C{컬럼 검증}
    C -->|Host/Guest/sigma 결측| C1[해당 행 제거]
    C -->|ppm 결측| C2[해당 ppm 컬럼 제거]
    C --> D[Dataset 생성<br/>x = [G]_t / [H]_t]

    D --> E{replicates?}
    E -->|Yes| F[동시 적합 준비<br/>K 공유, shift는 개별]
    E -->|No| G[각 파일별 적합]

    F --> H[모델 반복: 1:1, 1:2, 2:1, non-binding]
    G --> H

    H --> I[멀티스타트 logK 초기값]
    I --> J[비선형 최소제곱<br/>scipy.optimize.least_squares]
    J --> K[예측 계산]
    K --> K1[1:1: 닫힌해]
    K --> K2[1:2, 2:1: Newton–Raphson<br/>실패 시 bisection]
    K --> L{예측값 유한?}
    L -->|No| L1[모델 실패 처리]
    L -->|Yes| M[잔차 계산<br/>sigma 있으면 가중]

    M --> N[통계량 계산<br/>RSS/RMSE/BIC/AICc]
    M --> O[부트스트랩<br/>residual/points/parametric]
    M --> P[플롯 생성<br/>isotherm/residual/fraction/boot]
    M --> Q[요약/리포트 생성<br/>summary.csv/report.html/decision.txt]

    L1 --> Q
```

## Methods Summary

NMR chemical shift titration data are analyzed under a fast-exchange assumption, with observed shifts modeled as host-weighted population averages (host resonances only). Candidate models include 1:1 binding (H + G ⇌ HG), 1:2 binding (H + G ⇌ HG; HG + G ⇌ HG<sub>2</sub>), 2:1 binding (H + G ⇌ HG; H + HG ⇌ H<sub>2</sub>G), and a non-binding linear drift. Parameters are estimated by nonlinear least squares (scipy.optimize.least_squares), using sigma-weighted residuals when provided and unweighted residuals otherwise. The 1:1 model is solved analytically, while 1:2 an 2:1 are solved point-wise for free guest using Newton-Raphson with a bisection fallback; numerical failures are logged and affected models may be excluded from selection. Uncertainty is quantified by bootstrap resampling (default 1000) with small logK perturbations in each refit; residual bootstrap uses sigma-standardized residuals when available. Model comparison uses Gaussian log-likelihood BIC as the primary criterion with AICc as a supplementary metric; n is the total finite residual count across peaks and points, and when sigma is absent per-peak variance is estimated and counted as extra parameters. BIC/AICc provide relative support under a Gaussian error model rather than proof of a true model. Replicate titrations are handled by simultaneous fitting with shared binding constants and replicate-specific chemical shifts. At very large K or extreme concentration ratios, the free-guest root can approach the lower bound (1e-18), leading to saturation and reduced sensitivity to K.
