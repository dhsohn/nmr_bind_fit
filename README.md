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

## Methods Summary

NMR chemical shift titration data are analyzed under a fast-exchange assumption, with observed shifts modeled as
host-weighted population averages (host resonances only). Candidate models include 1:1 binding (H + G ⇌ HG),
1:2 binding (H + G ⇌ HG; HG + G ⇌ HG<sub>2</sub>), 2:1 binding (H + G ⇌ HG; H + HG ⇌ H<sub>2</sub>G), and a non-binding linear
drift. Parameters are estimated by nonlinear least squares (scipy.optimize.least_squares), using sigma-weighted
residuals when provided and unweighted residuals otherwise. The 1:1 model is solved analytically, while 1:2 and
2:1 are solved point-wise for free guest using Newton-Raphson with a bisection fallback; numerical failures are
logged and affected models may be excluded from selection. Uncertainty is quantified by bootstrap resampling
(default 1000) with small logK perturbations in each refit; residual bootstrap uses sigma-standardized residuals
when available. Model comparison uses Gaussian log-likelihood BIC as the primary criterion with AICc as a
supplementary metric; n is the total finite residual count across peaks and points, and when sigma is absent
per-peak variance is estimated and counted as extra parameters. BIC/AICc provide relative support under a
Gaussian error model rather than proof of a true model. Replicate titrations are handled by simultaneous fitting
with shared binding constants and replicate-specific chemical shifts. At very large K or extreme concentration
ratios, the free-guest root can approach the lower bound (1e-18), leading to saturation and reduced sensitivity
to K.
