# nmrbindfit

NMR chemical shift titration binding fits with model comparison (1:1, 1:2, 2:1, and non-binding).

## Install

This package is pure Python. Install dependencies with pip:

```
pip install numpy pandas scipy matplotlib
```

If you need Excel support:

```
pip install openpyxl
```

## Input format

CSV or XLSX with columns that include:

- Host Conc.  -> total host concentration in M
- Guest Conc. -> total guest concentration in M
- ppm columns -> one or more peak columns (e.g. ppm, ppm_H1, ppm_H2)
- optional sigma column for weighting

Example (CSV):

```
Host Conc.,Guest Conc.,ppm_H1,ppm_H2,sigma
1.0e-3,0.0,7.10,8.22,0.005
1.0e-3,2.5e-4,7.15,8.25,0.005
1.0e-3,5.0e-4,7.20,8.28,0.005
1.0e-3,1.0e-3,7.32,8.34,0.005
```

The CLI auto-detects host/guest columns and ppm columns. You can override with `--host-col`, `--guest-col`, `--ppm-cols`, and `--sigma-col`.

## Example usage

Single file fit with all models and 1000 bootstrap iterations (output folder is auto-named):

```
python -m nmrbindfit.cli fit --input data.csv --bootstrap 1000
```

If you see optimizer overflow warnings, constrain K:

```
python -m nmrbindfit.cli fit --input data.csv --k-min 1e0 --k-max 1e12
```

Multiple replicate files with shared binding constants:

```
python -m nmrbindfit.cli fit --input data1.xlsx data2.xlsx --replicates
```

## Outputs

The output directory is auto-created as `YYYYMMDD_HHMMSS_<input_name>` and contains:
- For multiple inputs, `<input_name>` becomes `<first_input>_plusN`.
- summary.csv (model comparison table)
- decision.txt (recommended model and diagnostics)
- report.html (plots + summary)
- model_*/ (plots, bootstrap correlation matrix, bootstrap histograms)

## Methods summary (manuscript-ready)

NMR chemical shift titration data are analyzed under a fast-exchange assumption, where the observed chemical
shift at each titration point is modeled as a population-weighted average of all species present in solution.
Four candidate models are considered: 1:1 binding (H + G <-> HG), 1:2 binding (H + G <-> HG; HG + G <-> HG2),
2:1 binding (H + G <-> HG; H + HG <-> H2G), and a non-binding control model (linear drift).

Model parameters are estimated by nonlinear least-squares fitting with multiple initializations to mitigate
local-minimum sensitivity. For multi-peak datasets, binding constants are shared across peaks while peak-specific
chemical shifts are fit separately. The 1:1 model is solved analytically, whereas 1:2 and 2:1 models are solved
point-wise using Newton-Raphson on the free guest concentration with mass-balance constraints; when
Newton-Raphson does not converge, a bracketing (bisection) fallback is used. Solver success and failure counts
are recorded to document numerical stability.

Uncertainty in fitted parameters is quantified by bootstrap resampling (default 1000 iterations). When
measurement uncertainties (sigma) are available, residuals are standardized by sigma prior to resampling;
otherwise, unweighted residuals are used. Standard errors and 95% confidence intervals are derived from the
bootstrap distributions, and bootstrap failure counts are reported as warnings when present.

Model selection prioritizes the Bayesian Information Criterion (BIC), computed from the Gaussian log-likelihood
(including sigma terms when provided). When sigma is not provided, per-peak variance is estimated and counted as
additional parameters. Corrected Akaike Information Criterion (AICc) is reported as a supplementary metric for
small-sample validation. For nested comparison of non-binding vs 1:1 models, an F test is reported when the 1:1
model is selected.

For replicate titration datasets, a simultaneous fitting is performed in which binding constants are shared
across replicates while chemical shifts are allowed to vary by replicate.

## Notes

- BIC is reported for model comparison using Gaussian log-likelihood (including sigma terms when provided); when sigma is not provided, variance is estimated per peak and counted as additional parameters. Corrected Akaike Information Criterion (AICc) is reported as a supplementary metric. The nested-model F test is reported when the 1:1 model is selected.
- 1:2 and 2:1 equilibrium points are solved with Newton-Raphson on free guest.
- summary.csv includes confidence intervals and standard errors.
- Residual bootstrap uses sigma-standardized residuals when sigma is provided; parametric bootstrap draws Gaussian noise scaled by sigma.
- bootstrap failures are listed in the model warnings.
- Solver success/failure counts (Newton and fallback) are included in report.html for 1:2 and 2:1 models.
