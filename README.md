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

Rows are dropped only when host/guest (and sigma if provided) are missing. If a ppm column contains missing
values, that peak column is excluded while remaining peaks are retained.

Example (CSV):

```
Host Conc.,Guest Conc.,ppm_H1,ppm_H2,sigma
1.0e-3,0.0,7.10,8.22,0.005
1.0e-3,2.5e-4,7.15,8.25,0.005
1.0e-3,5.0e-4,7.20,8.28,0.005
1.0e-3,1.0e-3,7.32,8.34,0.005
```

The CLI auto-detects host/guest columns and ppm columns. You can override with `--host-col`, `--guest-col`, `--ppm-cols`, and `--sigma-col`. The x-axis is always plotted as equivalents (G/H).

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
- For multiple inputs, `<input_name>` becomes `replicates`.
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
bootstrap distributions, and bootstrap failure counts are reported as warnings when present. Bootstrap refits
apply small random perturbations (std 0.1 in log10 K) to logK initial values to reduce local-minimum bias.

Model selection prioritizes the Bayesian Information Criterion (BIC), computed from the Gaussian log-likelihood
(including sigma terms when provided). When sigma is not provided, per-peak variance is estimated and counted as
additional parameters. Corrected Akaike Information Criterion (AICc) is reported as a supplementary metric for
small-sample validation.

Model selection metrics are computed from a Gaussian log-likelihood. When sigma is provided, point-specific
sigma values are used in the likelihood; otherwise, a separate variance is estimated per peak and counted as an
additional parameter to reflect potential peak-specific noise scales. The effective sample size n used in BIC
and AICc is defined as the number of finite residuals included in the likelihood across all peaks and titration
points (sum of n_points * n_peaks after filtering). BIC is an asymptotic approximation and can be sensitive to
model misspecification, correlated errors, or outliers; therefore, BIC-based rankings should be interpreted
alongside residual patterns and bootstrap uncertainty, with AICc reported as a supplementary small-sample metric.

For replicate titration datasets, a simultaneous fitting is performed in which binding constants are shared
across replicates while chemical shifts are allowed to vary by replicate.

Each model was fit by nonlinear least squares assuming Gaussian errors, and relative support among candidate
models was compared using Gaussian log-likelihood-based BIC (and AICc). Replicate datasets were handled with
simultaneous fitting that shared binding constants across repeats while estimating chemical-shift parameters
independently for each replicate.

## Chemical shift weighting (fast-exchange)

Observed chemical shifts are modeled as population-weighted averages of species under fast exchange. The
implementation uses host-based fractional weighting (e.g., in the 2:1 model the H2G complex contributes
2 * [H2G] / H_tot because it contains two host units). This assumes the observed signal is a host resonance.
If guest resonances are fitted, the weighting scheme must be revised accordingly; therefore the identity of the
fitted nucleus and the weighting definition should be stated explicitly in any report or manuscript.

## Equilibrium solving (per model)

For the 1:1 model, the complex concentration is computed using the analytical closed-form solution. For the 1:2
and 2:1 models, a one-dimensional root in the free guest concentration is solved at each titration point using
Newton-Raphson with multiple starting guesses; if Newton-Raphson fails to converge, a bisection fallback is used.
If both solvers fail at a point, an exception is raised and the failure is counted in the report. The numerical
method, convergence behavior, and failure handling should be described explicitly in any manuscript.

## Objective function and error model

Optimization uses nonlinear least squares via `scipy.optimize.least_squares`. When sigma values are provided,
residuals are weighted by sigma (r = (y_obs - y_calc) / sigma); otherwise, unweighted residuals are used. For
model selection, Gaussian log-likelihoods are computed from these residuals to obtain BIC and AICc. When sigma
is not provided, a separate variance is estimated per peak as RSS/n and included in the likelihood, and these
per-peak variances are counted as additional parameters in the BIC/AICc penalty term. This design choice should
be stated explicitly to ensure reproducibility of the reported criteria.

## Initialization and identifiability

Nonlinear least-squares fits can converge to different solutions depending on initialization. The code performs
multiple starting values for logK parameters, and the scan range and number of restarts should be reported in any
manuscript. For the 1:2 and 2:1 models, logK1 and logK2 can be strongly correlated, leading to identifiability
issues. In such cases, confidence intervals based on local linearization can be misleading and bootstrap
distributions may become non-normal or multimodal; reporting these distributions and parameter correlations
provides a more transparent assessment of uncertainty.

## Bootstrap interpretation notes

Residual bootstrap resamples titration points while applying the same indices across peaks, which preserves
cross-peak covariance structure to some extent. However, the independence of experimental noise across points and
peaks depends on the dataset, so the choice of residual vs points vs parametric bootstrap should be justified in
the manuscript. Bootstrap refits apply small random perturbations (std 0.1 in log10 K) to logK initial values to
mitigate local-minimum bias, but multimodal solutions can still be under-sampled; reporting this limitation
remains important.

## Limitations (numerical stability)

For the 1:2 and 2:1 models, equilibrium concentrations are solved point-wise using Newton-Raphson with a
bisection fallback. In regions of extreme binding strength (very large K) or highly skewed concentration ratios,
the free-guest root can approach the lower bound and the solver may become stiff, leading to slow convergence or
occasional failures. Such cases are logged, and failure counts are reported; these instances indicate numerical
instability in extreme parameter regimes and should be interpreted cautiously.

The solver enforces a lower bound on free guest (1e-18). When the true solution approaches this bound at very
large binding constants, the fit can saturate toward a fully bound regime, and K may be driven only toward larger
values because the data are no longer sensitive to K. This reflects a physical limitation in the measurable K
range rather than a unique, well-determined estimate.

## Notes

- BIC is reported for model comparison using Gaussian log-likelihood (including sigma terms when provided); when sigma is not provided, variance is estimated per peak and counted as additional parameters. Corrected Akaike Information Criterion (AICc) is reported as a supplementary metric.
- 1:2 and 2:1 equilibrium points are solved with Newton-Raphson on free guest.
- summary.csv includes confidence intervals and standard errors.
- Residual bootstrap uses sigma-standardized residuals when sigma is provided; parametric bootstrap draws Gaussian noise scaled by sigma.
- bootstrap failures are listed in the model warnings.
- Solver success/failure counts (Newton and fallback) are included in report.html for 1:2 and 2:1 models.
