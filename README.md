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
python -m nmrbindfit.cli --input data.csv --bootstrap 1000
```

Multiple replicate files with shared binding constants:

```
python -m nmrbindfit.cli --input data1.xlsx data2.xlsx --replicates
```

For reproducible, paper-oriented analysis, the CLI uses fixed strict policies: ppm columns
with missing values are dropped before fitting, and point-wise nonlinear solver failures in
1:2/2:1 models use fail-fast behavior. log10 K is constrained to [0, 12] (K in [1e0, 1e12]).

## Input Format

CSV or XLSX with:
- [H]t (total host concentration, M)
- [G]t (total guest concentration, M)
- one or more ppm columns (e.g., ppm, ppm_H1)

The concentration column names are fixed and must be exactly `[H]t` and `[G]t`.

Rows are dropped when host/guest values are missing. If any ppm column contains missing
values, that peak column is excluded while remaining peaks are retained. Point-wise solver
failures in 1:2/2:1 models use fail-fast behavior. The x-axis is always equivalents
([G]<sub>t</sub>/[H]<sub>t</sub>).

Example (CSV):

```
[H]t,[G]t,ppm_H1,ppm_H2
1.0e-3,0.0,7.10,8.22
1.0e-3,2.5e-4,7.15,8.25
1.0e-3,5.0e-4,7.20,8.28
1.0e-3,1.0e-3,7.32,8.34
```

## Outputs

The output directory is auto-created as `YYYYMMDD_HHMMSS_mmm_<input_name>` (or `..._replicates`) and contains:
- summary.csv (model comparison table)
- decision.txt (recommended model and diagnostics)
- report.html (plots + summary)
- model_*/ (plots, bootstrap histograms, correlation matrix)

## Flowchart

```mermaid
flowchart TD
    A[CLI nmrbindfit] --> B[Load input files CSV or XLSX]
    B --> C{Validate columns}
    C -->|Host or Guest missing| C1[Drop row]
    C -->|ppm missing| C2[Drop ppm column]
    C --> D[Build dataset x = equivalents]

    D --> E{Replicates?}
    E -->|Yes| F[Prepare simultaneous fit with shared K]
    E -->|No| G[Fit each file separately]

    F --> H[Loop models: 1:1, 1:2, 2:1, non-binding]
    G --> H

    H --> I[Multistart logK initials]
    I --> J[Nonlinear least squares using least_squares]
    J --> K[Predict shifts]
    K --> K1[1:1: closed form]
    K --> K2[1:2 and 2:1: Newton-Raphson then bisection]
    K --> L{Finite predictions?}
    L -->|No| L1[Mark model failed]
    L -->|Yes| M[Residuals]

    M --> N[Stats RSS RMSE BIC AICc]
    M --> O[Bootstrap resampling]
    M --> P[Plots isotherm residual fraction boot]
    M --> Q[Outputs summary.csv report.html decision.txt]

    L1 --> Q
```

## Methods Summary

NMR chemical shift titration data are interpreted under a fast-exchange assumption, with observed host-resonance shifts modeled as population-weighted averages of chemical states. Candidate stoichiometries are 1:1 binding (H + G <=> HG), 1:2 binding (H + G <=> HG; HG + G <=> HG2), 2:1 binding (H + G <=> HG; H + HG <=> H2G), and a non-binding linear drift model. Parameters are estimated by nonlinear least squares (scipy.optimize.least_squares). The 1:1 model is solved analytically, while 1:2 and 2:1 are solved numerically point-by-point with Newton-Raphson and bisection fallback. Uncertainty is evaluated by bootstrap resampling (default 1000) with small logK perturbations in each refit. Model comparison uses BIC as the primary ranking index with AICc as supporting information. These criteria indicate relative support among tested candidates and should be interpreted together with chemical plausibility and spectral consistency. One shared residual variance term is estimated for model comparison and counted as one additional information-criteria parameter (k = p + 1). Replicate titrations are handled by simultaneous fitting with shared binding constants and replicate-specific chemical shifts. At very large K or extreme concentration ratios, the free-guest root can approach the lower bound (1e-18), leading to saturation and reduced sensitivity to K.
