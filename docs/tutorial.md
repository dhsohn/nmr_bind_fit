# Tutorial: fitting synthetic NMR titration data

This tutorial runs `nmr_bind_fit` on synthetic host-guest NMR titration examples and explains how to read the main outputs.

## 1. Install the package

From a local clone:

```bash
python -m pip install -e ".[test,excel]"
```

Or from GitHub:

```bash
python -m pip install git+https://github.com/dhsohn/nmr_bind_fit.git
```

The tutorial commands assume you are working from a clone or downloaded source
tree that contains the repository's `examples/` directory. A direct GitHub
`pip install` installs the command-line tool, but it does not create an
`examples/` directory in your current working directory.

## 2. Inspect the example data

The example files live in `examples/` and use the required column names:

- `[H]t`: total host concentration in molar (M);
- `[G]t`: total guest concentration in molar (M);
- `ppm_*`: observed chemical-shift columns.

Convert other units (mM, µM) to molar before fitting.

For example:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv('examples/synthetic_11.csv').head())
PY
```

## 3. Run a single-file fit

```bash
nmr_bind_fit --input examples/synthetic_11.csv --bootstrap 0
```

This fits four candidate explanations to the same data:

1. 1:1 binding (`H + G <=> HG`)
2. sequential 1:2 binding (`H + G <=> HG`; `HG + G <=> HG2`)
3. sequential 2:1 binding (`H + G <=> HG`; `H + HG <=> H2G`)
4. non-binding linear drift

The output directory is named with a timestamp and the input stem, for example `20260630_120000_synthetic_11/`.

## 4. Read the output files

The most important outputs are:

- `report.html`: human-readable report with the provisional working model, methods text, plots, tables, and warnings;
- `model_*/dataset_*/`: dataset-scoped isotherm plots, residual plots, bound-fraction plots, and bootstrap histograms when available.

The lowest finite-BIC model is reported as the provisional working model among tested candidates. It is not an automatic chemical truth claim.

## 5. Add bootstrap uncertainty

```bash
nmr_bind_fit --input examples/synthetic_11.csv --bootstrap 200 --seed 1
```

For publication-facing runs, use more bootstrap iterations, for example `--bootstrap 1000`. Bootstrap intervals should be interpreted together with fit quality, saturation, spectral behavior, and model plausibility.

## 6. Try the non-binding control example

```bash
nmr_bind_fit --input examples/synthetic_nonbinding.csv --bootstrap 200 --seed 1
```

This example is useful for checking that a workflow includes a non-binding alternative rather than forcing a binding constant onto drift-like data.

## 7. Replicate/global fitting

When multiple replicate titrations should share binding constants but have replicate-specific chemical shifts, use:

```bash
nmr_bind_fit --input replicate_a.csv replicate_b.csv --replicates
```

This performs a simultaneous fit with shared `K` values and dataset-specific δ parameters.

## 8. Interpretation checklist

Before accepting a binding model, check:

- whether fast exchange is chemically justified by the spectra;
- whether the titration reaches a saturation region;
- whether the selected model is clearly separated from the next-best model by ΔBIC;
- whether the non-binding model is competitive;
- whether bootstrap confidence intervals are narrow enough to support the reported K;
- whether fitted δ values and stoichiometry are chemically plausible.

The software provides model-comparison evidence and diagnostic warnings; final chemical interpretation remains the user's responsibility.
