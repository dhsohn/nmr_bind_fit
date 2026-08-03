# Changelog

All notable changes to `nmr_bind_fit` will be documented in this file.

The format follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses semantic versioning once releases are tagged.

## [0.2.0] - 2026-08-03

### Added

- Community-maintenance scaffolding: GitHub issue forms (bug report, feature request), a pull-request template, and Dependabot configuration for GitHub Actions and pip.
- Python version badge in the README.
- Zenodo DOI metadata for the archived v0.1.0 release (10.5281/zenodo.21071370).
- JOSS-readiness project metadata, including license, citation metadata, contribution guide, and CI configuration.
- Example synthetic titration datasets and tutorial documentation.
- Draft JOSS paper, BibTeX references, Zenodo metadata, and release/DOI checklist.
- `py.typed` marker so downstream users receive the package's type information (PEP 561).

### Changed

- Replaced bootstrap resampling with asymptotic uncertainty from the covariance matrix of the
  converged fit, `cov = (RSS / dof) * inv(J^T J)`, with Student-t confidence intervals on
  `log10 K`. This fixes a calibration defect and simplifies the method at the same time. The
  bootstrap it replaces was anti-conservative on the small datasets typical of NMR titrations:
  a 200-run Monte-Carlo check of the nominal 95% interval on the bundled 1:1 synthetic design
  measured 0.83-0.92 coverage for the bootstrap against 0.950, 0.965 and 0.970 for the
  asymptotic interval at noise levels of 0.005, 0.01 and 0.02 ppm. On the 1:2 and 2:1 designs
  the asymptotic interval measured 0.94-0.98. The bootstrap also withheld the interval
  entirely whenever any refit failed its all-refits-must-succeed gate, which the asymptotic
  interval has no equivalent of. Intervals are computed in `log10 K` and converted to `K` for
  display, so they are asymmetric about `K`; the standard error quoted for `K` uses the delta
  method. Uncertainty is now always reported for a successful fit and needs no flag, no seed,
  and no refits.
- Parameter correlations now come from the fit covariance matrix instead of bootstrap samples,
  so `correlation.csv` is in the fitted-parameter basis and is written whenever the standard
  errors are nonzero.

- Simplified CLI input and output handling: glob-like literal filenames are accepted,
  duplicate dataset names receive deterministic numbered labels, positive numeric options
  are validated by the argument parser, and output directories use second-resolution
  timestamps with ordinal collision suffixes.
- Simplified internal typing and report artifacts without changing fitted results: the
  duplicated structural typing protocols were replaced by the concrete result dataclasses,
  the optimizer penalty-scale fallback collapses to a single degenerate-data branch, and
  isotherm/residual plot files use position-based names (`peak-0001`).
- Slimmed the HTML report. The table of contents is gone, per-model statistics are a compact
  table rather than a tile grid, and the stylesheet is a fraction of its former size. Model
  cards no longer repeat the Jacobian rank, condition number and log₁₀(K) sensitivity, because
  a reported fit has by definition cleared the identifiability gate those describe and the
  thresholds are already stated in the methods text; optimization penalty counts, equilibrium
  solver counts and per-peak R² now appear only when they carry information. Equilibrium solver
  counts are also labelled instead of showing their internal key names. The executive summary,
  methods sections, warnings, model comparison table and figures are unchanged.
- Dropped guards that re-checked what their callers already guarantee: directory tokens and
  HTML anchors are sanitized and bounded without a hash suffix, because callers prefix an
  ordinal that already keeps them distinct. Reporting decisions are unchanged; only the
  generated names for very long labels differ.
- Hardened model fitting and reporting guardrails: invalid multivalent binding constants
  are rejected, weak-binding solver brackets include the physical root, failed candidates
  remain visible in the model comparison table, and model selection now ranks only finite-BIC fits.
- Input handling now treats non-finite ppm values as invalid peak observations and reports
  rows dropped for missing required concentrations.
- Renamed the import package namespace from `core` to `nmr_bind_fit` while keeping the CLI command `nmr_bind_fit`.
- Expanded packaging metadata and optional dependency groups.
- Input concentrations are now required to be in molar (M); binding constants are reported in M⁻¹.
- The generated HTML report is now labeled consistently as `nmr_bind_fit` (previously "NMRBindFit").
- The CLI now reports expected input and validation errors as a concise message with a nonzero exit status instead of an uncaught traceback.
- `fit_models` now documents its parameters and defaults the optional ones, so it can be used as a library entry point with only datasets, model names, and log10(K) starts.

### Fixed

- Isolated independent multi-dataset report artifacts so plots cannot overwrite one another.
- Rejected underdetermined, rank-deficient, severely ill-conditioned, low-sensitivity, or bound-limited fits using response-unit-invariant diagnostics instead of reporting non-identifiable binding constants.
- Hardened 1:2 and 2:1 equilibrium bracketing and convergence behavior across extreme valid inputs.
- Added early input/configuration validation and nonzero CLI failures when no candidate model succeeds.

### Removed

- Removed the `missing_policy` parameter of `load_dataset` and `load_datasets`, together with
  its `mask` scheme. Only `drop-column` was ever reachable: the CLI hard-coded it, the option
  was undocumented, and `mask` had no consumer outside its own tests. A ppm column holding a
  missing or non-finite value is dropped, and a file whose columns are all dropped is rejected.
  With `mask` gone, `_validate_ppm_array` became unreachable — every retained column is finite
  by construction — and went with it.
- Removed the concentration validation from `load_dataset`. Non-positive host and negative
  guest concentrations are physically impossible and were already rejected at the fit boundary
  by `fit._validate_fit_design`, which checks the same thing; the loader no longer repeats it.
  Malformed input is still rejected with a clear error and a non-zero exit, now raised when the
  fit runs rather than at load time.
- Removed the `nmr_bind_fit.fit_bootstrap` module and everything specific to bootstrap
  resampling: `BootstrapResult`, BCa intervals and their delete-one jackknife refits, the
  residual, parametric and case resampling schemes, the censored-draw and
  all-refits-must-succeed gates, and bootstrap K histograms. `FitResult.bootstrap` is replaced
  by `FitResult.uncertainty`.
- Removed the `--bootstrap`, `--bootstrap-method`, `--bootstrap-ci-method`,
  `--bootstrap-logk-jitter` and `--seed` command-line flags, and the `bootstrap`,
  `bootstrap_method`, `bootstrap_ci_method`, `seed` and `logk_jitter` parameters of
  `fit_models` and `fit_model`. Uncertainty no longer involves resampling, so none of them
  have a meaning.
- Renamed `--bootstrap-ci-width` to `--ci-width`.
- Removed `summary.csv` from the output directory. Its model comparison table is rendered in
  `report.html` from the same rows, so a run now writes `report.html` and the per-model plot
  directories. The spreadsheet-formula cell escaping went with it, having nothing left to guard.
- Removed `decision.txt` from the output directory. The recommended provisional working model
  and the reasons behind it are reported in the executive summary of `report.html`, which is
  built from the same decision entries.
- Removed the optional `fit` positional command. Invoke `nmr_bind_fit --input ...`
  directly.
- Removed the `--concentration-unit` flag, which only relabeled report text while the fit and `K` bounds used raw numeric values. Supply concentrations in molar (M) instead.
- Removed unused private plot/report compatibility helpers and the undocumented
  `MIN_BOOTSTRAP_CI_SAMPLES` and `BootstrapResult.ci_warning` aliases. Result
  dataclasses now require the current diagnostic fields instead of preserving
  older positional-constructor layouts.

## [0.1.0] - 2026-06-30

### Added

- Initial public research-software package for NMR chemical-shift titration binding fits.
- Candidate model fitting for 1:1, sequential 1:2, sequential 2:1, and non-binding linear drift models.
- BIC/AICc model-comparison reporting, bootstrap uncertainty estimates, HTML reports, and CLI smoke-tested workflows.

[0.2.0]: https://github.com/dhsohn/nmr_bind_fit/releases/tag/v0.2.0
[0.1.0]: https://github.com/dhsohn/nmr_bind_fit/releases/tag/v0.1.0
