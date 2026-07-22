# Changelog

All notable changes to `nmr_bind_fit` will be documented in this file.

The format follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses semantic versioning once releases are tagged.

## [Unreleased]

### Added

- Community-maintenance scaffolding: GitHub issue forms (bug report, feature request), a pull-request template, and Dependabot configuration for GitHub Actions and pip.
- Python version badge in the README.
- Zenodo DOI metadata for the archived v0.1.0 release (10.5281/zenodo.21071370).
- JOSS-readiness project metadata, including license, citation metadata, contribution guide, and CI configuration.
- Example synthetic titration datasets and tutorial documentation.
- Draft JOSS paper, BibTeX references, Zenodo metadata, and release/DOI checklist.
- `py.typed` marker so downstream users receive the package's type information (PEP 561).

### Changed

- Simplified CLI input and output handling: glob-like literal filenames are accepted,
  duplicate dataset names receive deterministic numbered labels, positive numeric options
  are validated by the argument parser, and output directories use second-resolution
  timestamps with ordinal collision suffixes.
- Hardened model fitting and reporting guardrails: invalid multivalent binding constants
  are rejected, weak-binding solver brackets include the physical root, failed candidates
  remain visible in `summary.csv`, and model selection now ranks only finite-BIC fits.
- Bootstrap confidence intervals now require every requested pseudo-dataset to yield an
  uncensored acceptable refit after comparing jittered and full-data-optimum starts. A
  statistically competitive bound-limited or otherwise non-identifiable solution censors the
  draw, preventing convergence-filtered tail samples from being reported as unconditional
  intervals. BCa-style intervals use leave-one-titration-point jackknife refits and explicitly
  omit the requested estimate when unavailable rather than silently falling back to percentile.
- Input handling now treats non-finite ppm values as invalid peak observations and reports
  rows dropped for missing required concentrations.
- Renamed the import package namespace from `core` to `nmr_bind_fit` while keeping the CLI command `nmr_bind_fit`.
- Expanded packaging metadata and optional dependency groups.
- Input concentrations are now required to be in molar (M); binding constants are reported in M⁻¹.
- The generated HTML report is now labeled consistently as `nmr_bind_fit` (previously "NMRBindFit").
- The CLI now reports expected input and validation errors as a concise message with a nonzero exit status instead of an uncaught traceback.
- `fit_models` now documents its parameters and defaults the optional ones, so it can be used as a library entry point with only datasets, model names, and log10(K) starts.

### Fixed

- Isolated independent multi-dataset report artifacts so plots and bootstrap files cannot overwrite one another.
- Rejected underdetermined, rank-deficient, severely ill-conditioned, low-sensitivity, or bound-limited fits using response-unit-invariant diagnostics instead of reporting non-identifiable binding constants.
- Required sufficient successful bootstrap refits before reporting confidence intervals and replaced bootstrap-sample acceleration with explicitly labeled local-refit BCa jackknife estimates.
- Hardened 1:2 and 2:1 equilibrium bracketing and convergence behavior across extreme valid inputs.
- Added early input/configuration validation and nonzero CLI failures when no candidate model succeeds.

### Removed

- Removed the optional `fit` positional command. Invoke `nmr_bind_fit --input ...`
  directly.
- Removed the `--concentration-unit` flag, which only relabeled report text while the fit and `K` bounds used raw numeric values. Supply concentrations in molar (M) instead.

## [0.1.0] - 2026-06-30

### Added

- Initial public research-software package for NMR chemical-shift titration binding fits.
- Candidate model fitting for 1:1, sequential 1:2, sequential 2:1, and non-binding linear drift models.
- BIC/AICc model-comparison reporting, bootstrap uncertainty estimates, HTML reports, and CLI smoke-tested workflows.

[Unreleased]: https://github.com/dhsohn/nmr_bind_fit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dhsohn/nmr_bind_fit/releases/tag/v0.1.0
