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

### Changed

- Renamed the import package namespace from `core` to `nmr_bind_fit` while keeping the CLI command `nmr_bind_fit`.
- Expanded packaging metadata and optional dependency groups.
- Input concentrations are now required to be in molar (M); binding constants are reported in M⁻¹.

### Removed

- Removed the `--concentration-unit` flag, which only relabeled report text while the fit and `K` bounds used raw numeric values. Supply concentrations in molar (M) instead.

## [0.1.0] - 2026-06-30

### Added

- Initial public research-software package for NMR chemical-shift titration binding fits.
- Candidate model fitting for 1:1, sequential 1:2, sequential 2:1, and non-binding linear drift models.
- BIC/AICc model-comparison reporting, bootstrap uncertainty estimates, HTML reports, and CLI smoke-tested workflows.

[Unreleased]: https://github.com/dhsohn/nmr_bind_fit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dhsohn/nmr_bind_fit/releases/tag/v0.1.0
