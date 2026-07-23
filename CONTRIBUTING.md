# Contributing

Thank you for your interest in improving `nmr_bind_fit`.

## Development setup

```bash
git clone https://github.com/dhsohn/nmr_bind_fit.git
cd nmr_bind_fit
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,excel]"
python -m pytest -q
```

## Reporting issues

Please use GitHub issues for bug reports, reproducibility problems, and feature requests. Helpful bug reports include:

- the command that was run;
- the input format, with a minimal example if possible;
- the Python version and operating system;
- the generated `summary.csv`, `report.html`, or relevant traceback;
- whether the data are synthetic/example data or real unpublished data.

Please do not upload confidential spectra, unpublished chemical structures, or private experimental data unless you are allowed to share them publicly.

## Questions and support

Please use GitHub issues for usage questions, installation problems, reproducibility questions, and requests for clarification about the documented workflow. For questions involving unpublished or confidential data, describe the software behavior with a synthetic or anonymized minimal example rather than uploading private spectra, chemical structures, or raw experimental records.

## Pull requests

Pull requests should keep the workflow reproducible and conservative. In particular:

1. Add or update tests for changes in fitting, model comparison, reporting, or input/output behavior.
2. Keep model-selection language transparent: the lowest-BIC model is a provisional working model among tested candidates, not an automatic chemical truth claim.
3. Preserve fail-fast behavior for numerical solver failures unless the change explicitly documents and tests a different policy.
4. Run `python -m ruff check .` and `python -m pytest -q` before opening a pull request.

## Documentation changes

Documentation should distinguish between:

- statistical model ranking among tested candidates;
- uncertainty estimates such as bootstrap confidence intervals;
- chemical plausibility checks that require user judgment, such as fast-exchange behavior, saturation, peak assignment, and feasible stoichiometry.

## AI-assisted contributions

If generative AI tools are used to draft code, tests, or documentation, contributors remain responsible for reviewing, validating, and licensing the final contribution. Please mention substantial AI assistance in the pull request description when relevant.
