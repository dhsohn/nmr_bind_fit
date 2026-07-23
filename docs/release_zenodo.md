# Release and Zenodo DOI checklist

Archived v0.1.0 DOI: [10.5281/zenodo.21071370](https://doi.org/10.5281/zenodo.21071370).

This checklist prepares `nmr_bind_fit` for a JOSS-facing archived release. Do not invent a DOI: Zenodo creates the DOI only after the GitHub repository is connected to Zenodo and a GitHub release is published.

## Preconditions

- The JOSS-readiness branch has been reviewed and merged to `main`.
- The repository is public and issues are enabled.
- GitHub Actions CI is green on the commit to be released.
- The working tree is clean.
- The version in `pyproject.toml`, `nmr_bind_fit/__init__.py`, `CITATION.cff`, and `CHANGELOG.md` is consistent.
- The release is made after the repository satisfies the public-development-history requirement for the intended JOSS submission.

## Local verification before tagging

```bash
git checkout main
git pull --ff-only
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,excel]"
python -m pytest -q
nmr_bind_fit --input examples/synthetic_11.csv
python -m build
python -m pip check
```

Remove generated smoke-test output directories and build artifacts before committing release metadata updates.

## Zenodo setup

1. Log in to Zenodo using the GitHub account that owns or can administer `dhsohn/nmr_bind_fit`.
2. Open the GitHub integration settings in Zenodo.
3. Enable Zenodo archiving for `dhsohn/nmr_bind_fit`.
4. Confirm that `.zenodo.json` metadata is present on the release branch or `main`.

## GitHub release

For the first JOSS-readiness release, use the version already declared in the project metadata unless intentionally bumping it first.

```bash
git tag -a v0.1.0 -m "nmr_bind_fit v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 \
  --title "nmr_bind_fit v0.1.0" \
  --notes "Initial JOSS-readiness release with package namespace cleanup, CI, examples, tutorial documentation, citation metadata, and MIT license."
```

After the GitHub release is published, Zenodo should archive the release and mint a DOI.

## After Zenodo mints the DOI

Update these files with the real Zenodo DOI:

- `CITATION.cff`: add `doi: <zenodo-doi>`.
- `paper/paper.bib`: add `doi = {<zenodo-doi>}` to `nmrbindfit2026` and replace the DOI-pending note.
- `README.md`: update the citation section if desired.
- Optionally add a DOI badge to `README.md`.

Then commit the DOI update:

```bash
git checkout -b release/zenodo-doi-v0.1.0
# edit citation files with the real DOI
git add CITATION.cff paper/paper.bib README.md
git commit -m "Add Zenodo DOI for v0.1.0 release"
git push -u origin release/zenodo-doi-v0.1.0
```

## JOSS submission notes

- Submit after the public-development-history requirement is satisfied.
- Use the archived release DOI from Zenodo, not a repository URL alone, as the software archive reference.
- Disclose related publications that are published, under review, or nearing submission.
- Keep the JOSS paper focused on the software implementation, validation, and reusable workflow rather than new experimental findings.
- Update the AI usage disclosure in `paper/paper.md` to match the final preparation history.
