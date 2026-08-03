# Release and Zenodo DOI checklist

Zenodo mints two kinds of DOI for this repository. The concept DOI [10.5281/zenodo.21071369](https://doi.org/10.5281/zenodo.21071369) resolves to the most recent release and never changes; the README badge uses it. Each release also gets its own version DOI, which is what `CITATION.cff` and `paper/paper.bib` carry so that a citation names the exact archived code. Current archives: v0.2.0 at [10.5281/zenodo.21767384](https://doi.org/10.5281/zenodo.21767384), v0.1.0 at [10.5281/zenodo.21071370](https://doi.org/10.5281/zenodo.21071370).

This checklist prepares `nmr_bind_fit` for a JOSS-facing archived release. Do not invent a DOI: Zenodo creates the DOI only after the GitHub repository is connected to Zenodo and a GitHub release is published.

## Preconditions

- The JOSS-readiness branch has been reviewed and merged to `main`.
- The repository is public and issues are enabled.
- GitHub Actions CI is green on the commit to be released.
- The working tree is clean.
- The version in `pyproject.toml`, `nmr_bind_fit/__init__.py`, `CITATION.cff`, `CHANGELOG.md`, and `paper/paper.bib` is consistent.
- The `CHANGELOG.md` heading for the version being released carries the release date instead of `- unreleased`, and its link reference points at the release tag rather than the `compare` range.
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

Use the version already declared in the project metadata unless intentionally bumping it first.

```bash
git tag -a v0.2.0 -m "nmr_bind_fit v0.2.0"
git push origin v0.2.0
gh release create v0.2.0 \
  --title "nmr_bind_fit v0.2.0" \
  --notes "Replaces bootstrap resampling with asymptotic covariance uncertainty and Student-t intervals on log10 K, removes the bootstrap CLI flags and API arguments, and slims the report and CLI surfaces. See CHANGELOG.md for the full list."
```

After the GitHub release is published, Zenodo should archive the release and mint a DOI.

## After Zenodo mints the DOI

Update these files with the new version DOI:

- `CITATION.cff`: set `doi:` to the new version DOI. Leave the `identifiers` entry alone; it holds the concept DOI, which is the same for every release.
- `paper/paper.bib`: set `doi` and `url` on `nmrbindfit2026` to the new version DOI.
- `README.md`: update the citation section to name the new release and its version DOI. The badge already points at the concept DOI and needs no change.

Then commit the DOI update:

```bash
git checkout -b release/zenodo-doi-v0.2.0
# edit citation files with the real DOI
git add CITATION.cff paper/paper.bib README.md
git commit -m "Add Zenodo DOI for v0.2.0 release"
git push -u origin release/zenodo-doi-v0.2.0
```

## JOSS submission notes

- Submit after the public-development-history requirement is satisfied.
- Use the archived release DOI from Zenodo, not a repository URL alone, as the software archive reference.
- Disclose related publications that are published, under review, or nearing submission.
- Keep the JOSS paper focused on the software implementation, validation, and reusable workflow rather than new experimental findings.
- Update the AI usage disclosure in `paper/paper.md` to match the final preparation history.
