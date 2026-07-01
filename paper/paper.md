---
title: 'nmr_bind_fit: transparent model comparison and uncertainty reporting for NMR host--guest titrations'
tags:
  - Python
  - chemistry
  - NMR
  - supramolecular chemistry
  - host-guest chemistry
  - binding titration
  - model comparison
authors:
  - name: Dae Hyup Sohn
    corresponding: true
    affiliation: 1
affiliations:
 - name: Hanyang University, Republic of Korea
   index: 1
date: 30 June 2026
bibliography: paper.bib
---

# Summary

Nuclear magnetic resonance (NMR) chemical-shift titration is a common way to study molecular recognition and host--guest binding in solution. In a typical experiment, the chemical shifts of one or more host resonances are tracked while a guest is added. The resulting curves can be used to estimate binding constants, but the interpretation depends strongly on the assumed binding stoichiometry and on whether the titration data genuinely discriminate binding from non-binding drift.

`nmr_bind_fit` [@nmrbindfit2026] is a Python command-line workflow for reproducible NMR chemical-shift titration analysis. It fits 1:1, sequential 1:2, sequential 2:1, and non-binding candidate models under a fast-exchange assumption, ranks the fitted candidates using information criteria, estimates uncertainty with bootstrap resampling, and writes machine-readable summaries plus human-readable HTML reports. The software is intended as a transparent decision-support tool: it reports the best-supported working model among tested candidates, while making clear that final chemical interpretation must also consider spectral behavior, saturation, peak assignment, and feasible stoichiometry.

# Statement of need

Binding constants in supramolecular chemistry are most useful when they are accompanied by transparent model choice, uncertainty estimates, and chemically meaningful caveats [@Thordarson2011; @Hibbert2016; @Flood2023]. In practice, however, NMR titration curves are often analyzed under a preselected stoichiometry, and ambiguous curves can yield plausible-looking binding constants even when the data do not strongly support a binding interpretation. This risk is especially important for weak, noisy, undersaturated, or drift-like titrations, where a binding model may overfit systematic trends unless a non-binding alternative is tested [@Baker2024].

`nmr_bind_fit` addresses this need by making model comparison and reporting part of the standard analysis workflow. Instead of asking users to fit only a chosen model, the software fits a small candidate set relevant to host-resonance titrations: 1:1 binding, sequential 1:2 binding, sequential 2:1 binding, and a non-binding linear drift control. It then reports Bayesian Information Criterion (BIC) values as the primary ranking index, corrected Akaike Information Criterion (AICc) values as supporting information, bootstrap confidence intervals for binding constants, and warnings when the statistical discrimination or numerical diagnostics are weak. This supports more cautious reporting of NMR-derived binding constants and helps users document why a candidate model was treated as provisional rather than definitive.

# State of the field

The need for transparent binding-constant determination, uncertainty estimation, and alternatives to oversimplified graphical approaches has been emphasized repeatedly in supramolecular data-analysis guidance [@Thordarson2011; @Hibbert2016]. Existing workflows often involve spreadsheets, general-purpose nonlinear least-squares tools, custom scripts, or web-based fitting tools. These can be effective for individual analyses, but they may leave important choices implicit: which stoichiometries were tested, how non-binding behavior was ruled out, whether a more complex model was penalized for extra parameters, and how uncertainty or weak discrimination was communicated.

`nmr_bind_fit` is not intended to replace broad chemical modeling environments or bespoke analyses for unusual equilibria. Its contribution is narrower and workflow-oriented: it packages a reproducible, inspectable, command-line analysis path for a common class of host-resonance NMR titrations. The software combines mass-balance-based candidate models, consistent model comparison, bootstrap uncertainty, and report generation in a single open-source Python package. By emphasizing a non-binding control and provisional model language, it complements best-practice recommendations for transparent supramolecular analysis rather than presenting model selection as a fully automated chemical verdict.

# Software design

The package is organized around a conservative analysis pipeline. Input files are CSV or XLSX tables with total host concentration, total guest concentration, and one or more chemical-shift columns. The command-line interface loads and validates the data, builds equivalents as `[G]t/[H]t`, fits each candidate model, and writes a timestamped output directory containing `summary.csv`, `decision.txt`, `report.html`, and per-model plots.

The candidate models share a host-resonance fast-exchange interpretation in which observed shifts are population-weighted averages of chemical states, following the global/local and mass-balance perspective used in prior titration-analysis work [@Lowe2012; @Hargrove2010]. The 1:1 model is solved analytically from the quadratic mass-balance equation. The sequential 1:2 and 2:1 models solve the equilibrium point-by-point with Brent root finding from SciPy [@SciPy2020]. In the 2:1 model, the H2G species contributes two host units to the observed host-resonance population, so the host-basis weight is doubled. Parameters are estimated by multistart nonlinear least squares, and binding constants are optimized in `log10 K` space to improve numerical behavior. The command-line workflow requires input concentrations in molar (M) and constrains `log10 K` to [0, 12], corresponding to K values from 1 to 10^12 M⁻¹. This interval is a broad numerical search bound rather than a universal chemically meaningful range; fits whose optima or bootstrap intervals reach these bounds should be interpreted as boundary-limited and inspected or refit with chemically justified bounds.

The reporting layer intentionally separates statistical ranking from chemical acceptance. BIC [@Schwarz1978] is used as the primary model-comparison index because it penalizes the additional parameters in sequential binding models; AICc [@Hurvich1989] is reported as supporting information for small datasets. The decision text flags weak discrimination when the gap between the best and next-best BIC values is small, using conventional interpretive thresholds for BIC differences as context [@Kass1995]. Bootstrap resampling provides uncertainty estimates for fitted binding constants. The generated methods text and diagnostic tables are designed to make analyses reproducible and reviewable without requiring users to reconstruct hidden fitting choices.

# Research impact statement

`nmr_bind_fit` has been developed for active NMR titration work by the author and has already been used in an ongoing peer-reviewed host--guest study. The experimental findings from that study are reported separately and are not part of this software paper. The present submission focuses on the reusable software implementation, validation, examples, and reporting workflow.

The repository includes a test suite covering equilibrium solvers, model predictions, information criteria, bootstrap behavior, command-line orchestration, input handling, plots, and report language. It also includes deterministic synthetic examples for 1:1, 1:2, 2:1, and non-binding cases, together with tutorial documentation. These materials provide reproducible reference cases for users and reviewers, and they establish a basis for future benchmark studies of model discrimination limits in NMR binding analysis.

# AI usage disclosure

Generative AI tools were used during both the development of this software and the preparation of this paper:

- **Software (code, tests, documentation).** Anthropic Claude (Opus 4.8 and Opus 4.6, via Claude Code) and OpenAI GPT-5.5 were used to help draft and refactor parts of the implementation, tests, docstrings, and documentation, and to draft pull-request descriptions. Several commits in the repository carry `Co-Authored-By` trailers recording this assistance.
- **Code review.** Automated review tools, including OpenAI Codex, were used to review pull requests during development.
- **Paper.** Claude Opus 4.8 and OpenAI GPT-5.5 were used to review and refine drafts of this paper.

All AI-assisted output was reviewed, edited, tested, and verified by the author. The author made the scientific and software-design decisions — including the choice of binding models, the equilibrium formulations, and the statistical methodology — and remains responsible for the correctness, originality, licensing, and ethical compliance of the submitted software and paper.
