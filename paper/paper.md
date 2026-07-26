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
    orcid: 0000-0002-1743-1312
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

`nmr_bind_fit` [@nmrbindfit2026] is a Python command-line workflow for reproducible NMR chemical-shift titration analysis. It fits 1:1, sequential 1:2, sequential 2:1, and non-binding candidate models under a fast-exchange assumption, ranks the fitted candidates using information criteria, estimates parameter uncertainty from the asymptotic covariance matrix of each converged fit, and writes a self-contained HTML report with the fitted values, model comparison, diagnostics, and plots. Rather than treating binding stoichiometry as a fixed input, the workflow addresses two linked questions: which tested stoichiometric model is best supported by the isotherm, and whether a binding model is supported over a non-binding trend. The resulting ranking is reported as a provisional working model to be evaluated together with spectral behavior, saturation, peak assignment, chemical feasibility, and other orthogonal experimental evidence.

# Statement of need

Reliable estimation also depends on experimental design. Poorly chosen concentration ranges or insufficient sampling of the curved and saturating regions of a titration can leave binding constants weakly determined even when a numerical fit converges [@Hirose2001; @Thordarson2011].

Binding constants in supramolecular chemistry are most useful when they are accompanied by transparent model choice, uncertainty estimates, and chemically meaningful caveats [@Thordarson2011; @Hibbert2016; @Flood2023]. In practice, however, NMR titration curves are often analyzed under a preselected stoichiometry, and ambiguous curves can yield plausible-looking binding constants even when the data do not strongly support a binding interpretation. This risk is especially important for weak, noisy, undersaturated, or drift-like titrations, where a binding model may overfit systematic trends unless a non-binding alternative is tested [@Baker2024].

`nmr_bind_fit` addresses this need through a fixed and inspectable candidate-model workflow. The same isotherm is fitted to 1:1, sequential 1:2, sequential 2:1, and non-binding linear-drift models. The workflow therefore supports two forms of statistical discrimination: among plausible binding stoichiometries, and between binding and non-binding interpretations. BIC is reported as the primary ranking index, with AICc, confidence intervals, identifiability checks, and weak-discrimination warnings providing supporting evidence. This approach is particularly useful when a separate stoichiometric assessment, such as a continuous-variation or Job-plot experiment, is impractical or inconclusive. It does not replace independent experimental evidence; instead, it provides a provisional stoichiometric hypothesis that can be evaluated against the full experimental record.

# State of the field

Software-assisted determination of equilibrium constants from NMR chemical-shift data predates modern web and desktop workflows. EQNMR provided model-independent specification of equilibria, parameter constraints, and nonlinear least-squares fitting for fast-exchange NMR data [@Hynes1993]. Bindfit subsequently made fitting of common supramolecular binding models accessible through a web interface [@Hibbert2016]. More recent open-source tools offer substantially broader analysis environments. SupraFit supports NMR and isothermal titration calorimetry data, common 1:1, 1:2, and 2:1 binding systems, global and local fitting, and several uncertainty-analysis methods through graphical and command-line interfaces [@Huebler2022]. Musketeer provides a cross-platform graphical workflow for NMR, UV--visible, and fluorescence titrations and supports user-defined, multicomponent equilibrium models [@Soloviev2024].

The comparison below summarizes the standard workflow documented for each tool rather than every model that could be constructed through customization.

| Tool | Data | Standard workflow | Model evaluation |
| --- | --- | --- | --- |
| Bindfit | NMR; UV--visible; fluorescence | Web; selected 1:1, 1:2, or 2:1 model | Fit and residuals |
| SupraFit | NMR; ITC | GUI and CLI; common models; global/local fits | Monte Carlo; F tests |
| Musketeer | NMR; UV--visible; fluorescence | GUI; predefined/custom multicomponent equilibria | RMSE; visual and physical checks |
| `nmr_bind_fit` | NMR chemical shifts | CLI; fixed 1:1, 1:2, 2:1, and non-binding (NB) set | BIC/AICc; uncertainty/identifiability; stoichiometry and binding/NB screening |

`nmr_bind_fit` complements these broader, flexible analysis environments by making model comparison part of the standard analysis rather than a separate user-directed step. Every candidate is processed using the same estimation, uncertainty, diagnostic, and reporting procedures. The included non-binding linear-drift model asks whether a systematic trend can explain the data without invoking a binding equilibrium. Together, these features make two linked questions explicit: which tested stoichiometric model is best supported by the binding isotherm, and whether the observed trend warrants a binding interpretation rather than a non-binding explanation.

The contribution of `nmr_bind_fit` is therefore not a new equilibrium model or a replacement for tools that support arbitrary chemical systems. It is the integration of stoichiometric model discrimination, an explicit non-binding negative control, identifiability and uncertainty checks, provisional interpretation, and self-contained reporting into a single reproducible command-line workflow.

# Software design

The package is organized around a conservative analysis pipeline. Input files are CSV or XLSX tables with total host concentration, total guest concentration, and one or more chemical-shift columns. The command-line interface loads and validates the data, builds equivalents as `[G]t/[H]t`, fits each candidate model, and writes a timestamped output directory containing `report.html` and per-model plots.

The candidate models share a host-resonance fast-exchange interpretation in which observed shifts are population-weighted averages of chemical states, following the global/local and mass-balance perspective used in prior titration-analysis work [@Lowe2012; @Hargrove2010]. The 1:1 model is solved analytically from the quadratic mass-balance equation. The sequential 1:2 and 2:1 models solve the equilibrium point-by-point with Brent root finding from SciPy [@SciPy2020]. In the 2:1 model, the H2G species contributes two host units to the observed host-resonance population, so the host-basis weight is doubled. Parameters are estimated by multistart nonlinear least squares, and binding constants are optimized in `log10 K` space to improve numerical behavior. The command-line workflow requires input concentrations in molar (M) and constrains `log10 K` to [0, 12], corresponding to K values from 1 to 10^12 M⁻¹. This interval is a broad numerical search bound rather than a universal chemically meaningful range; fits whose optima or confidence intervals reach these bounds should be interpreted as boundary-limited and inspected or refit with chemically justified bounds.

The reporting layer intentionally separates statistical ranking from chemical acceptance. BIC [@Schwarz1978] is used as the primary model-comparison index because it penalizes the additional parameters in sequential binding models, while AICc [@Hurvich1989] is reported as supporting information for small datasets. The report flags weak discrimination when the gap between the best and next-best BIC values is small, using conventional interpretive thresholds for BIC differences as context [@Kass1995].

Accordingly, the model-comparison table addresses two linked scientific questions: whether the isotherm supports binding over non-binding drift and, if so, which of the tested stoichiometries receives the strongest relative support.

These rankings are conditional on both the tested candidate set and the assumed residual model. They do not establish the chemical truth of a binding model and may be unreliable when concentration errors, heteroscedastic residuals, unmodelled equilibria, impurities, or other systematic effects dominate the data. Similar limitations of purely numerical model-selection criteria have been emphasized for titration analysis [@Soloviev2024]. `nmr_bind_fit` therefore presents the selected model as a provisional statistical result and reports fitted curves, diagnostics, boundary warnings, and weak-discrimination warnings for inspection. Users must additionally consider spectral behavior, saturation, peak assignment, feasible stoichiometry, and experimental uncertainty before accepting a chemical interpretation.

Uncertainty estimates for fitted binding constants come from the asymptotic covariance matrix of the converged fit, with Student-t confidence intervals on `log10 K` at the residual degrees of freedom. These intervals are likewise conditional on the local linear approximation and residual assumptions of the fitted model.

# Research impact statement

`nmr_bind_fit` has been developed for active NMR titration work by the author and has already been used in an ongoing peer-reviewed host--guest study. The experimental findings from that study are reported separately and are not part of this software paper. The present submission focuses on the reusable software implementation, validation, examples, and reporting workflow.

The repository includes a test suite covering equilibrium solvers, model predictions, information criteria, uncertainty estimation, command-line orchestration, input handling, plots, and report language. It also includes deterministic synthetic examples for 1:1, 1:2, 2:1, and non-binding cases, together with tutorial documentation. These materials provide reproducible reference cases for users and reviewers, and they establish a basis for future benchmark studies of model discrimination limits in NMR binding analysis.

# AI usage disclosure

Generative AI tools were used during both the development of this software and the preparation of this paper:

- **Software (code, tests, documentation).** Anthropic Claude (Opus 4.8 and Opus 4.6, via Claude Code) and OpenAI GPT-5.5 were used to help draft and refactor parts of the implementation, tests, docstrings, and documentation, and to draft pull-request descriptions. Several commits in the repository carry `Co-Authored-By` trailers recording this assistance.
- **Code review.** Automated review tools, including OpenAI Codex, were used to review pull requests during development.
- **Paper.** Claude Opus 4.8 and OpenAI GPT-5.5 were used to review and refine drafts of this paper.

All AI-assisted output was reviewed, edited, tested, and verified by the author. The author made the scientific and software-design decisions — including the choice of binding models, the equilibrium formulations, and the statistical methodology — and remains responsible for the correctness, originality, licensing, and ethical compliance of the submitted software and paper.
