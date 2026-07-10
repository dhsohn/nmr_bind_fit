# Binding Model Selection — Statistical and Chemical Rationale

This document specifies how `nmr_bind_fit` selects a **working binding model**
(1:1, 1:2, 2:1, or non-binding) from NMR chemical-shift titration data, and why that
selection is **statistically and chemically defensible**. It is intended as a methods
reference suitable for citation in a manuscript; `README.md` covers installation and
basic usage only.

> **Guiding principle.** The package does not declare any model to be the ground truth.
> The information criteria report only *relative support among the tested candidates*; the
> final model must always be interpreted together with **chemical plausibility** and
> **spectral behavior**. This is why the report consistently labels the chosen model a
> *provisional (tentative) working model*. This emphasis on the reliability — not merely the
> point value — of a binding constant reflects the "quality binding constants" perspective of
> Flood, Vander Griend and Thordarson (2023).

---

## 1. Data interpretation — the fast-exchange assumption

Titration data are interpreted under the **fast-exchange regime**, in which the observed
host-resonance chemical shift is the **population-weighted average** of the free and bound
chemical states:

$$
\delta_\text{obs} = \sum_i x_i \, \delta_i
$$

- $x_i$ — mole fraction of host in state $i$
- $\delta_i$ — intrinsic chemical shift of that state

**Why this matters.** If fast exchange does not hold (intermediate or slow exchange), one
observes separate peaks or line broadening rather than a single averaged peak, and the
linear averaging relation breaks down. Before fitting, the user must confirm that the peak
moves continuously and does not split during the titration. This is a **chemical
precondition** that statistics cannot adjudicate; it is the user's responsibility.

All weights are defined on a **host basis** (host resonances are observed), and the x-axis is
always equivalents $[G]_t / [H]_t$.

---

## 2. Candidate binding models

Four candidates are fit independently to the same data.

| Model | Equilibrium | Constants | Species | δ params / peak |
|-------|-------------|-----------|---------|-----------------|
| **1:1** | H + G ⇌ HG | $K$ | H, HG | 2 |
| **1:2** | H + G ⇌ HG (K₁); HG + G ⇌ HG₂ (K₂) | $K_1, K_2$ | H, HG, HG₂ | 3 |
| **2:1** | H + G ⇌ HG (K₁); H + HG ⇌ H₂G (K₂) | $K_1, K_2$ | H, HG, H₂G | 3 |
| **Non-binding (nb)** | none, linear drift | — | a₀, a₁ | 2 |

(Implementation: `MODEL_SPECS` in `nmr_bind_fit/models.py`.)

### 2.1 Predictive equations

**1:1** — the equilibrium is solved **analytically** via the closed-form quadratic:

$$
\delta_\text{obs} = \frac{[H]}{[H]_t}\,\delta_H + \frac{[HG]}{[H]_t}\,\delta_{HG}
$$

**1:2 sequential binding** — the free-guest concentration is obtained from a cubic
polynomial solved numerically at each titration point; weights are the host-basis fractions
of H, HG, and HG₂.

**2:1 sequential binding** — H₂G contains **two host units**, so its weight is doubled:

$$
w_{H_2G} = \frac{2\,[H_2G]}{[H]_t}
$$

(Implementation: `_weights_21` in `nmr_bind_fit/models.py`.) This is not a numerical trick but a
direct consequence of host mass conservation.

**Non-binding (control model)** — the simple linear drift expected in the absence of binding:

$$
\delta = a_0 + a_1 \cdot \frac{[G]_t}{[H]_t}
$$

The non-binding model serves as a **negative control**. If it receives statistical support
comparable to or better than the binding models, this is strong evidence that the data do
not warrant a binding interpretation. This systematic fit-and-compare strategy — testing all
reasonable binding models rather than presupposing one — follows the best-practice
recommendations of Hibbert and Thordarson (2016) for supramolecular data analysis.
Explicitly modeling the non-binding case also guards against reporting a binding constant for
data that do not in fact support binding — the kind of false positive examined by Baker et al.
(2024).

---

## 3. Parameter estimation

- **Method**: nonlinear least squares (`scipy.optimize.least_squares`, Trust Region Reflective).
- **1:1**: equilibrium solved analytically (closed-form quadratic).
- **1:2 / 2:1**: solved point-by-point with Brent's method over the physical free-guest
  interval `[0, G_tot]` (`scipy.optimize.brentq`, scale-adaptive absolute tolerance,
  `rtol=8·eps`, `maxiter=200`). The full mass-balance relation is
  solved at every point — the simplifying approximation that free guest equals total guest is
  *not* used — in line with the rigorous equilibrium treatment of Hargrove et al. (2010).
- **K parameterization**: estimated as $\log_{10}(K)$ and **constrained to [0, 12]**
  → $K \in [1,\ 10^{12}]\ \text{M}^{-1}$ (concentrations in molar, M).
  This range ensures physically meaningful and numerically stable estimation. A
  $\log_{10}K$ estimate pinned at 0 or 12 indicates the estimate is not trustworthy (see §6).
- **Multistart**: fits are launched from a grid of $\log_{10}K$ starting values to avoid
  local minima; among successful fits, the solution with the lowest RSS is retained
  (`select_best_multistart` in `nmr_bind_fit/fit_optimizer.py`).
- **Identifiability gate**: a fit is excluded from model comparison unless it has positive
  residual degrees of freedom, a full-column-rank optimizer Jacobian with condition number
  at most $10^6$, and no fitted $\log_{10}K$ value on an active optimization bound.
- **Missing data**: rows with missing host/guest concentrations are dropped; any ppm column
  containing missing values is excluded entirely (remaining peaks retained).
- **Solver failures**: per-point solver failures in 1:2/2:1 use fail-fast behavior; penalty
  residual events are recorded in the report.
- **Replicates**: fit simultaneously with shared binding constants $K$ and replicate-specific
  δ parameters, improving the precision of K estimates. This *global* fitting strategy — using
  all data to constrain shared constants rather than fitting each peak or replicate in
  isolation — follows the global/local analysis of NMR titration data described by Lowe,
  Pfeffer and Thordarson (2012).

---

## 4. Statistical model comparison — the primary criterion

### 4.1 BIC (primary ranking index)

The **primary ranking index is the Bayesian Information Criterion (BIC)**:

$$
\text{BIC} = -2\,\ell(\hat\theta) + k \ln(n)
$$

- $\ell(\hat\theta)$ — maximized Gaussian log-likelihood
- $n$ — number of observations
- $k$ — number of estimated parameters

Under i.i.d. Gaussian residuals with the MLE variance $\hat\sigma^2 = \text{RSS}/n$, this
equals $n\ln(2\pi\,\text{RSS}/n) + n + k\ln(n)$; for ranking, the terms additive in $n$
cancel, giving the same ordering as $n\ln(\text{RSS}/n) + k\ln(n)$.

**Why BIC is primary.** The $k\ln(n)$ penalty penalizes parameter-rich models (1:2, 2:1)
more heavily. In titration data the more complex models almost always reduce RSS, so RSS
alone would always favor them (overfitting). BIC asks whether the added parameters justify
the improvement in fit — that is, it enforces **parsimony** — which is especially important
for small binding datasets.

### 4.2 AICc (supporting index)

The small-sample corrected AIC (AICc) is reported as **supporting information**:

$$
\text{AICc} = -2\,\ell(\hat\theta) + 2k + \frac{2k(k+1)}{n-k-1}
$$

When $n - k - 1 \le 0$, AICc is undefined (NaN) — a natural signal that there are too few
data points relative to the number of parameters for AICc to be trusted. AIC-family criteria
are more permissive toward complex models than BIC; when BIC and AICc point to different
models, that disagreement is itself evidence that the data do not strongly discriminate
between candidates.

### 4.3 Likelihood, n, and k — exact definitions

(Implementation: `nmr_bind_fit/stats.py`, `nmr_bind_fit/fit_criteria.py`.)

- **One shared variance is estimated.** Residuals from all datasets, points, and peaks are
  pooled to estimate a single residual variance $\sigma^2$ (Gaussian likelihood). This shared
  variance is counted as one information-criteria parameter, so **$k = p + 1$** ($p$ is the
  number of model parameters; +1 is the shared variance term).
- **Effective sample size $n$** is the total number of **finite residual scalars** across all
  datasets, titration points, and ppm peaks. Missing observations are excluded.
- **Inter-peak residual correlation is not modeled** (diagonal covariance assumed). Multiple
  peaks from the same molecule may in reality be correlated, but the tool treats them as
  independent for simplicity — see the limitations in §8.

> Note: model comparison (BIC/AICc) uses a **single pooled variance**, whereas uncertainty
> quantification (parametric bootstrap) may use per-peak variances. The two procedures serve
> different purposes (model ranking vs. uncertainty quantification) and are therefore not
> inconsistent.

---

## 5. Model-selection decision rules

(Implementation: `build_decisions` in `nmr_bind_fit/report_pipeline.py`.)

1. **Select the lowest-BIC model as the provisional working model.**
   The candidate with the lowest BIC is adopted as the "tentative working model among the
   tested candidates." The report flags it as the *recommended model* while always stating it
   is provisional.

2. **ΔBIC check (discriminating power).**
   Compute the BIC gap to the next-best candidate, $\Delta\text{BIC} = \text{BIC}\text{(2nd)} - \text{BIC}\text{(best)}$:

   $$\Delta\text{BIC} < 2 \;\Rightarrow\; \text{flag as "weak discrimination"}$$

   In this case the report advises treating model selection as provisional. By convention
   (Kass & Raftery, 1995), $\Delta\text{BIC} \ge 2$ is "positive," $\ge 6$ "strong," and
   $\ge 10$ "very strong" evidence. This tool conservatively uses **2** as the threshold for
   weak discrimination.

3. **Bootstrap CI-width check (estimate stability).**
   When `--bootstrap-ci-width` is set and the best model is a binding model, if the bootstrap
   confidence interval width for K exceeds the threshold, a "bootstrap CI too wide" warning is
   added — a sign that K is weakly identified by the data.

4. **Record the outcome.** All checks (ΔBIC, CI width, solver failures, penalty events, etc.)
   are written with their reasons to `decision.txt`, `summary.csv`, and `report.html`.

### Bootstrap uncertainty quantification

Reporting binding constants without a credible estimate of their uncertainty is of limited
value; resampling-based confidence intervals are used here in the spirit of the uncertainty-
estimation methods advocated by Hibbert and Thordarson (2016).

- Iterations via `--bootstrap` (default 1000); resampling via `--bootstrap-method`
  (`residual` default / `parametric` / `points`).
- CI method: percentile (2.5th/97.5th, default) or BCa-style local refitting
  (`--bootstrap-ci-method bca`). The acceleration term is estimated from delete-one jackknife
  refits initialized at the full-data optimum. Because the complete multistart estimator is not
  rerun for every bootstrap and jackknife sample, reports identify the method as `bca-local`.
- CI validity: at least 20 successful refits and at least 80% of the requested bootstrap
  iterations must succeed; otherwise the interval is reported as unavailable.
- Each bootstrap refit applies a small jitter to the $\log_{10}K$ start to explore the
  objective surface near the optimum.

---

## 6. Chemical plausibility checks — beyond statistics

Even when an information criterion points to a model, it must not be accepted unless the
following are satisfied. These are **chemical sanity checks** the user must judge from the
report.

1. **K is not pinned at a bound.** If $\log_{10}K$ reaches 0 or 12 (the constraint bounds),
   the true optimum lies outside the range or the data cannot determine K; that K (and model)
   is not trustworthy.

2. **The bootstrap CI is reasonably narrow.** If a K confidence interval spans several orders
   of magnitude, that binding constant is effectively undetermined — commonly seen for one of
   K₁/K₂ in 1:2/2:1.

3. **Estimated δ values are physically reasonable.** If a bound-state intrinsic shift is
   implausibly large (e.g., tens of ppm) or has a chemically counterintuitive sign, the model
   may be force-fitting the data.

4. **Saturation is reached.** The titration must extend far enough in equivalents that the
   curve approaches a plateau for K and δ to be jointly well-determined. Early curve segments
   alone cannot distinguish 1:1 from 1:2/2:1 — a common cause of small ΔBIC.

5. **Interpreting 1:2 / 2:1 cooperativity.** The ratio $K_2/K_1$ in a sequential model
   suggests cooperativity. Relative to the statistical expectation, a large ratio indicates
   positive cooperativity and a small ratio negative cooperativity. If this ratio is poorly
   determined, defer adopting the sequential model.

6. **Comparison with the non-binding model.** If the non-binding (control) model is within
   ΔBIC 2 of the binding model, the evidence for binding is weak, and no binding constant
   should be reported. Treating a marginal fit as genuine binding is a well-documented source
   of false positives (Baker et al., 2024).

7. **The stoichiometry must be chemically feasible.** Even if 2:1 (H₂G) or 1:2 (HG₂) is
   selected statistically, confirm independently that the stoichiometry is possible for the
   actual host–guest system (number of functional groups, structural constraints).

---

## 7. Residual diagnostics (informational)

With `--residual-diagnostics`, the following are computed (implementation:
`residual_diagnostics` in `nmr_bind_fit/stats.py`). **These do not automatically change model
selection; they are informational.**

- **Shapiro–Wilk test** — residual normality. A small p-value casts doubt on the Gaussian
  residual assumption underlying BIC/AICc. For large samples, the computation is capped at a
  maximum sample size.
- **Durbin–Watson statistic** — first-order residual autocorrelation. ≈2 indicates no
  correlation; near 0 indicates positive autocorrelation (the model fails to capture curve
  structure — possible model misspecification); near 4 indicates negative autocorrelation.

If autocorrelation is pronounced (e.g., residuals show a systematic pattern along the
titration order), the model fails to explain the structure of the data regardless of having
the lowest BIC, and the model itself should be reconsidered.

---

## 8. Limitations and caveats

- **Relies on the fast-exchange assumption.** Not applicable to intermediate/slow-exchange
  data (§1).
- **Models host resonances only.** Weights are defined on a host basis.
- **Inter-peak correlation not modeled.** The diagonal-covariance assumption can overstate the
  effective information content for strongly correlated multi-peak data, making ΔBIC appear
  larger than warranted.
- **Single pooled variance for model comparison.** Strong heteroscedasticity across peaks can
  affect the BIC comparison.
- **Statistics give relative support within the candidate set only.** Untested stoichiometries
  (e.g., 2:2 or higher order) are not in the comparison. "Lowest BIC" means "best of the four
  tested," not "the absolute truth."
- **Small samples.** With few titration points, AICc may be undefined (NaN) and BIC's
  discriminating power degrades sharply. Adequate titration points and a saturation region take
  precedence over statistics.

---

## 9. Where to find these rules in the report outputs

| Item | File | Content |
|------|------|---------|
| Model-comparison table (BIC, AICc, RMSE, R², …) | `summary.csv` | Per-model values of the §4 indices |
| Selection decision and reasons | `decision.txt` | Provisional working model, ΔBIC, warnings (§5) |
| Combined report | `report.html` | Methods + plots + decision paragraphs |
| Per-model diagnostics | `dataset_*/model_*/` | Dataset-scoped plots, bootstrap histograms, correlation matrix |

---

## Appendix: decision-flow summary

```
Fit all four candidates (1:1, 1:2, 2:1, non-binding)
        │
        ▼
Gaussian likelihood → BIC (primary), AICc (supporting)
   (shared variance, k = p+1, n = # finite residuals)
        │
        ▼
Lowest-BIC model = provisional working model
        │
        ├─ ΔBIC < 2 ?               → yes: flag "weak discrimination"
        ├─ K bootstrap CI too wide? → yes: warn "CI too wide"
        ├─ logK pinned at 0/12 ?    → yes: estimate untrustworthy (user judgment)
        ├─ Close to non-binding ?   → yes: weak evidence for binding (user judgment)
        └─ Residual autocorr./non-normal? → yes: possible misspecification (user judgment)
        │
        ▼
Confirm the final model together with chemical plausibility (§6)
```

> Statistics narrow the candidates; the final step is always decided by chemistry.

---

## References

**Supramolecular binding-constant determination and data analysis**

- Thordarson, P. (2011). Determining association constants from titration experiments in
  supramolecular chemistry. *Chemical Society Reviews*, 40(3), 1305–1323.
  DOI: 10.1039/c1cs15071e.
- Hibbert, D. B., & Thordarson, P. (2016). The death of the Job plot, transparency, open
  science and online tools, uncertainty estimation methods and other developments in
  supramolecular chemistry data analysis. *Chemical Communications*, 52, 12792–12805.
  DOI: 10.1039/c6cc03888c.
- Lowe, A. J., Pfeffer, F. M., & Thordarson, P. (2012). Determining binding constants from
  ¹H NMR titration data using global and local methods: a case study using
  [n]polynorbornane-based anion hosts. *Supramolecular Chemistry*, 24(8), 585–594.
  DOI: 10.1080/10610278.2012.688972.
- Hargrove, A. E., Zhong, Z., Sessler, J. L., & Anslyn, E. V. (2010). Algorithms for the
  determination of binding constants and enantiomeric excess in complex host:guest equilibria
  using optical measurements. *New Journal of Chemistry*, 34(2), 348–354.
  DOI: 10.1039/b9nj00498j.
- Flood, A. H., Vander Griend, D. A., & Thordarson, P. (2023). Driving to K-town: the quest
  for quality binding constants. *Supramolecular Chemistry*, 34(9–10), 320–325.
  DOI: 10.1080/10610278.2024.2359949.
- Baker, A. E., Hoogstra, M. N., Pehrson, N. J., Jipping, A. E., Daspit, O. R., Grabill, M. A.,
  Stonehouse, A. A., Vander Griend, D. A., Santoro, F., Brancaccio, D., Carosella, L.,
  Raniolo, S., & Limongelli, V. (2024). To bind or not to bind: diagnosing false positives for
  1:1 binding to G-Protein. *Supramolecular Chemistry*, 36, 18. DOI: 10.1080/10610278.2024.2440343.

**Statistics and model selection**

- Schwarz, G. (1978). Estimating the Dimension of a Model. *The Annals of Statistics*, 6(2),
  461–464.
- Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *Journal of the American Statistical
  Association*, 90(430), 773–795.
- Hurvich, C. M., & Tsai, C.-L. (1989). Regression and Time Series Model Selection in Small
  Samples. *Biometrika*, 76(2), 297–307.
