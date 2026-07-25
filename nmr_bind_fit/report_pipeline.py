"""Report-building pipeline extracted from CLI orchestration."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .equilibrium import SpeciesResult
from .fit import FitResult
from .fit_uncertainty import CONFIDENCE_LEVEL
from .io import Dataset
from .models import display_model_name, split_params_multi
from .plots import (
    plot_fraction_bound,
    plot_isotherms,
    plot_residuals,
)
from .report import DecisionEntry, ModelEntry, ParamEntry

SUMMARY_LABELS = {
    "dataset": "Dataset",
    "model": "Model",
    "status": "Status",
    "K": "Binding constant (M⁻¹)",
    "k_ci": "95 % CI",
    "R2": "R² (mean)",
    "BIC": "BIC",
    "AICc": "AICc",
}

LOGK_BOUND_ATOL = 1e-7


STATS_LABELS = {
    "n": "Observations (n)",
    "p": "Fitted parameters (p)",
    "dof": "Residual degrees of freedom",
    "R2": "Coefficient of determination (mean per-peak)",
    "R2_per_peak": "R² per peak",
    "RSS": "Residual sum of squares",
    "RMSE": "Root mean square error",
    "BIC": "Bayesian Information Criterion",
    "AICc": "Corrected Akaike Information Criterion",
    "penalty_events": "Optimization penalty events",
    "logk_se": "Standard error (log10 K)",
    "solver_points": "Equilibrium solver points",
    "solver_fail": "Equilibrium solver failures",
    "residual_n": "Residual count",
    "shapiro_stat": "Shapiro-Wilk statistic",
    "shapiro_p": "Shapiro-Wilk p-value",
    "durbin_watson": "Durbin-Watson statistic",
}


def _label_summary_key(key: str) -> str:
    return SUMMARY_LABELS.get(key, key)


def _label_stats_key(key: str) -> str:
    return STATS_LABELS.get(key, key)


def _safe_pow10(values: np.ndarray) -> np.ndarray:
    # Clip log10 inputs to avoid overflow in exp.
    log10_max = np.log(np.finfo(float).max) / np.log(10.0)
    log10_min = np.log(np.finfo(float).tiny) / np.log(10.0)
    clipped = np.clip(values, log10_min, log10_max)
    return np.exp(clipped * np.log(10.0))


def _safe_path_token(value: str) -> str:
    # Sanitize and bound free-form labels before using them as path components.
    # Callers prefix an ordinal, so this does not have to keep truncated labels
    # distinct on its own.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    safe = safe or "dataset"
    return safe[:80].rstrip("._-") or "dataset"


def _dataset_dir_tokens(labels: Sequence[str]) -> dict[str, str]:
    # Always include the ordinal so labels that differ only by case remain
    # distinct on the case-insensitive filesystems common on macOS and Windows.
    return {
        label: f"{idx:02d}_{_safe_path_token(label)}"
        for idx, label in enumerate(labels, start=1)
    }


def _replicate_dataset_dir_labels(datasets: Sequence[Dataset]) -> list[str]:
    # Build deterministic, collision-free directory labels per replicate dataset.
    labels: list[str] = []
    for idx, ds in enumerate(datasets, start=1):
        base = str(ds.name or f"dataset_{idx}")
        path = ds.path
        if path is not None:
            filename = Path(path).name
            if filename and filename != base:
                base = f"{base}_{filename}"
        labels.append(f"{idx:02d}_{_safe_path_token(base)}")
    return labels


def _format_dropped_peaks(datasets: Sequence[Dataset]) -> str:
    # Format dropped ppm columns for report warnings.
    items: list[str] = []
    multi = len(datasets) > 1
    for ds in datasets:
        dropped_peaks = ds.dropped_peaks
        if dropped_peaks:
            cols = ", ".join(dropped_peaks)
            if multi:
                items.append(f"{ds.name}: {cols}")
            else:
                items.append(cols)
    if not items:
        return "None"
    return "; ".join(items)


def _format_dropped_rows(datasets: Sequence[Dataset]) -> str:
    # Format concentration rows dropped before fitting for report warnings.
    items: list[str] = []
    multi = len(datasets) > 1
    for ds in datasets:
        dropped_rows = int(ds.dropped_rows)
        if dropped_rows > 0:
            if multi:
                items.append(f"{ds.name}: {dropped_rows}")
            else:
                items.append(str(dropped_rows))
    if not items:
        return "None"
    return "; ".join(items)


def _accumulate_solver_stats(species_list: list[SpeciesResult]) -> dict[str, int] | None:
    # Combine the per-point solver counts the report can show. A clean solve
    # reports nothing, so only the totals behind a failure are accumulated.
    totals = {"solver_points": 0, "solver_fail": 0}
    found = False
    for species in species_list:
        stats = species.solver_stats
        if stats is None:
            continue
        found = True
        totals["solver_points"] += int(stats.points)
        totals["solver_fail"] += int(stats.fail)
    if not found:
        return None
    return totals


def _build_param_entries(res: FitResult) -> list[ParamEntry]:
    # Convert fitted parameters into report entries with asymptotic standard errors.
    param_se = res.uncertainty.param_se if res.uncertainty is not None else None

    params = []
    for i, name in enumerate(res.param_names):
        value = float(res.params[i])
        se = float(param_se[i]) if param_se is not None else float("nan")
        if name in {"logK", "logK1", "logK2"}:
            k_value = float(_safe_pow10(np.array(value)))
            # Delta method: K = 10**logK, so SE(K) = K * ln(10) * SE(logK).
            k_se = k_value * float(np.log(10.0)) * se if np.isfinite(se) else float("nan")
            params.append(
                ParamEntry(
                    name=name.replace("logK", "K"),
                    value=k_value,
                    se=k_se,
                )
            )
        else:
            params.append(
                ParamEntry(
                    name=name,
                    value=value,
                    se=se,
                )
            )
    return params


def _collect_plot_artifacts(
    res: FitResult,
    model_name: str,
    ds_label: str,
    out_dir: Path,
    dataset_dir_token: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    # Write model plots and return relative PNG paths plus display labels.
    model_dir = out_dir / f"model_{_safe_path_token(model_name)}"
    if ds_label != "Simultaneous Fitting":
        token = dataset_dir_token or _safe_path_token(ds_label)
        model_dir = model_dir / f"dataset_{token}"
    elif len(res.datasets) > 1:
        model_dir = model_dir / f"dataset_{_safe_path_token(ds_label)}"
    model_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: list[str] = []
    plot_labels: dict[str, str] = {}
    replicate_dir_labels = _replicate_dataset_dir_labels(res.datasets) if len(res.datasets) > 1 else []
    model_spec = res.model
    logk, deltas = split_params_multi(res.params, model_spec, res.datasets)
    for idx, (ds, delta, residual) in enumerate(zip(res.datasets, deltas, res.residuals)):
        ds_dir = model_dir
        if len(res.datasets) > 1:
            ds_dir = model_dir / f"dataset_{replicate_dir_labels[idx]}"
        isotherm_files = plot_isotherms(model_spec, ds, logk, delta, ds_dir)
        residual_files = plot_residuals(model_spec, ds, residual, ds_dir)
        frac_files = plot_fraction_bound(model_spec, ds, logk, delta, ds_dir)
        for path in isotherm_files + residual_files + frac_files:
            if path.suffix.lower() == ".png":
                plot_paths.append(path.relative_to(out_dir).as_posix())
        for peak, path in zip(
            ds.y_cols,
            [path for path in isotherm_files if path.suffix.lower() == ".png"],
        ):
            plot_labels[path.relative_to(out_dir).as_posix()] = str(peak)
        for peak, path in zip(
            ds.y_cols,
            [path for path in residual_files if path.suffix.lower() == ".png"],
        ):
            plot_labels[path.relative_to(out_dir).as_posix()] = str(peak)

    uncertainty = res.uncertainty
    if uncertainty is not None and np.all(np.isfinite(uncertainty.correlation)):
        # The correlation matrix comes straight from the fit covariance, in the
        # fitted-parameter basis. A perfect fit has zero standard errors, which
        # leaves the correlation undefined and nothing worth writing.
        corr_path = model_dir / "correlation.csv"
        np.savetxt(corr_path, uncertainty.correlation, delimiter=",", fmt="%.6g")

    return plot_paths, plot_labels


def _logk_se_text(res: FitResult) -> str:
    # Format the asymptotic standard error in log10(K) space.
    if res.uncertainty is None or res.model.n_logk == 0:
        return "N/A"
    se_vals = np.asarray(res.uncertainty.param_se[: res.model.n_logk], dtype=float)
    if not np.all(np.isfinite(se_vals)):
        return "N/A"
    return ";".join(f"{value:.6g}" for value in se_vals)


def _k_ci(res: FitResult) -> tuple[np.ndarray, np.ndarray]:
    # Return the log10(K) confidence interval mapped into linear K space.
    if res.uncertainty is None or res.model.n_logk == 0:
        return (
            np.full((res.model.n_logk,), np.nan),
            np.full((res.model.n_logk,), np.nan),
        )
    return (
        _safe_pow10(np.asarray(res.uncertainty.logk_ci_low, dtype=float)),
        _safe_pow10(np.asarray(res.uncertainty.logk_ci_high, dtype=float)),
    )


def _logk_names(n_logk: int) -> list[str]:
    if n_logk == 1:
        return ["K"]
    return [f"K{i + 1}" for i in range(n_logk)]


def _format_k_values(k_vals: np.ndarray, n_logk: int) -> str:
    if n_logk == 0:
        return "N/A"
    names = _logk_names(n_logk)
    values = [f"{value:.6g}" if np.isfinite(value) else "N/A" for value in k_vals]
    if n_logk == 1:
        return values[0]
    return "; ".join(f"{name}={value}" for name, value in zip(names, values))


def _format_k_ci(k_ci_low: np.ndarray, k_ci_high: np.ndarray, n_logk: int) -> str:
    finite = np.isfinite(k_ci_low) & np.isfinite(k_ci_high)
    if k_ci_low.size == 0 or not np.any(finite):
        return "N/A"
    names = _logk_names(n_logk)
    intervals = []
    for idx, (low, high) in enumerate(zip(k_ci_low, k_ci_high)):
        interval = f"[{low:.6g}, {high:.6g}]" if np.isfinite(low) and np.isfinite(high) else "N/A"
        if n_logk == 1:
            intervals.append(interval)
        else:
            intervals.append(f"{names[idx]}={interval}")
    return "; ".join(intervals)


def _logk_bound_warnings(res: FitResult) -> list[str]:
    if res.model.n_logk == 0:
        return []
    # Compare against the bounds the fit actually used; when they are unknown
    # (e.g. an unbounded programmatic fit) no bound was active, so pinning a
    # valid estimate such as K=1 or K=1e12 would be misleading.
    bounds = res.logk_bounds
    if bounds is None:
        return []
    low, high = float(bounds[0]), float(bounds[1])
    logk_vals = np.asarray(res.params[: res.model.n_logk], dtype=float)
    names = _logk_names(res.model.n_logk)
    warnings: list[str] = []
    for name, value in zip(names, logk_vals):
        if not np.isfinite(value):
            continue
        if np.isclose(value, low, atol=LOGK_BOUND_ATOL, rtol=0.0):
            warnings.append(f"{name} is pinned at the lower log10(K) bound ({low:g})")
        elif np.isclose(value, high, atol=LOGK_BOUND_ATOL, rtol=0.0):
            warnings.append(f"{name} is pinned at the upper log10(K) bound ({high:g})")
    return warnings


def _solver_stats_for_result(res: FitResult) -> dict[str, int] | None:
    # Collect solver diagnostics only for nonlinear root-solved models.
    if res.model.name not in {"12", "21"}:
        return None
    return _accumulate_solver_stats(res.species)


def _build_model_warnings(
    args: argparse.Namespace,
    res: FitResult,
    solver_stats: dict[str, int] | None,
) -> list[str]:
    # Build per-model warning messages for report rendering.
    warnings = []
    datasets = res.datasets

    dropped_peaks = _format_dropped_peaks(datasets)
    if dropped_peaks != "None":
        warnings.append(f"Dropped chemical shift columns with missing or non-finite values: {dropped_peaks}")

    dropped_rows = _format_dropped_rows(datasets)
    if dropped_rows != "None":
        warnings.append(f"Dropped rows with missing required concentrations: {dropped_rows}")

    if not np.isfinite(res.bic):
        if res.n <= res.p + 1:
            warnings.append(
                "BIC/AICc unavailable: finite observations are not greater than fitted parameters plus variance"
            )
        elif np.isfinite(res.rss) and res.rss <= 0:
            warnings.append("BIC/AICc unavailable: residual variance is zero")
        else:
            warnings.append("BIC/AICc unavailable for this fit")
    elif not np.isfinite(res.aicc):
        warnings.append(
            "AICc unavailable: too few observations for the small-sample correction; BIC is still reported"
        )

    warnings.extend(_logk_bound_warnings(res))

    k_ci_low, k_ci_high = _k_ci(res)
    if (
        args.ci_width is not None
        and k_ci_low.size > 0
        and np.any(np.isfinite(k_ci_low) & np.isfinite(k_ci_high) & ((k_ci_high - k_ci_low) > args.ci_width))
    ):
        warnings.append("K confidence interval is wider than the requested threshold")

    penalty_count = int(res.penalty_count)
    if penalty_count > 0:
        warnings.append(f"optimization penalty residual events: {penalty_count}")

    if solver_stats is not None:
        n_fail = solver_stats["solver_fail"]
        n_points = solver_stats["solver_points"]
        if n_fail > 0 and n_points > 0:
            warnings.append(f"solver failures ({n_fail}/{n_points})")

    for ds, species in zip(datasets, res.species):
        stats = species.solver_stats
        if stats is None:
            continue
        failed_indices = list(stats.failed_indices)
        if not failed_indices:
            continue
        preview = ", ".join(str(idx) for idx in failed_indices[:10])
        if len(failed_indices) > 10:
            preview = f"{preview}, ..."
        warnings.append(f"{ds.name}: solver-failed points [{preview}]")

    return warnings


def _build_stats_dict(res: FitResult, solver_stats: dict[str, int] | None) -> dict[str, str]:
    """Build the stats block for one model card.

    Only reported fits reach this point, and reporting a fit already means it
    passed the identifiability gate, so the rank, condition number and logK
    sensitivity behind that gate would always read as passing values. The
    thresholds are stated once in the methods text instead. Counters that are
    only meaningful when something went wrong are included only then.
    """
    stats_base = {
        "R2": f"{res.r2:.6g}" if np.isfinite(res.r2) else "N/A",
        "RSS": f"{res.rss:.6g}",
        "RMSE": f"{res.rmse:.6g}",
        "BIC": f"{res.bic:.6g}" if np.isfinite(res.bic) else "N/A",
        "AICc": f"{res.aicc:.6g}" if np.isfinite(res.aicc) else "N/A",
    }
    if len(res.r2_per_peak) > 1:
        stats_base["R2_per_peak"] = ";".join(f"{value:.6g}" for value in res.r2_per_peak)
    stats_base["n"] = str(res.n)
    stats_base["p"] = str(res.p)
    stats_base["dof"] = str(res.dof)
    if res.penalty_count > 0:
        stats_base["penalty_events"] = str(res.penalty_count)
    if res.uncertainty is not None and res.model.n_logk > 0:
        stats_base["logk_se"] = _logk_se_text(res)
    if solver_stats is not None and solver_stats["solver_fail"] > 0:
        # A clean per-point solve is the norm; the counts only inform a failure.
        stats_base["solver_points"] = str(solver_stats["solver_points"])
        stats_base["solver_fail"] = str(solver_stats["solver_fail"])
    diagnostics = res.residual_diagnostics
    if diagnostics:
        if "residual_n" in diagnostics:
            stats_base["residual_n"] = f"{diagnostics['residual_n']:.0f}"
        if "shapiro_stat" in diagnostics:
            stats_base["shapiro_stat"] = f"{diagnostics['shapiro_stat']:.6g}"
        if "shapiro_p" in diagnostics:
            stats_base["shapiro_p"] = f"{diagnostics['shapiro_p']:.6g}"
        if "durbin_watson" in diagnostics:
            stats_base["durbin_watson"] = f"{diagnostics['durbin_watson']:.6g}"
    return {_label_stats_key(k): v for k, v in stats_base.items()}


def _build_summary_row(
    res: FitResult,
    ds_label: str,
    display_name: str,
) -> dict[str, str]:
    # Build one row of the model comparison table.
    logk_vals = res.params[: res.model.n_logk]
    k_vals = _safe_pow10(logk_vals)
    k_str = _format_k_values(k_vals, res.model.n_logk)

    k_ci_low, k_ci_high = _k_ci(res)
    k_ci_text = _format_k_ci(k_ci_low, k_ci_high, res.model.n_logk)

    summary_base = {
        "dataset": ds_label,
        "model": display_name,
        "status": "success",
        "K": k_str,
        "k_ci": k_ci_text,
        "R2": f"{res.r2:.6g}" if np.isfinite(res.r2) else "N/A",
        "BIC": f"{res.bic:.6g}" if np.isfinite(res.bic) else "N/A",
        "AICc": f"{res.aicc:.6g}" if np.isfinite(res.aicc) else "N/A",
    }
    return {_label_summary_key(k): v for k, v in summary_base.items()}


def _build_failure_summary_row(ds_label: str, display_name: str) -> dict[str, str]:
    # Keep failed candidates visible in the comparison table for auditability.
    summary_base = {
        "dataset": ds_label,
        "model": display_name,
        "status": "failed",
        "K": "N/A",
        "k_ci": "N/A",
        "R2": "N/A",
        "BIC": "N/A",
        "AICc": "N/A",
    }
    return {_label_summary_key(k): v for k, v in summary_base.items()}


def _build_model_entry(
    args: argparse.Namespace,
    key: str,
    model_name: str,
    res: FitResult,
    out_dir: Path,
    dataset_dir_token: str | None = None,
) -> tuple[ModelEntry, dict[str, str]]:
    # Build one report model section and its matching summary row.
    display_name = display_model_name(model_name)
    params = _build_param_entries(res)
    plot_paths, plot_labels = _collect_plot_artifacts(
        res,
        model_name,
        key,
        out_dir,
        dataset_dir_token,
    )
    solver_stats = _solver_stats_for_result(res)
    warnings = _build_model_warnings(args, res, solver_stats)
    stats_dict = _build_stats_dict(res, solver_stats)
    model_entry = ModelEntry(
        dataset=key,
        model=display_name,
        stats=stats_dict,
        params=params,
        plots=plot_paths,
        warnings=warnings,
        plot_labels=plot_labels,
    )
    summary_row = _build_summary_row(res, key, display_name)
    return model_entry, summary_row


def build_report_artifacts(
    args: argparse.Namespace,
    ordered_keys: list[str],
    results_by_key: dict[str, dict[str, FitResult]],
    out_dir: Path,
) -> tuple[list[dict[str, str]], list[ModelEntry], list[str]]:
    # Convert fit results into summary rows, model entries, and top-level warnings.
    summary_rows: list[dict[str, str]] = []
    model_entries: list[ModelEntry] = []
    report_warnings: list[str] = []
    dataset_dir_tokens = _dataset_dir_tokens(ordered_keys)

    for key in ordered_keys:
        model_map = results_by_key.get(key, {})
        for model_name, res in model_map.items():
            if res.success:
                continue
            report_warnings.append(
                f"{key}: excluded {display_model_name(model_name)} (fit failed: {res.message})"
            )
            summary_rows.append(_build_failure_summary_row(key, display_model_name(model_name)))
        for model_name, res in model_map.items():
            if not res.success:
                continue
            model_entry, summary_row = _build_model_entry(
                args,
                key,
                model_name,
                res,
                out_dir,
                dataset_dir_tokens.get(key),
            )
            model_entries.append(model_entry)
            summary_rows.append(summary_row)

    return summary_rows, model_entries, report_warnings


def _compose_methods_sections(args: argparse.Namespace, datasets: Sequence[Dataset]) -> list[dict[str, str]]:
    # Canonical structured methods content for the HTML report.
    sections: list[dict[str, str]] = []
    # K is reported in M⁻¹ (molar inputs).
    k_unit = "M⁻¹"

    # 1. Data Interpretation
    sections.append(
        {
            "title": "Data Interpretation",
            "content": (
                "NMR chemical shift titration data were interpreted under a fast-exchange assumption, "
                "with observed host-resonance chemical shifts modeled as population-weighted averages of "
                "the free and bound chemical states. Under this regime, the observed shift δ_obs at each "
                "titration point is expressed as: δ_obs = Σᵢ xᵢ · δᵢ, where xᵢ is the mole fraction of "
                "host in state i and δᵢ is the intrinsic chemical shift of that state."
            ),
        }
    )

    # 2. Binding Models
    replicate_note = ""
    if args.replicates and len(datasets) > 1:
        replicate_note = (
            " Replicate datasets were fit simultaneously with shared binding constants (K) and "
            "replicate-specific chemical shift parameters (δ), thereby improving precision in K estimates."
        )
    sections.append(
        {
            "title": "Binding Models",
            "content": (
                "Candidate binding stoichiometries evaluated were: "
                "(i) 1:1 binding (H + G ⇌ HG, characterized by K), "
                "(ii) 1:2 sequential binding (H + G ⇌ HG, K₁; HG + G ⇌ HG₂, K₂), "
                "(iii) 2:1 sequential binding (H + G ⇌ HG, K₁; H + HG ⇌ H₂G, K₂), and "
                "(iv) a non-binding linear drift control model (δ = a₀ + a₁·[G]ₜ/[H]ₜ)."
                + replicate_note
            ),
        }
    )

    # 3. Parameter Estimation
    sections.append(
        {
            "title": "Parameter Estimation",
            "content": (
                "Parameters were estimated by nonlinear least-squares minimization using "
                "scipy.optimize.least_squares (Trust Region Reflective algorithm). "
                "The 1:1 equilibrium was solved analytically via the closed-form quadratic solution. "
                "The 1:2 and 2:1 equilibria were solved numerically point-by-point using Brent's method "
                "over the physical free-guest bracket [0, [G]ₜ] (scipy.optimize.brentq; "
                "scale-adaptive xtol = 10⁻¹³ times the estimated free-guest scale, "
                "rtol = 8 machine epsilons, and a bracket-scale-adaptive iteration budget with a minimum of 200). "
                "Binding constants were parameterized as log₁₀(K) and constrained to [0, 12] "
                f"(K ∈ [1, 10¹²] {k_unit}) during optimization to ensure stable, physically meaningful estimation. "
                "Residuals were divided by one global observed-response scale during optimization, which leaves "
                "the least-squares minimum and relative residual weights unchanged while making termination "
                "behavior invariant to the response unit. "
                "Fits were excluded from model comparison unless they had positive residual degrees of freedom, "
                "a full-column-rank dimensionless Jacobian with condition number at most 10⁶, minimum dimensionless "
                "log₁₀(K) RMS sensitivity of at least 10⁻⁴, and no active log₁₀(K) bound. "
                "ppm columns containing missing or non-finite values were dropped before fitting. "
                "For 1:2 and 2:1 models, per-point solver failures used fail-fast behavior."
            ),
        }
    )

    # 4. Model Comparison
    sections.append(
        {
            "title": "Statistical Model Comparison",
            "content": (
                "Model comparison employed the Bayesian Information Criterion (BIC) as the primary "
                "ranking index, defined as BIC = −2 ℓ(θ̂) + k·ln(n), where ℓ(θ̂) is the maximized "
                "Gaussian log-likelihood, n is the number of observations, and k is the number of "
                "estimated parameters. Under the assumption of i.i.d. Gaussian residuals with MLE "
                "variance σ̂² = RSS/n, this equals n·ln(2π·RSS/n) + n + k·ln(n); for model ranking "
                "the terms additive in n cancel, yielding equivalent ordering to n·ln(RSS/n) + k·ln(n). "
                "The corrected Akaike Information Criterion (AICc) was reported as "
                "supporting information. These criteria were interpreted as measures of relative support "
                "among the tested candidates and considered together with chemical plausibility and "
                "spectral consistency. One shared residual variance term (σ²) was estimated for model "
                "comparison and counted as one additional information-criteria parameter (k = p + 1). "
                "The effective sample size n was defined as the total number of finite residual scalars "
                "across all datasets, titration points, and ppm peaks; missing observations were excluded. "
                "Residual correlation between peaks was not modeled (diagonal covariance assumed). "
                "A ΔBIC < 2 between the best and next-best model was flagged as weak discrimination."
            ),
        }
    )

    # 5. Uncertainty Quantification
    uncertainty_note = (
        "Parameter uncertainty was estimated from the asymptotic covariance matrix of the "
        "converged fit, cov = (RSS / dof) · (JᵀJ)⁻¹, where J is the Jacobian at the optimum and "
        "dof is the residual degrees of freedom. Standard errors are the square roots of its "
        f"diagonal, and the {CONFIDENCE_LEVEL:.0%} confidence interval for each log₁₀(K) is "
        "the estimate plus or minus the Student-t quantile for that dof times its standard error. "
        "Intervals are reported in log₁₀(K) and converted to K for display, so they are asymmetric "
        "about K; the standard error quoted for K itself uses the delta method, "
        "SE(K) = K · ln(10) · SE(log₁₀K). Parameter correlations are taken from the same "
        "covariance matrix. This is the standard large-sample approximation for nonlinear least "
        "squares and assumes the model is locally linear near the optimum, which the reported "
        "fits already satisfy through the rank, conditioning, and sensitivity gates described "
        "above."
    )
    sections.append(
        {
            "title": "Uncertainty Quantification",
            "content": uncertainty_note,
        }
    )

    return sections


def build_methods_sections(
    args: argparse.Namespace, datasets: Sequence[Dataset]
) -> list[dict[str, str]]:
    """Return structured methods as list of {title, content} dicts for detailed HTML reporting."""
    return _compose_methods_sections(args, datasets)


def build_decisions(
    args: argparse.Namespace,
    ordered_keys: list[str],
    results_by_key: dict[str, dict[str, FitResult]],
) -> list[DecisionEntry]:
    """Rank each dataset's candidates and record why the leader was chosen.

    Datasets with no finitely ranked candidate yield no entry; the reason is
    already reported through the warnings shown alongside them.
    """
    decision_entries: list[DecisionEntry] = []

    for key in ordered_keys:
        model_map = results_by_key.get(key, {})
        bic_sorted = sorted(
            (res for res in model_map.values() if res.success and np.isfinite(res.bic)),
            key=lambda r: r.bic,
        )
        if not bic_sorted:
            continue

        best = bic_sorted[0]
        reasons = [
            (f"Within the tested model set, this model gave the lowest Bayesian Information Criterion "
            f"(BIC={best.bic:.6g}). This ranking indicates relative support only and should be interpreted with "
            "chemical plausibility and spectral behavior")
        ]
        if len(bic_sorted) > 1:
            delta_bic = float(bic_sorted[1].bic - best.bic)
            if np.isfinite(delta_bic) and delta_bic < 2.0:
                reasons.append("BIC separation from the next candidate was small, so model discrimination is weak")
        if args.ci_width is not None and best.model.n_logk > 0:
            k_ci_low, k_ci_high = _k_ci(best)
            if k_ci_low.size > 0 and np.any(
                np.isfinite(k_ci_low)
                & np.isfinite(k_ci_high)
                & ((k_ci_high - k_ci_low) > args.ci_width)
            ):
                reasons.append("The K confidence interval is wider than the requested threshold")
        decision_entries.append(
            DecisionEntry(
                dataset=key,
                recommended_model=display_model_name(best.model.name),
                reasons=reasons,
            )
        )

    return decision_entries
