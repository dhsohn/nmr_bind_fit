"""Report-building pipeline extracted from CLI orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import split_params_multi
from .plots import plot_bootstrap_hist, plot_fraction_bound, plot_isotherms, plot_residuals
from .report import DecisionEntry, ModelEntry, ParamEntry


SUMMARY_LABELS = {
    "dataset": "Dataset",
    "model": "Model",
    "K": "Binding constant",
    "bootstrap_K_CI": "Confidence Interval(CI)",
    "bootstrap_K_SE": "Standard Error(SE)",
    "RSS": "Residual sum of squares",
    "RMSE": "Root mean square error",
    "BIC": "Bayesian Information Criterion",
    "AICc": "Corrected Akaike Information Criterion",
}


STATS_LABELS = {
    "RSS": "Residual sum of squares",
    "RMSE": "Root mean square error",
    "BIC": "Bayesian Information Criterion",
    "AICc": "Corrected Akaike Information Criterion",
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


def _safe_std(values: np.ndarray) -> float:
    # Scale before std to reduce catastrophic cancellation.
    vals = values[np.isfinite(values)]
    if vals.size <= 1:
        return float("nan")
    scale = np.max(np.abs(vals))
    if not np.isfinite(scale) or scale == 0:
        return 0.0
    scaled = vals / scale
    return float(np.std(scaled, ddof=1) * scale)


def _filter_finite_rows(values: np.ndarray) -> np.ndarray:
    # Drop rows with any non-finite values.
    if values.ndim == 1:
        return values[np.isfinite(values)]
    mask = np.all(np.isfinite(values), axis=1)
    return values[mask]


def _format_dropped_peaks(datasets: Sequence[object]) -> str:
    # Format dropped ppm columns for report warnings.
    items: List[str] = []
    multi = len(datasets) > 1
    for ds in datasets:
        dropped_peaks = getattr(ds, "dropped_peaks", [])
        if dropped_peaks:
            cols = ", ".join(dropped_peaks)
            if multi:
                items.append(f"{ds.name}: {cols}")
            else:
                items.append(cols)
    if not items:
        return "None"
    return "; ".join(items)


def _accumulate_solver_stats(species_list: List[object]) -> Optional[Dict[str, object]]:
    # Combine solver statistics across species lists.
    totals = {
        "solver_points": 0,
        "solver_newton_success": 0,
        "solver_newton_fail": 0,
        "solver_newton_max_iter": 0,
        "solver_fallback_success": 0,
        "solver_fallback_fail": 0,
        "solver_fallback_method": set(),
    }
    found = False
    for species in species_list:
        stats = getattr(species, "solver_stats", None)
        if stats is None:
            continue
        found = True
        totals["solver_points"] += int(getattr(stats, "points", 0))
        totals["solver_newton_success"] += int(getattr(stats, "newton_success", 0))
        totals["solver_newton_fail"] += int(getattr(stats, "newton_fail", 0))
        totals["solver_newton_max_iter"] += int(getattr(stats, "newton_max_iter", 0))
        totals["solver_fallback_success"] += int(getattr(stats, "fallback_success", 0))
        totals["solver_fallback_fail"] += int(getattr(stats, "fallback_fail", 0))
        method = getattr(stats, "fallback_method", None)
        if method:
            totals["solver_fallback_method"].add(str(method))
    if not found:
        return None
    methods = totals["solver_fallback_method"]
    totals["solver_fallback_method"] = ", ".join(sorted(methods)) if methods else "N/A"
    return totals


def _build_param_entries(res) -> List[ParamEntry]:
    # Convert fitted parameters into report entries and optional bootstrap SE.
    bootstrap_samples = None
    if res.bootstrap is not None and res.bootstrap.param_samples.size > 0:
        bootstrap_samples = res.bootstrap.param_samples

    params = []
    for i, name in enumerate(res.param_names):
        value = float(res.params[i])
        se = float("nan")
        if bootstrap_samples is not None and bootstrap_samples.shape[0] > 1:
            sample_col = bootstrap_samples[:, i]
            if name in {"logK", "logK1", "logK2"}:
                sample_col = _safe_pow10(sample_col)
            se = _safe_std(sample_col)
        if name in {"logK", "logK1", "logK2"}:
            params.append(
                ParamEntry(
                    name=name.replace("logK", "K"),
                    value=float(_safe_pow10(np.array(value))),
                    se=se,
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


def _collect_plot_paths(res, model_name: str, ds_label: str, out_dir: Path) -> List[str]:
    # Write model plots and return PNG paths relative to output root.
    model_dir = out_dir / f"model_{model_name}"
    if len(res.datasets) > 1:
        model_dir = model_dir / f"dataset_{ds_label}"
    model_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: List[str] = []
    logk, deltas = split_params_multi(res.params, res.model, res.datasets)
    for ds, delta, residual in zip(res.datasets, deltas, res.residuals):
        ds_dir = model_dir
        if len(res.datasets) > 1:
            ds_dir = model_dir / f"dataset_{ds.name}"
        isotherm_files = plot_isotherms(res.model, ds, logk, delta, ds_dir)
        residual_files = plot_residuals(res.model, ds, residual, ds_dir)
        frac_files = plot_fraction_bound(res.model, ds, logk, delta, ds_dir)
        for path in isotherm_files + residual_files + frac_files:
            if path.suffix.lower() == ".png":
                plot_paths.append(str(path.relative_to(out_dir)))

    if res.bootstrap is not None and res.bootstrap.param_samples.shape[0] > 1:
        samples = res.bootstrap.param_samples.copy()
        for idx, name in enumerate(res.param_names):
            if name in {"logK", "logK1", "logK2"}:
                samples[:, idx] = _safe_pow10(samples[:, idx])
        data = _filter_finite_rows(samples)
        if data.shape[0] > 1:
            scale = np.nanmax(np.abs(data), axis=0)
            scale[~np.isfinite(scale) | (scale == 0)] = 1.0
            scaled = data / scale
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(scaled, rowvar=False)
            corr_path = model_dir / "correlation.csv"
            np.savetxt(corr_path, corr, delimiter=",", fmt="%.6g")

    if res.bootstrap is not None and res.model.n_logk > 0:
        k_names = ["K"] if res.model.n_logk == 1 else ["K1", "K2"]
        k_samples = _safe_pow10(res.bootstrap.param_samples[:, : res.model.n_logk])
        boot_files = plot_bootstrap_hist(k_samples, k_names, model_dir)
        for path in boot_files:
            if path.suffix.lower() == ".png":
                plot_paths.append(str(path.relative_to(out_dir)))

    return plot_paths


def _bootstrap_k_samples(res) -> np.ndarray:
    # Return finite bootstrap K samples (linear scale) for warnings and summary.
    if res.bootstrap is None or res.model.n_logk == 0 or res.bootstrap.param_samples.size == 0:
        return np.full((0, res.model.n_logk), np.nan)
    k_samples = _safe_pow10(res.bootstrap.param_samples[:, : res.model.n_logk])
    return _filter_finite_rows(k_samples)


def _solver_stats_for_result(res) -> Optional[Dict[str, object]]:
    # Collect solver diagnostics only for nonlinear root-solved models.
    if res.model.name not in {"12", "21"}:
        return None
    solver_stats = _accumulate_solver_stats(res.species)
    if solver_stats is None:
        return {
            "solver_points": "N/A",
            "solver_newton_success": "N/A",
            "solver_newton_fail": "N/A",
            "solver_newton_max_iter": "N/A",
            "solver_fallback_success": "N/A",
            "solver_fallback_fail": "N/A",
            "solver_fallback_method": "N/A",
        }
    return solver_stats


def _build_model_warnings(args: argparse.Namespace, res, solver_stats: Optional[Dict[str, object]]) -> List[str]:
    # Build per-model warning messages for report rendering.
    warnings = []

    dropped_peaks = _format_dropped_peaks(res.datasets)
    if dropped_peaks != "None":
        warnings.append(f"Dropped chemical shift columns with missing values: {dropped_peaks}")

    k_samples = _bootstrap_k_samples(res)
    if args.bootstrap_ci_width is not None and k_samples.size > 0:
        k_ci_low = np.percentile(k_samples, 2.5, axis=0)
        k_ci_high = np.percentile(k_samples, 97.5, axis=0)
        if np.any((k_ci_high - k_ci_low) > args.bootstrap_ci_width):
            warnings.append("bootstrap CI too wide")

    if res.bootstrap is not None and res.bootstrap.n_boot > 0:
        n_fail = res.bootstrap.n_boot - res.bootstrap.n_success
        if n_fail > 0:
            warnings.append(f"bootstrap failures: {n_fail} of {res.bootstrap.n_boot} iterations")

    if solver_stats is not None and solver_stats.get("solver_fallback_fail", 0) not in {"N/A", None}:
        n_fail = int(solver_stats.get("solver_fallback_fail", 0))
        n_points = int(solver_stats.get("solver_points", 0))
        if n_fail > 0 and n_points > 0:
            warnings.append(f"solver fallback failures ({n_fail}/{n_points})")

    return warnings


def _build_stats_dict(res, solver_stats: Optional[Dict[str, object]]) -> Dict[str, str]:
    # Build stats block for report tables.
    stats_base = {
        "RSS": f"{res.rss:.6g}",
        "RMSE": f"{res.rmse:.6g}",
        "BIC": f"{res.bic:.6g}",
        "AICc": f"{res.aicc:.6g}" if np.isfinite(res.aicc) else "N/A",
    }
    if solver_stats is not None:
        stats_base.update(
            {
                "solver_points": str(solver_stats["solver_points"]),
                "solver_newton_success": str(solver_stats["solver_newton_success"]),
                "solver_newton_fail": str(solver_stats["solver_newton_fail"]),
                "solver_newton_max_iter": str(solver_stats["solver_newton_max_iter"]),
                "solver_fallback_success": str(solver_stats["solver_fallback_success"]),
                "solver_fallback_fail": str(solver_stats["solver_fallback_fail"]),
                "solver_fallback_method": str(solver_stats["solver_fallback_method"]),
            }
        )
    return {_label_stats_key(k): v for k, v in stats_base.items()}


def _build_summary_row(res, ds_label: str, display_name: str) -> Dict[str, str]:
    # Build one row for summary.csv.
    logk_vals = res.params[: res.model.n_logk]
    k_vals = _safe_pow10(logk_vals)
    k_str = ";".join(f"{v:.6g}" for v in k_vals) if res.model.n_logk else "N/A"

    k_samples = _bootstrap_k_samples(res)
    if k_samples.size == 0:
        boot_k_ci = "N/A"
        boot_k_se = "N/A"
    else:
        k_ci_low = np.percentile(k_samples, 2.5, axis=0)
        k_ci_high = np.percentile(k_samples, 97.5, axis=0)
        boot_k_ci = ";".join(f"[{l:.6g}, {h:.6g}]" for l, h in zip(k_ci_low, k_ci_high))
        if k_samples.shape[0] > 1:
            boot_k_se_vals = [_safe_std(k_samples[:, i]) for i in range(k_samples.shape[1])]
            boot_k_se = ";".join(f"{v:.6g}" for v in boot_k_se_vals)
        else:
            boot_k_se = "N/A"

    summary_base = {
        "dataset": ds_label,
        "model": display_name,
        "K": k_str,
        "bootstrap_K_CI": boot_k_ci,
        "bootstrap_K_SE": boot_k_se,
        "RSS": f"{res.rss:.6g}",
        "RMSE": f"{res.rmse:.6g}",
        "BIC": f"{res.bic:.6g}",
        "AICc": f"{res.aicc:.6g}" if np.isfinite(res.aicc) else "N/A",
    }
    return {_label_summary_key(k): v for k, v in summary_base.items()}


def _build_model_entry(
    args: argparse.Namespace,
    key: str,
    model_name: str,
    res,
    out_dir: Path,
    display_model_name: Callable[[str], str],
) -> Tuple[ModelEntry, Dict[str, str]]:
    # Build one report model section and its matching summary row.
    display_name = display_model_name(model_name)
    params = _build_param_entries(res)
    plot_paths = _collect_plot_paths(res, model_name, key, out_dir)
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
    )
    summary_row = _build_summary_row(res, key, display_name)
    return model_entry, summary_row


def build_report_artifacts(
    args: argparse.Namespace,
    ordered_keys: List[str],
    results_by_key: Dict[str, Dict[str, object]],
    failures_by_key: Dict[str, List[Tuple[str, str]]],
    out_dir: Path,
    display_model_name: Callable[[str], str],
) -> Tuple[List[Dict[str, str]], List[ModelEntry], List[str]]:
    # Convert fit results into summary rows, model entries, and top-level warnings.
    summary_rows: List[Dict[str, str]] = []
    model_entries: List[ModelEntry] = []
    report_warnings: List[str] = []

    for key in ordered_keys:
        model_map = results_by_key.get(key, {})
        failures = failures_by_key.get(key, [])
        for model_name, message in failures:
            report_warnings.append(
                f"{key}: excluded {display_model_name(model_name)} (optimizer did not converge: {message})"
            )
        if not model_map:
            continue
        for model_name, res in model_map.items():
            model_entry, summary_row = _build_model_entry(
                args,
                key,
                model_name,
                res,
                out_dir,
                display_model_name,
            )
            model_entries.append(model_entry)
            summary_rows.append(summary_row)

    return summary_rows, model_entries, report_warnings


def build_methods_text(args: argparse.Namespace, datasets: Sequence[object]) -> str:
    # Build static methods narrative with runtime bootstrap/replicate notes.
    bootstrap_note = (
        f"Bootstrap uncertainty was evaluated with {args.bootstrap} iterations using "
        f"{args.bootstrap_method} resampling."
        if args.bootstrap > 0
        else "Bootstrap uncertainty was not evaluated."
    )
    replicate_note = ""
    if args.replicates and len(datasets) > 1:
        replicate_note = (
            "Replicate datasets were fit simultaneously with shared binding constants and "
            "replicate-specific chemical shifts. "
        )
    return (
        "NMR chemical shift titration data were interpreted under a fast-exchange assumption, with observed "
        "host-resonance shifts modeled as population-weighted averages of chemical states. Candidate stoichiometries "
        "were 1:1 binding (H + G <=> HG), 1:2 binding (H + G <=> HG; HG + G <=> HG2), 2:1 binding "
        "(H + G <=> HG; H + HG <=> H2G), and a non-binding linear drift model. Parameters were estimated by "
        "nonlinear least squares (scipy.optimize.least_squares). The 1:1 equilibrium was solved analytically, while "
        "1:2 and 2:1 equilibria were solved numerically point-by-point with Newton-Raphson and bisection fallback. "
        f"Bootstrap refits used a small logK start perturbation (std {args.bootstrap_logk_jitter:.3g} in log10 K). "
        "Model comparison used BIC as the primary ranking index and AICc as supporting information. These criteria "
        "were interpreted as relative support among tested candidates and considered together with chemical "
        "plausibility and spectral consistency. One shared residual variance term was estimated for model comparison. "
        + replicate_note
        + bootstrap_note
    )


def build_decisions(
    args: argparse.Namespace,
    ordered_keys: List[str],
    results_by_key: Dict[str, Dict[str, object]],
    failures_by_key: Dict[str, List[Tuple[str, str]]],
    display_model_name: Callable[[str], str],
) -> Tuple[List[str], List[DecisionEntry]]:
    # Build decision.txt lines and structured report decision entries.
    decisions: List[str] = []
    decision_entries: List[DecisionEntry] = []

    for key in ordered_keys:
        model_map = results_by_key.get(key, {})
        failures = failures_by_key.get(key, [])
        bic_sorted = sorted(model_map.values(), key=lambda r: r.bic)
        decisions.append(f"Dataset: {key}")
        if failures:
            decisions.append("Warnings:")
            for model_name, message in failures:
                decisions.append(
                    f"- excluded {display_model_name(model_name)} (optimizer did not converge: {message})"
                )
        if not bic_sorted:
            decisions.append("No successful model fits; see warnings.")
            decisions.append("")
            continue

        best = bic_sorted[0]
        best_display = display_model_name(best.model.name)
        decisions.append(
            f"Tentative working model among tested candidates: {best_display} (lowest BIC among candidates)"
        )
        reasons = [
            f"Within the tested model set, this model gave the lowest Bayesian Information Criterion "
            f"(BIC={best.bic:.6g}). This ranking indicates relative support only and should be interpreted with "
            "chemical plausibility and spectral behavior"
        ]
        if len(bic_sorted) > 1:
            delta_bic = float(bic_sorted[1].bic - best.bic)
            decisions.append(f"- delta BIC to next candidate: {delta_bic:.6g}")
            if np.isfinite(delta_bic) and delta_bic < 2.0:
                decisions.append("- BIC separation is small; treat model selection as provisional")
                reasons.append("BIC separation from the next candidate was small, so model discrimination is weak")
        if args.bootstrap_ci_width is not None and best.bootstrap is not None and best.model.n_logk > 0:
            k_samples = _bootstrap_k_samples(best)
            if k_samples.size > 0:
                k_ci_low = np.percentile(k_samples, 2.5, axis=0)
                k_ci_high = np.percentile(k_samples, 97.5, axis=0)
                if np.any((k_ci_high - k_ci_low) > args.bootstrap_ci_width):
                    decisions.append("- bootstrap CI too wide")
                    reasons.append("Bootstrap confidence interval width exceeds the specified threshold")
        decisions.append("")
        decision_entries.append(
            DecisionEntry(
                dataset=key,
                recommended_model=best_display,
                reasons=reasons,
            )
        )

    return decisions, decision_entries
