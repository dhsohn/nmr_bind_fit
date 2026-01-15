"""CLI entry point for NMR binding fits."""

from __future__ import annotations

import argparse
import glob
from datetime import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .fit import fit_models
from .io import load_datasets
from .models import split_params_multi
from .plots import plot_bootstrap_hist, plot_fraction_bound, plot_isotherms, plot_residuals
from .report import DecisionEntry, ModelEntry, ParamEntry, write_decision_txt, write_report_html, write_summary_csv
from .stats import f_test


MODEL_LABELS = {
    "11": "H : G = 1 : 1",
    "12": "H : G = 1 : 2",
    "21": "H : G = 2 : 1",
    "nb": "non-binding",
}

DEFAULT_MODEL_NAMES = ["11", "12", "21", "nb"]


def _parse_k_starts(value: Optional[str]) -> List[float]:
    if not value:
        return [10**i for i in range(1, 9)]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _resolve_inputs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            path = Path(pattern)
            if path.exists():
                matches = [pattern]
        for match in matches:
            paths.append(Path(match))
    if not paths:
        raise FileNotFoundError("No input files found.")
    return paths


def _safe_output_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return safe or "output"


def _auto_output_dir(paths: List[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(paths) == 1:
        base = paths[0].stem
    else:
        base = "replicates"
    return Path(f"{timestamp}_{_safe_output_name(base)}")


def _dataset_key(result) -> str:
    if len(result.datasets) == 1:
        return result.datasets[0].name
    return "Simultaneous Fitting"


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


def _format_ftest_value(value: object) -> str:
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f"{float(value):.6g}"
    if value is None:
        return "Not available"
    return str(value)


def _safe_pow10(values: np.ndarray) -> np.ndarray:
    log10_max = np.log(np.finfo(float).max) / np.log(10.0)
    log10_min = np.log(np.finfo(float).tiny) / np.log(10.0)
    clipped = np.clip(values, log10_min, log10_max)
    return np.exp(clipped * np.log(10.0))


def _safe_std(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size <= 1:
        return float("nan")
    scale = np.max(np.abs(vals))
    if not np.isfinite(scale) or scale == 0:
        return 0.0
    scaled = vals / scale
    return float(np.std(scaled, ddof=1) * scale)


def _filter_finite_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[np.isfinite(values)]
    mask = np.all(np.isfinite(values), axis=1)
    return values[mask]


def _display_model_name(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _accumulate_solver_stats(species_list: List[object]) -> Optional[Dict[str, object]]:
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


def run_fit(args: argparse.Namespace) -> None:
    paths = _resolve_inputs(args.input)
    datasets = load_datasets(
        paths,
        host_col=args.host_col,
        guest_col=args.guest_col,
        ppm_cols=args.ppm_cols,
        sigma_col=args.sigma_col,
        xaxis=args.xaxis,
    )

    model_names = DEFAULT_MODEL_NAMES

    k_starts = _parse_k_starts(args.k_starts)
    if any(v <= 0 for v in k_starts):
        raise ValueError("All K starts must be positive.")
    logk_starts = [float(np.log10(v)) for v in k_starts]
    logk_bounds = None
    if args.k_min is not None or args.k_max is not None:
        if args.k_min is None or args.k_max is None:
            raise ValueError("Both --k-min and --k-max must be provided.")
        if args.k_min <= 0 or args.k_max <= 0:
            raise ValueError("K bounds must be positive.")
        if args.k_min > args.k_max:
            raise ValueError("--k-min must be less than or equal to --k-max.")
        logk_bounds = (float(np.log10(args.k_min)), float(np.log10(args.k_max)))

    results = fit_models(
        datasets,
        model_names,
        logk_starts=logk_starts,
        global_replicates=args.global_replicates,
        max_nfev=args.max_nfev,
        bootstrap=args.bootstrap,
        bootstrap_method=args.bootstrap_method,
        seed=args.seed,
        logk_bounds=logk_bounds,
    )

    out_dir = _auto_output_dir(paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index results
    results_by_key: Dict[str, Dict[str, object]] = {}
    for res in results:
        key = _dataset_key(res)
        results_by_key.setdefault(key, {})[res.model.name] = res

    # Best model per dataset for conditional F test
    best_by_key = {
        key: min(model_map.values(), key=lambda r: r.bic) for key, model_map in results_by_key.items()
    }

    # F-test (nb vs 1:1) only when 1:1 is the selected model
    ftest_by_key: Dict[str, dict] = {}
    for key, model_map in results_by_key.items():
        if best_by_key[key].model.name != "11":
            continue
        entry = {"label": "non-binding vs 1:1"}
        if "nb" in model_map and "11" in model_map:
            res_nb = model_map["nb"]
            res_11 = model_map["11"]
            if res_nb.p <= res_11.p:
                res_small, res_large = res_nb, res_11
            else:
                res_small, res_large = res_11, res_nb
            ftest = f_test(res_small.rss_weighted, res_large.rss_weighted, res_small.p, res_large.p, res_large.n)
            if ftest:
                entry.update(stat=ftest.f_stat, p=ftest.p_value, crit=ftest.f_crit, note="Nested")
            else:
                entry.update(
                    stat="Not available (insufficient data)",
                    p="Not available (insufficient data)",
                    crit="Not available (insufficient data)",
                    note="Nested",
                )
        else:
            entry.update(
                stat="Not available (model missing)",
                p="Not available (model missing)",
                crit="Not available (model missing)",
                note="Model missing",
            )
        ftest_by_key[key] = entry

    summary_rows: List[Dict[str, str]] = []
    model_entries: List[ModelEntry] = []

    for key, model_map in results_by_key.items():
        for model_name, res in model_map.items():
            ds_names = [ds.name for ds in res.datasets]
            ds_label = key

            # Build param summaries
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
                    k_value = float(_safe_pow10(np.array(value)))
                    k_se = se
                    k_name = name.replace("logK", "K")
                    params.append(
                        ParamEntry(
                            name=k_name,
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

            # Plot outputs
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
                isotherm_files = plot_isotherms(res.model, ds, logk, delta, args.xaxis, ds_dir)
                residual_files = plot_residuals(res.model, ds, residual, args.xaxis, ds_dir)
                frac_files = plot_fraction_bound(res.model, ds, logk, delta, args.xaxis, ds_dir)
                for path in isotherm_files + residual_files + frac_files:
                    if path.suffix.lower() == ".png":
                        plot_paths.append(str(path.relative_to(out_dir)))

            # correlation matrix (bootstrap-based)
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

            # bootstrap hist
            if res.bootstrap is not None and res.model.n_logk > 0:
                k_names = ["K"] if res.model.n_logk == 1 else ["K1", "K2"]
                k_samples = _safe_pow10(res.bootstrap.param_samples[:, : res.model.n_logk])
                boot_files = plot_bootstrap_hist(k_samples, k_names, model_dir)
                for path in boot_files:
                    if path.suffix.lower() == ".png":
                        plot_paths.append(str(path.relative_to(out_dir)))

            warnings = []
            if args.bootstrap_ci_width is not None and res.bootstrap is not None and res.model.n_logk > 0:
                if res.bootstrap.param_samples.size > 0:
                    k_samples = _safe_pow10(res.bootstrap.param_samples[:, : res.model.n_logk])
                    k_samples = _filter_finite_rows(k_samples)
                    if k_samples.size > 0:
                        k_ci_low = np.percentile(k_samples, 2.5, axis=0)
                        k_ci_high = np.percentile(k_samples, 97.5, axis=0)
                        width = k_ci_high - k_ci_low
                        if np.any(width > args.bootstrap_ci_width):
                            warnings.append("bootstrap CI too wide")
            if res.bootstrap is not None and res.bootstrap.n_boot > 0:
                n_fail = res.bootstrap.n_boot - res.bootstrap.n_success
                if n_fail > 0:
                    warnings.append(
                        f"bootstrap failures: {n_fail} of {res.bootstrap.n_boot} iterations"
                    )

            solver_stats = None
            if res.model.name in {"12", "21"}:
                solver_stats = _accumulate_solver_stats(res.species)
                if solver_stats is None:
                    solver_stats = {
                        "solver_points": "N/A",
                        "solver_newton_success": "N/A",
                        "solver_newton_fail": "N/A",
                        "solver_newton_max_iter": "N/A",
                        "solver_fallback_success": "N/A",
                        "solver_fallback_fail": "N/A",
                        "solver_fallback_method": "N/A",
                    }
                elif solver_stats["solver_fallback_fail"] > 0:
                    warnings.append(
                        f"solver fallback failures ({solver_stats['solver_fallback_fail']}/{solver_stats['solver_points']})"
                    )

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
            stats_dict = {_label_stats_key(k): v for k, v in stats_base.items()}

            display_name = _display_model_name(model_name)
            model_entries.append(
                ModelEntry(
                    dataset=ds_label,
                    model=display_name,
                    stats=stats_dict,
                    params=params,
                    plots=plot_paths,
                    warnings=warnings,
                )
            )

            # Summary row
            logk_vals = res.params[: res.model.n_logk]
            k_vals = _safe_pow10(logk_vals)
            k_str = ";".join(f"{v:.6g}" for v in k_vals) if res.model.n_logk else "N/A"

            if res.bootstrap is not None and res.bootstrap.param_samples.size > 0 and res.model.n_logk > 0:
                k_samples = _safe_pow10(res.bootstrap.param_samples[:, : res.model.n_logk])
                k_samples = _filter_finite_rows(k_samples)
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
            else:
                boot_k_ci = "N/A"
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
            summary_rows.append({_label_summary_key(k): v for k, v in summary_base.items()})

    write_summary_csv(summary_rows, out_dir / "summary.csv")

    # Methods summary text for report
    bootstrap_note = (
        f"Bootstrap confidence intervals were estimated with {args.bootstrap} iterations using "
        f"{args.bootstrap_method} resampling."
        if args.bootstrap > 0
        else "Bootstrap confidence intervals were not computed."
    )
    replicate_note = ""
    if args.global_replicates and len(datasets) > 1:
        replicate_note = (
            "Replicate datasets were fit simultaneously with shared binding constants and "
            "replicate-specific chemical shifts. "
        )
    methods_text = (
        "Chemical shift data were fit under a fast-exchange assumption, with observed shifts modeled as "
        "population-weighted averages of species. The 1:1 model was solved analytically, while 1:2 and 2:1 "
        "models were solved per titration point using Newton-Raphson on free guest concentration with mass "
        "balance constraints, with bisection fallback when Newton-Raphson did not converge; solver "
        "success and failure counts were recorded. Nonlinear least squares fitting used binding constants and multi-start "
        "initialization; for multi-peak data, binding constants were shared across peaks with peak-specific "
        "chemical shifts. "
        + replicate_note
        + "Uncertainty was quantified using bootstrap resampling, and parameter standard "
        "errors are reported from the bootstrap distributions. Residual bootstrap uses "
        "sigma-standardized residuals when sigma is provided, while parametric bootstrap "
        "draws Gaussian noise scaled by sigma. Bayesian Information Criterion(BIC) is computed from the Gaussian "
        "log-likelihood (including sigma terms when provided); when sigma is not provided, variance "
        "is estimated per peak and counted as additional parameters. Corrected Akaike Information Criterion (AICc) "
        "is reported as a supplementary metric for small-sample validation. Model selection prioritized BIC. "
        "When the 1:1 model is selected, "
        "a nested-model F test versus the non-binding model is reported. "
        + bootstrap_note
    )

    decision_entries: List[DecisionEntry] = []

    # Decision text
    decisions: List[str] = []
    for key, model_map in results_by_key.items():
        bic_sorted = sorted(model_map.values(), key=lambda r: r.bic)
        best = bic_sorted[0]
        best_name = best.model.name
        best_display = _display_model_name(best_name)
        decisions.append(f"Dataset: {key}")
        decisions.append(f"Recommended model: {best_display} (lowest BIC)")
        reasons: List[str] = []
        reasons.append(
            f"The selected model has the lowest Bayesian Information Criterion (BIC={best.bic:.6g}); "
            "BIC penalizes model complexity, so lower values indicate a better balance of fit and parsimony"
        )
        if len(bic_sorted) > 1:
            decisions.append(f"- next best BIC: {bic_sorted[1].bic:.6g}")
        if args.bootstrap_ci_width is not None and best.bootstrap is not None and best.model.n_logk > 0:
            if best.bootstrap.param_samples.size > 0:
                k_samples = _safe_pow10(best.bootstrap.param_samples[:, : best.model.n_logk])
                k_ci_low = np.percentile(k_samples, 2.5, axis=0)
                k_ci_high = np.percentile(k_samples, 97.5, axis=0)
                width = k_ci_high - k_ci_low
                if np.any(width > args.bootstrap_ci_width):
                    decisions.append("- bootstrap CI too wide")
                    reasons.append("Bootstrap confidence interval width exceeds the specified threshold")
        ftest_entry = ftest_by_key.get(key, {})
        if isinstance(ftest_entry.get("stat"), (int, float, np.floating)):
            f_line = (
                f"F test non-binding vs 1:1: F={_format_ftest_value(ftest_entry.get('stat'))}, "
                f"p={_format_ftest_value(ftest_entry.get('p'))}, "
                f"Fcrit={_format_ftest_value(ftest_entry.get('crit'))}."
            )
            decisions.append(f"- {f_line}")
            f_support = ""
            if isinstance(ftest_entry.get("p"), (int, float, np.floating)):
                if float(ftest_entry.get("p")) < 0.05:
                    f_support = "This supports the binding model over the non-binding model at alpha=0.05"
                else:
                    f_support = "This does not support a statistically significant improvement for the binding model at alpha=0.05"
            reasons.append(
                "The F test compares the nested non-binding and 1:1 binding models; "
                f"{f_line} {f_support}".strip()
            )
        decisions.append("")
        decision_entries.append(
            DecisionEntry(
                dataset=key,
                recommended_model=best_display,
                reasons=reasons,
            )
        )

    write_decision_txt(decisions, out_dir / "decision.txt")

    write_report_html(
        summary_rows,
        model_entries,
        decision_entries=decision_entries,
        methods_text=methods_text,
        output_path=out_dir / "report.html",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nmrbindfit")
    sub = parser.add_subparsers(dest="command", required=True)

    fit_p = sub.add_parser("fit", help="Fit binding models")
    fit_p.add_argument("--input", nargs="+", required=True, help="Input CSV/XLSX files")
    fit_p.add_argument("--xaxis", choices=["eq", "guest"], default="eq", help="X axis for plots")
    fit_p.add_argument("--host-col", default=None, help="Host concentration column")
    fit_p.add_argument("--guest-col", default=None, help="Guest concentration column")
    fit_p.add_argument("--ppm-cols", default=None, help="Comma-separated ppm columns")
    fit_p.add_argument("--sigma-col", default=None, help="Sigma column for weighting")
    fit_p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")
    fit_p.add_argument("--bootstrap-method", choices=["residual", "points", "parametric"], default="residual")
    fit_p.add_argument("--k-starts", default=None, help="Comma-separated K starts")
    fit_p.add_argument("--k-min", type=float, default=None, help="Minimum K bound")
    fit_p.add_argument("--k-max", type=float, default=None, help="Maximum K bound")
    fit_p.add_argument(
        "--global-replicates",
        "--replicates",
        dest="global_replicates",
        action="store_true",
        help="Fit replicate inputs with shared binding constants",
    )
    fit_p.add_argument("--max-nfev", type=int, default=5000, help="Max optimizer evaluations")
    fit_p.add_argument("--seed", type=int, default=None, help="Random seed")
    fit_p.add_argument(
        "--bootstrap-ci-width",
        type=float,
        default=None,
        help="Warn if bootstrap K CI width exceeds this threshold",
    )
    fit_p.set_defaults(func=run_fit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
