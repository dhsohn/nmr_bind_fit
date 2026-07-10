"""Report-building pipeline extracted from CLI orchestration."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from .fit_bootstrap import MIN_BOOTSTRAP_CI_SUCCESS_RATE, MIN_BOOTSTRAP_CI_SUCCESSES
from .models import ModelSpec, split_params_multi
from .plots import (
    plot_bootstrap_hist,
    plot_fraction_bound,
    plot_isotherms,
    plot_residuals,
)
from .report import DecisionEntry, ModelEntry, ParamEntry
from .types import DatasetLike, FitResultLike, SpeciesLike

SUMMARY_LABELS = {
    "dataset": "Dataset",
    "model": "Model",
    "status": "Status",
    "K": "Binding constant (M⁻¹)",
    "bootstrap_K_CI": "95 % CI",
    "R2": "R² (mean)",
    "BIC": "BIC",
    "AICc": "AICc",
    "notes": "Notes",
}

LOGK_BOUND_ATOL = 1e-7


STATS_LABELS = {
    "n": "Observations (n)",
    "p": "Fitted parameters (p)",
    "dof": "Residual degrees of freedom",
    "jacobian_rank": "Jacobian rank",
    "jacobian_condition": "Jacobian condition number",
    "R2": "Coefficient of determination (mean per-peak)",
    "R2_per_peak": "R² per peak",
    "RSS": "Residual sum of squares",
    "RMSE": "Root mean square error",
    "BIC": "Bayesian Information Criterion",
    "AICc": "Corrected Akaike Information Criterion",
    "penalty_events": "Optimization penalty events",
    "bootstrap_logK_SE": "Bootstrap SE (log10 K)",
    "bootstrap_success": "Successful bootstrap refits",
    "bootstrap_ci_method": "Bootstrap CI method used",
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


def _append_png_plot_paths(plot_paths: List[str], paths: Sequence[Path], out_dir: Path) -> None:
    for path in paths:
        if path.suffix.lower() == ".png":
            plot_paths.append(path.relative_to(out_dir).as_posix())


def _filter_finite_rows(values: np.ndarray) -> np.ndarray:
    # Drop rows with any non-finite values.
    if values.ndim == 1:
        return values[np.isfinite(values)]
    mask = np.all(np.isfinite(values), axis=1)
    return values[mask]


def _as_int(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _safe_path_token(value: str) -> str:
    # Sanitize and bound free-form labels before using them as path components.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    safe = safe or "dataset"
    if len(safe) <= 80:
        return safe
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    prefix = safe[:67].rstrip("._-") or "dataset"
    return f"{prefix}-{digest}"


def _dataset_dir_tokens(labels: Sequence[str]) -> Dict[str, str]:
    # Preserve distinct dataset directories even when labels sanitize to the same token.
    safe_labels = [_safe_path_token(label) for label in labels]
    counts = Counter(safe_labels)
    used: set[str] = set()
    tokens: Dict[str, str] = {}

    for idx, (label, safe) in enumerate(zip(labels, safe_labels), start=1):
        candidates = []
        if counts[safe] == 1:
            candidates.append(safe)
        candidates.append(f"{idx:02d}_{safe}")

        token = next((candidate for candidate in candidates if candidate not in used), "")
        if not token:
            suffix = 1
            token = f"{idx:02d}_{safe}_{suffix}"
            while token in used:
                suffix += 1
                token = f"{idx:02d}_{safe}_{suffix}"

        used.add(token)
        tokens[label] = token

    return tokens


def _replicate_dataset_dir_labels(datasets: Sequence[DatasetLike]) -> List[str]:
    # Build deterministic, collision-free directory labels per replicate dataset.
    labels: List[str] = []
    for idx, ds in enumerate(datasets, start=1):
        base = str(ds.name or f"dataset_{idx}")
        path = ds.path
        if path is not None:
            filename = Path(path).name
            if filename and filename != base:
                base = f"{base}_{filename}"
        labels.append(f"{idx:02d}_{_safe_path_token(base)}")
    return labels


def _format_dropped_peaks(datasets: Sequence[DatasetLike]) -> str:
    # Format dropped ppm columns for report warnings.
    items: List[str] = []
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


def _format_dropped_rows(datasets: Sequence[DatasetLike]) -> str:
    # Format concentration rows dropped before fitting for report warnings.
    items: List[str] = []
    multi = len(datasets) > 1
    for ds in datasets:
        dropped_rows = int(getattr(ds, "dropped_rows", 0))
        if dropped_rows > 0:
            if multi:
                items.append(f"{ds.name}: {dropped_rows}")
            else:
                items.append(str(dropped_rows))
    if not items:
        return "None"
    return "; ".join(items)


def _accumulate_solver_stats(species_list: List[SpeciesLike]) -> Optional[Dict[str, object]]:
    # Combine solver statistics across species lists.
    totals = {
        "solver_points": 0,
        "solver_success": 0,
        "solver_fail": 0,
        "solver_method": set(),
    }
    found = False
    for species in species_list:
        stats = species.solver_stats
        if stats is None:
            continue
        found = True
        totals["solver_points"] += int(stats.points)
        totals["solver_success"] += int(stats.success)
        totals["solver_fail"] += int(stats.fail)
        method = stats.method
        if method:
            totals["solver_method"].add(str(method))
    if not found:
        return None
    methods = totals["solver_method"]
    totals["solver_method"] = ", ".join(sorted(methods)) if methods else "N/A"
    return totals


def _build_param_entries(res: FitResultLike) -> List[ParamEntry]:
    # Convert fitted parameters into report entries and optional bootstrap SE.
    bootstrap_samples = None
    if (
        res.bootstrap is not None
        and getattr(res.bootstrap, "ci_valid", True)
        and res.bootstrap.param_samples.size > 0
    ):
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


def _collect_plot_artifacts(
    res: FitResultLike,
    model_name: str,
    ds_label: str,
    out_dir: Path,
    dataset_dir_token: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str]]:
    # Write model plots and return relative PNG paths plus display labels.
    model_dir = out_dir / f"model_{_safe_path_token(model_name)}"
    if ds_label != "Simultaneous Fitting":
        token = dataset_dir_token or _safe_path_token(ds_label)
        model_dir = model_dir / f"dataset_{token}"
    elif len(res.datasets) > 1:
        model_dir = model_dir / f"dataset_{_safe_path_token(ds_label)}"
    model_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: List[str] = []
    plot_labels: Dict[str, str] = {}
    replicate_dir_labels = _replicate_dataset_dir_labels(res.datasets) if len(res.datasets) > 1 else []
    model_spec = cast(ModelSpec, res.model)
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
        _append_png_plot_paths(plot_paths, boot_files, out_dir)

    return plot_paths, plot_labels


def _collect_plot_paths(
    res: FitResultLike,
    model_name: str,
    ds_label: str,
    out_dir: Path,
    dataset_dir_token: Optional[str] = None,
) -> List[str]:
    """Compatibility wrapper returning only relative PNG paths."""
    plot_paths, _ = _collect_plot_artifacts(
        res,
        model_name,
        ds_label,
        out_dir,
        dataset_dir_token,
    )
    return plot_paths


def _bootstrap_k_samples(res: FitResultLike) -> np.ndarray:
    # Return finite bootstrap K samples (linear scale) for warnings and summary.
    if res.bootstrap is None or res.model.n_logk == 0 or res.bootstrap.param_samples.size == 0:
        return np.full((0, res.model.n_logk), np.nan)
    k_samples = _safe_pow10(res.bootstrap.param_samples[:, : res.model.n_logk])
    return _filter_finite_rows(k_samples)


def _bootstrap_logk_samples(res: FitResultLike) -> np.ndarray:
    # Return finite bootstrap logK samples for SE reporting.
    if (
        res.bootstrap is None
        or res.model.n_logk == 0
        or not getattr(res.bootstrap, "ci_valid", True)
    ):
        return np.full((0, res.model.n_logk), np.nan)
    samples = getattr(res.bootstrap, "logk_samples", np.full((0, res.model.n_logk), np.nan))
    if samples.size == 0:
        return np.full((0, res.model.n_logk), np.nan)
    return _filter_finite_rows(samples)


def _bootstrap_logk_se_text(res: FitResultLike) -> str:
    # Format bootstrap SE in log10(K) space.
    samples = _bootstrap_logk_samples(res)
    if samples.shape[0] <= 1:
        return "N/A"
    se_vals = [_safe_std(samples[:, i]) for i in range(samples.shape[1])]
    return ";".join(f"{value:.6g}" for value in se_vals)


def _bootstrap_k_ci(res: FitResultLike) -> Tuple[np.ndarray, np.ndarray]:
    # Return selected bootstrap CI in linear K space.
    if (
        res.bootstrap is None
        or res.model.n_logk == 0
        or not getattr(res.bootstrap, "ci_valid", True)
    ):
        return np.full((res.model.n_logk,), np.nan), np.full((res.model.n_logk,), np.nan)
    low_log = np.asarray(
        getattr(res.bootstrap, "ci_low", np.full((res.model.n_logk,), np.nan)),
        dtype=float,
    )
    high_log = np.asarray(
        getattr(res.bootstrap, "ci_high", np.full((res.model.n_logk,), np.nan)),
        dtype=float,
    )
    if low_log.shape != (res.model.n_logk,) or high_log.shape != (res.model.n_logk,):
        return np.full((res.model.n_logk,), np.nan), np.full((res.model.n_logk,), np.nan)
    return _safe_pow10(low_log), _safe_pow10(high_log)


def _logk_names(n_logk: int) -> List[str]:
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


def _logk_bound_warnings(res: FitResultLike) -> List[str]:
    if res.model.n_logk == 0 or not hasattr(res, "params"):
        return []
    # Compare against the bounds the fit actually used; when they are unknown
    # (e.g. an unbounded programmatic fit) no bound was active, so pinning a
    # valid estimate such as K=1 or K=1e12 would be misleading.
    bounds = getattr(res, "logk_bounds", None)
    if bounds is None:
        return []
    low, high = float(bounds[0]), float(bounds[1])
    logk_vals = np.asarray(res.params[: res.model.n_logk], dtype=float)
    names = _logk_names(res.model.n_logk)
    warnings: List[str] = []
    for name, value in zip(names, logk_vals):
        if not np.isfinite(value):
            continue
        if np.isclose(value, low, atol=LOGK_BOUND_ATOL, rtol=0.0):
            warnings.append(f"{name} is pinned at the lower log10(K) bound ({low:g})")
        elif np.isclose(value, high, atol=LOGK_BOUND_ATOL, rtol=0.0):
            warnings.append(f"{name} is pinned at the upper log10(K) bound ({high:g})")
    return warnings


def _solver_stats_for_result(res: FitResultLike) -> Optional[Dict[str, object]]:
    # Collect solver diagnostics only for nonlinear root-solved models.
    if res.model.name not in {"12", "21"}:
        return None
    if not hasattr(res, "species"):
        return None
    solver_stats = _accumulate_solver_stats(res.species)
    if solver_stats is None:
        return {
            "solver_points": "N/A",
            "solver_success": "N/A",
            "solver_fail": "N/A",
            "solver_method": "N/A",
        }
    return solver_stats


def _build_model_warnings(
    args: argparse.Namespace,
    res: FitResultLike,
    solver_stats: Optional[Dict[str, object]],
) -> List[str]:
    # Build per-model warning messages for report rendering.
    warnings = []
    datasets = getattr(res, "datasets", [])

    dropped_peaks = _format_dropped_peaks(datasets)
    if dropped_peaks != "None":
        warnings.append(f"Dropped chemical shift columns with missing or non-finite values: {dropped_peaks}")

    dropped_rows = _format_dropped_rows(datasets)
    if dropped_rows != "None":
        warnings.append(f"Dropped rows with missing required concentrations: {dropped_rows}")

    if not np.isfinite(res.bic):
        if getattr(res, "n", 0) <= getattr(res, "p", 0) + 1:
            warnings.append(
                "BIC/AICc unavailable: finite observations are not greater than fitted parameters plus variance"
            )
        elif np.isfinite(getattr(res, "rss", np.nan)) and res.rss <= 0:
            warnings.append("BIC/AICc unavailable: residual variance is zero")
        else:
            warnings.append("BIC/AICc unavailable for this fit")
    elif not np.isfinite(getattr(res, "aicc", float("nan"))):
        warnings.append(
            "AICc unavailable: too few observations for the small-sample correction; BIC is still reported"
        )

    warnings.extend(_logk_bound_warnings(res))

    k_ci_low, k_ci_high = _bootstrap_k_ci(res)
    if args.bootstrap_ci_width is not None and k_ci_low.size > 0:
        if np.any(np.isfinite(k_ci_low) & np.isfinite(k_ci_high) & ((k_ci_high - k_ci_low) > args.bootstrap_ci_width)):
            warnings.append("bootstrap CI too wide")

    if res.bootstrap is not None:
        n_boot = int(getattr(res.bootstrap, "n_boot", 0))
        n_success = int(getattr(res.bootstrap, "n_success", 0))
        if n_boot > 0:
            n_fail = n_boot - n_success
            if n_fail > 0:
                warnings.append(f"bootstrap failures: {n_fail} of {n_boot} iterations")
        if not getattr(res.bootstrap, "ci_valid", True):
            ci_message = str(getattr(res.bootstrap, "ci_message", "")).strip()
            warnings.append(ci_message or "bootstrap confidence interval is unavailable")

    penalty_count = int(getattr(res, "penalty_count", 0))
    if penalty_count > 0:
        warnings.append(f"optimization penalty residual events: {penalty_count}")

    if solver_stats is not None and solver_stats.get("solver_fail", 0) not in {"N/A", None}:
        n_fail = _as_int(solver_stats.get("solver_fail", 0))
        n_points = _as_int(solver_stats.get("solver_points", 0))
        if n_fail > 0 and n_points > 0:
            warnings.append(f"solver failures ({n_fail}/{n_points})")

    for ds, species in zip(datasets, getattr(res, "species", [])):
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


def _build_stats_dict(res: FitResultLike, solver_stats: Optional[Dict[str, object]]) -> Dict[str, str]:
    # Build stats block for report tables.
    r2_per_peak = getattr(res, "r2_per_peak", [])
    r2_per_peak_text = (
        ";".join(f"{value:.6g}" for value in r2_per_peak)
        if r2_per_peak
        else "N/A"
    )
    stats_base = {
        "R2": f"{res.r2:.6g}" if np.isfinite(res.r2) else "N/A",
        "R2_per_peak": r2_per_peak_text,
        "bootstrap_logK_SE": _bootstrap_logk_se_text(res),
        "RSS": f"{res.rss:.6g}",
        "RMSE": f"{res.rmse:.6g}",
        "BIC": f"{res.bic:.6g}" if np.isfinite(res.bic) else "N/A",
        "AICc": f"{res.aicc:.6g}" if np.isfinite(res.aicc) else "N/A",
        "penalty_events": str(res.penalty_count),
    }
    for attr in ("n", "p", "dof", "jacobian_rank"):
        value = getattr(res, attr, None)
        if value is not None:
            stats_base[attr] = str(value)
    jacobian_condition = getattr(res, "jacobian_condition", None)
    if jacobian_condition is not None:
        stats_base["jacobian_condition"] = f"{float(jacobian_condition):.6g}"
    if res.bootstrap is not None:
        n_success = getattr(res.bootstrap, "n_success", None)
        n_boot = getattr(res.bootstrap, "n_boot", None)
        if n_success is not None and n_boot is not None:
            stats_base["bootstrap_success"] = f"{n_success} / {n_boot}"
        ci_method_used = getattr(res.bootstrap, "ci_method_used", None)
        if ci_method_used:
            stats_base["bootstrap_ci_method"] = str(ci_method_used)
    if solver_stats is not None:
        stats_base.update(
            {
                "solver_points": str(solver_stats["solver_points"]),
                "solver_success": str(solver_stats["solver_success"]),
                "solver_fail": str(solver_stats["solver_fail"]),
                "solver_method": str(solver_stats["solver_method"]),
            }
        )
    diagnostics = getattr(res, "residual_diagnostics", {})
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
    res: FitResultLike,
    ds_label: str,
    display_name: str,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    # Build one row for summary.csv.
    logk_vals = res.params[: res.model.n_logk]
    k_vals = _safe_pow10(logk_vals)
    k_str = _format_k_values(k_vals, res.model.n_logk)

    k_ci_low, k_ci_high = _bootstrap_k_ci(res)
    boot_k_ci = _format_k_ci(k_ci_low, k_ci_high, res.model.n_logk)

    summary_base = {
        "dataset": ds_label,
        "model": display_name,
        "status": "success",
        "K": k_str,
        "bootstrap_K_CI": boot_k_ci,
        "R2": f"{res.r2:.6g}" if np.isfinite(res.r2) else "N/A",
        "BIC": f"{res.bic:.6g}" if np.isfinite(res.bic) else "N/A",
        "AICc": f"{res.aicc:.6g}" if np.isfinite(res.aicc) else "N/A",
        "notes": "; ".join(str(warning) for warning in (warnings or [])),
    }
    return {_label_summary_key(k): v for k, v in summary_base.items()}


def _build_failure_summary_row(ds_label: str, display_name: str, message: str) -> Dict[str, str]:
    # Keep failed candidates visible in summary.csv for auditability.
    summary_base = {
        "dataset": ds_label,
        "model": display_name,
        "status": "failed",
        "K": "N/A",
        "bootstrap_K_CI": "N/A",
        "R2": "N/A",
        "BIC": "N/A",
        "AICc": "N/A",
        "notes": f"fit failed: {message}",
    }
    return {_label_summary_key(k): v for k, v in summary_base.items()}


def _build_model_entry(
    args: argparse.Namespace,
    key: str,
    model_name: str,
    res: FitResultLike,
    out_dir: Path,
    display_model_name: Callable[[str], str],
    dataset_dir_token: Optional[str] = None,
) -> Tuple[ModelEntry, Dict[str, str]]:
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
    summary_row = _build_summary_row(res, key, display_name, warnings)
    return model_entry, summary_row


def build_report_artifacts(
    args: argparse.Namespace,
    ordered_keys: List[str],
    results_by_key: Dict[str, Dict[str, FitResultLike]],
    failures_by_key: Dict[str, List[Tuple[str, str]]],
    out_dir: Path,
    display_model_name: Callable[[str], str],
) -> Tuple[List[Dict[str, str]], List[ModelEntry], List[str]]:
    # Convert fit results into summary rows, model entries, and top-level warnings.
    summary_rows: List[Dict[str, str]] = []
    model_entries: List[ModelEntry] = []
    report_warnings: List[str] = []
    dataset_dir_tokens = _dataset_dir_tokens(ordered_keys)

    for key in ordered_keys:
        model_map = cast(Dict[str, FitResultLike], results_by_key.get(key, {}))
        failures = failures_by_key.get(key, [])
        for model_name, message in failures:
            report_warnings.append(
                f"{key}: excluded {display_model_name(model_name)} (fit failed: {message})"
            )
            summary_rows.append(_build_failure_summary_row(key, display_model_name(model_name), message))
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
                dataset_dir_tokens.get(key),
            )
            model_entries.append(model_entry)
            summary_rows.append(summary_row)

    return summary_rows, model_entries, report_warnings


def _compose_methods_sections(args: argparse.Namespace, datasets: Sequence[DatasetLike]) -> List[Dict[str, str]]:
    # Shared source for both structured methods sections and plain-text methods summary.
    sections: List[Dict[str, str]] = []
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
                "rtol = 8 machine epsilons, maxiter = 200). "
                "Binding constants were parameterized as log₁₀(K) and constrained to [0, 12] "
                f"(K ∈ [1, 10¹²] {k_unit}) during optimization to ensure stable, physically meaningful estimation. "
                "Fits were excluded from model comparison unless they had positive residual degrees of freedom, "
                "a full-column-rank Jacobian with condition number at most 10⁶, and no active log₁₀(K) bound. "
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
    method_desc = {
        "residual": (
            "Residual resampling: fitted residuals were mean-centered per peak column and "
            "resampled with replacement across titration points, then added back to the "
            "fitted values to generate pseudo-datasets."
        ),
        "parametric": (
            "Parametric resampling: Gaussian noise was generated independently for each "
            "peak column using the column-wise residual standard deviation (ddof = 1), then "
            "added to the fitted values. This per-column variance estimate allows for modest "
            "heteroscedasticity across peaks while the model-comparison criteria (BIC/AICc) "
            "use a single pooled variance; the two serve different inferential purposes "
            "(uncertainty quantification vs. model ranking) and are therefore not inconsistent."
        ),
        "points": (
            "Case (points) resampling: entire titration-point rows (concentrations and "
            "observed shifts) were resampled with replacement, preserving the covariate "
            "structure within each row."
        ),
    }
    bootstrap_ci_method = str(getattr(args, "bootstrap_ci_method", "percentile")).strip().lower()
    if bootstrap_ci_method == "bca":
        ci_desc = (
            "The 95% confidence interval was derived from BCa-style adjusted bootstrap quantiles for the "
            "local warm-start refit estimator, with the acceleration term estimated from leave-one-out "
            "jackknife fits initialized at the full-data optimum. This avoids claiming a full multistart "
            "BCa estimator for potentially multimodal objectives."
        )
    else:
        ci_desc = (
            "The 95% confidence interval was derived from the 2.5th and 97.5th percentiles of the "
            "bootstrap distribution."
        )
    bootstrap_note = (
        f"Bootstrap uncertainty was evaluated using {args.bootstrap} iterations with "
        f"{args.bootstrap_method} resampling. "
        f"{method_desc.get(args.bootstrap_method, '')} "
        f"Bootstrap refits used a small log₁₀(K) start perturbation "
        f"(σ = {args.bootstrap_logk_jitter:.3g}) to explore the objective surface near the optimum. "
        f"A confidence interval was reported only when at least max({MIN_BOOTSTRAP_CI_SUCCESSES}, "
        f"{MIN_BOOTSTRAP_CI_SUCCESS_RATE:.0%} of requested refits) fits succeeded; otherwise the interval "
        "and bootstrap-derived standard errors were reported as unavailable. "
        f"{ci_desc}"
        if args.bootstrap > 0
        else "Bootstrap uncertainty was not evaluated in this analysis."
    )
    sections.append(
        {
            "title": "Uncertainty Quantification",
            "content": bootstrap_note,
        }
    )

    return sections


def build_methods_text(args: argparse.Namespace, datasets: Sequence[DatasetLike]) -> str:
    # Flatten structured methods sections into one text block.
    sections = _compose_methods_sections(args, datasets)
    return " ".join(section["content"] for section in sections)


def build_methods_sections(
    args: argparse.Namespace, datasets: Sequence[DatasetLike]
) -> List[Dict[str, str]]:
    """Return structured methods as list of {title, content} dicts for detailed HTML reporting."""
    return _compose_methods_sections(args, datasets)


def build_decisions(
    args: argparse.Namespace,
    ordered_keys: List[str],
    results_by_key: Dict[str, Dict[str, FitResultLike]],
    failures_by_key: Dict[str, List[Tuple[str, str]]],
    display_model_name: Callable[[str], str],
) -> Tuple[List[str], List[DecisionEntry]]:
    # Build decision.txt lines and structured report decision entries.
    decisions: List[str] = []
    decision_entries: List[DecisionEntry] = []

    for key in ordered_keys:
        model_map = cast(Dict[str, FitResultLike], results_by_key.get(key, {}))
        failures = failures_by_key.get(key, [])
        bic_sorted = sorted((res for res in model_map.values() if np.isfinite(res.bic)), key=lambda r: r.bic)
        decisions.append(f"Dataset: {key}")
        model_warning_lines: List[str] = []
        for model_name, res in model_map.items():
            model_warnings = _build_model_warnings(args, res, _solver_stats_for_result(res))
            for warning in model_warnings:
                model_warning_lines.append(f"- {display_model_name(model_name)}: {warning}")
        if failures or model_warning_lines:
            decisions.append("Warnings:")
            for model_name, message in failures:
                decisions.append(
                    f"- excluded {display_model_name(model_name)} (fit failed: {message})"
                )
            decisions.extend(model_warning_lines)
        if not bic_sorted:
            if model_map:
                decisions.append("No model had a finite BIC for ranking; see warnings.")
            else:
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
        if best.bootstrap is not None and not getattr(best.bootstrap, "ci_valid", True):
            ci_message = str(getattr(best.bootstrap, "ci_message", "")).strip()
            warning = ci_message or "Bootstrap uncertainty is unavailable for the selected model."
            decisions.append(f"- {warning}")
            reasons.append(warning)
        if args.bootstrap_ci_width is not None and best.bootstrap is not None and best.model.n_logk > 0:
            k_ci_low, k_ci_high = _bootstrap_k_ci(best)
            if k_ci_low.size > 0 and np.any(
                np.isfinite(k_ci_low)
                & np.isfinite(k_ci_high)
                & ((k_ci_high - k_ci_low) > args.bootstrap_ci_width)
            ):
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
