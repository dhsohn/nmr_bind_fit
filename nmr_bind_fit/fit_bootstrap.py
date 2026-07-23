"""Bootstrap resampling for binding-constant uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Type

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.stats import chi2, norm

from .fit_optimizer import optimizer_result_is_identifiable
from .io import Dataset
from .models import ModelSpec

MIN_BOOTSTRAP_CI_SUCCESSES = 20
MIN_BOOTSTRAP_CI_SUCCESS_RATE = 1.0
BOOTSTRAP_COMPETITIVE_CONFIDENCE = 0.95


@dataclass
class BootstrapResult:
    param_samples: np.ndarray
    logk_samples: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_low_percentile: np.ndarray
    ci_high_percentile: np.ndarray
    ci_low_bca: np.ndarray
    ci_high_bca: np.ndarray
    ci_method: str
    ci_method_used: str
    ci_valid: bool
    ci_message: str
    n_success: int
    n_boot: int


def _bca_ci(
    samples: np.ndarray,
    original_stat: float,
    jackknife_stats: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Return a BCa interval for the local warm-start refit estimator."""
    vals = np.asarray(samples, dtype=float)
    vals = vals[np.isfinite(vals)]
    jackknife = np.asarray(jackknife_stats, dtype=float)
    jackknife = jackknife[np.isfinite(jackknife)]
    if vals.size < 3 or jackknife.size < 3 or not np.isfinite(original_stat):
        return float("nan"), float("nan")

    n = int(vals.size)
    prop_less = float(np.mean(vals < original_stat))
    # Avoid ±inf in inverse-normal transform.
    eps = 0.5 / n
    prop_less = float(np.clip(prop_less, eps, 1.0 - eps))
    z0 = float(norm.ppf(prop_less))

    jack_mean = float(np.mean(jackknife))
    diffs = jack_mean - jackknife
    denom = float(np.sum(diffs**2))
    if denom <= 0.0 or not np.isfinite(denom):
        a = 0.0
    else:
        num = float(np.sum(diffs**3))
        a = num / (6.0 * (denom ** 1.5))
        if not np.isfinite(a):
            a = 0.0

    z_low = float(norm.ppf(alpha / 2.0))
    z_high = float(norm.ppf(1.0 - alpha / 2.0))

    def _adjusted_prob(z_alpha: float) -> float:
        z_term = z0 + z_alpha
        denom_term = 1.0 - a * z_term
        if denom_term == 0.0:
            return float("nan")
        p = float(norm.cdf(z0 + (z_term / denom_term)))
        return float(np.clip(p, 0.0, 1.0))

    p_low = _adjusted_prob(z_low)
    p_high = _adjusted_prob(z_high)
    if not np.isfinite(p_low) or not np.isfinite(p_high):
        return float("nan"), float("nan")
    return (
        float(np.percentile(vals, 100.0 * p_low)),
        float(np.percentile(vals, 100.0 * p_high)),
    )


def _bootstrap_ci_requirement(n_boot: int) -> int:
    return max(MIN_BOOTSTRAP_CI_SUCCESSES, int(np.ceil(MIN_BOOTSTRAP_CI_SUCCESS_RATE * n_boot)))


def _bootstrap_ci_status(n_success: int, n_boot: int) -> Tuple[bool, str]:
    required = _bootstrap_ci_requirement(n_boot)
    if n_success < required:
        if n_boot >= MIN_BOOTSTRAP_CI_SUCCESSES and n_success < n_boot:
            requirement = (
                f"all {n_boot} requested refits must succeed so the interval is not "
                "conditional on optimizer convergence"
            )
        else:
            requirement = f"at least {required} successful refits are required"
        return (
            False,
            f"Bootstrap uncertainty unavailable: {n_success}/{n_boot} refits succeeded; "
            f"{requirement}.",
        )
    return True, ""


def _competitive_objective_tolerance(best_rss: float, dof: int, n_logk: int) -> float:
    """Return a conservative RSS window for statistically competitive fits."""
    numerical_tolerance = max(1e-12, abs(best_rss) * 1e-10)
    if best_rss <= 0.0 or dof <= 0:
        return numerical_tolerance
    cutoff = float(chi2.ppf(BOOTSTRAP_COMPETITIVE_CONFIDENCE, max(1, n_logk)))
    profile_tolerance = cutoff * best_rss / dof
    if not np.isfinite(profile_tolerance) or profile_tolerance <= 0.0:
        return numerical_tolerance
    return max(numerical_tolerance, profile_tolerance)


def _residual_bootstrap(
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> Dataset:
    idx = rng.integers(0, ds.n_points, size=ds.n_points)
    col_means = np.nanmean(residuals, axis=0, keepdims=True)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    centered = residuals - col_means
    resampled = centered[idx, :]
    for col_idx in range(centered.shape[1]):
        finite_residuals = centered[:, col_idx][np.isfinite(centered[:, col_idx])]
        missing = ~np.isfinite(resampled[:, col_idx])
        if not np.any(missing):
            continue
        if finite_residuals.size == 0:
            resampled[missing, col_idx] = 0.0
        else:
            resampled[missing, col_idx] = rng.choice(finite_residuals, size=int(np.count_nonzero(missing)))
    y_boot = y_pred + resampled
    y_boot[~np.isfinite(ds.y)] = np.nan
    return Dataset(
        name=ds.name,
        path=ds.path,
        h_tot=ds.h_tot,
        g_tot=ds.g_tot,
        x=ds.x,
        y=y_boot,
        y_cols=ds.y_cols,
        dropped_peaks=ds.dropped_peaks,
        dropped_rows=ds.dropped_rows,
    )


def _parametric_bootstrap(
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    scale_factor: float = 1.0,
) -> Dataset:
    scale = np.nanstd(residuals, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    if np.isfinite(scale_factor) and scale_factor > 0:
        scale = scale * float(scale_factor)
    noise = rng.normal(0.0, 1.0, size=y_pred.shape) * scale.reshape(1, -1)
    y_boot = y_pred + noise
    y_boot[~np.isfinite(ds.y)] = np.nan
    return Dataset(
        name=ds.name,
        path=ds.path,
        h_tot=ds.h_tot,
        g_tot=ds.g_tot,
        x=ds.x,
        y=y_boot,
        y_cols=ds.y_cols,
        dropped_peaks=ds.dropped_peaks,
        dropped_rows=ds.dropped_rows,
    )


def _points_bootstrap(
    rng: np.random.Generator,
    ds: Dataset,
) -> Dataset:
    idx = rng.integers(0, ds.n_points, size=ds.n_points)
    return Dataset(
        name=ds.name,
        path=ds.path,
        h_tot=ds.h_tot[idx],
        g_tot=ds.g_tot[idx],
        x=ds.x[idx],
        y=ds.y[idx],
        y_cols=ds.y_cols,
        dropped_peaks=ds.dropped_peaks,
        dropped_rows=ds.dropped_rows,
    )


def _resample_bootstrap_dataset(
    method: str,
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    parametric_scale_factor: float = 1.0,
) -> Dataset:
    if method == "points":
        return _points_bootstrap(rng, ds)
    if method == "parametric":
        return _parametric_bootstrap(rng, ds, y_pred, residuals, scale_factor=parametric_scale_factor)
    return _residual_bootstrap(rng, ds, y_pred, residuals)


def _delete_dataset_row(ds: Dataset, row: int) -> Dataset:
    return Dataset(
        name=ds.name,
        path=ds.path,
        h_tot=np.delete(ds.h_tot, row, axis=0),
        g_tot=np.delete(ds.g_tot, row, axis=0),
        x=np.delete(ds.x, row, axis=0),
        y=np.delete(ds.y, row, axis=0),
        y_cols=ds.y_cols,
        dropped_peaks=ds.dropped_peaks,
        dropped_rows=ds.dropped_rows,
    )


def _jitter_bootstrap_start(
    params: np.ndarray,
    model: ModelSpec,
    rng: np.random.Generator,
    logk_jitter: float,
    logk_bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    params0 = params.copy()
    if model.n_logk and logk_jitter > 0:
        jitter = rng.normal(0.0, logk_jitter, size=model.n_logk)
        params0[: model.n_logk] = params0[: model.n_logk] + jitter
        if logk_bounds is not None:
            params0[: model.n_logk] = np.clip(params0[: model.n_logk], logk_bounds[0], logk_bounds[1])
    return params0


def accept_bootstrap_fit(
    params_fit: np.ndarray,
    res: OptimizeResult,
    model: ModelSpec,
    datasets: List[Dataset],
    predict_all_fn: Callable[..., Tuple[List[np.ndarray], List[object], List[np.ndarray]]],
    numeric_exceptions: Tuple[Type[BaseException], ...],
    solver_failure_mode: str = "fail-fast",
) -> bool:
    if not bool(getattr(res, "success", False)):
        return False
    if not bool(np.all(np.isfinite(params_fit))):
        return False
    if not optimizer_result_is_identifiable(
        res,
        params_fit.size,
        model.n_logk,
        model,
        datasets,
    ):
        return False
    try:
        y_pred_list, _, _ = predict_all_fn(
            params_fit,
            model,
            datasets,
            solver_failure_mode=solver_failure_mode,
        )
    except numeric_exceptions:
        return False
    for ds, y_pred in zip(datasets, y_pred_list):
        valid_mask = np.isfinite(ds.y)
        if not np.any(valid_mask):
            return False
        if not np.all(np.isfinite(y_pred[valid_mask])):
            return False
    return True


def _jackknife_param_samples(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
    fit_with_initial_fn: Callable[..., Tuple[np.ndarray, OptimizeResult]],
    param_bounds_fn: Callable[[np.ndarray, ModelSpec, Optional[Tuple[float, float]]], Tuple[np.ndarray, np.ndarray]],
    predict_all_fn: Callable[..., Tuple[List[np.ndarray], List[object], List[np.ndarray]]],
    numeric_exceptions: Tuple[Type[BaseException], ...],
    logk_bounds: Optional[Tuple[float, float]],
    max_nfev: int,
    solver_failure_mode: str,
) -> Tuple[np.ndarray, int]:
    samples = []
    expected = 0
    for dataset_idx, ds in enumerate(datasets):
        for row in range(ds.n_points):
            if not np.any(np.isfinite(ds.y[row])):
                continue
            expected += 1
            jackknife_datasets = list(datasets)
            jackknife_datasets[dataset_idx] = _delete_dataset_row(ds, row)
            params0 = params.copy()
            bounds = param_bounds_fn(params0, model, logk_bounds)
            fit_kwargs = {"max_nfev": max_nfev, "bounds": bounds}
            if solver_failure_mode != "fail-fast":
                fit_kwargs["solver_failure_mode"] = solver_failure_mode
            try:
                params_fit, result = fit_with_initial_fn(
                    model,
                    jackknife_datasets,
                    params0,
                    **fit_kwargs,
                )
            except numeric_exceptions:
                continue
            if accept_bootstrap_fit(
                params_fit,
                result,
                model,
                jackknife_datasets,
                predict_all_fn=predict_all_fn,
                numeric_exceptions=numeric_exceptions,
                solver_failure_mode=solver_failure_mode,
            ):
                samples.append(params_fit)
    if not samples:
        return np.full((0, len(params)), np.nan), expected
    return np.vstack(samples), expected


def bootstrap_params(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
    n_boot: int,
    method: str,
    seed: Optional[int],
    logk_bounds: Optional[Tuple[float, float]],
    logk_jitter: float,
    predict_all_fn: Callable[..., Tuple[List[np.ndarray], List[object], List[np.ndarray]]],
    fit_with_initial_fn: Callable[..., Tuple[np.ndarray, OptimizeResult]],
    param_bounds_fn: Callable[[np.ndarray, ModelSpec, Optional[Tuple[float, float]]], Tuple[np.ndarray, np.ndarray]],
    numeric_exceptions: Tuple[Type[BaseException], ...],
    ci_method: str = "percentile",
    max_nfev: int = 5000,
    solver_failure_mode: str = "fail-fast",
) -> BootstrapResult:
    if ci_method not in {"percentile", "bca"}:
        raise ValueError("ci_method must be one of: percentile, bca")
    if method not in {"residual", "points", "parametric"}:
        raise ValueError("method must be one of: residual, points, parametric")
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")
    if not np.isfinite(logk_jitter) or logk_jitter < 0:
        raise ValueError("logk_jitter must be finite and non-negative")

    rng = np.random.default_rng(seed)
    param_samples = []
    n_success = 0
    logk_jitter = float(logk_jitter)

    y_pred_list, _, residuals = predict_all_fn(
        params,
        model,
        datasets,
        solver_failure_mode=solver_failure_mode,
    )
    total_valid = int(sum(np.count_nonzero(np.isfinite(ds.y)) for ds in datasets))
    dof = int(total_valid - len(params))
    parametric_scale_factor = float(np.sqrt(total_valid / dof)) if dof > 0 else 1.0

    for _ in range(n_boot):
        boot_datasets: List[Dataset] = []
        for ds, y_pred, res in zip(datasets, y_pred_list, residuals):
            boot = _resample_bootstrap_dataset(
                method,
                rng,
                ds,
                y_pred,
                res,
                parametric_scale_factor=parametric_scale_factor,
            )
            boot_datasets.append(boot)

        jittered_start = _jitter_bootstrap_start(params, model, rng, logk_jitter, logk_bounds)
        starts = [jittered_start]
        if not np.array_equal(jittered_start, params):
            # Retry the same pseudo-dataset from the full-data optimum. This
            # recovers start-induced failures without replacing a failed tail
            # draw with a newly sampled pseudo-dataset.
            starts.append(params.copy())

        converged_candidates = []
        for params0 in starts:
            bounds = param_bounds_fn(params0, model, logk_bounds)
            fit_kwargs = {"max_nfev": max_nfev, "bounds": bounds}
            if solver_failure_mode != "fail-fast":
                fit_kwargs["solver_failure_mode"] = solver_failure_mode
            try:
                params_fit, res = fit_with_initial_fn(
                    model,
                    boot_datasets,
                    params0,
                    **fit_kwargs,
                )
            except numeric_exceptions:
                continue
            if not bool(getattr(res, "success", False)) or not np.all(np.isfinite(params_fit)):
                continue
            try:
                candidate_rss = float(np.sum(np.asarray(res.fun, dtype=float) ** 2))
            except (TypeError, ValueError, FloatingPointError):
                continue
            if not np.isfinite(candidate_rss):
                continue
            accepted = accept_bootstrap_fit(
                params_fit,
                res,
                model,
                boot_datasets,
                predict_all_fn=predict_all_fn,
                numeric_exceptions=numeric_exceptions,
                solver_failure_mode=solver_failure_mode,
            )
            converged_candidates.append((candidate_rss, accepted, params_fit))

        if not converged_candidates:
            continue
        best_rss = min(candidate[0] for candidate in converged_candidates)
        boot_valid = int(
            sum(np.count_nonzero(np.isfinite(ds.y)) for ds in boot_datasets)
        )
        objective_tolerance = _competitive_objective_tolerance(
            best_rss,
            boot_valid - len(params),
            model.n_logk,
        )
        competitive = [
            candidate
            for candidate in converged_candidates
            if candidate[0] <= best_rss + objective_tolerance
        ]
        # A statistically competitive converged solution that is bound-limited,
        # ill-conditioned, or otherwise invalid is evidence of a censored
        # pseudo-dataset tail. Do not replace it with an interior basin whose
        # objective is indistinguishable at the 95% profile-likelihood scale.
        if any(not candidate[1] for candidate in competitive):
            continue
        accepted_candidates = [candidate for candidate in competitive if candidate[1]]
        if not accepted_candidates:
            continue
        accepted_params = min(accepted_candidates, key=lambda candidate: candidate[0])[2]
        param_samples.append(accepted_params)
        n_success += 1

    if not param_samples:
        samples = np.full((0, len(params)), np.nan)
    else:
        samples = np.vstack(param_samples)

    logk_samples = samples[:, : model.n_logk] if model.n_logk else np.full((samples.shape[0], 0), np.nan)

    ci_low_percentile = np.full((model.n_logk,), np.nan)
    ci_high_percentile = np.full((model.n_logk,), np.nan)
    ci_low_bca = np.full((model.n_logk,), np.nan)
    ci_high_bca = np.full((model.n_logk,), np.nan)
    ci_valid, ci_message = _bootstrap_ci_status(n_success, n_boot)
    if model.n_logk == 0 and ci_valid:
        ci_method_used = "not-applicable"
    else:
        ci_method_used = "unavailable"

    if logk_samples.shape[0] >= MIN_BOOTSTRAP_CI_SUCCESSES and logk_samples.size > 0:
        ci_low_percentile = np.percentile(logk_samples, 2.5, axis=0)
        ci_high_percentile = np.percentile(logk_samples, 97.5, axis=0)

        if ci_valid and ci_method == "bca":
            jackknife_samples, jackknife_expected = _jackknife_param_samples(
                params=params,
                model=model,
                datasets=datasets,
                fit_with_initial_fn=fit_with_initial_fn,
                param_bounds_fn=param_bounds_fn,
                predict_all_fn=predict_all_fn,
                numeric_exceptions=numeric_exceptions,
                logk_bounds=logk_bounds,
                max_nfev=max_nfev,
                solver_failure_mode=solver_failure_mode,
            )
            if jackknife_expected < 3 or jackknife_samples.shape[0] != jackknife_expected:
                ci_valid = False
                ci_message = (
                    "BCa CI unavailable: all delete-one jackknife refits must succeed "
                    f"({jackknife_samples.shape[0]}/{jackknife_expected} succeeded)."
                )
            else:
                for i in range(model.n_logk):
                    bca_low, bca_high = _bca_ci(
                        logk_samples[:, i],
                        float(params[i]),
                        jackknife_samples[:, i],
                        alpha=0.05,
                    )
                    ci_low_bca[i] = bca_low
                    ci_high_bca[i] = bca_high
                if not np.all(np.isfinite(ci_low_bca) & np.isfinite(ci_high_bca)):
                    ci_valid = False
                    ci_message = "BCa CI unavailable: adjusted quantiles could not be estimated."

    if ci_valid and ci_method == "bca":
        ci_low = ci_low_bca.copy()
        ci_high = ci_high_bca.copy()
        ci_method_used = "bca-local"
    elif ci_valid and ci_method == "percentile":
        ci_low = ci_low_percentile.copy()
        ci_high = ci_high_percentile.copy()
        ci_method_used = "percentile"
    else:
        ci_low = np.full((model.n_logk,), np.nan)
        ci_high = np.full((model.n_logk,), np.nan)

    return BootstrapResult(
        param_samples=samples,
        logk_samples=logk_samples,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_low_percentile=ci_low_percentile,
        ci_high_percentile=ci_high_percentile,
        ci_low_bca=ci_low_bca,
        ci_high_bca=ci_high_bca,
        ci_method=ci_method,
        ci_method_used=ci_method_used,
        ci_valid=ci_valid,
        ci_message=ci_message,
        n_success=n_success,
        n_boot=n_boot,
    )
