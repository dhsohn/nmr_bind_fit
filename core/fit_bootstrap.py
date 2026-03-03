from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Type

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.stats import norm

from .io import Dataset
from .models import ModelSpec


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
    n_success: int
    n_boot: int


def _bca_ci_approx(
    samples: np.ndarray,
    original_stat: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    # Approximate BCa interval using bootstrap samples with jackknife-over-samples acceleration.
    vals = np.asarray(samples, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3 or not np.isfinite(original_stat):
        return float("nan"), float("nan")

    n = int(vals.size)
    prop_less = float(np.mean(vals < original_stat))
    # Avoid ±inf in inverse-normal transform.
    eps = 0.5 / n
    prop_less = float(np.clip(prop_less, eps, 1.0 - eps))
    z0 = float(norm.ppf(prop_less))

    # Jackknife-over-bootstrap-samples approximation for acceleration.
    total = float(np.sum(vals))
    jackknife = (total - vals) / float(n - 1)
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
    centered[~np.isfinite(centered)] = 0.0
    resampled = centered[idx, :]
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
    )


def _parametric_bootstrap(
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> Dataset:
    scale = np.nanstd(residuals, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale), scale, 0.0)
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
    )


def _resample_bootstrap_dataset(
    method: str,
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> Dataset:
    if method == "points":
        return _points_bootstrap(rng, ds)
    if method == "parametric":
        return _parametric_bootstrap(rng, ds, y_pred, residuals)
    return _residual_bootstrap(rng, ds, y_pred, residuals)


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

    for _ in range(n_boot):
        boot_datasets: List[Dataset] = []
        for ds, y_pred, res in zip(datasets, y_pred_list, residuals):
            boot = _resample_bootstrap_dataset(method, rng, ds, y_pred, res)
            boot_datasets.append(boot)

        params0 = _jitter_bootstrap_start(params, model, rng, logk_jitter, logk_bounds)
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
        if not accept_bootstrap_fit(
            params_fit,
            res,
            model,
            boot_datasets,
            predict_all_fn=predict_all_fn,
            numeric_exceptions=numeric_exceptions,
            solver_failure_mode=solver_failure_mode,
        ):
            continue
        param_samples.append(params_fit)
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

    if logk_samples.size > 0:
        ci_low_percentile = np.percentile(logk_samples, 2.5, axis=0)
        ci_high_percentile = np.percentile(logk_samples, 97.5, axis=0)

        if ci_method == "bca":
            for i in range(model.n_logk):
                bca_low, bca_high = _bca_ci_approx(logk_samples[:, i], float(params[i]), alpha=0.05)
                ci_low_bca[i] = bca_low
                ci_high_bca[i] = bca_high

    if ci_method == "bca":
        ci_low = np.where(np.isfinite(ci_low_bca), ci_low_bca, ci_low_percentile)
        ci_high = np.where(np.isfinite(ci_high_bca), ci_high_bca, ci_high_percentile)
    else:
        ci_low = ci_low_percentile.copy()
        ci_high = ci_high_percentile.copy()

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
        n_success=n_success,
        n_boot=n_boot,
    )
