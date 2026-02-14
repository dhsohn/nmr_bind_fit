"""Model fitting and bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import itertools
import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from .io import Dataset
from .models import MODEL_SPECS, ModelSpec, predict_dataset, split_params_multi
from .stats import aicc_from_loglik, bic_from_loglik, gaussian_loglik


@dataclass
class BootstrapResult:
    param_samples: np.ndarray
    logk_samples: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_success: int
    n_boot: int


@dataclass
class FitResult:
    model: ModelSpec
    datasets: List[Dataset]
    params: np.ndarray
    param_names: List[str]
    success: bool
    message: str
    rss: float
    rmse: float
    r2: float
    bic: float
    aicc: float
    n: int
    p: int
    dof: int
    y_pred: List[np.ndarray]
    species: List
    residuals: List[np.ndarray]
    bootstrap: Optional[BootstrapResult]
    penalty_count: int


class ModelFitError(RuntimeError):
    """Recoverable per-model fit failure for multi-model sweeps."""


_NUMERIC_EXCEPTIONS = (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


def _residual_penalty_scale(y: np.ndarray) -> float:
    vals = np.asarray(y, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 1e3
    scale = float(np.max(np.abs(finite)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return float(scale * 1e3)


def _init_delta(model: ModelSpec, dataset: Dataset) -> np.ndarray:
    """Heuristic start values for chemical shift parameters."""
    # Use endpoints to seed delta values for faster convergence.
    y0 = dataset.y[0]
    y1 = dataset.y[-1]
    if model.name == "nb":
        x0 = dataset.x[0]
        x1 = dataset.x[-1]
        slope = np.zeros_like(y0)
        if x1 != x0:
            slope = (y1 - y0) / (x1 - x0)
        return np.column_stack([y0, slope])
    if model.n_delta_per_peak == 2:
        return np.column_stack([y0, y1])
    if model.n_delta_per_peak == 3:
        return np.column_stack([y0, y1, y1])
    raise ValueError("Unsupported delta parameter count")


def _build_initial_params(
    model: ModelSpec,
    datasets: List[Dataset],
    logk_values: Sequence[float],
) -> np.ndarray:
    """Assemble the full parameter vector for one start."""
    # Concatenate logK values followed by per-dataset delta parameters.
    params: List[float] = []
    params.extend(list(logk_values))
    for ds in datasets:
        delta = _init_delta(model, ds)
        params.extend(delta.ravel().tolist())
    return np.array(params, dtype=float)


def _param_names_multi(model: ModelSpec, datasets: List[Dataset]) -> List[str]:
    """Parameter names for multi-dataset fits (replicates)."""
    names: List[str] = []
    if model.n_logk == 1:
        names.append("logK")
    elif model.n_logk == 2:
        names.extend(["logK1", "logK2"])
    # Name delta parameters by species, dataset, and peak.
    for ds in datasets:
        for peak in ds.y_cols:
            for label in model.species_labels:
                names.append(f"{label}_{ds.name}_{peak}")
    return names


def _residual_vector(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
    solver_failure_mode: str = "fail-fast",
    penalty_counter: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Return concatenated residuals used by least_squares."""
    logk, deltas = split_params_multi(params, model, datasets)
    residuals = []
    for ds, delta in zip(datasets, deltas):
        valid_mask = np.isfinite(ds.y)
        n_valid = int(np.count_nonzero(valid_mask))
        if n_valid == 0:
            continue
        penalty = np.full((n_valid,), _residual_penalty_scale(ds.y), dtype=float)
        try:
            y_pred, _ = predict_dataset(model, ds, logk, delta, solver_failure_mode=solver_failure_mode)
        except _NUMERIC_EXCEPTIONS:
            if penalty_counter is not None:
                penalty_counter["count"] = penalty_counter.get("count", 0) + 1
            residuals.append(penalty)
            continue
        if not np.all(np.isfinite(y_pred[valid_mask])):
            if penalty_counter is not None:
                penalty_counter["count"] = penalty_counter.get("count", 0) + 1
            residuals.append(penalty)
            continue
        res = ds.y - y_pred
        res_valid = res[valid_mask]
        if not np.all(np.isfinite(res_valid)):
            if penalty_counter is not None:
                penalty_counter["count"] = penalty_counter.get("count", 0) + 1
            residuals.append(penalty)
            continue
        residuals.append(res_valid.ravel())
    if not residuals:
        return np.array([], dtype=float)
    return np.concatenate(residuals)


def _predict_all(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
    solver_failure_mode: str = "fail-fast",
) -> Tuple[List[np.ndarray], List, List[np.ndarray]]:
    """Predict shifts and residuals on the original ppm scale."""
    # Run model prediction for each dataset and keep raw residuals.
    logk, deltas = split_params_multi(params, model, datasets)
    y_pred_list = []
    species_list = []
    residuals = []
    for ds, delta in zip(datasets, deltas):
        y_pred, species = predict_dataset(model, ds, logk, delta, solver_failure_mode=solver_failure_mode)
        y_pred_list.append(y_pred)
        species_list.append(species)
        residuals.append(np.where(np.isfinite(ds.y), ds.y - y_pred, np.nan))
    return y_pred_list, species_list, residuals


def _rss_value(residuals: List[np.ndarray]) -> float:
    """Return residual sum of squares."""
    rss = 0.0
    for res in residuals:
        rss += float(np.nansum(res**2))
    return rss


def _r2_score(datasets: List[Dataset], y_pred_list: List[np.ndarray]) -> float:
    """R2 is reported for reference (not used for selection)."""
    # Flatten all datasets into one vector for a global R2.
    y_all = []
    y_pred_all = []
    for ds, y_pred in zip(datasets, y_pred_list):
        mask = np.isfinite(ds.y) & np.isfinite(y_pred)
        if not np.any(mask):
            continue
        y_all.append(ds.y[mask].ravel())
        y_pred_all.append(y_pred[mask].ravel())
    if not y_all:
        return float("nan")
    y_all = np.concatenate(y_all)
    y_pred_all = np.concatenate(y_pred_all)
    ss_res = np.sum((y_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _fit_with_initial(
    model: ModelSpec,
    datasets: List[Dataset],
    params0: np.ndarray,
    max_nfev: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    solver_failure_mode: str = "fail-fast",
) -> Tuple[np.ndarray, OptimizeResult]:
    """Run least_squares for one initialization."""
    # Use bounded trust-region least squares on concatenated residuals.
    penalty_counter: Dict[str, int] = {"count": 0}

    def residual_fn(current_params: np.ndarray, current_model: ModelSpec, current_datasets: List[Dataset]) -> np.ndarray:
        return _residual_vector(
            current_params,
            current_model,
            current_datasets,
            solver_failure_mode=solver_failure_mode,
            penalty_counter=penalty_counter,
        )

    res = least_squares(
        residual_fn,
        params0,
        args=(model, datasets),
        method="trf",
        max_nfev=max_nfev,
        x_scale="jac",
        bounds=bounds,
    )
    setattr(res, "penalty_count", int(penalty_counter.get("count", 0)))
    return res.x, res


def _param_bounds(
    params0: np.ndarray,
    model: ModelSpec,
    logk_bounds: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply optional logK bounds while leaving shift parameters free."""
    # Only constrain logK parameters, leave delta unbounded.
    if logk_bounds is None or model.n_logk == 0:
        return (np.full_like(params0, -np.inf), np.full_like(params0, np.inf))
    lower = np.full_like(params0, -np.inf)
    upper = np.full_like(params0, np.inf)
    lower[: model.n_logk] = logk_bounds[0]
    upper[: model.n_logk] = logk_bounds[1]
    return lower, upper


def _total_observations(datasets: List[Dataset]) -> int:
    # Count total scalar observations across all datasets and peaks.
    return int(sum(np.count_nonzero(np.isfinite(ds.y)) for ds in datasets))


def _failed_fit_result(
    model: ModelSpec,
    datasets: List[Dataset],
    params: np.ndarray,
    param_names: List[str],
    message: str,
    species: Optional[List] = None,
) -> FitResult:
    # Build a consistent failure payload used across early-return paths.
    n = _total_observations(datasets)
    p = int(len(params))
    dof = int(n - p)
    return FitResult(
        model=model,
        datasets=datasets,
        params=params,
        param_names=param_names,
        success=False,
        message=message,
        rss=float("nan"),
        rmse=float("nan"),
        r2=float("nan"),
        bic=float("nan"),
        aicc=float("nan"),
        n=n,
        p=p,
        dof=dof,
        y_pred=[],
        species=species if species is not None else [],
        residuals=[],
        bootstrap=None,
        penalty_count=0,
    )


def _select_best_multistart(
    model: ModelSpec,
    datasets: List[Dataset],
    logk_grid: Sequence[Tuple[float, ...]],
    max_nfev: int,
    logk_bounds: Optional[Tuple[float, float]],
    solver_failure_mode: str = "fail-fast",
) -> Tuple[Optional[np.ndarray], Optional[OptimizeResult]]:
    # Prefer converged starts and keep best failure only as fallback diagnostics.
    best_success_params = None
    best_success_res = None
    best_success_rss = None
    best_failed_params = None
    best_failed_res = None
    best_failed_rss = None

    for logk_vals in logk_grid:
        params0 = _build_initial_params(model, datasets, logk_vals)
        bounds = _param_bounds(params0, model, logk_bounds)
        fit_kwargs = {"max_nfev": max_nfev, "bounds": bounds}
        if solver_failure_mode != "fail-fast":
            fit_kwargs["solver_failure_mode"] = solver_failure_mode
        try:
            params, res = _fit_with_initial(model, datasets, params0, **fit_kwargs)
        except _NUMERIC_EXCEPTIONS:
            continue
        rss = float(np.sum(res.fun**2))
        if bool(getattr(res, "success", False)):
            if best_success_rss is None or rss < best_success_rss:
                best_success_rss = rss
                best_success_params = params
                best_success_res = res
        elif best_failed_rss is None or rss < best_failed_rss:
            best_failed_rss = rss
            best_failed_params = params
            best_failed_res = res

    if best_success_params is not None and best_success_res is not None:
        return best_success_params, best_success_res
    return best_failed_params, best_failed_res


def _build_logk_grid(
    model: ModelSpec,
    logk_starts: Sequence[float],
    logk_bounds: Optional[Tuple[float, float]],
) -> List[Tuple[float, ...]]:
    # Expand multistart seeds across the model's logK dimensions.
    if model.n_logk == 0:
        return [()]
    starts = list(logk_starts)
    if logk_bounds is not None:
        starts = [v for v in starts if logk_bounds[0] <= v <= logk_bounds[1]]
        if not starts:
            raise ValueError("No K starts within bounds.")
    return list(itertools.product(starts, repeat=model.n_logk))


def _nonfinite_prediction_message(datasets: List[Dataset], species_list: List[object]) -> str:
    # Build a diagnostics message when equilibrium predictions are non-finite.
    fail_points = 0
    total_points = 0
    for ds, species in zip(datasets, species_list):
        stats = getattr(species, "solver_stats", None)
        if stats is not None:
            fail_points += int(getattr(stats, "fallback_fail", 0))
            total_points += int(getattr(stats, "points", ds.n_points))
    message = "Equilibrium solver produced non-finite predictions."
    if total_points > 0:
        message = f"{message} Failed points: {fail_points}/{total_points}."
    return message


def _information_criteria(
    datasets: List[Dataset],
    residuals: List[np.ndarray],
    p: int,
) -> Tuple[float, float]:
    # Compute BIC/AICc from one shared residual variance across all residuals.
    stacked = np.concatenate([res.ravel() for res in residuals]) if residuals else np.array([], dtype=float)
    loglik, n_loglik, n_sigma = gaussian_loglik(stacked)
    if not np.isfinite(loglik):
        names = ", ".join(ds.name for ds in datasets)
        raise RuntimeError(f"BIC calculation failed: log-likelihood is NaN for datasets: {names}.")
    bic_p = p + n_sigma
    if n_loglik <= 0:
        raise RuntimeError("BIC calculation failed: no valid log-likelihood terms.")
    bic = bic_from_loglik(loglik, n_loglik, bic_p)
    aicc = aicc_from_loglik(loglik, n_loglik, bic_p)
    return bic, aicc


def _build_successful_fit_result(
    model: ModelSpec,
    datasets: List[Dataset],
    best_params: np.ndarray,
    best_res: OptimizeResult,
    bootstrap: int,
    bootstrap_method: str,
    seed: Optional[int],
    logk_bounds: Optional[Tuple[float, float]],
    logk_jitter: float,
    max_nfev: int,
    solver_failure_mode: str,
) -> FitResult:
    # Evaluate diagnostics/statistics and build the final successful FitResult.
    param_names = _param_names_multi(model, datasets)
    y_pred_list, species_list, residuals = _predict_all(
        best_params,
        model,
        datasets,
        solver_failure_mode=solver_failure_mode,
    )
    if not all(np.all(np.isfinite(y_pred[np.isfinite(ds.y)])) for ds, y_pred in zip(datasets, y_pred_list)):
        return _failed_fit_result(
            model=model,
            datasets=datasets,
            params=best_params,
            param_names=param_names,
            message=_nonfinite_prediction_message(datasets, species_list),
            species=species_list,
        )

    rss = _rss_value(residuals)
    n = _total_observations(datasets)
    p = int(len(best_params))
    dof = int(n - p)
    rmse = float(np.sqrt(rss / n)) if n > 0 else float("nan")
    r2 = _r2_score(datasets, y_pred_list)
    bic, aicc = _information_criteria(datasets, residuals, p)

    bootstrap_result = None
    if bootstrap > 0:
        bootstrap_result = bootstrap_params(
            best_params,
            model,
            datasets,
            bootstrap,
            bootstrap_method,
            seed=seed,
            logk_bounds=logk_bounds,
            logk_jitter=logk_jitter,
            max_nfev=max_nfev,
            solver_failure_mode=solver_failure_mode,
        )

    return FitResult(
        model=model,
        datasets=datasets,
        params=best_params,
        param_names=param_names,
        success=bool(best_res.success),
        message=str(best_res.message),
        rss=rss,
        rmse=rmse,
        r2=r2,
        bic=bic,
        aicc=aicc,
        n=n,
        p=p,
        dof=dof,
        y_pred=y_pred_list,
        species=species_list,
        residuals=residuals,
        bootstrap=bootstrap_result,
        penalty_count=int(getattr(best_res, "penalty_count", 0)),
    )


def fit_model(
    datasets: List[Dataset],
    model_name: str,
    logk_starts: Sequence[float],
    max_nfev: int = 5000,
    bootstrap: int = 0,
    bootstrap_method: str = "residual",
    seed: Optional[int] = None,
    logk_bounds: Optional[Tuple[float, float]] = None,
    logk_jitter: float = 0.1,
    solver_failure_mode: str = "fail-fast",
) -> FitResult:
    """Fit one model to one or multiple datasets with multistart logK."""
    model = MODEL_SPECS[model_name]
    try:
        logk_grid = _build_logk_grid(model, logk_starts, logk_bounds)
    except ValueError as exc:
        raise ModelFitError(str(exc)) from exc

    best_params, best_res = _select_best_multistart(
        model,
        datasets,
        logk_grid,
        max_nfev=max_nfev,
        logk_bounds=logk_bounds,
        solver_failure_mode=solver_failure_mode,
    )

    if best_params is None or best_res is None:
        raise ModelFitError(f"Fit failed for model {model_name}")

    # Return early if the optimizer failed to converge.
    if not best_res.success:
        param_names = _param_names_multi(model, datasets)
        return _failed_fit_result(
            model=model,
            datasets=datasets,
            params=best_params,
            param_names=param_names,
            message=str(best_res.message),
        )

    try:
        return _build_successful_fit_result(
            model=model,
            datasets=datasets,
            best_params=best_params,
            best_res=best_res,
            bootstrap=bootstrap,
            bootstrap_method=bootstrap_method,
            seed=seed,
            logk_bounds=logk_bounds,
            logk_jitter=logk_jitter,
            max_nfev=max_nfev,
            solver_failure_mode=solver_failure_mode,
        )
    except RuntimeError as exc:
        if str(exc).startswith("BIC calculation failed:"):
            raise ModelFitError(str(exc)) from exc
        raise


def _residual_bootstrap(
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> Dataset:
    """Residual bootstrap by resampling centered residuals."""
    # Resample residuals by index to preserve autocorrelation structure.
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
    """Parametric bootstrap using Gaussian noise estimated from residuals."""
    # Inject Gaussian noise using per-peak residual standard deviation.
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
    # Sample complete titration points with replacement.
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
    # Dispatch bootstrap resampling strategy for one dataset.
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
    # Apply bounded random jitter to logK start values.
    params0 = params.copy()
    if model.n_logk and logk_jitter > 0:
        jitter = rng.normal(0.0, logk_jitter, size=model.n_logk)
        params0[: model.n_logk] = params0[: model.n_logk] + jitter
        if logk_bounds is not None:
            params0[: model.n_logk] = np.clip(params0[: model.n_logk], logk_bounds[0], logk_bounds[1])
    return params0


def _accept_bootstrap_fit(
    params_fit: np.ndarray,
    res: OptimizeResult,
    model: ModelSpec,
    datasets: List[Dataset],
    solver_failure_mode: str = "fail-fast",
) -> bool:
    # Count only converged refits with finite parameter vectors.
    if not bool(getattr(res, "success", False)):
        return False
    if not bool(np.all(np.isfinite(params_fit))):
        return False
    try:
        y_pred_list, _, _ = _predict_all(
            params_fit,
            model,
            datasets,
            solver_failure_mode=solver_failure_mode,
        )
    except _NUMERIC_EXCEPTIONS:
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
    max_nfev: int = 5000,
    solver_failure_mode: str = "fail-fast",
) -> BootstrapResult:
    """Run bootstrap refits and collect parameter samples."""
    # Refit the model on resampled datasets to estimate parameter dispersion.
    rng = np.random.default_rng(seed)
    param_samples = []
    n_success = 0
    logk_jitter = float(logk_jitter)

    y_pred_list, _, residuals = _predict_all(
        params,
        model,
        datasets,
        solver_failure_mode=solver_failure_mode,
    )

    for _ in range(n_boot):
        # Resample each dataset consistently across peaks.
        boot_datasets: List[Dataset] = []
        for ds, y_pred, res in zip(datasets, y_pred_list, residuals):
            boot = _resample_bootstrap_dataset(method, rng, ds, y_pred, res)
            boot_datasets.append(boot)

        # Small random perturbations reduce local-minimum bias.
        params0 = _jitter_bootstrap_start(params, model, rng, logk_jitter, logk_bounds)
        bounds = _param_bounds(params0, model, logk_bounds)
        fit_kwargs = {"max_nfev": max_nfev, "bounds": bounds}
        if solver_failure_mode != "fail-fast":
            fit_kwargs["solver_failure_mode"] = solver_failure_mode
        try:
            params_fit, res = _fit_with_initial(
                model,
                boot_datasets,
                params0,
                **fit_kwargs,
            )
        except _NUMERIC_EXCEPTIONS:
            continue
        if not _accept_bootstrap_fit(
            params_fit,
            res,
            model,
            boot_datasets,
            solver_failure_mode=solver_failure_mode,
        ):
            continue
        param_samples.append(params_fit)
        n_success += 1

    # Stack successful samples and compute percentile intervals.
    if not param_samples:
        samples = np.full((0, len(params)), np.nan)
    else:
        samples = np.vstack(param_samples)

    logk_samples = samples[:, : model.n_logk] if model.n_logk else np.full((samples.shape[0], 0), np.nan)

    if logk_samples.size == 0:
        ci_low = np.full((model.n_logk,), np.nan)
        ci_high = np.full((model.n_logk,), np.nan)
    else:
        ci_low = np.percentile(logk_samples, 2.5, axis=0)
        ci_high = np.percentile(logk_samples, 97.5, axis=0)

    return BootstrapResult(
        param_samples=samples,
        logk_samples=logk_samples,
        ci_low=ci_low,
        ci_high=ci_high,
        n_success=n_success,
        n_boot=n_boot,
    )


def _exception_failure_result(
    model_name: str,
    datasets: List[Dataset],
    exc: Exception,
) -> FitResult:
    # Convert per-model exceptions into a normal failure result payload.
    model = MODEL_SPECS[model_name]
    n_delta = sum(ds.n_peaks * model.n_delta_per_peak for ds in datasets)
    n_params = int(model.n_logk + n_delta)
    params = np.full((n_params,), np.nan, dtype=float)
    param_names = _param_names_multi(model, datasets)
    message = f"{type(exc).__name__}: {exc}"
    return _failed_fit_result(
        model=model,
        datasets=datasets,
        params=params,
        param_names=param_names,
        message=message,
    )


def _iter_fit_jobs(
    datasets: List[Dataset],
    model_names: Sequence[str],
    replicates: bool,
):
    # Yield (datasets_for_fit, model_name) jobs for shared and per-dataset modes.
    if replicates:
        for model_name in model_names:
            yield datasets, model_name
        return
    for ds in datasets:
        for model_name in model_names:
            yield [ds], model_name


def fit_models(
    datasets: List[Dataset],
    model_names: Sequence[str],
    logk_starts: Sequence[float],
    replicates: bool,
    max_nfev: int,
    bootstrap: int,
    bootstrap_method: str,
    seed: Optional[int],
    logk_bounds: Optional[Tuple[float, float]],
    logk_jitter: float,
    solver_failure_mode: str = "fail-fast",
) -> List[FitResult]:
    """Fit all requested models, optionally as simultaneous replicates."""
    results = []
    for fit_datasets, model_name in _iter_fit_jobs(datasets, model_names, replicates):
        # Fit one model to either all replicates or a single dataset.
        try:
            result = fit_model(
                fit_datasets,
                model_name,
                logk_starts,
                max_nfev=max_nfev,
                bootstrap=bootstrap,
                bootstrap_method=bootstrap_method,
                seed=seed,
                logk_bounds=logk_bounds,
                logk_jitter=logk_jitter,
                solver_failure_mode=solver_failure_mode,
            )
        except ModelFitError as exc:
            result = _exception_failure_result(model_name, fit_datasets, exc)
        results.append(result)
    return results
