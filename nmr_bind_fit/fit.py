"""Model fitting orchestration for NMR binding titrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeResult

from .fit_bootstrap import BootstrapResult
from .fit_bootstrap import bootstrap_params as _bootstrap_params_impl
from .fit_criteria import information_criteria
from .fit_optimizer import build_logk_grid, param_bounds, select_best_multistart
from .fit_optimizer import fit_with_initial as _optimizer_fit_with_initial
from .io import Dataset
from .models import MODEL_SPECS, ModelSpec, predict_dataset, split_params_multi
from .stats import residual_diagnostics as _residual_diagnostics_impl


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
    r2_per_peak: List[float]
    bic: float
    aicc: float
    n: int
    p: int
    dof: int
    y_pred: List[np.ndarray]
    species: List
    residuals: List[np.ndarray]
    residual_diagnostics: Dict[str, float]
    bootstrap: Optional[BootstrapResult]
    penalty_count: int


class ModelFitError(RuntimeError):
    pass


_NUMERIC_EXCEPTIONS = (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


def _residual_penalty_scale(y: np.ndarray) -> float:
    vals = np.asarray(y, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 1.0
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 0:
        q75, q25 = np.percentile(finite, [75.0, 25.0])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 0:
        span = float(np.max(finite) - np.min(finite))
        scale = span
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(finite))) * 1e-2
    if not np.isfinite(scale) or scale <= 0:
        scale = 1e-3
    return float(max(scale * 8.0, 1e-3))


def _init_delta(model: ModelSpec, dataset: Dataset) -> np.ndarray:
    y = np.asarray(dataset.y, dtype=float)
    x = np.asarray(dataset.x, dtype=float)
    y0 = np.full((dataset.n_peaks,), np.nan, dtype=float)
    y1 = np.full((dataset.n_peaks,), np.nan, dtype=float)
    x0 = np.full((dataset.n_peaks,), np.nan, dtype=float)
    x1 = np.full((dataset.n_peaks,), np.nan, dtype=float)

    for peak_idx in range(dataset.n_peaks):
        mask = np.isfinite(y[:, peak_idx]) & np.isfinite(x)
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        first = int(indices[0])
        last = int(indices[-1])
        y0[peak_idx] = y[first, peak_idx]
        y1[peak_idx] = y[last, peak_idx]
        x0[peak_idx] = x[first]
        x1[peak_idx] = x[last]

    if not np.all(np.isfinite(y0) & np.isfinite(y1)):
        raise ValueError("Each ppm column must contain at least one finite value for fitting.")

    if model.name == "nb":
        slope = np.zeros_like(y0)
        span = x1 - x0
        mask = np.isfinite(span) & (span != 0)
        slope[mask] = (y1[mask] - y0[mask]) / span[mask]
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
    params: List[float] = []
    params.extend(list(logk_values))
    for ds in datasets:
        delta = _init_delta(model, ds)
        params.extend(delta.ravel().tolist())
    return np.array(params, dtype=float)


def _param_names_multi(model: ModelSpec, datasets: List[Dataset]) -> List[str]:
    names: List[str] = []
    if model.n_logk == 1:
        names.append("logK")
    elif model.n_logk == 2:
        names.extend(["logK1", "logK2"])
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
    logk, deltas = split_params_multi(params, model, datasets)
    residuals = []
    for ds, delta in zip(datasets, deltas):
        valid_mask = np.isfinite(ds.y)
        n_valid = int(np.count_nonzero(valid_mask))
        if n_valid == 0:
            continue
        penalty_scale = _residual_penalty_scale(ds.y)

        y_pred = None
        try:
            y_pred, _ = predict_dataset(model, ds, logk, delta, solver_failure_mode=solver_failure_mode)
        except _NUMERIC_EXCEPTIONS:
            y_pred = None

        if solver_failure_mode == "fail-fast" and model.is_binding:
            retry_needed = y_pred is None
            if y_pred is not None:
                retry_needed = not np.all(np.isfinite(y_pred[valid_mask]))
            if retry_needed:
                y_pred_retry = None
                try:
                    y_pred_retry, _ = predict_dataset(model, ds, logk, delta, solver_failure_mode="continue")
                except _NUMERIC_EXCEPTIONS:
                    y_pred_retry = None

                best_pred = y_pred
                best_count = -1
                if best_pred is not None:
                    best_count = int(np.count_nonzero(np.isfinite(best_pred[valid_mask])))
                if y_pred_retry is not None:
                    retry_count = int(np.count_nonzero(np.isfinite(y_pred_retry[valid_mask])))
                    if retry_count > best_count:
                        best_pred = y_pred_retry
                y_pred = best_pred

        if y_pred is None:
            if penalty_counter is not None:
                penalty_counter["count"] = penalty_counter.get("count", 0) + 1
            residuals.append(np.full((n_valid,), penalty_scale, dtype=float))
            continue

        res = ds.y - y_pred
        res_valid = res[valid_mask]
        finite_mask = np.isfinite(res_valid)
        ds_residual = np.full((n_valid,), penalty_scale, dtype=float)
        if np.any(finite_mask):
            ds_residual[finite_mask] = res_valid[finite_mask]
        residuals.append(ds_residual)

        failed_count = n_valid - int(np.count_nonzero(finite_mask))
        if failed_count > 0:
            if penalty_counter is not None:
                penalty_counter["count"] = penalty_counter.get("count", 0) + 1
    if not residuals:
        return np.array([], dtype=float)
    return np.concatenate(residuals)


def _predict_all(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
    solver_failure_mode: str = "fail-fast",
) -> Tuple[List[np.ndarray], List, List[np.ndarray]]:
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
    rss = 0.0
    for res in residuals:
        rss += float(np.nansum(res**2))
    return rss


def _r2_score(datasets: List[Dataset], y_pred_list: List[np.ndarray]) -> Tuple[float, List[float]]:
    r2_per_peak: List[float] = []
    for ds, y_pred in zip(datasets, y_pred_list):
        for peak_idx in range(ds.n_peaks):
            y_obs_col = ds.y[:, peak_idx]
            y_pred_col = y_pred[:, peak_idx]
            mask = np.isfinite(y_obs_col) & np.isfinite(y_pred_col)
            if not np.any(mask):
                continue
            obs = y_obs_col[mask]
            pred = y_pred_col[mask]
            ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
            if ss_tot <= 0:
                continue
            ss_res = float(np.sum((obs - pred) ** 2))
            r2_per_peak.append(float(1.0 - ss_res / ss_tot))
    if not r2_per_peak:
        return float("nan"), []
    return float(np.mean(r2_per_peak)), r2_per_peak


def _fit_with_initial(
    model: ModelSpec,
    datasets: List[Dataset],
    params0: np.ndarray,
    max_nfev: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    solver_failure_mode: str = "fail-fast",
) -> Tuple[np.ndarray, OptimizeResult]:
    """Bind fit.py's residual-vector policy to the lower-level optimizer."""
    return _optimizer_fit_with_initial(
        model=model,
        datasets=datasets,
        params0=params0,
        residual_vector_fn=_residual_vector,
        max_nfev=max_nfev,
        bounds=bounds,
        solver_failure_mode=solver_failure_mode,
    )


def _total_observations(datasets: List[Dataset]) -> int:
    return int(sum(np.count_nonzero(np.isfinite(ds.y)) for ds in datasets))


def _failed_fit_result(
    model: ModelSpec,
    datasets: List[Dataset],
    params: np.ndarray,
    param_names: List[str],
    message: str,
    species: Optional[List] = None,
) -> FitResult:
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
        r2_per_peak=[],
        bic=float("nan"),
        aicc=float("nan"),
        n=n,
        p=p,
        dof=dof,
        y_pred=[],
        species=species if species is not None else [],
        residuals=[],
        residual_diagnostics={},
        bootstrap=None,
        penalty_count=0,
    )


def _nonfinite_prediction_message(datasets: List[Dataset], species_list: List[object]) -> str:
    fail_points = 0
    total_points = 0
    for ds, species in zip(datasets, species_list):
        stats = getattr(species, "solver_stats", None)
        if stats is not None:
            fail_points += int(getattr(stats, "fail", 0))
            total_points += int(getattr(stats, "points", ds.n_points))
    message = "Equilibrium solver produced non-finite predictions."
    if total_points > 0:
        message = f"{message} Failed points: {fail_points}/{total_points}."
    return message


def _build_successful_fit_result(
    model: ModelSpec,
    datasets: List[Dataset],
    best_params: np.ndarray,
    best_res: OptimizeResult,
    bootstrap: int,
    bootstrap_method: str,
    bootstrap_ci_method: str,
    seed: Optional[int],
    logk_bounds: Optional[Tuple[float, float]],
    logk_jitter: float,
    max_nfev: int,
    solver_failure_mode: str,
    compute_residual_diagnostics: bool,
) -> FitResult:
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
    r2, r2_per_peak = _r2_score(datasets, y_pred_list)
    bic, aicc = information_criteria(datasets, residuals, p)
    diag: Dict[str, float] = {}
    if compute_residual_diagnostics:
        finite_residuals = []
        for res in residuals:
            for peak_idx in range(res.shape[1]):
                series = res[:, peak_idx]
                finite_residuals.append(series[np.isfinite(series)])
        finite_residuals = [series for series in finite_residuals if series.size > 0]
        if finite_residuals:
            stacked = np.concatenate(finite_residuals)
            diag = _residual_diagnostics_impl(
                stacked,
                include_durbin_watson=len(finite_residuals) == 1,
            )

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
            ci_method=bootstrap_ci_method,
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
        r2_per_peak=r2_per_peak,
        bic=bic,
        aicc=aicc,
        n=n,
        p=p,
        dof=dof,
        y_pred=y_pred_list,
        species=species_list,
        residuals=residuals,
        residual_diagnostics=diag,
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
    bootstrap_ci_method: str = "percentile",
    seed: Optional[int] = None,
    logk_bounds: Optional[Tuple[float, float]] = None,
    logk_jitter: float = 0.1,
    solver_failure_mode: str = "fail-fast",
    residual_diagnostics: bool = False,
) -> FitResult:
    model = MODEL_SPECS[model_name]
    try:
        logk_grid = build_logk_grid(model, logk_starts, logk_bounds)
    except ValueError as exc:
        raise ModelFitError(str(exc)) from exc

    best_params, best_res = select_best_multistart(
        model,
        datasets,
        logk_grid,
        max_nfev=max_nfev,
        logk_bounds=logk_bounds,
        build_initial_params_fn=_build_initial_params,
        fit_with_initial_fn=_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
        solver_failure_mode=solver_failure_mode,
    )

    if best_params is None or best_res is None:
        raise ModelFitError(f"Fit failed for model {model_name}")

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
            bootstrap_ci_method=bootstrap_ci_method,
            seed=seed,
            logk_bounds=logk_bounds,
            logk_jitter=logk_jitter,
            max_nfev=max_nfev,
            solver_failure_mode=solver_failure_mode,
            compute_residual_diagnostics=residual_diagnostics,
        )
    except RuntimeError as exc:
        raise ModelFitError(str(exc)) from exc


def bootstrap_params(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
    n_boot: int,
    method: str,
    seed: Optional[int],
    logk_bounds: Optional[Tuple[float, float]],
    logk_jitter: float,
    ci_method: str = "percentile",
    max_nfev: int = 5000,
    solver_failure_mode: str = "fail-fast",
) -> BootstrapResult:
    """Run bootstrap resampling using fit.py prediction and optimizer policies."""
    return _bootstrap_params_impl(
        params=params,
        model=model,
        datasets=datasets,
        n_boot=n_boot,
        method=method,
        ci_method=ci_method,
        seed=seed,
        logk_bounds=logk_bounds,
        logk_jitter=logk_jitter,
        predict_all_fn=_predict_all,
        fit_with_initial_fn=_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
        max_nfev=max_nfev,
        solver_failure_mode=solver_failure_mode,
    )


def _exception_failure_result(
    model_name: str,
    datasets: List[Dataset],
    exc: Exception,
) -> FitResult:
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
    replicates: bool = False,
    max_nfev: int = 5000,
    bootstrap: int = 0,
    bootstrap_method: str = "residual",
    seed: Optional[int] = None,
    logk_bounds: Optional[Tuple[float, float]] = None,
    logk_jitter: float = 0.1,
    solver_failure_mode: str = "fail-fast",
    bootstrap_ci_method: str = "percentile",
    residual_diagnostics: bool = False,
) -> List[FitResult]:
    """Fit candidate models and return one :class:`FitResult` per fit job.

    This is the primary programmatic entry point (mirrored by the ``nmr_bind_fit``
    command-line interface). Each requested model is fitted by multistart
    nonlinear least squares; failures are captured as unsuccessful ``FitResult``
    objects rather than raised, so the returned list always has one entry per job.

    Args:
        datasets: Datasets to fit (see :func:`nmr_bind_fit.io.load_datasets`).
        model_names: Candidate model codes, e.g. ``["11", "12", "21", "nb"]``.
        logk_starts: log10(K) multistart initial values; the grid is the
            Cartesian product across a model's binding constants.
        replicates: If True, fit all datasets simultaneously with shared binding
            constants and dataset-specific chemical shifts. If False (default),
            fit each dataset independently.
        max_nfev: Maximum optimizer function evaluations per start.
        bootstrap: Number of bootstrap refits for uncertainty; 0 disables it.
        bootstrap_method: Resampling scheme: ``"residual"``, ``"points"``, or
            ``"parametric"``.
        seed: Seed for the bootstrap random generator (None for nondeterministic).
        logk_bounds: Optional ``(low, high)`` bounds on log10(K).
        logk_jitter: Std. dev. of the log10(K) start perturbation per refit.
        solver_failure_mode: Per-point equilibrium-solver policy for 1:2/2:1
            models: ``"fail-fast"`` or ``"continue"``.
        bootstrap_ci_method: Confidence-interval method: ``"percentile"`` or
            ``"bca"``.
        residual_diagnostics: If True, compute informational residual
            diagnostics (Shapiro-Wilk, Durbin-Watson).

    Returns:
        One :class:`FitResult` per (dataset-group, model) job, in job order.
        Check ``FitResult.success`` before using the fitted values.
    """
    results = []
    for fit_datasets, model_name in _iter_fit_jobs(datasets, model_names, replicates):
        try:
            result = fit_model(
                fit_datasets,
                model_name,
                logk_starts,
                max_nfev=max_nfev,
                bootstrap=bootstrap,
                bootstrap_method=bootstrap_method,
                bootstrap_ci_method=bootstrap_ci_method,
                seed=seed,
                logk_bounds=logk_bounds,
                logk_jitter=logk_jitter,
                solver_failure_mode=solver_failure_mode,
                residual_diagnostics=residual_diagnostics,
            )
        except ModelFitError as exc:
            result = _exception_failure_result(model_name, fit_datasets, exc)
        results.append(result)
    return results
