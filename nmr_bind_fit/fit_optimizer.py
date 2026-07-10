"""Nonlinear least-squares optimization and multistart selection."""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, Type

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from .io import Dataset
from .models import ModelSpec

MAX_JACOBIAN_CONDITION = 1e6
MIN_LOGK_RMS_SENSITIVITY = 1e-4


def _peak_response_scale(values: np.ndarray) -> float:
    """Return a response-unit-equivariant scale for one observed peak."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    span = float(np.max(finite) - np.min(finite))
    magnitude = float(np.max(np.abs(finite)))
    scale = max(span, magnitude * np.sqrt(np.finfo(float).eps))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _identifiability_scales(
    model: ModelSpec,
    datasets: List[Dataset],
    parameter_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build row and parameter scales for a dimensionless fit Jacobian."""
    row_scales: List[float] = []
    parameter_scales: List[float] = [1.0] * model.n_logk
    for ds in datasets:
        y = np.asarray(ds.y, dtype=float)
        if y.ndim != 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        peak_scales = np.array(
            [_peak_response_scale(y[:, peak_idx]) for peak_idx in range(y.shape[1])],
            dtype=float,
        )
        if not np.all(np.isfinite(peak_scales)) or np.any(peak_scales <= 0.0):
            return np.array([], dtype=float), np.array([], dtype=float)
        scale_grid = np.broadcast_to(peak_scales.reshape(1, -1), y.shape)
        row_scales.extend(scale_grid[np.isfinite(y)].tolist())
        parameter_scales.extend(
            np.repeat(peak_scales, model.n_delta_per_peak).tolist()
        )
    row_array = np.asarray(row_scales, dtype=float)
    parameter_array = np.asarray(parameter_scales, dtype=float)
    if parameter_array.shape != (parameter_count,):
        return np.array([], dtype=float), np.array([], dtype=float)
    return row_array, parameter_array


def _optimization_residual_scale(datasets: List[Dataset]) -> float:
    """Return one global response scale without changing relative residual weights."""
    scales: List[float] = []
    for ds in datasets:
        y = np.asarray(ds.y, dtype=float)
        if y.ndim != 2:
            continue
        scales.extend(_peak_response_scale(y[:, idx]) for idx in range(y.shape[1]))
    finite = np.asarray(scales, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1.0
    return float(np.max(finite))


def _scaled_jacobian(
    result: OptimizeResult,
    parameter_count: int,
    model: Optional[ModelSpec] = None,
    datasets: Optional[List[Dataset]] = None,
) -> Optional[np.ndarray]:
    jacobian = getattr(result, "jac", None)
    if jacobian is None:
        return None
    if hasattr(jacobian, "toarray"):
        jacobian = jacobian.toarray()
    values = np.asarray(jacobian, dtype=float)
    if values.ndim != 2 or values.shape[1] != parameter_count or not np.all(np.isfinite(values)):
        return None
    residual_scale = float(getattr(result, "residual_scale", 1.0))
    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        return None
    values = values * residual_scale
    if model is None or datasets is None:
        return values
    row_scales, parameter_scales = _identifiability_scales(
        model,
        datasets,
        parameter_count,
    )
    if row_scales.shape != (values.shape[0],) or parameter_scales.shape != (parameter_count,):
        return None
    return (values / row_scales.reshape(-1, 1)) * parameter_scales.reshape(1, -1)


def jacobian_rank_and_condition(
    result: OptimizeResult,
    parameter_count: int,
    model: Optional[ModelSpec] = None,
    datasets: Optional[List[Dataset]] = None,
) -> Tuple[int, float]:
    """Return rank and condition number of the dimensionless fit Jacobian."""
    values = _scaled_jacobian(result, parameter_count, model, datasets)
    if values is None:
        return 0, float("inf")
    try:
        singular_values = np.linalg.svd(values, compute_uv=False)
        rank = int(np.linalg.matrix_rank(values))
    except np.linalg.LinAlgError:
        return 0, float("inf")
    if singular_values.size == 0 or singular_values[-1] <= 0:
        return rank, float("inf")
    return rank, float(singular_values[0] / singular_values[-1])


def jacobian_logk_rms_sensitivity(
    result: OptimizeResult,
    parameter_count: int,
    n_logk: int,
    model: Optional[ModelSpec] = None,
    datasets: Optional[List[Dataset]] = None,
) -> float:
    """Return the minimum dimensionless RMS sensitivity among logK columns."""
    if n_logk == 0:
        return float("nan")
    values = _scaled_jacobian(result, parameter_count, model, datasets)
    if values is None or values.shape[0] == 0 or values.shape[1] < n_logk:
        return 0.0
    sensitivities = np.linalg.norm(values[:, :n_logk], axis=0) / np.sqrt(values.shape[0])
    if not np.all(np.isfinite(sensitivities)):
        return 0.0
    return float(np.min(sensitivities))


def optimizer_result_is_identifiable(
    result: OptimizeResult,
    parameter_count: int,
    n_logk: int,
    model: Optional[ModelSpec] = None,
    datasets: Optional[List[Dataset]] = None,
) -> bool:
    """Require a full-rank, well-conditioned fit away from logK bounds."""
    if getattr(result, "jac", None) is None:
        return True
    rank, condition = jacobian_rank_and_condition(
        result,
        parameter_count,
        model,
        datasets,
    )
    if rank < parameter_count or not np.isfinite(condition) or condition > MAX_JACOBIAN_CONDITION:
        return False
    if model is not None and datasets is not None and n_logk > 0:
        sensitivity = jacobian_logk_rms_sensitivity(
            result,
            parameter_count,
            n_logk,
            model,
            datasets,
        )
        if sensitivity < MIN_LOGK_RMS_SENSITIVITY:
            return False
    active_mask = np.asarray(getattr(result, "active_mask", np.zeros(parameter_count)), dtype=int)
    return bool(active_mask.size >= n_logk and not np.any(active_mask[:n_logk]))


def param_bounds(
    params0: np.ndarray,
    model: ModelSpec,
    logk_bounds: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build lower/upper parameter bounds, constraining only the logK entries."""
    if logk_bounds is None or model.n_logk == 0:
        return np.full_like(params0, -np.inf), np.full_like(params0, np.inf)
    lower = np.full_like(params0, -np.inf)
    upper = np.full_like(params0, np.inf)
    lower[: model.n_logk] = logk_bounds[0]
    upper[: model.n_logk] = logk_bounds[1]
    return lower, upper


def fit_with_initial(
    model: ModelSpec,
    datasets: List[Dataset],
    params0: np.ndarray,
    residual_vector_fn: Callable[..., np.ndarray],
    max_nfev: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    solver_failure_mode: str = "fail-fast",
) -> Tuple[np.ndarray, OptimizeResult]:
    """Run a single bounded least-squares fit from one initial guess."""
    penalty_counter = {"count": 0}
    residual_scale = _optimization_residual_scale(datasets)

    def residual_fn(current_params: np.ndarray, current_model: ModelSpec, current_datasets: List[Dataset]) -> np.ndarray:
        residuals = residual_vector_fn(
            current_params,
            current_model,
            current_datasets,
            solver_failure_mode=solver_failure_mode,
            penalty_counter=penalty_counter,
        )
        return residuals / residual_scale

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
    setattr(res, "residual_scale", residual_scale)
    return res.x, res


def build_logk_grid(
    model: ModelSpec,
    logk_starts: Sequence[float],
    logk_bounds: Optional[Tuple[float, float]],
) -> List[Tuple[float, ...]]:
    """Build the multistart grid of logK starting points within bounds."""
    if model.n_logk == 0:
        return [()]
    starts = list(logk_starts)
    if logk_bounds is not None:
        starts = [v for v in starts if logk_bounds[0] <= v <= logk_bounds[1]]
        if not starts:
            raise ValueError("No K starts within bounds.")
    return list(itertools.product(starts, repeat=model.n_logk))


def select_best_multistart(
    model: ModelSpec,
    datasets: List[Dataset],
    logk_grid: Sequence[Tuple[float, ...]],
    max_nfev: int,
    logk_bounds: Optional[Tuple[float, float]],
    build_initial_params_fn: Callable[[ModelSpec, List[Dataset], Sequence[float]], np.ndarray],
    fit_with_initial_fn: Callable[..., Tuple[np.ndarray, OptimizeResult]],
    param_bounds_fn: Callable[[np.ndarray, ModelSpec, Optional[Tuple[float, float]]], Tuple[np.ndarray, np.ndarray]],
    numeric_exceptions: Tuple[Type[BaseException], ...],
    solver_failure_mode: str = "fail-fast",
) -> Tuple[Optional[np.ndarray], Optional[OptimizeResult]]:
    """Fit from every grid start, preferring converged fits over failed starts."""
    best_success_params = None
    best_success_res = None
    best_success_rss = None
    best_failed_params = None
    best_failed_res = None
    best_failed_rss = None

    for logk_vals in logk_grid:
        params0 = build_initial_params_fn(model, datasets, logk_vals)
        bounds = param_bounds_fn(params0, model, logk_bounds)
        fit_kwargs = {"max_nfev": max_nfev, "bounds": bounds}
        if solver_failure_mode != "fail-fast":
            fit_kwargs["solver_failure_mode"] = solver_failure_mode
        try:
            params, res = fit_with_initial_fn(model, datasets, params0, **fit_kwargs)
        except numeric_exceptions:
            continue
        rss = float(np.sum(res.fun**2))
        if not np.isfinite(rss):
            continue
        if bool(getattr(res, "success", False)) and optimizer_result_is_identifiable(
            res,
            params.size,
            model.n_logk,
            model,
            datasets,
        ):
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
