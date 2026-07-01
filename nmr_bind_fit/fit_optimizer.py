"""Nonlinear least-squares optimization and multistart selection."""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, Type

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from .io import Dataset
from .models import ModelSpec


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

    def residual_fn(current_params: np.ndarray, current_model: ModelSpec, current_datasets: List[Dataset]) -> np.ndarray:
        return residual_vector_fn(
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
    """Fit from every grid start and return the lowest-RSS successful result."""
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
