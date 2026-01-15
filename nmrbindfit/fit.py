"""Model fitting and bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import itertools
import numpy as np
from scipy.optimize import least_squares

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
    rss_weighted: float
    rmse: float
    r2: float
    reduced_chi2: float
    bic: float
    aicc: float
    n: int
    p: int
    dof: int
    y_pred: List[np.ndarray]
    species: List
    residuals: List[np.ndarray]
    bootstrap: Optional[BootstrapResult]


def _init_delta(model: ModelSpec, dataset: Dataset) -> np.ndarray:
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
) -> np.ndarray:
    logk, deltas = split_params_multi(params, model, datasets)
    residuals = []
    for ds, delta in zip(datasets, deltas):
        try:
            y_pred, _ = predict_dataset(model, ds, logk, delta)
        except Exception:
            res = np.full_like(ds.y, 1e6)
            residuals.append(res.ravel())
            continue
        if not np.all(np.isfinite(y_pred)):
            res = np.full_like(ds.y, 1e6)
        else:
            res = ds.y - y_pred
            if ds.sigma is not None:
                res = res / ds.sigma
            if not np.all(np.isfinite(res)):
                res = np.full_like(ds.y, 1e6)
        residuals.append(res.ravel())
    return np.concatenate(residuals)


def _predict_all(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
) -> Tuple[List[np.ndarray], List, List[np.ndarray]]:
    logk, deltas = split_params_multi(params, model, datasets)
    y_pred_list = []
    species_list = []
    residuals = []
    for ds, delta in zip(datasets, deltas):
        y_pred, species = predict_dataset(model, ds, logk, delta)
        y_pred_list.append(y_pred)
        species_list.append(species)
        residuals.append(ds.y - y_pred)
    return y_pred_list, species_list, residuals


def _rss_values(datasets: List[Dataset], residuals: List[np.ndarray]) -> Tuple[float, float]:
    rss = 0.0
    rss_weighted = 0.0
    for ds, res in zip(datasets, residuals):
        rss += float(np.sum(res**2))
        if ds.sigma is not None:
            rss_weighted += float(np.sum((res / ds.sigma) ** 2))
        else:
            rss_weighted += float(np.sum(res**2))
    return rss, rss_weighted


def _r2_score(datasets: List[Dataset], y_pred_list: List[np.ndarray]) -> float:
    y_all = []
    y_pred_all = []
    for ds, y_pred in zip(datasets, y_pred_list):
        y_all.append(ds.y.ravel())
        y_pred_all.append(y_pred.ravel())
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
) -> Tuple[np.ndarray, least_squares]:
    res = least_squares(
        _residual_vector,
        params0,
        args=(model, datasets),
        method="trf",
        max_nfev=max_nfev,
        x_scale="jac",
        bounds=bounds,
    )
    return res.x, res


def _param_bounds(
    params0: np.ndarray,
    model: ModelSpec,
    logk_bounds: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if logk_bounds is None or model.n_logk == 0:
        return (np.full_like(params0, -np.inf), np.full_like(params0, np.inf))
    lower = np.full_like(params0, -np.inf)
    upper = np.full_like(params0, np.inf)
    lower[: model.n_logk] = logk_bounds[0]
    upper[: model.n_logk] = logk_bounds[1]
    return lower, upper


def fit_model(
    datasets: List[Dataset],
    model_name: str,
    logk_starts: Sequence[float],
    max_nfev: int = 5000,
    bootstrap: int = 0,
    bootstrap_method: str = "residual",
    seed: Optional[int] = None,
    logk_bounds: Optional[Tuple[float, float]] = None,
) -> FitResult:
    model = MODEL_SPECS[model_name]
    if model.n_logk == 0:
        logk_grid = [()]
    else:
        if logk_bounds is not None:
            logk_starts = [v for v in logk_starts if logk_bounds[0] <= v <= logk_bounds[1]]
            if not logk_starts:
                raise ValueError("No K starts within bounds.")
        logk_grid = list(itertools.product(logk_starts, repeat=model.n_logk))

    best_params = None
    best_res = None
    best_rss = None

    for logk_vals in logk_grid:
        params0 = _build_initial_params(model, datasets, logk_vals)
        bounds = _param_bounds(params0, model, logk_bounds)
        try:
            params, res = _fit_with_initial(model, datasets, params0, max_nfev=max_nfev, bounds=bounds)
        except Exception:
            continue
        rss = float(np.sum(res.fun**2))
        if best_rss is None or rss < best_rss:
            best_rss = rss
            best_params = params
            best_res = res

    if best_params is None or best_res is None:
        raise RuntimeError(f"Fit failed for model {model_name}")

    y_pred_list, species_list, residuals = _predict_all(best_params, model, datasets)
    rss, rss_weighted = _rss_values(datasets, residuals)

    n = int(sum(ds.n_points * ds.n_peaks for ds in datasets))
    p = int(len(best_params))
    dof = int(n - p)
    rmse = float(np.sqrt(rss / n)) if n > 0 else float("nan")
    r2 = _r2_score(datasets, y_pred_list)
    reduced_chi2 = float(rss_weighted / dof) if dof > 0 else float("nan")

    loglik_total = 0.0
    n_loglik = 0
    sigma_param_extra = 0
    for ds, res in zip(datasets, residuals):
        loglik, n_ll, n_sigma = gaussian_loglik(res, ds.sigma, per_peak=True)
        if not np.isfinite(loglik):
            raise RuntimeError(f"BIC calculation failed: log-likelihood is NaN for dataset {ds.name}.")
        if ds.sigma is None and n_sigma > 0:
            sigma_param_extra += n_sigma
        if np.isfinite(loglik):
            loglik_total += loglik
            n_loglik += n_ll
    bic_p = p + sigma_param_extra
    if n_loglik <= 0:
        raise RuntimeError("BIC calculation failed: no valid log-likelihood terms.")
    bic = bic_from_loglik(loglik_total, n_loglik, bic_p)
    aicc = aicc_from_loglik(loglik_total, n_loglik, bic_p)

    param_names = _param_names_multi(model, datasets)

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
        )

    return FitResult(
        model=model,
        datasets=datasets,
        params=best_params,
        param_names=param_names,
        success=bool(best_res.success),
        message=str(best_res.message),
        rss=rss,
        rss_weighted=rss_weighted,
        rmse=rmse,
        r2=r2,
        reduced_chi2=reduced_chi2,
        bic=bic,
        aicc=aicc,
        n=n,
        p=p,
        dof=dof,
        y_pred=y_pred_list,
        species=species_list,
        residuals=residuals,
        bootstrap=bootstrap_result,
    )


def _residual_bootstrap(
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> Dataset:
    idx = rng.integers(0, ds.n_points, size=ds.n_points)
    if ds.sigma is None:
        centered = residuals - np.mean(residuals, axis=0, keepdims=True)
        resampled = centered[idx, :]
        y_boot = y_pred + resampled
    else:
        std_res = residuals / ds.sigma
        std_res = std_res - np.mean(std_res, axis=0, keepdims=True)
        resampled = std_res[idx, :]
        y_boot = y_pred + resampled * ds.sigma
    return Dataset(
        name=ds.name,
        path=ds.path,
        h_tot=ds.h_tot,
        g_tot=ds.g_tot,
        x=ds.x,
        y=y_boot,
        y_cols=ds.y_cols,
        sigma=ds.sigma,
    )


def _parametric_bootstrap(
    rng: np.random.Generator,
    ds: Dataset,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> Dataset:
    if ds.sigma is None:
        scale = np.std(residuals, axis=0, ddof=1)
        scale = np.where(np.isfinite(scale), scale, 0.0)
        noise = rng.normal(0.0, 1.0, size=y_pred.shape) * scale.reshape(1, -1)
    else:
        noise = rng.normal(0.0, 1.0, size=y_pred.shape) * ds.sigma
    y_boot = y_pred + noise
    return Dataset(
        name=ds.name,
        path=ds.path,
        h_tot=ds.h_tot,
        g_tot=ds.g_tot,
        x=ds.x,
        y=y_boot,
        y_cols=ds.y_cols,
        sigma=ds.sigma,
    )


def bootstrap_params(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
    n_boot: int,
    method: str,
    seed: Optional[int],
    logk_bounds: Optional[Tuple[float, float]],
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    param_samples = []
    n_success = 0
    logk_jitter = 0.1

    y_pred_list, species_list, residuals = _predict_all(params, model, datasets)

    for _ in range(n_boot):
        boot_datasets: List[Dataset] = []
        for ds, y_pred, res in zip(datasets, y_pred_list, residuals):
            if method == "points":
                idx = rng.integers(0, ds.n_points, size=ds.n_points)
                boot = Dataset(
                    name=ds.name,
                    path=ds.path,
                    h_tot=ds.h_tot[idx],
                    g_tot=ds.g_tot[idx],
                    x=ds.x[idx],
                    y=ds.y[idx],
                    y_cols=ds.y_cols,
                    sigma=ds.sigma[idx] if ds.sigma is not None else None,
                )
            elif method == "parametric":
                boot = _parametric_bootstrap(rng, ds, y_pred, res)
            else:
                boot = _residual_bootstrap(rng, ds, y_pred, res)
            boot_datasets.append(boot)

        params0 = params.copy()
        if model.n_logk:
            jitter = rng.normal(0.0, logk_jitter, size=model.n_logk)
            params0[: model.n_logk] = params0[: model.n_logk] + jitter
            if logk_bounds is not None:
                params0[: model.n_logk] = np.clip(
                    params0[: model.n_logk], logk_bounds[0], logk_bounds[1]
                )
        bounds = _param_bounds(params0, model, logk_bounds)
        try:
            params_fit, _ = _fit_with_initial(model, boot_datasets, params0, max_nfev=2000, bounds=bounds)
        except Exception:
            continue
        param_samples.append(params_fit)
        n_success += 1

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
) -> List[FitResult]:
    results = []
    if replicates:
        for model_name in model_names:
            results.append(
                fit_model(
                    datasets,
                    model_name,
                    logk_starts,
                    max_nfev=max_nfev,
                    bootstrap=bootstrap,
                    bootstrap_method=bootstrap_method,
                    seed=seed,
                    logk_bounds=logk_bounds,
                )
            )
    else:
        for ds in datasets:
            for model_name in model_names:
                results.append(
                    fit_model(
                        [ds],
                        model_name,
                        logk_starts,
                        max_nfev=max_nfev,
                        bootstrap=bootstrap,
                        bootstrap_method=bootstrap_method,
                        seed=seed,
                        logk_bounds=logk_bounds,
                    )
                )
    return results
