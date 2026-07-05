"""Statistical diagnostics and confidence intervals."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import shapiro


def gaussian_loglik(
    residuals: np.ndarray,
) -> Tuple[float, int, int]:
    """Gaussian log-likelihood with one shared residual variance."""
    res = np.asarray(residuals, dtype=float)
    # Estimate one variance term from all finite residuals.
    mask = np.isfinite(res)
    res = res[mask]
    n = int(res.size)
    if n == 0:
        return float("nan"), 0, 0
    rss = float(np.sum(res**2))
    if not np.isfinite(rss) or rss <= 0:
        return float("nan"), n, 1
    sigma2 = rss / n
    loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    return float(loglik), n, 1


def bic_from_loglik(loglik: float, n: int, p: int) -> float:
    """Bayesian Information Criterion from log-likelihood."""
    if n <= 0 or not np.isfinite(loglik):
        return float("nan")
    return float(-2.0 * loglik + p * np.log(n))


def aicc_from_loglik(loglik: float, n: int, p: int) -> float:
    """Small-sample corrected AIC from log-likelihood."""
    if n <= 0 or not np.isfinite(loglik):
        return float("nan")
    denom = n - p - 1
    if denom <= 0:
        return float("nan")
    return float(-2.0 * loglik + 2.0 * p + (2.0 * p * (p + 1)) / denom)


def residual_diagnostics(
    residuals: np.ndarray,
    *,
    shapiro_max_n: int = 5000,
    include_durbin_watson: bool = True,
) -> Dict[str, float]:
    """Compute informational residual diagnostics (normality, autocorrelation).

    These are reporting aids, not acceptance tests. The Durbin-Watson statistic
    treats the input as a single ordered series; callers that pass pooled
    residuals should disable it because lag-1 differences at concatenation
    boundaries are not physically meaningful.
    """
    res = np.asarray(residuals, dtype=float)
    flat = res[np.isfinite(res)]
    result: Dict[str, float] = {}
    if flat.size == 0:
        return result

    result["residual_n"] = float(flat.size)

    if flat.size >= 3:
        sample = flat[: min(int(flat.size), int(shapiro_max_n))]
        sample_range = float(np.max(sample) - np.min(sample))
        if sample_range > 0.0:
            stat, p_value = shapiro(sample)
            result["shapiro_stat"] = float(stat)
            result["shapiro_p"] = float(p_value)
            result["shapiro_n"] = float(sample.size)

    if include_durbin_watson and flat.size >= 2:
        denom = float(np.sum(flat**2))
        if denom > 0:
            diff = np.diff(flat)
            dw = float(np.sum(diff**2) / denom)
            result["durbin_watson"] = dw
    return result
