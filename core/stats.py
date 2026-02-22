"""Statistical diagnostics and confidence intervals."""

from __future__ import annotations

from typing import Tuple

import numpy as np


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
    sigma2 = rss / n
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = 1e-30
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
