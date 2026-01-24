"""Statistical diagnostics and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class QuadraticResult:
    a: float
    se: float
    ci_low: float
    ci_high: float
    t_value: float


@dataclass
class SVDResult:
    significant: int
    possible: int
    ratios: np.ndarray


def gaussian_loglik(
    residuals: np.ndarray,
    sigma: Optional[np.ndarray],
    per_peak: bool = False,
) -> Tuple[float, int, int]:
    """Gaussian log-likelihood for residuals (optionally per-peak variance)."""
    res = np.asarray(residuals, dtype=float)
    if sigma is None:
        # Estimate variance from residuals when sigma is not supplied.
        if not np.all(np.isfinite(res)):
            return float("nan"), 0, 0
        if per_peak and res.ndim == 2:
            # Estimate a separate variance for each peak when sigma is absent.
            n_peaks = int(res.shape[1])
            loglik = 0.0
            n_total = 0
            for idx in range(n_peaks):
                col = res[:, idx]
                mask = np.isfinite(col)
                col = col[mask]
                n = int(col.size)
                if n == 0:
                    continue
                rss = float(np.sum(col**2))
                sigma2 = rss / n
                if not np.isfinite(sigma2) or sigma2 <= 0:
                    sigma2 = 1e-30
                loglik += -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
                n_total += n
            if n_total == 0:
                return float("nan"), 0, 0
            return float(loglik), n_total, n_peaks

        n = int(res.size)
        if n == 0:
            return float("nan"), 0, 0
        rss = float(np.sum(res**2))
        sigma2 = rss / n
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = 1e-30
        loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
        return float(loglik), n, 1

    # Align provided sigma to residual shape and filter invalid entries.
    sig = np.asarray(sigma, dtype=float)
    if sig.shape != res.shape:
        sig = np.broadcast_to(sig, res.shape)
    mask = np.isfinite(res) & np.isfinite(sig) & (sig > 0)
    res = res[mask]
    sig = sig[mask]
    n = int(res.size)
    if n == 0:
        return float("nan"), 0, 0
    loglik = -0.5 * np.sum(np.log(2.0 * np.pi * sig * sig) + (res / sig) ** 2)
    return float(loglik), n, 0


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


def quadratic_nonlinearity(x: np.ndarray, y: np.ndarray) -> QuadraticResult:
    """Quadratic fit to diagnose curvature in y vs x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Fit a quadratic via least squares to estimate curvature.
    X = np.column_stack([x**2, x, np.ones_like(x)])
    coeff, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    a = coeff[0]
    n = len(y)
    p = 3
    dof = max(n - p, 1)
    if residuals.size == 0:
        rss = np.sum((y - X @ coeff) ** 2)
    else:
        rss = residuals[0]
    sigma2 = rss / dof
    # Use the covariance of the quadratic coefficient for confidence bounds.
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[0, 0]))
    t_val = stats.t.ppf(0.975, dof)
    ci_low = float(a - t_val * se)
    ci_high = float(a + t_val * se)
    return QuadraticResult(a=float(a), se=se, ci_low=ci_low, ci_high=ci_high, t_value=float(t_val))


def svd_diagnosis(y: np.ndarray) -> SVDResult:
    """Apply the 2x/30% singular value rule."""
    y = np.asarray(y, dtype=float)
    # rows = peaks, cols = points
    if y.ndim != 2:
        raise ValueError("SVD input must be 2D")
    if y.shape[0] < 2 or y.shape[1] < 2:
        return SVDResult(significant=0, possible=0, ratios=np.array([]))

    u, s, vt = np.linalg.svd(y, full_matrices=False)
    ratios = s[:-1] / s[1:]
    significant = 0
    possible = 0
    # Identify the first ratio crossing the 2x (significant) or 30% (possible) threshold.
    for i, ratio in enumerate(ratios):
        if ratio > 2.0:
            significant = i + 1
            break
        if ratio > 0.3 and possible == 0:
            possible = i + 1
    if possible == 0 and significant > 0:
        possible = significant
    return SVDResult(significant=significant, possible=possible, ratios=ratios)
