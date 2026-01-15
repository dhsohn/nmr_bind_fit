"""Statistical diagnostics and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class FTestResult:
    f_stat: float
    p_value: float
    f_crit: float
    df_num: int
    df_den: int


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
    res = np.asarray(residuals, dtype=float)
    if sigma is None:
        if per_peak and res.ndim == 2:
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
    if n <= 0 or not np.isfinite(loglik):
        return float("nan")
    return float(-2.0 * loglik + p * np.log(n))


def f_test(rss_small: float, rss_large: float, p_small: int, p_large: int, n: int) -> Optional[FTestResult]:
    if p_large <= p_small:
        return None
    df_num = p_large - p_small
    df_den = n - p_large
    if df_den <= 0:
        return None
    num = (rss_small - rss_large) / df_num
    den = rss_large / df_den
    if den <= 0:
        return None
    f_stat = num / den
    p_val = stats.f.sf(f_stat, df_num, df_den)
    f_crit = stats.f.ppf(0.95, df_num, df_den)
    return FTestResult(
        f_stat=float(f_stat),
        p_value=float(p_val),
        f_crit=float(f_crit),
        df_num=int(df_num),
        df_den=int(df_den),
    )


def quadratic_nonlinearity(x: np.ndarray, y: np.ndarray) -> QuadraticResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
    for i, ratio in enumerate(ratios):
        if ratio > 2.0:
            significant = i + 1
            break
        if ratio > 0.3 and possible == 0:
            possible = i + 1
    if possible == 0 and significant > 0:
        possible = significant
    return SVDResult(significant=significant, possible=possible, ratios=ratios)


def covariance_from_jacobian(jac: np.ndarray, rss: float, dof: int) -> Optional[np.ndarray]:
    if dof <= 0:
        return None
    try:
        jtj = jac.T @ jac
        cov = (rss / dof) * np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return None
    return cov


def param_ci(params: np.ndarray, cov: Optional[np.ndarray], dof: int) -> Dict[str, np.ndarray]:
    if cov is None:
        return {"se": np.full_like(params, np.nan), "ci_low": np.full_like(params, np.nan), "ci_high": np.full_like(params, np.nan)}
    se = np.sqrt(np.diag(cov))
    t_val = stats.t.ppf(0.975, max(dof, 1))
    ci_low = params - t_val * se
    ci_high = params + t_val * se
    return {"se": se, "ci_low": ci_low, "ci_high": ci_high}
