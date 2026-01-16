"""Equilibrium solvers for binding models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class SpeciesResult:
    h: np.ndarray
    g: np.ndarray
    hg: np.ndarray
    hg2: Optional[np.ndarray] = None
    h2g: Optional[np.ndarray] = None
    solver_stats: Optional["SolverStats"] = None


@dataclass
class SolverStats:
    points: int = 0
    newton_success: int = 0
    newton_fail: int = 0
    newton_max_iter: int = 0
    fallback_success: int = 0
    fallback_fail: int = 0
    fallback_method: str = "bisection"


def solve_11(h_tot: np.ndarray, g_tot: np.ndarray, k: float) -> SpeciesResult:
    """Closed form 1:1 solution using the quadratic mass-balance equation."""
    h_tot = np.asarray(h_tot, dtype=float)
    g_tot = np.asarray(g_tot, dtype=float)
    k = float(k)

    if k <= 0:
        raise ValueError("K must be positive.")

    # Quadratic in [HG], written to avoid catastrophic cancellation.
    term = h_tot + g_tot + 1.0 / k
    discr = term**2 - 4.0 * h_tot * g_tot
    discr = np.maximum(discr, 0.0)
    hg = 0.5 * (term - np.sqrt(discr))
    h = h_tot - hg
    g = g_tot - hg
    return SpeciesResult(h=h, g=g, hg=hg)


def _log_or_neg_inf(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return float("-inf")
    return float(math.log(value))


def _logsumexp(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return float("-inf")
    max_val = float(np.max(vals[finite]))
    return float(max_val + np.log(np.sum(np.exp(vals[finite] - max_val))))


def _scale_species_from_logs(
    log_species: np.ndarray,
    h_tot: float,
    stoich: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Scale log-populations so that weighted host total matches h_tot."""
    if h_tot <= 0 or not math.isfinite(h_tot):
        return tuple(0.0 for _ in log_species)
    log_stoich = np.log(np.asarray(stoich, dtype=float))
    log_total = _logsumexp(log_species + log_stoich)
    if not math.isfinite(log_total):
        return tuple(0.0 for _ in log_species)
    log_scale = math.log(h_tot) - log_total
    values = np.exp(log_species + log_scale)
    return tuple(float(v) for v in values)


def _newton_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    lower: float,
    upper: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[Optional[float], str]:
    """Newton-Raphson with simple safeguards for a 1D root."""
    x = float(np.clip(x0, lower, upper))
    for _ in range(max_iter):
        fx = f(x)
        if not np.isfinite(fx):
            return None, "nonfinite"
        if abs(fx) < tol:
            return x, "success"
        dfx = df(x)
        if not np.isfinite(dfx) or dfx == 0:
            return None, "derivative"
        step = fx / dfx
        x_new = x - step
        if not np.isfinite(x_new) or x_new <= lower or x_new >= upper:
            x_new = x - 0.5 * step
        if not np.isfinite(x_new) or x_new <= lower or x_new >= upper:
            x_new = 0.5 * (x + np.clip(x_new, lower, upper))
        if abs(x_new - x) < tol * max(1.0, abs(x)):
            fx_new = f(x_new)
            if np.isfinite(fx_new) and abs(fx_new) < tol:
                return x_new, "success"
            return None, "convergence"
        x = x_new
    return None, "max_iter"


def _solve_newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    guesses: Tuple[float, ...],
    lower: float,
    upper: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[Optional[float], str]:
    statuses = []
    for guess in guesses:
        x, status = _newton_1d(f, df, guess, lower, upper, tol=tol, max_iter=max_iter)
        if x is not None and np.isfinite(x):
            return x, "success"
        statuses.append(status)
    if "max_iter" in statuses:
        return None, "max_iter"
    return None, "fail"


def _solve_bisection(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Optional[float]:
    """Bracketed bisection used only when a sign change is available."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None
    f_low = f(lower)
    f_high = f(upper)
    if not np.isfinite(f_low) or not np.isfinite(f_high):
        return None
    if f_low == 0:
        return float(lower)
    if f_high == 0:
        return float(upper)
    if (f_low > 0 and f_high > 0) or (f_low < 0 and f_high < 0):
        return None

    lo = lower
    hi = upper
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if not np.isfinite(f_mid):
            return None
        if abs(f_mid) < tol or abs(hi - lo) < tol * max(1.0, abs(mid)):
            return float(mid)
        if (f_low >= 0 and f_mid <= 0) or (f_low <= 0 and f_mid >= 0):
            hi = mid
            f_high = f_mid
        else:
            lo = mid
            f_low = f_mid
    return None


def solve_12_point(
    h_tot: float,
    g_tot: float,
    k1: float,
    k2: float,
    x0: Optional[float] = None,
    stats: Optional[SolverStats] = None,
) -> Tuple[float, float, float, float]:
    """Solve 1:2 binding for a single point via free-guest root finding."""
    if h_tot <= 0 and g_tot <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if g_tot <= 0:
        return float(h_tot), 0.0, 0.0, 0.0

    with np.errstate(over="ignore", invalid="ignore"):
        A = k1 * k2
        prod = A * (2.0 * h_tot - g_tot)
    # Switch to a rescaled polynomial when k1*k2 overflows.
    use_scaled = not np.isfinite(A) or not np.isfinite(prod)

    if use_scaled:
        k2_inv = 1.0 / k2 if k2 != 0 else float("inf")
        k1k2_inv = k2_inv / k1 if k1 != 0 else float("inf")
        b = k2_inv + (2.0 * h_tot - g_tot)
        c = k1k2_inv + (h_tot - g_tot) * k2_inv
        d = -g_tot * k1k2_inv

        def f(g: float) -> float:
            if not np.isfinite(b) or not np.isfinite(c) or not np.isfinite(d):
                return float("nan")
            return ((g + b) * g + c) * g + d

        def df(g: float) -> float:
            if not np.isfinite(b) or not np.isfinite(c):
                return float("nan")
            return (3.0 * g + 2.0 * b) * g + c
    else:
        B = k1 + prod
        C = 1.0 + k1 * (h_tot - g_tot)
        D = -g_tot

        def f(g: float) -> float:
            return ((A * g + B) * g + C) * g + D

        def df(g: float) -> float:
            return (3.0 * A * g + 2.0 * B) * g + C

    # Enforce a small lower bound to keep log terms finite.
    lower = 1e-18
    upper = max(g_tot, lower * 10.0)
    guesses = []
    if x0 is not None:
        guesses.append(float(x0))
    guesses.extend([g_tot, g_tot * 0.5, g_tot * 0.1, upper])
    g, status = _solve_newton(f, df, tuple(guesses), lower, upper)
    if g is None:
        if stats is not None:
            stats.newton_fail += 1
            if status == "max_iter":
                stats.newton_max_iter += 1
        g = _solve_bisection(f, lower, upper)
        if g is None:
            if stats is not None:
                stats.fallback_fail += 1
            raise RuntimeError("Equilibrium solver failed.")
        if stats is not None:
            stats.fallback_success += 1
    else:
        if stats is not None:
            stats.newton_success += 1

    logk1 = _log_or_neg_inf(k1)
    logk2 = _log_or_neg_inf(k2)
    logg = _log_or_neg_inf(g)
    log_species = np.array(
        [0.0, logk1 + logg, logk1 + logk2 + 2.0 * logg],
        dtype=float,
    )
    h, hg, hg2 = _scale_species_from_logs(log_species, h_tot, (1.0, 1.0, 1.0))
    return h, g, hg, hg2


def solve_21_point(
    h_tot: float,
    g_tot: float,
    k1: float,
    k2: float,
    x0: Optional[float] = None,
    stats: Optional[SolverStats] = None,
) -> Tuple[float, float, float, float]:
    """Solve 2:1 binding for a single point via free-guest root finding."""
    if h_tot <= 0 and g_tot <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if g_tot <= 0:
        return float(h_tot), 0.0, 0.0, 0.0

    def _h_and_bh_from_g(g: float) -> Tuple[float, float]:
        if g <= 0 or h_tot <= 0:
            return float(h_tot), float(h_tot)
        b = 1.0 + k1 * g
        if not np.isfinite(b) or b <= 0:
            return 0.0, 0.0
        c = 8.0 * k1 * k2 * h_tot * g
        # Log-space evaluation stabilizes the quadratic formula at large K.
        log_b = np.log(b)
        log_c = np.log(c) if c > 0 else -np.inf
        log_discr = np.logaddexp(2.0 * log_b, log_c)
        log_sqrt = 0.5 * log_discr
        log_denom = np.logaddexp(log_b, log_sqrt)
        if not np.isfinite(log_denom):
            return 0.0, 0.0
        h = np.exp(np.log(2.0 * h_tot) - log_denom)
        log_r = log_sqrt - log_b
        if not np.isfinite(log_r):
            r = 1.0
        else:
            r = np.exp(log_r)
        b_h = (2.0 * h_tot) / (1.0 + r)
        return h, b_h

    def f(g: float) -> float:
        h, b_h = _h_and_bh_from_g(g)
        if not np.isfinite(h) or not np.isfinite(b_h):
            return float("nan")
        term1 = b_h - h
        term2 = 0.5 * (h_tot - b_h)
        return g + term1 + term2 - g_tot

    def df(g: float) -> float:
        eps = max(1e-12, 1e-6 * abs(g))
        g_low = max(lower, g - eps)
        g_high = min(upper, g + eps)
        if g_high == g_low:
            return float("nan")
        f_low = f(g_low)
        f_high = f(g_high)
        if not np.isfinite(f_low) or not np.isfinite(f_high):
            return float("nan")
        return (f_high - f_low) / (g_high - g_low)

    # Enforce a small lower bound to keep log terms finite.
    lower = 1e-18
    upper = max(g_tot, lower * 10.0)
    guesses = []
    if x0 is not None:
        guesses.append(float(x0))
    guesses.extend([g_tot, g_tot * 0.5, g_tot * 0.1, upper])
    g, status = _solve_newton(f, df, tuple(guesses), lower, upper)
    if g is None:
        if stats is not None:
            stats.newton_fail += 1
            if status == "max_iter":
                stats.newton_max_iter += 1
        g = _solve_bisection(f, lower, upper)
        if g is None:
            if stats is not None:
                stats.fallback_fail += 1
            raise RuntimeError("Equilibrium solver failed.")
        if stats is not None:
            stats.fallback_success += 1
    else:
        if stats is not None:
            stats.newton_success += 1

    h_raw, _ = _h_and_bh_from_g(g)
    logh = _log_or_neg_inf(h_raw)
    logk1 = _log_or_neg_inf(k1)
    logk2 = _log_or_neg_inf(k2)
    logg = _log_or_neg_inf(g)
    log_species = np.array(
        [logh, logk1 + logh + logg, logk1 + logk2 + 2.0 * logh + logg],
        dtype=float,
    )
    h, hg, h2g = _scale_species_from_logs(log_species, h_tot, (1.0, 1.0, 2.0))
    return h, g, hg, h2g


def solve_12(
    h_tot: np.ndarray,
    g_tot: np.ndarray,
    k1: float,
    k2: float,
) -> SpeciesResult:
    """Solve 1:2 binding across all points; aborts on the first failure."""
    h_tot = np.asarray(h_tot, dtype=float)
    g_tot = np.asarray(g_tot, dtype=float)
    h = np.full_like(h_tot, np.nan)
    g = np.full_like(g_tot, np.nan)
    hg = np.full_like(h_tot, np.nan)
    hg2 = np.full_like(h_tot, np.nan)

    g_prev = None
    stats = SolverStats()
    for i, (h0, g0) in enumerate(zip(h_tot, g_tot)):
        stats.points += 1
        try:
            h_i, g_i, hg_i, hg2_i = solve_12_point(h0, g0, k1, k2, x0=g_prev, stats=stats)
        except RuntimeError:
            break
        h[i] = h_i
        g[i] = g_i
        hg[i] = hg_i
        hg2[i] = hg2_i
        g_prev = g_i

    return SpeciesResult(h=h, g=g, hg=hg, hg2=hg2, solver_stats=stats)


def solve_21(
    h_tot: np.ndarray,
    g_tot: np.ndarray,
    k1: float,
    k2: float,
) -> SpeciesResult:
    """Solve 2:1 binding across all points; aborts on the first failure."""
    h_tot = np.asarray(h_tot, dtype=float)
    g_tot = np.asarray(g_tot, dtype=float)
    h = np.full_like(h_tot, np.nan)
    g = np.full_like(g_tot, np.nan)
    hg = np.full_like(h_tot, np.nan)
    h2g = np.full_like(h_tot, np.nan)

    g_prev = None
    stats = SolverStats()
    for i, (h0, g0) in enumerate(zip(h_tot, g_tot)):
        stats.points += 1
        try:
            h_i, g_i, hg_i, h2g_i = solve_21_point(h0, g0, k1, k2, x0=g_prev, stats=stats)
        except RuntimeError:
            break
        h[i] = h_i
        g[i] = g_i
        hg[i] = hg_i
        h2g[i] = h2g_i
        g_prev = g_i

    return SpeciesResult(h=h, g=g, hg=hg, h2g=h2g, solver_stats=stats)
