"""Equilibrium solvers for binding models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq


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
    success: int = 0
    fail: int = 0
    method: str = "brentq"
    failed_indices: List[int] = field(default_factory=list)


def _normalize_failure_mode(failure_mode: str) -> str:
    if failure_mode not in {"fail-fast", "continue"}:
        raise ValueError("failure_mode must be one of: fail-fast, continue")
    return failure_mode


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
    # Use -inf for non-positive values to keep log-space formulas stable.
    if value <= 0 or not math.isfinite(value):
        return float("-inf")
    return float(math.log(value))


def _logsumexp(values: np.ndarray) -> float:
    # Stable log-sum-exp for combining log-populations.
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
    # Rescale log-populations so stoichiometric totals match the host mass balance.
    if h_tot <= 0 or not math.isfinite(h_tot):
        return tuple(0.0 for _ in log_species)
    log_stoich = np.log(np.asarray(stoich, dtype=float))
    log_total = _logsumexp(log_species + log_stoich)
    if not math.isfinite(log_total):
        return tuple(0.0 for _ in log_species)
    log_scale = math.log(h_tot) - log_total
    values = np.exp(log_species + log_scale)
    return tuple(float(v) for v in values)


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
    else:
        B = k1 + prod
        C = 1.0 + k1 * (h_tot - g_tot)
        D = -g_tot

        def f(g: float) -> float:
            return ((A * g + B) * g + C) * g + D

    # Adaptive lower bound: estimate minimum free guest under full saturation.
    with np.errstate(over="ignore"):
        binding_capacity = k1 * h_tot * (1.0 + 2.0 * k2 * h_tot)
    if np.isfinite(binding_capacity) and binding_capacity > 0:
        lower = max(1e-300, g_tot / binding_capacity * 1e-6)
    else:
        lower = 1e-300
    upper = max(g_tot, lower * 10.0)

    try:
        g = brentq(f, lower, upper, xtol=1e-50, rtol=1e-15)
        if stats is not None:
            stats.success += 1
    except ValueError:
        if stats is not None:
            stats.fail += 1
        raise RuntimeError("Equilibrium solver failed.")

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
        # Solve for free host using a log-space quadratic to avoid overflow.
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

    # Adaptive lower bound: estimate minimum free guest under full saturation.
    with np.errstate(over="ignore"):
        binding_capacity = k1 * h_tot * (1.0 + k2 * h_tot)
    if np.isfinite(binding_capacity) and binding_capacity > 0:
        lower = max(1e-300, g_tot / binding_capacity * 1e-6)
    else:
        lower = 1e-300
    upper = max(g_tot, lower * 10.0)

    try:
        g = brentq(f, lower, upper, xtol=1e-50, rtol=1e-15)
        if stats is not None:
            stats.success += 1
    except ValueError:
        if stats is not None:
            stats.fail += 1
        raise RuntimeError("Equilibrium solver failed.")

    # Derive species directly from the root decomposition so both
    # host and guest mass balances hold by construction of f(g)=0.
    h_raw, b_h = _h_and_bh_from_g(g)
    h = max(0.0, h_raw)
    hg = max(0.0, b_h - h_raw)
    h2g = max(0.0, 0.5 * (h_tot - b_h))

    return h, g, hg, h2g


def solve_12(
    h_tot: np.ndarray,
    g_tot: np.ndarray,
    k1: float,
    k2: float,
    failure_mode: str = "fail-fast",
) -> SpeciesResult:
    """Solve 1:2 binding across all points; aborts on the first failure."""
    mode = _normalize_failure_mode(failure_mode)
    h_tot = np.asarray(h_tot, dtype=float)
    g_tot = np.asarray(g_tot, dtype=float)
    h = np.full_like(h_tot, np.nan)
    g = np.full_like(g_tot, np.nan)
    hg = np.full_like(h_tot, np.nan)
    hg2 = np.full_like(h_tot, np.nan)

    g_prev = None
    stats = SolverStats()
    for i, (h0, g0) in enumerate(zip(h_tot, g_tot)):
        # Use the previous free-guest solution as the next initial guess.
        stats.points += 1
        try:
            h_i, g_i, hg_i, hg2_i = solve_12_point(h0, g0, k1, k2, x0=g_prev, stats=stats)
        except RuntimeError:
            stats.failed_indices.append(i)
            if mode == "continue":
                continue
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
    failure_mode: str = "fail-fast",
) -> SpeciesResult:
    """Solve 2:1 binding across all points; aborts on the first failure."""
    mode = _normalize_failure_mode(failure_mode)
    h_tot = np.asarray(h_tot, dtype=float)
    g_tot = np.asarray(g_tot, dtype=float)
    h = np.full_like(h_tot, np.nan)
    g = np.full_like(g_tot, np.nan)
    hg = np.full_like(h_tot, np.nan)
    h2g = np.full_like(h_tot, np.nan)

    g_prev = None
    stats = SolverStats()
    for i, (h0, g0) in enumerate(zip(h_tot, g_tot)):
        # Use the previous free-guest solution as the next initial guess.
        stats.points += 1
        try:
            h_i, g_i, hg_i, h2g_i = solve_21_point(h0, g0, k1, k2, x0=g_prev, stats=stats)
        except RuntimeError:
            stats.failed_indices.append(i)
            if mode == "continue":
                continue
            break
        h[i] = h_i
        g[i] = g_i
        hg[i] = hg_i
        h2g[i] = h2g_i
        g_prev = g_i

    return SpeciesResult(h=h, g=g, hg=hg, h2g=h2g, solver_stats=stats)
