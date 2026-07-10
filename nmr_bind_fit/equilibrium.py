"""Equilibrium solvers for binding models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

_ROOT_XTOL_REL = 1e-13
_ROOT_RTOL = 8.0 * np.finfo(float).eps
_ROOT_MAXITER = 200


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


def _record_solver_success(stats: Optional[SolverStats]) -> None:
    if stats is not None:
        stats.success += 1


def _raise_solver_failure(
    stats: Optional[SolverStats], cause: Optional[BaseException] = None
) -> None:
    if stats is not None:
        stats.fail += 1
    error = RuntimeError("Equilibrium solver failed.")
    if cause is None:
        raise error
    raise error from cause


def _validate_point_inputs(
    h_tot: float,
    g_tot: float,
    k1: float,
    k2: float,
    stats: Optional[SolverStats],
) -> Tuple[float, float, float, float]:
    try:
        values = tuple(float(value) for value in (h_tot, g_tot, k1, k2))
    except (TypeError, ValueError) as exc:
        _raise_solver_failure(stats, exc)

    h_value, g_value, k1_value, k2_value = values
    if (
        not all(math.isfinite(value) for value in values)
        or h_value < 0.0
        or g_value < 0.0
        or k1_value <= 0.0
        or k2_value <= 0.0
    ):
        _raise_solver_failure(stats, ValueError("Invalid equilibrium inputs."))
    return h_value, g_value, k1_value, k2_value


def _solve_free_guest_root(
    residual: Callable[[float], float],
    g_tot: float,
    stats: Optional[SolverStats],
    tolerance_scale: Optional[float] = None,
) -> float:
    """Solve within the physical free-guest interval, including endpoint roots."""
    lower = 0.0
    upper = float(g_tot)
    smallest = float(np.nextafter(0.0, 1.0))
    scale = upper if tolerance_scale is None else min(upper, tolerance_scale)
    xtol = max(smallest, _ROOT_XTOL_REL * max(smallest, scale))

    try:
        f_lower = float(residual(lower))
        f_upper = float(residual(upper))
        if not math.isfinite(f_lower) or not math.isfinite(f_upper):
            raise ValueError("Non-finite equilibrium residual at bracket endpoint.")
        if f_lower == 0.0:
            root = lower
        elif f_upper == 0.0:
            root = upper
        else:
            if f_lower > 0.0 or f_upper < 0.0:
                raise ValueError("Equilibrium root is not bracketed physically.")
            root = float(
                brentq(
                    residual,
                    lower,
                    upper,
                    xtol=xtol,
                    rtol=_ROOT_RTOL,
                    maxiter=_ROOT_MAXITER,
                )
            )
        if not math.isfinite(root) or root < lower or root > upper:
            raise RuntimeError("Equilibrium root lies outside the physical bracket.")
    except (FloatingPointError, OverflowError, RuntimeError, ValueError) as exc:
        _raise_solver_failure(stats, exc)

    _record_solver_success(stats)
    return root


def _free_guest_tolerance_scale(g_tot: float, log_binding_capacity: float) -> float:
    """Estimate the free-guest scale without moving the physical lower bound."""
    if log_binding_capacity <= 0.0:
        return g_tot
    smallest = float(np.nextafter(0.0, 1.0))
    log_scale = math.log(g_tot) - log_binding_capacity
    return math.exp(max(math.log(smallest), log_scale))


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
    stats: Optional[SolverStats] = None,
) -> Tuple[float, float, float, float]:
    """Solve 1:2 binding for a single point via free-guest root finding."""
    h_tot, g_tot, k1, k2 = _validate_point_inputs(h_tot, g_tot, k1, k2, stats)
    if h_tot == 0.0 and g_tot == 0.0:
        _record_solver_success(stats)
        return 0.0, 0.0, 0.0, 0.0
    if g_tot == 0.0:
        _record_solver_success(stats)
        return float(h_tot), 0.0, 0.0, 0.0

    logk1 = _log_or_neg_inf(k1)
    logk2 = _log_or_neg_inf(k2)
    log_h_tot = _log_or_neg_inf(h_tot)

    def species_from_g(g: float) -> Tuple[float, float, float]:
        logg = _log_or_neg_inf(g)
        log_species = np.array(
            [0.0, logk1 + logg, logk1 + logk2 + 2.0 * logg],
            dtype=float,
        )
        return _scale_species_from_logs(log_species, h_tot, (1.0, 1.0, 1.0))

    def residual(g: float) -> float:
        _, hg, hg2 = species_from_g(g)
        return (g - g_tot) + hg + 2.0 * hg2

    log_capacity = logk1 + log_h_tot + float(
        np.logaddexp(0.0, math.log(2.0) + logk2 + log_h_tot)
    )
    tolerance_scale = _free_guest_tolerance_scale(g_tot, log_capacity)
    g = _solve_free_guest_root(residual, g_tot, stats, tolerance_scale)
    h, hg, hg2 = species_from_g(g)

    return h, g, hg, hg2


def solve_21_point(
    h_tot: float,
    g_tot: float,
    k1: float,
    k2: float,
    stats: Optional[SolverStats] = None,
) -> Tuple[float, float, float, float]:
    """Solve 2:1 binding for a single point via free-guest root finding."""
    h_tot, g_tot, k1, k2 = _validate_point_inputs(h_tot, g_tot, k1, k2, stats)
    if h_tot == 0.0 and g_tot == 0.0:
        _record_solver_success(stats)
        return 0.0, 0.0, 0.0, 0.0
    if g_tot == 0.0:
        _record_solver_success(stats)
        return float(h_tot), 0.0, 0.0, 0.0

    logk1 = math.log(k1)
    logk2 = math.log(k2)
    log_h_tot = _log_or_neg_inf(h_tot)

    def _log_h_from_g(g: float) -> float:
        # Solve for free host using a log-space quadratic to avoid overflow.
        if g <= 0.0 or h_tot == 0.0:
            return log_h_tot
        logg = math.log(g)
        log_b = float(np.logaddexp(0.0, logk1 + logg))
        log_c = math.log(8.0) + logk1 + logk2 + log_h_tot + logg
        log_sqrt = 0.5 * float(np.logaddexp(2.0 * log_b, log_c))
        log_two_h = math.log(2.0) + log_h_tot
        log_denom = float(np.logaddexp(log_b, log_sqrt))
        return log_two_h - log_denom

    def species_from_g(g: float) -> Tuple[float, float, float]:
        logg = _log_or_neg_inf(g)
        logh = _log_h_from_g(g)
        log_species = np.array(
            [logh, logk1 + logh + logg, logk1 + logk2 + 2.0 * logh + logg],
            dtype=float,
        )
        return _scale_species_from_logs(log_species, h_tot, (1.0, 1.0, 2.0))

    def residual(g: float) -> float:
        _, hg, h2g = species_from_g(g)
        return (g - g_tot) + hg + h2g

    log_capacity = logk1 + log_h_tot + float(
        np.logaddexp(0.0, logk2 + log_h_tot)
    )
    tolerance_scale = _free_guest_tolerance_scale(g_tot, log_capacity)
    g = _solve_free_guest_root(residual, g_tot, stats, tolerance_scale)

    h, hg, h2g = species_from_g(g)

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
    k1, k2 = _validate_positive_finite_constants(k1, k2)
    h_tot, g_tot = _validate_total_arrays(h_tot, g_tot)
    h = np.full_like(h_tot, np.nan)
    g = np.full_like(g_tot, np.nan)
    hg = np.full_like(h_tot, np.nan)
    hg2 = np.full_like(h_tot, np.nan)

    stats = SolverStats()
    for i, (h0, g0) in enumerate(zip(h_tot, g_tot)):
        stats.points += 1
        fail_before = stats.fail
        try:
            h_i, g_i, hg_i, hg2_i = solve_12_point(h0, g0, k1, k2, stats=stats)
        except RuntimeError:
            if stats.fail == fail_before:
                stats.fail += 1
            stats.failed_indices.append(i)
            if mode == "continue":
                continue
            raise
        h[i] = h_i
        g[i] = g_i
        hg[i] = hg_i
        hg2[i] = hg2_i

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
    k1, k2 = _validate_positive_finite_constants(k1, k2)
    h_tot, g_tot = _validate_total_arrays(h_tot, g_tot)
    h = np.full_like(h_tot, np.nan)
    g = np.full_like(g_tot, np.nan)
    hg = np.full_like(h_tot, np.nan)
    h2g = np.full_like(h_tot, np.nan)

    stats = SolverStats()
    for i, (h0, g0) in enumerate(zip(h_tot, g_tot)):
        stats.points += 1
        fail_before = stats.fail
        try:
            h_i, g_i, hg_i, h2g_i = solve_21_point(h0, g0, k1, k2, stats=stats)
        except RuntimeError:
            if stats.fail == fail_before:
                stats.fail += 1
            stats.failed_indices.append(i)
            if mode == "continue":
                continue
            raise
        h[i] = h_i
        g[i] = g_i
        hg[i] = hg_i
        h2g[i] = h2g_i

    return SpeciesResult(h=h, g=g, hg=hg, h2g=h2g, solver_stats=stats)
