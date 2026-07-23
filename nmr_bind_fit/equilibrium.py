"""Equilibrium solvers for binding models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, NoReturn

import numpy as np
from scipy.optimize import brentq

_ROOT_XTOL_REL = 1e-13
_ROOT_RTOL = 8.0 * np.finfo(float).eps
_ROOT_MAXITER = 200
_ROOT_ITER_SAFETY_MARGIN = 64
_ROOT_ITER_SAFETY_FACTOR = 2


@dataclass
class SpeciesResult:
    h: np.ndarray
    g: np.ndarray
    hg: np.ndarray
    hg2: np.ndarray | None = None
    h2g: np.ndarray | None = None
    solver_stats: SolverStats | None = None


@dataclass
class SolverStats:
    points: int = 0
    success: int = 0
    fail: int = 0
    method: str = "brentq"
    failed_indices: list[int] = field(default_factory=list)


def _validate_failure_mode(failure_mode: str) -> str:
    if failure_mode not in {"fail-fast", "continue"}:
        raise ValueError("failure_mode must be one of: fail-fast, continue")
    return failure_mode


def _validate_positive_finite_constants(*values: float) -> tuple[float, ...]:
    constants = tuple(float(value) for value in values)
    if any((not math.isfinite(value)) or value <= 0.0 for value in constants):
        raise ValueError("Binding constants must be positive and finite.")
    return constants


def _validate_total_concentration_arrays(
    h_tot: np.ndarray, g_tot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h_arr = np.asarray(h_tot, dtype=float)
    g_arr = np.asarray(g_tot, dtype=float)
    if h_arr.shape != g_arr.shape:
        raise ValueError("Host and guest total arrays must have matching shape.")
    if h_arr.ndim != 1:
        raise ValueError("Host and guest total arrays must be one-dimensional.")
    return h_arr, g_arr


def _record_solver_success(stats: SolverStats | None) -> None:
    if stats is not None:
        stats.success += 1


def _raise_solver_failure(
    stats: SolverStats | None, original_error: BaseException | None = None
) -> NoReturn:
    """Record a failed solve and raise a consistent solver error."""
    if stats is not None:
        stats.fail += 1

    if original_error is not None:
        raise RuntimeError("Equilibrium solver failed.") from original_error
    raise RuntimeError("Equilibrium solver failed.")


def _validate_point_inputs(
    h_tot: float,
    g_tot: float,
    k1: float,
    k2: float,
    stats: SolverStats | None,
) -> tuple[float, float, float, float]:
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
    stats: SolverStats | None,
    tolerance_scale: float | None = None,
) -> float:
    """Solve within the physical free-guest interval, including endpoint roots."""
    lower = 0.0
    upper = float(g_tot)
    smallest = float(np.nextafter(0.0, 1.0))
    scale = upper if tolerance_scale is None else min(upper, tolerance_scale)
    xtol = max(smallest, _ROOT_XTOL_REL * max(smallest, scale))
    maxiter = _ROOT_MAXITER
    if upper > lower and xtol > 0.0:
        bisection_steps = math.ceil(max(0.0, (math.log(upper - lower) - math.log(xtol)) / math.log(2.0)))
        maxiter = max(
            maxiter,
            _ROOT_ITER_SAFETY_FACTOR * bisection_steps + _ROOT_ITER_SAFETY_MARGIN,
        )

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
                    maxiter=maxiter,
                )
            )
        if not math.isfinite(root) or root < lower or root > upper:
            raise RuntimeError("Equilibrium root lies outside the physical bracket.")
    except (FloatingPointError, OverflowError, RuntimeError, ValueError) as exc:
        _raise_solver_failure(stats, exc)

    _record_solver_success(stats)
    return root


def _solve_extreme_free_guest_log_root(
    log_residual: Callable[[float], float],
    g_tot: float,
    stats: SolverStats | None,
) -> float:
    """Solve a subnormal or unrepresentable free-guest root in log space."""
    log_smallest = math.log(float(np.nextafter(0.0, 1.0)))
    upper = math.log(g_tot)
    try:
        f_upper = float(log_residual(upper))
        if not math.isfinite(f_upper) or f_upper < 0.0:
            raise ValueError("Extreme equilibrium root is not bracketed.")

        lower = min(log_smallest, upper - 64.0)
        f_lower = float(log_residual(lower))
        step = 64.0
        for _ in range(64):
            if not math.isfinite(f_lower):
                raise ValueError("Non-finite log-space equilibrium residual.")
            if f_lower <= 0.0:
                break
            step *= 2.0
            lower = log_smallest - step
            f_lower = float(log_residual(lower))
        else:
            raise ValueError("Could not bracket extreme equilibrium root in log space.")

        root = float(
            brentq(
                log_residual,
                lower,
                upper,
                xtol=1e-12,
                rtol=_ROOT_RTOL,
                maxiter=_ROOT_MAXITER,
            )
        )
        if not math.isfinite(root):
            raise RuntimeError("Non-finite log-space equilibrium root.")
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
    """Closed-form 1:1 solution with scaled, cancellation-safe arithmetic."""
    h_tot, g_tot = _validate_total_concentration_arrays(h_tot, g_tot)
    (k,) = _validate_positive_finite_constants(k)
    if (
        not np.all(np.isfinite(h_tot))
        or not np.all(np.isfinite(g_tot))
        or np.any(h_tot < 0.0)
        or np.any(g_tot < 0.0)
    ):
        raise ValueError("Total concentrations must be non-negative and finite.")

    if k >= 1.0:
        # Scale the conventional concentration-space quadratic by its largest
        # term, then evaluate the small root via 2c/a instead of subtraction.
        inv_k = 1.0 / k
        scale = np.maximum(np.maximum(h_tot, g_tot), inv_k)
        h_scaled = h_tot / scale
        g_scaled = g_tot / scale
        inv_k_scaled = inv_k / scale
        term = h_scaled + g_scaled + inv_k_scaled
        discriminant = (
            (h_scaled - g_scaled) ** 2
            + 2.0 * inv_k_scaled * (h_scaled + g_scaled)
            + inv_k_scaled**2
        )
        denominator = term + np.sqrt(discriminant)
        hg = np.minimum(h_tot, g_tot) * (
            2.0 * np.maximum(h_scaled, g_scaled) / denominator
        )
    else:
        # For very small K, avoid forming 1/K. K*H and K*G cannot overflow
        # because K < 1 and the validated totals are finite.
        kh = k * h_tot
        kg = k * g_tot
        scale = np.maximum(np.maximum(kh, kg), 1.0)
        kh_scaled = kh / scale
        kg_scaled = kg / scale
        one_scaled = 1.0 / scale
        term = kh_scaled + kg_scaled + one_scaled
        discriminant = (
            (kh_scaled - kg_scaled) ** 2
            + 2.0 * one_scaled * (kh_scaled + kg_scaled)
            + one_scaled**2
        )
        denominator = term + np.sqrt(discriminant)
        hg = np.minimum(h_tot, g_tot) * (
            2.0 * np.maximum(kh_scaled, kg_scaled) / denominator
        )

    hg = np.minimum(np.maximum(hg, 0.0), np.minimum(h_tot, g_tot))
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
    stoich: tuple[float, ...],
) -> tuple[float, ...]:
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
    stats: SolverStats | None = None,
) -> tuple[float, float, float, float]:
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

    def species_from_logg(logg: float) -> tuple[float, float, float]:
        log_species = np.array(
            [0.0, logk1 + logg, logk1 + logk2 + 2.0 * logg],
            dtype=float,
        )
        return _scale_species_from_logs(log_species, h_tot, (1.0, 1.0, 1.0))

    def species_from_g(g: float) -> tuple[float, float, float]:
        return species_from_logg(_log_or_neg_inf(g))

    def residual(g: float) -> float:
        _, hg, hg2 = species_from_g(g)
        return (g - g_tot) + hg + 2.0 * hg2

    def log_residual(logg: float) -> float:
        _, hg, hg2 = species_from_logg(logg)
        return (math.exp(logg) - g_tot) + hg + 2.0 * hg2

    log_capacity = logk1 + log_h_tot + float(
        np.logaddexp(0.0, math.log(2.0) + logk2 + log_h_tot)
    )
    tolerance_scale = _free_guest_tolerance_scale(g_tot, log_capacity)
    smallest = float(np.nextafter(0.0, 1.0))
    use_log_root = tolerance_scale < np.finfo(float).tiny
    if g_tot > smallest and residual(smallest) >= 0.0:
        use_log_root = True
    if use_log_root:
        logg = _solve_extreme_free_guest_log_root(log_residual, g_tot, stats)
        g = math.exp(logg)
        h, hg, hg2 = species_from_logg(logg)
    else:
        g = _solve_free_guest_root(residual, g_tot, stats, tolerance_scale)
        h, hg, hg2 = species_from_g(g)

    return h, g, hg, hg2


def solve_21_point(
    h_tot: float,
    g_tot: float,
    k1: float,
    k2: float,
    stats: SolverStats | None = None,
) -> tuple[float, float, float, float]:
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

    def _log_h_from_logg(logg: float) -> float:
        # Solve for free host using a log-space quadratic to avoid overflow.
        if h_tot == 0.0:
            return log_h_tot
        log_b = float(np.logaddexp(0.0, logk1 + logg))
        log_c = math.log(8.0) + logk1 + logk2 + log_h_tot + logg
        log_sqrt = 0.5 * float(np.logaddexp(2.0 * log_b, log_c))
        log_two_h = math.log(2.0) + log_h_tot
        log_denom = float(np.logaddexp(log_b, log_sqrt))
        return log_two_h - log_denom

    def _log_h_from_g(g: float) -> float:
        if g <= 0.0:
            return log_h_tot
        return _log_h_from_logg(math.log(g))

    def species_from_logg(logg: float) -> tuple[float, float, float]:
        logh = _log_h_from_logg(logg)
        log_species = np.array(
            [logh, logk1 + logh + logg, logk1 + logk2 + 2.0 * logh + logg],
            dtype=float,
        )
        return _scale_species_from_logs(log_species, h_tot, (1.0, 1.0, 2.0))

    def species_from_g(g: float) -> tuple[float, float, float]:
        return species_from_logg(_log_or_neg_inf(g))

    def residual(g: float) -> float:
        _, hg, h2g = species_from_g(g)
        return (g - g_tot) + hg + h2g

    def log_residual(logg: float) -> float:
        _, hg, h2g = species_from_logg(logg)
        return (math.exp(logg) - g_tot) + hg + h2g

    log_capacity = logk1 + log_h_tot + float(
        np.logaddexp(0.0, logk2 + log_h_tot)
    )
    tolerance_scale = _free_guest_tolerance_scale(g_tot, log_capacity)
    smallest = float(np.nextafter(0.0, 1.0))
    use_log_root = tolerance_scale < np.finfo(float).tiny
    if g_tot > smallest and residual(smallest) >= 0.0:
        use_log_root = True
    if use_log_root:
        logg = _solve_extreme_free_guest_log_root(log_residual, g_tot, stats)
        g = math.exp(logg)
        h, hg, h2g = species_from_logg(logg)
    else:
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
    mode = _validate_failure_mode(failure_mode)
    k1, k2 = _validate_positive_finite_constants(k1, k2)
    h_tot, g_tot = _validate_total_concentration_arrays(h_tot, g_tot)
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
    mode = _validate_failure_mode(failure_mode)
    k1, k2 = _validate_positive_finite_constants(k1, k2)
    h_tot, g_tot = _validate_total_concentration_arrays(h_tot, g_tot)
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
