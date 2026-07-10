import numpy as np
import pytest

import nmr_bind_fit.equilibrium as equilibrium
from nmr_bind_fit.equilibrium import (
    SolverStats,
    solve_11,
    solve_12,
    solve_12_point,
    solve_21,
    solve_21_point,
)


def test_solve_11_mass_balance():
    # Mass balance should hold for the closed-form 1:1 solver.
    h0 = np.array([1e-3, 1e-3, 1e-3])
    g0 = np.array([0.0, 5e-4, 2e-3])
    k = 1e4
    species = solve_11(h0, g0, k)
    np.testing.assert_allclose(species.h + species.hg, h0, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(species.g + species.hg, g0, rtol=1e-6, atol=1e-12)


def test_solve_12_mass_balance():
    # Mass balance should hold for the 1:2 solver across points.
    h0 = np.array([1e-3, 1e-3])
    g0 = np.array([1e-3, 2e-3])
    k1 = 1e4
    k2 = 1e3
    species = solve_12(h0, g0, k1, k2)
    assert species.hg2 is not None
    np.testing.assert_allclose(species.h + species.hg + species.hg2, h0, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(species.g + species.hg + 2 * species.hg2, g0, rtol=1e-6, atol=1e-10)


def test_solve_21_mass_balance():
    # Mass balance should hold for the 2:1 solver across points.
    h0 = np.array([1e-3, 2e-3])
    g0 = np.array([1e-3, 1e-3])
    k1 = 1e4
    k2 = 1e3
    species = solve_21(h0, g0, k1, k2)
    assert species.h2g is not None
    np.testing.assert_allclose(species.h + species.hg + 2 * species.h2g, h0, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(species.g + species.hg + species.h2g, g0, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("k", [1e1, 1e4, 1e8, 1e12])
def test_solve_11_mass_balance_extreme_k_sweep(k):
    h0 = np.array([1e-6, 1e-4, 1e-3], dtype=float)
    g0 = np.array([1e-9, 5e-4, 2e-3], dtype=float)

    species = solve_11(h0, g0, k)

    assert np.all(np.isfinite(species.h))
    assert np.all(np.isfinite(species.g))
    assert np.all(np.isfinite(species.hg))
    np.testing.assert_allclose(species.h + species.hg, h0, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(species.g + species.hg, g0, rtol=1e-5, atol=1e-12)


def test_solve_12_continue_mode_records_failed_point_and_continues(monkeypatch):
    calls = 0

    def fake_solve_12_point(h_tot, g_tot, k1, k2, stats=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic failure")
        return float(h_tot), 0.25, 0.0, 0.0

    monkeypatch.setattr(equilibrium, "solve_12_point", fake_solve_12_point)

    h0 = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0, 5e-4, 1e-3], dtype=float)
    species = solve_12(h0, g0, 1e4, 1e3, failure_mode="continue")

    assert calls == 3
    assert species.solver_stats is not None
    assert species.solver_stats.fail == 1
    assert species.solver_stats.failed_indices == [1]
    assert np.isnan(species.h[1])
    assert np.isfinite(species.h[2])


def test_solve_12_rejects_unknown_failure_mode():
    h0 = np.array([1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0, 5e-4], dtype=float)

    with pytest.raises(ValueError, match="failure_mode"):
        solve_12(h0, g0, 1e4, 1e3, failure_mode="unknown")


@pytest.mark.parametrize("solver", [solve_12, solve_21])
@pytest.mark.parametrize("k1,k2", [(0.0, 1e3), (1e3, 0.0), (-1.0, 1e3), (np.nan, 1e3)])
def test_multivalent_solvers_reject_invalid_k_domains(solver, k1, k2):
    h0 = np.array([1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0, 5e-4], dtype=float)

    with pytest.raises(ValueError, match="positive and finite"):
        solver(h0, g0, k1, k2)


@pytest.mark.parametrize("solver", [solve_12, solve_21])
def test_multivalent_solvers_reject_mismatched_array_shapes(solver):
    h0 = np.array([1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0], dtype=float)

    with pytest.raises(ValueError, match="matching shape"):
        solver(h0, g0, 1e4, 1e3)


def test_solve_12_fail_fast_raises_on_point_failure(monkeypatch):
    def fake_solve_12_point(h_tot, g_tot, k1, k2, stats=None):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(equilibrium, "solve_12_point", fake_solve_12_point)

    h0 = np.array([1e-3], dtype=float)
    g0 = np.array([5e-4], dtype=float)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        solve_12(h0, g0, 1e4, 1e3, failure_mode="fail-fast")


@pytest.mark.parametrize("solver", [solve_12, solve_21])
def test_multivalent_solvers_keep_weak_binding_root_bracket(solver):
    h0 = np.array([1e-9], dtype=float)
    g0 = np.array([1e-3], dtype=float)

    species = solver(h0, g0, 1.0, 1.0)

    assert np.all(np.isfinite(species.g))


HIGH_K_PARAMS = [
    (1e8, 1e6),
    (1e10, 1e8),
    (1e12, 1e10),
    (1e12, 1e12),
]


@pytest.mark.parametrize("k1,k2", HIGH_K_PARAMS)
def test_solve_12_high_k_low_free_guest(k1, k2):
    h0 = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    g0 = np.array([5e-4, 1e-3, 2e-3], dtype=float)

    species = solve_12(h0, g0, k1, k2)

    assert species.hg2 is not None
    assert np.all(np.isfinite(species.h))
    assert np.all(np.isfinite(species.g))
    assert np.all(np.isfinite(species.hg))
    assert np.all(np.isfinite(species.hg2))
    np.testing.assert_allclose(
        species.h + species.hg + species.hg2, h0, rtol=1e-5, atol=1e-12
    )
    np.testing.assert_allclose(
        species.g + species.hg + 2 * species.hg2, g0, rtol=1e-5, atol=1e-12
    )


@pytest.mark.parametrize("k1,k2", HIGH_K_PARAMS)
def test_solve_21_high_k_low_free_guest(k1, k2):
    h0 = np.array([1e-3, 2e-3, 3e-3], dtype=float)
    g0 = np.array([5e-4, 1e-3, 1e-3], dtype=float)

    species = solve_21(h0, g0, k1, k2)

    assert species.h2g is not None
    assert np.all(np.isfinite(species.h))
    assert np.all(np.isfinite(species.g))
    assert np.all(np.isfinite(species.hg))
    assert np.all(np.isfinite(species.h2g))
    np.testing.assert_allclose(
        species.h + species.hg + 2 * species.h2g, h0, rtol=1e-5, atol=1e-12
    )
    np.testing.assert_allclose(
        species.g + species.hg + species.h2g, g0, rtol=1e-5, atol=1e-12
    )


@pytest.mark.parametrize("k1,k2", [(1e10, 1e8), (1e12, 1e12)])
def test_solve_12_near_saturation(k1, k2):
    h0 = np.array([1e-3, 1e-3], dtype=float)
    g0 = np.array([1e-6, 1e-3], dtype=float)

    species = solve_12(h0, g0, k1, k2)

    assert species.hg2 is not None
    assert np.all(np.isfinite(species.h))
    assert np.all(np.isfinite(species.g))
    np.testing.assert_allclose(
        species.h + species.hg + species.hg2, h0, rtol=1e-5, atol=1e-12
    )
    np.testing.assert_allclose(
        species.g + species.hg + 2 * species.hg2, g0, rtol=1e-5, atol=1e-12
    )


@pytest.mark.parametrize("k1,k2", [(1e10, 1e8), (1e12, 1e12)])
def test_solve_21_near_saturation(k1, k2):
    h0 = np.array([2e-3, 2e-3], dtype=float)
    g0 = np.array([1e-6, 1e-3], dtype=float)

    species = solve_21(h0, g0, k1, k2)

    assert species.h2g is not None
    assert np.all(np.isfinite(species.h))
    assert np.all(np.isfinite(species.g))
    np.testing.assert_allclose(
        species.h + species.hg + 2 * species.h2g, h0, rtol=1e-5, atol=1e-12
    )
    np.testing.assert_allclose(
        species.g + species.hg + species.h2g, g0, rtol=1e-5, atol=1e-12
    )


@pytest.mark.parametrize(
    "solver,host_balance,guest_balance",
    [
        (
            solve_12_point,
            lambda h, hg, complex_2: h + hg + complex_2,
            lambda g, hg, complex_2: g + hg + 2.0 * complex_2,
        ),
        (
            solve_21_point,
            lambda h, hg, complex_2: h + hg + 2.0 * complex_2,
            lambda g, hg, complex_2: g + hg + complex_2,
        ),
    ],
)
def test_point_solvers_keep_root_in_physical_interval_for_dilute_host(
    monkeypatch, solver, host_balance, guest_balance
):
    h_tot = 1e-9
    g_tot = 1e-6
    calls = []
    scipy_brentq = equilibrium.brentq

    def tracking_brentq(function, lower, upper, **kwargs):
        calls.append((lower, upper, kwargs))
        return scipy_brentq(function, lower, upper, **kwargs)

    monkeypatch.setattr(equilibrium, "brentq", tracking_brentq)

    h, g, hg, complex_2 = solver(h_tot, g_tot, 100.0, 1e5)

    assert calls
    lower, upper, options = calls[0]
    assert lower == 0.0
    assert upper == g_tot
    assert 0.0 < options["xtol"] < g_tot
    assert options["maxiter"] > 100
    assert 0.0 <= g <= g_tot
    np.testing.assert_allclose(
        host_balance(h, hg, complex_2), h_tot, rtol=1e-10, atol=1e-22
    )
    np.testing.assert_allclose(
        guest_balance(g, hg, complex_2), g_tot, rtol=1e-10, atol=1e-20
    )


def test_solve_21_converges_for_small_guest_case_that_exceeded_default_maxiter():
    h_tot = 2e-3
    g_tot = 5e-7

    h, g, hg, h2g = solve_21_point(h_tot, g_tot, 5e4, 5e2)

    assert 0.0 <= g <= g_tot
    np.testing.assert_allclose(h + hg + 2.0 * h2g, h_tot, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(g + hg + h2g, g_tot, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("solver", [solve_12_point, solve_21_point])
def test_point_solvers_handle_physical_endpoint_roots(solver):
    stats = SolverStats()

    h, g, hg, complex_2 = solver(0.0, 2e-3, 1e4, 1e3, stats=stats)

    assert (h, g, hg, complex_2) == (0.0, 2e-3, 0.0, 0.0)
    assert stats.success == 1
    assert stats.fail == 0


@pytest.mark.parametrize("solver", [solve_12_point, solve_21_point])
def test_point_solvers_count_brent_runtime_failure(monkeypatch, solver):
    stats = SolverStats()

    def fail_brentq(*args, **kwargs):
        raise RuntimeError("synthetic convergence failure")

    monkeypatch.setattr(equilibrium, "brentq", fail_brentq)

    with pytest.raises(RuntimeError, match="Equilibrium solver failed"):
        solver(1e-3, 5e-4, 1e4, 1e3, stats=stats)

    assert stats.success == 0
    assert stats.fail == 1
