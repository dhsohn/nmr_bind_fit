import numpy as np
import pytest

import nmrbindfit.equilibrium as equilibrium
from nmrbindfit.equilibrium import solve_11, solve_12, solve_21


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


def test_solve_12_continue_mode_keeps_last_success_seed(monkeypatch):
    x0_seen = []

    def fake_solve_12_point(h_tot, g_tot, k1, k2, x0=None, stats=None):
        x0_seen.append(x0)
        if len(x0_seen) == 2:
            raise RuntimeError("synthetic failure")
        return float(h_tot), 0.25, 0.0, 0.0

    monkeypatch.setattr(equilibrium, "solve_12_point", fake_solve_12_point)

    h0 = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0, 5e-4, 1e-3], dtype=float)
    species = solve_12(h0, g0, 1e4, 1e3, failure_mode="continue")

    assert x0_seen == [None, 0.25, 0.25]
    assert species.solver_stats is not None
    assert species.solver_stats.failed_indices == [1]
    assert np.isnan(species.h[1])
    assert np.isfinite(species.h[2])


def test_solve_12_rejects_unknown_failure_mode():
    h0 = np.array([1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0, 5e-4], dtype=float)

    with pytest.raises(ValueError, match="failure_mode"):
        solve_12(h0, g0, 1e4, 1e3, failure_mode="unknown")


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
