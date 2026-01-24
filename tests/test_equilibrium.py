import numpy as np

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
    np.testing.assert_allclose(species.h + species.hg + species.hg2, h0, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(species.g + species.hg + 2 * species.hg2, g0, rtol=1e-6, atol=1e-10)


def test_solve_21_mass_balance():
    # Mass balance should hold for the 2:1 solver across points.
    h0 = np.array([1e-3, 2e-3])
    g0 = np.array([1e-3, 1e-3])
    k1 = 1e4
    k2 = 1e3
    species = solve_21(h0, g0, k1, k2)
    np.testing.assert_allclose(species.h + species.hg + 2 * species.h2g, h0, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(species.g + species.hg + species.h2g, g0, rtol=1e-6, atol=1e-10)
