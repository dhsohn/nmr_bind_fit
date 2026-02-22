from pathlib import Path

import numpy as np

import core.fit as fit
from core.io import Dataset
from core.stats import aicc_from_loglik, bic_from_loglik, gaussian_loglik


def _dataset(name: str, n_points: int, n_peaks: int) -> Dataset:
    h_tot = np.full((n_points,), 1e-3, dtype=float)
    g_tot = np.linspace(0.0, 1e-3, n_points, dtype=float)
    x = g_tot / h_tot
    y = np.zeros((n_points, n_peaks), dtype=float)
    return Dataset(
        name=name,
        path=Path(f"{name}.csv"),
        h_tot=h_tot,
        g_tot=g_tot,
        x=x,
        y=y,
        y_cols=[f"ppm{i + 1}" for i in range(n_peaks)],
        dropped_peaks=[],
    )


def test_information_criteria_uses_shared_variance_including_sigma_single_dataset():
    ds = _dataset("single", n_points=4, n_peaks=2)
    residuals = [np.array([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2], [0.05, -0.15]], dtype=float)]
    p = 5

    bic, aicc = fit._information_criteria([ds], residuals, p)

    loglik, n_obs, n_sigma = gaussian_loglik(residuals[0].ravel())
    assert n_sigma == 1
    expected_bic = bic_from_loglik(loglik, n_obs, p + n_sigma)
    expected_aicc = aicc_from_loglik(loglik, n_obs, p + n_sigma)
    np.testing.assert_allclose(bic, expected_bic)
    np.testing.assert_allclose(aicc, expected_aicc)


def test_information_criteria_uses_one_shared_variance_including_sigma_multiple_datasets():
    ds1 = _dataset("d1", n_points=3, n_peaks=1)
    ds2 = _dataset("d2", n_points=3, n_peaks=1)
    residuals = [
        np.array([[0.1], [-0.1], [0.2]], dtype=float),
        np.array([[0.3], [-0.2], [0.1]], dtype=float),
    ]
    p = 4

    bic, aicc = fit._information_criteria([ds1, ds2], residuals, p)

    stacked = np.concatenate([res.ravel() for res in residuals])
    loglik, n_obs, n_sigma = gaussian_loglik(stacked)
    assert n_sigma == 1
    expected_bic = bic_from_loglik(loglik, n_obs, p + n_sigma)
    expected_aicc = aicc_from_loglik(loglik, n_obs, p + n_sigma)
    np.testing.assert_allclose(bic, expected_bic)
    np.testing.assert_allclose(aicc, expected_aicc)
