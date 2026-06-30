from pathlib import Path

import numpy as np

import nmr_bind_fit.fit as fit
from nmr_bind_fit.io import Dataset


def _dataset(y: np.ndarray) -> Dataset:
    h_tot = np.full((y.shape[0],), 1e-3, dtype=float)
    g_tot = np.linspace(0.0, 1e-3, y.shape[0], dtype=float)
    x = g_tot / h_tot
    return Dataset(
        name="sample",
        path=Path("sample.csv"),
        h_tot=h_tot,
        g_tot=g_tot,
        x=x,
        y=y,
        y_cols=[f"ppm{i + 1}" for i in range(y.shape[1])],
        dropped_peaks=[],
    )


def test_r2_score_is_mean_of_per_peak_r2():
    y_obs = np.array(
        [
            [0.0, 100.0],
            [1.0, 101.0],
            [2.0, 102.0],
        ],
        dtype=float,
    )
    y_pred = np.array(
        [
            [0.0, 100.0],
            [1.0, 101.0],
            [1.0, 101.0],
        ],
        dtype=float,
    )
    ds = _dataset(y_obs)

    r2, r2_per_peak = fit._r2_score([ds], [y_pred])

    assert len(r2_per_peak) == 2
    np.testing.assert_allclose(r2_per_peak, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(r2, 0.5, atol=1e-12)


def test_r2_score_ignores_constant_peaks():
    y_obs = np.array(
        [
            [1.0, 10.0],
            [1.0, 11.0],
            [1.0, 12.0],
        ],
        dtype=float,
    )
    y_pred = np.array(
        [
            [0.0, 10.0],
            [0.0, 11.0],
            [0.0, 11.0],
        ],
        dtype=float,
    )
    ds = _dataset(y_obs)

    r2, r2_per_peak = fit._r2_score([ds], [y_pred])

    assert len(r2_per_peak) == 1
    np.testing.assert_allclose(r2_per_peak, [0.5], atol=1e-12)
    np.testing.assert_allclose(r2, 0.5, atol=1e-12)
