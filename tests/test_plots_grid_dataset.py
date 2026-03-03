from pathlib import Path

import numpy as np

from core.io import Dataset
from core.plots import _grid_dataset


def _dataset(h_tot: np.ndarray) -> Dataset:
    x = np.array([0.0, 0.5, 1.0], dtype=float)
    g_tot = x * h_tot
    y = np.column_stack([np.linspace(7.0, 7.3, 3)])
    return Dataset(
        name="sample",
        path=Path("sample.csv"),
        h_tot=h_tot,
        g_tot=g_tot,
        x=x,
        y=y,
        y_cols=["ppm1"],
        dropped_peaks=[],
    )


def test_grid_dataset_uses_constant_host_when_cv_is_low():
    ds = _dataset(np.array([1.0e-3, 1.0e-3, 1.0e-3], dtype=float))

    grid = _grid_dataset(ds, n=7)

    assert np.allclose(grid.h_tot, np.median(ds.h_tot))
    np.testing.assert_allclose(grid.g_tot, grid.x * grid.h_tot)


def test_grid_dataset_interpolates_host_when_cv_is_high():
    ds = _dataset(np.array([1.0e-3, 2.0e-3, 1.0e-3], dtype=float))

    grid = _grid_dataset(ds, n=5)

    assert not np.allclose(grid.h_tot, np.median(ds.h_tot))
    np.testing.assert_allclose([grid.h_tot[0], grid.h_tot[2], grid.h_tot[-1]], [1.0e-3, 2.0e-3, 1.0e-3])
    np.testing.assert_allclose(grid.g_tot, grid.x * grid.h_tot)
