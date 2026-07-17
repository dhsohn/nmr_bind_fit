from pathlib import Path

import numpy as np

from nmr_bind_fit.io import Dataset
from nmr_bind_fit.plots import _grid_dataset, _safe_file_stems


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


def test_safe_file_stems_disambiguates_sanitized_peak_collisions():
    stems = _safe_file_stems(["ppm 1", "ppm_1", "ppm:2"])

    assert stems == ["01_ppm_1", "02_ppm_1", "ppm_2"]
    assert len(stems) == len(set(stems))


def test_safe_file_stems_avoids_collision_with_existing_prefix():
    # A distinct peak whose sanitized stem equals the auto-generated prefix of a
    # duplicated stem must still resolve to a unique filename stem so its plots
    # are not overwritten.
    stems = _safe_file_stems(["ppm", "ppm", "01 ppm"])

    assert len(stems) == len(set(stems))
    assert stems[2] not in {stems[0], stems[1]}


def test_safe_file_stems_are_bounded_and_collision_free_for_long_labels():
    stems = _safe_file_stems(
        [
            "ppm/" + "a" * 300,
            "ppm?" + "a" * 300,
            "측정값" * 200,
            "측정값" * 200,
        ]
    )

    assert len(stems) == len(set(stems))
    assert all(len(stem) <= 70 for stem in stems)
