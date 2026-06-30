import numpy as np
import pandas as pd
import pytest

from nmr_bind_fit.io import _compute_equivalents, load_dataset


def test_load_dataset_drops_incomplete_ppm_column(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
            "ppm2": [8.1, np.nan, 8.3],
        }
    )
    df.to_csv(path, index=False)

    with pytest.warns(RuntimeWarning, match="Dropping ppm columns with missing values: ppm2"):
        ds = load_dataset(path, ppm_cols=["ppm1", "ppm2"])

    assert ds.y_cols == ["ppm1"]
    assert ds.dropped_peaks == ["ppm2"]
    assert ds.y.shape == (3, 1)


def test_load_dataset_drops_rows_missing_required_columns(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "[H]t": [1e-3, np.nan, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
        }
    )
    df.to_csv(path, index=False)

    ds = load_dataset(
        path,
        ppm_cols=["ppm1"],
    )

    assert ds.n_points == 2
    assert ds.y.shape == (2, 1)


def test_load_dataset_masks_missing_ppm_values_when_requested(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, np.nan, 7.3],
            "ppm2": [8.1, 8.2, 8.3],
        }
    )
    df.to_csv(path, index=False)

    ds = load_dataset(
        path,
        ppm_cols=["ppm1", "ppm2"],
        missing_policy="mask",
    )

    assert ds.y_cols == ["ppm1", "ppm2"]
    assert ds.dropped_peaks == []
    assert ds.y.shape == (3, 2)
    assert np.isnan(ds.y[1, 0])


def test_load_dataset_reads_xlsx(tmp_path):
    pytest.importorskip("openpyxl")

    path = tmp_path / "sample.xlsx"
    df = pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
        }
    )
    df.to_excel(path, index=False)

    ds = load_dataset(path, ppm_cols=["ppm1"])

    assert ds.n_points == 3
    assert ds.y.shape == (3, 1)
    assert ds.y_cols == ["ppm1"]


def test_load_dataset_auto_detects_bracket_concentration_columns(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
        }
    )
    df.to_csv(path, index=False)

    ds = load_dataset(path)

    assert ds.n_points == 3
    assert ds.y.shape == (3, 1)
    assert ds.y_cols == ["ppm1"]


def test_load_dataset_rejects_legacy_host_guest_columns(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "Host Conc.": [1e-3, 1e-3, 1e-3],
            "Guest Conc.": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
        }
    )
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"Expected columns: \[H\]t, \[G\]t"):
        load_dataset(path)


def test_compute_equivalents_is_defensive_for_zero_host_values():
    h_tot = np.array([1e-3, 0.0], dtype=float)
    g_tot = np.array([5e-4, 1e-3], dtype=float)

    out = _compute_equivalents(h_tot, g_tot)

    np.testing.assert_allclose(out, np.array([0.5, 0.0], dtype=float))
