import numpy as np
import pandas as pd
import pytest

from nmr_bind_fit.io import load_dataset, load_datasets


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

    with pytest.warns(RuntimeWarning, match="Dropping ppm columns with missing or non-finite values: ppm2"):
        ds = load_dataset(path, ppm_cols=["ppm1", "ppm2"])

    assert ds.y_cols == ["ppm1"]
    assert ds.dropped_peaks == ["ppm2"]
    assert ds.dropped_rows == 0
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
    assert ds.dropped_rows == 1
    assert ds.y.shape == (2, 1)


def test_load_dataset_rejects_when_all_required_rows_are_dropped(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "[H]t": [np.nan, np.nan],
            "[G]t": [0.0, 5e-4],
            "ppm1": [7.1, 7.2],
        }
    )
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="No rows remain after dropping rows"):
        load_dataset(path, ppm_cols=["ppm1"])


def test_load_dataset_rejects_missing_explicit_ppm_column(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "[H]t": [1e-3],
            "[G]t": [0.0],
            "ppm1": [7.1],
        }
    )
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Requested ppm columns not found: missing"):
        load_dataset(path, ppm_cols=["missing"])


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
    assert ds.dropped_rows == 0
    assert ds.y.shape == (3, 2)
    assert np.isnan(ds.y[1, 0])


def test_load_dataset_drops_ppm_column_with_inf(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
            "ppm2": [8.1, np.inf, 8.3],
        }
    )
    df.to_csv(path, index=False)

    with pytest.warns(RuntimeWarning, match="non-finite values: ppm2"):
        ds = load_dataset(path, ppm_cols=["ppm1", "ppm2"])

    assert ds.y_cols == ["ppm1"]
    assert ds.dropped_peaks == ["ppm2"]


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


def test_load_dataset_drops_missing_concentration_row_before_ppm_numeric_validation(tmp_path):
    path = tmp_path / "droppable_placeholder.csv"
    pd.DataFrame(
        {
            "[H]t": [1e-3, np.nan, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, "not measured", 7.3],
        }
    ).to_csv(path, index=False)

    ds = load_dataset(path)

    assert ds.n_points == 2
    np.testing.assert_allclose(ds.y[:, 0], [7.1, 7.3])


def test_load_dataset_rejects_ppm_columns_without_finite_observations(tmp_path):
    path = tmp_path / "no_finite_ppm.csv"
    pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3],
            "[G]t": [0.0, 1e-3],
            "ppm1": [np.nan, np.nan],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="No finite ppm observations remain"):
        load_dataset(path, missing_policy="mask")


def test_load_dataset_reports_missing_explicit_ppm_columns(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "[H]t": [1e-3],
            "[G]t": [0.0],
            "ppm1": [7.1],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Requested ppm columns not found: ppm_missing"):
        load_dataset(path, ppm_cols=["ppm1", "ppm_missing"])


def test_load_datasets_rejects_an_explicit_empty_ppm_column_list(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "[H]t": [1e-3],
            "[G]t": [0.0],
            "ppm1": [7.1],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="No ppm columns detected"):
        load_datasets([path], ppm_cols=",")


def test_load_dataset_requires_a_regular_existing_file(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        load_dataset(missing)

    directory = tmp_path / "input_dir"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        load_dataset(directory)


