import numpy as np
import pandas as pd
import pytest

from nmrbindfit.io import load_dataset


def test_load_dataset_drops_incomplete_ppm_column(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "host": [1e-3, 1e-3, 1e-3],
            "guest": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
            "ppm2": [8.1, np.nan, 8.3],
        }
    )
    df.to_csv(path, index=False)

    with pytest.warns(RuntimeWarning, match="Dropping ppm columns with missing values: ppm2"):
        ds = load_dataset(path, host_col="host", guest_col="guest", ppm_cols=["ppm1", "ppm2"])

    assert ds.y_cols == ["ppm1"]
    assert ds.dropped_peaks == ["ppm2"]
    assert ds.y.shape == (3, 1)


def test_load_dataset_drops_rows_missing_required_columns(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "host": [1e-3, np.nan, 1e-3],
            "guest": [0.0, 5e-4, 1e-3],
            "ppm1": [7.1, 7.2, 7.3],
        }
    )
    df.to_csv(path, index=False)

    ds = load_dataset(
        path,
        host_col="host",
        guest_col="guest",
        ppm_cols=["ppm1"],
    )

    assert ds.n_points == 2
    assert ds.y.shape == (2, 1)
