import numpy as np
import pytest

from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS, predict_dataset, split_params_multi


def test_predict_shape_11():
    # Sanity check prediction shape for the 1:1 model.
    h0 = np.array([1e-3, 1e-3, 1e-3])
    g0 = np.array([0.0, 5e-4, 1e-3])
    x = g0 / h0
    y = np.column_stack([np.linspace(7.0, 7.5, 3), np.linspace(8.0, 8.2, 3)])
    ds = Dataset(
        name="test",
        path=__import__("pathlib").Path("dummy.csv"),
        h_tot=h0,
        g_tot=g0,
        x=x,
        y=y,
        y_cols=["ppm1", "ppm2"],
        dropped_peaks=[],
    )
    model = MODEL_SPECS["11"]
    logk = np.array([4.0])
    delta = np.array([[7.0, 7.5], [8.0, 8.2]])
    y_pred, species = predict_dataset(model, ds, logk, delta)
    assert y_pred.shape == y.shape


def test_split_params_rejects_trailing_values():
    ds = Dataset(
        name="test",
        path=__import__("pathlib").Path("dummy.csv"),
        h_tot=np.array([1e-3]),
        g_tot=np.array([0.0]),
        x=np.array([0.0]),
        y=np.array([[7.0]]),
        y_cols=["ppm1"],
        dropped_peaks=[],
    )
    model = MODEL_SPECS["11"]

    with pytest.raises(ValueError, match="unused trailing"):
        split_params_multi(np.array([4.0, 7.0, 7.5, 99.0]), model, [ds])


def test_predict_rejects_nonfinite_logk():
    ds = Dataset(
        name="test",
        path=__import__("pathlib").Path("dummy.csv"),
        h_tot=np.array([1e-3]),
        g_tot=np.array([0.0]),
        x=np.array([0.0]),
        y=np.array([[7.0]]),
        y_cols=["ppm1"],
        dropped_peaks=[],
    )
    model = MODEL_SPECS["11"]
    delta = np.array([[7.0, 7.5]])

    with pytest.raises(ValueError, match="logK"):
        predict_dataset(model, ds, np.array([np.inf]), delta)
