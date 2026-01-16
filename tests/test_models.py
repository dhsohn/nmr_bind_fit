import numpy as np

from nmrbindfit.io import Dataset
from nmrbindfit.models import MODEL_SPECS, predict_dataset


def test_predict_shape_11():
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
        sigma=None,
        dropped_peaks=[],
    )
    model = MODEL_SPECS["11"]
    logk = np.array([4.0])
    delta = np.array([[7.0, 7.5], [8.0, 8.2]])
    y_pred, species = predict_dataset(model, ds, logk, delta)
    assert y_pred.shape == y.shape
