from pathlib import Path
from types import SimpleNamespace

import numpy as np

import nmrbindfit.fit as fit
from nmrbindfit.io import Dataset
from nmrbindfit.models import MODEL_SPECS


def _make_dataset() -> Dataset:
    h0 = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0, 5e-4, 1e-3], dtype=float)
    x = g0 / h0
    y = np.array([[7.0], [7.2], [7.5]], dtype=float)
    return Dataset(
        name="test",
        path=Path("dummy.csv"),
        h_tot=h0,
        g_tot=g0,
        x=x,
        y=y,
        y_cols=["ppm1"],
        dropped_peaks=[],
    )


def test_residual_vector_uses_finite_residuals_and_pointwise_penalty(monkeypatch):
    ds = _make_dataset()
    model = MODEL_SPECS["11"]

    def fake_predict_dataset(model, dataset, logk, delta, solver_failure_mode="fail-fast"):
        y_pred = dataset.y.copy()
        y_pred[0, 0] -= 0.05
        y_pred[1, 0] = np.nan
        y_pred[2, 0] += 0.10
        return y_pred, SimpleNamespace(solver_stats=None)

    monkeypatch.setattr(fit, "predict_dataset", fake_predict_dataset)

    params = np.array([4.0, 7.0, 7.5], dtype=float)
    penalty_counter = {"count": 0}
    residual = fit._residual_vector(params, model, [ds], penalty_counter=penalty_counter)

    penalty = fit._residual_penalty_scale(ds.y)
    np.testing.assert_allclose(residual, np.array([0.05, penalty, -0.10], dtype=float))
    assert penalty_counter["count"] == 1


def test_residual_vector_retries_continue_mode_before_full_penalty(monkeypatch):
    ds = _make_dataset()
    model = MODEL_SPECS["12"]
    call_modes = []

    def fake_predict_dataset(model, dataset, logk, delta, solver_failure_mode="fail-fast"):
        call_modes.append(solver_failure_mode)
        if solver_failure_mode == "fail-fast":
            raise RuntimeError("synthetic solver failure")
        y_pred = dataset.y.copy()
        y_pred[2, 0] = np.nan
        return y_pred, SimpleNamespace(solver_stats=None)

    monkeypatch.setattr(fit, "predict_dataset", fake_predict_dataset)

    params = np.array([4.0, 3.5, 7.0, 7.3, 7.5], dtype=float)
    penalty_counter = {"count": 0}
    residual = fit._residual_vector(
        params,
        model,
        [ds],
        solver_failure_mode="fail-fast",
        penalty_counter=penalty_counter,
    )

    penalty = fit._residual_penalty_scale(ds.y)
    np.testing.assert_allclose(residual[:2], np.array([0.0, 0.0], dtype=float))
    np.testing.assert_allclose(residual[2:], np.array([penalty], dtype=float))
    assert call_modes == ["fail-fast", "continue"]
    assert penalty_counter["count"] == 1


def test_residual_vector_retries_continue_mode_on_nonfinite_prediction(monkeypatch):
    ds = _make_dataset()
    model = MODEL_SPECS["12"]
    call_modes = []

    def fake_predict_dataset(model, dataset, logk, delta, solver_failure_mode="fail-fast"):
        call_modes.append(solver_failure_mode)
        y_pred = dataset.y.copy()
        if solver_failure_mode == "fail-fast":
            y_pred[1, 0] = np.nan
        return y_pred, SimpleNamespace(solver_stats=None)

    monkeypatch.setattr(fit, "predict_dataset", fake_predict_dataset)

    params = np.array([4.0, 3.5, 7.0, 7.3, 7.5], dtype=float)
    penalty_counter = {"count": 0}
    residual = fit._residual_vector(
        params,
        model,
        [ds],
        solver_failure_mode="fail-fast",
        penalty_counter=penalty_counter,
    )

    np.testing.assert_allclose(residual, np.zeros((ds.n_points,), dtype=float))
    assert call_modes == ["fail-fast", "continue"]
    assert penalty_counter["count"] == 0


def test_residual_vector_uses_full_penalty_when_prediction_unavailable(monkeypatch):
    ds = _make_dataset()
    model = MODEL_SPECS["12"]

    def fake_predict_dataset(model, dataset, logk, delta, solver_failure_mode="fail-fast"):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(fit, "predict_dataset", fake_predict_dataset)

    params = np.array([4.0, 3.5, 7.0, 7.3, 7.5], dtype=float)
    penalty_counter = {"count": 0}
    residual = fit._residual_vector(
        params,
        model,
        [ds],
        solver_failure_mode="fail-fast",
        penalty_counter=penalty_counter,
    )

    penalty = fit._residual_penalty_scale(ds.y)
    np.testing.assert_allclose(residual, np.full((ds.n_points,), penalty, dtype=float))
    assert penalty_counter["count"] == 1
