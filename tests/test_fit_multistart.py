from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import nmr_bind_fit.fit as fit
from nmr_bind_fit.fit_optimizer import param_bounds, select_best_multistart
from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS, ModelSpec


class _DummyResult(OptimizeResult):
    def __init__(self, success: bool, rss: float, message: str):
        super().__init__(success=success, fun=np.array([np.sqrt(rss)], dtype=float), message=message)


def _make_dataset() -> Dataset:
    h0 = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    g0 = np.array([0.0, 5e-4, 1e-3], dtype=float)
    x = g0 / h0
    y = np.column_stack([np.linspace(7.0, 7.5, 3)])
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


def _build_initial_params(model: ModelSpec, datasets: list[Dataset], logk_vals: Sequence[float]) -> np.ndarray:
    ds = datasets[0]
    return np.array([*logk_vals, ds.y[0, 0], ds.y[-1, 0]], dtype=float)


def _select_with_fake_optimizer(fit_with_initial_fn):
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    return select_best_multistart(
        model,
        [ds],
        logk_grid=[(1.0,), (2.0,)],
        max_nfev=10,
        logk_bounds=None,
        build_initial_params_fn=_build_initial_params,
        fit_with_initial_fn=fit_with_initial_fn,
        param_bounds_fn=param_bounds,
        numeric_exceptions=(RuntimeError,),
    )


def test_select_best_multistart_prefers_success_over_lower_rss_failure():
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0, _DummyResult(success=False, rss=1.0, message="failed_low_rss")
        return params0 + 0.05, _DummyResult(success=True, rss=4.0, message="converged")

    params, res = _select_with_fake_optimizer(fake_fit_with_initial)

    assert params is not None
    assert res is not None
    assert res.success is True
    assert res.message == "converged"
    assert params[0] > 2.0


def test_select_best_multistart_all_fail_uses_best_failure():
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0, _DummyResult(success=False, rss=4.0, message="failed_high_rss")
        return params0, _DummyResult(success=False, rss=1.0, message="failed_low_rss")

    params, res = _select_with_fake_optimizer(fake_fit_with_initial)

    assert params is not None
    assert res is not None
    assert res.success is False
    assert res.message == "failed_low_rss"


def test_select_best_multistart_skips_numeric_exception_and_keeps_success():
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("numeric failure")
        return params0, _DummyResult(success=True, rss=2.0, message="converged")

    params, res = _select_with_fake_optimizer(fake_fit_with_initial)

    assert params is not None
    assert res is not None
    assert res.success is True
    assert res.message == "converged"


def test_fit_models_records_failure_and_continues_single_dataset(monkeypatch):
    ds = _make_dataset()
    real_fit_model = fit.fit_model

    def fake_fit_model(datasets, model_name, logk_starts, **kwargs):
        if model_name == "11":
            raise fit.ModelFitError("forced model crash")
        return real_fit_model(datasets, model_name, logk_starts, **kwargs)

    monkeypatch.setattr(fit, "fit_model", fake_fit_model)

    results = fit.fit_models(
        datasets=[ds],
        model_names=["11", "nb"],
        logk_starts=[1.0],
        replicates=False,
        max_nfev=100,
        bootstrap=0,
        bootstrap_method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
    )

    assert len(results) == 2
    assert results[0].success is False
    assert results[0].model.name == "11"
    assert "ModelFitError: forced model crash" in results[0].message
    assert results[1].success is True
    assert results[1].model.name == "nb"


def test_fit_models_records_failure_and_continues_replicates(monkeypatch):
    ds1 = _make_dataset()
    ds2 = _make_dataset()
    ds2.name = "test2"
    real_fit_model = fit.fit_model

    def fake_fit_model(datasets, model_name, logk_starts, **kwargs):
        if model_name == "11":
            raise fit.ModelFitError("bad start grid")
        return real_fit_model(datasets, model_name, logk_starts, **kwargs)

    monkeypatch.setattr(fit, "fit_model", fake_fit_model)

    results = fit.fit_models(
        datasets=[ds1, ds2],
        model_names=["11", "nb"],
        logk_starts=[1.0],
        replicates=True,
        max_nfev=100,
        bootstrap=0,
        bootstrap_method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
    )

    assert len(results) == 2
    assert results[0].success is False
    assert results[0].model.name == "11"
    assert "ModelFitError: bad start grid" in results[0].message
    assert results[1].success is True
    assert results[1].model.name == "nb"


def test_fit_models_propagates_unexpected_exception(monkeypatch):
    ds = _make_dataset()
    real_fit_model = fit.fit_model

    def fake_fit_model(datasets, model_name, logk_starts, **kwargs):
        if model_name == "11":
            raise RuntimeError("unexpected bug")
        return real_fit_model(datasets, model_name, logk_starts, **kwargs)

    monkeypatch.setattr(fit, "fit_model", fake_fit_model)

    with pytest.raises(RuntimeError, match="unexpected bug"):
        fit.fit_models(
            datasets=[ds],
            model_names=["11", "nb"],
            logk_starts=[1.0],
            replicates=False,
            max_nfev=100,
            bootstrap=0,
            bootstrap_method="residual",
            seed=0,
            logk_bounds=None,
            logk_jitter=0.0,
        )


def test_select_best_multistart_propagates_unexpected_exception():
    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds, solver_failure_mode="fail-fast"):
        raise IndexError("unexpected coding bug")

    with pytest.raises(IndexError, match="unexpected coding bug"):
        _select_with_fake_optimizer(fake_fit_with_initial)


def test_fit_model_reports_residual_diagnostics_when_enabled():
    ds = _make_dataset()

    result = fit.fit_model(
        [ds],
        "nb",
        logk_starts=[1.0],
        max_nfev=100,
        bootstrap=0,
        residual_diagnostics=True,
    )

    assert result.success is True
    assert "residual_n" in result.residual_diagnostics
