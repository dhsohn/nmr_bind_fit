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


@pytest.mark.parametrize("replicates", [False, True])
def test_fit_models_rejects_empty_dataset_list_consistently(replicates):
    with pytest.raises(ValueError, match="At least one dataset is required"):
        fit.fit_models([], ["11"], logk_starts=[4.0], replicates=replicates)


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
    assert params[0] == 2.05


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


def test_select_best_multistart_skips_nonfinite_rss():
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0, _DummyResult(success=True, rss=float("nan"), message="nonfinite")
        return params0, _DummyResult(success=True, rss=2.0, message="converged")

    params, res = _select_with_fake_optimizer(fake_fit_with_initial)

    assert params is not None
    assert res is not None
    assert res.message == "converged"


def test_select_best_multistart_prefers_identifiable_success():
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0, OptimizeResult(
                success=True,
                fun=np.array([1.0]),
                jac=np.zeros((3, params0.size)),
                message="rank deficient",
            )
        return params0, OptimizeResult(
            success=True,
            fun=np.array([2.0]),
            jac=np.eye(params0.size),
            message="identifiable",
        )

    params, res = _select_with_fake_optimizer(fake_fit_with_initial)

    assert params is not None
    assert res is not None
    assert res.message == "identifiable"


def test_select_best_multistart_rejects_logk_at_active_bound():
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0, OptimizeResult(
                success=True,
                fun=np.array([1.0]),
                jac=np.eye(params0.size),
                active_mask=np.array([1, 0, 0]),
                message="bound limited",
            )
        return params0, OptimizeResult(
            success=True,
            fun=np.array([2.0]),
            jac=np.eye(params0.size),
            active_mask=np.zeros(params0.size, dtype=int),
            message="interior",
        )

    params, res = _select_with_fake_optimizer(fake_fit_with_initial)

    assert params is not None
    assert res is not None
    assert res.message == "interior"


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


@pytest.mark.parametrize("start", [309.0, -400.0])
def test_fit_models_rejects_nonrepresentable_logk_starts_early(start):
    with pytest.raises(ValueError, match="positive finite binding constants"):
        fit.fit_models(
            [_make_dataset()],
            ["11"],
            logk_starts=[start],
            logk_bounds=None,
        )


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


def test_fit_model_initializes_from_finite_masked_endpoints():
    ds = Dataset(
        name="masked_endpoint",
        path=Path("dummy.csv"),
        h_tot=np.array([1e-3, 1e-3, 1e-3, 1e-3], dtype=float),
        g_tot=np.array([0.0, 3e-4, 6e-4, 1e-3], dtype=float),
        x=np.array([0.0, 0.3, 0.6, 1.0], dtype=float),
        y=np.array([[np.nan], [7.1], [7.2], [7.3]], dtype=float),
        y_cols=["ppm1"],
        dropped_peaks=[],
    )

    result = fit.fit_model(
        [ds],
        "nb",
        logk_starts=[1.0],
        max_nfev=100,
        bootstrap=0,
    )

    assert result.success is True
    assert np.all(np.isfinite(result.params))


def test_fit_models_captures_all_masked_peak_column():
    # A ppm column with no finite observations cannot seed the initial
    # parameter vector. fit_models must capture this as an unsuccessful
    # FitResult (one result per job) instead of raising and aborting the run.
    ds = Dataset(
        name="masked_peak",
        path=Path("dummy.csv"),
        h_tot=np.array([1e-3, 1e-3, 1e-3, 1e-3], dtype=float),
        g_tot=np.array([0.0, 3e-4, 6e-4, 1e-3], dtype=float),
        x=np.array([0.0, 0.3, 0.6, 1.0], dtype=float),
        y=np.column_stack(
            [
                np.array([7.0, 7.1, 7.2, 7.3], dtype=float),
                np.array([np.nan, np.nan, np.nan, np.nan], dtype=float),
            ]
        ),
        y_cols=["ppm1", "ppm2"],
        dropped_peaks=[],
    )

    results = fit.fit_models(
        datasets=[ds],
        model_names=["11"],
        logk_starts=[1.0],
        replicates=False,
        max_nfev=100,
        bootstrap=0,
        logk_bounds=(0.0, 12.0),
        logk_jitter=0.0,
    )

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].model.name == "11"
    assert "at least one finite value" in results[0].message
