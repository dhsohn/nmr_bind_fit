from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult

from nmr_bind_fit.fit_bootstrap import accept_bootstrap_fit, bootstrap_params
from nmr_bind_fit.fit_optimizer import param_bounds
from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS

_NUMERIC_EXCEPTIONS = (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


class _DummyResult(OptimizeResult):
    def __init__(self, success: bool):
        super().__init__(success=success, fun=np.array([1.0], dtype=float), message="ok" if success else "failed")


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


def _finite_predict_all(params, model, datasets, solver_failure_mode="fail-fast"):
    y_pred_list = [np.array(ds.y, copy=True) for ds in datasets]
    residuals = [np.zeros_like(ds.y) for ds in datasets]
    return y_pred_list, [], residuals


def test_bootstrap_counts_only_converged_refits():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0 + 0.1, _DummyResult(success=False)
        if call_count["n"] == 2:
            return params0 + 0.2, _DummyResult(success=True)
        return params0 + 0.3, _DummyResult(success=True)

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=3,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert out.n_success == 2
    assert out.n_boot == 3
    assert out.param_samples.shape == (2, 3)
    np.testing.assert_allclose(out.param_samples[0], params + 0.2)
    np.testing.assert_allclose(out.param_samples[1], params + 0.3)


def test_bootstrap_excludes_nonfinite_params():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            bad = params0.copy()
            bad[0] = np.nan
            return bad, _DummyResult(success=True)
        return params0 + 0.2, _DummyResult(success=True)

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=2,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert out.n_success == 1
    assert out.n_boot == 2
    assert out.param_samples.shape == (1, 3)
    np.testing.assert_allclose(out.param_samples[0], params + 0.2)


def test_bootstrap_all_nonconverged_yields_no_samples():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        return params0, _DummyResult(success=False)

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=2,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert out.n_success == 0
    assert out.n_boot == 2
    assert out.param_samples.shape == (0, 3)
    assert out.logk_samples.shape == (0, 1)
    assert np.isnan(out.ci_low).all()
    assert np.isnan(out.ci_high).all()
    assert np.isnan(out.ci_low_percentile).all()
    assert np.isnan(out.ci_high_percentile).all()
    assert np.isnan(out.ci_low_bca).all()
    assert np.isnan(out.ci_high_bca).all()
    assert out.ci_method == "percentile"


def test_accept_bootstrap_fit_rejects_nonfinite_predictions():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)

    def fake_predict_all(params, model, datasets, solver_failure_mode="fail-fast"):
        bad = np.full_like(ds.y, np.nan)
        return [bad], [], [np.zeros_like(ds.y)]

    accepted = accept_bootstrap_fit(
        params,
        cast(OptimizeResult, _DummyResult(success=True)),
        model,
        [ds],
        predict_all_fn=fake_predict_all,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert accepted is False


def test_bootstrap_supports_bca_ci_method():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        return params0 + np.array([0.05, 0.0, 0.0]), _DummyResult(success=True)

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=5,
        method="residual",
        ci_method="bca",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert out.ci_method == "bca"
    assert out.ci_low.shape == (1,)
    assert out.ci_high.shape == (1,)
    assert out.ci_low_percentile.shape == (1,)
    assert out.ci_high_percentile.shape == (1,)
    assert out.ci_low_bca.shape == (1,)
    assert out.ci_high_bca.shape == (1,)
