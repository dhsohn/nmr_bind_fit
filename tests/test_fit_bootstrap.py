from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult

from nmr_bind_fit.fit_bootstrap import (
    MIN_BOOTSTRAP_CI_SAMPLES,
    MIN_BOOTSTRAP_CI_SUCCESSES,
    _bootstrap_ci_requirement,
    _delete_dataset_row,
    _residual_bootstrap,
    accept_bootstrap_fit,
    bootstrap_params,
)
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


def test_bootstrap_minimum_sample_name_remains_compatible_with_stricter_contract():
    assert MIN_BOOTSTRAP_CI_SAMPLES == MIN_BOOTSTRAP_CI_SUCCESSES == 20
    assert _bootstrap_ci_requirement(20) == 20
    assert _bootstrap_ci_requirement(100) == 100


def test_bootstrap_tail_failures_never_produce_an_unconditional_ci():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)
    draws = iter(np.linspace(2.0, 6.0, 100))

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        logk = next(draws)
        fitted = params0.copy()
        fitted[0] = logk
        return fitted, _DummyResult(success=logk <= 5.2)

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=100,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert out.n_success == 80
    assert out.ci_valid is False
    assert out.ci_method_used == "unavailable"
    assert "all 100 requested refits must succeed" in out.ci_message
    assert np.isnan(out.ci_low).all()
    assert np.isnan(out.ci_high).all()
    assert np.isfinite(out.ci_low_percentile).all()
    assert np.isfinite(out.ci_high_percentile).all()


def test_bootstrap_retries_failed_jittered_start_from_full_data_optimum():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)
    calls = {"count": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        calls["count"] += 1
        first_attempt = calls["count"] % 2 == 1
        return params0, _DummyResult(success=not first_attempt)

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=20,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.1,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert calls["count"] == 40
    assert out.n_success == 20
    assert out.ci_valid is True


def test_bootstrap_retry_does_not_replace_better_bound_tail_with_worse_interior_fit():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)
    calls = {"count": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        calls["count"] += 1
        fitted = params0.copy()
        if calls["count"] % 2 == 1:
            fitted[0] = 12.0
            return fitted, OptimizeResult(
                success=True,
                fun=np.array([0.0]),
                jac=np.eye(3),
                active_mask=np.array([1, 0, 0]),
            )
        fitted[0] = 4.0
        return fitted, OptimizeResult(
            success=True,
            fun=np.array([np.sqrt(500.0)]),
            jac=np.eye(3),
            active_mask=np.zeros(3, dtype=int),
        )

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=20,
        method="residual",
        seed=0,
        logk_bounds=(0.0, 12.0),
        logk_jitter=0.1,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert calls["count"] == 40
    assert out.n_success == 0
    assert out.ci_valid is False
    assert np.isnan(out.ci_low).all()
    assert np.isnan(out.ci_high).all()


def test_bootstrap_retry_censors_statistically_competitive_bound_fit():
    n_points = 100
    h0 = np.full(n_points, 1e-3, dtype=float)
    g0 = np.linspace(0.0, 2e-3, n_points)
    ds = Dataset(
        name="near_tie",
        path=Path("near_tie.csv"),
        h_tot=h0,
        g_tot=g0,
        x=g0 / h0,
        y=np.linspace(7.0, 7.5, n_points).reshape(-1, 1),
        y_cols=["ppm1"],
        dropped_peaks=[],
    )
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)
    calls = {"count": 0}
    jacobian = np.zeros((n_points, len(params)), dtype=float)
    jacobian[: len(params), :] = np.eye(len(params))

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        calls["count"] += 1
        fitted = params0.copy()
        residual = np.zeros(n_points, dtype=float)
        if calls["count"] % 2 == 1:
            fitted[0] = 12.0
            residual[0] = np.sqrt(1.000001)
            active_mask = np.array([1, 0, 0])
        else:
            fitted[0] = 4.0
            residual[0] = 1.0
            active_mask = np.zeros(3, dtype=int)
        return fitted, OptimizeResult(
            success=True,
            fun=residual,
            jac=jacobian,
            active_mask=active_mask,
        )

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=20,
        method="residual",
        seed=0,
        logk_bounds=(0.0, 12.0),
        logk_jitter=0.1,
        predict_all_fn=_finite_predict_all,
        fit_with_initial_fn=fake_fit_with_initial,
        param_bounds_fn=param_bounds,
        numeric_exceptions=_NUMERIC_EXCEPTIONS,
    )

    assert calls["count"] == 40
    assert out.n_success == 0
    assert out.ci_valid is False
    assert np.isnan(out.ci_low).all()
    assert np.isnan(out.ci_high).all()


def test_delete_dataset_row_preserves_input_drop_metadata():
    ds = _make_dataset()
    ds.dropped_rows = 4

    out = _delete_dataset_row(ds, 1)

    assert out.n_points == ds.n_points - 1
    assert out.dropped_rows == 4


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
    assert out.ci_valid is False
    assert out.ci_method_used == "unavailable"
    assert "1/2 refits succeeded" in out.ci_message
    assert np.isnan(out.ci_low).all()
    assert np.isnan(out.ci_high).all()
    assert "Bootstrap CI omitted" in out.ci_warning


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


def test_residual_bootstrap_does_not_treat_missing_residuals_as_zero():
    ds = _make_dataset()
    ds.y = np.array([[7.0], [np.nan], [7.3]], dtype=float)
    y_pred = np.zeros_like(ds.y)
    residuals = np.array([[1.0], [np.nan], [3.0]], dtype=float)

    class FakeRng:
        def integers(self, low, high=None, size=None):
            return np.ones(size, dtype=int)

        def choice(self, values, size=None):
            return np.full(size, values[0], dtype=float)

    out = _residual_bootstrap(FakeRng(), ds, y_pred, residuals)

    assert np.isnan(out.y[1, 0])
    assert np.all(out.y[[0, 2], 0] != 0.0)


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
        n_boot=20,
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
    assert out.ci_method_used == "bca-local"
    assert out.ci_valid is True
    assert out.ci_message == ""
    assert out.ci_low.shape == (1,)
    assert out.ci_high.shape == (1,)
    assert out.ci_low_percentile.shape == (1,)
    assert out.ci_high_percentile.shape == (1,)
    assert out.ci_low_bca.shape == (1,)
    assert out.ci_high_bca.shape == (1,)
    assert np.isfinite(out.ci_low_bca).all()
    assert np.isfinite(out.ci_high_bca).all()


def test_bca_does_not_silently_fallback_when_jackknife_refit_fails():
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] > 20:
            return params0, _DummyResult(success=False)
        return params0 + np.array([0.05, 0.0, 0.0]), _DummyResult(success=True)

    out = bootstrap_params(
        params=params,
        model=model,
        datasets=[ds],
        n_boot=20,
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

    assert out.n_success == 20
    assert out.ci_valid is False
    assert out.ci_method_used == "unavailable"
    assert "jackknife refits" in out.ci_message
    assert np.isnan(out.ci_low).all()
    assert np.isnan(out.ci_high).all()
    assert np.isfinite(out.ci_low_percentile).all()
    assert np.isfinite(out.ci_high_percentile).all()


def test_nonbinding_bootstrap_se_uses_same_minimum_success_contract():
    ds = _make_dataset()
    model = MODEL_SPECS["nb"]
    params = np.array([7.0, 0.5], dtype=float)

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        return params0, _DummyResult(success=True)

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

    assert out.n_success == 2
    assert out.ci_valid is False
    assert out.ci_method_used == "unavailable"
    assert "Bootstrap uncertainty unavailable" in out.ci_message
    assert out.ci_warning == out.ci_message
