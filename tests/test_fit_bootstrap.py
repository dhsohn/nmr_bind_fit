from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult

import nmr_bind_fit.fit as fit
from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS


class _DummyResult:
    def __init__(self, success: bool):
        self.success = success
        self.fun = np.array([1.0], dtype=float)
        self.message = "ok" if success else "failed"


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


def test_bootstrap_counts_only_converged_refits(monkeypatch):
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

    monkeypatch.setattr(fit, "_fit_with_initial", fake_fit_with_initial)

    out = fit.bootstrap_params(
        params,
        model,
        [ds],
        n_boot=3,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
    )

    assert out.n_success == 2
    assert out.n_boot == 3
    assert out.param_samples.shape == (2, 3)
    np.testing.assert_allclose(out.param_samples[0], params + 0.2)
    np.testing.assert_allclose(out.param_samples[1], params + 0.3)


def test_bootstrap_excludes_nonfinite_params(monkeypatch):
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

    monkeypatch.setattr(fit, "_fit_with_initial", fake_fit_with_initial)

    out = fit.bootstrap_params(
        params,
        model,
        [ds],
        n_boot=2,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
    )

    assert out.n_success == 1
    assert out.n_boot == 2
    assert out.param_samples.shape == (1, 3)
    np.testing.assert_allclose(out.param_samples[0], params + 0.2)


def test_bootstrap_all_nonconverged_yields_no_samples(monkeypatch):
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        return params0, _DummyResult(success=False)

    monkeypatch.setattr(fit, "_fit_with_initial", fake_fit_with_initial)

    out = fit.bootstrap_params(
        params,
        model,
        [ds],
        n_boot=2,
        method="residual",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
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


def test_accept_bootstrap_fit_rejects_nonfinite_predictions(monkeypatch):
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)

    def fake_predict_all(params, model, datasets, solver_failure_mode="fail-fast"):
        bad = np.full_like(ds.y, np.nan)
        return [bad], [], [np.zeros_like(ds.y)]

    monkeypatch.setattr(fit, "_predict_all", fake_predict_all)

    accepted = fit._accept_bootstrap_fit(
        params,
        cast(OptimizeResult, _DummyResult(success=True)),
        model,
        [ds],
    )

    assert accepted is False


def test_bootstrap_supports_bca_ci_method(monkeypatch):
    ds = _make_dataset()
    model = MODEL_SPECS["11"]
    params = np.array([4.0, 7.0, 7.5], dtype=float)

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        return params0 + np.array([0.05, 0.0, 0.0]), _DummyResult(success=True)

    monkeypatch.setattr(fit, "_fit_with_initial", fake_fit_with_initial)

    out = fit.bootstrap_params(
        params,
        model,
        [ds],
        n_boot=5,
        method="residual",
        ci_method="bca",
        seed=0,
        logk_bounds=None,
        logk_jitter=0.0,
    )

    assert out.ci_method == "bca"
    assert out.ci_low.shape == (1,)
    assert out.ci_high.shape == (1,)
    assert out.ci_low_percentile.shape == (1,)
    assert out.ci_high_percentile.shape == (1,)
    assert out.ci_low_bca.shape == (1,)
    assert out.ci_high_bca.shape == (1,)
