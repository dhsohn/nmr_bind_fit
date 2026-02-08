from pathlib import Path

import numpy as np

import nmrbindfit.fit as fit
from nmrbindfit.io import Dataset
from nmrbindfit.models import MODEL_SPECS


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
