from pathlib import Path

import numpy as np

import nmrbindfit.fit as fit
from nmrbindfit.io import Dataset


class _DummyResult:
    def __init__(self, success: bool, rss: float, message: str):
        self.success = success
        self.fun = np.array([np.sqrt(rss)], dtype=float)
        self.message = message


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
        sigma=None,
        dropped_peaks=[],
    )


def test_multistart_prefers_success_over_lower_rss_failure(monkeypatch):
    ds = _make_dataset()
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0, _DummyResult(success=False, rss=1.0, message="failed_low_rss")
        return params0 + 0.05, _DummyResult(success=True, rss=4.0, message="converged")

    monkeypatch.setattr(fit, "_fit_with_initial", fake_fit_with_initial)

    result = fit.fit_model(
        [ds],
        "11",
        logk_starts=[1.0, 2.0],
        max_nfev=10,
        bootstrap=0,
    )

    assert result.success is True
    assert result.message == "converged"
    assert result.params[0] > 2.0


def test_multistart_all_fail_uses_best_failure(monkeypatch):
    ds = _make_dataset()
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return params0, _DummyResult(success=False, rss=4.0, message="failed_high_rss")
        return params0, _DummyResult(success=False, rss=1.0, message="failed_low_rss")

    monkeypatch.setattr(fit, "_fit_with_initial", fake_fit_with_initial)

    result = fit.fit_model(
        [ds],
        "11",
        logk_starts=[1.0, 2.0],
        max_nfev=10,
        bootstrap=0,
    )

    assert result.success is False
    assert result.message == "failed_low_rss"
    assert np.isnan(result.rss)


def test_multistart_skips_exception_and_keeps_success(monkeypatch):
    ds = _make_dataset()
    call_count = {"n": 0}

    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("numeric failure")
        return params0, _DummyResult(success=True, rss=2.0, message="converged")

    monkeypatch.setattr(fit, "_fit_with_initial", fake_fit_with_initial)

    result = fit.fit_model(
        [ds],
        "11",
        logk_starts=[1.0, 2.0],
        max_nfev=10,
        bootstrap=0,
    )

    assert result.success is True
    assert result.message == "converged"
