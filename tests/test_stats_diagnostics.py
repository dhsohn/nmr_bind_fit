import numpy as np

from nmr_bind_fit.stats import residual_diagnostics


def test_residual_diagnostics_returns_shapiro_and_dw():
    residuals = np.array([0.1, -0.2, 0.05, 0.0, -0.1, 0.2], dtype=float)

    out = residual_diagnostics(residuals)

    assert "residual_n" in out
    assert out["residual_n"] == 6.0
    assert "shapiro_stat" in out
    assert "shapiro_p" in out
    assert "durbin_watson" in out


def test_residual_diagnostics_handles_empty_input():
    out = residual_diagnostics(np.array([np.nan, np.inf, -np.inf], dtype=float))
    assert out == {}


def test_residual_diagnostics_can_suppress_durbin_watson_for_pooled_input():
    residuals = np.array([0.1, -0.1, 0.05, -0.05], dtype=float)

    out = residual_diagnostics(residuals, include_durbin_watson=False)

    assert "durbin_watson" not in out
