from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.stats import t as student_t

from nmr_bind_fit import fit
from nmr_bind_fit.fit import _param_names_multi, fit_models
from nmr_bind_fit.fit_optimizer import param_bounds, select_best_multistart
from nmr_bind_fit.fit_uncertainty import CONFIDENCE_LEVEL, parameter_uncertainty
from nmr_bind_fit.io import Dataset, load_dataset
from nmr_bind_fit.models import (
    MODEL_SPECS,
    ModelSpec,
    predict_dataset,
    split_params_multi,
)
from nmr_bind_fit.stats import (
    aicc_from_loglik,
    bic_from_loglik,
    gaussian_loglik,
    information_criteria,
    residual_diagnostics,
)


class _DummyResult(OptimizeResult):
    def __init__(self, success: bool, rss: float, message: str):
        super().__init__(success=success, fun=np.array([np.sqrt(rss)], dtype=float), message=message)


def _named_dataset(
    name: str,
    h_tot: np.ndarray,
    g_tot: np.ndarray,
    y: np.ndarray,
    *,
    path: str | None = None,
) -> Dataset:
    return Dataset(
        name=name,
        path=Path(path or f"{name}.csv"),
        h_tot=h_tot,
        g_tot=g_tot,
        x=g_tot / h_tot,
        y=y,
        y_cols=[f"ppm{i + 1}" for i in range(y.shape[1])],
        dropped_peaks=[],
    )


def test_binding_fit_rejects_dataset_without_positive_guest():
    n = 20
    ds = _named_dataset(
        "zero_guest",
        np.full(n, 1e-3),
        np.zeros(n),
        (7.0 + np.linspace(-0.01, 0.01, n)).reshape(-1, 1),
    )

    result = fit_models(
        [ds],
        ["11"],
        logk_starts=[1.0, 4.0, 9.0],
        logk_bounds=(0.0, 12.0),
    )[0]

    assert result.success is False
    assert "without positive guest concentrations" in result.message
    assert result.uncertainty is None
    assert result.logk_bounds == (0.0, 12.0)


def test_fit_rejects_nonpositive_host_concentration():
    # Physically impossible input (a mistyped CSV) is rejected at the fit
    # boundary; the loader no longer repeats this check. A negative host
    # exercises the same ``h_tot <= 0`` guard as a zero would, without the
    # divide-by-zero the equivalents helper produces for an exact zero.
    ds = _named_dataset(
        "bad_host",
        np.array([1e-3, -1e-3, 2e-3]),
        np.array([0.0, 5e-4, 1e-3]),
        np.array([[7.0], [7.1], [7.2]]),
    )

    result = fit_models([ds], ["11"], logk_starts=[4.0], logk_bounds=(0.0, 12.0))[0]

    assert result.success is False
    assert "invalid host concentrations" in result.message


def test_fit_rejects_negative_guest_concentration():
    ds = _named_dataset(
        "bad_guest",
        np.full(3, 1e-3),
        np.array([0.0, -5e-4, 1e-3]),
        np.array([[7.0], [7.1], [7.2]]),
    )

    result = fit_models([ds], ["11"], logk_starts=[4.0], logk_bounds=(0.0, 12.0))[0]

    assert result.success is False
    assert "invalid guest concentrations" in result.message


def test_fit_rejects_nonpositive_residual_degrees_of_freedom():
    ds = _named_dataset(
        "two_points",
        np.full(2, 1e-3),
        np.array([0.0, 1e-3]),
        np.array([[7.0], [7.2]]),
    )

    result = fit_models([ds], ["11"], logk_starts=[4.0])[0]

    assert result.success is False
    assert "positive residual degrees of freedom are required" in result.message
    assert result.n == 2
    assert result.p == 3
    assert result.dof == -1


def test_fit_rejects_rank_deficient_positive_guest_design():
    n = 12
    ds = _named_dataset(
        "constant_condition",
        np.full(n, 1e-3),
        np.full(n, 5e-4),
        (7.0 + np.linspace(-0.01, 0.01, n)).reshape(-1, 1),
    )

    result = fit_models([ds], ["11"], logk_starts=[4.0])[0]

    assert result.success is False
    assert "Jacobian rank" in result.message
    assert result.jacobian_rank < result.p


@pytest.mark.parametrize("start", [8.0, 10.0, 11.9])
def test_fit_rejects_practically_flat_high_affinity_solution(start):
    n = 20
    h_tot = np.full(n, 1e-3)
    g_tot = np.linspace(0.0, 5e-4, n)
    ds = _named_dataset(
        "high_affinity_limit",
        h_tot,
        g_tot,
        (7.0 + 0.1 * g_tot / h_tot).reshape(-1, 1),
    )

    result = fit_models(
        [ds],
        ["11"],
        logk_starts=[start],
        logk_bounds=(0.0, 12.0),
    )[0]

    assert result.success is False
    assert "dimensionless logK RMS sensitivity" in result.message
    assert result.jacobian_logk_sensitivity < 1e-4
    assert result.logk_bounds == (0.0, 12.0)


@pytest.mark.parametrize(
    "model_name,filename,expected_logk",
    [
        ("11", "synthetic_11.csv", [4.0]),
        ("12", "synthetic_12.csv", [4.1, 3.2]),
        ("21", "synthetic_21.csv", [4.0, 3.4]),
    ],
)
def test_fit_and_identifiability_are_invariant_to_response_unit_rescaling(
    model_name,
    filename,
    expected_logk,
):
    source = load_dataset(Path(__file__).parents[1] / "examples" / filename)
    results = []
    for scale in (1e-9, 1.0, 1e9):
        dataset = Dataset(
            name=source.name,
            path=source.path,
            h_tot=source.h_tot,
            g_tot=source.g_tot,
            x=source.x,
            y=source.y * scale,
            y_cols=source.y_cols,
            dropped_peaks=source.dropped_peaks,
        )
        result = fit_models(
            [dataset],
            [model_name],
            logk_starts=[4.0],
            logk_bounds=(0.0, 12.0),
            max_nfev=2000,
        )[0]
        assert result.success is True
        results.append(result)

    reference = results[1]
    np.testing.assert_allclose(
        reference.params[: reference.model.n_logk],
        expected_logk,
        rtol=0.0,
        atol=5e-4,
    )
    for result in results:
        np.testing.assert_allclose(
            result.params[: result.model.n_logk],
            reference.params[: reference.model.n_logk],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.jacobian_condition,
            reference.jacobian_condition,
            rtol=5e-5,
        )
        np.testing.assert_allclose(
            result.jacobian_logk_sensitivity,
            reference.jacobian_logk_sensitivity,
            rtol=5e-5,
        )


def test_nonbinding_control_is_selected_without_forcing_a_binding_model():
    dataset = load_dataset(Path(__file__).parents[1] / "examples" / "synthetic_nonbinding.csv")

    results = fit_models(
        [dataset],
        ["11", "nb"],
        logk_starts=[4.0],
        logk_bounds=(0.0, 12.0),
        max_nfev=200,
    )

    successful = [result for result in results if result.success and np.isfinite(result.bic)]
    assert successful
    assert min(successful, key=lambda result: result.bic).model.name == "nb"


def test_masked_endpoint_uses_finite_peak_endpoints_for_initialization():
    h_tot = np.full(5, 1e-3)
    g_tot = np.linspace(0.0, 4e-3, 5)
    ds = _named_dataset(
        "masked",
        h_tot,
        g_tot,
        np.array([[np.nan], [7.1], [7.2], [7.3], [7.4]]),
    )

    result = fit_models([ds], ["nb"], logk_starts=[4.0])[0]

    assert result.success is True
    assert result.jacobian_rank == result.p
    assert np.all(np.isfinite(result.params))


def test_replicate_parameter_names_disambiguate_duplicate_dataset_stems():
    h_tot = np.full(4, 1e-3)
    g_tot = np.linspace(0.0, 1e-3, 4)
    y = np.linspace(7.0, 7.3, 4).reshape(-1, 1)
    ds1 = _named_dataset("sample", h_tot, g_tot, y, path="a/sample.csv")
    ds2 = _named_dataset("sample", h_tot, g_tot, y, path="b/sample.csv")
    ds3 = _named_dataset("1_sample", h_tot, g_tot, y, path="c/1_sample.csv")

    names = _param_names_multi(MODEL_SPECS["11"], [ds1, ds2, ds3])

    assert len(names) == len(set(names))
    assert any("1_sample" in name for name in names)
    assert any("2_sample" in name for name in names)
    assert any("3_1_sample" in name for name in names)


def test_programmatic_api_rejects_nonfinite_starts_before_model_selection():
    h_tot = np.full(5, 1e-3)
    g_tot = np.linspace(0.0, 1e-3, 5)
    y = np.linspace(7.0, 7.3, 5).reshape(-1, 1)
    ds = _named_dataset("sample", h_tot, g_tot, y)

    with pytest.raises(ValueError, match="logk_starts must contain only finite values"):
        fit_models([ds], ["11", "nb"], logk_starts=[float("nan")])


def test_programmatic_invalid_dataset_shape_returns_failed_result():
    ds = Dataset(
        name="invalid",
        path=Path("invalid.csv"),
        h_tot=np.full(4, 1e-3),
        g_tot=np.linspace(0.0, 1e-3, 4),
        x=np.linspace(0.0, 1.0, 4),
        y=np.linspace(7.0, 7.3, 4),
        y_cols=["ppm1"],
        dropped_peaks=[],
    )

    result = fit_models([ds], ["11"], logk_starts=[4.0])[0]

    assert result.success is False
    assert "must contain at least one point and one peak" in result.message


def _multistart_dataset() -> Dataset:
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
    ds = _multistart_dataset()
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


def test_fit_models_rejects_empty_dataset_list():
    # The guard runs before the replicates branch is reached, so one case covers it.
    with pytest.raises(ValueError, match="At least one dataset is required"):
        fit.fit_models([], ["11"], logk_starts=[4.0])


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
    ds = _multistart_dataset()
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
        logk_bounds=None,
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
            [_multistart_dataset()],
            ["11"],
            logk_starts=[start],
            logk_bounds=None,
        )


def test_fit_models_records_failure_and_continues_replicates(monkeypatch):
    ds1 = _multistart_dataset()
    ds2 = _multistart_dataset()
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
        logk_bounds=None,
    )

    assert len(results) == 2
    assert results[0].success is False
    assert results[0].model.name == "11"
    assert "ModelFitError: bad start grid" in results[0].message
    assert results[1].success is True
    assert results[1].model.name == "nb"


def test_fit_models_propagates_unexpected_exception(monkeypatch):
    ds = _multistart_dataset()
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
            logk_bounds=None,
        )


def test_select_best_multistart_propagates_unexpected_exception():
    def fake_fit_with_initial(model, datasets, params0, max_nfev, bounds, solver_failure_mode="fail-fast"):
        raise IndexError("unexpected coding bug")

    with pytest.raises(IndexError, match="unexpected coding bug"):
        _select_with_fake_optimizer(fake_fit_with_initial)


def test_fit_model_reports_residual_diagnostics_when_enabled():
    ds = _multistart_dataset()

    result = fit.fit_model(
        [ds],
        "nb",
        logk_starts=[1.0],
        max_nfev=100,
        residual_diagnostics=True,
    )

    assert result.success is True
    assert "residual_n" in result.residual_diagnostics


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
        logk_bounds=(0.0, 12.0),
    )

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].model.name == "11"
    assert "at least one finite value" in results[0].message


def _penalty_dataset() -> Dataset:
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
    ds = _penalty_dataset()
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
    ds = _penalty_dataset()
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
    ds = _penalty_dataset()
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
    ds = _penalty_dataset()
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


def _r2_dataset(y: np.ndarray) -> Dataset:
    h_tot = np.full((y.shape[0],), 1e-3, dtype=float)
    g_tot = np.linspace(0.0, 1e-3, y.shape[0], dtype=float)
    x = g_tot / h_tot
    return Dataset(
        name="sample",
        path=Path("sample.csv"),
        h_tot=h_tot,
        g_tot=g_tot,
        x=x,
        y=y,
        y_cols=[f"ppm{i + 1}" for i in range(y.shape[1])],
        dropped_peaks=[],
    )


def test_r2_score_is_mean_of_per_peak_r2():
    y_obs = np.array(
        [
            [0.0, 100.0],
            [1.0, 101.0],
            [2.0, 102.0],
        ],
        dtype=float,
    )
    y_pred = np.array(
        [
            [0.0, 100.0],
            [1.0, 101.0],
            [1.0, 101.0],
        ],
        dtype=float,
    )
    ds = _r2_dataset(y_obs)

    r2, r2_per_peak = fit._r2_score([ds], [y_pred])

    assert len(r2_per_peak) == 2
    np.testing.assert_allclose(r2_per_peak, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(r2, 0.5, atol=1e-12)


def test_r2_score_ignores_constant_peaks():
    y_obs = np.array(
        [
            [1.0, 10.0],
            [1.0, 11.0],
            [1.0, 12.0],
        ],
        dtype=float,
    )
    y_pred = np.array(
        [
            [0.0, 10.0],
            [0.0, 11.0],
            [0.0, 11.0],
        ],
        dtype=float,
    )
    ds = _r2_dataset(y_obs)

    r2, r2_per_peak = fit._r2_score([ds], [y_pred])

    assert len(r2_per_peak) == 1
    np.testing.assert_allclose(r2_per_peak, [0.5], atol=1e-12)
    np.testing.assert_allclose(r2, 0.5, atol=1e-12)


def test_split_params_rejects_trailing_values():
    ds = Dataset(
        name="test",
        path=__import__("pathlib").Path("dummy.csv"),
        h_tot=np.array([1e-3]),
        g_tot=np.array([0.0]),
        x=np.array([0.0]),
        y=np.array([[7.0]]),
        y_cols=["ppm1"],
        dropped_peaks=[],
    )
    model = MODEL_SPECS["11"]

    with pytest.raises(ValueError, match="unused trailing"):
        split_params_multi(np.array([4.0, 7.0, 7.5, 99.0]), model, [ds])


def test_predict_rejects_nonfinite_logk():
    ds = Dataset(
        name="test",
        path=__import__("pathlib").Path("dummy.csv"),
        h_tot=np.array([1e-3]),
        g_tot=np.array([0.0]),
        x=np.array([0.0]),
        y=np.array([[7.0]]),
        y_cols=["ppm1"],
        dropped_peaks=[],
    )
    model = MODEL_SPECS["11"]
    delta = np.array([[7.0, 7.5]])

    with pytest.raises(ValueError, match="logK"):
        predict_dataset(model, ds, np.array([np.inf]), delta)


@pytest.mark.parametrize(
    ("residuals", "n_model_params"),
    [
        ([np.array([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2], [0.05, -0.15]])], 5),
        ([np.array([[0.1], [-0.1], [0.2]]), np.array([[0.3], [-0.2], [0.1]])], 4),
    ],
)
def test_information_criteria_uses_one_shared_variance_parameter(residuals, n_model_params):
    bic, aicc = information_criteria(residuals, n_model_params)

    stacked = np.concatenate([res.ravel() for res in residuals])
    loglik, n_obs, n_variance_params = gaussian_loglik(stacked)
    assert n_variance_params == 1
    n_ic_params = n_model_params + n_variance_params
    expected_bic = bic_from_loglik(loglik, n_obs, n_ic_params)
    expected_aicc = aicc_from_loglik(loglik, n_obs, n_ic_params)
    np.testing.assert_allclose(bic, expected_bic)
    np.testing.assert_allclose(aicc, expected_aicc)


def test_information_criteria_keeps_bic_when_aicc_is_underpowered():
    residuals = [np.array([[0.1], [-0.1], [0.2]], dtype=float)]

    bic, aicc = information_criteria(residuals, n_model_params=2)
    loglik, n_obs, n_variance_params = gaussian_loglik(residuals[0].ravel())

    np.testing.assert_allclose(bic, bic_from_loglik(loglik, n_obs, 2 + n_variance_params))
    assert np.isnan(aicc)


def test_information_criteria_returns_nan_for_zero_rss():
    residuals = [np.zeros((5, 1), dtype=float)]

    bic, aicc = information_criteria(residuals, n_model_params=2)

    assert np.isnan(bic)
    assert np.isnan(aicc)


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


def test_standard_errors_match_the_closed_form_least_squares_covariance():
    # A straight-line design whose covariance can be worked out by hand:
    # X'X = [[4, 6], [6, 14]], det 20, so cov = s2 * [[14, -6], [-6, 4]] / 20
    # with s2 = RSS/dof = 4/2 = 2.
    jac = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    fun = np.array([1.0, -1.0, 1.0, -1.0])
    params = np.array([3.0, 0.5])

    out = parameter_uncertainty(params, OptimizeResult(jac=jac, fun=fun), n_logk=1)

    np.testing.assert_allclose(out.param_se, [np.sqrt(1.4), np.sqrt(0.4)])
    # Correlation of the design, independent of the residual scale.
    expected_corr = -6.0 / np.sqrt(4.0 * 14.0)
    np.testing.assert_allclose(out.correlation[0, 1], expected_corr)
    np.testing.assert_allclose(np.diag(out.correlation), [1.0, 1.0])


def test_interval_uses_the_student_t_quantile_for_the_residual_dof():
    jac = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    fun = np.array([1.0, -1.0, 1.0, -1.0])
    params = np.array([3.0, 0.5])

    out = parameter_uncertainty(params, OptimizeResult(jac=jac, fun=fun), n_logk=1)

    dof = 2
    critical = float(student_t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, dof))
    # A normal quantile would be 1.96; the small-sample t quantile is much wider.
    assert critical > 4.0
    np.testing.assert_allclose(out.logk_ci_low, [3.0 - critical * np.sqrt(1.4)])
    np.testing.assert_allclose(out.logk_ci_high, [3.0 + critical * np.sqrt(1.4)])


def test_perfect_fit_reports_zero_error_and_undefined_correlation():
    # Zero residual is degenerate but honest: no spread, so no correlation.
    jac = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    fun = np.zeros(4)
    params = np.array([3.0, 0.5])

    out = parameter_uncertainty(params, OptimizeResult(jac=jac, fun=fun), n_logk=1)

    np.testing.assert_allclose(out.param_se, [0.0, 0.0])
    np.testing.assert_allclose(out.logk_ci_low, [3.0])
    np.testing.assert_allclose(out.logk_ci_high, [3.0])
    assert not np.all(np.isfinite(out.correlation))


def test_only_the_binding_constants_get_intervals():
    jac = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    fun = np.array([1.0, -1.0, 1.0, -1.0])
    params = np.array([3.0, 0.5])

    out = parameter_uncertainty(params, OptimizeResult(jac=jac, fun=fun), n_logk=0)

    assert out.param_se.shape == (2,)
    assert out.logk_ci_low.shape == (0,)
    assert out.logk_ci_high.shape == (0,)


def test_fit_models_attaches_uncertainty_that_brackets_the_estimate():
    # End-to-end wiring: a real fit must carry a usable interval, so a
    # regression that drops the uncertainty call cannot pass silently.
    dataset = load_dataset(Path(__file__).parents[1] / "examples" / "synthetic_11.csv")
    rng = np.random.default_rng(5)
    dataset.y = dataset.y + rng.normal(0.0, 0.005, size=dataset.y.shape)

    result = fit_models(
        [dataset],
        ["11"],
        logk_starts=[3.0, 4.0, 5.0],
        logk_bounds=(0.0, 12.0),
    )[0]

    assert result.success
    assert result.uncertainty is not None
    assert result.uncertainty.param_se.shape == (result.p,)
    assert np.all(result.uncertainty.param_se > 0.0)
    logk = float(result.params[0])
    assert result.uncertainty.logk_ci_low[0] < logk < result.uncertainty.logk_ci_high[0]
    assert result.uncertainty.correlation.shape == (result.p, result.p)


def test_noisier_data_widens_the_reported_interval():
    base = load_dataset(Path(__file__).parents[1] / "examples" / "synthetic_11.csv")
    widths = []
    for sigma in (0.002, 0.02):
        dataset = load_dataset(Path(__file__).parents[1] / "examples" / "synthetic_11.csv")
        rng = np.random.default_rng(11)
        dataset.y = base.y + rng.normal(0.0, sigma, size=base.y.shape)
        result = fit_models(
            [dataset],
            ["11"],
            logk_starts=[3.0, 4.0, 5.0],
            logk_bounds=(0.0, 12.0),
        )[0]
        assert result.success
        widths.append(
            float(result.uncertainty.logk_ci_high[0] - result.uncertainty.logk_ci_low[0])
        )

    assert widths[1] > widths[0]
