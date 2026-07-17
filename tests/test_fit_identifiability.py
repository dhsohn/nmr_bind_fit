from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from nmr_bind_fit.fit import _param_names_multi, fit_models
from nmr_bind_fit.io import Dataset, load_dataset
from nmr_bind_fit.models import MODEL_SPECS


def _dataset(
    name: str,
    h_tot: np.ndarray,
    g_tot: np.ndarray,
    y: np.ndarray,
    *,
    path: Optional[str] = None,
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
    ds = _dataset(
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
        bootstrap=20,
        seed=1,
    )[0]

    assert result.success is False
    assert "without positive guest concentrations" in result.message
    assert result.bootstrap is None
    assert result.logk_bounds == (0.0, 12.0)


def test_fit_rejects_nonpositive_residual_degrees_of_freedom():
    ds = _dataset(
        "two_points",
        np.full(2, 1e-3),
        np.array([0.0, 1e-3]),
        np.array([[7.0], [7.2]]),
    )

    result = fit_models([ds], ["11"], logk_starts=[4.0], bootstrap=0)[0]

    assert result.success is False
    assert "positive residual degrees of freedom are required" in result.message
    assert result.n == 2
    assert result.p == 3
    assert result.dof == -1


def test_fit_rejects_rank_deficient_positive_guest_design():
    n = 12
    ds = _dataset(
        "constant_condition",
        np.full(n, 1e-3),
        np.full(n, 5e-4),
        (7.0 + np.linspace(-0.01, 0.01, n)).reshape(-1, 1),
    )

    result = fit_models([ds], ["11"], logk_starts=[4.0], bootstrap=0)[0]

    assert result.success is False
    assert "Jacobian rank" in result.message
    assert result.jacobian_rank < result.p


@pytest.mark.parametrize("start", [8.0, 10.0, 11.9])
def test_fit_rejects_practically_flat_high_affinity_solution(start):
    n = 20
    h_tot = np.full(n, 1e-3)
    g_tot = np.linspace(0.0, 5e-4, n)
    ds = _dataset(
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
        bootstrap=0,
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
            logk_starts=[3.0, 5.0],
            logk_bounds=(0.0, 12.0),
            max_nfev=2000,
            bootstrap=0,
        )[0]
        assert result.success is True
        np.testing.assert_allclose(
            result.params[: result.model.n_logk],
            expected_logk,
            rtol=0.0,
            atol=5e-4,
        )
        results.append(result)

    reference = results[1]
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
        bootstrap=0,
    )

    successful = [result for result in results if result.success and np.isfinite(result.bic)]
    assert successful
    assert min(successful, key=lambda result: result.bic).model.name == "nb"


def test_masked_endpoint_uses_finite_peak_endpoints_for_initialization():
    h_tot = np.full(5, 1e-3)
    g_tot = np.linspace(0.0, 4e-3, 5)
    ds = _dataset(
        "masked",
        h_tot,
        g_tot,
        np.array([[np.nan], [7.1], [7.2], [7.3], [7.4]]),
    )

    result = fit_models([ds], ["nb"], logk_starts=[4.0], bootstrap=0)[0]

    assert result.success is True
    assert result.jacobian_rank == result.p
    assert np.all(np.isfinite(result.params))


def test_replicate_parameter_names_disambiguate_duplicate_dataset_stems():
    h_tot = np.full(4, 1e-3)
    g_tot = np.linspace(0.0, 1e-3, 4)
    y = np.linspace(7.0, 7.3, 4).reshape(-1, 1)
    ds1 = _dataset("sample", h_tot, g_tot, y, path="a/sample.csv")
    ds2 = _dataset("sample", h_tot, g_tot, y, path="b/sample.csv")
    ds3 = _dataset("1_sample", h_tot, g_tot, y, path="c/1_sample.csv")

    names = _param_names_multi(MODEL_SPECS["11"], [ds1, ds2, ds3])

    assert len(names) == len(set(names))
    assert any("1_sample" in name for name in names)
    assert any("2_sample" in name for name in names)
    assert any("3_1_sample" in name for name in names)


def test_programmatic_api_rejects_nonfinite_starts_before_model_selection():
    h_tot = np.full(5, 1e-3)
    g_tot = np.linspace(0.0, 1e-3, 5)
    y = np.linspace(7.0, 7.3, 5).reshape(-1, 1)
    ds = _dataset("sample", h_tot, g_tot, y)

    with pytest.raises(ValueError, match="logk_starts must contain only finite values"):
        fit_models([ds], ["11", "nb"], logk_starts=[float("nan")], bootstrap=0)


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

    result = fit_models([ds], ["11"], logk_starts=[4.0], bootstrap=0)[0]

    assert result.success is False
    assert "must contain at least one point and one peak" in result.message
