from types import SimpleNamespace

import numpy as np
import pytest

from nmr_bind_fit.fit import FitResult, fit_models
from nmr_bind_fit.fit_bootstrap import BootstrapResult


def test_fit_result_accepts_legacy_positional_constructor():
    result = FitResult(
        SimpleNamespace(name="11", n_logk=1),
        [],
        np.array([4.0]),
        ["logK"],
        True,
        "ok",
        1.0,
        1.0,
        0.9,
        [0.9],
        2.0,
        3.0,
        4,
        1,
        3,
        [],
        [],
        [],
        {},
        None,
        0,
        (0.0, 12.0),
    )

    assert result.logk_bounds == (0.0, 12.0)
    assert result.jacobian_rank == 0
    assert np.isinf(result.jacobian_condition)
    assert np.isnan(result.jacobian_logk_sensitivity)


def test_bootstrap_result_accepts_legacy_positional_constructor():
    empty = np.empty((0, 1))
    result = BootstrapResult(
        empty,
        empty,
        np.array([1.0]),
        np.array([2.0]),
        np.array([1.0]),
        np.array([2.0]),
        np.array([np.nan]),
        np.array([np.nan]),
        "percentile",
        20,
        20,
    )

    assert result.n_success == 20
    assert result.n_boot == 20
    assert result.ci_valid is True
    assert result.ci_method_used == ""


@pytest.mark.parametrize("replicates", [False, True])
def test_fit_models_rejects_empty_dataset_list_consistently(replicates):
    with pytest.raises(ValueError, match="At least one dataset is required"):
        fit_models([], ["11"], logk_starts=[4.0], replicates=replicates)
