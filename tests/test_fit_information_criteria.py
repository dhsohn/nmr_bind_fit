import numpy as np
import pytest

from nmr_bind_fit.fit_criteria import information_criteria
from nmr_bind_fit.stats import aicc_from_loglik, bic_from_loglik, gaussian_loglik


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
