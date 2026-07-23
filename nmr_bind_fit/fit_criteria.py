"""Information criteria (BIC, AICc) for candidate-model comparison."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .stats import aicc_from_loglik, bic_from_loglik, gaussian_loglik


def information_criteria(
    residuals: Sequence[np.ndarray],
    n_model_params: int,
) -> tuple[float, float]:
    stacked = np.concatenate([res.ravel() for res in residuals]) if residuals else np.array([], dtype=float)
    loglik, n_obs, n_variance_params = gaussian_loglik(stacked)
    n_ic_params = n_model_params + n_variance_params
    if n_obs <= 0:
        return float("nan"), float("nan")
    if not np.isfinite(loglik):
        return float("nan"), float("nan")
    bic = bic_from_loglik(loglik, n_obs, n_ic_params)
    aicc = aicc_from_loglik(loglik, n_obs, n_ic_params)
    return bic, aicc
