"""Information criteria (BIC, AICc) for candidate-model comparison."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .io import Dataset
from .stats import aicc_from_loglik, bic_from_loglik, gaussian_loglik


def information_criteria(
    datasets: List[Dataset],
    residuals: List[np.ndarray],
    p: int,
) -> Tuple[float, float]:
    stacked = np.concatenate([res.ravel() for res in residuals]) if residuals else np.array([], dtype=float)
    loglik, n_loglik, n_sigma = gaussian_loglik(stacked)
    bic_p = p + n_sigma
    if n_loglik <= 0:
        return float("nan"), float("nan")
    if n_loglik <= bic_p:
        return float("nan"), float("nan")
    if not np.isfinite(loglik):
        return float("nan"), float("nan")
    bic = bic_from_loglik(loglik, n_loglik, bic_p)
    aicc = aicc_from_loglik(loglik, n_loglik, bic_p)
    return bic, aicc
