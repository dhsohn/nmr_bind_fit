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
    if not np.isfinite(loglik):
        names = ", ".join(ds.name for ds in datasets)
        raise RuntimeError(f"BIC calculation failed: log-likelihood is NaN for datasets: {names}.")
    bic_p = p + n_sigma
    if n_loglik <= 0:
        raise RuntimeError("BIC calculation failed: no valid log-likelihood terms.")
    bic = bic_from_loglik(loglik, n_loglik, bic_p)
    aicc = aicc_from_loglik(loglik, n_loglik, bic_p)
    return bic, aicc
