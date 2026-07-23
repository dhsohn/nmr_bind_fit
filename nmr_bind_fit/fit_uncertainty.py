"""Asymptotic parameter uncertainty for nonlinear least-squares binding fits.

Standard errors come from the covariance matrix of the converged fit,
``cov = (RSS / dof) * inv(J^T J)``, and binding-constant intervals use the
Student-t quantile for the residual degrees of freedom. This is the standard
large-sample approximation for nonlinear least squares. It assumes the model is
locally linear near the optimum, which the identifiability gate already enforces
by rejecting rank-deficient or ill-conditioned Jacobians before a fit is
reported.

The optimizer divides residuals by one global response scale, so ``res.fun`` is
``r / s`` and ``res.jac`` is ``J / s``. That scale cancels in
``(sum(fun**2) / dof) * inv(jac.T @ jac)``, so no unscaling is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.stats import t as student_t

CONFIDENCE_LEVEL = 0.95


@dataclass
class Uncertainty:
    param_se: np.ndarray
    logk_ci_low: np.ndarray
    logk_ci_high: np.ndarray
    correlation: np.ndarray


def parameter_uncertainty(
    params: np.ndarray,
    res: OptimizeResult,
    n_logk: int,
) -> Uncertainty:
    """Return standard errors, log10(K) t intervals, and parameter correlations.

    Raises:
        numpy.linalg.LinAlgError: If the normal matrix is exactly singular.
            Callers already treat that as a failed fit.
    """
    jacobian = res.jac
    if hasattr(jacobian, "toarray"):
        jacobian = jacobian.toarray()
    jacobian = np.asarray(jacobian, dtype=float)
    residuals = np.asarray(res.fun, dtype=float)

    n_params = int(np.size(params))
    dof = int(residuals.size - n_params)
    covariance = (float(np.sum(residuals**2)) / dof) * np.linalg.inv(jacobian.T @ jacobian)

    variance = np.diag(covariance)
    # A perfect fit (zero residual, as in the noise-free synthetic examples) has
    # zero variance, which is a degenerate but honest answer rather than an error.
    param_se = np.sqrt(np.where(variance > 0.0, variance, 0.0))

    critical = float(student_t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, dof))
    logk = np.asarray(params[:n_logk], dtype=float)
    logk_se = param_se[:n_logk]
    logk_ci_low = logk - critical * logk_se
    logk_ci_high = logk + critical * logk_se

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(param_se, param_se)

    return Uncertainty(
        param_se=param_se,
        logk_ci_low=logk_ci_low,
        logk_ci_high=logk_ci_high,
        correlation=correlation,
    )
