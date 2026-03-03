"""Shared protocol types used across modules."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np


class DatasetLike(Protocol):
    """Protocol for dataset-shaped objects used by fit/model/plot layers."""

    name: str
    path: Path
    h_tot: np.ndarray
    g_tot: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_cols: List[str]
    dropped_peaks: List[str]

    @property
    def n_points(self) -> int:
        ...

    @property
    def n_peaks(self) -> int:
        ...


class SolverStatsLike(Protocol):
    points: int
    success: int
    fail: int
    method: str
    failed_indices: List[int]


class SpeciesLike(Protocol):
    solver_stats: Optional[SolverStatsLike]


class BootstrapLike(Protocol):
    param_samples: np.ndarray
    logk_samples: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_low_percentile: np.ndarray
    ci_high_percentile: np.ndarray
    ci_low_bca: np.ndarray
    ci_high_bca: np.ndarray
    ci_method: str
    n_success: int
    n_boot: int


class ModelLike(Protocol):
    name: str
    n_logk: int


class FitResultLike(Protocol):
    model: ModelLike
    datasets: List[DatasetLike]
    params: np.ndarray
    param_names: List[str]
    success: bool
    message: str
    rss: float
    rmse: float
    r2: float
    r2_per_peak: List[float]
    bic: float
    aicc: float
    residual_diagnostics: dict
    species: List[SpeciesLike]
    residuals: List[np.ndarray]
    bootstrap: Optional[BootstrapLike]
    penalty_count: int
