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
    newton_success: int
    newton_fail: int
    newton_max_iter: int
    fallback_success: int
    fallback_fail: int
    fallback_method: str
    failed_indices: List[int]


class SpeciesLike(Protocol):
    solver_stats: Optional[SolverStatsLike]


class BootstrapLike(Protocol):
    param_samples: np.ndarray
    logk_samples: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
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
    bic: float
    aicc: float
    species: List[SpeciesLike]
    residuals: List[np.ndarray]
    bootstrap: Optional[BootstrapLike]
    penalty_count: int
