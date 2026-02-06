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
    sigma: Optional[np.ndarray]
    dropped_peaks: List[str]

    @property
    def n_points(self) -> int:
        ...

    @property
    def n_peaks(self) -> int:
        ...
