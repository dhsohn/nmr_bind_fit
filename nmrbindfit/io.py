"""Input/output helpers for NMR binding fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Dataset:
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
        return int(self.y.shape[0])

    @property
    def n_peaks(self) -> int:
        return int(self.y.shape[1])


def _norm_col(name: str) -> str:
    # Normalize column names for fuzzy matching.
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_first(columns: Sequence[str], candidates: Iterable[str]) -> Optional[str]:
    # Map normalized names to original columns for quick lookup.
    norm_map = {_norm_col(c): c for c in columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _find_ppm_columns(columns: Sequence[str]) -> List[str]:
    # Infer ppm columns by name when they are not explicitly supplied.
    ppm_cols = [c for c in columns if "ppm" in _norm_col(c)]
    return ppm_cols


def _split_cols(value: Optional[str]) -> Optional[List[str]]:
    # Parse a comma-separated list into clean column names.
    if value is None:
        return None
    cols = [c.strip() for c in value.split(",") if c.strip()]
    return cols or None


def _read_table(path: Path) -> pd.DataFrame:
    # Load tabular data based on file extension.
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _resolve_concentration_columns(
    columns: Sequence[str],
    host_col: Optional[str],
    guest_col: Optional[str],
) -> Tuple[str, str]:
    # Guess host/guest concentration columns when not explicitly provided.
    resolved_host = host_col
    resolved_guest = guest_col

    if resolved_host is None:
        resolved_host = _find_first(
            columns,
            [
                "host conc",
                "host concentration",
                "host",
                "h0",
                "htot",
                "h_tot",
            ],
        )
    if resolved_guest is None:
        resolved_guest = _find_first(
            columns,
            [
                "guest conc",
                "guest concentration",
                "guest",
                "g0",
                "gtot",
                "g_tot",
            ],
        )
    if resolved_host is None or resolved_guest is None:
        raise ValueError("Host or guest concentration column not found.")
    return resolved_host, resolved_guest


def _resolve_ppm_cols(columns: Sequence[str], ppm_cols: Optional[Sequence[str]]) -> List[str]:
    # Determine ppm columns, either by explicit list or name heuristic.
    if ppm_cols is None:
        cols = _find_ppm_columns(columns)
    else:
        cols = list(ppm_cols)
    if not cols:
        raise ValueError("No ppm columns detected. Use --ppm-cols to specify.")
    return cols


def _subset_input_columns(
    df: pd.DataFrame,
    host_col: str,
    guest_col: str,
    ppm_cols: Sequence[str],
    sigma_col: Optional[str],
) -> pd.DataFrame:
    # Keep only columns needed for fitting.
    use_cols = [host_col, guest_col] + list(ppm_cols)
    if sigma_col is not None:
        use_cols.append(sigma_col)
    return df[use_cols].copy()


def _drop_missing_required(
    data: pd.DataFrame,
    host_col: str,
    guest_col: str,
    sigma_col: Optional[str],
) -> pd.DataFrame:
    # Drop rows missing required concentration (or sigma) values.
    required_cols = [host_col, guest_col]
    if sigma_col is not None:
        required_cols.append(sigma_col)
    return data.dropna(axis=0, how="any", subset=required_cols)


def _drop_incomplete_ppm_columns(
    data: pd.DataFrame,
    ppm_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Remove ppm columns containing any missing value.
    dropped_ppm = [col for col in ppm_cols if data[col].isna().any()]
    if dropped_ppm:
        warnings.warn(
            "Dropping ppm columns with missing values: " + ", ".join(dropped_ppm),
            RuntimeWarning,
        )
    kept_ppm = [col for col in ppm_cols if col not in dropped_ppm]
    if not kept_ppm:
        raise ValueError("No ppm columns remain after dropping columns with missing values.")
    return data[kept_ppm], kept_ppm, dropped_ppm


def _validate_concentration_arrays(h_tot: np.ndarray, g_tot: np.ndarray) -> None:
    # Validate concentration arrays prior to model fitting.
    if not np.all(np.isfinite(h_tot)) or np.any(h_tot <= 0):
        raise ValueError("Host concentration values must be positive and finite.")
    if not np.all(np.isfinite(g_tot)) or np.any(g_tot < 0):
        raise ValueError("Guest concentration values must be non-negative and finite.")


def _extract_sigma(data: pd.DataFrame, sigma_col: Optional[str], n_peaks: int) -> Optional[np.ndarray]:
    # Expand scalar sigma values across peaks when provided.
    if sigma_col is None:
        return None
    sigma_raw = data[sigma_col].to_numpy(dtype=float).reshape(-1, 1)
    if not np.all(np.isfinite(sigma_raw)) or np.any(sigma_raw <= 0):
        raise ValueError("Sigma values must be positive and finite.")
    return np.repeat(sigma_raw, n_peaks, axis=1)


def _compute_equivalents(h_tot: np.ndarray, g_tot: np.ndarray) -> np.ndarray:
    # Compute equivalents (G/H) for plotting and x-axis usage.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h_tot != 0, g_tot / h_tot, 0.0)


def load_dataset(
    path: Path,
    host_col: Optional[str] = None,
    guest_col: Optional[str] = None,
    ppm_cols: Optional[Sequence[str]] = None,
    sigma_col: Optional[str] = None,
) -> Dataset:
    """Load a single dataset from CSV or XLSX."""
    df = _read_table(path)
    columns = list(df.columns)
    host_col, guest_col = _resolve_concentration_columns(columns, host_col, guest_col)
    ppm_cols = _resolve_ppm_cols(columns, ppm_cols)

    if sigma_col is not None and sigma_col not in columns:
        raise ValueError(f"Sigma column not found: {sigma_col}")

    data = _subset_input_columns(df, host_col, guest_col, ppm_cols, sigma_col)
    data = _drop_missing_required(data, host_col, guest_col, sigma_col)

    ppm_data, ppm_cols, dropped_ppm = _drop_incomplete_ppm_columns(data, ppm_cols)
    use_cols = [host_col, guest_col] + ppm_cols
    if sigma_col is not None:
        use_cols.append(sigma_col)
    data = data[use_cols]

    # Extract numeric arrays and validate concentrations.
    h_tot = data[host_col].to_numpy(dtype=float)
    g_tot = data[guest_col].to_numpy(dtype=float)
    _validate_concentration_arrays(h_tot, g_tot)

    # Extract ppm values and optional sigma weights.
    y = ppm_data.to_numpy(dtype=float)
    sigma = _extract_sigma(data, sigma_col, y.shape[1])
    x = _compute_equivalents(h_tot, g_tot)

    name = path.stem
    # Package into the Dataset dataclass used by the fitter.
    return Dataset(
        name=name,
        path=path,
        h_tot=h_tot,
        g_tot=g_tot,
        x=x,
        y=y,
        y_cols=ppm_cols,
        sigma=sigma,
        dropped_peaks=dropped_ppm,
    )


def load_datasets(
    paths: Sequence[Path],
    host_col: Optional[str],
    guest_col: Optional[str],
    ppm_cols: Optional[str],
    sigma_col: Optional[str],
) -> List[Dataset]:
    """Load multiple datasets."""
    # Reuse column parsing for all input paths.
    ppm_cols_list = _split_cols(ppm_cols)
    datasets = []
    for path in paths:
        datasets.append(
            load_dataset(
                path,
                host_col=host_col,
                guest_col=guest_col,
                ppm_cols=ppm_cols_list,
                sigma_col=sigma_col,
            )
        )
    return datasets
