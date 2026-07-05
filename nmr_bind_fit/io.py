"""Input/output helpers for NMR binding fits."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
    dropped_peaks: List[str]
    dropped_rows: int = 0

    @property
    def n_points(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_peaks(self) -> int:
        return int(self.y.shape[1])


REQUIRED_HOST_COL = "[H]t"
REQUIRED_GUEST_COL = "[G]t"


def _norm_col(name: str) -> str:
    # Normalize column names for fuzzy matching.
    return re.sub(r"[^a-z0-9]+", "", name.lower())


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
) -> Tuple[str, str]:
    if REQUIRED_HOST_COL not in columns or REQUIRED_GUEST_COL not in columns:
        raise ValueError("Required concentration columns not found. Expected columns: [H]t, [G]t.")
    return REQUIRED_HOST_COL, REQUIRED_GUEST_COL


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
) -> pd.DataFrame:
    # Keep only columns needed for fitting.
    use_cols = [host_col, guest_col] + list(ppm_cols)
    return df.loc[:, use_cols].copy()


def _drop_missing_required(
    data: pd.DataFrame,
    host_col: str,
    guest_col: str,
) -> Tuple[pd.DataFrame, int]:
    # Drop rows missing required concentration values.
    required_cols = [host_col, guest_col]
    before = int(len(data))
    dropped = data.dropna(axis=0, how="any", subset=required_cols)
    return dropped, before - int(len(dropped))


def _drop_incomplete_ppm_columns(
    data: pd.DataFrame,
    ppm_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Remove ppm columns containing any missing value.
    ppm_cols_list = list(ppm_cols)
    ppm_view = data.loc[:, ppm_cols_list].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(ppm_view.to_numpy(dtype=float))
    missing_by_col = pd.Series(~finite_mask.all(axis=0), index=ppm_cols_list)
    dropped_ppm = [col for col in ppm_cols_list if bool(missing_by_col.get(col, False))]
    if dropped_ppm:
        warnings.warn(
            "Dropping ppm columns with missing or non-finite values: " + ", ".join(dropped_ppm),
            RuntimeWarning,
        )
    kept_ppm = [col for col in ppm_cols if col not in dropped_ppm]
    if not kept_ppm:
        raise ValueError("No ppm columns remain after dropping columns with missing or non-finite values.")
    return ppm_view.loc[:, kept_ppm].copy(), kept_ppm, dropped_ppm


def _apply_missing_policy(
    data: pd.DataFrame,
    ppm_cols: Sequence[str],
    missing_policy: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if missing_policy == "drop-column":
        return _drop_incomplete_ppm_columns(data, ppm_cols)
    if missing_policy == "mask":
        kept_ppm = list(ppm_cols)
        if not kept_ppm:
            raise ValueError("No ppm columns remain after applying missing-value policy.")
        ppm_view = data.loc[:, kept_ppm].apply(pd.to_numeric, errors="coerce")
        ppm_array = ppm_view.to_numpy(dtype=float, copy=True)
        ppm_array[~np.isfinite(ppm_array)] = np.nan
        return pd.DataFrame(ppm_array, columns=kept_ppm, index=data.index), kept_ppm, []
    raise ValueError("missing_policy must be one of: drop-column, mask")


def _validate_concentration_arrays(h_tot: np.ndarray, g_tot: np.ndarray) -> None:
    # Validate concentration arrays prior to model fitting.
    if not np.all(np.isfinite(h_tot)) or np.any(h_tot <= 0):
        raise ValueError("Host concentration values must be positive and finite.")
    if not np.all(np.isfinite(g_tot)) or np.any(g_tot < 0):
        raise ValueError("Guest concentration values must be non-negative and finite.")


def _compute_equivalents(h_tot: np.ndarray, g_tot: np.ndarray) -> np.ndarray:
    # Compute equivalents (G/H) for plotting and x-axis usage.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h_tot != 0, g_tot / h_tot, 0.0)


def load_dataset(
    path: Path,
    ppm_cols: Optional[Sequence[str]] = None,
    missing_policy: str = "drop-column",
) -> Dataset:
    """Load a single dataset from CSV or XLSX."""
    df = _read_table(path)
    columns = list(df.columns)
    host_col, guest_col = _resolve_concentration_columns(columns)
    ppm_cols = _resolve_ppm_cols(columns, ppm_cols)

    data = _subset_input_columns(df, host_col, guest_col, ppm_cols)
    data, dropped_required_rows = _drop_missing_required(data, host_col, guest_col)

    ppm_data, ppm_cols, dropped_ppm = _apply_missing_policy(data, ppm_cols, missing_policy)
    use_cols = [host_col, guest_col] + ppm_cols
    data = data.loc[:, use_cols].copy()

    # Extract numeric arrays and validate concentrations.
    h_tot = np.asarray(data.loc[:, host_col], dtype=float)
    g_tot = np.asarray(data.loc[:, guest_col], dtype=float)
    _validate_concentration_arrays(h_tot, g_tot)

    # Extract ppm values.
    y = np.asarray(ppm_data, dtype=float)
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
        dropped_peaks=dropped_ppm,
        dropped_rows=dropped_required_rows,
    )


def load_datasets(
    paths: Sequence[Path],
    ppm_cols: Optional[str],
    missing_policy: str = "drop-column",
) -> List[Dataset]:
    """Load multiple datasets."""
    # Reuse column parsing for all input paths.
    ppm_cols_list = _split_cols(ppm_cols)
    datasets = []
    for path in paths:
        datasets.append(
            load_dataset(
                path,
                ppm_cols=ppm_cols_list,
                missing_policy=missing_policy,
            )
        )
    return datasets
