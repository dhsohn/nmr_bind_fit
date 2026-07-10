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
    ppm_cols = [c for c in columns if "ppm" in _norm_col(str(c))]
    return ppm_cols


def _split_cols(value: Optional[str]) -> Optional[List[str]]:
    # Parse a comma-separated list into clean column names.
    if value is None:
        return None
    return [c.strip() for c in value.split(",") if c.strip()]


def _read_table(path: Path) -> pd.DataFrame:
    # Load tabular data based on file extension.
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            if suffix == ".xls":
                message = (
                    "Reading .xls files requires the optional 'xlrd' dependency; "
                    "install xlrd or convert the file to .xlsx or CSV."
                )
            else:
                message = (
                    "Reading .xlsx files requires the optional Excel dependency; "
                    "install nmr_bind_fit[excel] or convert the file to CSV."
                )
            raise ValueError(message) from exc
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input file is empty or has no header: {path}") from exc


def _validate_input_file(path: Path) -> None:
    """Require a concrete regular input file before invoking pandas."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a regular file: {path}")


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
        missing = [col for col in cols if col not in columns]
        if missing:
            available = ", ".join(str(col) for col in columns)
            raise ValueError(
                "Requested ppm columns not found: "
                + ", ".join(missing)
                + f". Available columns: {available}"
            )
    if not cols:
        raise ValueError("No ppm columns detected. Use --ppm-cols to specify.")
    duplicates = sorted({col for col in cols if cols.count(col) > 1})
    if duplicates:
        raise ValueError("Duplicate ppm columns specified: " + ", ".join(map(str, duplicates)))
    missing = [col for col in cols if col not in columns]
    if missing:
        raise ValueError("Specified ppm columns not found: " + ", ".join(map(str, missing)))
    reserved = [col for col in cols if col in {REQUIRED_HOST_COL, REQUIRED_GUEST_COL}]
    if reserved:
        raise ValueError(
            "Concentration columns cannot also be used as ppm columns: "
            + ", ".join(map(str, reserved))
        )
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


def _coerce_numeric_columns(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert fitting columns to numeric values with a column-specific error."""
    converted = data.copy()
    for col in columns:
        try:
            converted[col] = pd.to_numeric(converted[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{col}' must contain only numeric values.") from exc
    return converted


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
    ppm_view = data.loc[:, ppm_cols_list]
    ppm_values = np.asarray(ppm_view, dtype=float)
    incomplete_by_col = ~np.all(np.isfinite(ppm_values), axis=0)
    dropped_ppm = [col for col, incomplete in zip(ppm_cols_list, incomplete_by_col) if bool(incomplete)]
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


def _validate_ppm_array(y: np.ndarray, ppm_cols: Sequence[str]) -> None:
    """Ensure every retained peak has at least one finite observation."""
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("No ppm observations remain after input filtering.")
    finite = np.isfinite(y)
    if not np.any(finite):
        raise ValueError("No finite ppm observations remain after input filtering.")
    empty_cols = [col for col, has_value in zip(ppm_cols, np.any(finite, axis=0)) if not has_value]
    if empty_cols:
        raise ValueError(
            "ppm columns contain no finite observations: " + ", ".join(map(str, empty_cols))
        )


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
    path = Path(path)
    _validate_input_file(path)
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"Input dataset contains no data rows: {path}")
    columns = list(df.columns)
    host_col, guest_col = _resolve_concentration_columns(columns)
    ppm_cols = _resolve_ppm_cols(columns, ppm_cols)

    data = _subset_input_columns(df, host_col, guest_col, ppm_cols)
    data = _drop_missing_required(data, host_col, guest_col)
    if data.empty:
        raise ValueError("No observations remain after dropping rows with missing concentration values.")
    data = _coerce_numeric_columns(data, [host_col, guest_col, *ppm_cols])

    ppm_data, ppm_cols, dropped_ppm = _apply_missing_policy(data, ppm_cols, missing_policy)
    use_cols = [host_col, guest_col] + ppm_cols
    data = data.loc[:, use_cols].copy()

    # Extract numeric arrays and validate concentrations.
    h_tot = np.asarray(data.loc[:, host_col], dtype=float)
    g_tot = np.asarray(data.loc[:, guest_col], dtype=float)
    _validate_concentration_arrays(h_tot, g_tot)

    # Extract ppm values.
    y = np.asarray(ppm_data, dtype=float)
    _validate_ppm_array(y, ppm_cols)
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
