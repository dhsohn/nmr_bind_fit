"""Generate deterministic synthetic CSV examples for nmr_bind_fit."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS, predict_dataset

ROOT = Path(__file__).resolve().parent


def _base_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_tot = np.full(10, 1.0e-3)
    equivalents = np.array([0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    g_tot = h_tot * equivalents
    return h_tot, g_tot, equivalents


def _dataset(name: str, h_tot: np.ndarray, g_tot: np.ndarray, y_cols: list[str]) -> Dataset:
    return Dataset(
        name=name,
        path=ROOT / name,
        h_tot=h_tot,
        g_tot=g_tot,
        x=g_tot / h_tot,
        y=np.zeros((h_tot.size, len(y_cols))),
        y_cols=y_cols,
        dropped_peaks=[],
    )


def _write_csv(path: Path, columns: list[str], rows: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([f"{value:.6g}" for value in row])


def _write_model_example(filename: str, model_name: str, logk: list[float], delta: list[list[float]]) -> None:
    h_tot, g_tot, _ = _base_grid()
    y_cols = ["ppm_H1", "ppm_H2"]
    ds = _dataset(filename, h_tot, g_tot, y_cols)
    y_pred, _ = predict_dataset(
        MODEL_SPECS[model_name],
        ds,
        np.asarray(logk, dtype=float),
        np.asarray(delta, dtype=float),
    )
    rows = []
    for idx in range(h_tot.size):
        rows.append([h_tot[idx], g_tot[idx], *np.round(y_pred[idx, :], 6)])
    _write_csv(ROOT / filename, ["[H]t", "[G]t", *y_cols], rows)


def main() -> None:
    _write_model_example(
        "synthetic_11.csv",
        "11",
        [4.0],
        [
            [7.100, 7.520],
            [8.220, 8.540],
        ],
    )
    _write_model_example(
        "synthetic_12.csv",
        "12",
        [4.1, 3.2],
        [
            [7.100, 7.420, 7.610],
            [8.220, 8.470, 8.630],
        ],
    )
    _write_model_example(
        "synthetic_21.csv",
        "21",
        [4.0, 3.4],
        [
            [7.100, 7.470, 7.690],
            [8.220, 8.510, 8.700],
        ],
    )

    h_tot, g_tot, equivalents = _base_grid()
    rows = []
    for idx in range(h_tot.size):
        rows.append(
            [
                h_tot[idx],
                g_tot[idx],
                round(7.100 + 0.018 * float(equivalents[idx]), 6),
                round(8.220 - 0.012 * float(equivalents[idx]), 6),
            ]
        )
    _write_csv(ROOT / "synthetic_nonbinding.csv", ["[H]t", "[G]t", "ppm_H1", "ppm_H2"], rows)


if __name__ == "__main__":
    main()
