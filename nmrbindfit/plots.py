"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import Dataset
from .models import ModelSpec, predict_dataset, fraction_bound
from .types import DatasetLike

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def _grid_dataset(ds: DatasetLike, n: int = 200) -> DatasetLike:
    # Create a dense grid of equivalents for smooth fit curves.
    eq_vals = np.linspace(np.min(ds.x), np.max(ds.x), n)
    h_ref = float(np.median(ds.h_tot))
    h_vals = np.full_like(eq_vals, h_ref)
    g_vals = eq_vals * h_ref
    x_vals = eq_vals

    return Dataset(
        name=ds.name,
        path=ds.path,
        h_tot=h_vals,
        g_tot=g_vals,
        x=x_vals,
        y=np.zeros((len(x_vals), ds.n_peaks)),
        y_cols=ds.y_cols,
        dropped_peaks=ds.dropped_peaks,
    )


def _save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    # Keep output format handling in one place.
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


def _prepare_isotherm_curve(
    model: ModelSpec,
    ds: DatasetLike,
    logk: np.ndarray,
    delta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Compute smooth model curve values on a dense equivalents grid.
    grid_ds = _grid_dataset(ds)
    y_grid, _ = predict_dataset(model, grid_ds, logk, delta)
    order = np.argsort(grid_ds.x)
    return grid_ds.x[order], y_grid[order, :]


def _prepare_fraction_bound_values(
    model: ModelSpec,
    ds: DatasetLike,
    logk: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    # Compute model-implied fraction bound for the observed x values.
    _, species = predict_dataset(model, ds, logk, delta)
    return fraction_bound(model, species, ds.h_tot)


def plot_isotherms(
    model: ModelSpec,
    ds: DatasetLike,
    logk: np.ndarray,
    delta: np.ndarray,
    out_dir: Path,
) -> List[Path]:
    # Plot data points alongside the fitted curve for each peak.
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []

    x_curve, y_curve = _prepare_isotherm_curve(model, ds, logk, delta)

    for i, peak in enumerate(ds.y_cols):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ds.x, ds.y[:, i], color="#2b2d42", label="data")
        ax.plot(x_curve, y_curve[:, i], color="#d90429", label="fit")
        ax.set_xlabel(r"[G]$_t$ / [H]$_t$")
        ax.set_ylabel("ppm")
        ax.legend()
        fig.tight_layout()
        png_path = out_dir / f"isotherm_{peak}.png"
        pdf_path = out_dir / f"isotherm_{peak}.pdf"
        _save_figure(fig, png_path, pdf_path)
        files.extend([png_path, pdf_path])
    return files


def plot_residuals(
    model: ModelSpec,
    ds: DatasetLike,
    residuals: np.ndarray,
    out_dir: Path,
) -> List[Path]:
    # Plot residuals by peak with a zero baseline.
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for i, peak in enumerate(ds.y_cols):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axhline(0.0, color="#6c757d", linewidth=1)
        ax.scatter(ds.x, residuals[:, i], color="#2b2d42")
        ax.set_xlabel(r"[G]$_t$ / [H]$_t$")
        ax.set_ylabel("residual")
        fig.tight_layout()
        png_path = out_dir / f"residual_{peak}.png"
        pdf_path = out_dir / f"residual_{peak}.pdf"
        _save_figure(fig, png_path, pdf_path)
        files.extend([png_path, pdf_path])
    return files


def plot_bootstrap_hist(
    samples: np.ndarray,
    names: List[str],
    out_dir: Path,
    title: str = "bootstrap K",
) -> List[Path]:
    # Draw bootstrap histograms for K (or K1/K2) samples.
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    if samples.size == 0:
        return files
    for i, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(samples[:, i], bins=20, color="#2b2d42", alpha=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        fig.tight_layout()
        png_path = out_dir / f"bootstrap_{name}.png"
        pdf_path = out_dir / f"bootstrap_{name}.pdf"
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        files.extend([png_path, pdf_path])
    return files


def plot_fraction_bound(
    model: ModelSpec,
    ds: DatasetLike,
    logk: np.ndarray,
    delta: np.ndarray,
    out_dir: Path,
) -> List[Path]:
    # Skip non-binding models since they do not define fraction bound.
    if not model.is_binding:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    f = _prepare_fraction_bound_values(model, ds, logk, delta)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(ds.x, f, color="#2b2d42")
    ax.set_xlabel(r"[G]$_t$ / [H]$_t$")
    ax.set_ylabel("fraction bound")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    png_path = out_dir / "fraction_bound.png"
    pdf_path = out_dir / "fraction_bound.pdf"
    _save_figure(fig, png_path, pdf_path)
    return [png_path, pdf_path]
