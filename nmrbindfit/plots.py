"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import Dataset
from .models import ModelSpec, predict_dataset, fraction_bound


def _grid_dataset(ds: Dataset, xaxis: str, n: int = 200) -> Dataset:
    if xaxis == "guest":
        g_vals = np.linspace(np.min(ds.g_tot), np.max(ds.g_tot), n)
        h_ref = float(np.median(ds.h_tot))
        h_vals = np.full_like(g_vals, h_ref)
        x_vals = g_vals
    else:
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
        sigma=None,
    )


def plot_isotherms(
    model: ModelSpec,
    ds: Dataset,
    logk: np.ndarray,
    delta: np.ndarray,
    xaxis: str,
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []

    grid_ds = _grid_dataset(ds, xaxis)
    y_grid, _ = predict_dataset(model, grid_ds, logk, delta)

    for i, peak in enumerate(ds.y_cols):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ds.x, ds.y[:, i], color="#2b2d42", label="data")
        order = np.argsort(grid_ds.x)
        ax.plot(grid_ds.x[order], y_grid[order, i], color="#d90429", label="fit")
        ax.set_xlabel("guest" if xaxis == "guest" else "eq")
        ax.set_ylabel("ppm")
        ax.set_title(f"{model.name} fit - {peak}")
        ax.legend()
        fig.tight_layout()
        png_path = out_dir / f"isotherm_{peak}.png"
        pdf_path = out_dir / f"isotherm_{peak}.pdf"
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        files.extend([png_path, pdf_path])
    return files


def plot_residuals(
    model: ModelSpec,
    ds: Dataset,
    residuals: np.ndarray,
    xaxis: str,
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for i, peak in enumerate(ds.y_cols):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axhline(0.0, color="#6c757d", linewidth=1)
        ax.scatter(ds.x, residuals[:, i], color="#2b2d42")
        ax.set_xlabel("guest" if xaxis == "guest" else "eq")
        ax.set_ylabel("residual")
        ax.set_title(f"{model.name} residual - {peak}")
        fig.tight_layout()
        png_path = out_dir / f"residual_{peak}.png"
        pdf_path = out_dir / f"residual_{peak}.pdf"
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        files.extend([png_path, pdf_path])
    return files


def plot_bootstrap_hist(
    samples: np.ndarray,
    names: List[str],
    out_dir: Path,
    title: str = "bootstrap K",
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    if samples.size == 0:
        return files
    for i, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(samples[:, i], bins=20, color="#2b2d42", alpha=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.set_title(title)
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
    ds: Dataset,
    logk: np.ndarray,
    delta: np.ndarray,
    xaxis: str,
    out_dir: Path,
) -> List[Path]:
    if not model.is_binding:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred, species = predict_dataset(model, ds, logk, delta)
    f = fraction_bound(model, species, ds.h_tot)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(ds.x, f, color="#2b2d42")
    ax.set_xlabel("guest" if xaxis == "guest" else "eq")
    ax.set_ylabel("fraction bound")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model.name} fraction bound")
    fig.tight_layout()
    png_path = out_dir / "fraction_bound.png"
    pdf_path = out_dir / "fraction_bound.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]
