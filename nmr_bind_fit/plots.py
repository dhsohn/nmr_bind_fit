"""Plotting helpers – publication-quality figures."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import Dataset
from .models import ModelSpec, fraction_bound, predict_dataset
from .types import DatasetLike

# ── Publication-quality defaults ──────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

# Colour palette — muted academic style
_CLR_DATA = "#2b2d42"
_CLR_FIT = "#d90429"
_CLR_ZERO = "#94a3b8"
_CLR_HIST = "#334155"


def _safe_file_stem(value: str) -> str:
    """Return a bounded ASCII stem suitable for compatibility callers."""
    text = str(value)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "peak"
    if len(safe) <= 64:
        return safe
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    prefix = safe[:51].rstrip("._-") or "peak"
    return f"{prefix}-{digest}"


def _safe_file_stems(values: List[str]) -> List[str]:
    """Sanitize peak labels into deterministic, collision-free stems.

    Plot artifacts use :func:`_peak_file_token`, but this helper remains for
    callers that relied on the readable stems introduced on ``main``.
    """
    stems = [_safe_file_stem(value) for value in values]
    counts = Counter(stems)
    used: set[str] = set()
    unique: List[str] = []
    for idx, stem in enumerate(stems, start=1):
        candidates = []
        if counts[stem] == 1:
            candidates.append(stem)
        candidates.append(f"{idx:02d}_{stem}")
        chosen = next((candidate for candidate in candidates if candidate not in used), "")
        if not chosen:
            suffix = 1
            chosen = f"{idx:02d}_{stem}_{suffix}"
            while chosen in used:
                suffix += 1
                chosen = f"{idx:02d}_{stem}_{suffix}"
        used.add(chosen)
        unique.append(chosen)
    return unique


def _peak_file_token(peak: str, index: int) -> str:
    """Return a bounded, collision-resistant filesystem token for a peak label."""
    digest = hashlib.sha256(str(peak).encode("utf-8")).hexdigest()[:16]
    return f"peak-{index:04d}-{digest}"


def _style_axes(ax: plt.Axes) -> None:
    """Remove top/right spines and lighten remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.tick_params(colors="#475569")


def _grid_dataset(ds: DatasetLike, n: int = 200) -> DatasetLike:
    # Create a dense grid of equivalents for smooth fit curves.
    eq_obs = np.asarray(ds.x, dtype=float)
    h_obs = np.asarray(ds.h_tot, dtype=float)
    eq_vals = np.linspace(np.min(eq_obs), np.max(eq_obs), n)

    finite_h = h_obs[np.isfinite(h_obs)]
    h_ref = float(np.median(finite_h)) if finite_h.size else 1.0
    if finite_h.size and np.isfinite(h_ref):
        h_mean = float(np.mean(finite_h))
        h_cv = float(np.std(finite_h) / abs(h_mean)) if h_mean != 0 else float("inf")
    else:
        h_cv = 0.0

    if h_cv <= 0.01:
        h_vals = np.full_like(eq_vals, h_ref)
    else:
        mask = np.isfinite(eq_obs) & np.isfinite(h_obs)
        if np.count_nonzero(mask) < 2:
            h_vals = np.full_like(eq_vals, h_ref)
        else:
            eq_sorted = eq_obs[mask]
            h_sorted = h_obs[mask]
            order = np.argsort(eq_sorted)
            eq_sorted = eq_sorted[order]
            h_sorted = h_sorted[order]

            eq_unique, inv = np.unique(eq_sorted, return_inverse=True)
            h_unique = np.zeros_like(eq_unique, dtype=float)
            counts = np.zeros_like(eq_unique, dtype=float)
            np.add.at(h_unique, inv, h_sorted)
            np.add.at(counts, inv, 1.0)
            h_unique = h_unique / counts

            if eq_unique.size == 1:
                h_vals = np.full_like(eq_vals, float(h_unique[0]))
            else:
                h_vals = np.interp(eq_vals, eq_unique, h_unique)
    g_vals = eq_vals * h_vals
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
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
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
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(ds.x, ds.y[:, i], color=_CLR_DATA, s=28, zorder=3,
                   label="Observed", edgecolors="white", linewidths=0.4)
        ax.plot(x_curve, y_curve[:, i], color=_CLR_FIT, linewidth=1.6,
                label="Fitted", zorder=2)
        ax.set_xlabel(r"$[\mathrm{G}]_{\mathrm{t}}$ / $[\mathrm{H}]_{\mathrm{t}}$")
        ax.set_ylabel(r"$\delta$ (ppm)")
        ax.legend(frameon=False, loc="best")
        _style_axes(ax)
        fig.tight_layout()
        peak_token = _peak_file_token(peak, i + 1)
        png_path = out_dir / f"isotherm_{peak_token}.png"
        pdf_path = out_dir / f"isotherm_{peak_token}.pdf"
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
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(0.0, color=_CLR_ZERO, linewidth=0.8, linestyle="--")
        ax.scatter(ds.x, residuals[:, i], color=_CLR_DATA, s=24, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.set_xlabel(r"$[\mathrm{G}]_{\mathrm{t}}$ / $[\mathrm{H}]_{\mathrm{t}}$")
        ax.set_ylabel("Residual (ppm)")
        _style_axes(ax)
        fig.tight_layout()
        peak_token = _peak_file_token(peak, i + 1)
        png_path = out_dir / f"residual_{peak_token}.png"
        pdf_path = out_dir / f"residual_{peak_token}.pdf"
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
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(samples[:, i], bins=25, color=_CLR_HIST, alpha=0.85,
                edgecolor="white", linewidth=0.5)
        if name == "K":
            xlabel = r"$K$ (M$^{-1}$)"
        elif name.startswith("K") and name[1:].isdigit():
            xlabel = rf"$K_{name[1:]}$ (M$^{{-1}}$)"
        else:
            xlabel = f"{name} (M$^{{-1}}$)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        _style_axes(ax)
        fig.tight_layout()
        png_path = out_dir / f"bootstrap_{name}.png"
        pdf_path = out_dir / f"bootstrap_{name}.pdf"
        _save_figure(fig, png_path, pdf_path)
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
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(ds.x, f, color=_CLR_DATA, s=28, zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.set_xlabel(r"$[\mathrm{G}]_{\mathrm{t}}$ / $[\mathrm{H}]_{\mathrm{t}}$")
    ax.set_ylabel("Host Fraction Bound")
    ax.set_ylim(-0.02, 1.05)
    _style_axes(ax)
    fig.tight_layout()
    png_path = out_dir / "fraction_bound.png"
    pdf_path = out_dir / "fraction_bound.pdf"
    _save_figure(fig, png_path, pdf_path)
    return [png_path, pdf_path]
