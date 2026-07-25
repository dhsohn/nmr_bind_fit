"""Binding model definitions and prediction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .equilibrium import SpeciesResult, solve_11, solve_12, solve_21
from .io import Dataset


@dataclass
class ModelSpec:
    name: str
    n_logk: int
    n_delta_per_peak: int
    species_labels: list[str]
    is_binding: bool


# Model definitions for each supported binding stoichiometry.
MODEL_SPECS = {
    "11": ModelSpec(
        name="11",
        n_logk=1,
        n_delta_per_peak=2,
        species_labels=["H", "HG"],
        is_binding=True,
    ),
    "12": ModelSpec(
        name="12",
        n_logk=2,
        n_delta_per_peak=3,
        species_labels=["H", "HG", "HG2"],
        is_binding=True,
    ),
    "21": ModelSpec(
        name="21",
        n_logk=2,
        n_delta_per_peak=3,
        species_labels=["H", "HG", "H2G"],
        is_binding=True,
    ),
    "nb": ModelSpec(
        name="nb",
        n_logk=0,
        n_delta_per_peak=2,
        species_labels=["a0", "a1"],
        is_binding=False,
    ),
}


MODEL_LABELS = {
    "11": "H : G = 1 : 1",
    "12": "H : G = 1 : 2",
    "21": "H : G = 2 : 1",
    "nb": "non-binding",
}


def display_model_name(name: str) -> str:
    """Return the report label for a model code."""
    return MODEL_LABELS.get(name, name)


def split_params_multi(
    params: np.ndarray,
    model: ModelSpec,
    datasets: Sequence[Dataset],
) -> tuple[np.ndarray, list[np.ndarray]]:
    # Walk the parameter vector across datasets for replicate fits.
    params = np.asarray(params, dtype=float)
    logk = params[: model.n_logk]
    deltas = []
    idx = model.n_logk
    for ds in datasets:
        count = ds.n_peaks * model.n_delta_per_peak
        if idx + count > params.size:
            raise ValueError("Parameter vector is shorter than expected for the selected model/datasets.")
        delta = params[idx : idx + count].reshape(ds.n_peaks, model.n_delta_per_peak)
        deltas.append(delta)
        idx += count
    if idx != params.size:
        raise ValueError("Parameter vector has unused trailing values for the selected model/datasets.")
    return logk, deltas


def _logk_to_k(value: float, label: str) -> float:
    logk_value = float(value)
    if not np.isfinite(logk_value):
        raise ValueError(f"{label} must be finite.")
    try:
        k = 10.0 ** logk_value
    except OverflowError as exc:
        raise ValueError(f"{label} overflows when converted to K.") from exc
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"{label} must convert to a positive finite K.")
    return float(k)


def _weights_11(species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    # Host-based fractions for host resonances under fast exchange.
    w_h = species.h / h_tot
    w_hg = species.hg / h_tot
    return np.vstack([w_h, w_hg]).T


def _weights_12(species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    # Host-based fractions for host resonances under fast exchange.
    hg2 = species.hg2
    if hg2 is None:
        raise ValueError("1:2 species populations are missing HG2.")
    w_h = species.h / h_tot
    w_hg = species.hg / h_tot
    w_hg2 = hg2 / h_tot
    return np.vstack([w_h, w_hg, w_hg2]).T


def _weights_21(species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    # H2G contributes two host units, so its weight is doubled.
    h2g = species.h2g
    if h2g is None:
        raise ValueError("2:1 species populations are missing H2G.")
    w_h = species.h / h_tot
    w_hg = species.hg / h_tot
    w_h2g = (2.0 * h2g) / h_tot
    return np.vstack([w_h, w_hg, w_h2g]).T


def predict_dataset(
    model: ModelSpec,
    dataset: Dataset,
    logk: np.ndarray,
    delta: np.ndarray,
    solver_failure_mode: str = "fail-fast",
) -> tuple[np.ndarray, SpeciesResult]:
    """Return predicted shifts and species populations for one dataset."""
    h_tot = dataset.h_tot
    g_tot = dataset.g_tot

    if model.name == "11":
        # Fit parameters are in log10(K); convert to linear K for equilibrium.
        k = _logk_to_k(logk[0], "logK")
        species = solve_11(h_tot, g_tot, k)
        weights = _weights_11(species, h_tot)
        y_pred = weights @ delta.T
        return y_pred, species

    if model.name == "12":
        # Stepwise binding constants for 1:2.
        k1 = _logk_to_k(logk[0], "logK1")
        k2 = _logk_to_k(logk[1], "logK2")
        species = solve_12(h_tot, g_tot, k1, k2, failure_mode=solver_failure_mode)
        weights = _weights_12(species, h_tot)
        y_pred = weights @ delta.T
        return y_pred, species

    if model.name == "21":
        # Stepwise binding constants for 2:1.
        k1 = _logk_to_k(logk[0], "logK1")
        k2 = _logk_to_k(logk[1], "logK2")
        species = solve_21(h_tot, g_tot, k1, k2, failure_mode=solver_failure_mode)
        weights = _weights_21(species, h_tot)
        y_pred = weights @ delta.T
        return y_pred, species

    if model.name == "nb":
        # Non-binding model: linear drift with equivalents.
        a0 = delta[:, 0]
        a1 = delta[:, 1]
        y_pred = a0.reshape(1, -1) + dataset.x.reshape(-1, 1) * a1.reshape(1, -1)
        return y_pred, SpeciesResult(h=h_tot, g=g_tot, hg=np.zeros_like(h_tot))

    raise ValueError(f"Unknown model: {model.name}")


def fraction_bound(model: ModelSpec, species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    """Host-basis bound fraction for host-resonance data."""
    # Normalize bound species on a per-host basis for host-only resonances.
    if model.name == "11":
        return species.hg / h_tot
    if model.name == "12":
        hg2 = species.hg2
        if hg2 is None:
            raise ValueError("1:2 species populations are missing HG2.")
        return (species.hg + hg2) / h_tot
    if model.name == "21":
        h2g = species.h2g
        if h2g is None:
            raise ValueError("2:1 species populations are missing H2G.")
        return (species.hg + 2.0 * h2g) / h_tot
    return np.full_like(h_tot, np.nan)
