"""Binding model definitions and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .equilibrium import SpeciesResult, solve_11, solve_12, solve_21
from .io import Dataset


@dataclass
class ModelSpec:
    name: str
    n_logk: int
    n_delta_per_peak: int
    species_labels: List[str]
    is_binding: bool


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


def model_param_names(model: ModelSpec, peak_names: List[str]) -> List[str]:
    names: List[str] = []
    if model.n_logk == 1:
        names.append("logK")
    elif model.n_logk == 2:
        names.extend(["logK1", "logK2"])

    for peak in peak_names:
        for label in model.species_labels:
            names.append(f"{label}_{peak}")
    return names


def split_params(
    params: np.ndarray,
    model: ModelSpec,
    dataset: Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    logk = params[: model.n_logk]
    delta = params[model.n_logk :].reshape(dataset.n_peaks, model.n_delta_per_peak)
    return logk, delta


def split_params_multi(
    params: np.ndarray,
    model: ModelSpec,
    datasets: List[Dataset],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    logk = params[: model.n_logk]
    deltas = []
    idx = model.n_logk
    for ds in datasets:
        count = ds.n_peaks * model.n_delta_per_peak
        delta = params[idx : idx + count].reshape(ds.n_peaks, model.n_delta_per_peak)
        deltas.append(delta)
        idx += count
    return logk, deltas


def _weights_11(species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    w_h = species.h / h_tot
    w_hg = species.hg / h_tot
    return np.vstack([w_h, w_hg]).T


def _weights_12(species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    w_h = species.h / h_tot
    w_hg = species.hg / h_tot
    w_hg2 = species.hg2 / h_tot
    return np.vstack([w_h, w_hg, w_hg2]).T


def _weights_21(species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    w_h = species.h / h_tot
    w_hg = species.hg / h_tot
    w_h2g = (2.0 * species.h2g) / h_tot
    return np.vstack([w_h, w_hg, w_h2g]).T


def predict_dataset(
    model: ModelSpec,
    dataset: Dataset,
    logk: np.ndarray,
    delta: np.ndarray,
) -> Tuple[np.ndarray, SpeciesResult]:
    h_tot = dataset.h_tot
    g_tot = dataset.g_tot

    if model.name == "11":
        k = 10 ** float(logk[0])
        species = solve_11(h_tot, g_tot, k)
        weights = _weights_11(species, h_tot)
        y_pred = weights @ delta.T
        return y_pred, species

    if model.name == "12":
        k1 = 10 ** float(logk[0])
        k2 = 10 ** float(logk[1])
        species = solve_12(h_tot, g_tot, k1, k2)
        weights = _weights_12(species, h_tot)
        y_pred = weights @ delta.T
        return y_pred, species

    if model.name == "21":
        k1 = 10 ** float(logk[0])
        k2 = 10 ** float(logk[1])
        species = solve_21(h_tot, g_tot, k1, k2)
        weights = _weights_21(species, h_tot)
        y_pred = weights @ delta.T
        return y_pred, species

    if model.name == "nb":
        a0 = delta[:, 0]
        a1 = delta[:, 1]
        y_pred = a0.reshape(1, -1) + dataset.x.reshape(-1, 1) * a1.reshape(1, -1)
        return y_pred, SpeciesResult(h=h_tot, g=g_tot, hg=np.zeros_like(h_tot))

    raise ValueError(f"Unknown model: {model.name}")


def fraction_bound(model: ModelSpec, species: SpeciesResult, h_tot: np.ndarray) -> np.ndarray:
    if model.name == "11":
        return species.hg / h_tot
    if model.name == "12":
        return (species.hg + species.hg2) / h_tot
    if model.name == "21":
        return (species.hg + 2.0 * species.h2g) / h_tot
    return np.full_like(h_tot, np.nan)
