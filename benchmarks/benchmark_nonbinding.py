"""Time one candidate-model fit on the bundled non-binding example."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from nmr_bind_fit.fit import fit_models
from nmr_bind_fit.io import load_dataset
from nmr_bind_fit.models import MODEL_SPECS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "examples" / "synthetic_nonbinding.csv"
MODEL_NAMES = ("11", "12", "21", "nb")
DEFAULT_LOGK_STARTS = tuple(float(value) for value in range(1, 9))


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--logk-starts",
        nargs="+",
        type=float,
        default=list(DEFAULT_LOGK_STARTS),
        help="log10(K) starts; defaults mirror the CLI's K=1e1...1e8 grid",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=5000,
        help="maximum optimizer evaluations per start (CLI default: 5000)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_dataset(args.input)
    config = {
        "event": "config",
        "input": str(args.input.resolve()),
        "model": args.model,
        "logk_starts": args.logk_starts,
        "max_nfev": args.max_nfev,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    print(json.dumps(config, sort_keys=True), flush=True)

    model = MODEL_SPECS[args.model]
    n_starts = len(args.logk_starts) ** model.n_logk
    started = perf_counter()
    result = fit_models(
        [dataset],
        [args.model],
        logk_starts=args.logk_starts,
        logk_bounds=(0.0, 12.0),
        max_nfev=args.max_nfev,
        bootstrap=0,
    )[0]
    elapsed = perf_counter() - started
    fitted_logk = np.asarray(result.params[: model.n_logk], dtype=float)
    record = {
        "event": "result",
        "model": args.model,
        "n_starts": n_starts,
        "elapsed_seconds": round(elapsed, 6),
        "success": bool(result.success),
        "message": result.message,
        "bic": _finite_or_none(result.bic),
        "logk": [_finite_or_none(value) for value in fitted_logk],
    }
    print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
