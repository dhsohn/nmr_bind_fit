"""CLI entry point for NMR binding fits."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .fit import FitResult, fit_models
from .io import Dataset, load_datasets
from .models import display_model_name
from .report import write_report_html
from .report_pipeline import (
    build_decisions,
    build_methods_sections,
    build_report_artifacts,
)

DEFAULT_MODEL_NAMES = ["11", "12", "21", "nb"]
STRICT_SOLVER_FAILURE_MODE = "fail-fast"
STRICT_K_MIN = 1e0
STRICT_K_MAX = 1e12
SIMULTANEOUS_FIT_LABEL = "Simultaneous Fitting"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parse_k_starts(value: str | None) -> list[float]:
    # Parse comma-separated starts or default to log-spaced values.
    if value is None:
        return [10**i for i in range(1, 9)]
    try:
        return [float(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError("--k-starts must be a comma-separated list of numeric values.") from exc


def _resolve_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    resolved_paths: list[Path] = []

    for pattern in patterns:
        literal = Path(pattern)
        if literal.exists():
            # An input that names an existing path is used verbatim, so a
            # filename containing glob metacharacters is never silently replaced
            # by a different file the pattern would otherwise match.
            candidates = [literal]
        else:
            candidates = [Path(match) for match in sorted(glob.glob(pattern))]
            if not candidates:
                raise FileNotFoundError(f"Input file or pattern not found: {pattern}")
        for path in candidates:
            if not path.is_file():
                raise ValueError(f"Input path is not a regular file: {path}")
            paths.append(path)
            resolved_paths.append(path.resolve())

    if not paths:
        raise FileNotFoundError("No input files found.")
    duplicates = sorted(str(path) for path, count in Counter(resolved_paths).items() if count > 1)

    if duplicates:
        raise ValueError("Duplicate input files detected: " + ", ".join(duplicates))
    return paths


def _safe_output_name(name: str) -> str:
    # Normalize a user-derived output directory label and limit its length.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    safe = safe or "output"
    safe = safe[:80].rstrip("._-")
    return safe or "output"


def _auto_output_dir(paths: list[Path]) -> Path:
    # Local wall-clock time, kept timezone-aware so the stamp is unambiguous.
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    base = paths[0].stem if len(paths) == 1 else "replicates"
    return Path(f"{timestamp}_{_safe_output_name(base)}")


def _reserve_output_dir(paths: list[Path]) -> Path:
    base = _auto_output_dir(paths)
    candidate = base
    suffix = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = base.with_name(f"{base.name}_{suffix:02d}")
            suffix += 1


def _build_dataset_labels(datasets: Sequence[Dataset]) -> dict[int, str]:
    names = [dataset.name for dataset in datasets]
    if len(names) == len(set(names)):
        return {id(dataset): dataset.name for dataset in datasets}
    return {
        id(dataset): f"{index}. {dataset.name}"
        for index, dataset in enumerate(datasets, start=1)
    }


def _resolve_logk_config(
    args: argparse.Namespace,
) -> tuple[list[float], tuple[float, float]]:
    k_starts = _parse_k_starts(args.k_starts)
    if not k_starts:
        raise ValueError("--k-starts must include at least one positive value.")
    if any(not np.isfinite(v) for v in k_starts):
        raise ValueError("All K starts must be finite.")
    if any(v <= 0 for v in k_starts):
        raise ValueError("All K starts must be positive.")
    if any(v < STRICT_K_MIN or v > STRICT_K_MAX for v in k_starts):
        raise ValueError(f"All K starts must be within [{STRICT_K_MIN:.0e}, {STRICT_K_MAX:.0e}].")
    logk_starts = [float(np.log10(v)) for v in k_starts]
    logk_bounds = (float(np.log10(STRICT_K_MIN)), float(np.log10(STRICT_K_MAX)))
    return logk_starts, logk_bounds


def _index_results(
    results: Sequence[FitResult],
    dataset_labels: dict[int, str],
) -> dict[str, dict[str, FitResult]]:
    results_by_key: dict[str, dict[str, FitResult]] = {}
    for result in results:
        if len(result.datasets) > 1:
            key = SIMULTANEOUS_FIT_LABEL
        else:
            key = dataset_labels[id(result.datasets[0])]
        model_results = results_by_key.setdefault(key, {})
        model_results[result.model.name] = result
    return results_by_key


def run_fit(args: argparse.Namespace) -> None:
    logk_starts, logk_bounds = _resolve_logk_config(args)

    # Resolve input patterns and load datasets from disk.
    paths = _resolve_inputs(args.input)
    datasets = load_datasets(paths, ppm_cols=args.ppm_cols)
    if args.replicates and len(datasets) < 2:
        raise ValueError("--replicates requires at least two input datasets.")
    dataset_labels = _build_dataset_labels(datasets)

    # Fit all requested models.
    results = fit_models(
        datasets,
        DEFAULT_MODEL_NAMES,
        logk_starts=logk_starts,
        replicates=args.replicates,
        max_nfev=args.max_nfev,
        logk_bounds=logk_bounds,
        solver_failure_mode=STRICT_SOLVER_FAILURE_MODE,
        residual_diagnostics=args.residual_diagnostics,
    )

    # Index every result by its report dataset and model keys.
    results_by_key = _index_results(results, dataset_labels)
    ordered_keys = list(results_by_key)
    if not any(result.success for model_map in results_by_key.values() for result in model_map.values()):
        failure_details = [
            f"{key} / {display_model_name(model_name)}: {result.message}"
            for key in ordered_keys
            for model_name, result in results_by_key[key].items()
            if not result.success
        ]
        detail_text = "; ".join(failure_details)
        message = "All model fits failed; no report was generated."
        if detail_text:
            message = f"{message} {detail_text}"
        raise ValueError(message)

    # Prepare output directory for reports and plots only after at least one
    # model has produced a reportable result.
    out_dir = _reserve_output_dir(paths)

    summary_rows, model_entries, report_warnings = build_report_artifacts(
        args,
        ordered_keys,
        results_by_key,
        out_dir,
    )

    methods_sections = build_methods_sections(args, datasets)
    decision_entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
    )

    write_report_html(
        summary_rows,
        model_entries,
        decision_entries=decision_entries,
        warnings=report_warnings,
        output_path=out_dir / "report.html",
        methods_sections=methods_sections,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nmr_bind_fit")
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV/XLSX files")
    parser.add_argument("--ppm-cols", default=None, help="Comma-separated ppm columns")
    parser.add_argument("--k-starts", default=None, help="Comma-separated K starts")
    parser.add_argument(
        "--replicates",
        action="store_true",
        help="Fit replicate inputs with shared binding constants",
    )
    parser.add_argument("--max-nfev", type=_positive_int, default=5000, help="Max optimizer evaluations")
    parser.add_argument(
        "--residual-diagnostics",
        action="store_true",
        help="Compute informational residual diagnostics (Shapiro-Wilk, Durbin-Watson)",
    )
    parser.add_argument(
        "--ci-width",
        type=_positive_float,
        default=None,
        help="Warn if the K confidence-interval width exceeds this threshold",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_fit(args)
    except (FileNotFoundError, ValueError) as exc:
        # Report expected input/validation problems as a clean message and a
        # nonzero exit status instead of an uncaught traceback.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
