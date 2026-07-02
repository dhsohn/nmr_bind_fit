"""CLI entry point for NMR binding fits."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fit import fit_models
from .io import load_datasets
from .report import write_decision_txt, write_report_html, write_summary_csv
from .report_pipeline import (
    build_decisions,
    build_methods_sections,
    build_methods_text,
    build_report_artifacts,
)
from .types import DatasetLike, FitResultLike

MODEL_LABELS = {
    "11": "H : G = 1 : 1",
    "12": "H : G = 1 : 2",
    "21": "H : G = 2 : 1",
    "nb": "non-binding",
}

DEFAULT_MODEL_NAMES = ["11", "12", "21", "nb"]
STRICT_MISSING_POLICY = "drop-column"
STRICT_SOLVER_FAILURE_MODE = "fail-fast"
STRICT_K_MIN = 1e0
STRICT_K_MAX = 1e12


def _non_negative_int(value: str) -> int:
    # Argparse type converter that rejects negative integers.
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--bootstrap must be non-negative.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("--bootstrap must be non-negative.")
    return parsed


def _parse_k_starts(value: Optional[str]) -> List[float]:
    # Parse comma-separated starts or default to log-spaced values.
    if not value:
        return [10**i for i in range(1, 9)]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _resolve_inputs(patterns: List[str]) -> List[Path]:
    # Expand glob patterns and validate file existence.
    paths: List[Path] = []
    resolved_paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            path = Path(pattern)
            if path.exists():
                matches = [pattern]
        for match in matches:
            path = Path(match)
            paths.append(path)
            resolved_paths.append(path.resolve())
    if not paths:
        raise FileNotFoundError("No input files found.")
    counts = Counter(resolved_paths)
    duplicates = sorted(str(path) for path, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("Duplicate input files detected: " + ", ".join(duplicates))
    return paths


def _safe_output_name(name: str) -> str:
    # Sanitize output names for filesystem safety.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return safe or "output"


def _auto_output_dir(paths: List[Path]) -> Path:
    # Build a timestamped output directory based on input names.
    now = datetime.now()
    timestamp = f"{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}"
    if len(paths) == 1:
        base = paths[0].stem
    else:
        base = "replicates"
    return Path(f"{timestamp}_{_safe_output_name(base)}")


def _build_dataset_labels(datasets: Sequence[DatasetLike]) -> Dict[int, str]:
    # Build collision-free labels so same-stem files do not overwrite each other.
    labels = [str(getattr(ds, "name", "dataset")) for ds in datasets]
    counts = Counter(labels)

    # First disambiguation pass: append file name.
    for idx, ds in enumerate(datasets):
        if counts[labels[idx]] > 1:
            path = getattr(ds, "path", None)
            filename = Path(path).name if path is not None else labels[idx]
            labels[idx] = f"{labels[idx]} ({filename})"

    counts = Counter(labels)
    # Second pass: append full path if file names still collide.
    for idx, ds in enumerate(datasets):
        if counts[labels[idx]] > 1:
            path = getattr(ds, "path", None)
            path_text = str(path) if path is not None else labels[idx]
            base_name = str(getattr(ds, "name", labels[idx]))
            labels[idx] = f"{base_name} ({path_text})"

    counts = Counter(labels)
    seen: Dict[str, int] = {}
    deduped: List[str] = []
    # Final pass: force uniqueness even if the exact same path is repeated.
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        if counts[label] > 1:
            deduped.append(f"{label} #{seen[label]}")
        else:
            deduped.append(label)

    return {id(ds): label for ds, label in zip(datasets, deduped)}


def _dataset_key(result: FitResultLike, dataset_labels: Dict[int, str]) -> str:
    # Normalize dataset key for grouping summaries.
    if len(result.datasets) == 1:
        ds = result.datasets[0]
        return dataset_labels.get(id(ds), ds.name) or ds.name
    return "Simultaneous Fitting"


def _resolve_logk_config(args: argparse.Namespace) -> Tuple[List[float], Optional[Tuple[float, float]]]:
    k_starts = _parse_k_starts(args.k_starts)
    if not k_starts:
        raise ValueError("--k-starts must include at least one positive value.")
    if any(v <= 0 for v in k_starts):
        raise ValueError("All K starts must be positive.")
    if any(v < STRICT_K_MIN or v > STRICT_K_MAX for v in k_starts):
        raise ValueError(f"All K starts must be within [{STRICT_K_MIN:.0e}, {STRICT_K_MAX:.0e}].")
    if args.bootstrap_logk_jitter < 0:
        raise ValueError("--bootstrap-logk-jitter must be non-negative.")
    logk_starts = [float(np.log10(v)) for v in k_starts]
    logk_bounds = (float(np.log10(STRICT_K_MIN)), float(np.log10(STRICT_K_MAX)))
    return logk_starts, logk_bounds


def _index_results(
    results: Sequence[FitResultLike],
    dataset_labels: Dict[int, str],
) -> Tuple[List[str], Dict[str, Dict[str, FitResultLike]], Dict[str, List[Tuple[str, str]]]]:
    # Build ordered successful result map and per-dataset failure list.
    results_by_key: Dict[str, Dict[str, FitResultLike]] = {}
    failures_by_key: Dict[str, List[Tuple[str, str]]] = {}
    ordered_keys: List[str] = []
    for res in results:
        key = _dataset_key(res, dataset_labels)
        if key not in ordered_keys:
            ordered_keys.append(key)
        if not res.success:
            failures_by_key.setdefault(key, []).append((res.model.name, res.message))
            continue
        results_by_key.setdefault(key, {})[res.model.name] = res
    return ordered_keys, results_by_key, failures_by_key


def _display_model_name(name: str) -> str:
    # Map internal model codes to friendly labels.
    return MODEL_LABELS.get(name, name)


def run_fit(args: argparse.Namespace) -> None:
    # Defensively re-validate for callers that build args directly and bypass
    # the parser (the CLI type converter already rejects negative --bootstrap).
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative.")
    args.bootstrap_ci_method = str(getattr(args, "bootstrap_ci_method", "percentile")).strip().lower()
    if args.bootstrap_ci_method not in {"percentile", "bca"}:
        raise ValueError("--bootstrap-ci-method must be one of: percentile, bca.")
    args.residual_diagnostics = bool(getattr(args, "residual_diagnostics", False))
    # Resolve input patterns and load datasets from disk.
    paths = _resolve_inputs(args.input)
    datasets = load_datasets(
        paths,
        ppm_cols=args.ppm_cols,
        missing_policy=STRICT_MISSING_POLICY,
    )
    dataset_labels = _build_dataset_labels(datasets)

    model_names = DEFAULT_MODEL_NAMES

    logk_starts, logk_bounds = _resolve_logk_config(args)

    # Fit all requested models.
    results = fit_models(
        datasets,
        model_names,
        logk_starts=logk_starts,
        replicates=args.replicates,
        max_nfev=args.max_nfev,
        bootstrap=args.bootstrap,
        bootstrap_method=args.bootstrap_method,
        bootstrap_ci_method=args.bootstrap_ci_method,
        seed=args.seed,
        logk_bounds=logk_bounds,
        logk_jitter=args.bootstrap_logk_jitter,
        solver_failure_mode=STRICT_SOLVER_FAILURE_MODE,
        residual_diagnostics=args.residual_diagnostics,
    )

    # Prepare output directory for reports and plots.
    out_dir = _auto_output_dir(paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index results by dataset key and collect failures.
    ordered_keys, results_by_key, failures_by_key = _index_results(results, dataset_labels)

    summary_rows, model_entries, report_warnings = build_report_artifacts(
        args,
        ordered_keys,
        results_by_key,
        failures_by_key,
        out_dir,
        display_model_name=_display_model_name,
    )

    write_summary_csv(summary_rows, out_dir / "summary.csv")

    methods_text = build_methods_text(args, datasets)
    methods_sections = build_methods_sections(args, datasets)
    decisions, decision_entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
        failures_by_key,
        display_model_name=_display_model_name,
    )

    write_decision_txt(decisions, out_dir / "decision.txt")

    write_report_html(
        summary_rows,
        model_entries,
        decision_entries=decision_entries,
        methods_text=methods_text,
        warnings=report_warnings,
        output_path=out_dir / "report.html",
        methods_sections=methods_sections,
    )


def build_parser() -> argparse.ArgumentParser:
    # Define CLI flags and defaults.
    parser = argparse.ArgumentParser(prog="nmr_bind_fit")
    parser.add_argument(
        "command",
        nargs="?",
        default="fit",
        choices=["fit"],
        help="Command (default: fit)",
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV/XLSX files")
    parser.add_argument("--ppm-cols", default=None, help="Comma-separated ppm columns")
    parser.add_argument("--bootstrap", type=_non_negative_int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--bootstrap-method", choices=["residual", "points", "parametric"], default="residual")
    parser.add_argument(
        "--bootstrap-ci-method",
        choices=["percentile", "bca"],
        default="percentile",
        help="Bootstrap CI method (default: percentile)",
    )
    parser.add_argument(
        "--bootstrap-logk-jitter",
        type=float,
        default=0.1,
        help="Standard deviation for logK jitter per bootstrap refit (log10 units)",
    )
    parser.add_argument("--k-starts", default=None, help="Comma-separated K starts")
    parser.add_argument(
        "--replicates",
        action="store_true",
        help="Fit replicate inputs with shared binding constants",
    )
    parser.add_argument("--max-nfev", type=int, default=5000, help="Max optimizer evaluations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--residual-diagnostics",
        action="store_true",
        help="Compute informational residual diagnostics (Shapiro-Wilk, Durbin-Watson)",
    )
    parser.add_argument(
        "--bootstrap-ci-width",
        type=float,
        default=None,
        help="Warn if bootstrap K CI width exceeds this threshold",
    )
    parser.set_defaults(
        func=run_fit,
        missing_policy=STRICT_MISSING_POLICY,
        solver_failure_mode=STRICT_SOLVER_FAILURE_MODE,
    )

    return parser


def main() -> None:
    # Parse CLI arguments and run the requested command.
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        # Report expected input/validation problems as a clean message and a
        # nonzero exit status instead of an uncaught traceback.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
