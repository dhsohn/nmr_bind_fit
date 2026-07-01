"""Report writers (summary CSV, decision text, HTML) for fit results."""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .report_html_renderer import write_report_html as _write_report_html_impl


@dataclass
class ParamEntry:
    name: str
    value: float
    se: float


@dataclass
class ModelEntry:
    dataset: str
    model: str
    stats: Dict[str, str]
    params: List[ParamEntry]
    plots: List[str]
    warnings: List[str]


@dataclass
class DecisionEntry:
    dataset: str
    recommended_model: str
    reasons: List[str]


def _rationale_text(reasons: Sequence[str]) -> str:
    cleaned = []
    for reason in reasons:
        text = reason.strip()
        if not text:
            continue
        if text.endswith("."):
            text = text[:-1]
        cleaned.append(text)
    if not cleaned:
        return "No supporting criteria were recorded."
    return ". ".join(cleaned) + "."


def _decision_paragraphs(entries: Sequence[DecisionEntry]) -> List[str]:
    paragraphs = []
    for entry in entries:
        rationale = _rationale_text(entry.reasons)
        if entry.dataset == "Simultaneous Fitting":
            text = (
                "Based on simultaneous fitting of replicate titration datasets, the "
                f"{entry.recommended_model} model was selected as a provisional working model among the evaluated "
                f"candidates. {rationale}"
            )
        else:
            text = (
                f"For the {entry.dataset} dataset, the {entry.recommended_model} model was selected as a provisional "
                f"working model among the evaluated candidates. {rationale}"
            )
        paragraphs.append(f"<p>{html.escape(text)}</p>")
    return paragraphs


def write_summary_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_decision_txt(decisions: Sequence[str], path: Path) -> None:
    with path.open("w") as f:
        for line in decisions:
            f.write(line.rstrip() + "\n")


def write_report_html(
    summary_rows: Sequence[Dict[str, str]],
    model_entries: Sequence[ModelEntry],
    decision_entries: Optional[Sequence[DecisionEntry]],
    methods_text: Optional[str],
    warnings: Optional[Sequence[str]],
    output_path: Path,
    *,
    methods_sections: Optional[Sequence[Dict[str, str]]] = None,
) -> None:
    _write_report_html_impl(
        summary_rows=summary_rows,
        model_entries=model_entries,
        decision_entries=decision_entries,
        methods_text=methods_text,
        warnings=warnings,
        output_path=output_path,
        methods_sections=methods_sections,
        decision_paragraphs_fn=_decision_paragraphs,
    )
