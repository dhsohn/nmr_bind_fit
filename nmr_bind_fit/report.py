"""Report writers (summary CSV, decision text, HTML) for fit results."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
    stats: dict[str, str]
    params: list[ParamEntry]
    plots: list[str]
    warnings: list[str]
    plot_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class DecisionEntry:
    dataset: str
    recommended_model: str
    reasons: list[str]


def _rationale_text(reasons: Sequence[str]) -> str:
    cleaned = []
    for reason in reasons:
        text = reason.strip()
        if not text:
            continue
        text = text.removesuffix(".")
        cleaned.append(text)
    if not cleaned:
        return "No supporting criteria were recorded."
    return ". ".join(cleaned) + "."


def _decision_paragraphs(entries: Sequence[DecisionEntry]) -> list[str]:
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


def write_report_html(
    summary_rows: Sequence[dict[str, str]],
    model_entries: Sequence[ModelEntry],
    decision_entries: Sequence[DecisionEntry] | None,
    methods_text: str | None,
    warnings: Sequence[str] | None,
    output_path: Path,
    *,
    methods_sections: Sequence[dict[str, str]] | None = None,
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
