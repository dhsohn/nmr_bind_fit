"""Report generation for NMR binding fits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import html


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
        if entry.dataset in {"GLOBAL", "Simultaneous Fitting"}:
            model_name = entry.recommended_model.replace(" : ", ":")
            if model_name == "non-binding":
                text = (
                    "Based on simultaneous fitting of replicate titration datasets, the non-binding model "
                    f"is recommended. {rationale}"
                )
            else:
                text = (
                    "Based on simultaneous fitting of replicate titration datasets, the "
                    f"{model_name} binding model is recommended. {rationale}"
                )
        else:
            text = (
                f"For the {entry.dataset} dataset, the {entry.recommended_model} model is recommended. "
                f"{rationale}"
            )
        paragraphs.append(f"<p>{html.escape(text)}</p>")
    return paragraphs


def write_summary_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    import csv

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


def _table_from_rows(rows: Sequence[Dict[str, str]]) -> str:
    if not rows:
        return "<p>No summary rows.</p>"
    headers = rows[0].keys()
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(row[h]))}</td>" for h in headers)
        body_rows.append(f"<tr>{tds}</tr>")
    tbody = "".join(body_rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def write_report_html(
    summary_rows: Sequence[Dict[str, str]],
    model_entries: Sequence[ModelEntry],
    decision_entries: Optional[Sequence[DecisionEntry]],
    methods_text: Optional[str],
    output_path: Path,
) -> None:
    summary_table = _table_from_rows(summary_rows)
    decision_paragraphs = _decision_paragraphs(decision_entries or [])
    methods_parts = []
    if methods_text:
        methods_parts.append(f"<p>{html.escape(methods_text)}</p>")
    methods_parts.extend(decision_paragraphs)
    methods_block = (
        "<section><h2>Methods summary</h2>" + "".join(methods_parts) + "</section>"
        if methods_parts
        else ""
    )

    sections = []
    for entry in model_entries:
        stats_items = "".join(
            f"<li><strong>{html.escape(k)}</strong>: {html.escape(str(v))}</li>" for k, v in entry.stats.items()
        )
        param_rows = []
        for p in entry.params:
            se_text = f"{p.se:.6g}" if math.isfinite(p.se) else "N/A"
            param_rows.append(
                f"<tr><td>{html.escape(p.name)}</td><td>{p.value:.6g}</td><td>{se_text}</td></tr>"
            )
        param_table = (
            "<table><thead><tr>"
            "<th>Parameter</th>"
            "<th>Value</th>"
            "<th>Standard Error(SE)</th>"
            "</tr></thead>"
            f"<tbody>{''.join(param_rows)}</tbody></table>"
        )
        plots_html = "".join(
            f"<div class='plot'><img src='{html.escape(p)}' alt='{html.escape(p)}'></div>"
            for p in entry.plots
        )
        warn_html = "".join(f"<li>{html.escape(w)}</li>" for w in entry.warnings)
        warn_block = f"<ul>{warn_html}</ul>" if entry.warnings else "<p>None</p>"
        sections.append(
            "<section>"
            f"<h2>({html.escape(entry.dataset)}){html.escape(entry.model)}</h2>"
            f"<h3>Stats</h3><ul>{stats_items}</ul>"
            f"<h3>Parameters</h3>{param_table}"
            f"<h3>Warnings</h3>{warn_block}"
            f"<h3>Plots</h3><div class='plots'>{plots_html}</div>"
            "</section>"
        )

    html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NMR binding fit report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2, h3 {{ color: #1d3557; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    section {{ margin-top: 32px; }}
    .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .plot img {{ width: 100%; border: 1px solid #ddd; background: #fff; }}
  </style>
</head>
<body>
  <h1>NMR binding fit report</h1>
  {methods_block}
  {summary_table}
  {''.join(sections)}
</body>
</html>
"""

    output_path.write_text(html_doc)
