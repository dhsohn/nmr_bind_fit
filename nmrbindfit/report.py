"""Report generation for NMR binding fits – publication-quality HTML."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    # Clean up reasons and make a single sentence block.
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
    # Render decisions into HTML paragraphs for the report.
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
    import csv

    if not rows:
        return
    # Preserve column order based on the first row.
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_decision_txt(decisions: Sequence[str], path: Path) -> None:
    # Write the decision narrative line-by-line.
    with path.open("w") as f:
        for line in decisions:
            f.write(line.rstrip() + "\n")


# ── Numeric helpers ───────────────────────────────────────────────────────────

_NUMERIC_COLS = {
    "Binding constant",
    "Confidence Interval(CI)",
    "Standard Error(SE)",
    "Residual sum of squares",
    "Root mean square error",
    "Bayesian Information Criterion",
    "Corrected Akaike Information Criterion",
}


def _cell_class(header: str) -> str:
    """Return CSS class for right-aligning numeric columns."""
    if header in _NUMERIC_COLS:
        return ' class="num"'
    return ""


# ── Figure counter ────────────────────────────────────────────────────────────

class _FigCounter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


# ── HTML builder ──────────────────────────────────────────────────────────────

_CSS = r"""
/* ── Reset & Base ────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --navy: #1a365d;
  --navy-light: #2c5282;
  --slate-700: #334155;
  --slate-500: #64748b;
  --slate-400: #94a3b8;
  --slate-200: #e2e8f0;
  --slate-100: #f1f5f9;
  --slate-50: #f8fafc;
  --accent: #c53030;
  --accent-bg: #fff5f5;
  --warn-bg: #fffbeb;
  --warn-border: #f59e0b;
  --success-bg: #f0fdf4;
  --success-border: #22c55e;
  --white: #ffffff;
  --text: #1e293b;
  --text-secondary: #475569;
  --radius: 6px;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.05);
  --shadow: 0 1px 3px rgba(0,0,0,.1), 0 1px 2px rgba(0,0,0,.06);
}

html { font-size: 15px; }

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--text);
  background: var(--slate-50);
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}

.container {
  max-width: 960px;
  margin: 0 auto;
  padding: 40px 32px 64px;
  background: var(--white);
  min-height: 100vh;
}

/* ── Header ──────────────────────────────────────── */
.report-header {
  border-bottom: 3px solid var(--navy);
  padding-bottom: 20px;
  margin-bottom: 36px;
}
.report-header h1 {
  font-family: 'Crimson Text', Georgia, 'Times New Roman', serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--navy);
  letter-spacing: -0.02em;
  margin-bottom: 6px;
}
.report-header .subtitle {
  font-size: 0.85rem;
  color: var(--slate-500);
  font-weight: 400;
}

/* ── Table of Contents ───────────────────────────── */
.toc {
  background: var(--slate-100);
  border: 1px solid var(--slate-200);
  border-radius: var(--radius);
  padding: 20px 28px;
  margin-bottom: 36px;
}
.toc h2 {
  font-family: 'Crimson Text', Georgia, serif;
  font-size: 1.15rem;
  color: var(--navy);
  margin-bottom: 10px;
}
.toc ol {
  padding-left: 0;
  list-style: none;
  columns: 2;
  column-gap: 32px;
}
.toc li {
  font-size: 0.88rem;
  color: var(--text-secondary);
  padding: 2px 0;
  break-inside: avoid;
}
.toc a {
  color: var(--navy-light);
  text-decoration: none;
  transition: color .15s;
}
.toc a:hover { color: var(--accent); }

/* ── Executive Summary Card ──────────────────────── */
.exec-summary {
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
  color: var(--white);
  border-radius: var(--radius);
  padding: 24px 28px;
  margin-bottom: 36px;
  box-shadow: var(--shadow);
}
.exec-summary h2 {
  font-family: 'Crimson Text', Georgia, serif;
  font-size: 1.2rem;
  margin-bottom: 12px;
  opacity: .9;
}
.exec-summary p {
  font-size: 0.92rem;
  line-height: 1.65;
  opacity: .92;
}

/* ── Section Headings ────────────────────────────── */
h2.section-title {
  font-family: 'Crimson Text', Georgia, serif;
  font-size: 1.4rem;
  color: var(--navy);
  margin-top: 44px;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--slate-200);
}
h3.subsection-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--slate-700);
  margin-top: 24px;
  margin-bottom: 8px;
}
h4.model-sub {
  font-size: 0.93rem;
  font-weight: 600;
  color: var(--slate-700);
  margin-top: 18px;
  margin-bottom: 6px;
}

/* ── Methods Detail ──────────────────────────────── */
.methods-section { margin-bottom: 12px; }
.methods-section h3 {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--navy-light);
  margin-bottom: 4px;
}
.methods-section p {
  font-size: 0.88rem;
  color: var(--text-secondary);
  line-height: 1.7;
  text-align: justify;
}

/* ── Tables ──────────────────────────────────────── */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 12px 0 20px;
  font-size: 0.85rem;
  box-shadow: var(--shadow-sm);
  border-radius: var(--radius);
  overflow: hidden;
}
thead th {
  background: var(--navy);
  color: var(--white);
  padding: 10px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 0.82rem;
  letter-spacing: 0.01em;
}
tbody td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--slate-200);
  color: var(--text);
}
tbody tr:nth-child(even) { background: var(--slate-50); }
tbody tr:hover { background: #eef2ff; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best-model td { background: var(--success-bg) !important; font-weight: 500; }

/* ── Stats List ──────────────────────────────────── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 8px;
  margin: 8px 0 16px;
}
.stat-item {
  background: var(--slate-50);
  border: 1px solid var(--slate-200);
  border-radius: var(--radius);
  padding: 10px 14px;
}
.stat-item .stat-label {
  font-size: 0.75rem;
  color: var(--slate-500);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 2px;
}
.stat-item .stat-value {
  font-size: 1rem;
  font-weight: 600;
  color: var(--navy);
  font-variant-numeric: tabular-nums;
}

/* ── Warnings ────────────────────────────────────── */
.warning-box {
  background: var(--warn-bg);
  border-left: 4px solid var(--warn-border);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 14px 18px;
  margin: 12px 0;
}
.warning-box .warn-title {
  font-weight: 600;
  font-size: 0.88rem;
  color: #92400e;
  margin-bottom: 6px;
}
.warning-box ul { padding-left: 18px; }
.warning-box li {
  font-size: 0.85rem;
  color: #78350f;
  padding: 2px 0;
}
.no-warnings {
  font-size: 0.85rem;
  color: var(--slate-400);
  font-style: italic;
}

/* ── Plots / Figures ─────────────────────────────── */
.figure-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 20px;
  margin: 16px 0 24px;
}
.figure-card {
  border: 1px solid var(--slate-200);
  border-radius: var(--radius);
  overflow: hidden;
  background: var(--white);
  box-shadow: var(--shadow-sm);
}
.figure-card img {
  width: 100%;
  display: block;
  background: var(--white);
}
.figure-card figcaption {
  padding: 8px 14px;
  font-size: 0.8rem;
  color: var(--text-secondary);
  border-top: 1px solid var(--slate-200);
  background: var(--slate-50);
}
.figure-card figcaption strong {
  color: var(--navy);
}

/* ── Model Section Card ──────────────────────────── */
.model-card {
  border: 1px solid var(--slate-200);
  border-radius: var(--radius);
  padding: 24px;
  margin: 20px 0;
  box-shadow: var(--shadow-sm);
  background: var(--white);
}
.model-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--slate-200);
}
.model-card-header h3 {
  font-family: 'Crimson Text', Georgia, serif;
  font-size: 1.15rem;
  color: var(--navy);
  margin: 0;
}
.model-badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 12px;
  background: var(--slate-100);
  color: var(--slate-500);
}
.model-badge.best {
  background: var(--success-bg);
  color: #166534;
  border: 1px solid var(--success-border);
}

/* ── Footer ──────────────────────────────────────── */
.report-footer {
  margin-top: 48px;
  padding-top: 20px;
  border-top: 2px solid var(--slate-200);
  font-size: 0.8rem;
  color: var(--slate-400);
  text-align: center;
}
.report-footer a { color: var(--navy-light); text-decoration: none; }

/* ── Print ───────────────────────────────────────── */
@media print {
  body { background: white; }
  .container { max-width: 100%; padding: 0; box-shadow: none; }
  .toc { break-inside: avoid; }
  .model-card { break-inside: avoid; }
  .figure-card { break-inside: avoid; }
  .report-footer .no-print { display: none; }
  a { color: inherit !important; text-decoration: none !important; }
  table { font-size: 0.8rem; }
}
"""


def _slug(text: str) -> str:
    """Make a simple anchor slug from text."""
    return text.lower().replace(" ", "-").replace(":", "").replace("(", "").replace(")", "")


def _fig_caption(fig_counter: _FigCounter, plot_path: str) -> str:
    """Derive a human-readable caption from a plot file path."""
    stem = Path(plot_path).stem
    n = fig_counter.next()
    # Map common prefixes
    if stem.startswith("isotherm_"):
        peak = stem.replace("isotherm_", "")
        return f"<strong>Figure {n}.</strong> Binding isotherm – {html.escape(peak)}"
    if stem.startswith("residual_"):
        peak = stem.replace("residual_", "")
        return f"<strong>Figure {n}.</strong> Fit residuals – {html.escape(peak)}"
    if stem == "fraction_bound":
        return f"<strong>Figure {n}.</strong> Fraction bound"
    if stem.startswith("bootstrap_"):
        k = stem.replace("bootstrap_", "")
        return f"<strong>Figure {n}.</strong> Bootstrap distribution – {html.escape(k)}"
    return f"<strong>Figure {n}.</strong> {html.escape(stem)}"


def _best_models_map(decision_entries: Optional[Sequence[DecisionEntry]]) -> Dict[str, str]:
    best_models: Dict[str, str] = {}
    if decision_entries:
        for d in decision_entries:
            best_models[d.dataset] = d.recommended_model
    return best_models


def _render_exec_summary(decision_entries: Optional[Sequence[DecisionEntry]]) -> str:
    decision_paragraphs = _decision_paragraphs(decision_entries or [])
    if not decision_paragraphs:
        return ""
    return (
        '<div class="exec-summary" id="executive-summary">'
        "<h2>Executive Summary</h2>"
        + "".join(decision_paragraphs)
        + "</div>"
    )


def _render_methods_block(
    methods_text: Optional[str],
    methods_sections: Optional[Sequence[Dict[str, str]]],
) -> str:
    if methods_sections:
        parts = []
        for sec in methods_sections:
            title = html.escape(sec["title"])
            content = html.escape(sec["content"])
            parts.append(
                f'<div class="methods-section">'
                f"<h3>{title}</h3>"
                f"<p>{content}</p>"
                f"</div>"
            )
        return (
            '<h2 class="section-title" id="methods">Methods</h2>'
            + "".join(parts)
        )
    if methods_text:
        return (
            '<h2 class="section-title" id="methods">Methods</h2>'
            f'<div class="methods-section"><p>{html.escape(methods_text)}</p></div>'
        )
    return ""


def _render_warnings_block(warnings: Optional[Sequence[str]]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    return (
        '<div class="warning-box">'
        '<div class="warn-title">⚠ Analysis Warnings</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _render_summary_table(summary_rows: Sequence[Dict[str, str]], best_models: Dict[str, str]) -> str:
    if not summary_rows:
        return ""
    # Exclude "Dataset" column from HTML to avoid horizontal overflow.
    headers = [h for h in summary_rows[0].keys() if h != "Dataset"]
    thead = "".join(
        f'<th{_cell_class(h)}>{html.escape(h)}</th>' for h in headers
    )
    body_rows = []
    for row in summary_rows:
        model_val = row.get("Model", "")
        ds_val = row.get("Dataset", "")
        is_best = best_models.get(ds_val) == model_val
        tr_class = ' class="best-model"' if is_best else ""
        tds = "".join(
            f'<td{_cell_class(h)}>{html.escape(str(row[h]))}</td>'
            for h in headers
        )
        body_rows.append(f"<tr{tr_class}>{tds}</tr>")
    tbody = "".join(body_rows)
    return (
        '<h2 class="section-title" id="model-comparison">Model Comparison</h2>'
        f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"
    )


def _render_stats_grid(stats: Dict[str, str]) -> str:
    stats_items = []
    for k, v in stats.items():
        stats_items.append(
            f'<div class="stat-item">'
            f'<div class="stat-label">{html.escape(k)}</div>'
            f'<div class="stat-value">{html.escape(str(v))}</div>'
            f"</div>"
        )
    return f'<div class="stats-grid">{"".join(stats_items)}</div>'


def _render_param_table(params: Sequence[ParamEntry]) -> str:
    param_rows = []
    for p in params:
        se_text = f"{p.se:.6g}" if math.isfinite(p.se) else "N/A"
        param_rows.append(
            f'<tr><td>{html.escape(p.name)}</td><td class="num">{p.value:.6g}</td>'
            f'<td class="num">{se_text}</td></tr>'
        )
    return (
        "<table><thead><tr>"
        "<th>Parameter</th>"
        '<th class="num">Value</th>'
        '<th class="num">Standard Error (SE)</th>'
        "</tr></thead>"
        f"<tbody>{''.join(param_rows)}</tbody></table>"
    )


def _render_model_warning_block(warnings: Sequence[str]) -> str:
    if warnings:
        warn_items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        return (
            '<div class="warning-box">'
            '<div class="warn-title">⚠ Model Warnings</div>'
            f"<ul>{warn_items}</ul></div>"
        )
    return '<p class="no-warnings">No warnings.</p>'


def _render_plot_grid(plot_paths: Sequence[str], fig_counter: _FigCounter) -> str:
    if not plot_paths:
        return ""
    fig_items = []
    for p in plot_paths:
        caption = _fig_caption(fig_counter, p)
        fig_items.append(
            f'<figure class="figure-card">'
            f'<img src="{html.escape(p)}" alt="{html.escape(Path(p).stem)}" loading="lazy">'
            f"<figcaption>{caption}</figcaption>"
            f"</figure>"
        )
    return f'<div class="figure-grid">{"".join(fig_items)}</div>'


def _render_model_card(entry: ModelEntry, best_models: Dict[str, str], fig_counter: _FigCounter) -> str:
    is_best = best_models.get(entry.dataset) == entry.model
    badge = '<span class="model-badge best">★ Best Model</span>' if is_best else ""
    slug = _slug(f"{entry.dataset}-{entry.model}")

    stats_grid = _render_stats_grid(entry.stats)
    param_table = _render_param_table(entry.params)
    warn_block = _render_model_warning_block(entry.warnings)
    plots_html = _render_plot_grid(entry.plots, fig_counter)

    display_title = f"{html.escape(entry.model)}"
    if entry.dataset != "Simultaneous Fitting":
        display_title += (
            f" <span style='font-weight:400;color:var(--slate-500)'>— {html.escape(entry.dataset)}</span>"
        )

    return (
        f'<div class="model-card" id="{slug}">'
        f'<div class="model-card-header"><h3>{display_title}</h3>{badge}</div>'
        f'<h4 class="model-sub">Goodness-of-Fit Statistics</h4>'
        f"{stats_grid}"
        f'<h4 class="model-sub">Fitted Parameters</h4>'
        f"{param_table}"
        f'<h4 class="model-sub">Warnings</h4>'
        f"{warn_block}"
        f'<h4 class="model-sub">Figures</h4>'
        f"{plots_html}"
        f"</div>"
    )


def _render_model_sections(
    model_entries: Sequence[ModelEntry],
    best_models: Dict[str, str],
    fig_counter: _FigCounter,
) -> List[str]:
    return [_render_model_card(entry, best_models, fig_counter) for entry in model_entries]


def _render_toc(
    exec_html: str,
    methods_block: str,
    summary_table: str,
    model_entries: Sequence[ModelEntry],
) -> str:
    toc_items = []
    toc_idx = 1
    if exec_html:
        toc_items.append(f'<li><a href="#executive-summary">{toc_idx}. Executive Summary</a></li>')
        toc_idx += 1
    if methods_block:
        toc_items.append(f'<li><a href="#methods">{toc_idx}. Methods</a></li>')
        toc_idx += 1
    if summary_table:
        toc_items.append(f'<li><a href="#model-comparison">{toc_idx}. Model Comparison</a></li>')
        toc_idx += 1
    for entry in model_entries:
        slug = _slug(f"{entry.dataset}-{entry.model}")
        label = html.escape(entry.model)
        if entry.dataset != "Simultaneous Fitting":
            label += f" ({html.escape(entry.dataset)})"
        toc_items.append(f'<li><a href="#{slug}">{toc_idx}. {label}</a></li>')
        toc_idx += 1

    if not toc_items:
        return ""
    return (
        '<nav class="toc"><h2>Contents</h2>'
        f'<ol>{"".join(toc_items)}</ol></nav>'
    )


def _render_model_detail_heading(model_sections: Sequence[str]) -> str:
    if model_sections:
        return '<h2 class="section-title" id="model-details">Model Details</h2>'
    return ""


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
    """Write a publication-quality HTML report."""
    now = datetime.now()
    fig_counter = _FigCounter()
    best_models = _best_models_map(decision_entries)
    exec_html = _render_exec_summary(decision_entries)
    methods_block = _render_methods_block(methods_text, methods_sections)
    warnings_block = _render_warnings_block(warnings)
    summary_table = _render_summary_table(summary_rows, best_models)
    model_sections = _render_model_sections(model_entries, best_models, fig_counter)
    toc_html = _render_toc(exec_html, methods_block, summary_table, model_entries)
    model_detail_heading = _render_model_detail_heading(model_sections)

    # ── Assemble document ─────────────────────────────────────────────────
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NMR Binding Fit Report</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">
  <header class="report-header">
    <h1>NMR Binding Fit Report</h1>
    <div class="subtitle">Generated by NMRBindFit &middot; {now.strftime("%Y-%m-%d %H:%M:%S")}</div>
  </header>

  {toc_html}
  {exec_html}
  {methods_block}
  {warnings_block}
  {summary_table}
  {model_detail_heading}
  {''.join(model_sections)}

  <footer class="report-footer">
    <p>This report was generated automatically by <strong>NMRBindFit</strong>.</p>
    <p>For publication, please cite the software appropriately and verify all fitted parameters with independent analysis.</p>
  </footer>
</div>
</body>
</html>
"""

    output_path.write_text(html_doc)
