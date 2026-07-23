"""HTML rendering of fit summaries, model tables, and decision reports."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_NUMERIC_COLS = {
    "Binding constant",
    "Binding constant (M⁻¹)",
    "95 % CI",
    "R² (mean)",
    "BIC",
    "AICc",
}


def _cell_class(header: str) -> str:
    if header in _NUMERIC_COLS:
        return ' class="num"'
    return ""


class _FigCounter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


_CSS = r"""
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, Roboto, Arial, sans-serif;
  color: #222;
  background: #fff;
  line-height: 1.6;
}

.container { max-width: 900px; margin: 0 auto; padding: 32px 24px 56px; }

.report-header { border-bottom: 2px solid #444; padding-bottom: 12px; margin-bottom: 28px; }
.report-header h1 { font-size: 1.8rem; }
.report-header .subtitle { font-size: 0.85rem; color: #666; }

h2.section-title { font-size: 1.3rem; margin: 36px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #ccc; }
h4.model-sub { font-size: 0.95rem; margin: 16px 0 6px; }

.exec-summary { margin-bottom: 28px; }
.exec-summary p { margin-bottom: 8px; }

.methods-section { margin-bottom: 10px; }
.methods-section h3 { font-size: 0.95rem; }
.methods-section p { font-size: 0.9rem; color: #444; text-align: justify; }

table { border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 0.85rem; }
th, td { padding: 6px 10px; border-bottom: 1px solid #ddd; text-align: left; }
thead th { background: #f2f2f2; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best-model td { background: #f0f6ec; font-weight: 500; }

.warning-box { border-left: 3px solid #c89b3c; background: #fdf8ec; padding: 10px 14px; margin: 10px 0; }
.warning-box .warn-title { font-weight: 600; font-size: 0.88rem; margin-bottom: 4px; }
.warning-box ul { padding-left: 18px; }
.warning-box li { font-size: 0.85rem; }
.no-warnings { font-size: 0.85rem; color: #777; font-style: italic; }

.model-card { border: 1px solid #ddd; border-radius: 4px; padding: 16px 20px; margin-bottom: 24px; }
.model-card-header { display: flex; align-items: baseline; gap: 10px; }
.model-card-header h3 { font-size: 1.05rem; }
.model-badge.best { font-size: 0.75rem; color: #4a7a3a; }
.dataset-label { font-weight: 400; color: #666; }

.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; margin: 10px 0 18px; }
figure img { width: 100%; display: block; }
figcaption { font-size: 0.78rem; color: #555; padding-top: 4px; }

.report-footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #ddd; font-size: 0.8rem; color: #666; }

@media print {
  .container { max-width: none; padding: 0; }
  .model-card { break-inside: avoid; }
}
"""


def _slug(text: str) -> str:
    # Model cards are keyed by dataset label and model name, which are already
    # distinct, so the slug only has to be a valid bounded HTML id.
    safe = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-_")
    return safe[:60].rstrip("-_") or "section"


def _plot_alt_text(plot_path: str, plot_label: str | None = None) -> str:
    stem = Path(plot_path).stem
    if stem.startswith("isotherm_"):
        peak = plot_label or stem.removeprefix("isotherm_")
        return f"Binding isotherm – {peak}"
    if stem.startswith("residual_"):
        peak = plot_label or stem.removeprefix("residual_")
        return f"Fit residuals – {peak}"
    if stem == "fraction_bound":
        return "Fraction bound"
    return stem


def _fig_caption(
    fig_counter: _FigCounter,
    plot_path: str,
    plot_label: str | None = None,
) -> str:
    stem = Path(plot_path).stem
    n = fig_counter.next()
    if stem.startswith("isotherm_"):
        peak = plot_label or stem.removeprefix("isotherm_")
        return f"<strong>Figure {n}.</strong> Binding isotherm – {html.escape(peak)}"
    if stem.startswith("residual_"):
        peak = plot_label or stem.removeprefix("residual_")
        return f"<strong>Figure {n}.</strong> Fit residuals – {html.escape(peak)}"
    if stem == "fraction_bound":
        return f"<strong>Figure {n}.</strong> Fraction bound"
    return f"<strong>Figure {n}.</strong> {html.escape(stem)}"


def _best_models_map(decision_entries: Sequence[object] | None) -> dict[str, str]:
    best_models: dict[str, str] = {}
    if decision_entries:
        for d in decision_entries:
            best_models[d.dataset] = d.recommended_model
    return best_models


def _render_exec_summary(
    decision_entries: Sequence[object] | None,
    decision_paragraphs_fn: Callable[[Sequence[object]], list[str]] | None,
) -> str:
    if decision_paragraphs_fn is None:
        return ""
    decision_paragraphs = decision_paragraphs_fn(decision_entries or [])
    if not decision_paragraphs:
        return ""
    return (
        '<div class="exec-summary" id="executive-summary">'
        "<h2>Executive Summary</h2>"
        + "".join(decision_paragraphs)
        + "</div>"
    )


def _render_methods_block(
    methods_text: str | None,
    methods_sections: Sequence[dict[str, str]] | None,
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


def _render_warnings_block(warnings: Sequence[str] | None) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    return (
        '<div class="warning-box">'
        '<div class="warn-title">⚠ Analysis Warnings</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _render_summary_table(summary_rows: Sequence[dict[str, str]], best_models: dict[str, str]) -> str:
    if not summary_rows:
        return ""
    headers = list(summary_rows[0].keys())
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


def _render_stats_table(stats: dict[str, str]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td>"
        f'<td class="num">{html.escape(str(v))}</td></tr>'
        for k, v in stats.items()
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _render_param_table(params: Sequence[object]) -> str:
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


def _render_plot_grid(
    plot_paths: Sequence[str],
    fig_counter: _FigCounter,
    plot_labels: dict[str, str] | None = None,
) -> str:
    if not plot_paths:
        return ""
    fig_items = []
    for p in plot_paths:
        plot_label = (plot_labels or {}).get(p)
        caption = _fig_caption(fig_counter, p, plot_label)
        alt_text = _plot_alt_text(p, plot_label)
        fig_items.append(
            f"<figure>"
            f'<img src="{html.escape(p)}" alt="{html.escape(alt_text)}" loading="lazy">'
            f"<figcaption>{caption}</figcaption>"
            f"</figure>"
        )
    return f'<div class="figure-grid">{"".join(fig_items)}</div>'


def _render_model_card(entry: object, best_models: dict[str, str], fig_counter: _FigCounter) -> str:
    is_best = best_models.get(entry.dataset) == entry.model
    badge = '<span class="model-badge best">★ Best Model</span>' if is_best else ""
    slug = _slug(f"{entry.dataset}-{entry.model}")

    stats_table = _render_stats_table(entry.stats)
    param_table = _render_param_table(entry.params)
    warn_block = _render_model_warning_block(entry.warnings)
    plots_html = _render_plot_grid(
        entry.plots,
        fig_counter,
        entry.plot_labels,
    )

    display_title = f"{html.escape(entry.model)}"
    if entry.dataset != "Simultaneous Fitting":
        display_title += f' <span class="dataset-label">— {html.escape(entry.dataset)}</span>'

    return (
        f'<div class="model-card" id="{slug}">'
        f'<div class="model-card-header"><h3>{display_title}</h3>{badge}</div>'
        f'<h4 class="model-sub">Goodness-of-Fit Statistics</h4>'
        f"{stats_table}"
        f'<h4 class="model-sub">Fitted Parameters</h4>'
        f"{param_table}"
        f'<h4 class="model-sub">Warnings</h4>'
        f"{warn_block}"
        f'<h4 class="model-sub">Figures</h4>'
        f"{plots_html}"
        f"</div>"
    )


def _render_model_sections(
    model_entries: Sequence[object],
    best_models: dict[str, str],
    fig_counter: _FigCounter,
) -> list[str]:
    return [_render_model_card(entry, best_models, fig_counter) for entry in model_entries]


def _render_model_detail_heading(model_sections: Sequence[str]) -> str:
    if model_sections:
        return '<h2 class="section-title" id="model-details">Model Details</h2>'
    return ""


def write_report_html(
    summary_rows: Sequence[dict[str, str]],
    model_entries: Sequence[object],
    decision_entries: Sequence[object] | None,
    methods_text: str | None,
    warnings: Sequence[str] | None,
    output_path: Path,
    *,
    methods_sections: Sequence[dict[str, str]] | None = None,
    decision_paragraphs_fn: Callable[[Sequence[object]], list[str]] | None = None,
) -> None:
    # Local wall-clock time, kept timezone-aware so the stamp is unambiguous.
    now = datetime.now(timezone.utc).astimezone()
    fig_counter = _FigCounter()
    best_models = _best_models_map(decision_entries)
    exec_html = _render_exec_summary(decision_entries, decision_paragraphs_fn)
    methods_block = _render_methods_block(methods_text, methods_sections)
    warnings_block = _render_warnings_block(warnings)
    summary_table = _render_summary_table(summary_rows, best_models)
    model_sections = _render_model_sections(model_entries, best_models, fig_counter)
    model_detail_heading = _render_model_detail_heading(model_sections)

    html_doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>NMR Binding Fit Report</title>
  <style>{_CSS}</style>
</head>
<body>
<div class=\"container\">
  <header class=\"report-header\">
    <h1>NMR Binding Fit Report</h1>
    <div class=\"subtitle\">Generated by nmr_bind_fit &middot; {now.strftime("%Y-%m-%d %H:%M:%S")}</div>
  </header>

  {exec_html}
  {methods_block}
  {warnings_block}
  {summary_table}
  {model_detail_heading}
  {''.join(model_sections)}

  <footer class=\"report-footer\">
    <p>This report was generated automatically by <strong>nmr_bind_fit</strong>.</p>
    <p>For publication, please cite the software appropriately and verify all fitted parameters with independent analysis.</p>
  </footer>
</div>
</body>
</html>
"""

    output_path.write_text(html_doc, encoding="utf-8")
