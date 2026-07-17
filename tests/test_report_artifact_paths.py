import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

import nmr_bind_fit.report_pipeline as report_pipeline
from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS
from nmr_bind_fit.plots import plot_residuals
from nmr_bind_fit.report_html_renderer import (
    _fig_caption,
    _FigCounter,
    _render_summary_table,
    _slug,
)


def _dataset(name: str, peak_names: Optional[List[str]] = None) -> Dataset:
    peaks = peak_names or ["ppm_H1"]
    x = np.array([0.0, 0.5, 1.0], dtype=float)
    y = np.column_stack(
        [np.array([7.0 + i, 7.1 + i, 7.2 + i], dtype=float) for i in range(len(peaks))]
    )
    return Dataset(
        name=name,
        path=Path(f"{name}.csv"),
        h_tot=np.full(3, 1e-3, dtype=float),
        g_tot=x * 1e-3,
        x=x,
        y=y,
        y_cols=peaks,
        dropped_peaks=[],
    )


def _fit_result(ds: Dataset) -> SimpleNamespace:
    param_samples = np.array(
        [
            [1.9, 7.0, 7.2],
            [2.0, 7.0, 7.2],
            [2.1, 7.0, 7.2],
        ],
        dtype=float,
    )
    bootstrap = SimpleNamespace(
        param_samples=param_samples,
        logk_samples=param_samples[:, :1],
        ci_low=np.array([1.8], dtype=float),
        ci_high=np.array([2.2], dtype=float),
        n_boot=4,
        n_success=3,
        ci_valid=False,
        ci_method_used="unavailable",
        ci_message="bootstrap CI requires more successful refits",
    )
    return SimpleNamespace(
        model=MODEL_SPECS["11"],
        datasets=[ds],
        params=np.array([2.0, 7.0, 7.2], dtype=float),
        param_names=["logK", "delta_free", "delta_bound"],
        residuals=[np.zeros_like(ds.y)],
        species=[SimpleNamespace(solver_stats=None)],
        bootstrap=bootstrap,
        r2=0.99,
        r2_per_peak=[0.99],
        rss=0.01,
        rmse=0.02,
        bic=1.0,
        aicc=2.0,
        penalty_count=0,
        residual_diagnostics={},
        n=3,
        p=3,
        dof=0,
        jacobian_rank=2,
        jacobian_condition=123.0,
    )


def test_independent_results_use_collision_free_dataset_scopes(tmp_path, monkeypatch):
    datasets = [_dataset("first"), _dataset("second")]

    def fake_isotherms(model, ds, logk, delta, out_dir):
        del model, logk, delta
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "isotherm_peak-0001-cHBtX0gx.png"
        path.write_text(ds.name, encoding="utf-8")
        return [path]

    def fake_bootstrap(samples, names, out_dir):
        del samples, names
        path = out_dir / "bootstrap_K.png"
        path.write_text(str(out_dir), encoding="utf-8")
        return [path]

    monkeypatch.setattr(report_pipeline, "plot_isotherms", fake_isotherms)
    monkeypatch.setattr(report_pipeline, "plot_residuals", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_fraction_bound", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_bootstrap_hist", fake_bootstrap)

    keys = ["sample/a", "sample?a"]
    summary_rows, entries, warnings = report_pipeline.build_report_artifacts(
        args=SimpleNamespace(bootstrap_ci_width=None),
        ordered_keys=keys,
        results_by_key={key: {"11": _fit_result(ds)} for key, ds in zip(keys, datasets)},
        failures_by_key={},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    isotherms = [
        tmp_path / next(path for path in entry.plots if "isotherm_" in path)
        for entry in entries
    ]
    bootstraps = [
        tmp_path / next(path for path in entry.plots if "bootstrap_" in path)
        for entry in entries
    ]

    assert isotherms[0] != isotherms[1]
    assert bootstraps[0] != bootstraps[1]
    assert isotherms[0].read_text(encoding="utf-8") == "first"
    assert isotherms[1].read_text(encoding="utf-8") == "second"
    assert isotherms[0].relative_to(tmp_path).parts[0] == "model_11"
    assert isotherms[1].relative_to(tmp_path).parts[0] == "model_11"
    assert isotherms[0].relative_to(tmp_path).parts[1].startswith("dataset_01_")
    assert isotherms[1].relative_to(tmp_path).parts[1].startswith("dataset_02_")
    assert all((path.parent / "correlation.csv").is_file() for path in isotherms)
    assert [row["Dataset"] for row in summary_rows] == keys
    assert entries[0].stats["Observations (n)"] == "3"
    assert entries[0].stats["Fitted parameters (p)"] == "3"
    assert entries[0].stats["Residual degrees of freedom"] == "0"
    assert entries[0].stats["Jacobian rank"] == "2"
    assert entries[0].stats["Jacobian condition number"] == "123"
    assert entries[0].stats["Successful bootstrap refits"] == "3 / 4"
    assert entries[0].stats["Bootstrap CI method used"] == "unavailable"
    assert entries[0].stats["Bootstrap SE (log10 K)"] == "N/A"
    assert all(np.isnan(param.se) for param in entries[0].params)
    assert summary_rows[0]["95 % CI"] == "N/A"
    assert warnings == []
    assert all("bootstrap CI requires more successful refits" in entry.warnings for entry in entries)
    assert set(entries[0].plot_labels.values()) == {"ppm_H1"}


def test_dataset_directory_tokens_are_case_insensitive_safe():
    tokens = report_pipeline._dataset_dir_tokens(["Sample", "sample"])

    assert tokens == {"Sample": "01_Sample", "sample": "02_sample"}
    assert len({token.casefold() for token in tokens.values()}) == 2


def test_slash_containing_peak_labels_use_safe_unique_files_and_original_captions(tmp_path):
    peak_names = ["ppm/H1", "ppm?H1"]
    ds = _dataset("sample", peak_names)

    files = plot_residuals(MODEL_SPECS["nb"], ds, np.zeros_like(ds.y), tmp_path)
    png_files = [path for path in files if path.suffix == ".png"]

    assert len(png_files) == 2
    assert len({path.name for path in png_files}) == 2
    assert all(path.parent == tmp_path and path.is_file() for path in png_files)
    assert all(re.fullmatch(r"residual_peak-\d{4}-[A-Za-z0-9_-]+\.png", path.name) for path in png_files)

    counter = _FigCounter()
    captions = [
        _fig_caption(counter, str(path), peak)
        for path, peak in zip(png_files, peak_names)
    ]
    assert "ppm/H1" in captions[0]
    assert "ppm?H1" in captions[1]


def test_long_peak_label_uses_bounded_filename_and_full_display_caption(tmp_path):
    peak = "ppm_" + "x" * 300
    ds = _dataset("sample", [peak])

    files = plot_residuals(MODEL_SPECS["nb"], ds, np.zeros_like(ds.y), tmp_path)
    png_path = next(path for path in files if path.suffix == ".png")

    assert len(png_path.name) < 100
    assert png_path.is_file()
    assert peak in _fig_caption(_FigCounter(), str(png_path), peak)


def test_html_model_comparison_retains_dataset_column():
    rows = [
        {"Dataset": "dataset A", "Model": "1:1", "BIC": "1.0"},
        {"Dataset": "dataset B", "Model": "1:1", "BIC": "2.0"},
    ]

    rendered = _render_summary_table(rows, {})

    assert "<th>Dataset</th>" in rendered
    assert "<td>dataset A</td>" in rendered
    assert "<td>dataset B</td>" in rendered


def test_html_slug_is_attribute_safe_and_bounded():
    value = 'sample" data-bad="yes ' + "x" * 300

    slug = _slug(value)

    assert len(slug) <= 71
    assert re.fullmatch(r"[a-z0-9_-]+", slug)
    assert '"' not in slug
    assert "=" not in slug
