import argparse
import csv
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pytest

import nmr_bind_fit.report_pipeline as report_pipeline
from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS
from nmr_bind_fit.plots import plot_residuals
from nmr_bind_fit.report import DecisionEntry, _decision_paragraphs, write_summary_csv
from nmr_bind_fit.report_html_renderer import (
    _fig_caption,
    _FigCounter,
    _render_summary_table,
    _slug,
)
from nmr_bind_fit.report_pipeline import (
    _build_model_warnings,
    _build_summary_row,
    build_decisions,
    build_methods_sections,
    build_report_artifacts,
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


def _fit_result(
    ds: Dataset,
    *,
    n_boot: int = 20,
    n_success: Optional[int] = None,
    ci_valid: bool = True,
    ci_method_used: Optional[str] = None,
    ci_message: Optional[str] = None,
) -> SimpleNamespace:
    success_count = n_boot if n_success is None else n_success
    offsets = np.linspace(-0.1, 0.1, success_count, dtype=float)
    param_samples = np.column_stack(
        [
            2.0 + offsets,
            7.0 + 0.2 * offsets,
            7.2 - 0.1 * offsets,
        ]
    )
    bootstrap = SimpleNamespace(
        param_samples=param_samples,
        logk_samples=param_samples[:, :1],
        ci_low=np.array([1.8], dtype=float),
        ci_high=np.array([2.2], dtype=float),
        n_boot=n_boot,
        n_success=success_count,
        ci_valid=ci_valid,
        ci_method_used=ci_method_used or ("percentile" if ci_valid else "unavailable"),
        ci_message=ci_message if ci_message is not None else (
            "" if ci_valid else "bootstrap CI requires more successful refits"
        ),
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
        jacobian_logk_sensitivity=0.5,
        logk_bounds=None,
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
    assert entries[0].stats["Successful bootstrap refits"] == "20 / 20"
    assert entries[0].stats["Bootstrap CI method used"] == "percentile"
    assert float(entries[0].stats["Bootstrap SE (log10 K)"]) > 0.0
    assert all(np.isfinite(param.se) for param in entries[0].params)
    assert summary_rows[0]["95 % CI"] != "N/A"
    assert warnings == []
    assert all(not entry.warnings for entry in entries)
    assert set(entries[0].plot_labels.values()) == {"ppm_H1"}


@pytest.mark.parametrize(("n_boot", "n_success"), [(20, 19), (19, 19)])
def test_incomplete_or_too_small_bootstrap_withholds_raw_distribution_artifacts(
    tmp_path,
    monkeypatch,
    n_boot,
    n_success,
):
    ds = _dataset("sample")

    def unexpected_bootstrap_plot(*args):
        raise AssertionError("bootstrap histogram must not be generated")

    monkeypatch.setattr(report_pipeline, "plot_isotherms", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_residuals", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_fraction_bound", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_bootstrap_hist", unexpected_bootstrap_plot)

    message = "Bootstrap uncertainty unavailable: the raw sample is incomplete or too small."
    summary_rows, entries, report_warnings = report_pipeline.build_report_artifacts(
        args=SimpleNamespace(bootstrap_ci_width=None),
        ordered_keys=["sample"],
        results_by_key={
            "sample": {
                "11": _fit_result(
                    ds,
                    n_boot=n_boot,
                    n_success=n_success,
                    ci_valid=False,
                    ci_message=message,
                )
            }
        },
        failures_by_key={},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    assert report_warnings == []
    assert not any("bootstrap_" in path for path in entries[0].plots)
    assert list(tmp_path.rglob("correlation.csv")) == []
    assert entries[0].stats["Bootstrap SE (log10 K)"] == "N/A"
    assert all(np.isnan(param.se) for param in entries[0].params)
    assert summary_rows[0]["95 % CI"] == "N/A"
    assert message in entries[0].warnings


def test_bca_only_failure_keeps_complete_raw_distribution_artifacts(tmp_path, monkeypatch):
    ds = _dataset("sample")

    def fake_bootstrap(samples, names, out_dir):
        assert samples.shape == (20, 1)
        assert names == ["K"]
        path = out_dir / "bootstrap_K.png"
        path.write_text("complete distribution", encoding="utf-8")
        return [path]

    monkeypatch.setattr(report_pipeline, "plot_isotherms", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_residuals", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_fraction_bound", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_bootstrap_hist", fake_bootstrap)

    message = "BCa CI unavailable: all delete-one jackknife refits must succeed."
    summary_rows, entries, report_warnings = report_pipeline.build_report_artifacts(
        args=SimpleNamespace(bootstrap_ci_width=None),
        ordered_keys=["sample"],
        results_by_key={
            "sample": {
                "11": _fit_result(
                    ds,
                    ci_valid=False,
                    ci_method_used="unavailable",
                    ci_message=message,
                )
            }
        },
        failures_by_key={},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    assert report_warnings == []
    bootstrap_path = tmp_path / next(
        path for path in entries[0].plots if "bootstrap_" in path
    )
    assert bootstrap_path.read_text(encoding="utf-8") == "complete distribution"
    assert list(tmp_path.rglob("correlation.csv"))
    assert float(entries[0].stats["Bootstrap SE (log10 K)"]) > 0.0
    assert all(np.isfinite(param.se) for param in entries[0].params)
    assert summary_rows[0]["95 % CI"] == "N/A"
    assert message in entries[0].warnings


def test_bootstrap_reportability_rejects_nonfinite_or_mismatched_samples():
    result = _fit_result(_dataset("sample"))
    result.bootstrap.param_samples[0, 0] = np.nan

    assert not report_pipeline._bootstrap_samples_reportable(result.bootstrap)

    result = _fit_result(_dataset("sample"))
    result.bootstrap.param_samples = result.bootstrap.param_samples[:-1]

    assert not report_pipeline._bootstrap_samples_reportable(result.bootstrap)


def test_complete_nonbinding_bootstrap_reports_parameter_distribution_without_k_histogram(
    tmp_path,
    monkeypatch,
):
    ds = _dataset("sample")
    result = _fit_result(ds)
    result.model = MODEL_SPECS["nb"]
    result.params = np.array([7.0, 0.5], dtype=float)
    result.param_names = ["delta_0", "slope"]
    result.bootstrap.param_samples = result.bootstrap.param_samples[:, 1:]
    result.bootstrap.logk_samples = np.empty((20, 0), dtype=float)

    def unexpected_bootstrap_plot(*args):
        raise AssertionError("a non-binding model has no K histogram")

    monkeypatch.setattr(report_pipeline, "plot_isotherms", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_residuals", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_fraction_bound", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_bootstrap_hist", unexpected_bootstrap_plot)

    summary_rows, entries, report_warnings = report_pipeline.build_report_artifacts(
        args=SimpleNamespace(bootstrap_ci_width=None),
        ordered_keys=["sample"],
        results_by_key={"sample": {"nb": result}},
        failures_by_key={},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    assert report_warnings == []
    assert list(tmp_path.rglob("correlation.csv"))
    assert all(np.isfinite(param.se) for param in entries[0].params)
    assert entries[0].stats["Bootstrap SE (log10 K)"] == "N/A"
    assert not any("bootstrap_" in path for path in entries[0].plots)
    assert summary_rows[0]["Binding constant (M⁻¹)"] == "N/A"
    assert summary_rows[0]["95 % CI"] == "N/A"


def test_dataset_directory_tokens_are_case_insensitive_safe():
    tokens = report_pipeline._dataset_dir_tokens(["Sample", "sample"])

    # Labels differing only by case must stay distinct on the case-insensitive
    # filesystems common on macOS and Windows.
    assert len({token.casefold() for token in tokens.values()}) == 2


def test_replicate_dataset_dir_labels_are_collision_free():
    datasets = [
        SimpleNamespace(name="sample", path="a/sample.csv"),
        SimpleNamespace(name="sample", path="b/sample.csv"),
        SimpleNamespace(name="sample", path="b/sample.csv"),
    ]

    labels = report_pipeline._replicate_dataset_dir_labels(datasets)

    assert len(labels) == len(set(labels)) == 3
    assert all(label.startswith(f"{index:02d}_") for index, label in enumerate(labels, start=1))


def test_report_path_tokens_are_bounded_and_collision_resistant():
    tokens = [
        report_pipeline._safe_path_token("sample/" + "a" * 300),
        report_pipeline._safe_path_token("sample?" + "a" * 300),
    ]

    assert all(len(token) <= 80 for token in tokens)
    assert len(set(tokens)) == 2


def test_slash_containing_peak_labels_use_safe_unique_files_and_original_captions(tmp_path):
    peak_names = ["ppm/H1", "ppm?H1"]
    ds = _dataset("sample", peak_names)

    files = plot_residuals(MODEL_SPECS["nb"], ds, np.zeros_like(ds.y), tmp_path)
    png_files = [path for path in files if path.suffix == ".png"]

    assert len(png_files) == 2
    assert len({path.name for path in png_files}) == 2
    assert all(path.parent == tmp_path and path.is_file() for path in png_files)
    assert all(re.fullmatch(r"residual_peak-\d{4}\.png", path.name) for path in png_files)

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


def test_write_summary_csv_rejects_empty_rows_explicitly(tmp_path):
    output_path = tmp_path / "summary.csv"

    with pytest.raises(ValueError, match="without at least one successful fit result"):
        write_summary_csv([], output_path)

    assert not output_path.exists()


def test_write_summary_csv_neutralizes_formulas_but_preserves_numeric_negatives(tmp_path):
    # A dataset label is the input file stem, so an input named "=cmd.csv" puts
    # that text straight into the first column and a spreadsheet would evaluate
    # it. Negative and exponent-form statistics must survive unchanged.
    output_path = tmp_path / "summary.csv"
    rows = [
        {
            "Dataset": '=HYPERLINK("https://example.invalid", "sample")',
            "Model": "@malicious-name",
            "BIC": "-12.5",
            "AICc": "-1.25e-3",
            "Notes": " \t+FORMULA(1)",
        }
    ]

    write_summary_csv(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        written = next(csv.DictReader(handle))

    assert written["Dataset"].startswith("'=HYPERLINK")
    assert written["Model"] == "'@malicious-name"
    assert written["BIC"] == "-12.5"
    assert written["AICc"] == "-1.25e-3"
    assert written["Notes"] == "' \t+FORMULA(1)"


def _result(**overrides):
    # A complete FitResult-shaped stand-in; override only what a test exercises.
    base = dict(
        model=SimpleNamespace(name="11", n_logk=1),
        datasets=[],
        params=np.array([2.0, 7.0, 7.5], dtype=float),
        param_names=["logK", "H", "HG"],
        bootstrap=None,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
        species=[],
        residual_diagnostics={},
        n=10,
        p=3,
        dof=7,
        jacobian_rank=3,
        jacobian_condition=100.0,
        jacobian_logk_sensitivity=0.5,
        logk_bounds=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_decisions_uses_provisional_language():
    args = argparse.Namespace(bootstrap_ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": _result(model=SimpleNamespace(name="11", n_logk=1), bic=10.0),
            "12": _result(model=SimpleNamespace(name="12", n_logk=2), bic=10.9),
        }
    }
    failures_by_key = {}

    decisions, entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
        failures_by_key,
        display_model_name=lambda name: name,
    )

    assert any("Tentative working model among tested candidates" in line for line in decisions)
    assert any("delta BIC to next candidate" in line for line in decisions)
    assert any("model selection as provisional" in line for line in decisions)
    assert len(entries) == 1
    assert "relative support only" in entries[0].reasons[0]


def test_decision_paragraphs_use_provisional_working_model_language():
    paragraphs = _decision_paragraphs(
        [DecisionEntry(dataset="sample", recommended_model="H : G = 1 : 1", reasons=["Lowest BIC among tested models"])]
    )

    assert len(paragraphs) == 1
    assert "provisional working model" in paragraphs[0]
    assert "best supported" not in paragraphs[0]


def test_build_decisions_uses_fit_failed_wording_for_exclusions():
    args = argparse.Namespace(bootstrap_ci_width=None)
    decisions, entries = build_decisions(
        args,
        ordered_keys=["dataset_a"],
        results_by_key={"dataset_a": {}},
        failures_by_key={"dataset_a": [("11", "ModelFitError: forced model crash")]},
        display_model_name=lambda name: name,
    )

    assert len(entries) == 0
    assert any("fit failed: ModelFitError: forced model crash" in line for line in decisions)


def test_build_decisions_propagates_unavailable_bootstrap_uncertainty():
    args = argparse.Namespace(bootstrap_ci_width=None)
    result = _result(
        bic=10.0,
        bootstrap=SimpleNamespace(
            ci_valid=False,
            ci_message="Bootstrap uncertainty unavailable: 1/1000 refits succeeded.",
            n_boot=1000,
            n_success=1,
            ci_method_used="unavailable",
        ),
    )

    decisions, entries = build_decisions(
        args,
        ordered_keys=["dataset_a"],
        results_by_key={"dataset_a": {"11": result}},
        failures_by_key={},
        display_model_name=lambda name: name,
    )

    assert any("1/1000 refits succeeded" in line for line in decisions)
    assert any("1/1000 refits succeeded" in reason for reason in entries[0].reasons)


def test_build_report_artifacts_uses_fit_failed_wording_for_exclusions(tmp_path):
    summary_rows, model_entries, warnings = build_report_artifacts(
        args=argparse.Namespace(),
        ordered_keys=["dataset_a"],
        results_by_key={"dataset_a": {}},
        failures_by_key={"dataset_a": [("11", "ModelFitError: forced model crash")]},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    assert len(summary_rows) == 1
    assert summary_rows[0]["Status"] == "failed"
    assert "fit failed: ModelFitError: forced model crash" in summary_rows[0]["Notes"]
    assert model_entries == []
    assert warnings == ["dataset_a: excluded 11 (fit failed: ModelFitError: forced model crash)"]


def test_build_methods_sections_uses_brent_and_molar_k_units():
    args = SimpleNamespace(
        replicates=False,
        bootstrap=0,
        bootstrap_method="residual",
        bootstrap_logk_jitter=0.1,
    )
    ds = SimpleNamespace(name="sample", path="sample.csv")

    sections = build_methods_sections(args, [ds])
    param_section = next(section for section in sections if section["title"] == "Parameter Estimation")
    content = param_section["content"]

    # The methods text must name the solver family and report K in molar units.
    # Exact tolerances and phrasing may change with the implementation.
    assert "Brent" in content
    assert "brentq" in content
    assert "M⁻¹" in content
    assert "Newton" not in content


def test_build_methods_sections_mentions_bca_when_selected():
    args = SimpleNamespace(
        replicates=False,
        bootstrap=100,
        bootstrap_method="residual",
        bootstrap_ci_method="bca",
        bootstrap_logk_jitter=0.1,
    )
    ds = SimpleNamespace(name="sample", path="sample.csv")

    sections = build_methods_sections(args, [ds])
    uq_section = next(section for section in sections if section["title"] == "Uncertainty Quantification")
    # The selected interval method and its jackknife basis must be disclosed;
    # the surrounding prose may change.
    content = uq_section["content"]
    assert "BCa" in content
    assert "jackknife" in content


def test_build_summary_row_uses_selected_ci_and_reports_logk_se():
    bootstrap = SimpleNamespace(
        param_samples=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float),
        logk_samples=np.array([[1.0], [2.0], [3.0]], dtype=float),
        ci_low=np.array([1.2], dtype=float),
        ci_high=np.array([2.8], dtype=float),
        ci_low_percentile=np.array([1.05], dtype=float),
        ci_high_percentile=np.array([2.95], dtype=float),
        ci_low_bca=np.array([1.2], dtype=float),
        ci_high_bca=np.array([2.8], dtype=float),
        ci_method="bca",
        ci_valid=True,
        n_success=3,
        n_boot=3,
    )
    res = SimpleNamespace(
        model=SimpleNamespace(n_logk=1),
        params=np.array([2.0], dtype=float),
        bootstrap=bootstrap,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
    )

    row = _build_summary_row(res, "sample", "11")

    assert row["95 % CI"] == f"[{10**1.2:.6g}, {10**2.8:.6g}]"
    assert row["Status"] == "success"


def test_build_summary_row_labels_sequential_k_values_and_ci():
    bootstrap = SimpleNamespace(
        param_samples=np.array([[1.0, 2.0], [1.1, 2.1]], dtype=float),
        logk_samples=np.array([[1.0, 2.0], [1.1, 2.1]], dtype=float),
        ci_low=np.array([0.9, 1.9], dtype=float),
        ci_high=np.array([1.2, 2.2], dtype=float),
        ci_low_percentile=np.array([0.9, 1.9], dtype=float),
        ci_high_percentile=np.array([1.2, 2.2], dtype=float),
        ci_low_bca=np.array([np.nan, np.nan], dtype=float),
        ci_high_bca=np.array([np.nan, np.nan], dtype=float),
        ci_method="percentile",
        ci_valid=True,
        n_success=2,
        n_boot=2,
    )
    res = SimpleNamespace(
        model=SimpleNamespace(n_logk=2),
        params=np.array([1.0, 2.0], dtype=float),
        bootstrap=bootstrap,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
    )

    row = _build_summary_row(res, "sample", "12")

    assert row["Binding constant (M⁻¹)"] == "K1=10; K2=100"
    assert row["95 % CI"] == f"K1=[{10**0.9:.6g}, {10**1.2:.6g}]; K2=[{10**1.9:.6g}, {10**2.2:.6g}]"


def _pinned_k_result(logk_bounds):
    return SimpleNamespace(
        model=MODEL_SPECS["11"],
        datasets=[],
        params=np.array([12.0, 7.0, 7.5], dtype=float),
        param_names=["logK", "H", "HG"],
        bootstrap=None,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
        species=[],
        residual_diagnostics={},
        n=10,
        p=3,
        logk_bounds=logk_bounds,
    )


def test_bound_pinned_k_is_reported_in_warnings_and_summary_notes():
    res = _pinned_k_result((0.0, 12.0))

    warnings = _build_model_warnings(argparse.Namespace(bootstrap_ci_width=None), res, None)
    row = _build_summary_row(res, "sample", "11", warnings)

    assert any("upper log10(K) bound" in warning for warning in warnings)
    assert "upper log10(K) bound" in row["Notes"]


def test_bound_pinned_warning_uses_actual_bounds_not_cli_constants():
    # A logK of 12 is only "pinned" if 12 was the active upper bound. With no
    # bounds (a programmatic fit) or a wider bound, K=1e12 is a valid estimate
    # and must not be flagged.
    unbounded = _build_model_warnings(
        argparse.Namespace(bootstrap_ci_width=None), _pinned_k_result(None), None
    )
    wider = _build_model_warnings(
        argparse.Namespace(bootstrap_ci_width=None), _pinned_k_result((0.0, 15.0)), None
    )

    assert not any("log10(K) bound" in warning for warning in unbounded)
    assert not any("log10(K) bound" in warning for warning in wider)


def test_aicc_only_unavailable_is_explained_in_warnings():
    # An underpowered fit can keep a finite BIC while the AICc small-sample
    # correction is undefined (NaN). The report must explain the resulting
    # AICc=N/A instead of showing it silently.
    res = SimpleNamespace(
        model=MODEL_SPECS["11"],
        datasets=[],
        params=np.array([3.0, 7.0, 7.5], dtype=float),
        param_names=["logK", "H", "HG"],
        bootstrap=None,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=float("nan"),
        penalty_count=0,
        species=[],
        residual_diagnostics={},
        n=5,
        p=4,
        logk_bounds=(0.0, 12.0),
    )

    warnings = _build_model_warnings(argparse.Namespace(bootstrap_ci_width=None), res, None)
    row = _build_summary_row(res, "sample", "11", warnings)

    assert any("AICc unavailable: too few observations" in warning for warning in warnings)
    assert not any("BIC/AICc unavailable" in warning for warning in warnings)
    assert "AICc unavailable" in row["Notes"]


def test_build_decisions_excludes_nonfinite_bic_from_ranking():
    args = argparse.Namespace(bootstrap_ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": _result(bic=float("nan")),
        }
    }

    decisions, entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
        failures_by_key={},
        display_model_name=lambda name: name,
    )

    assert entries == []
    assert any("No model had a finite BIC" in line for line in decisions)
