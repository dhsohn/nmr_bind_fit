from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nmr_bind_fit import report_pipeline
from nmr_bind_fit.io import Dataset
from nmr_bind_fit.models import MODEL_SPECS
from nmr_bind_fit.plots import plot_residuals
from nmr_bind_fit.report import (
    DecisionEntry,
    _decision_paragraphs,
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


def _dataset(name: str, peak_names: list[str] | None = None) -> Dataset:
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
    with_uncertainty: bool = True,
) -> SimpleNamespace:
    uncertainty = None
    if with_uncertainty:
        uncertainty = SimpleNamespace(
            param_se=np.array([0.05, 0.02, 0.03], dtype=float),
            logk_ci_low=np.array([1.8], dtype=float),
            logk_ci_high=np.array([2.2], dtype=float),
            correlation=np.eye(3, dtype=float),
        )
    return SimpleNamespace(
        model=MODEL_SPECS["11"],
        datasets=[ds],
        params=np.array([2.0, 7.0, 7.2], dtype=float),
        param_names=["logK", "delta_free", "delta_bound"],
        residuals=[np.zeros_like(ds.y)],
        species=[SimpleNamespace(solver_stats=None)],
        uncertainty=uncertainty,
        r2=0.99,
        r2_per_peak=[0.99],
        rss=0.01,
        rmse=0.02,
        bic=1.0,
        aicc=2.0,
        penalty_count=0,
        residual_diagnostics={},
        success=True,
        message="ok",
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

    monkeypatch.setattr(report_pipeline, "plot_isotherms", fake_isotherms)
    monkeypatch.setattr(report_pipeline, "plot_residuals", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_fraction_bound", lambda *args: [])

    keys = ["sample/a", "sample?a"]
    summary_rows, entries, warnings = report_pipeline.build_report_artifacts(
        args=SimpleNamespace(ci_width=None),
        ordered_keys=keys,
        results_by_key={key: {"11": _fit_result(ds)} for key, ds in zip(keys, datasets)},
        out_dir=tmp_path,
    )

    isotherms = [
        tmp_path / next(path for path in entry.plots if "isotherm_" in path)
        for entry in entries
    ]

    assert isotherms[0] != isotherms[1]
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
    # Reporting a fit already means it cleared the identifiability gate, so the
    # rank and condition number behind that gate are not repeated per model.
    assert not any("Jacobian" in label for label in entries[0].stats)
    # Counters that only matter on failure stay out of a clean card.
    assert "Optimization penalty events" not in entries[0].stats
    assert float(entries[0].stats["Standard error (log10 K)"]) > 0.0
    assert all(np.isfinite(param.se) for param in entries[0].params)
    assert summary_rows[0]["95 % CI"] != "N/A"
    assert warnings == []
    assert all(not entry.warnings for entry in entries)
    assert set(entries[0].plot_labels.values()) == {"ppm_H1"}


def test_nonbinding_model_reports_parameter_errors_but_no_k_interval(
    tmp_path,
    monkeypatch,
):
    ds = _dataset("sample")
    result = _fit_result(ds)
    result.model = MODEL_SPECS["nb"]
    result.params = np.array([7.0, 0.5], dtype=float)
    result.param_names = ["delta_0", "slope"]

    monkeypatch.setattr(report_pipeline, "plot_isotherms", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_residuals", lambda *args: [])
    monkeypatch.setattr(report_pipeline, "plot_fraction_bound", lambda *args: [])

    summary_rows, entries, report_warnings = report_pipeline.build_report_artifacts(
        args=SimpleNamespace(ci_width=None),
        ordered_keys=["sample"],
        results_by_key={"sample": {"nb": result}},
        out_dir=tmp_path,
    )

    assert report_warnings == []
    assert list(tmp_path.rglob("correlation.csv"))
    assert all(np.isfinite(param.se) for param in entries[0].params)
    assert "Standard error (log10 K)" not in entries[0].stats
    assert summary_rows[0]["Binding constant (M⁻¹)"] == "N/A"
    assert summary_rows[0]["95 % CI"] == "N/A"


def test_replicate_dataset_dir_labels_are_collision_free():
    datasets = [
        SimpleNamespace(name="sample", path="a/sample.csv"),
        SimpleNamespace(name="sample", path="b/sample.csv"),
        SimpleNamespace(name="sample", path="b/sample.csv"),
    ]

    labels = report_pipeline._replicate_dataset_dir_labels(datasets)

    assert len(labels) == len(set(labels)) == 3
    assert all(label.startswith(f"{index:02d}_") for index, label in enumerate(labels, start=1))


def test_report_path_tokens_are_sanitized_and_bounded():
    # Distinctness comes from the ordinal prefix the callers add, not from here.
    token = report_pipeline._safe_path_token("sample/" + "a" * 300)

    assert len(token) <= 80
    assert re.fullmatch(r"[A-Za-z0-9._-]+", token)


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


def _result(**overrides):
    # A complete FitResult-shaped stand-in; override only what a test exercises.
    base = {
        "model": SimpleNamespace(name="11", n_logk=1),
        "datasets": [],
        "params": np.array([2.0, 7.0, 7.5], dtype=float),
        "param_names": ["logK", "H", "HG"],
        "uncertainty": None,
        "r2": 0.9,
        "r2_per_peak": [0.9],
        "rss": 1.0,
        "rmse": 0.5,
        "bic": 10.0,
        "aicc": 11.0,
        "penalty_count": 0,
        "species": [],
        "residual_diagnostics": {},
        "n": 10,
        "p": 3,
        "dof": 7,
        "jacobian_rank": 3,
        "jacobian_condition": 100.0,
        "jacobian_logk_sensitivity": 0.5,
        "logk_bounds": None,
        "success": True,
        "message": "ok",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_decisions_uses_provisional_language():
    args = argparse.Namespace(ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": _result(model=SimpleNamespace(name="11", n_logk=1), bic=10.0),
            "12": _result(model=SimpleNamespace(name="12", n_logk=2), bic=10.9),
        }
    }
    entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
    )

    assert len(entries) == 1
    assert entries[0].recommended_model == "H : G = 1 : 1"
    assert "relative support only" in entries[0].reasons[0]
    # The two candidates are 0.9 apart, so discrimination is flagged as weak.
    assert any("discrimination is weak" in reason for reason in entries[0].reasons)


def test_decision_paragraphs_use_provisional_working_model_language():
    paragraphs = _decision_paragraphs(
        [DecisionEntry(dataset="sample", recommended_model="H : G = 1 : 1", reasons=["Lowest BIC among tested models"])]
    )

    assert len(paragraphs) == 1
    assert "provisional working model" in paragraphs[0]
    assert "best supported" not in paragraphs[0]


def test_build_report_artifacts_uses_fit_failed_wording_for_exclusions(tmp_path):
    summary_rows, model_entries, warnings = build_report_artifacts(
        args=argparse.Namespace(),
        ordered_keys=["dataset_a"],
        results_by_key={
            "dataset_a": {
                "11": _result(
                    success=False,
                    message="ModelFitError: forced model crash",
                )
            }
        },
        out_dir=tmp_path,
    )

    assert len(summary_rows) == 1
    assert summary_rows[0]["Status"] == "failed"
    assert "Notes" not in summary_rows[0]
    assert model_entries == []
    assert warnings == [
        "dataset_a: excluded H : G = 1 : 1 (fit failed: ModelFitError: forced model crash)"
    ]


def test_build_methods_sections_uses_brent_and_molar_k_units():
    args = SimpleNamespace(
        replicates=False,
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


def test_build_methods_sections_discloses_asymptotic_covariance_interval():
    args = SimpleNamespace(
        replicates=False,
    )
    ds = SimpleNamespace(name="sample", path="sample.csv")

    sections = build_methods_sections(args, [ds])
    uq_section = next(section for section in sections if section["title"] == "Uncertainty Quantification")
    # The covariance basis, the Student-t interval, and the local-linearity
    # assumption must all be disclosed; the surrounding prose may change.
    content = uq_section["content"]
    assert "covariance" in content
    assert "Student-t" in content
    assert "95%" in content
    assert "locally linear" in content


def test_build_summary_row_uses_selected_ci_and_reports_logk_se():
    uncertainty = SimpleNamespace(
        param_se=np.array([0.1], dtype=float),
        logk_ci_low=np.array([1.2], dtype=float),
        logk_ci_high=np.array([2.8], dtype=float),
        correlation=np.eye(1, dtype=float),
    )
    res = SimpleNamespace(
        model=SimpleNamespace(n_logk=1),
        params=np.array([2.0], dtype=float),
        uncertainty=uncertainty,
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
    uncertainty = SimpleNamespace(
        param_se=np.array([0.05, 0.05], dtype=float),
        logk_ci_low=np.array([0.9, 1.9], dtype=float),
        logk_ci_high=np.array([1.2, 2.2], dtype=float),
        correlation=np.eye(2, dtype=float),
    )
    res = SimpleNamespace(
        model=SimpleNamespace(n_logk=2),
        params=np.array([1.0, 2.0], dtype=float),
        uncertainty=uncertainty,
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
        uncertainty=None,
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


def test_bound_pinned_k_is_reported_in_warnings():
    res = _pinned_k_result((0.0, 12.0))

    warnings = _build_model_warnings(argparse.Namespace(ci_width=None), res, None)

    assert any("upper log10(K) bound" in warning for warning in warnings)


def test_lower_bound_pinned_k_is_not_reported_in_warnings():
    res = _pinned_k_result((0.0, 12.0))
    res.params[0] = 0.0

    warnings = _build_model_warnings(argparse.Namespace(ci_width=None), res, None)

    assert not any("log10(K) bound" in warning for warning in warnings)


def test_bound_pinned_warning_uses_actual_bounds_not_cli_constants():
    # A logK of 12 is only "pinned" if 12 was the active upper bound. With no
    # bounds (a programmatic fit) or a wider bound, K=1e12 is a valid estimate
    # and must not be flagged.
    unbounded = _build_model_warnings(
        argparse.Namespace(ci_width=None), _pinned_k_result(None), None
    )
    wider = _build_model_warnings(
        argparse.Namespace(ci_width=None), _pinned_k_result((0.0, 15.0)), None
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
        uncertainty=None,
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

    warnings = _build_model_warnings(argparse.Namespace(ci_width=None), res, None)

    assert any("AICc unavailable: too few observations" in warning for warning in warnings)
    assert not any("BIC/AICc unavailable" in warning for warning in warnings)


def test_build_decisions_excludes_nonfinite_bic_from_ranking():
    args = argparse.Namespace(ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": _result(bic=float("nan")),
        }
    }

    entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
    )

    # No finitely ranked candidate yields no recommendation; the reason reaches
    # the reader through the warnings shown alongside the dataset.
    assert entries == []
