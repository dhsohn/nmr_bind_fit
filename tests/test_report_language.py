import argparse
from types import SimpleNamespace

import numpy as np

from nmr_bind_fit.models import MODEL_SPECS
from nmr_bind_fit.report import DecisionEntry, _decision_paragraphs
from nmr_bind_fit.report_pipeline import (
    _build_model_warnings,
    _build_summary_row,
    _collect_plot_paths,
    _replicate_dataset_dir_labels,
    build_decisions,
    build_methods_sections,
    build_report_artifacts,
)


def test_build_decisions_uses_provisional_language():
    args = argparse.Namespace(bootstrap_ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": SimpleNamespace(model=SimpleNamespace(name="11", n_logk=1), bic=10.0, bootstrap=None),
            "12": SimpleNamespace(model=SimpleNamespace(name="12", n_logk=2), bic=10.9, bootstrap=None),
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


def test_replicate_dataset_dir_labels_are_collision_free():
    ds1 = SimpleNamespace(name="sample", path="a/sample.csv")
    ds2 = SimpleNamespace(name="sample", path="b/sample.csv")
    ds3 = SimpleNamespace(name="sample", path="b/sample.csv")

    labels = _replicate_dataset_dir_labels([ds1, ds2, ds3])

    assert len(labels) == 3
    assert len(set(labels)) == 3
    assert labels[0].startswith("01_")
    assert labels[1].startswith("02_")
    assert labels[2].startswith("03_")


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

    assert "Brent's method" in content
    assert "scipy.optimize.brentq" in content
    assert "xtol=1e-50, rtol=1e-15" in content
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
    assert "BCa-style bootstrap quantiles" in uq_section["content"]
    assert "leave-one-titration-point jackknife refits" in uq_section["content"]


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


def test_bound_pinned_k_is_reported_in_warnings_and_summary_notes():
    res = SimpleNamespace(
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
    )

    warnings = _build_model_warnings(argparse.Namespace(bootstrap_ci_width=None), res, None)
    row = _build_summary_row(res, "sample", "11", warnings)

    assert any("upper log10(K) bound" in warning for warning in warnings)
    assert "upper log10(K) bound" in row["Notes"]


def test_collect_plot_paths_scopes_single_dataset_outputs_by_dataset_label(tmp_path, monkeypatch):
    ds = SimpleNamespace(
        name="sample",
        path=tmp_path / "sample.csv",
        n_peaks=1,
        y_cols=["ppm1"],
        dropped_peaks=[],
    )
    res = SimpleNamespace(
        model=MODEL_SPECS["11"],
        datasets=[ds],
        params=np.array([4.0, 7.0, 7.5], dtype=float),
        residuals=[np.zeros((3, 1), dtype=float)],
        bootstrap=None,
        param_names=["logK", "H_sample_ppm1", "HG_sample_ppm1"],
    )

    def fake_plot(name):
        def _impl(model, ds, *args):
            out_dir = args[-1]
            return [out_dir / f"{name}.png"]

        return _impl

    monkeypatch.setattr("nmr_bind_fit.report_pipeline.plot_isotherms", fake_plot("isotherm_ppm1"))
    monkeypatch.setattr("nmr_bind_fit.report_pipeline.plot_residuals", fake_plot("residual_ppm1"))
    monkeypatch.setattr("nmr_bind_fit.report_pipeline.plot_fraction_bound", fake_plot("fraction_bound"))

    paths_a = _collect_plot_paths(res, "11", "dataset A", tmp_path)
    paths_b = _collect_plot_paths(res, "11", "dataset B", tmp_path)

    assert all(path.startswith("model_11/dataset_dataset_A/") for path in paths_a)
    assert all(path.startswith("model_11/dataset_dataset_B/") for path in paths_b)
    assert set(paths_a).isdisjoint(paths_b)


def test_build_report_artifacts_keeps_sanitized_dataset_dirs_unique(tmp_path, monkeypatch):
    ds_a = SimpleNamespace(
        name="sample A",
        path=tmp_path / "sample A.csv",
        n_peaks=1,
        y_cols=["ppm1"],
        dropped_peaks=[],
    )
    ds_b = SimpleNamespace(
        name="sample_A",
        path=tmp_path / "sample_A.csv",
        n_peaks=1,
        y_cols=["ppm1"],
        dropped_peaks=[],
    )

    def make_result(ds):
        return SimpleNamespace(
            model=MODEL_SPECS["11"],
            datasets=[ds],
            params=np.array([4.0, 7.0, 7.5], dtype=float),
            residuals=[np.zeros((3, 1), dtype=float)],
            species=[SimpleNamespace(solver_stats=None)],
            bootstrap=None,
            param_names=["logK", f"H_{ds.name}_ppm1", f"HG_{ds.name}_ppm1"],
            r2=0.9,
            r2_per_peak=[0.9],
            rss=1.0,
            rmse=0.5,
            bic=10.0,
            aicc=11.0,
            penalty_count=0,
            residual_diagnostics={},
        )

    def fake_plot(name):
        def _impl(model, ds, *args):
            out_dir = args[-1]
            return [out_dir / f"{name}.png"]

        return _impl

    monkeypatch.setattr("nmr_bind_fit.report_pipeline.plot_isotherms", fake_plot("isotherm_ppm1"))
    monkeypatch.setattr("nmr_bind_fit.report_pipeline.plot_residuals", fake_plot("residual_ppm1"))
    monkeypatch.setattr("nmr_bind_fit.report_pipeline.plot_fraction_bound", fake_plot("fraction_bound"))

    _, model_entries, _ = build_report_artifacts(
        args=argparse.Namespace(bootstrap_ci_width=None),
        ordered_keys=["sample A.csv", "sample_A.csv"],
        results_by_key={
            "sample A.csv": {"11": make_result(ds_a)},
            "sample_A.csv": {"11": make_result(ds_b)},
        },
        failures_by_key={},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    paths_by_dataset = {entry.dataset: entry.plots for entry in model_entries}

    assert all(path.startswith("model_11/dataset_01_sample_A.csv/") for path in paths_by_dataset["sample A.csv"])
    assert all(path.startswith("model_11/dataset_02_sample_A.csv/") for path in paths_by_dataset["sample_A.csv"])
    assert set(paths_by_dataset["sample A.csv"]).isdisjoint(paths_by_dataset["sample_A.csv"])


def test_build_decisions_excludes_nonfinite_bic_from_ranking():
    args = argparse.Namespace(bootstrap_ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": SimpleNamespace(model=SimpleNamespace(name="11", n_logk=1), bic=float("nan"), bootstrap=None),
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
